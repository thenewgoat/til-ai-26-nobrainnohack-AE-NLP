"""Hybrid post-opener rollout collection: drive each learner slot with a
HybridController and record ONLY post-handover ticks into a HybridRolloutBuffer
whose fields match `hybrid_ppo.ppo_update_hybrid`. train_selfplay.py is untouched
(additive). Opponents here are RandomAgents; the league pool is wired in Plan 5.4.
"""
import random
from dataclasses import dataclass

import numpy as np
import torch

from critic import STATE_PLANES, STATE_SCALARS, encode_global_state
from evaluate import RandomAgent, ScriptedAgent
from features import (FeatureBuilder, NUM_BASES, BASE_FIELDS, RAW_AGENT_SHAPE,
                      RAW_BASE_SHAPE, STACKED_GRID_CHANNELS, STACKED_SCALARS)
from hybrid_controller import ActorRuntime, HybridController
from policy import NUM_ACTIONS, SymbolicTransformerActor
from scripted.geometry import STAY
from scripted.handover import HandoverTrigger
from til_environment import bomberman_env
from til_environment.config import default_config

# torch's default file_descriptor tensor-sharing opens one socket per storage and
# exhausts fds when many workers each ship a state_dict-built buffer back. Match
# train_selfplay.py and share via the filesystem instead.
torch.multiprocessing.set_sharing_strategy("file_system")

SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]
EPISODE_LEN = 200
GRID_SIZE = 16


@dataclass
class HybridRolloutBuffer:
    grid: np.ndarray
    base_feats: np.ndarray
    raw_agent: np.ndarray
    raw_base: np.ndarray
    scalar: np.ndarray
    gstate: np.ndarray
    gscalar: np.ndarray
    masks: np.ndarray
    proposed_actions: np.ndarray
    executed_actions: np.ndarray
    actor_queried: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    env_rewards: np.ndarray
    dones: np.ndarray
    forward_bias: float = 0.0

    @property
    def size(self):
        return self.grid.shape[0]


def _new_hybrid_buffer(n):
    return HybridRolloutBuffer(
        grid=np.zeros((n, STACKED_GRID_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32),
        base_feats=np.zeros((n, NUM_BASES, BASE_FIELDS), np.float32),
        raw_agent=np.zeros((n, *RAW_AGENT_SHAPE), np.float32),
        raw_base=np.zeros((n, *RAW_BASE_SHAPE), np.float32),
        scalar=np.zeros((n, STACKED_SCALARS), np.float32),
        gstate=np.zeros((n, STATE_PLANES, GRID_SIZE, GRID_SIZE), np.float32),
        gscalar=np.zeros((n, STATE_SCALARS), np.float32),
        masks=np.zeros((n, NUM_ACTIONS), bool),
        proposed_actions=np.zeros(n, np.int64),
        executed_actions=np.zeros(n, np.int64),
        actor_queried=np.zeros(n, bool),
        logprobs=np.zeros(n, np.float32),
        rewards=np.zeros(n, np.float32),
        env_rewards=np.zeros(n, np.float32),
        dones=np.zeros(n, np.float32),
        forward_bias=0.0,
    )


def build_hybrid_buffer(transitions, forward_bias=0.0):
    """Stack a list of per-tick transition dicts (in per-(episode,slot)-contiguous
    order) into a HybridRolloutBuffer. Empty list -> a zero-length buffer."""
    buf = _new_hybrid_buffer(len(transitions))
    for i, tr in enumerate(transitions):
        buf.grid[i] = tr["grid"]
        buf.base_feats[i] = tr["base_feats"]
        buf.raw_agent[i] = tr["raw_agent"]
        buf.raw_base[i] = tr["raw_base"]
        buf.scalar[i] = tr["scalar"]
        buf.gstate[i] = tr["gstate"]
        buf.gscalar[i] = tr["gscalar"]
        buf.masks[i] = tr["mask"]
        buf.proposed_actions[i] = tr["proposed"]
        buf.executed_actions[i] = tr["executed"]
        buf.actor_queried[i] = tr["actor_queried"]
        buf.logprobs[i] = tr["logp"]
        buf.rewards[i] = tr["reward"]
        buf.env_rewards[i] = tr["env_reward"]
        buf.dones[i] = tr["done"]
    buf.forward_bias = float(forward_bias)
    return buf


_FROZEN_CACHE = {}      # path -> loaded actor (per process; reused across episodes)


def _make_frozen_hybrid(path, trigger):
    """Build a self-play opponent: a frozen HybridAgent (scripted opener + the
    checkpoint's RL) from a saved actor `path`. The actor is cached per process."""
    from hybrid_eval import HybridAgent       # lazy: avoid import cycle
    actor = _FROZEN_CACHE.get(path)
    if actor is None:
        actor = SymbolicTransformerActor.from_checkpoint(path).eval()
        _FROZEN_CACHE[path] = actor
    return HybridAgent(actor, trigger=trigger)


def collect_hybrid_rollout(actor, learner_slots, num_episodes, seed0,
                           trigger=None, post_params=None, forward_bias=0.0,
                           anti_idle_penalty=0.0, opponent_names=None,
                           randomize_slot=False, frozen_paths=None,
                           selfplay_prob=0.0):
    """Roll out `num_episodes`; drive each learner slot with a HybridController.

    Records ONLY post-handover ticks (pre-handover ticks warm the controller but
    are excluded from gradient/GAE). Per-(episode,slot) runs are appended
    contiguously so GAE boundaries (via `dones`) are correct. The handover-boundary
    reward is dropped (it is the consequence of the last scripted action: `prev`
    is None until the first post-handover write). Anti-idle subtracts `penalty`
    only on actor-queried proposed-STAY ticks. Opponents are RandomAgents (the
    league pool is Plan 5.4). Returns a HybridRolloutBuffer."""
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    dynamics = env.unwrapped.dynamics
    # ActorRuntime moves `actor` to CPU in place for per-step inference; remember
    # the caller's device so we can restore it before returning (ppo_update_hybrid
    # assumes the actor stays on its training device).
    orig_device = next(actor.parameters()).device
    runtime = ActorRuntime(actor)
    trigger = trigger or HandoverTrigger()
    learner_slots = tuple(learner_slots)
    transitions = []
    for ep in range(num_episodes):
        seed = seed0 + ep
        random.seed(seed)
        env.reset(seed=seed)
        # randomize_slot: learner plays a seed-derived random slot 0-5 (board stays
        # novice/seed-88) so it learns from all spawns, not just agent_0.
        ep_slots = (random.choice(SLOTS),) if randomize_slot else learner_slots
        opp_slots = [s for s in SLOTS if s not in ep_slots]
        opp_agents = {}
        for s in opp_slots:
            if frozen_paths and random.random() < selfplay_prob:
                opp_agents[s] = _make_frozen_hybrid(random.choice(frozen_paths), trigger)
            elif opponent_names:
                opp_agents[s] = ScriptedAgent(random.choice(opponent_names))
            else:
                opp_agents[s] = RandomAgent()
        for ag in opp_agents.values():
            ag.reset()
        controllers = {
            s: HybridController(runtime, trigger, post_params=post_params,
                                feature_builder=FeatureBuilder(),
                                forward_bias=forward_bias)
            for s in ep_slots}
        per_slot_run = {s: [] for s in ep_slots}
        prev_tr = {s: None for s in ep_slots}
        for slot in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            done = term or trunc
            if slot in ep_slots:
                if prev_tr[slot] is not None:
                    r = float(reward)
                    prev_tr[slot]["env_reward"] = r
                    if (anti_idle_penalty and prev_tr[slot]["actor_queried"]
                            and prev_tr[slot]["proposed"] == STAY):
                        r -= anti_idle_penalty
                    prev_tr[slot]["reward"] = r
                    prev_tr[slot]["done"] = 1.0 if done else 0.0
                if done:
                    env.step(None)
                    prev_tr[slot] = None
                    continue
                step = int(np.asarray(obs["step"]).flat[0])
                action, decision = controllers[slot].step(obs)
                if decision is None:
                    env.step(action)            # pre-handover: warmed, not recorded
                    continue
                gp, gsc = encode_global_state(dynamics, slot, step)
                grid, base_feats, raw_agent, raw_base, scalar = decision.features
                proposed = (decision.proposed_action
                            if decision.proposed_action is not None
                            else decision.executed_action)
                tr = dict(
                    grid=grid, base_feats=base_feats, raw_agent=raw_agent,
                    raw_base=raw_base, scalar=scalar, gstate=gp, gscalar=gsc,
                    mask=np.asarray(decision.action_mask, bool).reshape(-1),
                    proposed=int(proposed),
                    executed=int(decision.executed_action),
                    actor_queried=bool(decision.actor_queried),
                    logp=float(decision.old_proposal_logp),
                    reward=0.0, env_reward=0.0, done=0.0)
                per_slot_run[slot].append(tr)
                prev_tr[slot] = tr
                env.step(action)
            else:
                env.step(None if done else opp_agents[slot].action(obs))
        for s in ep_slots:                      # append each slot's run contiguously
            transitions.extend(per_slot_run[s])
    env.close()
    actor.to(orig_device)   # undo ActorRuntime's in-place CPU move
    return build_hybrid_buffer(transitions, forward_bias)


# ── parallel rollout ─────────────────────────────────────────────────────────
def _concat_hybrid_buffers(bufs):
    """Concatenate HybridRolloutBuffers field-wise, preserving order. Each buffer
    is internally per-(episode,slot)-contiguous and done-terminated, so GAE (which
    resets on `dones`) is correct regardless of cross-buffer order."""
    fields = ("grid", "base_feats", "raw_agent", "raw_base", "scalar", "gstate",
              "gscalar", "masks", "proposed_actions", "executed_actions",
              "actor_queried", "logprobs", "rewards", "env_rewards", "dones")
    merged = {f: np.concatenate([getattr(b, f) for b in bufs]) for f in fields}
    return HybridRolloutBuffer(forward_bias=bufs[0].forward_bias, **merged)


def _worker_hybrid_rollout(state_dict, cfg, learner_slots, num_episodes, seed0,
                           trigger, post_params, forward_bias, anti_idle_penalty,
                           opponent_names, randomize_slot=False, frozen_paths=None,
                           selfplay_prob=0.0):
    """Pool-worker entry point (top-level + picklable). Rebuild the actor on CPU
    from `state_dict`/`cfg`, then run `collect_hybrid_rollout` for this chunk.
    `frozen_paths` are checkpoint paths (workers load them as self-play opponents)."""
    torch.set_num_threads(1)        # one thread per worker — no core oversubscription
    torch.manual_seed(seed0)
    actor = SymbolicTransformerActor(**cfg)
    actor.load_state_dict(state_dict)
    actor.eval()
    return collect_hybrid_rollout(
        actor, learner_slots, num_episodes, seed0, trigger=trigger,
        post_params=post_params, forward_bias=forward_bias,
        anti_idle_penalty=anti_idle_penalty, opponent_names=opponent_names,
        randomize_slot=randomize_slot, frozen_paths=frozen_paths,
        selfplay_prob=selfplay_prob)


def collect_hybrid_rollout_parallel(actor, learner_slots, num_episodes, seed0,
                                    pool, num_workers, trigger=None,
                                    post_params=None, forward_bias=0.0,
                                    anti_idle_penalty=0.0, opponent_names=None,
                                    progress=False, randomize_slot=False,
                                    frozen_paths=None, selfplay_prob=0.0):
    """Parallel `collect_hybrid_rollout`: split `num_episodes` into contiguous
    chunks across `min(num_workers, num_episodes)` pool workers and concatenate.
    Episode `ep` keeps seed `seed0 + ep` regardless of chunking. The caller's
    `actor` is NOT mutated — workers rebuild from a CPU `state_dict` copy. `pool`
    is a caller-owned multiprocessing Pool."""
    workers = max(1, min(num_workers, num_episodes))
    base, extra = divmod(num_episodes, workers)
    state_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
    cfg = actor.cfg
    tasks, offset = [], 0
    for i in range(workers):
        count = base + (1 if i < extra else 0)
        if count == 0:
            continue
        tasks.append((state_dict, cfg, learner_slots, count, seed0 + offset,
                      trigger, post_params, forward_bias, anti_idle_penalty,
                      opponent_names, randomize_slot, frozen_paths, selfplay_prob))
        offset += count
    if progress:
        from tqdm.auto import tqdm
        bufs = list(tqdm(pool.imap_unordered(_worker_hybrid_rollout_star, tasks),
                         total=len(tasks), desc="rollout", unit="chunk",
                         leave=False))
    else:
        bufs = pool.starmap(_worker_hybrid_rollout, tasks)
    return _concat_hybrid_buffers(bufs)


def _worker_hybrid_rollout_star(args):
    """imap-friendly single-arg adapter (order-preserving dispatch + tqdm)."""
    return _worker_hybrid_rollout(*args)
