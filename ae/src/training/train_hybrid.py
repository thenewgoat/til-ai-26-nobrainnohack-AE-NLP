"""Hybrid post-opener PPO training loop.

Ties together collect_hybrid_rollout (5.3), the reused GAE + centralized-critic
values, and ppo_update_hybrid (5.2/5.1) with the forward-bias + entropy schedules
and an optional critic warm-up. Builds the actor + critic with dropout=0 so the
ppo_update_hybrid determinism assert holds. train_selfplay.py is untouched.

Opponents here are scripted (opponent_names). The self-play league + paired-
continuation acceptance eval are Plan 5.5.
"""
import json
import multiprocessing
import os
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from critic import CentralizedCritic
from hybrid_ppo import HybridPPOConfig, forward_bias_value, ppo_update_hybrid
from hybrid_rollout import collect_hybrid_rollout, collect_hybrid_rollout_parallel
from policy import SymbolicTransformerActor
from train_selfplay import compute_advantages, critic_values, make_optimizer

GAMMA = 0.99
GAE_LAMBDA = 0.95


def critic_only_update(critic, opt, buf, returns, cfg, device):
    """Warm-up step: fit the centralized critic to `returns`; the actor is left
    untouched (no actor term in the loss -> set_to_none keeps actor grads None ->
    Adam skips the actor group). Mirrors ppo_update_hybrid's value-loss path."""
    gstate = torch.from_numpy(buf.gstate).to(device)
    gscalar = torch.from_numpy(buf.gscalar).to(device)
    ret = torch.from_numpy(np.asarray(returns, np.float32)).to(device)
    n = buf.size
    mb_size = max(1, n // cfg.num_minibatches)
    inds = np.arange(n)
    critic.train()
    vl = torch.zeros((), device=device)
    for _ in range(cfg.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, n, mb_size):
            mb = torch.from_numpy(inds[start:start + mb_size]).to(device)
            v = critic(gstate[mb], gscalar[mb])
            vl = 0.5 * ((v - ret[mb]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            (cfg.vf_coef * vl).backward()
            nn.utils.clip_grad_norm_(
                [p for p in critic.parameters() if p.grad is not None],
                cfg.max_grad_norm)
            opt.step()
    return {"value_loss": float(vl.detach()), "warmup": True}


def train_hybrid(total_updates=3000, episodes_per_update=4,
                 learner_slots=("agent_0",), seed0=0, cfg=None,
                 forward_bias_init=0.0, anti_idle_penalty=0.0, critic_warmup=0,
                 trigger=None, opponent_names=None, post_params=None,
                 d_model=64, n_layers=4, n_heads=4, device="cpu",
                 checkpoint_dir=None, checkpoint_every=0, log_path=None,
                 eval_every=0, eval_fn=None, actor=None, critic=None,
                 rollout_workers=1, randomize_slot=False,
                 snapshot_every=0, selfplay_prob=0.0, **_ignored):
    """Run `total_updates` of hybrid PPO. Returns (actor, critic, history).

    Builds actor + critic with dropout=0 (the ppo_update_hybrid determinism assert
    requires it). `forward_bias` follows the anneal+zero-hold schedule; `ent_coef`
    anneals linearly from cfg.ent_coef to cfg.ent_coef_final. The first
    `critic_warmup` updates fit the critic only (actor frozen). Unknown kwargs are
    ignored (`**_ignored`) so callers can pass extra diagnostics flags."""
    cfg = cfg or HybridPPOConfig()
    actor = actor or SymbolicTransformerActor(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=0.0)
    critic = critic or CentralizedCritic(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=0.0)
    actor.to(device)
    critic.to(device)
    opt = make_optimizer(actor, critic, cfg)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def _log(rec):
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")

    # Parallel rollout: spawn a reusable worker pool (spawn ctx avoids the
    # CUDA-fork footgun — workers rebuild the actor on CPU from a state_dict).
    # Effective parallelism is min(rollout_workers, episodes_per_update).
    pool = None
    if rollout_workers and rollout_workers > 1:
        n_proc = min(rollout_workers, episodes_per_update)
        if n_proc > 1:
            pool = multiprocessing.get_context("spawn").Pool(n_proc)

    history = []
    frozen_paths = []        # self-play league: frozen actor snapshots (grows over training)
    pbar = tqdm(range(total_updates), desc="hybrid-ppo", unit="upd")
    try:
      for update in pbar:
        fb = forward_bias_value(update, total_updates, forward_bias_init)
        frac = update / max(1, total_updates - 1)
        ent = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * frac
        cfg_u = replace(cfg, ent_coef=ent)
        seed = seed0 + update * episodes_per_update
        if pool is not None:
            buf = collect_hybrid_rollout_parallel(
                actor, learner_slots, episodes_per_update, seed,
                pool=pool, num_workers=rollout_workers, trigger=trigger,
                post_params=post_params, forward_bias=fb,
                anti_idle_penalty=anti_idle_penalty,
                opponent_names=opponent_names, progress=False,
                randomize_slot=randomize_slot, frozen_paths=frozen_paths,
                selfplay_prob=selfplay_prob)
        else:
            buf = collect_hybrid_rollout(
                actor, learner_slots, episodes_per_update, seed,
                trigger=trigger, post_params=post_params, forward_bias=fb,
                anti_idle_penalty=anti_idle_penalty,
                opponent_names=opponent_names, randomize_slot=randomize_slot,
                frozen_paths=frozen_paths, selfplay_prob=selfplay_prob)
        if buf.size == 0:
            rec = {"update": update, "size": 0, "skipped": True}
            history.append(rec)
            _log(rec)
            continue
        vals = critic_values(critic, buf.gstate, buf.gscalar, device)
        adv, ret = compute_advantages(buf.rewards, vals, buf.dones,
                                      GAMMA, GAE_LAMBDA)
        if update < critic_warmup:
            m = critic_only_update(critic, opt, buf, ret, cfg_u, device)
            m.setdefault("policy_loss", 0.0)
            m.setdefault("approx_kl", 0.0)
            m["n_active"] = 0
        else:
            m = ppo_update_hybrid(actor, critic, opt, buf, adv, ret, cfg_u, device)
        m["update"] = update
        m["size"] = int(buf.size)
        m["forward_bias"] = fb
        m["ent_coef"] = ent
        m["post_handover_return"] = float(buf.env_rewards.sum()
                                          / max(1, float(buf.dones.sum())))
        history.append(m)
        _log(m)
        pbar.set_postfix(ret=round(m.get("post_handover_return", 0.0), 1),
                         kl=round(m.get("approx_kl", 0.0), 4),
                         ent=round(m.get("entropy", 0.0), 2),
                         ev=round(m.get("explained_variance", 0.0), 2),
                         lr=f"{m.get('actor_lr', 0.0):.1e}",
                         ploss=round(m.get("policy_loss", 0.0), 3),
                         n=int(m.get("n_active", 0)))
        if (checkpoint_dir and checkpoint_every
                and (update + 1) % checkpoint_every == 0):
            actor.save_checkpoint(f"{checkpoint_dir}/actor_{update + 1}.pt")
        # self-play league: snapshot the current actor and add it to the frozen
        # opponent pool so future rollouts face past versions of the agent.
        if (checkpoint_dir and snapshot_every
                and (update + 1) % snapshot_every == 0):
            snap = f"{checkpoint_dir}/snap_{update + 1}.pt"
            actor.save_checkpoint(snap)
            frozen_paths.append(snap)
            m["league_size"] = len(frozen_paths)
        if (eval_fn is not None and eval_every
                and (update + 1) % eval_every == 0):
            ev = eval_fn(update + 1, actor)
            if ev:
                _log({"update": update + 1, "eval": ev})
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return actor, critic, history
