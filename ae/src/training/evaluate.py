"""Standalone evaluation harness for AE agents.

Runs the novice AEC env over a fixed seed set, scoring a policy-under-test in
slot agent_0 against a configurable opponent set in agent_1..agent_5. Reports
mean score and win-rate. Used to gate the BC stage (spec C §6.1) and rung
promotion (§6.2). Regression checks must always pass the SAME seeds + opponents
to two checkpoints — never compare across opponent sets.
"""
import random
from dataclasses import dataclass

import numpy as np
import torch

from features import FeatureBuilder
from policy import SymbolicTransformerActor
from scripted import greedy as _greedy_mod
from scripted.belief import Belief
from scripted.decide import act
from scripted.map_prior import MapPrior
from scripted.strategies import STRATEGIES
from til_environment import bomberman_env
from til_environment.config import default_config

EPISODE_LEN = 200


# ----- agent adapters: each maps observation -> action int ---------------- #
class RandomAgent:
    """Uniform-random over legal actions."""
    name = "random"

    def reset(self):
        pass

    def action(self, observation):
        mask = np.asarray(observation["action_mask"], dtype=bool).reshape(-1)
        legal = np.flatnonzero(mask)
        return int(random.choice(legal)) if len(legal) else 4


class ScriptedAgent:
    """A named scripted strategy."""

    def __init__(self, strategy_name="balanced"):
        self.name = f"scripted:{strategy_name}"
        self.strategy = STRATEGIES[strategy_name]
        self.prior = MapPrior.load()
        self.belief = Belief()
        self._started = False

    def reset(self):
        self._started = False

    def action(self, observation):
        step = int(np.asarray(observation["step"]).flat[0])
        if step == 0 or not self._started:
            self.prior.identify_team(observation["base_location"])
            self.belief.reset(self.prior)
            self._started = True
        self.belief.update(observation)
        return int(act(self.belief, observation["action_mask"], self.strategy))


class NoisyScriptedAgent:
    """ScriptedAgent wrapper that with prob ε replaces the strategy's action
    with a uniform random LEGAL action. Used in training rollouts only; eval
    constructs plain ScriptedAgent so eval scores stay comparable."""

    def __init__(self, strategy_name, epsilon, rng=None):
        self._inner = ScriptedAgent(strategy_name)
        self.name = f"scripted:{strategy_name}"   # bookkeeping-equivalent to inner
        self._eps = float(epsilon)
        self._rng = rng or random.Random()

    def reset(self):
        self._inner.reset()

    def action(self, observation):
        if self._eps > 0 and self._rng.random() < self._eps:
            mask = np.asarray(observation["action_mask"], dtype=bool).reshape(-1)
            legal = np.flatnonzero(mask)
            if len(legal):
                return int(self._rng.choice(legal.tolist()))
        return self._inner.action(observation)


class GreedyAgent:
    """The standalone greedy AE agent (A* to nearest live enemy base)."""
    name = "greedy"

    def __init__(self):
        self.prior = MapPrior.load()
        self.belief = Belief()
        self._started = False

    def reset(self):
        self._started = False

    def action(self, observation):
        step = int(np.asarray(observation["step"]).flat[0])
        if step == 0 or not self._started:
            self.prior.identify_team(observation["base_location"])
            self.belief.reset(self.prior)
            self._started = True
        self.belief.update(observation)
        return int(_greedy_mod.act(self.belief, observation["action_mask"]))


class NeuralAgent:
    """A trained SymbolicTransformerActor. By default samples from a
    masked Categorical at temperature 1.0 (uniform over legal when logits
    are flat); higher T → more random opponent for league diversity.

    Device-aware: input tensors are moved to the actor's device on every
    call, so a CUDA-resident actor works without the caller having to
    `.to('cpu')`-copy first."""

    def __init__(self, actor, name="neural", temperature=1.0):
        self.name = name
        self.actor = actor
        self.actor.eval()
        self.fb = FeatureBuilder()
        self._t = float(temperature)
        # Cache once; the actor's device shouldn't change after construction.
        self._device = next(actor.parameters()).device

    def reset(self):
        self.fb = FeatureBuilder()

    def action(self, observation):
        grid, base_feats, raw_agent, raw_base, scalar = self.fb.build(
            observation)
        mask = np.asarray(observation["action_mask"], dtype=bool).reshape(-1)
        d = self._device
        with torch.no_grad():
            logits = self.actor(
                torch.from_numpy(grid).unsqueeze(0).to(d),
                torch.from_numpy(base_feats).unsqueeze(0).to(d),
                torch.from_numpy(raw_agent).unsqueeze(0).to(d),
                torch.from_numpy(raw_base).unsqueeze(0).to(d),
                torch.from_numpy(scalar).unsqueeze(0).to(d),
            )[0]
        mask_t = torch.from_numpy(mask).to(d)
        scaled = torch.where(mask_t, logits / self._t,
                              torch.full_like(logits, -1e8))
        dist = torch.distributions.Categorical(logits=scaled)
        return int(dist.sample().item())


@dataclass
class EvalResult:
    episodes: int
    mean_score: float
    win_rate: float
    per_seed_scores: list


def evaluate_policy(agent, opponents, seeds, randomize_slot=False,
                    opponent_bank=None, setups=None):
    """Run one episode per seed.

    Default: `agent` in agent_0, `opponents` in agent_1..5 (fixed scenario).

    `randomize_slot=True`: per seed, place `agent` in a seed-derived random slot
    (0-5). `opponent_bank` (list of strategy names): per seed, draw the 5 opponents
    from it (seed-derived) instead of using `opponents`. Both derivations are pure
    functions of the seed, so paired A/B arms get the SAME slot + opponents per
    seed. The board stays novice/seed-88 (not un-pinned); diversity is from
    spawn-slot + opponent-mix only.

    `setups`: an explicit list of `(learner_idx, seed)` cells (overrides `seeds`/
    `randomize_slot`) — used by the structured grid eval to cover every slot
    equally. Opponents per cell are still drawn from `opponent_bank` via the seed.

    Returns an EvalResult for `agent`. The eval env always uses the UNMODIFIED
    default reward config (spec C §6 reward shaping is training-only).
    """
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    slots = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]

    cells = list(setups) if setups is not None else [(None, s) for s in seeds]
    scores, wins = [], 0
    for fixed_idx, seed in cells:
        setup = random.Random(seed)          # seed-derived setup (paired A/B safe)
        if fixed_idx is not None:
            learner_idx = fixed_idx
        else:
            learner_idx = setup.randrange(len(slots)) if randomize_slot else 0
        if opponent_bank:
            opp_list = [ScriptedAgent(setup.choice(opponent_bank))
                        for _ in range(len(slots) - 1)]
        else:
            opp_list = list(opponents)
        by_slot = {slots[learner_idx]: agent}
        oi = 0
        for i in range(len(slots)):
            if i == learner_idx:
                continue
            by_slot[slots[i]] = opp_list[oi]
            oi += 1

        random.seed(seed)                    # global RNG for episode stochasticity
        env.reset(seed=seed)
        for a in by_slot.values():
            a.reset()
        totals = {s: 0.0 for s in slots}
        for slot in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            totals[slot] += float(reward)
            if term or trunc:
                env.step(None)
                continue
            env.step(by_slot[slot].action(obs))
        my = totals[slots[learner_idx]]
        scores.append(my)
        if my > max(totals[s] for i, s in enumerate(slots) if i != learner_idx):
            wins += 1
    env.close()
    return EvalResult(
        episodes=len(cells),
        mean_score=float(np.mean(scores)) if scores else 0.0,
        win_rate=wins / len(cells) if cells else 0.0,
        per_seed_scores=scores,
    )
