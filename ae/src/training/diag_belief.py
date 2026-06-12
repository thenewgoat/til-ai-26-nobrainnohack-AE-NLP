"""Belief-vs-ground-truth foraging diagnostic.

Runs the novice (seed-88) AEC env with a foraging scripted strategy in ONE
chosen slot, twice per episode (paired):

  * BELIEF mode  — the agent forages off its normal Belief (prior minus
    `collected`, updated only from what its viewcone has seen). The real agent.
  * TRUTH  mode  — identical agent, except `belief.remaining_collectibles()`
    is overridden each tick to return prior values ONLY for cells that hold a
    currently-ACTIVE collectible in the env registry. Flips *presence* to
    ground truth (kills respawn/presence drift) while keeping value scale
    identical, isolating the cost of belief error.

The novice map is pinned (seed 88) so layout is identical; we vary the agent's
*starting location* (which of the 6 slots/bases it occupies) and the opponents
(strategy sampled per episode from a pool + epsilon noise). Opponents are paired
per episode (same RNG stream for belief and truth runs).

Run one slot (from repo root):
  PYTHONPATH=ae/src:ae/src/training \
  til-26-ae/.venv/bin/python ae/src/training/diag_belief.py \
    --slot 0 --episodes 30 --out ae/runs/belief_diag
"""
import argparse
import os
import random

import numpy as np

from scripted.belief import Belief
from scripted.decide import act
from scripted.map_prior import MapPrior
from scripted.strategies import STRATEGIES
from til_environment import bomberman_env
from til_environment.config import default_config
from til_environment.entities.base import EntityStatus
from til_environment.entities.static import Mission, Recon, Resource

SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]
OPP_POOL = ["balanced", "base_rusher", "forager", "hunter_killer",
            "adaptive", "camper"]


def _dynamics(env):
    d = getattr(env, "dynamics", None)
    return d if d is not None else env.unwrapped.dynamics


def _true_present_cells(env):
    reg = _dynamics(env).registry
    present = set()
    for cls in (Mission, Recon, Resource):
        for e in reg.query().type(cls).status(EntityStatus.ACTIVE).all():
            present.add((int(e.position[0]), int(e.position[1])))
    return present


class ScriptedAgent:
    def __init__(self, strategy_name):
        self.strategy = STRATEGIES[strategy_name]
        self.prior = MapPrior.load()
        self.belief = Belief()
        self._started = False

    def reset(self):
        self._started = False

    def _sync(self, obs):
        step = int(np.asarray(obs["step"]).flat[0])
        if step == 0 or not self._started:
            self.prior.identify_team(obs["base_location"])
            self.belief.reset(self.prior)
            self._started = True
        self.belief.update(obs)

    def action(self, obs):
        self._sync(obs)
        return int(act(self.belief, obs["action_mask"], self.strategy))


class PerfectScriptedAgent(ScriptedAgent):
    """Same agent, but collectible *presence* is read from env ground truth."""

    def __init__(self, strategy_name, env):
        super().__init__(strategy_name)
        self.env = env

    def action(self, obs):
        self._sync(obs)
        present = _true_present_cells(self.env)
        prior_coll = self.prior.collectibles
        self.belief.remaining_collectibles = (
            lambda: {c: v for c, v in prior_coll.items() if c in present}
        )
        return int(act(self.belief, obs["action_mask"], self.strategy))


class NoisyScripted:
    def __init__(self, strategy_name, eps, rng):
        self._inner = ScriptedAgent(strategy_name)
        self._eps = eps
        self._rng = rng

    def reset(self):
        self._inner.reset()

    def action(self, obs):
        if self._eps > 0 and self._rng.random() < self._eps:
            mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
            legal = np.flatnonzero(mask)
            if len(legal):
                return int(self._rng.choice(legal.tolist()))
        return self._inner.action(obs)


def _build_opponents(slot_seed, ep, eps):
    """5 opponents with strategies sampled from the pool + noise, reproducible
    from (slot_seed, ep) so the belief and truth runs see identical opponents."""
    pick_rng = random.Random(7000 + slot_seed * 991 + ep)
    noise_rng = random.Random(9000 + slot_seed * 991 + ep)
    strats = [pick_rng.choice(OPP_POOL) for _ in range(5)]
    return [NoisyScripted(s, eps, noise_rng) for s in strats]


def _run_episode(env, by_slot, our_slot, seed):
    random.seed(seed)
    env.reset(seed=seed)
    for a in by_slot.values():
        a.reset()
    totals = {s: 0.0 for s in SLOTS}
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        totals[slot] += float(reward)
        if term or trunc:
            env.step(None)
            continue
        env.step(by_slot[slot].action(obs))
    return totals[our_slot]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--strategy", default="collector")
    ap.add_argument("--opp-eps", type=float, default=0.15)
    ap.add_argument("--out", default="ae/runs/belief_diag")
    args = ap.parse_args()

    our_slot = SLOTS[args.slot]
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"slot_{args.slot}.csv")
    rows = []
    for ep in range(args.episodes):
        seed = args.slot * 10000 + ep

        opponents = _build_opponents(args.slot, ep, args.opp_eps)
        by_slot = {our_slot: ScriptedAgent(args.strategy)}
        for opp, s in zip(opponents, [s for s in SLOTS if s != our_slot]):
            by_slot[s] = opp
        b = _run_episode(env, by_slot, our_slot, seed)

        opponents = _build_opponents(args.slot, ep, args.opp_eps)  # same streams
        by_slot = {our_slot: PerfectScriptedAgent(args.strategy, env)}
        for opp, s in zip(opponents, [s for s in SLOTS if s != our_slot]):
            by_slot[s] = opp
        t = _run_episode(env, by_slot, our_slot, seed)

        rows.append((seed, b, t))
        print(f"slot {args.slot} ep {ep:2d}: belief {b:8.2f} | truth {t:8.2f} "
              f"| delta {t - b:+8.2f}", flush=True)

    env.close()
    with open(csv_path, "w") as f:
        f.write("seed,belief,truth\n")
        for seed, b, t in rows:
            f.write(f"{seed},{b},{t}\n")

    b = np.array([r[1] for r in rows])
    t = np.array([r[2] for r in rows])
    d = t - b
    print(f"SUMMARY slot={args.slot} n={len(rows)} "
          f"belief_mean={b.mean():.2f} truth_mean={t.mean():.2f} "
          f"delta_mean={d.mean():+.2f} delta_std={d.std():.2f} "
          f"wins={int((d > 0).sum())}", flush=True)


if __name__ == "__main__":
    main()
