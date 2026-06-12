"""Probe: how far ahead does forage_chain plan, and where do respawned
(belief-blind) cells rank? Answers 'does truth fail to kick in because the
planner doesn't look far enough?'

Re-implements forage_chain's chain construction faithfully (copied from
scripted.layers.forage_chain) but returns the whole chain + costs so we can
measure planning depth, and on each tick logs the distance/rate of the nearest
respawned (only_truth) cell vs the chosen first target.
"""
import argparse
import random

import numpy as np

from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.layers import (_bfs_facing, _centre_prox, _enemy_distances,
                             _trace_first_action)
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner
from scripted.strategies import STRATEGIES
from til_environment import bomberman_env
from til_environment.config import default_config
from til_environment.entities.base import EntityStatus
from til_environment.entities.static import Mission, Recon, Resource


def true_present(env):
    reg = (getattr(env, "dynamics", None) or env.unwrapped.dynamics).registry
    s = set()
    for cls in (Mission, Recon, Resource):
        for e in reg.query().type(cls).status(EntityStatus.ACTIVE).all():
            s.add((int(e.position[0]), int(e.position[1])))
    return s


def _min_cost_facing(cost, c):
    best = None
    for f in range(4):
        st = (c, f)
        if st in cost and (best is None or cost[st] < best):
            best = cost[st]
    return best


def chain_debug(belief, danger, remaining, params):
    """Faithful copy of forage_chain's loop, returning the chain + costs."""
    enemy_distances = _enemy_distances(belief)
    start_state = (belief.location, belief.facing)
    chained = []
    chain_value = 0.0
    chain_cost = 0
    current_state = start_state
    first_action = None
    first_cell = None
    first_cost = None

    def rate_of(c, v, min_cost):
        boost = 1.0 + params.centre_value_weight * _centre_prox(belief, c)
        k = sum(1 for ed in enemy_distances
                if (c in ed) and ed[c] + 0.5 < min_cost)
        if k > 0:
            boost *= params.contested_value_factor ** k
        return (v * boost) / min_cost

    first_cost_map = None
    while True:
        cost, parent = _bfs_facing(belief, danger, current_state)
        if first_cost_map is None:
            first_cost_map = cost  # BFS from the agent's CURRENT position
        best_c, best_cost, best_facing, best_rate = None, 0, 0, 0.0
        for c, v in remaining.items():
            if c in chained:
                continue
            mc = _min_cost_facing(cost, c)
            if mc is None or mc == 0:
                continue
            r = rate_of(c, v, mc)
            if r > best_rate:
                best_rate, best_c, best_cost, best_facing = r, c, mc, _min_cost_facing(cost, c) and \
                    [f for f in range(4) if (c, f) in cost and cost[(c, f)] == mc][0]
        if best_c is None:
            break
        if chain_cost > 0 and best_rate <= chain_value / chain_cost:
            break
        if first_action is None:
            first_action = _trace_first_action(parent, current_state,
                                               (best_c, best_facing))
            first_cell, first_cost = best_c, best_cost
        chained.append(best_c)
        chain_value += remaining[best_c]
        chain_cost += best_cost
        current_state = (best_c, best_facing)

    return {
        "chain": chained, "chain_len": len(chained), "chain_cost": chain_cost,
        "first_cell": first_cell, "first_cost": first_cost,
        "first_rate": (rate_of(first_cell, remaining[first_cell], first_cost)
                       if first_cell else 0.0),
        "cost_from_now": first_cost_map, "rate_of": rate_of,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", default="collector")
    args = ap.parse_args()

    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    strategy = STRATEGIES[args.strategy]
    params = strategy.params
    prior = MapPrior.load()
    belief = Belief()
    started = False
    random.seed(args.seed)
    env.reset(seed=args.seed)

    chain_lens, chain_costs, first_costs = [], [], []
    ot_ticks = 0
    ot_in_chain = 0
    ot_is_first = 0
    ot_dist = []
    ot_rate_ratio = []   # best only_truth rate / chosen first-target rate

    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if slot != "agent_0":
            if not (term or trunc):
                mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
                legal = np.flatnonzero(mask)
                env.step(int(legal[0]) if len(legal) else 4)
            else:
                env.step(None)
            continue
        if term or trunc:
            env.step(None)
            continue
        step = int(np.asarray(obs["step"]).flat[0])
        if step == 0 or not started:
            prior.identify_team(obs["base_location"])
            belief.reset(prior)
            started = True
        belief.update(obs)

        native = {c: v for c, v in prior.collectibles.items()
                  if c not in belief.collected}
        present = true_present(env)
        truth = {c: v for c, v in prior.collectibles.items() if c in present}
        only_truth = set(truth) - set(native)

        danger = DangerMap(belief.enemy_bombs, belief)
        info = chain_debug(belief, danger, truth, params)
        if info["first_cell"] is not None:
            chain_lens.append(info["chain_len"])
            chain_costs.append(info["chain_cost"])
            first_costs.append(info["first_cost"])

        if only_truth and info["first_cell"] is not None:
            ot_ticks += 1
            cost = info["cost_from_now"]
            # nearest respawned cell distance + best rate among them
            best_d, best_r = None, 0.0
            for c in only_truth:
                mc = _min_cost_facing(cost, c)
                if mc is None:
                    continue
                if best_d is None or mc < best_d:
                    best_d = mc
                r = info["rate_of"](c, truth[c], mc)
                best_r = max(best_r, r)
            if best_d is not None:
                ot_dist.append(best_d)
                if info["first_rate"] > 0:
                    ot_rate_ratio.append(best_r / info["first_rate"])
            if any(c in info["chain"] for c in only_truth):
                ot_in_chain += 1
            if info["first_cell"] in only_truth:
                ot_is_first += 1

        # execute truth agent
        belief.remaining_collectibles = lambda t=truth: t
        from scripted.decide import act
        env.step(int(act(belief, obs["action_mask"], strategy)))

    env.close()

    def stats(a):
        a = np.array(a, dtype=float)
        if len(a) == 0:
            return "n=0"
        return (f"n={len(a)} mean={a.mean():.1f} median={np.median(a):.0f} "
                f"p90={np.percentile(a,90):.0f} max={a.max():.0f}")

    print(f"\n=== HORIZON PROBE (seed {args.seed}, {args.strategy}) ===")
    print(f"chain length (collectibles chained/tick): {stats(chain_lens)}")
    print(f"chain cost   (ticks to chain end)       : {stats(chain_costs)}")
    print(f"first-target cost (committed reach)      : {stats(first_costs)}")
    print(f"\nrespawned (only_truth) cells present on {ot_ticks} ticks")
    print(f"  dist agent->nearest respawned cell     : {stats(ot_dist)}")
    print(f"  a respawned cell entered the chain     : {ot_in_chain} ticks")
    print(f"  a respawned cell was the FIRST target  : {ot_is_first} ticks")
    print(f"  best_respawn_rate / chosen_rate        : {stats(ot_rate_ratio)}")
    print("  (ratio<1 => respawned cell correctly judged worse, not a horizon "
          "miss)")


if __name__ == "__main__":
    main()
