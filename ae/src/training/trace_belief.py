"""Single-episode trace: PROVE the truth agent forages off ground truth.

Runs ONE novice episode with the collector strategy in slot 0, using the
PerfectScriptedAgent (collectible presence overridden to env ground truth).
At every tick where the agent is alive we compute, side by side:

  native  = what the REAL belief would think is present
            ({prior cells} - belief.collected)
  truth   = what is ACTUALLY present right now
            ({prior cells} that hold an ACTIVE collectible in the env)

and the forage layer's chosen action under each. We log:
  * only_truth  : present now, but belief thinks gone (respawn the belief is
                  blind to)  -> the agent SHOULD chase these
  * only_belief : belief thinks present, but actually gone/collecting
  * fa_native vs fa_truth : forage_chain's first action under each set
  * source      : which cascade layer produced the executed action

Then summary counts prove the mechanism is live (or not):
  - ticks where the two sets disagreed
  - ticks where forage_chain chose a DIFFERENT action because of it
  - collection events on only_truth cells (smoking gun: agent collected a
    respawned collectible the belief agent didn't know existed)
"""
import argparse
import random

import numpy as np

from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.decide import act
from scripted.layers import forage_chain
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner
from scripted.strategies import STRATEGIES
from til_environment import bomberman_env
from til_environment.config import default_config
from til_environment.entities.base import EntityStatus
from til_environment.entities.static import Mission, Recon, Resource

ACT_NAME = {0: "FWD", 1: "BACK", 2: "LEFT", 3: "RIGHT", 4: "STAY", 5: "BOMB"}


def true_present(env):
    reg = (getattr(env, "dynamics", None) or env.unwrapped.dynamics).registry
    s = set()
    for cls in (Mission, Recon, Resource):
        for e in reg.query().type(cls).status(EntityStatus.ACTIVE).all():
            s.add((int(e.position[0]), int(e.position[1])))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", default="collector")
    ap.add_argument("--max-print", type=int, default=25)
    args = ap.parse_args()

    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    strategy = STRATEGIES[args.strategy]
    prior = MapPrior.load()
    belief = Belief()
    started = False

    random.seed(args.seed)
    env.reset(seed=args.seed)

    n_alive = 0
    n_setdiff = 0
    n_actiondiff = 0
    n_collect_truthonly = 0
    n_only_truth_ticks = 0
    n_only_belief_ticks = 0
    ever_only_truth = set()
    printed = 0
    prev_collected = set()

    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if slot != "agent_0":
            if not (term or trunc):
                # opponents: first legal action, keep episode moving
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
        n_alive += 1

        # --- the two collectible sets ---
        native = {c: v for c, v in prior.collectibles.items()
                  if c not in belief.collected}
        present = true_present(env)
        truth = {c: v for c, v in prior.collectibles.items() if c in present}
        only_truth = set(truth) - set(native)
        only_belief = set(native) - set(truth)

        # --- forage action under each set (forage_chain is side-effect free) ---
        danger = DangerMap(belief.enemy_bombs, belief)
        planner = build_planner(belief, danger)
        belief.remaining_collectibles = lambda n=native: n
        fa_native = forage_chain(belief, danger, planner, strategy.params)
        belief.remaining_collectibles = lambda t=truth: t
        fa_truth = forage_chain(belief, danger, planner, strategy.params)

        setdiff = bool(only_truth or only_belief)
        actiondiff = (fa_native != fa_truth)
        if setdiff:
            n_setdiff += 1
        if actiondiff:
            n_actiondiff += 1
        if only_truth:
            n_only_truth_ticks += 1
            ever_only_truth |= only_truth
        if only_belief:
            n_only_belief_ticks += 1

        # smoking gun: did the agent just collect a cell that was only_truth?
        newly = belief.collected - prev_collected
        collected_truthonly = [c for c in newly if c in only_belief]  # belief
        # a cell the belief thought present and just got marked collected is
        # normal; the interesting case is the agent SITTING on a respawned
        # (only_truth) cell that belief was blind to:
        on_truthonly = belief.location in only_truth
        if on_truthonly:
            n_collect_truthonly += 1
        prev_collected = set(belief.collected)

        if setdiff and printed < args.max_print:
            printed += 1
            print(f"t={step:3d} loc={belief.location} "
                  f"|native|={len(native)} |truth|={len(truth)} "
                  f"only_truth={sorted(only_truth)} "
                  f"only_belief={sorted(only_belief)} "
                  f"forage: native={ACT_NAME.get(fa_native)} "
                  f"truth={ACT_NAME.get(fa_truth)} "
                  f"{'<-- DIFFERS' if actiondiff else ''}"
                  f"{' [on respawn belief is blind to]' if on_truthonly else ''}",
                  flush=True)

        # execute the TRUTH agent for real
        belief.remaining_collectibles = lambda t=truth: t
        action = int(act(belief, obs["action_mask"], strategy))
        env.step(action)

    env.close()
    print("\n=== PROOF SUMMARY (seed %d, %s) ===" % (args.seed, args.strategy))
    print(f"alive ticks                         : {n_alive}")
    print(f"ticks belief vs truth sets DISAGREE : {n_setdiff} "
          f"({100*n_setdiff/max(n_alive,1):.1f}%)")
    print(f"ticks forage action DIFFERS         : {n_actiondiff} "
          f"({100*n_actiondiff/max(n_alive,1):.1f}%)")
    print(f"ticks agent stood on a respawn the "
          f"belief was blind to: {n_collect_truthonly}")
    print(f"ticks only_belief (belief over-counts)  : {n_only_belief_ticks}")
    print(f"ticks only_truth  (belief blind to cell): {n_only_truth_ticks}")
    print(f"distinct cells ever only_truth          : "
          f"{len(ever_only_truth)} {sorted(ever_only_truth)[:20]}")


if __name__ == "__main__":
    main()
