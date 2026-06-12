"""Collect balanced-family winner trajectories from a scripted tournament.

Each episode places the six scripted strategies in the six fixed novice slots
by a seeded random permutation, runs the episode recording every slot's
(obs, action) stream, and — when a balanced-family agent wins cleanly (the
quality gate) — keeps the winner's raw stream AND replays it through a fresh
FeatureBuilder to also build the winner's five-tensor BCSamples. The output
file stores `samples` (flat, directly train_bc-able), `episodes[*].steps`
(durable raw-obs archive), and `meta`. See the design spec.
"""
import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import torch

# ae/src holds scripted/, imported transitively via evaluate -> features.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from bc import BCSample
from evaluate import ScriptedAgent
from features import (BASE_FIELDS, FeatureBuilder,
                      NUM_BASES, RAW_AGENT_SHAPE, RAW_BASE_SHAPE,
                      STACKED_GRID_CHANNELS, STACKED_SCALARS)
from measure_tournament import assign_roster, build_env, episode_outcome

# The two compatible "balanced family" strategies — the only winners kept.
FAMILY = ("balanced", "balanced_extreme")


def passes_gate(outcome, value_threshold, margin_threshold):
    """True if the episode's winner is a strong balanced-family agent."""
    return (outcome["winner_strategy"] in FAMILY
            and outcome["winner_value"] >= value_threshold
            and outcome["winner_margin"] >= margin_threshold)


def run_episode_recorded(env, roster, episode_seed):
    """Run one episode with `roster` {slot: strategy}.

    Returns ({slot: cumulative reward}, {slot: [(obs_copy, action), ...]}),
    recording a deep-copied observation and the action for every acting step.
    Mirrors evaluate.py's agent_iter loop.
    """
    agents = {slot: ScriptedAgent(strat) for slot, strat in roster.items()}
    random.seed(episode_seed)
    env.reset(seed=episode_seed)
    for agent in agents.values():
        agent.reset()
    totals = {slot: 0.0 for slot in roster}
    streams = {slot: [] for slot in roster}
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        totals[slot] += float(reward)
        if term or trunc:
            env.step(None)
            continue
        action = int(agents[slot].action(obs))
        streams[slot].append((copy.deepcopy(obs), action))
        env.step(action)
    return totals, streams


def build_winner_samples(winner_steps):
    """Replay a winner's recorded obs stream through a fresh FeatureBuilder.

    `winner_steps` is [{obs, action}, ...] from run_episode_recorded — the
    winner's contiguous-from-step-0 acting steps, so the stateful belief stays
    in sync. Default teacher_strategy="balanced" matches what NeuralAEManager
    uses at serving, so collection/serving features agree. Returns one BCSample
    per step in the post-refactor five-tensor form.
    """
    fb = FeatureBuilder()
    samples = []
    for step in winner_steps:
        obs = step["obs"]
        grid, base_feats, raw_agent, raw_base, scalar = fb.build(obs)
        mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
        samples.append(BCSample(
            grid=grid, base_feats=base_feats, raw_agent=raw_agent,
            raw_base=raw_base, scalar=scalar, mask=mask.copy(),
            action=step["action"]))
    return samples


def collect(n_episodes, value_threshold, margin_threshold):
    """Run `n_episodes` tournaments; keep winning balanced-family streams and
    their replayed BCSamples.

    Returns (samples, episodes, meta):
      - samples: flat list[BCSample] across all kept episodes (train_bc-ready);
      - episodes: list of {episode_id, winner_strategy, winner_value,
        winner_margin, steps:[{obs, action}, ...]} for re-replay;
      - meta: run-level summary including the feature_builder_contract string
        for staleness detection.
    """
    env = build_env()
    t0 = time.time()
    samples_all = []
    episodes = []
    try:
        for ep in range(n_episodes):
            roster = assign_roster(ep)
            totals, streams = run_episode_recorded(env, roster, ep)
            outcome = episode_outcome(roster, totals)
            kept = passes_gate(outcome, value_threshold, margin_threshold)
            if kept:
                winner_stream = streams[outcome["winner_slot"]]
                winner_steps = [{"obs": obs, "action": action}
                                for obs, action in winner_stream]
                ep_samples = build_winner_samples(winner_steps)
                samples_all.extend(ep_samples)
                episodes.append({
                    "episode_id": ep,
                    "winner_strategy": outcome["winner_strategy"],
                    "winner_value": outcome["winner_value"],
                    "winner_margin": outcome["winner_margin"],
                    "steps": winner_steps,
                })
            print(f"  ep {ep}: winner {outcome['winner_strategy']} "
                  f"value {outcome['winner_value']:.1f} "
                  f"margin {outcome['winner_margin']:.1f} "
                  f"{'KEPT' if kept else 'skip'}  [{time.time() - t0:.0f}s]",
                  flush=True)
    finally:
        env.close()

    meta = {
        "episodes_run": n_episodes,
        "episodes_kept": len(episodes),
        "total_samples": len(samples_all),
        "value_threshold": value_threshold,
        "margin_threshold": margin_threshold,
        "duration_seconds": time.time() - t0,
        "feature_builder_contract": (
            f"grid[{STACKED_GRID_CHANNELS},16,16] "
            f"base_feats[{NUM_BASES},{BASE_FIELDS}] "
            f"raw_agent{tuple(RAW_AGENT_SHAPE)} "
            f"raw_base{tuple(RAW_BASE_SHAPE)} "
            f"scalar[{STACKED_SCALARS}]"),
        "winners": [{"episode_id": e["episode_id"],
                     "winner_strategy": e["winner_strategy"],
                     "winner_value": e["winner_value"],
                     "winner_margin": e["winner_margin"]}
                    for e in episodes],
    }
    return samples_all, episodes, meta


def main():
    parser = argparse.ArgumentParser(
        description="Collect balanced-family winner trajectories + BCSamples")
    parser.add_argument("--episodes", type=int, default=400,
                        help="number of tournament episodes (default 400)")
    parser.add_argument("--value-threshold", type=float, default=275.0,
                        help="minimum winner cumulative reward (default 275)")
    parser.add_argument("--margin-threshold", type=float, default=10.0,
                        help="minimum winner victory margin (default 10)")
    args = parser.parse_args()

    samples, episodes, meta = collect(args.episodes, args.value_threshold,
                                      args.margin_threshold)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "family_winner.pt")
    torch.save({"samples": samples, "episodes": episodes, "meta": meta},
               out_path)

    print(f"\nkept {meta['episodes_kept']}/{meta['episodes_run']} episodes, "
          f"{meta['total_samples']} winner BCSamples")
    print(f"saved {out_path}  [{meta['duration_seconds']:.0f}s]")


if __name__ == "__main__":
    main()
