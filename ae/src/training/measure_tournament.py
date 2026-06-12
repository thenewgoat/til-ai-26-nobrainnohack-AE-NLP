"""Read-only scripted 8-way tournament: measure which AE strategy wins.

Each episode places a seeded random sample of six of the eight scripted
strategies in the six fixed novice slots, runs the episode, and records
per-slot cumulative reward. Per-episode outcomes append to logs/tournament.jsonl
(crash-safe, resumable); at the end the script aggregates and writes
logs/tournament_summary.json. See the design spec for the decision rule.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

# ae/src holds scripted/ and features.py; this file now lives at ae/src/training/,
# so the parent directory is ae/src/. pytest's conftest.py also adds it for tests;
# a standalone run needs this.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from evaluate import ScriptedAgent
from til_environment import bomberman_env
from til_environment.config import default_config

SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]

# The named scripted strategies. test_strategy_names_match_registry asserts
# this stays in sync with scripted.strategies.STRATEGIES.
STRATEGY_NAMES = ["balanced", "balanced_extreme", "base_rusher",
                  "base_rusher_extreme", "collector", "camper", "forager",
                  "lean_rush", "defender", "adaptive",
                  "balanced_extreme_opening"]


def assign_roster(episode_seed):
    """Return {slot: strategy} — a seeded random sample of `len(SLOTS)`
    strategies, one per slot, drawn without replacement from STRATEGY_NAMES."""
    perm = random.Random(episode_seed).sample(STRATEGY_NAMES, len(SLOTS))
    return dict(zip(SLOTS, perm))


def episode_outcome(roster, cumulative_rewards):
    """Winner slot/strategy/value plus winner_margin (top minus second-best)."""
    ranked = sorted(cumulative_rewards.items(), key=lambda kv: kv[1],
                    reverse=True)
    winner_slot, winner_value = ranked[0]
    second_value = ranked[1][1]
    return {
        "winner_slot": winner_slot,
        "winner_strategy": roster[winner_slot],
        "winner_value": float(winner_value),
        "winner_margin": float(winner_value - second_value),
    }


def _distribution(values):
    """mean/min/max/p10/p50/p90 of a list of floats."""
    if not values:
        return {k: 0.0 for k in ("mean", "min", "max", "p10", "p50", "p90")}
    arr = np.asarray(values, dtype=float)
    p10, p50, p90 = (float(x) for x in np.percentile(arr, [10, 50, 90]))
    return {"mean": float(arr.mean()), "min": float(arr.min()),
            "max": float(arr.max()), "p10": p10, "p50": p50, "p90": p90}


def aggregate(records):
    """Summary statistics over a list of episode outcome records.

    Each record has: roster {slot: strategy}, cumulative_rewards {slot: float},
    winner_slot, winner_strategy, winner_value, winner_margin.
    """
    n = len(records)
    per_strategy = {}
    for strat in STRATEGY_NAMES:
        wins = sum(1 for r in records if r["winner_strategy"] == strat)
        rewards = []
        for r in records:
            slot = next((s for s, v in r["roster"].items() if v == strat), None)
            if slot is not None:
                rewards.append(r["cumulative_rewards"][slot])
        per_strategy[strat] = {
            "win_count": wins,
            "win_rate": wins / n if n else 0.0,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "median_reward": float(np.median(rewards)) if rewards else 0.0,
        }

    shares = [per_strategy[s]["win_rate"] for s in STRATEGY_NAMES]
    entropy = -sum(p * math.log2(p) for p in shares if p > 0.0)

    win_rate_by_slot = {
        slot: (sum(1 for r in records if r["winner_slot"] == slot) / n
               if n else 0.0)
        for slot in SLOTS
    }

    return {
        "episodes": n,
        "per_strategy": per_strategy,
        "win_entropy_bits": entropy,
        "win_entropy_normalized": (entropy / math.log2(len(STRATEGY_NAMES))
                                   if n else 0.0),
        "winner_value": _distribution([r["winner_value"] for r in records]),
        "winner_margin": _distribution([r["winner_margin"] for r in records]),
        "win_rate_by_slot": win_rate_by_slot,
    }


def build_env():
    """Create the novice Bomberman env, configured exactly as evaluate.py."""
    cfg = default_config()
    cfg.env.novice = True
    return bomberman_env.basic_env(env_wrappers=[], cfg=cfg)


def run_episode(env, roster, episode_seed):
    """Run one episode with `roster` {slot: strategy}; return {slot: cumulative
    reward}. Mirrors evaluate.py's agent_iter accumulation loop."""
    agents = {slot: ScriptedAgent(strat) for slot, strat in roster.items()}
    random.seed(episode_seed)
    env.reset(seed=episode_seed)
    for agent in agents.values():
        agent.reset()
    totals = {slot: 0.0 for slot in roster}
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        totals[slot] += float(reward)
        if term or trunc:
            env.step(None)
            continue
        env.step(agents[slot].action(obs))
    return totals


def read_records(jsonl_path):
    """Load all episode records from a tournament.jsonl file (or [] if none)."""
    if not os.path.exists(jsonl_path):
        return []
    records = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pending_episodes(n_episodes, jsonl_path):
    """Episode ids in range(n_episodes) not already present in jsonl_path."""
    done = {r["episode_id"] for r in read_records(jsonl_path)}
    return [i for i in range(n_episodes) if i not in done]


DOMINANCE_THRESHOLD = 0.70          # win-rate at/above which a strategy "dominates"


def decision_hint(summary):
    """One-line guidance mapping the win distribution to the spec's decision rule."""
    top_strategy, top_stats = max(summary["per_strategy"].items(),
                                  key=lambda kv: kv[1]["win_rate"])
    top_rate = top_stats["win_rate"]
    if top_rate >= DOMINANCE_THRESHOLD:
        return (f"Case 1: '{top_strategy}' dominates "
                f"({top_rate:.0%} wins). DAgger-clone it directly; do not "
                f"build clone-the-winner.")
    return (f"Wins are spread (entropy {summary['win_entropy_bits']:.2f} bits; "
            f"top '{top_strategy}' at {top_rate:.0%}). Inspect whether the "
            f"winning strategies are behaviorally compatible: compatible -> "
            f"Approach C (restricted-roster BC), contradictory -> Approach B "
            f"(strategy-conditioned BC).")


def print_summary(summary):
    """Print the aggregate table and decision hint to stdout."""
    print(f"\n=== Tournament summary ({summary['episodes']} episodes) ===")
    print(f"{'strategy':<22}{'wins':>6}{'win%':>8}{'mean_rwd':>12}"
          f"{'median_rwd':>12}")
    for strat in STRATEGY_NAMES:
        st = summary["per_strategy"][strat]
        print(f"{strat:<22}{st['win_count']:>6}{st['win_rate']:>7.0%} "
              f"{st['mean_reward']:>11.2f}{st['median_reward']:>12.2f}")
    print(f"win entropy: {summary['win_entropy_bits']:.3f} bits "
          f"({summary['win_entropy_normalized']:.2f} normalized)")
    wv, wm = summary["winner_value"], summary["winner_margin"]
    print(f"winner value : mean {wv['mean']:.2f}  "
          f"p10/p50/p90 {wv['p10']:.2f}/{wv['p50']:.2f}/{wv['p90']:.2f}")
    print(f"winner margin: mean {wm['mean']:.2f}  "
          f"p10/p50/p90 {wm['p10']:.2f}/{wm['p50']:.2f}/{wm['p90']:.2f}")
    print("win-rate by slot: " + "  ".join(
        f"{slot}={rate:.0%}" for slot, rate in
        summary["win_rate_by_slot"].items()))
    print(f"\nHINT: {summary['decision_hint']}")


def main():
    parser = argparse.ArgumentParser(description="Scripted AE tournament measurement")
    parser.add_argument("--episodes", type=int, default=120,
                        help="number of tournament episodes (default 120)")
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    jsonl_path = os.path.join(log_dir, "tournament.jsonl")
    summary_path = os.path.join(log_dir, "tournament_summary.json")

    pending = pending_episodes(args.episodes, jsonl_path)
    print(f"{len(pending)} episodes to run, "
          f"{args.episodes - len(pending)} already done", flush=True)

    env = build_env()
    t0 = time.time()
    with open(jsonl_path, "a") as fh:
        for ep in pending:
            roster = assign_roster(ep)
            totals = run_episode(env, roster, ep)
            outcome = episode_outcome(roster, totals)
            record = {"episode_id": ep, "episode_seed": ep, "roster": roster,
                      "cumulative_rewards": totals, **outcome}
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(f"  ep {ep}: winner {outcome['winner_strategy']} "
                  f"value {outcome['winner_value']:.2f} "
                  f"margin {outcome['winner_margin']:.2f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    env.close()

    summary = aggregate(read_records(jsonl_path))
    summary["decision_hint"] = decision_hint(summary)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print_summary(summary)
    print(f"\nsaved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
