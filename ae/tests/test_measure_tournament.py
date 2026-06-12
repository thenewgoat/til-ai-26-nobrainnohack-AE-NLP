"""Tests for the scripted tournament measurement tool."""
import math

from measure_tournament import SLOTS, STRATEGY_NAMES, assign_roster


def test_strategy_names_match_registry():
    from scripted.strategies import STRATEGIES
    assert set(STRATEGY_NAMES) == set(STRATEGIES)


def test_assign_roster_fills_every_slot_with_distinct_strategies():
    roster = assign_roster(0)
    assert set(roster.keys()) == set(SLOTS)
    values = list(roster.values())
    assert len(values) == len(SLOTS)
    assert len(set(values)) == len(SLOTS)              # distinct
    assert set(values) <= set(STRATEGY_NAMES)          # all are known strategies


def test_assign_roster_is_deterministic():
    assert assign_roster(7) == assign_roster(7)


def test_assign_roster_varies_with_seed():
    rosters = {tuple(assign_roster(s).items()) for s in range(20)}
    assert len(rosters) > 1


def test_episode_outcome_blowout():
    from measure_tournament import episode_outcome
    roster = {"agent_0": "balanced", "agent_1": "camper", "agent_2": "collector",
              "agent_3": "base_rusher", "agent_4": "balanced_extreme",
              "agent_5": "base_rusher_extreme"}
    rewards = {"agent_0": 1.0, "agent_1": 2.0, "agent_2": 9.0,
               "agent_3": 0.0, "agent_4": -1.0, "agent_5": 3.0}
    out = episode_outcome(roster, rewards)
    assert out["winner_slot"] == "agent_2"
    assert out["winner_strategy"] == "collector"
    assert out["winner_value"] == 9.0
    assert out["winner_margin"] == 6.0          # 9.0 - 3.0 (second best)


def test_episode_outcome_near_tie():
    from measure_tournament import episode_outcome
    roster = {s: STRATEGY_NAMES[i] for i, s in enumerate(SLOTS)}
    rewards = {"agent_0": 5.0, "agent_1": 4.9, "agent_2": 1.0,
               "agent_3": 0.0, "agent_4": 0.0, "agent_5": 0.0}
    out = episode_outcome(roster, rewards)
    assert out["winner_slot"] == "agent_0"
    assert abs(out["winner_margin"] - 0.1) < 1e-9


def _record(winner_slot, roster, rewards):
    """Build a synthetic episode record for aggregate() tests."""
    from measure_tournament import episode_outcome
    rec = {"roster": roster, "cumulative_rewards": rewards}
    rec.update(episode_outcome(roster, rewards))
    return rec


def _uniform_roster():
    return {s: STRATEGY_NAMES[i] for i, s in enumerate(SLOTS)}


def test_aggregate_win_counts_and_entropy_even():
    from measure_tournament import aggregate
    # _uniform_roster() uses only the first len(SLOTS)=6 strategies; the
    # remainder are absent from every roster so their win_count must be 0.
    roster = _uniform_roster()
    strategies_in_roster = list(roster.values())
    records = []
    for i, slot in enumerate(SLOTS):
        rewards = {s: (10.0 if s == slot else 0.0) for s in SLOTS}
        records.append(_record(slot, roster, rewards))
    summary = aggregate(records)
    assert summary["episodes"] == 6
    for s in strategies_in_roster:
        assert summary["per_strategy"][s]["win_count"] == 1
        assert abs(summary["per_strategy"][s]["win_rate"] - 1 / 6) < 1e-9
    for s in STRATEGY_NAMES:
        if s not in strategies_in_roster:
            assert summary["per_strategy"][s]["win_count"] == 0
    # entropy over the 6 active strategies is log2(6); normalised divides by
    # log2(len(STRATEGY_NAMES)) since the remaining strategies have win_rate=0.
    assert abs(summary["win_entropy_bits"] - math.log2(6)) < 1e-9
    expected_normalized = math.log2(6) / math.log2(len(STRATEGY_NAMES))
    assert abs(summary["win_entropy_normalized"] - expected_normalized) < 1e-9


def test_aggregate_entropy_zero_when_one_strategy_dominates():
    from measure_tournament import aggregate
    roster = _uniform_roster()              # agent_0 -> "balanced"
    records = []
    for _ in range(6):
        rewards = {s: (10.0 if s == "agent_0" else 0.0) for s in SLOTS}
        records.append(_record("agent_0", roster, rewards))
    summary = aggregate(records)
    assert summary["per_strategy"]["balanced"]["win_count"] == 6
    assert summary["win_entropy_bits"] == 0.0


def test_aggregate_per_strategy_mean_reward():
    from measure_tournament import aggregate
    # "balanced" sits in agent_0 in record A, agent_1 in record B
    rec_a = _record("agent_0", _uniform_roster(),
                    {"agent_0": 8.0, "agent_1": 1.0, "agent_2": 0.0,
                     "agent_3": 0.0, "agent_4": 0.0, "agent_5": 0.0})
    roster_b = {"agent_1": "balanced", "agent_0": "camper",
                "agent_2": "collector", "agent_3": "base_rusher",
                "agent_4": "balanced_extreme", "agent_5": "base_rusher_extreme"}
    rec_b = _record("agent_1", roster_b,
                    {"agent_1": 4.0, "agent_0": 1.0, "agent_2": 0.0,
                     "agent_3": 0.0, "agent_4": 0.0, "agent_5": 0.0})
    summary = aggregate([rec_a, rec_b])
    # balanced's rewards: 8.0 (rec_a, agent_0) and 4.0 (rec_b, agent_1)
    assert abs(summary["per_strategy"]["balanced"]["mean_reward"] - 6.0) < 1e-9
    assert summary["winner_value"]["max"] == 8.0
    assert summary["win_rate_by_slot"]["agent_0"] == 0.5


def test_run_episode_returns_per_slot_rewards():
    # Runs one real ~9s scripted episode; asserts the result shape only.
    from measure_tournament import run_episode, build_env
    env = build_env()
    try:
        totals = run_episode(env, assign_roster(0), episode_seed=0)
    finally:
        env.close()
    assert set(totals.keys()) == set(SLOTS)
    assert all(isinstance(v, float) for v in totals.values())


def test_pending_episodes_skips_completed(tmp_path):
    import json
    from measure_tournament import pending_episodes
    jsonl = tmp_path / "tournament.jsonl"
    jsonl.write_text(
        json.dumps({"episode_id": 0}) + "\n"
        + json.dumps({"episode_id": 2}) + "\n")
    assert pending_episodes(5, str(jsonl)) == [1, 3, 4]


def test_pending_episodes_all_when_no_file(tmp_path):
    from measure_tournament import pending_episodes
    missing = tmp_path / "nope.jsonl"
    assert pending_episodes(3, str(missing)) == [0, 1, 2]


def test_decision_hint_flags_dominant_strategy():
    from measure_tournament import decision_hint
    summary = {
        "per_strategy": {s: {"win_rate": 0.0} for s in STRATEGY_NAMES},
        "win_entropy_bits": 0.5,
    }
    summary["per_strategy"]["base_rusher"]["win_rate"] = 0.8
    hint = decision_hint(summary)
    assert "base_rusher" in hint
    assert "Case 1" in hint


def test_decision_hint_flags_spread():
    from measure_tournament import decision_hint
    summary = {
        "per_strategy": {s: {"win_rate": 1 / 6} for s in STRATEGY_NAMES},
        "win_entropy_bits": 2.58,
    }
    hint = decision_hint(summary)
    assert "spread" in hint.lower()


def test_assign_roster_can_select_adaptive():
    # With 10 strategies in the pool and 6 slots, `adaptive` must be drawable
    # by at least some seed. Sample 200 rosters and confirm `adaptive` appears
    # in at least one of them.
    seen = set()
    for seed in range(200):
        seen |= set(assign_roster(seed).values())
    assert "adaptive" in seen, "adaptive never sampled across 200 rosters"


def test_assign_roster_yields_six_distinct_strategies():
    # Sanity: a single roster is 6 distinct strategies all from STRATEGY_NAMES.
    roster = assign_roster(7)
    values = list(roster.values())
    assert len(set(values)) == 6
    assert set(values) <= set(STRATEGY_NAMES)
