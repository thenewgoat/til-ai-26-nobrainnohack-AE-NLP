"""Tests for the balanced-family winner collection tool."""
from collect_family_winner import passes_gate
from features import STACKED_GRID_CHANNELS, STACKED_SCALARS


def _outcome(strategy, value, margin):
    return {"winner_slot": "agent_0", "winner_strategy": strategy,
            "winner_value": value, "winner_margin": margin}


def test_gate_accepts_strong_family_winner():
    assert passes_gate(_outcome("balanced", 400.0, 100.0), 275.0, 10.0)
    assert passes_gate(_outcome("balanced_extreme", 300.0, 50.0), 275.0, 10.0)


def test_gate_rejects_non_family_winner():
    assert not passes_gate(_outcome("camper", 400.0, 100.0), 275.0, 10.0)
    assert not passes_gate(_outcome("collector", 999.0, 999.0), 275.0, 10.0)


def test_gate_rejects_low_value_winner():
    assert not passes_gate(_outcome("balanced", 200.0, 100.0), 275.0, 10.0)


def test_gate_rejects_low_margin_winner():
    assert not passes_gate(_outcome("balanced", 400.0, 3.0), 275.0, 10.0)


def test_run_episode_recorded_returns_rewards_and_streams():
    # Runs one real ~9s scripted episode; asserts structure only.
    from collect_family_winner import run_episode_recorded
    from measure_tournament import assign_roster, build_env, SLOTS
    env = build_env()
    try:
        totals, streams = run_episode_recorded(env, assign_roster(0),
                                               episode_seed=0)
    finally:
        env.close()
    assert set(totals.keys()) == set(SLOTS)
    assert all(isinstance(v, float) for v in totals.values())
    assert set(streams.keys()) == set(SLOTS)
    for slot in SLOTS:
        stream = streams[slot]
        assert isinstance(stream, list) and len(stream) > 0
        obs, action = stream[0]
        assert isinstance(obs, dict) and "action_mask" in obs
        assert isinstance(action, int)


def test_build_winner_samples_yields_one_per_step_with_correct_shapes():
    # Runs one real ~9s episode, takes agent_0's contiguous-from-step-0 stream,
    # replays it through FeatureBuilder, asserts BCSample shapes match the
    # post-refactor five-tensor contract.
    from collect_family_winner import build_winner_samples, run_episode_recorded
    from measure_tournament import assign_roster, build_env, SLOTS
    env = build_env()
    try:
        _, streams = run_episode_recorded(env, assign_roster(0),
                                          episode_seed=0)
    finally:
        env.close()
    stream = streams[SLOTS[0]]
    winner_steps = [{"obs": obs, "action": action} for obs, action in stream]
    samples = build_winner_samples(winner_steps)
    assert len(samples) == len(winner_steps)
    s = samples[0]
    assert s.grid.shape == (STACKED_GRID_CHANNELS, 16, 16)
    assert s.base_feats.shape == (5, 11)
    assert s.raw_agent.shape == (7, 5, 25)
    assert s.raw_base.shape == (7, 7, 25)
    assert s.scalar.shape == (STACKED_SCALARS,)
    assert s.mask.shape == (6,) and s.mask.dtype == bool
    assert isinstance(s.action, int)


def test_collect_smoke(tmp_path):
    # Runs 3 real scripted episodes (~30s); thresholds 0 so every family win
    # is kept. Verifies the dataset schema and a torch.save round-trip.
    import torch
    from collect_family_winner import FAMILY, collect

    samples, episodes, meta = collect(3, value_threshold=0.0,
                                      margin_threshold=0.0)

    assert meta["episodes_run"] == 3
    assert meta["episodes_kept"] == len(episodes)
    assert meta["total_samples"] == len(samples)
    for key in ("value_threshold", "margin_threshold", "duration_seconds",
                "winners", "feature_builder_contract"):
        assert key in meta

    for ep in episodes:
        assert set(ep.keys()) == {"episode_id", "winner_strategy",
                                  "winner_value", "winner_margin", "steps"}
        assert ep["winner_strategy"] in FAMILY        # gate keeps family only
        assert len(ep["steps"]) > 0
        step = ep["steps"][0]
        assert set(step.keys()) == {"obs", "action"}
        assert isinstance(step["obs"], dict)
        assert isinstance(step["action"], int)

    if samples:
        s = samples[0]
        assert s.grid.shape == (STACKED_GRID_CHANNELS, 16, 16)
        assert s.base_feats.shape == (5, 11)
        assert s.raw_agent.shape == (7, 5, 25)
        assert s.raw_base.shape == (7, 7, 25)
        assert s.scalar.shape == (STACKED_SCALARS,)

    path = tmp_path / "family_winner.pt"
    torch.save({"samples": samples, "episodes": episodes, "meta": meta},
               str(path))
    reloaded = torch.load(str(path), weights_only=False)
    assert reloaded["meta"]["episodes_run"] == 3
    assert len(reloaded["samples"]) == len(samples)
    assert len(reloaded["episodes"]) == len(episodes)
