"""DAgger rollout produces (feature, action) pairs labeled by the teacher."""
import numpy as np

from bc import collect_dagger_dataset, BCSample
from features import STACKED_GRID_CHANNELS, STACKED_SCALARS
from policy import SymbolicTransformerActor, NUM_ACTIONS


def test_teacher_only_dataset_shapes():
    """Beta=1.0: the teacher drives every step; dataset has correct shapes."""
    ds = collect_dagger_dataset(
        teacher_strategy="balanced",
        rollout_policy=None,       # beta=1.0 -> teacher controls
        beta=1.0,
        num_episodes=2,
        seeds=[0, 1],
    )
    assert len(ds) > 0
    s = ds[0]
    assert isinstance(s, BCSample)
    assert s.grid.shape == (STACKED_GRID_CHANNELS, 16, 16)
    assert s.scalar.shape == (STACKED_SCALARS,)
    assert 0 <= s.action < NUM_ACTIONS
    assert s.mask.shape == (NUM_ACTIONS,)


def test_actions_are_legal_under_mask():
    ds = collect_dagger_dataset("balanced", None, 1.0, 1, [3])
    for s in ds:
        assert s.mask[s.action]      # the teacher label is always a legal move


def test_mixed_rollout_runs():
    """Beta<1: a fresh actor drives some steps; labels still come from teacher."""
    actor = SymbolicTransformerActor()
    ds = collect_dagger_dataset("balanced", actor, beta=0.5, num_episodes=1,
                                seeds=[7])
    assert len(ds) > 0


def test_bcsample_carries_five_tensors():
    from bc import collect_dagger_dataset
    from features import STACKED_GRID_CHANNELS, STACKED_SCALARS
    ds = collect_dagger_dataset("balanced", None, 1.0, 1, [0])
    s = ds[0]
    assert s.grid.shape == (STACKED_GRID_CHANNELS, 16, 16)
    assert s.base_feats.shape == (5, 11)
    assert s.raw_agent.shape == (7, 5, 25)
    assert s.raw_base.shape == (7, 7, 25)
    assert s.scalar.shape == (STACKED_SCALARS,)
    assert s.mask.shape == (6,)


def test_dataset_contains_bomb_labels():
    """Regression (spec C Task 21): the teacher must demonstrate PLACE_BOMB.

    A dataset with zero bomb labels trains a clone structurally unable to bomb
    — it can never destroy an enemy base, the bulk of a strong score. This
    fails if collect_dagger_dataset drives the non-learner slots with the
    teacher (teacher-vs-teacher contention makes a competent strategy oscillate
    in place and never bomb); it passes with the varied opponent pool.
    """
    ds = collect_dagger_dataset(
        teacher_strategy="balanced",
        rollout_policy=None,
        beta=1.0,
        num_episodes=6,
        seeds=list(range(6)),
    )
    actions = {s.action for s in ds}
    assert 5 in actions, (
        f"no PLACE_BOMB (action 5) labels in dataset; actions present={sorted(actions)}"
    )


def test_default_opponent_pool_excludes_teacher():
    from bc import _default_opponent_pool
    from scripted.strategies import STRATEGIES
    pool = _default_opponent_pool(teacher_name="balanced_extreme_opening")
    # build an agent from each non-RandomAgent factory and read its name
    names = []
    for factory in pool:
        ag = factory()
        names.append(ag.name)
    assert "scripted:balanced_extreme_opening" not in names
    # every other registered strategy IS represented
    for name in STRATEGIES:
        if name == "balanced_extreme_opening":
            continue
        assert f"scripted:{name}" in names


def test_default_opponent_pool_keeps_random_padding():
    from bc import _default_opponent_pool, RandomAgent
    pool = _default_opponent_pool(teacher_name="balanced_extreme_opening")
    random_count = sum(1 for f in pool if f is RandomAgent)
    assert random_count == 4


def test_collect_dagger_dataset_parallel_matches_sequential_count():
    """Parallel collection should produce the same number of BCSamples as
    sequential — episode count is the only invariant we can compare cheaply
    (trajectories differ across runs due to per-worker RNG seeding).
    """
    from bc import collect_dagger_dataset
    n_ep = 4
    seeds = list(range(n_ep))
    seq = collect_dagger_dataset("balanced_extreme_opening", None, 1.0,
                                  n_ep, seeds, num_workers=1, progress=False)
    par = collect_dagger_dataset("balanced_extreme_opening", None, 1.0,
                                  n_ep, seeds, num_workers=2, progress=False)
    # Same episode count and per-episode length (200 steps), so total samples
    # are equal regardless of which slot the learner occupies each episode.
    # Each episode contributes exactly one BCSample per env step the learner
    # was active for — that's deterministic given the seed list, so totals
    # must match exactly.
    assert len(par) == len(seq) > 0


def test_opponent_pool_meta_round_trip():
    from bc import _opponent_pool_meta, _opponent_pool_from_meta, RandomAgent
    from scripted.strategies import STRATEGIES
    meta = _opponent_pool_meta("balanced_extreme_opening")
    pool = _opponent_pool_from_meta(meta)
    # 4 RandomAgent classes + (len(STRATEGIES) - 1) scripted lambdas
    randoms = [f for f in pool if f is RandomAgent]
    scripteds = [f for f in pool if f is not RandomAgent]
    assert len(randoms) == 4
    assert len(scripteds) == len(STRATEGIES) - 1
    # Each lambda must produce a ScriptedAgent with a distinct strategy name
    names = sorted(f().name for f in scripteds)
    expected = sorted(f"scripted:{s}" for s in STRATEGIES
                       if s != "balanced_extreme_opening")
    assert names == expected


def test_collect_dagger_dataset_custom_pool_rejects_parallel():
    """Custom opponent_pool can't be pickled — must error early when num_workers>1."""
    import pytest
    from bc import collect_dagger_dataset, RandomAgent
    with pytest.raises(ValueError, match="custom opponent_pool"):
        collect_dagger_dataset("balanced", None, 1.0, 2, [0, 1],
                                opponent_pool=[RandomAgent], num_workers=2)


def test_bc_gate_works_with_cuda_actor(tmp_path):
    """bc_gate must not crash on a CUDA-resident actor (regression for the
    R3 device-mismatch crash)."""
    import torch
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    from bc import bc_gate
    from policy import SymbolicTransformerActor
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    actor = actor.to("cuda")
    # tiny seed list keeps the test bounded; we only care that bc_gate runs
    # to completion without raising the device-mismatch error.
    passed, detail = bc_gate(actor, "balanced_extreme_opening",
                              seeds=list(range(2)), tolerance=1.0)
    # don't assert on `passed` — a tiny random-init actor will fail the gate;
    # the test is about not crashing.
    assert "clone" in detail
    assert "teacher" in detail
