"""Rollout collection gathers transitions from all learner-controlled slots."""
import numpy as np

from train_selfplay import OpponentRegistry, collect_rollout, RolloutBuffer
from league import League
from policy import SymbolicTransformerActor
from features import STACKED_GRID_CHANNELS, STACKED_SCALARS


def test_rollout_collects_from_multiple_slots():
    actor = SymbolicTransformerActor()
    league = League()
    reg = OpponentRegistry(league)
    # learner controls slots 0,1,2 ; opponents fill 3,4,5
    buf = collect_rollout(actor, reg, learner_slots=("agent_0", "agent_1",
                                                     "agent_2"),
                          num_episodes=1, seed0=0)
    assert isinstance(buf, RolloutBuffer)
    # 3 slots * 200 steps = 600 learner transitions in one episode
    assert buf.size == 3 * 200
    from critic import STATE_PLANES, STATE_SCALARS
    from features import NUM_BASES, BASE_FIELDS, RAW_AGENT_SHAPE, RAW_BASE_SHAPE
    assert buf.grid.shape == (buf.size, STACKED_GRID_CHANNELS, 16, 16)
    assert buf.base_feats.shape == (buf.size, NUM_BASES, BASE_FIELDS)
    assert buf.raw_agent.shape == (buf.size, *RAW_AGENT_SHAPE)
    assert buf.raw_base.shape == (buf.size, *RAW_BASE_SHAPE)
    assert buf.scalar.shape == (buf.size, STACKED_SCALARS)
    # encoded global state for the centralized critic, per learner transition
    assert buf.gstate.shape == (buf.size, STATE_PLANES, 16, 16)
    assert buf.gscalar.shape == (buf.size, STATE_SCALARS)
    assert buf.actions.shape == (buf.size,)
    assert buf.dones.sum() == 3        # one episode-end per learner slot


def test_single_slot_rollout_size():
    actor = SymbolicTransformerActor()
    reg = OpponentRegistry(League())
    buf = collect_rollout(actor, reg, learner_slots=("agent_0",),
                          num_episodes=2, seed0=10)
    assert buf.size == 2 * 200


def test_rollout_run_contiguous_layout():
    """Buffer must be run-contiguous: each (episode, slot) run of EPISODE_LEN
    transitions occupies a contiguous block, and dones==1.0 at exactly the last
    index of each run (EPISODE_LEN-1, 2*EPISODE_LEN-1, 3*EPISODE_LEN-1) and
    0.0 everywhere else."""
    from train_selfplay import EPISODE_LEN
    actor = SymbolicTransformerActor()
    reg = OpponentRegistry(League())
    # 3 learner slots, 1 episode -> 3 runs of EPISODE_LEN each
    buf = collect_rollout(actor, reg, learner_slots=("agent_0", "agent_1",
                                                     "agent_2"),
                          num_episodes=1, seed0=0)
    assert buf.size == 3 * EPISODE_LEN
    # Each run must end with dones==1.0 at its last index
    for j in range(3):
        last_idx = (j + 1) * EPISODE_LEN - 1
        assert buf.dones[last_idx] == 1.0, (
            f"run {j}: dones[{last_idx}] = {buf.dones[last_idx]}, expected 1.0")
    # All other positions must be 0.0
    terminal_indices = {(j + 1) * EPISODE_LEN - 1 for j in range(3)}
    for i in range(buf.size):
        if i not in terminal_indices:
            assert buf.dones[i] == 0.0, (
                f"dones[{i}] = {buf.dones[i]}, expected 0.0 (non-terminal)")


def test_collect_rollout_parallel_buffer_size():
    import multiprocessing
    from train_selfplay import collect_rollout_parallel
    actor = SymbolicTransformerActor()
    slots = ("agent_0", "agent_1", "agent_2")
    with multiprocessing.Pool(2) as pool:
        buf, outcomes = collect_rollout_parallel(
            actor, slots, num_episodes=2, seed0=0, opponent_members=None,
            reward_shaper=None, pool=pool, num_workers=2)
    assert isinstance(buf, RolloutBuffer)
    assert buf.size == 3 * 200 * 2          # 3 slots * 200 steps * 2 episodes
    assert outcomes == []                   # random opponents -> no league outcomes


def test_collect_rollout_parallel_one_worker_size_matches_sequential():
    import multiprocessing
    from train_selfplay import collect_rollout_parallel
    actor = SymbolicTransformerActor()
    slots = ("agent_0", "agent_1", "agent_2")
    seq = collect_rollout(actor, OpponentRegistry(League()), slots,
                          num_episodes=3, seed0=0)
    with multiprocessing.Pool(1) as pool:
        par, _ = collect_rollout_parallel(actor, slots, 3, 0, None, None,
                                          pool, num_workers=1)
    assert par.size == seq.size


def test_collect_rollout_parallel_multiworker_size():
    import multiprocessing
    from train_selfplay import collect_rollout_parallel
    actor = SymbolicTransformerActor()
    slots = ("agent_0", "agent_1", "agent_2")
    with multiprocessing.Pool(4) as pool:
        b1, _ = collect_rollout_parallel(actor, slots, 4, 0, None, None,
                                         pool, num_workers=1)
        b4, _ = collect_rollout_parallel(actor, slots, 4, 0, None, None,
                                         pool, num_workers=4)
    assert b1.size == b4.size == 3 * 200 * 4


def test_collect_rollout_parallel_outcomes_per_opponent_slot():
    import multiprocessing
    from train_selfplay import collect_rollout_parallel
    actor = SymbolicTransformerActor()
    slots = ("agent_0", "agent_1", "agent_2")
    members = list(League().anchors())          # scripted Members -> real bookkeeping
    with multiprocessing.Pool(2) as pool:
        _, outcomes = collect_rollout_parallel(
            actor, slots, num_episodes=2, seed0=0, opponent_members=members,
            reward_shaper=None, pool=pool, num_workers=2)
    # 3 opponent slots * 2 episodes -> 6 (member, won) outcomes
    assert len(outcomes) == 2 * 3
    for member, won in outcomes:
        assert won in (True, False)


def test_collect_rollout_uses_kind_mix_when_opponent_members_none():
    """When opponent_members is None and use_sample_mix=True, opponents are
    drawn via OpponentRegistry.sample_slot_opponent — verifying the rollout
    completes and writes one full episode worth of buffer."""
    from train_selfplay import collect_rollout
    actor = SymbolicTransformerActor()
    reg = OpponentRegistry(League())
    buf = collect_rollout(
        actor, reg, learner_slots=("agent_0",),
        num_episodes=1, seed0=42,
        opponent_members=None,
        use_sample_mix=True,
    )
    assert isinstance(buf, RolloutBuffer)
    assert buf.size == 200
