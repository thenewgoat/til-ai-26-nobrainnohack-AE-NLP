"""Per-episode learner-slot rotation: subset size, buffer slice, advantages."""
import numpy as np
import pytest

from train_selfplay import (collect_rollout, OpponentRegistry, RolloutBuffer,
                              EPISODE_LEN, compute_advantages)
from league import League
from policy import SymbolicTransformerActor


SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]


def test_rotate_emits_correct_buffer_size():
    actor = SymbolicTransformerActor()
    reg = OpponentRegistry(League())
    # 4 episodes with K ∈ [1, 3] random per episode
    buf = collect_rollout(
        actor, reg, learner_slots=("rotate", 1, 3),
        num_episodes=4, seed0=11,
    )
    # k_ep ∈ {1,2,3}; total writes = sum(k_ep) * EPISODE_LEN.
    # Allocated worst-case is 3 * EPISODE_LEN * 4 = 2400.
    assert buf.size <= 3 * EPISODE_LEN * 4
    assert buf.size >= 1 * EPISODE_LEN * 4
    assert buf.size % EPISODE_LEN == 0


def test_rotate_subset_size_is_in_range():
    """Run 8 episodes with K∈[1,2] and confirm every episode contributed a
    multiple of EPISODE_LEN to the buffer (no partial runs)."""
    actor = SymbolicTransformerActor()
    reg = OpponentRegistry(League())
    buf = collect_rollout(
        actor, reg, learner_slots=("rotate", 1, 2),
        num_episodes=8, seed0=0,
    )
    assert buf.size % EPISODE_LEN == 0
    n_runs = buf.size // EPISODE_LEN
    # 8 episodes * K∈{1,2} → 8..16 runs total
    assert 8 <= n_runs <= 16


def test_rotate_dones_at_run_boundaries():
    """compute_advantages relies on dones[t]==1 at every slot-run boundary."""
    actor = SymbolicTransformerActor()
    reg = OpponentRegistry(League())
    buf = collect_rollout(
        actor, reg, learner_slots=("rotate", 1, 3),
        num_episodes=2, seed0=5,
    )
    n_runs = buf.size // EPISODE_LEN
    for j in range(n_runs):
        assert buf.dones[(j + 1) * EPISODE_LEN - 1] == 1.0
    # No other positions have done==1
    terminals = {(j + 1) * EPISODE_LEN - 1 for j in range(n_runs)}
    for i in range(buf.size):
        if i not in terminals:
            assert buf.dones[i] == 0.0


def test_compute_advantages_handles_variable_run_count():
    """GAE over a mixed-K buffer must zero the bootstrap at every done==1."""
    rewards = np.array([1, 1, 1, 1, 1,    # run A
                        2, 2, 2, 2, 2,    # run B
                        0, 0, 0, 0, 0],   # run C
                       dtype=np.float32)
    values = np.zeros_like(rewards)
    dones = np.zeros_like(rewards)
    dones[4] = 1.0
    dones[9] = 1.0
    dones[14] = 1.0
    adv, ret = compute_advantages(rewards, values, dones, gamma=1.0,
                                   gae_lambda=1.0)
    # Per run, GAE with gamma=1 and lambda=1 reduces to sum-of-future-rewards.
    expected = np.array([5,4,3,2,1, 10,8,6,4,2, 0,0,0,0,0], dtype=np.float32)
    np.testing.assert_allclose(adv, expected, atol=1e-5)
