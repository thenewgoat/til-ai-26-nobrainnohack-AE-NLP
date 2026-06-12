"""Parallel hybrid rollout: buffer concat + multiprocess collection.

The parallel path splits episodes across pool workers (each rebuilds the actor on
CPU from a state_dict), so it must (a) concatenate the per-worker HybridRollout
buffers faithfully, (b) produce a valid buffer, and (c) leave the caller's actor
untouched (workers get a CPU copy; the parent actor never moves).
"""
import multiprocessing

import numpy as np

from hybrid_rollout import (collect_hybrid_rollout, collect_hybrid_rollout_parallel,
                            _concat_hybrid_buffers)
from policy import SymbolicTransformerActor
from scripted.handover import HandoverTrigger


def _actor():
    return SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)


def _trig():
    return HandoverTrigger(step_fallback=5)


def test_concat_hybrid_buffers_sums_size_and_preserves_fields():
    a = _actor()
    b1 = collect_hybrid_rollout(a, ["agent_0"], 1, 0, trigger=_trig())
    b2 = collect_hybrid_rollout(a, ["agent_0"], 1, 1, trigger=_trig())
    c = _concat_hybrid_buffers([b1, b2])
    assert c.size == b1.size + b2.size
    assert c.grid.shape == (c.size, *b1.grid.shape[1:])
    np.testing.assert_array_equal(
        c.proposed_actions, np.concatenate([b1.proposed_actions, b2.proposed_actions]))
    np.testing.assert_array_equal(
        c.dones, np.concatenate([b1.dones, b2.dones]))
    np.testing.assert_array_equal(
        c.masks, np.concatenate([b1.masks, b2.masks]))
    assert c.forward_bias == b1.forward_bias


def test_parallel_rollout_valid_buffer_and_actor_untouched():
    actor = _actor()
    dev_before = next(actor.parameters()).device
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2) as pool:
        buf = collect_hybrid_rollout_parallel(
            actor, ["agent_0"], num_episodes=2, seed0=0, pool=pool, num_workers=2,
            trigger=_trig())
    assert buf.size > 0
    for i in range(buf.size):
        assert buf.masks[i][buf.executed_actions[i]]   # executed action always legal
    assert buf.dones.sum() >= 2.0                       # two episodes -> two dones
    # The caller's actor must be left exactly where it was (workers use a copy).
    assert next(actor.parameters()).device == dev_before


def test_parallel_caps_workers_to_episode_count():
    # 4 workers but 1 episode: must not error, must still collect the episode.
    actor = _actor()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        buf = collect_hybrid_rollout_parallel(
            actor, ["agent_0"], num_episodes=1, seed0=0, pool=pool, num_workers=4,
            trigger=_trig())
    assert buf.size > 0
    assert buf.dones.sum() >= 1.0
