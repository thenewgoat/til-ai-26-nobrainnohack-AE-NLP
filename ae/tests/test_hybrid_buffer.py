import numpy as np

from hybrid_rollout import build_hybrid_buffer
from critic import STATE_PLANES, STATE_SCALARS
from features import (NUM_BASES, BASE_FIELDS, RAW_AGENT_SHAPE, RAW_BASE_SHAPE,
                      STACKED_GRID_CHANNELS, STACKED_SCALARS)
from policy import NUM_ACTIONS

GRID = 16


def _tr(proposed=1, executed=1, actor_queried=True, logp=-0.7, reward=0.5,
        env_reward=0.5, done=0.0):
    return dict(
        grid=np.zeros((STACKED_GRID_CHANNELS, GRID, GRID), np.float32),
        base_feats=np.zeros((NUM_BASES, BASE_FIELDS), np.float32),
        raw_agent=np.zeros(RAW_AGENT_SHAPE, np.float32),
        raw_base=np.zeros(RAW_BASE_SHAPE, np.float32),
        scalar=np.zeros(STACKED_SCALARS, np.float32),
        gstate=np.zeros((STATE_PLANES, GRID, GRID), np.float32),
        gscalar=np.zeros(STATE_SCALARS, np.float32),
        mask=np.ones(NUM_ACTIONS, bool),
        proposed=proposed, executed=executed, actor_queried=actor_queried,
        logp=logp, reward=reward, env_reward=env_reward, done=done)


def test_build_stacks_fields_and_sets_size():
    trs = [_tr(proposed=1, actor_queried=True), _tr(proposed=4, actor_queried=False)]
    buf = build_hybrid_buffer(trs, forward_bias=0.5)
    assert buf.size == 2
    assert buf.grid.shape == (2, STACKED_GRID_CHANNELS, GRID, GRID)
    assert buf.gstate.shape == (2, STATE_PLANES, GRID, GRID)
    assert buf.masks.shape == (2, NUM_ACTIONS) and buf.masks.dtype == bool
    assert buf.proposed_actions.tolist() == [1, 4]
    assert buf.actor_queried.tolist() == [True, False]
    assert buf.forward_bias == 0.5
    assert buf.proposed_actions.dtype == np.int64
    assert buf.actor_queried.dtype == bool
    assert buf.logprobs.dtype == np.float32


def test_build_preserves_reward_done_order():
    trs = [_tr(reward=1.0, done=0.0), _tr(reward=2.0, done=1.0)]
    buf = build_hybrid_buffer(trs, forward_bias=0.0)
    assert buf.rewards.tolist() == [1.0, 2.0]
    assert buf.dones.tolist() == [0.0, 1.0]


def test_build_empty_is_zero_size():
    buf = build_hybrid_buffer([], forward_bias=0.0)
    assert buf.size == 0
    assert buf.grid.shape[0] == 0
