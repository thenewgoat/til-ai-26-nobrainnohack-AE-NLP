"""K=5 frame-stack: shape, newest-first ordering, warmup zero-pad, reset."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np

import features as F
from features import FeatureBuilder
from til_environment import bomberman_env
from til_environment.config import default_config


def _step_env_n(n_steps=4, seed=0):
    """Reset the novice env and step it `n_steps` times via a random-walk
    policy; return a list of observations for agent_0."""
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=seed)
    target = "agent_0"
    observed = []
    seen_steps = 0
    for slot in env.agent_iter():
        obs, _, term, trunc, _ = env.last()
        if slot == target:
            observed.append(dict(obs))
            seen_steps += 1
            if seen_steps > n_steps:
                break
        if term or trunc:
            env.step(None)
            continue
        mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
        legal = np.flatnonzero(mask)
        env.step(int(legal[0]) if len(legal) else 4)
    env.close()
    return observed


def test_stacked_grid_shape():
    obs = _step_env_n(1)[0]
    fb = FeatureBuilder()
    grid, _, _, _, _ = fb.build(obs)
    assert grid.shape == (F.STACKED_GRID_CHANNELS, F.GRID_SIZE, F.GRID_SIZE)
    assert grid.dtype == np.float32


def test_stacked_scalar_shape():
    obs = _step_env_n(1)[0]
    fb = FeatureBuilder()
    _, _, _, _, scalar = fb.build(obs)
    assert scalar.shape == (F.STACKED_SCALARS,)
    assert scalar.dtype == np.float32


def test_first_frame_zero_pads_past_slots():
    """At step 0 only the newest (channels [0:GC), scalars [0:FS)) is non-zero;
    past slots are exact zeros."""
    obs = _step_env_n(1)[0]
    fb = FeatureBuilder()
    grid, _, _, _, scalar = fb.build(obs)
    past_grid = grid[F.GRID_CHANNELS:]                # channels 17..84
    past_scalar = scalar[F.FEATURE_SCALARS:]          # entries 10..49
    assert np.all(past_grid == 0.0)
    assert np.all(past_scalar == 0.0)


def test_history_fills_after_K_steps():
    """After K builds the deque is full; channels [(K-1)*GC : K*GC) hold the
    oldest frame and should equal what build() produced at step 0."""
    observed = _step_env_n(F.STACK - 1)
    fb = FeatureBuilder()
    first_grid_alone = None
    last = None
    for i, obs in enumerate(observed):
        grid, _, _, _, scalar = fb.build(obs)
        if i == 0:
            first_grid_alone = grid[:F.GRID_CHANNELS].copy()
        last = grid
    # After STACK total builds, the OLDEST frame in the stack is the one we
    # built first. Layout (newest-first):
    #   slot 0 (channels [0:GC))               = newest
    #   slot K-1 (channels [(K-1)*GC:K*GC))    = oldest
    oldest_slice = last[(F.STACK - 1) * F.GRID_CHANNELS : F.STACK * F.GRID_CHANNELS]
    np.testing.assert_array_equal(oldest_slice, first_grid_alone)


def test_episode_reset_clears_history():
    """A second episode (a second step==0 observation) re-zeros past slots."""
    first_ep = _step_env_n(F.STACK + 1, seed=0)
    fb = FeatureBuilder()
    for obs in first_ep:
        fb.build(obs)
    # synthesise a step==0 observation by re-resetting the env at a different seed
    second_first = _step_env_n(1, seed=99)[0]
    grid, _, _, _, scalar = fb.build(second_first)
    past_grid = grid[F.GRID_CHANNELS:]
    past_scalar = scalar[F.FEATURE_SCALARS:]
    assert np.all(past_grid == 0.0), (
        "history not cleared across episodes: past channels non-zero")
    assert np.all(past_scalar == 0.0)


def test_newest_slice_matches_single_frame_intent():
    """Channels [0:GRID_CHANNELS) of the stacked output should carry the
    current frame's signal."""
    obs = _step_env_n(1)[0]
    fb = FeatureBuilder()
    grid, _, _, _, scalar = fb.build(obs)
    newest_grid = grid[:F.GRID_CHANNELS]
    # the newest slice must carry some non-trivial signal (walls / self /
    # collectibles are guaranteed non-zero in a fresh novice episode).
    assert newest_grid.sum() > 0.0
