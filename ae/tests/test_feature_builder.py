"""FeatureBuilder emits the five-tensor contract with correct shapes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "training"))

import numpy as np
import pytest

import features as F
from features import FeatureBuilder
from til_environment import bomberman_env
from til_environment.config import default_config


def _first_obs():
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=0)
    slot = env.agents[0]
    obs, _, _, _, _ = env.last()
    env.close()
    return obs


def test_build_returns_five_tensors_with_frozen_shapes():
    obs = _first_obs()
    fb = FeatureBuilder()
    grid, base_feats, raw_agent, raw_base, scalar = fb.build(obs)
    assert grid.shape == (F.STACKED_GRID_CHANNELS, F.GRID_SIZE, F.GRID_SIZE)
    assert base_feats.shape == (F.NUM_BASES, F.BASE_FIELDS)
    assert raw_agent.shape == F.RAW_AGENT_SHAPE
    assert raw_base.shape == F.RAW_BASE_SHAPE
    assert scalar.shape == (F.STACKED_SCALARS,)
    for t in (grid, base_feats, raw_agent, raw_base, scalar):
        assert t.dtype == np.float32


def test_raw_agent_passes_through_unaltered():
    obs = _first_obs()
    fb = FeatureBuilder()
    _, _, raw_agent, _, _ = fb.build(obs)
    expected = np.asarray(obs["agent_viewcone"], dtype=np.float32)
    np.testing.assert_array_equal(raw_agent, expected)


def test_raw_base_zeroed_when_viewcone_degenerate():
    obs = _first_obs()
    obs = dict(obs)
    obs["base_viewcone"] = np.zeros((1, 1, 25), dtype=np.float32)
    fb = FeatureBuilder()
    _, _, _, raw_base, _ = fb.build(obs)
    assert raw_base.shape == F.RAW_BASE_SHAPE
    assert not raw_base.any()


def test_raw_agent_zeroed_when_viewcone_degenerate():
    obs = _first_obs()
    obs = dict(obs)
    obs["agent_viewcone"] = np.zeros((1, 1, 25), dtype=np.float32)
    fb = FeatureBuilder()
    _, _, raw_agent, _, _ = fb.build(obs)
    assert raw_agent.shape == F.RAW_AGENT_SHAPE
    assert not raw_agent.any()
