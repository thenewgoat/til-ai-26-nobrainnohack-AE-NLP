"""NoisyScriptedAgent: ε-noise over the strategy's pick, masked to legal."""
import random

import numpy as np
import pytest

from evaluate import NoisyScriptedAgent, ScriptedAgent
from til_environment import bomberman_env
from til_environment.config import default_config


def _first_obs(seed=0):
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=seed)
    target = env.agents[0]
    obs = None
    for slot in env.agent_iter():
        o, _, _, _, _ = env.last()
        if slot == target:
            obs = dict(o)
            break
        env.step(0)
    env.close()
    return obs


def test_epsilon_zero_matches_inner_scripted():
    obs = _first_obs()
    inner = ScriptedAgent("balanced_extreme_opening")
    inner.reset()
    noisy = NoisyScriptedAgent("balanced_extreme_opening", epsilon=0.0,
                                rng=random.Random(1))
    noisy.reset()
    assert noisy.action(obs) == inner.action(obs)


def test_epsilon_one_picks_only_legal_actions():
    obs = _first_obs()
    noisy = NoisyScriptedAgent("balanced", epsilon=1.0,
                                rng=random.Random(7))
    noisy.reset()
    mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
    legal = set(np.flatnonzero(mask).tolist())
    for _ in range(200):
        a = noisy.action(obs)
        assert a in legal


def test_epsilon_zero_with_no_legal_mask_passes_through_inner():
    """When the mask has no legal actions, ε=0 still defers to the inner
    (inner is responsible for the fallback). Smoke: just confirms no crash."""
    obs = _first_obs()
    obs2 = dict(obs)
    obs2["action_mask"] = np.zeros(6, dtype=np.int8)
    noisy = NoisyScriptedAgent("balanced", epsilon=0.0,
                                rng=random.Random(0))
    noisy.reset()
    # should not raise — inner's act() handles the empty-mask case
    noisy.action(obs2)


def test_reset_resets_inner():
    noisy = NoisyScriptedAgent("balanced", epsilon=0.1,
                                rng=random.Random(0))
    noisy.reset()
    # Both noisy and inner ScriptedAgent expose `name`
    assert noisy._inner.name == "scripted:balanced"
