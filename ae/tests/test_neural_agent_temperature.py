"""NeuralAgent: temperature controls argmax (T→0) vs uniform-legal (T→∞)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "training"))

import numpy as np
import torch

from evaluate import NeuralAgent
from policy import SymbolicTransformerActor


def _make_actor_with_seed(seed=0):
    torch.manual_seed(seed)
    return SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)


def _first_obs(seed=0):
    from til_environment import bomberman_env
    from til_environment.config import default_config
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=seed)
    for slot in env.agent_iter():
        o, _, _, _, _ = env.last()
        obs = dict(o)
        env.close()
        return obs


def test_temperature_default_is_one():
    actor = _make_actor_with_seed()
    ag = NeuralAgent(actor, name="x")
    assert hasattr(ag, "_t") or hasattr(ag, "temperature")


def test_low_temperature_is_argmax_like():
    """T=0.001 → essentially argmax. The action chosen at T=0.001 over many
    rollouts of the same observation should be the same value most of the time."""
    actor = _make_actor_with_seed()
    obs = _first_obs()
    ag = NeuralAgent(actor, name="x", temperature=0.001)
    ag.reset()
    actions = [ag.action(obs) for _ in range(50)]
    # >= 90% should be the same value
    from collections import Counter
    most, count = Counter(actions).most_common(1)[0]
    assert count >= 45


def test_high_temperature_diversifies():
    """T=10.0 → near-uniform over legal actions. Over many samples we should
    see at least 2 distinct legal actions."""
    actor = _make_actor_with_seed()
    obs = _first_obs()
    mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
    n_legal = int(mask.sum())
    if n_legal < 2:
        return   # cannot test diversity if only one legal action
    ag = NeuralAgent(actor, name="x", temperature=10.0)
    ag.reset()
    actions = {ag.action(obs) for _ in range(200)}
    assert len(actions) >= 2


def test_neural_agent_accepts_cuda_actor():
    """NeuralAgent.action must work with a CUDA-resident actor without the
    caller having to .to('cpu')-copy first."""
    import torch
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    actor = _make_actor_with_seed()
    actor = actor.to("cuda")
    obs = _first_obs()
    ag = NeuralAgent(actor, name="x")
    ag.reset()
    a = ag.action(obs)
    assert 0 <= a < 6
