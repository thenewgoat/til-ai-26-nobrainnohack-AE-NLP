"""visualize._spec_to_agent parses per-slot policy specs."""
import os

import pytest
import torch

from visualize import _spec_to_agent
from evaluate import RandomAgent, ScriptedAgent, NeuralAgent
from policy import SymbolicTransformerActor


def test_random_spec():
    agent, label = _spec_to_agent("random")
    assert isinstance(agent, RandomAgent)
    assert label == "random"


def test_scripted_spec():
    agent, label = _spec_to_agent("scripted:balanced")
    assert isinstance(agent, ScriptedAgent)
    assert label == "scripted:balanced"


def test_ckpt_spec(tmp_path):
    ckpt = os.path.join(str(tmp_path), "tiny_actor.pt")
    SymbolicTransformerActor().save_checkpoint(ckpt)
    agent, label = _spec_to_agent(f"ckpt:{ckpt}")
    assert isinstance(agent, NeuralAgent)
    # label is ckpt:<basename>, not the full path
    assert label == "ckpt:tiny_actor.pt"


def test_bad_spec_raises():
    with pytest.raises(ValueError):
        _spec_to_agent("nonsense")
    with pytest.raises(ValueError):
        _spec_to_agent("scripted:no_such_strategy")
