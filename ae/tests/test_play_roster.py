"""Tests for play.py's roster helpers (SLOT_AGENTS validation, spec parsing).

play.py lives in til-26-ae/ rather than ae/src/, so we prepend that directory
to sys.path before importing it. The til-26-ae/.venv is also the only venv
with both pytest and til_environment, so these tests must be run with it
(see plan header for the exact pytest command).
"""
import sys
from pathlib import Path

# til-26-ae/ holds play.py and is editable-installed til_environment.
_TIL26_AE = Path(__file__).resolve().parents[2] / "til-26-ae"
sys.path.insert(0, str(_TIL26_AE))

import pytest

import play  # noqa: E402  (sys.path inserted above)


def test_validate_accepts_canonical_roster():
    play._validate_slot_agents(
        ["human", "random", "random", "random", "random", "random"])


def test_validate_rejects_wrong_length():
    with pytest.raises(ValueError, match="length 5"):
        play._validate_slot_agents(["human", "random", "random", "random", "random"])
    with pytest.raises(ValueError, match="length 7"):
        play._validate_slot_agents(["human"] + ["random"] * 6)


def test_validate_rejects_no_human():
    with pytest.raises(ValueError, match="exactly one"):
        play._validate_slot_agents(["random"] * 6)


def test_validate_rejects_multiple_human():
    with pytest.raises(ValueError, match="exactly one"):
        play._validate_slot_agents(
            ["human", "human", "random", "random", "random", "random"])


def test_spec_to_agent_human_sentinel():
    agent, label = play._spec_to_agent("human")
    assert agent is None
    assert label == "human"


def test_spec_to_agent_random():
    agent, label = play._spec_to_agent("random")
    assert label == "random"
    assert agent.__class__.__name__ == "RandomAgent"


def test_spec_to_agent_greedy():
    agent, label = play._spec_to_agent("greedy")
    assert label == "greedy"
    assert agent.__class__.__name__ == "GreedyAgent"


def test_spec_to_agent_scripted_adaptive():
    agent, label = play._spec_to_agent("scripted:adaptive")
    assert label == "scripted:adaptive"
    assert agent.__class__.__name__ == "ScriptedAgent"


def test_spec_to_agent_unknown_scripted_raises():
    with pytest.raises(ValueError, match="unknown scripted strategy"):
        play._spec_to_agent("scripted:does_not_exist_xyz")


def test_spec_to_agent_bad_spec_raises():
    with pytest.raises(ValueError, match="bad agent spec"):
        play._spec_to_agent("totally_unknown")


def test_spec_to_agent_ckpt_missing_path():
    with pytest.raises(ValueError, match="checkpoint not found"):
        play._spec_to_agent("ckpt:/nonexistent/path/that/should/not/exist.pt")


def test_slot_agents_default_is_valid():
    # The shipped default must be a sane roster the user can run immediately.
    play._validate_slot_agents(play.SLOT_AGENTS)
    assert len(play.SLOT_AGENTS) == 6
    assert play.SLOT_AGENTS.count("human") == 1


def test_build_slot_agents_returns_full_roster():
    specs = ["human", "random", "random", "random", "random", "random"]
    possible = [f"agent_{i}" for i in range(6)]
    slot_agents, player_slot = play._build_slot_agents(specs, possible)
    assert set(slot_agents.keys()) == set(possible)
    assert player_slot == "agent_0"
    # human slot has the sentinel agent
    assert slot_agents["agent_0"] == (None, "human")
    # bot slots have real agents
    for slot in possible[1:]:
        agent, label = slot_agents[slot]
        assert agent is not None
        assert label == "random"


def test_build_slot_agents_human_in_middle_slot():
    specs = ["random", "random", "human", "random", "random", "random"]
    possible = [f"agent_{i}" for i in range(6)]
    slot_agents, player_slot = play._build_slot_agents(specs, possible)
    assert player_slot == "agent_2"
    assert slot_agents["agent_2"] == (None, "human")
    assert slot_agents["agent_0"][0] is not None


def test_build_slot_agents_propagates_validation_errors():
    with pytest.raises(ValueError, match="exactly one"):
        play._build_slot_agents(["random"] * 6, [f"agent_{i}" for i in range(6)])
