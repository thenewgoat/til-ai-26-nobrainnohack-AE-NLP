import numpy as np
import pytest

from ae_manager import HybridAEManager
from export_onnx import export_actor
from policy import SymbolicTransformerActor


def _obs(mask, step=0):
    return {
        "agent_viewcone": np.zeros((7, 5, 25), np.float32).tolist(),
        "base_viewcone": np.zeros((7, 7, 25), np.float32).tolist(),
        "direction": 0, "location": [3, 3], "base_location": [1, 1],
        "health": [60.0], "frozen_ticks": 0, "base_health": [100.0],
        "team_resources": [0.0], "team_bombs": 2, "step": step,
        "action_mask": mask,
    }


def test_fatal_when_actor_path_unset(monkeypatch):
    monkeypatch.delenv("AE_RL_ACTOR_PATH", raising=False)
    with pytest.raises((RuntimeError, ValueError)):
        HybridAEManager()


def test_fatal_when_actor_path_does_not_exist(monkeypatch):
    monkeypatch.setenv("AE_RL_ACTOR_PATH", "/nonexistent/rl_actor.onnx")
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        HybridAEManager()


def test_pre_handover_serves_opener_action(tmp_path, monkeypatch):
    onnx = tmp_path / "rl_actor.onnx"
    export_actor(SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2), str(onnx))
    monkeypatch.setenv("AE_RL_ACTOR_PATH", str(onnx))
    mgr = HybridAEManager()
    action = mgr.ae(_obs([1, 0, 0, 0, 0, 0], step=0))   # only FORWARD legal
    assert action == 0


def test_post_handover_serves_legal_actor_action(tmp_path, monkeypatch):
    onnx = tmp_path / "rl_actor.onnx"
    export_actor(SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2), str(onnx))
    monkeypatch.setenv("AE_RL_ACTOR_PATH", str(onnx))
    mgr = HybridAEManager()
    mgr.ae(_obs([1, 1, 1, 1, 1, 1], step=0))            # warm the builder
    action = mgr.ae(_obs([1, 1, 1, 1, 1, 1], step=60))  # step fallback -> handover
    assert 0 <= action < 6
    assert mgr.controller.handover_fired is True
