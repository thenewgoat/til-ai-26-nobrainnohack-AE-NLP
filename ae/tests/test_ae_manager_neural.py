"""NeuralAEManager serves a trained ONNX policy via the feature builder."""
import numpy as np

from ae_manager import NeuralAEManager
from export_onnx import export_actor
from policy import SymbolicTransformerActor


def test_neural_manager_returns_legal_action(tmp_path):
    onnx_path = tmp_path / "policy.onnx"
    export_actor(SymbolicTransformerActor(), str(onnx_path))
    mgr = NeuralAEManager(onnx_path=str(onnx_path))

    obs = {
        "agent_viewcone": np.zeros((7, 5, 25), np.float32).tolist(),
        "base_viewcone": np.zeros((7, 7, 25), np.float32).tolist(),
        "direction": 0, "location": [3, 3], "base_location": [1, 1],
        "health": [60.0], "frozen_ticks": 0, "base_health": [100.0],
        "team_resources": [0.0], "team_bombs": 2, "step": 0,
        "action_mask": [1, 0, 0, 0, 0, 0],
    }
    action = mgr.ae(obs)
    assert action == 0                       # only action 0 is legal


def test_neural_manager_resets_belief_on_step0(tmp_path):
    onnx_path = tmp_path / "policy.onnx"
    export_actor(SymbolicTransformerActor(), str(onnx_path))
    mgr = NeuralAEManager(onnx_path=str(onnx_path))
    base = {
        "agent_viewcone": np.zeros((7, 5, 25), np.float32).tolist(),
        "base_viewcone": np.zeros((7, 7, 25), np.float32).tolist(),
        "direction": 0, "location": [3, 3], "base_location": [1, 1],
        "health": [60.0], "frozen_ticks": 0, "base_health": [100.0],
        "team_resources": [0.0], "team_bombs": 2,
        "action_mask": [1, 1, 1, 1, 1, 1],
    }
    mgr.ae({**base, "step": 0})
    mgr.ae({**base, "step": 1})
    mgr.ae({**base, "step": 0})              # new round -> belief re-bootstraps
    assert mgr.fb._started is True
