import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from export_onnx import export_actor
from hybrid_controller import OnnxActorRuntime
from policy import SymbolicTransformerActor
from scripted.geometry import FORWARD, STAY


def _features():
    return (np.zeros((85, 16, 16), np.float32), np.zeros((5, 11), np.float32),
            np.zeros((7, 5, 25), np.float32), np.zeros((7, 7, 25), np.float32),
            np.zeros(50, np.float32))


def _export(tmp_path, actor):
    p = tmp_path / "actor.onnx"
    export_actor(actor, str(p))
    return str(p)


def test_onnx_query_returns_legal_argmax(tmp_path):
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    rt = OnnxActorRuntime.from_path(_export(tmp_path, actor))
    mask = np.zeros(6, bool)
    mask[FORWARD] = True
    mask[STAY] = True
    action, logp, entropy = rt.query(_features(), mask)
    assert action in (FORWARD, STAY)
    assert isinstance(logp, float) and isinstance(entropy, float)


def test_onnx_query_respects_mask_under_forward_bias(tmp_path):
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    rt = OnnxActorRuntime.from_path(_export(tmp_path, actor))
    mask = np.ones(6, bool)
    mask[FORWARD] = False
    action, _, _ = rt.query(_features(), mask, forward_bias=50.0)
    assert action != FORWARD


def test_onnx_matches_torch_masked_argmax(tmp_path):
    import torch
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2).eval()
    rt = OnnxActorRuntime.from_path(_export(tmp_path, actor))
    feats = _features()
    mask = np.ones(6, bool)
    mask[0] = False
    with torch.no_grad():
        g, bf, ra, rb, sc = (torch.as_tensor(x).unsqueeze(0) for x in feats)
        logits = actor.forward(g, bf, ra, rb, sc)[0].numpy()
    torch_argmax = int(np.argmax(np.where(mask, logits, -1e8)))
    onnx_action, _, _ = rt.query(feats, mask)
    assert onnx_action == torch_argmax


def test_module_imports_without_torch():
    src = Path(__file__).resolve().parent.parent / "src"
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import hybrid_controller as hc\n"
        "assert hasattr(hc, 'OnnxActorRuntime') and hasattr(hc, 'HybridController')\n"
        "assert hasattr(hc, 'post_handover_decision') and hasattr(hc, 'RLDecision')\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src}{os.pathsep}{src / 'training'}"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
