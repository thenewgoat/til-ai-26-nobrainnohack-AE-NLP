"""SymbolicTransformerActor exports to ONNX with logits matching torch."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "training"))

import numpy as np
import onnxruntime as ort
import torch

from export_onnx import export_actor
from policy import SymbolicTransformerActor, STACKED_GRID_CHANNELS, STACKED_SCALARS


def _inputs(n=2):
    return {
        "grid": np.random.randn(n, STACKED_GRID_CHANNELS, 16, 16).astype(np.float32),
        "base_feats": np.random.randn(n, 5, 11).astype(np.float32),
        "raw_agent": np.random.randn(n, 7, 5, 25).astype(np.float32),
        "raw_base": np.random.randn(n, 7, 7, 25).astype(np.float32),
        "scalar": np.random.randn(n, STACKED_SCALARS).astype(np.float32),
    }


def test_onnx_logits_match_torch(tmp_path):
    actor = SymbolicTransformerActor(d_model=32, n_layers=2, n_heads=4)
    actor.eval()
    onnx_path = tmp_path / "actor.onnx"
    export_actor(actor, str(onnx_path))

    inp = _inputs(2)
    with torch.no_grad():
        torch_logits = actor(
            torch.from_numpy(inp["grid"]),
            torch.from_numpy(inp["base_feats"]),
            torch.from_numpy(inp["raw_agent"]),
            torch.from_numpy(inp["raw_base"]),
            torch.from_numpy(inp["scalar"]),
        ).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    onnx_logits = sess.run(["logits"], inp)[0]
    np.testing.assert_allclose(torch_logits, onnx_logits, rtol=1e-3, atol=1e-4)


def test_onnx_export_dynamic_batch(tmp_path):
    actor = SymbolicTransformerActor(d_model=32, n_layers=2, n_heads=4)
    actor.eval()
    onnx_path = tmp_path / "actor.onnx"
    export_actor(actor, str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path))
    # exported with a dynamic batch axis: a batch size != the trace size works
    for n in (1, 5):
        out = sess.run(["logits"], _inputs(n))[0]
        assert out.shape == (n, 6)
