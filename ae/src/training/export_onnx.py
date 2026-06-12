"""Export a trained SymbolicTransformerActor to ONNX for container serving.

The served container drops torch and runs onnxruntime (smaller image, faster
cold start - spec C section 8). The exported graph takes the five named inputs
of the observation contract (grid, base_feats, raw_agent, raw_base, scalar) and
returns `logits` [N,6].

CLI: `python export_onnx.py` exports ae/src/policy_transformer_bc.pt ->
ae/src/policy.onnx.
"""
import os
import sys

import torch

# ae/src holds policy.py; conftest.py adds it for tests but a standalone run
# (CLI or `uv run python -c ...`) needs this.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from policy import SymbolicTransformerActor, STACKED_GRID_CHANNELS, STACKED_SCALARS

_INPUT_NAMES = ["grid", "base_feats", "raw_agent", "raw_base", "scalar"]


def _dummy_inputs():
    return (torch.zeros(1, STACKED_GRID_CHANNELS, 16, 16),
            torch.zeros(1, 5, 11),
            torch.zeros(1, 7, 5, 25), torch.zeros(1, 7, 7, 25),
            torch.zeros(1, STACKED_SCALARS))


def export_actor(actor, out_path):
    """Trace `actor` to an ONNX file at `out_path`."""
    actor.eval()
    dynamic = {name: {0: "batch"} for name in _INPUT_NAMES}
    dynamic["logits"] = {0: "batch"}
    torch.onnx.export(
        actor,
        _dummy_inputs(),
        out_path,
        input_names=_INPUT_NAMES,
        output_names=["logits"],
        dynamic_axes=dynamic,
        opset_version=17,
        dynamo=False,
    )
    return out_path


def export_transformer(pt_path, onnx_path):
    """Load a SymbolicTransformerActor {state_dict,cfg} checkpoint and export."""
    actor = SymbolicTransformerActor.from_checkpoint(pt_path)
    return export_actor(actor, onnx_path)


def main():
    """CLI: export ae/src/policy_transformer_bc.pt -> ae/src/policy.onnx."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    pt_path = os.path.join(src, "policy_transformer_bc.pt")
    onnx_path = os.path.join(src, "policy.onnx")
    export_transformer(pt_path, onnx_path)
    print(f"exported {pt_path} -> {onnx_path}")


if __name__ == "__main__":
    main()
