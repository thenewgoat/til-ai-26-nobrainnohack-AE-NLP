"""Train SymbolicTransformerActor on the family-winner BC dataset.

Loads ae/training/logs/family_winner.pt (produced by collect_family_winner.py),
runs bc.train_bc on the included BCSamples, evaluates the clone with bc_gate
against both balanced-family teachers, and saves the trained actor as
ae/src/policy_family_winner_bc.pt.

Model scale is read from env vars (TF_D_MODEL, TF_N_LAYERS, TF_N_HEADS,
TF_FFN_DIM, TF_DROPOUT) so scale sweeps need no code edit.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import torch

from bc import bc_gate, train_bc
from policy import SymbolicTransformerActor


def _config_from_env():
    """Build the SymbolicTransformerActor config from TF_* env vars."""
    cfg = {
        "d_model": int(os.environ.get("TF_D_MODEL", 64)),
        "n_layers": int(os.environ.get("TF_N_LAYERS", 4)),
        "n_heads": int(os.environ.get("TF_N_HEADS", 4)),
        "dropout": float(os.environ.get("TF_DROPOUT", 0.1)),
    }
    ffn = os.environ.get("TF_FFN_DIM")
    if ffn is not None:
        cfg["ffn_dim"] = int(ffn)
    return cfg


def main():
    t0 = time.time()
    cfg = _config_from_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  |  transformer config: {cfg}", flush=True)
    actor = SymbolicTransformerActor(**cfg).to(device)

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "logs", "family_winner.pt")
    print(f"loading {data_path} ...", flush=True)
    data = torch.load(data_path, weights_only=False)
    samples = data["samples"]
    m = data["meta"]
    print(f"  {len(samples)} samples, "
          f"kept {m['episodes_kept']}/{m['episodes_run']} episodes",
          flush=True)

    print("training...", flush=True)
    train_bc(actor, samples, epochs=20, verbose=True)
    print(f"trained  [{time.time() - t0:.0f}s]", flush=True)

    # bc_gate's NeuralAgent builds CPU tensors; move the actor back for eval
    # and so the saved checkpoint is portable.
    actor.cpu()
    for teacher in ("balanced", "balanced_extreme"):
        passed, detail = bc_gate(actor, teacher)
        flag = "PASS" if passed else "FAIL"
        print(f"BC GATE vs {teacher}: {flag}  {detail}", flush=True)

    out = os.path.join(here, "..", "policy_family_winner_bc.pt")
    actor.save_checkpoint(out)
    print(f"saved {out}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
