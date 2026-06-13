"""DAgger behavior cloning of the `balanced` scripted teacher into the
SymbolicTransformerActor, then the BC gate.

DAgger pipeline (200 pure-teacher episodes + 2 x 100 DAgger rounds) training
the transformer actor. Model scale is read from env vars
(TF_D_MODEL, TF_N_LAYERS, TF_N_HEADS, TF_FFN_DIM, TF_DROPOUT) so scale sweeps
need no code edit. Saves ae/src/policy_transformer_bc.pt as {state_dict, cfg}.
"""
import copy
import os
import sys
import time

# Must precede `import torch` — the CUDA caching allocator reads this once at
# import time. `expandable_segments:True` lets the allocator grow / shrink
# segments to reduce fragmentation, which matters when the BC dataset is
# materialised on the GPU (3.5+ GB on round 1, ~12 GB by round 3 at K=5 stack).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from bc import bc_gate, collect_dagger_dataset, train_bc
from policy import SymbolicTransformerActor

TEACHER = os.environ.get("BC_TEACHER", "balanced_extreme_opening")
NUM_WORKERS = int(os.environ.get("BC_WORKERS", 1))
# 10 (vs original 20) — the dataset grows monotonically across DAgger rounds,
# so each round already gets ~1.5× more gradient steps than the last; halving
# epochs/round cuts the late-epoch overfitting spikes observed in long runs.
EPOCHS = int(os.environ.get("BC_EPOCHS", 10))
# 3e-4 (vs original 1e-3) — at near-zero training loss the 1e-3 LR was large
# enough to cause loss spikes. 3e-4 is a more typical Adam LR for transformers.
LR = float(os.environ.get("BC_LR", 3e-4))


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
    print(f"BC teacher: {TEACHER}", flush=True)
    print(f"BC workers: {NUM_WORKERS}", flush=True)
    print(f"BC epochs/round: {EPOCHS}  lr: {LR}", flush=True)
    print(f"transformer config: {cfg}", flush=True)
    actor = SymbolicTransformerActor(**cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = actor.to(device)
    if device.type == "cuda":
        print(f"actor moved to {device} "
              f"({torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)",
              flush=True)
    else:
        print("actor on CPU — BC training will be ~1000× slower than GPU. "
              "Verify CUDA is available.", flush=True)

    # Round 1 — pure teacher (beta = 1.0).
    print("R1: collecting 200 pure-teacher episodes...", flush=True)
    ds = collect_dagger_dataset(TEACHER, None, 1.0, 200, list(range(200)),
                                num_workers=NUM_WORKERS)
    print(f"  R1 dataset: {len(ds)} samples  [{time.time() - t0:.0f}s]",
          flush=True)
    train_bc(actor, ds, epochs=EPOCHS, lr=LR, verbose=True)
    print(f"  R1 trained  [{time.time() - t0:.0f}s]", flush=True)

    # Rounds 2-3 — DAgger aggregation with the partially-trained actor.
    for rnd in range(2):
        seeds = list(range(1000 + rnd * 1000, 1100 + rnd * 1000))
        print(f"R{rnd + 2}: collecting 100 DAgger episodes (beta=0.5)...",
              flush=True)
        cpu_actor = copy.deepcopy(actor).to("cpu")
        cpu_actor.eval()
        more = collect_dagger_dataset(TEACHER, cpu_actor, 0.5, 100, seeds,
                                      num_workers=NUM_WORKERS)
        ds += more
        print(f"  dataset now {len(ds)} samples  [{time.time() - t0:.0f}s]",
              flush=True)
        train_bc(actor, ds, epochs=EPOCHS, lr=LR, verbose=True)
        print(f"  R{rnd + 2} trained  [{time.time() - t0:.0f}s]", flush=True)

    print(f"FINAL dataset: {len(ds)} samples", flush=True)

    # Save BEFORE the gate so a gate-side crash (e.g. device mismatch, OOM)
    # cannot lose the trained model.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "policy_transformer_bc.pt")
    actor.save_checkpoint(out)
    print(f"saved {out}  [total {time.time() - t0:.0f}s]", flush=True)

    try:
        passed, detail = bc_gate(actor, TEACHER)
        print(f"BC GATE {'PASS' if passed else 'FAIL'}: {detail}", flush=True)
    except Exception as e:
        print(f"BC GATE crashed (model already saved): {type(e).__name__}: {e}",
              flush=True)


if __name__ == "__main__":
    main()
