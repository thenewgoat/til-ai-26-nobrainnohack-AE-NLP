"""One-off driver: large-dataset behavior cloning of the `balanced` scripted
agent (the HP-aware-offense teacher) into the SymbolicTransformerActor, then
the BC gate.

Same DAgger pipeline as bc.main(), scaled up: 200 pure-teacher episodes +
2 x 100 DAgger rounds = 400 episodes (~80k teacher-labelled samples).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from bc import bc_gate, collect_dagger_dataset, train_bc
from policy import SymbolicTransformerActor

TEACHER = "balanced"


def main():
    t0 = time.time()
    actor = SymbolicTransformerActor()

    # Round 1 — pure teacher (beta = 1.0).
    print("R1: collecting 200 pure-teacher episodes...", flush=True)
    ds = collect_dagger_dataset(TEACHER, None, 1.0, 200, list(range(200)))
    print(f"  R1 dataset: {len(ds)} samples  [{time.time() - t0:.0f}s]",
          flush=True)
    train_bc(actor, ds, epochs=20)
    print(f"  R1 trained  [{time.time() - t0:.0f}s]", flush=True)

    # Rounds 2-3 — DAgger aggregation with the partially-trained actor.
    for rnd in range(2):
        seeds = list(range(1000 + rnd * 1000, 1100 + rnd * 1000))
        print(f"R{rnd + 2}: collecting 100 DAgger episodes (beta=0.5)...",
              flush=True)
        more = collect_dagger_dataset(TEACHER, actor, 0.5, 100, seeds)
        ds += more
        print(f"  dataset now {len(ds)} samples  [{time.time() - t0:.0f}s]",
              flush=True)
        train_bc(actor, ds, epochs=20)
        print(f"  R{rnd + 2} trained  [{time.time() - t0:.0f}s]", flush=True)

    print(f"FINAL dataset: {len(ds)} samples", flush=True)

    passed, detail = bc_gate(actor, TEACHER)
    print(f"BC GATE {'PASS' if passed else 'FAIL'}: {detail}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "policy_bc.pt")
    actor.save_checkpoint(out)
    print(f"saved {out}  [total {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
