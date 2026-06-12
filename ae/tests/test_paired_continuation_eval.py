import numpy as np

from hybrid_eval import HybridAgent, paired_continuation_eval
from policy import SymbolicTransformerActor
from scripted.handover import HandoverTrigger


def test_paired_eval_produces_per_seed_deltas():
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    agent = HybridAgent(actor, trigger=HandoverTrigger(step_fallback=5))
    seeds = [0, 1]
    out = paired_continuation_eval(agent, seeds)
    assert len(out["deltas"]) == len(seeds)
    assert len(out["per_seed_a"]) == len(seeds)
    assert len(out["per_seed_b"]) == len(seeds)
    assert np.isfinite(out["mean"])
    assert out["ci_lo"] <= out["mean"] <= out["ci_hi"]
    # the delta is exactly B - A per seed (pairing is correct)
    for d, a, b in zip(out["deltas"], out["per_seed_a"], out["per_seed_b"]):
        assert abs(d - (b - a)) < 1e-9
