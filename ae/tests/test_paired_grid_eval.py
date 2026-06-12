"""Structured aggregate eval: 6 starting slots x N opponent configs, paired A/B.

Gives a balanced per-slot benchmark (every spawn covered equally) plus an
aggregate score, instead of random-seed coverage. Setups are seed-derived so the
A and B arms see the same (slot, opponents) per cell.
"""
from evaluate import ScriptedAgent
from hybrid_eval import paired_grid_eval


def test_grid_covers_all_slots_and_is_reproducible():
    bank = ["balanced", "adaptive", "defender"]
    kw = dict(opener_name="balanced_extreme_opening", opponent_bank=bank,
              n_slots=6, configs_per_slot=3, base_seed=0)
    out = paired_grid_eval(ScriptedAgent("balanced_extreme_opening"), **kw)
    assert out["n"] == 18                                  # 6 slots x 3 configs
    assert set(out["per_slot_mean"]) == set(range(6))      # every spawn covered
    # paired arms use the same per-cell setup, and the whole grid is reproducible
    out2 = paired_grid_eval(ScriptedAgent("balanced_extreme_opening"), **kw)
    assert out["deltas"] == out2["deltas"]
    assert "mean" in out and "ci_lo" in out and "ci_hi" in out
