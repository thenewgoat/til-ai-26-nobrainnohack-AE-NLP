"""Parallel grid eval must give identical results to the serial one (the setups
are seed-derived and deterministic, so parallelization only changes wall-clock)."""
from evaluate import ScriptedAgent
from hybrid_eval import paired_grid_eval, paired_grid_eval_parallel


def test_parallel_grid_matches_serial():
    bank = ["balanced", "adaptive"]
    common = dict(opener_name="balanced_extreme_opening", opponent_bank=bank,
                  n_slots=2, configs_per_slot=1)
    serial = paired_grid_eval(ScriptedAgent("balanced_extreme_opening"), **common)
    par = paired_grid_eval_parallel(("scripted", "balanced_extreme_opening"),
                                    num_workers=2, **common)
    assert par["deltas"] == serial["deltas"]
    assert par["per_slot_mean"] == serial["per_slot_mean"]
