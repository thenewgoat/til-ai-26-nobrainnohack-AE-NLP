"""Linear random-share schedule + mix-tuple derivation."""
import math

from train_selfplay import _interp_random_share, _compute_mix


def test_interp_at_first_update_returns_start():
    assert _interp_random_share(1, 1000, 0.4, 0.15) == 0.4


def test_interp_at_final_update_returns_end():
    assert math.isclose(_interp_random_share(1000, 1000, 0.4, 0.15), 0.15, abs_tol=1e-9)


def test_interp_at_midpoint_returns_average():
    v = _interp_random_share(501, 1001, 0.4, 0.15)
    assert math.isclose(v, (0.4 + 0.15) / 2, abs_tol=1e-6)


def test_interp_clamps_above_final():
    assert math.isclose(_interp_random_share(2000, 1000, 0.4, 0.15), 0.15, abs_tol=1e-9)


def test_interp_total_updates_one():
    assert _interp_random_share(1, 1, 0.4, 0.15) == 0.15


def test_compute_mix_league_absorbs_slack():
    league, random_, noisy, raw = _compute_mix(0.4, 0.15, 0.10)
    assert math.isclose(league, 1.0 - 0.4 - 0.15 - 0.10, abs_tol=1e-6)
    assert random_ == 0.4
    assert noisy == 0.15
    assert raw == 0.10


def test_compute_mix_sums_to_one():
    for r in (0.1, 0.25, 0.4, 0.6):
        mix = _compute_mix(r, 0.15, 0.10)
        assert math.isclose(sum(mix), 1.0, abs_tol=1e-6)


def test_compute_mix_clamps_negative_league_to_epsilon():
    # If shares sum > 1, league would go negative; clamp to 1e-6.
    league, _, _, _ = _compute_mix(0.7, 0.2, 0.2)
    assert league == 1e-6
