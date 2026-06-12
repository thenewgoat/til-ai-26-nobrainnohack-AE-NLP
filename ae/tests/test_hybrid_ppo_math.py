import numpy as np

from hybrid_ppo import forward_bias_value, normalize_active_advantages


def test_forward_bias_starts_at_init():
    assert forward_bias_value(0, 1000, 1.0) == 1.0


def test_forward_bias_zero_in_hold_tail():
    assert forward_bias_value(850, 1000, 1.0, hold_frac=0.15) == 0.0
    assert forward_bias_value(999, 1000, 1.0, hold_frac=0.15) == 0.0


def test_forward_bias_monotonically_decreases_through_anneal():
    vals = [forward_bias_value(u, 1000, 1.0, hold_frac=0.15) for u in range(0, 851, 50)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert vals[0] == 1.0 and vals[-1] == 0.0


def test_forward_bias_inert_when_init_zero_or_trivial_horizon():
    assert forward_bias_value(5, 1000, 0.0) == 0.0
    assert forward_bias_value(0, 1, 1.0) == 0.0


def test_normalize_active_advantages_uses_only_active_stats():
    adv = np.array([10.0, -10.0, 100.0, 3.0], np.float32)
    active = np.array([True, True, False, False])
    out = normalize_active_advantages(adv, active)
    assert abs(out[active].mean()) < 1e-5
    assert abs(out[active].std() - 1.0) < 1e-4
    assert out[0] == -out[1]


def test_normalize_active_advantages_no_active_is_identity():
    adv = np.array([1.0, 2.0, 3.0], np.float32)
    active = np.array([False, False, False])
    out = normalize_active_advantages(adv, active)
    assert np.allclose(out, adv)


import torch

from hybrid_ppo import masked_mean, masked_surrogate


def test_masked_surrogate_ratio_one_gives_zero_kl():
    n = 8
    logp = torch.full((n,), -1.2)
    adv = torch.randn(n)
    active = torch.ones(n, dtype=torch.bool)
    loss, approx_kl, clipfrac, n_active = masked_surrogate(
        logp.clone(), logp.clone(), adv, active, clip_coef=0.2)
    assert abs(approx_kl) < 1e-6
    assert clipfrac == 0.0
    assert n_active == n


def test_masked_surrogate_only_counts_active_rows():
    n = 6
    new_logp = torch.tensor([-1.0, -1.0, -5.0, -5.0, -1.0, -1.0])
    old_logp = torch.full((n,), -1.0)
    adv = torch.ones(n)
    active = torch.tensor([True, True, False, False, True, True])
    loss, approx_kl, clipfrac, n_active = masked_surrogate(
        new_logp, old_logp, adv, active, clip_coef=0.2)
    assert n_active == 4
    assert abs(approx_kl) < 1e-6
    assert torch.isfinite(loss)


def test_masked_surrogate_no_active_returns_detached_zero():
    n = 4
    logp = torch.zeros(n, requires_grad=True)
    active = torch.zeros(n, dtype=torch.bool)
    loss, approx_kl, clipfrac, n_active = masked_surrogate(
        logp, logp, torch.ones(n), active, clip_coef=0.2)
    assert n_active == 0
    assert float(loss) == 0.0
    assert loss.grad_fn is None


def test_masked_surrogate_clips_large_ratios():
    new_logp = torch.tensor([0.0])
    old_logp = torch.tensor([-2.0])
    adv = torch.tensor([1.0])
    active = torch.ones(1, dtype=torch.bool)
    loss, _, clipfrac, _ = masked_surrogate(new_logp, old_logp, adv, active, clip_coef=0.2)
    assert abs(float(loss) - (-1.2)) < 1e-4
    assert clipfrac == 1.0


def test_masked_mean_over_active():
    x = torch.tensor([1.0, 2.0, 100.0, 100.0])
    active = torch.tensor([True, True, False, False])
    assert abs(float(masked_mean(x, active)) - 1.5) < 1e-6


def test_masked_mean_no_active_is_detached_zero():
    x = torch.zeros(3, requires_grad=True)
    out = masked_mean(x, torch.zeros(3, dtype=torch.bool))
    assert float(out) == 0.0
    assert out.grad_fn is None


# --- review #1: non-negative k3 approx_kl estimator ---

def test_approx_kl_is_nonnegative_when_naive_estimator_would_be_negative():
    # new > old on the sampled actions -> naive mean(old-new) is NEGATIVE; the
    # k3 estimator ((ratio-1) - log_ratio) is non-negative and is the value that
    # drives target_kl early-abort.
    n = 8
    new_logp = torch.full((n,), -0.5)
    old_logp = torch.full((n,), -1.0)
    adv = torch.zeros(n)
    active = torch.ones(n, dtype=torch.bool)
    _, approx_kl, _, _ = masked_surrogate(new_logp, old_logp, adv, active, clip_coef=0.2)
    assert approx_kl >= 0.0
    # exp(0.5) - 1 - 0.5 = 0.148721...
    assert abs(approx_kl - 0.148721) < 1e-4


# --- review #2: empty-active zero must be NaN-proof ---

def test_masked_surrogate_no_active_returns_zero_even_with_nan_rows():
    n = 4
    nan_logp = torch.full((n,), float("nan"), requires_grad=True)
    active = torch.zeros(n, dtype=torch.bool)
    loss, _, _, n_active = masked_surrogate(
        nan_logp, nan_logp, torch.ones(n), active, clip_coef=0.2)
    assert n_active == 0
    assert float(loss) == 0.0
    assert torch.isfinite(loss)
    assert loss.grad_fn is None


def test_masked_mean_no_active_zero_even_with_nan_rows():
    x = torch.tensor([float("nan"), float("nan"), 1.0])
    out = masked_mean(x, torch.zeros(3, dtype=torch.bool))
    assert float(out) == 0.0
    assert torch.isfinite(out)
    assert out.grad_fn is None
