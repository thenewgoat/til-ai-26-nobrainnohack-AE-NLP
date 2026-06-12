import numpy as np
import torch

from critic import CentralizedCritic, STATE_PLANES, STATE_SCALARS
from features import (NUM_BASES, BASE_FIELDS, RAW_AGENT_SHAPE, RAW_BASE_SHAPE,
                      STACKED_GRID_CHANNELS, STACKED_SCALARS)
from hybrid_ppo import HybridPPOConfig, ppo_update_hybrid
from hybrid_rollout import build_hybrid_buffer
from policy import NUM_ACTIONS, SymbolicTransformerActor
from train_selfplay import compute_advantages, make_optimizer

GRID = 16


def _tr(actor_queried, proposed=1, executed=1, reward=0.0, done=0.0, mask=None):
    return dict(
        grid=np.random.randn(STACKED_GRID_CHANNELS, GRID, GRID).astype(np.float32),
        base_feats=np.zeros((NUM_BASES, BASE_FIELDS), np.float32),
        raw_agent=np.zeros(RAW_AGENT_SHAPE, np.float32),
        raw_base=np.zeros(RAW_BASE_SHAPE, np.float32),
        scalar=np.zeros(STACKED_SCALARS, np.float32),
        gstate=np.random.randn(STATE_PLANES, GRID, GRID).astype(np.float32),
        gscalar=np.zeros(STATE_SCALARS, np.float32),
        mask=(np.ones(NUM_ACTIONS, bool) if mask is None else mask),
        proposed=proposed, executed=executed, actor_queried=actor_queried,
        logp=0.0, reward=reward, env_reward=reward, done=done)


def _params_snapshot(m):
    return [p.detach().clone() for p in m.parameters()]


def _params_changed(before, m):
    return any(not torch.equal(b, p) for b, p in zip(before, m.parameters()))


# ---- Invariant 1: distribution reconstruction (determinism) ----

def test_distribution_reconstruction_deterministic_with_dropout_zero():
    a = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2, dropout=0.0).eval()
    g = torch.zeros(1, STACKED_GRID_CHANNELS, GRID, GRID)
    bf = torch.zeros(1, NUM_BASES, BASE_FIELDS)
    ra = torch.zeros(1, *RAW_AGENT_SHAPE)
    rb = torch.zeros(1, *RAW_BASE_SHAPE)
    sc = torch.zeros(1, STACKED_SCALARS)
    mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
    act = torch.tensor([1])
    _, lp1, _ = a.act(g, bf, ra, rb, sc, mask, action=act)
    _, lp2, _ = a.act(g, bf, ra, rb, sc, mask, action=act)
    assert torch.allclose(lp1, lp2, atol=1e-6)


# ---- Invariant 2: gradient routing ----

def test_all_forced_escape_buffer_leaves_actor_unchanged():
    a = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2, dropout=0.0)
    c = CentralizedCritic(d_model=16, n_layers=1, n_heads=2, dropout=0.0)
    buf = build_hybrid_buffer([_tr(False) for _ in range(8)], forward_bias=0.0)
    vals = np.zeros(buf.size, np.float32)
    adv, ret = compute_advantages(buf.rewards, vals, buf.dones, 0.99, 0.95)
    cfg = HybridPPOConfig(num_minibatches=2, update_epochs=1)
    opt = make_optimizer(a, c, cfg)
    before_a, before_c = _params_snapshot(a), _params_snapshot(c)
    ppo_update_hybrid(a, c, opt, buf, adv, ret, cfg, "cpu", check_determinism=False)
    assert not _params_changed(before_a, a)
    assert _params_changed(before_c, c)


def test_active_buffer_updates_actor():
    a = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2, dropout=0.0)
    c = CentralizedCritic(d_model=16, n_layers=1, n_heads=2, dropout=0.0)
    buf = build_hybrid_buffer([_tr(True, reward=1.0) for _ in range(8)], forward_bias=0.0)
    vals = np.zeros(buf.size, np.float32)
    adv, ret = compute_advantages(buf.rewards, vals, buf.dones, 0.99, 0.95)
    cfg = HybridPPOConfig(num_minibatches=2, update_epochs=1)
    opt = make_optimizer(a, c, cfg)
    before_a = _params_snapshot(a)
    ppo_update_hybrid(a, c, opt, buf, adv, ret, cfg, "cpu", check_determinism=False)
    assert _params_changed(before_a, a)


# ---- Invariant 3: GAE continuity across a run boundary ----

def test_gae_zeroes_bootstrap_at_done_boundary():
    rewards = np.array([0.0, 1.0, 0.0, 1.0], np.float32)
    values = np.array([5.0, 7.0, 9.0, 11.0], np.float32)
    dones = np.array([0.0, 1.0, 0.0, 0.0], np.float32)   # run A=[0,1], run B=[2,3]
    adv, ret = compute_advantages(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
    assert abs(adv[1] - (-6.0)) < 1e-5     # A terminal: r1 - v1 = 1 - 7
    assert abs(adv[0] - (-4.0)) < 1e-5     # A t=0: (0+v1-v0) + adv1 = 2 + (-6)
    assert abs(adv[3] - (-10.0)) < 1e-5    # B terminal (last index): r3 - v3 = 1 - 11
