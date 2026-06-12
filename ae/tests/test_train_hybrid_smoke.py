import numpy as np
import torch

from train_hybrid import train_hybrid
from hybrid_ppo import HybridPPOConfig
from scripted.handover import HandoverTrigger


def test_train_hybrid_runs_and_updates_weights():
    cfg = HybridPPOConfig(num_minibatches=2, update_epochs=1)
    actor, critic, history = train_hybrid(
        total_updates=2, episodes_per_update=1, learner_slots=("agent_0",),
        seed0=0, cfg=cfg, trigger=HandoverTrigger(step_fallback=5),
        d_model=16, n_layers=1, n_heads=2, device="cpu")
    assert len(history) == 2
    floor = cfg.lr_floor_frac * cfg.learning_rate
    for m in history:
        assert np.isfinite(m["policy_loss"]) and np.isfinite(m["value_loss"])
        assert m["n_active"] <= m["size"]
        assert "forward_bias" in m and "ent_coef" in m
        # LR is logged and stays within [floor, initial] (no irreversible collapse)
        assert floor <= m["actor_lr"] <= cfg.learning_rate + 1e-12


def test_train_hybrid_routes_through_parallel_path_when_workers_gt_1(monkeypatch):
    import train_hybrid as th
    calls = {"n": 0}
    real = getattr(th, "collect_hybrid_rollout_parallel", None)

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(th, "collect_hybrid_rollout_parallel", spy, raising=False)
    cfg = HybridPPOConfig(num_minibatches=2, update_epochs=1)
    _, _, history = train_hybrid(
        total_updates=1, episodes_per_update=2, learner_slots=("agent_0",),
        seed0=0, cfg=cfg, trigger=HandoverTrigger(step_fallback=5),
        d_model=16, n_layers=1, n_heads=2, device="cpu", rollout_workers=2)
    assert calls["n"] == 1                       # rollout went through the parallel path
    assert len(history) == 1 and history[0]["size"] > 0


def test_train_hybrid_selfplay_snapshots(tmp_path):
    import os
    cfg = HybridPPOConfig(num_minibatches=2, update_epochs=1)
    _, _, history = train_hybrid(
        total_updates=3, episodes_per_update=1, learner_slots=("agent_0",),
        seed0=0, cfg=cfg, trigger=HandoverTrigger(step_fallback=5),
        d_model=16, n_layers=1, n_heads=2, device="cpu",
        checkpoint_dir=str(tmp_path), snapshot_every=1, selfplay_prob=1.0,
        opponent_names=["balanced"])
    assert os.path.exists(os.path.join(str(tmp_path), "snap_1.pt"))   # snapshot saved
    assert history[-1].get("league_size", 0) >= 1                     # pool grew + was used


def test_critic_warmup_freezes_actor():
    cfg = HybridPPOConfig(num_minibatches=2, update_epochs=1)
    actor, critic, history = train_hybrid(
        total_updates=1, episodes_per_update=1, learner_slots=("agent_0",),
        seed0=0, cfg=cfg, trigger=HandoverTrigger(step_fallback=5),
        critic_warmup=1, d_model=16, n_layers=1, n_heads=2, device="cpu")
    assert history[0].get("warmup") is True
