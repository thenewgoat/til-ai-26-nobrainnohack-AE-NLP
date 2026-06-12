"""Actor LR adaptation: gentle decay with a floor on KL overshoot, and restore
toward the initial LR on healthy updates.

The old logic halved the actor LR irreversibly (×0.5, no floor, no restore), so a
few large-KL updates from random init collapsed the LR to ~0 and froze the policy
for the rest of training. The new logic decays gently, never below a floor, and
recovers when updates are well-behaved.
"""
import math

from hybrid_ppo import HybridPPOConfig, adapt_actor_lr


def _cfg(**kw):
    return HybridPPOConfig(**kw)


def test_large_overshoot_decays_by_decay_factor():
    cfg = _cfg()  # decay 0.7, halve_factor 2.0, target_kl 0.02, lr 1e-4
    new = adapt_actor_lr(1e-4, running_kl=0.10, cfg=cfg)   # 0.10 > 2*0.02
    assert math.isclose(new, 0.7e-4, rel_tol=1e-6)         # gentle, not ×0.5


def test_decay_never_below_floor():
    cfg = _cfg()                                           # floor = 0.1 * 1e-4 = 1e-5
    near_floor = 1.1e-5
    new = adapt_actor_lr(near_floor, running_kl=0.10, cfg=cfg)  # 0.7*1.1e-5 < floor
    assert math.isclose(new, 1e-5, rel_tol=1e-6)


def test_healthy_update_restores_toward_init():
    cfg = _cfg()
    new = adapt_actor_lr(5e-5, running_kl=0.0, cfg=cfg)    # <= target_kl -> restore
    assert math.isclose(new, 1.1 * 5e-5, rel_tol=1e-6)


def test_restore_capped_at_initial_lr():
    cfg = _cfg()
    new = adapt_actor_lr(9.9e-5, running_kl=0.0, cfg=cfg)  # 1.1*9.9e-5 > init
    assert math.isclose(new, 1e-4, rel_tol=1e-6)           # never exceeds initial LR


def test_mild_overshoot_holds_lr():
    cfg = _cfg()
    new = adapt_actor_lr(5e-5, running_kl=0.03, cfg=cfg)   # target_kl < kl <= 2*target_kl
    assert new == 5e-5
