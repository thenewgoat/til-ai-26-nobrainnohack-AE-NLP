from hybrid_ppo import HybridPPOConfig


def test_random_init_defaults():
    c = HybridPPOConfig()
    assert c.update_epochs == 1                  # random-init default (promote w/ evidence)
    assert c.target_kl == 0.02                   # NEW minibatch-level abort threshold
    assert c.kl_lr_halve_factor == 2.0
    assert c.clip_coef == 0.2
    assert 0.0 < c.learning_rate <= 1e-3         # random-init actor LR (pilot 1e-5..1e-4)
    assert c.critic_learning_rate == 1e-4
    assert c.ent_coef >= c.ent_coef_final
    assert c.num_minibatches == 8 and c.max_grad_norm == 0.5 and c.vf_coef == 0.5


def test_overridable():
    c = HybridPPOConfig(learning_rate=3e-5, target_kl=0.01, update_epochs=2)
    assert c.learning_rate == 3e-5 and c.target_kl == 0.01 and c.update_epochs == 2
