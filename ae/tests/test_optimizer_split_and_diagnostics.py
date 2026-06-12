"""Adam param-groups split actor/critic LRs; ppo_update returns diagnostics."""
import numpy as np
import torch

from train_selfplay import (PPOConfig, make_optimizer, ppo_update,
                             RolloutBuffer)
from policy import SymbolicTransformerActor
from critic import CentralizedCritic, STATE_PLANES, STATE_SCALARS
from features import (STACKED_GRID_CHANNELS, STACKED_SCALARS, NUM_BASES,
                       BASE_FIELDS, RAW_AGENT_SHAPE, RAW_BASE_SHAPE)


def test_make_optimizer_has_two_param_groups_with_distinct_lrs():
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    critic = CentralizedCritic(d_model=16, n_layers=1, n_heads=2)
    cfg = PPOConfig()
    opt = make_optimizer(actor, critic, cfg)
    assert len(opt.param_groups) == 2
    lrs = sorted([g["lr"] for g in opt.param_groups])
    assert lrs == sorted([cfg.learning_rate, cfg.critic_learning_rate])
    # the larger-LR group is the critic; verify by parameter membership
    critic_params = set(id(p) for p in critic.parameters())
    critic_group = [g for g in opt.param_groups
                    if all(id(p) in critic_params for p in g["params"])]
    assert len(critic_group) == 1
    assert critic_group[0]["lr"] == cfg.critic_learning_rate


def test_default_critic_lr_meaningfully_higher_than_actor():
    """Critic LR must be at least 5× the actor LR (the whole point of the
    split is that the critic stays close to its pretraining LR while the
    actor moves cautiously). Exact ratio is tuned per smoke-run findings
    (was 10× → 33× as of 2026-05-23 to keep approx_kl < 0.03)."""
    cfg = PPOConfig()
    ratio = cfg.critic_learning_rate / cfg.learning_rate
    assert ratio >= 5.0, f"critic_lr/actor_lr = {ratio:.1f}× — too close to actor"


def _tiny_buffer(n=32):
    """A small RolloutBuffer with the K=5 stacked shapes."""
    return RolloutBuffer(
        grid=np.random.randn(n, STACKED_GRID_CHANNELS, 16, 16).astype(np.float32),
        base_feats=np.random.randn(n, NUM_BASES, BASE_FIELDS).astype(np.float32),
        raw_agent=np.random.randn(n, *RAW_AGENT_SHAPE).astype(np.float32),
        raw_base=np.random.randn(n, *RAW_BASE_SHAPE).astype(np.float32),
        scalar=np.random.randn(n, STACKED_SCALARS).astype(np.float32),
        gstate=np.random.randn(n, STATE_PLANES, 16, 16).astype(np.float32),
        gscalar=np.random.randn(n, STATE_SCALARS).astype(np.float32),
        actions=np.random.randint(0, 6, size=n).astype(np.int64),
        logprobs=np.random.randn(n).astype(np.float32),
        rewards=np.random.randn(n).astype(np.float32),
        env_rewards=np.random.randn(n).astype(np.float32),
        dones=np.zeros(n, np.float32),
        masks=np.ones((n, 6), bool),
    )


def test_ppo_update_returns_all_diagnostic_keys():
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    critic = CentralizedCritic(d_model=16, n_layers=1, n_heads=2)
    cfg = PPOConfig(num_minibatches=2, update_epochs=1)
    opt = make_optimizer(actor, critic, cfg)
    buf = _tiny_buffer(n=16)
    values = np.zeros(16, np.float32)
    advantages = buf.rewards.copy()       # arbitrary
    returns = advantages + values
    losses = ppo_update(actor, critic, opt, buf, advantages, returns, cfg,
                         torch.device("cpu"))
    for key in ("policy_loss", "value_loss", "entropy",
                "explained_variance", "approx_kl", "clipfrac",
                "advantage_mean", "advantage_std"):
        assert key in losses, f"missing key {key}"
        assert isinstance(losses[key], float), f"{key} is not float"


def test_explained_variance_one_for_perfect_critic():
    """If values exactly equal returns (so returns - values = 0), EV == 1."""
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    critic = CentralizedCritic(d_model=16, n_layers=1, n_heads=2)
    cfg = PPOConfig(num_minibatches=2, update_epochs=1)
    opt = make_optimizer(actor, critic, cfg)
    buf = _tiny_buffer(n=16)
    returns = np.random.randn(16).astype(np.float32) * 5.0   # var > 0
    advantages = np.zeros(16, np.float32)                    # so values == returns
    losses = ppo_update(actor, critic, opt, buf, advantages, returns, cfg,
                         torch.device("cpu"))
    # advantages = 0  =>  values = returns - advantages = returns
    # returns - values = 0  =>  EV = 1
    assert abs(losses["explained_variance"] - 1.0) < 1e-5


def test_explained_variance_zero_for_constant_critic():
    """If values are constant (no signal), Var(returns - values) == Var(returns),
    so EV == 0. We achieve this by setting advantages = returns (so values = 0)."""
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    critic = CentralizedCritic(d_model=16, n_layers=1, n_heads=2)
    cfg = PPOConfig(num_minibatches=2, update_epochs=1)
    opt = make_optimizer(actor, critic, cfg)
    buf = _tiny_buffer(n=16)
    returns = np.random.randn(16).astype(np.float32) * 5.0
    advantages = returns.copy()   # so values = returns - advantages = 0
    losses = ppo_update(actor, critic, opt, buf, advantages, returns, cfg,
                         torch.device("cpu"))
    # values is the zero vector; EV = 1 - Var(returns - 0)/Var(returns) = 0
    assert abs(losses["explained_variance"]) < 1e-5
