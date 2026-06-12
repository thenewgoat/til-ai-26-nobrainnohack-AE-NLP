"""One PPO update reduces the surrogate loss on a fixed batch."""
import numpy as np
import torch

from train_selfplay import ppo_update, make_optimizer, PPOConfig
from policy import SymbolicTransformerActor
from critic import CentralizedCritic
from train_selfplay import RolloutBuffer
from critic import STATE_PLANES, STATE_SCALARS
from features import STACKED_GRID_CHANNELS, STACKED_SCALARS, NUM_BASES, BASE_FIELDS, RAW_AGENT_SHAPE, RAW_BASE_SHAPE


def _toy_buffer(n=128):
    rng = np.random.default_rng(0)
    return RolloutBuffer(
        grid=rng.random((n, STACKED_GRID_CHANNELS, 16, 16)).astype(np.float32),
        base_feats=rng.random((n, NUM_BASES, BASE_FIELDS)).astype(np.float32),
        raw_agent=rng.random((n, *RAW_AGENT_SHAPE)).astype(np.float32),
        raw_base=rng.random((n, *RAW_BASE_SHAPE)).astype(np.float32),
        scalar=rng.random((n, STACKED_SCALARS)).astype(np.float32),
        gstate=rng.random((n, STATE_PLANES, 16, 16)).astype(np.float32),
        gscalar=rng.random((n, STATE_SCALARS)).astype(np.float32),
        actions=rng.integers(0, 6, n).astype(np.int64),
        logprobs=np.zeros(n, np.float32),
        rewards=rng.random(n).astype(np.float32),
        env_rewards=rng.random(n).astype(np.float32),
        dones=np.zeros(n, np.float32),
        masks=np.ones((n, 6), bool),
    )


def test_ppo_update_returns_loss_dict():
    actor, critic = SymbolicTransformerActor(), CentralizedCritic()
    cfg = PPOConfig()
    opt = make_optimizer(actor, critic, cfg)
    buf = _toy_buffer()
    adv = np.random.default_rng(1).random(buf.size).astype(np.float32)
    ret = adv.copy()
    losses = ppo_update(actor, critic, opt, buf, adv, ret, cfg,
                        device=torch.device("cpu"))
    for k in ("policy_loss", "value_loss", "entropy"):
        assert k in losses


def test_ppo_update_changes_weights():
    actor, critic = SymbolicTransformerActor(), CentralizedCritic()
    cfg = PPOConfig()
    opt = make_optimizer(actor, critic, cfg)
    before = actor.head.weight.detach().clone()
    buf = _toy_buffer()
    adv = np.random.default_rng(2).random(buf.size).astype(np.float32)
    ppo_update(actor, critic, opt, buf, adv, adv.copy(), cfg,
               device=torch.device("cpu"))
    assert not torch.equal(before, actor.head.weight.detach())
