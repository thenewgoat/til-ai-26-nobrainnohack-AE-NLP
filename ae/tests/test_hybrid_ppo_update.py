import numpy as np
import pytest
import torch

from critic import CentralizedCritic
from hybrid_ppo import HybridPPOConfig, ppo_update_hybrid
from policy import SymbolicTransformerActor


def _tiny():
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2, dropout=0.0)
    critic = CentralizedCritic(d_model=16, n_layers=1, n_heads=2, dropout=0.0)
    return actor.eval(), critic.eval()


def _opt(actor, critic, lr=1e-3):
    # param_groups[0] MUST be the actor (ppo_update_hybrid halves group 0's LR)
    return torch.optim.Adam([
        {"params": list(actor.parameters()), "lr": lr},
        {"params": list(critic.parameters()), "lr": lr},
    ], eps=1e-5)


class _Buf:
    def __init__(self, n, proposed, actor_queried, old_logp, forward_bias=0.0):
        self.size = n
        self.grid = np.zeros((n, 85, 16, 16), np.float32)
        self.base_feats = np.zeros((n, 5, 11), np.float32)
        self.raw_agent = np.zeros((n, 7, 5, 25), np.float32)
        self.raw_base = np.zeros((n, 7, 7, 25), np.float32)
        self.scalar = np.zeros((n, 50), np.float32)
        self.masks = np.ones((n, 6), bool)
        self.proposed_actions = np.asarray(proposed, np.int64)
        self.logprobs = np.asarray(old_logp, np.float32)
        self.actor_queried = np.asarray(actor_queried, bool)
        self.gstate = np.zeros((n, 9, 16, 16), np.float32)
        self.gscalar = np.zeros((n, 6), np.float32)
        self.forward_bias = forward_bias


def _matching_old_logp(actor, buf):
    with torch.no_grad():
        _, lp, _ = actor.act(
            torch.from_numpy(buf.grid), torch.from_numpy(buf.base_feats),
            torch.from_numpy(buf.raw_agent), torch.from_numpy(buf.raw_base),
            torch.from_numpy(buf.scalar), torch.from_numpy(buf.masks),
            action=torch.from_numpy(buf.proposed_actions), logit_bias=None)
    return lp.numpy().astype(np.float32)


def _changed(params, snapshot):
    return any(not torch.equal(p, q) for p, q in zip(params, snapshot))


def _snap(module):
    return [p.detach().clone() for p in module.parameters()]


def test_reeval_determinism_holds_and_actor_trains():
    torch.manual_seed(0)
    actor, critic = _tiny()
    n = 6
    buf = _Buf(n, proposed=[0, 1, 2, 3, 4, 0], actor_queried=[True] * n, old_logp=np.zeros(n))
    buf.logprobs = _matching_old_logp(actor, buf)
    cfg = HybridPPOConfig(num_minibatches=2, target_kl=100.0)
    opt = _opt(actor, critic)
    a0 = _snap(actor)
    adv = np.array([1., 2., 3., 4., 5., 6.], np.float32)
    ret = np.zeros(n, np.float32)
    m = ppo_update_hybrid(actor, critic, opt, buf, adv, ret, cfg, "cpu", check_determinism=True)
    assert m["approx_kl"] >= 0.0 and m["n_active"] == n and not m["aborted"]
    assert _changed(actor.parameters(), a0)


def test_empty_active_is_critic_only():
    torch.manual_seed(0)
    actor, critic = _tiny()
    n = 6
    buf = _Buf(n, proposed=[0] * n, actor_queried=[False] * n, old_logp=np.zeros(n))
    cfg = HybridPPOConfig(num_minibatches=2, target_kl=100.0)
    opt = _opt(actor, critic)
    a0, c0 = _snap(actor), _snap(critic)
    adv = np.ones(n, np.float32)
    ret = np.ones(n, np.float32)
    m = ppo_update_hybrid(actor, critic, opt, buf, adv, ret, cfg, "cpu", check_determinism=False)
    assert m["n_active"] == 0
    assert not _changed(actor.parameters(), a0)
    assert _changed(critic.parameters(), c0)


def test_determinism_assert_raises_on_mismatch():
    actor, critic = _tiny()
    n = 4
    buf = _Buf(n, proposed=[0, 1, 2, 3], actor_queried=[True] * n, old_logp=np.zeros(n))
    buf.logprobs = _matching_old_logp(actor, buf) + 5.0
    cfg = HybridPPOConfig(num_minibatches=1, target_kl=100.0)
    opt = _opt(actor, critic)
    with pytest.raises(AssertionError):
        ppo_update_hybrid(actor, critic, opt, buf, np.ones(n, np.float32),
                          np.zeros(n, np.float32), cfg, "cpu", check_determinism=True)


def test_kl_abort_breaks_early_and_decays_lr():
    torch.manual_seed(0)
    actor, critic = _tiny()
    n = 8
    buf = _Buf(n, proposed=[0] * n, actor_queried=[True] * n, old_logp=np.zeros(n))
    cfg = HybridPPOConfig(num_minibatches=1, target_kl=1e-9)
    opt = _opt(actor, critic)          # _opt sets actor LR = 1e-3
    m = ppo_update_hybrid(actor, critic, opt, buf, np.ones(n, np.float32),
                          np.zeros(n, np.float32), cfg, "cpu", check_determinism=False)
    assert m["aborted"] is True
    assert m["minibatches"] == 1
    # large overshoot -> gentle decay by kl_lr_decay (0.7), not the old ×0.5 halve
    assert opt.param_groups[0]["lr"] == pytest.approx(0.7e-3)
    assert m["actor_lr"] == pytest.approx(0.7e-3)


def test_mixed_active_inactive_minibatch_trains_actor_on_active_rows():
    # A single minibatch with BOTH active and inactive rows exercises the internal
    # masking in masked_surrogate/masked_mean (the subtlest gradient-routing path).
    torch.manual_seed(0)
    actor, critic = _tiny()
    n = 4
    buf = _Buf(n, proposed=[0, 1, 2, 3], actor_queried=[True, False, True, False], old_logp=np.zeros(n))
    buf.logprobs = _matching_old_logp(actor, buf)
    cfg = HybridPPOConfig(num_minibatches=1, target_kl=100.0)   # one mixed minibatch
    opt = _opt(actor, critic)
    a0 = _snap(actor)
    adv = np.array([1., 2., 3., 4.], np.float32)
    m = ppo_update_hybrid(actor, critic, opt, buf, adv, np.zeros(n, np.float32),
                          cfg, "cpu", check_determinism=True)
    assert m["n_active"] == 2
    assert _changed(actor.parameters(), a0)        # active rows train the actor


def test_forward_bias_is_applied_in_the_reeval():
    # old_logp is seeded from a BIASED re-eval; the in-update re-eval must apply the
    # SAME forward bias or the determinism assert mismatches. So this passing proves
    # the bias actually reaches actor.act in ppo_update_hybrid.
    torch.manual_seed(0)
    actor, critic = _tiny()
    n = 4
    buf = _Buf(n, proposed=[0, 1, 2, 3], actor_queried=[True] * n, old_logp=np.zeros(n),
               forward_bias=1.0)
    lb = torch.zeros(6)
    lb[0] = 1.0                                     # FORWARD bias matching buf.forward_bias
    with torch.no_grad():
        _, lp, _ = actor.act(
            torch.from_numpy(buf.grid), torch.from_numpy(buf.base_feats),
            torch.from_numpy(buf.raw_agent), torch.from_numpy(buf.raw_base),
            torch.from_numpy(buf.scalar), torch.from_numpy(buf.masks),
            action=torch.from_numpy(buf.proposed_actions), logit_bias=lb)
    buf.logprobs = lp.numpy().astype(np.float32)
    cfg = HybridPPOConfig(num_minibatches=1, target_kl=100.0)
    opt = _opt(actor, critic)
    # check_determinism=True will RAISE if the re-eval omits the bias
    m = ppo_update_hybrid(actor, critic, opt, buf, np.array([1., 2., 3., 4.], np.float32),
                          np.zeros(n, np.float32), cfg, "cpu", check_determinism=True)
    assert not m["aborted"]


def test_forward_bias_mismatch_is_caught_by_determinism():
    # Seed old_logp WITHOUT bias but tell the update forward_bias=1.0 -> re-eval applies
    # the bias -> new_logp != old_logp -> determinism assert fires. Confirms the bias
    # genuinely changes the re-eval (not a no-op).
    torch.manual_seed(0)
    actor, critic = _tiny()
    n = 4
    buf = _Buf(n, proposed=[0, 1, 2, 3], actor_queried=[True] * n, old_logp=np.zeros(n),
               forward_bias=5.0)
    buf.logprobs = _matching_old_logp(actor, buf)   # UNBIASED old_logp
    cfg = HybridPPOConfig(num_minibatches=1, target_kl=100.0)
    opt = _opt(actor, critic)
    with pytest.raises(AssertionError):
        ppo_update_hybrid(actor, critic, opt, buf, np.ones(n, np.float32),
                          np.zeros(n, np.float32), cfg, "cpu", check_determinism=True)
