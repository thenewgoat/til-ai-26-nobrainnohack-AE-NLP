"""Pure PPO math primitives for the hybrid post-opener trainer.

These encode the reviewed corrections in isolation (no env, no training loop):
the actor is trained on its PROPOSAL over actor-queried ticks only (R1/R2),
advantages are normalized once over the actor-queried set (R4), and the forward
bias anneals to zero then holds (R6). Plan 5.2's `ppo_update_hybrid` wires these
into the minibatch loop.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class HybridPPOConfig:
    """Random-init PPO config for the hybrid post-opener trainer. Distinct from
    the BC-refiner `PPOConfig` in train_selfplay (random init wants movement, not
    a tiny LR). `target_kl` (NEW vs PPOConfig) drives the minibatch-level abort."""
    learning_rate: float = 1e-4          # actor LR — pilot-swept 1e-5..1e-4
    critic_learning_rate: float = 1e-4
    num_minibatches: int = 8
    update_epochs: int = 1               # random-init default; promote to 2 with evidence
    clip_coef: float = 0.2
    ent_coef: float = 0.001              # low: a high entropy bonus pins the action
    ent_coef_final: float = 0.0          # head toward uniform (head~=0). anneal ->0.
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02              # minibatch-level running-KL abort threshold
    kl_lr_halve_factor: float = 2.0      # decay actor LR if running KL > factor*target_kl
    kl_lr_decay: float = 0.7             # actor-LR multiplier on a large KL overshoot
    kl_lr_restore: float = 1.1           # actor-LR multiplier on a healthy update
    lr_floor_frac: float = 0.1           # actor LR never decays below frac*learning_rate


def adapt_actor_lr(current_lr, running_kl, cfg):
    """Return the next actor LR given this update's running KL.

    * large overshoot (running_kl > kl_lr_halve_factor*target_kl): decay by
      `kl_lr_decay`, but never below `lr_floor_frac * learning_rate`;
    * healthy update (running_kl <= target_kl): restore by `kl_lr_restore`,
      capped at the initial `learning_rate`;
    * mild overshoot (in between): hold.

    Unlike the old irreversible ×0.5 halve, this floors the decay and recovers on
    well-behaved updates, so an early-training KL transient can't permanently
    freeze the policy."""
    lr_init = cfg.learning_rate
    floor = cfg.lr_floor_frac * lr_init
    if cfg.target_kl is not None and running_kl > cfg.kl_lr_halve_factor * cfg.target_kl:
        return max(current_lr * cfg.kl_lr_decay, floor)
    if cfg.target_kl is not None and running_kl <= cfg.target_kl:
        return min(current_lr * cfg.kl_lr_restore, lr_init)
    return current_lr


def forward_bias_value(update, total_updates, init_bias, hold_frac=0.15):
    """Forward-bias magnitude for training `update`.

    Anneal linearly from `init_bias` to 0 over the first `(1 - hold_frac)` of
    training, then hold exactly 0 for the final `hold_frac` so the policy
    converges to the deploy (bias-free) distribution. Inert (0.0) when
    `init_bias <= 0` or the horizon is trivial."""
    if init_bias <= 0 or total_updates <= 1:
        return 0.0
    anneal_end = (1.0 - hold_frac) * total_updates
    progress = min(1.0, update / max(anneal_end - 1, 1))
    return float(init_bias * max(0.0, 1.0 - progress))


def normalize_active_advantages(adv, actor_queried):
    """Standardize advantages using ONLY the actor-queried entries' mean/std
    (computed once per rollout — R4). Returns a copy with the active entries
    standardized; inactive entries are left as-is (they're masked out of the
    actor loss). No active entries -> returned unchanged."""
    adv = np.asarray(adv, np.float32).copy()
    m = np.asarray(actor_queried, bool)
    if not m.any():
        return adv
    a = adv[m]
    adv[m] = (a - a.mean()) / (a.std() + 1e-8)
    return adv


import torch
import torch.nn as nn

from policy import NUM_ACTIONS
from scripted.geometry import FORWARD


def masked_mean(x, active):
    """Mean of `x` over the active rows; a graph-free zero tensor when no row is
    active (so the caller never adds a spurious actor-graph term that would let
    Adam step the actor on an empty minibatch)."""
    active = active.bool()
    if not bool(active.any()):
        return x.new_zeros(())          # exact zero (no graph, NaN-proof)
    return x[active].mean()


def masked_surrogate(new_logp, old_logp, adv_norm, active, clip_coef):
    """Clipped PPO surrogate over the actor-queried rows only (R1/R2).

    `new_logp`/`old_logp` are the PROPOSED action's log-probs. `adv_norm` is
    pre-normalized over the active set. Returns `(policy_loss, approx_kl,
    clipfrac, n_active)`; on an empty active set returns a graph-free zero loss
    so the caller can take a critic-only optimizer step without touching the
    actor."""
    active = active.bool()
    n_active = int(active.sum())
    if n_active == 0:
        # exact zero (no graph, NaN-proof) so the caller takes a critic-only step
        return new_logp.new_zeros(()), 0.0, 0.0, 0
    nl, ol, a = new_logp[active], old_logp[active], adv_norm[active]
    log_ratio = nl - ol
    ratio = log_ratio.exp()
    pg1 = -a * ratio
    pg2 = -a * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    policy_loss = torch.max(pg1, pg2).mean()
    with torch.no_grad():
        # k3 estimator (CleanRL): non-negative per sample, low variance — this is
        # the value that drives target_kl early-abort in Plan 5.2. The naive
        # mean(-log_ratio) can go negative on a minibatch and is unsuitable as a
        # stopping signal.
        approx_kl = float(((ratio - 1.0) - log_ratio).mean())
        clipfrac = float(((ratio - 1.0).abs() > clip_coef).float().mean())
    return policy_loss, approx_kl, clipfrac, n_active


def ppo_update_hybrid(actor, critic, opt, buf, advantages, returns, cfg, device,
                      check_determinism=True):
    """Proposal-policy PPO update for the hybrid post-opener trainer.

    Differs from the BC-refiner `ppo_update`:
      * re-evaluates the actor on its PROPOSED action with the stored forward bias
        (`logit_bias`) and mask — the PPO ratio is over the proposal (R1);
      * actor/entropy loss is masked to `actor_queried` ticks via `masked_surrogate`
        / `masked_mean`; the critic trains on ALL post-handover ticks (R2);
      * advantages are normalized ONCE over the active set (R4);
      * an all-inactive minibatch takes a critic-ONLY step (no actor graph, so Adam
        never steps the actor on zero gradient);
      * minibatch-level running k3-KL abort, with actor-LR halved on a large overshoot;
      * asserts re-eval determinism on the first active minibatch (catches mask/bias/
        dropout drift — the actor and critic must be built with dropout=0).

    `opt.param_groups[0]` must be the actor group (matches make_optimizer order).
    Returns a metrics dict; non-hybrid `ppo_update` is untouched."""
    n = buf.size
    # The LR-halve below mutates opt.param_groups[0] — guard it's the actor group.
    assert {id(p) for p in opt.param_groups[0]["params"]} == {id(p) for p in actor.parameters()}, \
        "opt.param_groups[0] must be the actor group (matches make_optimizer order)"
    g = torch.from_numpy(buf.grid).to(device)
    bf = torch.from_numpy(buf.base_feats).to(device)
    ra = torch.from_numpy(buf.raw_agent).to(device)
    rb = torch.from_numpy(buf.raw_base).to(device)
    sc = torch.from_numpy(buf.scalar).to(device)
    masks = torch.from_numpy(buf.masks).to(device)
    proposed = torch.from_numpy(buf.proposed_actions).to(device)
    old_logp = torch.from_numpy(buf.logprobs).to(device)
    actor_queried = torch.from_numpy(buf.actor_queried).to(device)
    gstate = torch.from_numpy(buf.gstate).to(device)
    gscalar = torch.from_numpy(buf.gscalar).to(device)
    ret = torch.from_numpy(np.asarray(returns, np.float32)).to(device)

    logit_bias = None
    if buf.forward_bias:
        logit_bias = torch.zeros(NUM_ACTIONS, device=device)
        logit_bias[FORWARD] = float(buf.forward_bias)

    adv_norm = torch.from_numpy(
        normalize_active_advantages(advantages, buf.actor_queried)).to(device)

    returns_np = np.asarray(returns, np.float32)
    values_np = returns_np - np.asarray(advantages, np.float32)
    var_returns = float(np.var(returns_np))
    explained_variance = (
        1.0 - float(np.var(returns_np - values_np)) / (var_returns + 1e-8)
        if var_returns > 0 else 0.0)

    mb_size = max(1, n // cfg.num_minibatches)
    inds = np.arange(n)
    actor.train()
    critic.train()
    pg = vl = ent = torch.zeros((), device=device)
    kl_sum = 0.0
    n_active_total = 0
    cf_sum = 0.0
    mb_count = 0
    running_kl = 0.0
    aborted = False
    checked = False
    for _ in range(cfg.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, n, mb_size):
            mb = inds[start:start + mb_size]
            mbt = torch.from_numpy(mb).to(device)
            _, new_logp, entropy = actor.act(
                g[mbt], bf[mbt], ra[mbt], rb[mbt], sc[mbt], masks[mbt],
                action=proposed[mbt], logit_bias=logit_bias)
            mq = actor_queried[mbt]

            if check_determinism and not checked and bool(mq.any()):
                with torch.no_grad():
                    a = mq.bool()
                    assert torch.allclose(new_logp[a], old_logp[mbt][a], atol=1e-4), (
                        "re-eval new_logp != stored old_logp on active rows — "
                        "mask/bias/temperature/dropout drift")
                checked = True

            policy_loss, approx_kl, clipfrac, n_act = masked_surrogate(
                new_logp, old_logp[mbt], adv_norm[mbt], mq, cfg.clip_coef)
            newvalue = critic(gstate[mbt], gscalar[mbt])
            vl = 0.5 * ((newvalue - ret[mbt]) ** 2).mean()
            if n_act > 0:
                pg = policy_loss
                ent = masked_mean(entropy, mq)
                loss = pg - cfg.ent_coef * ent + cfg.vf_coef * vl
            else:
                loss = cfg.vf_coef * vl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip = [p for p in (list(actor.parameters()) + list(critic.parameters()))
                    if p.grad is not None]
            nn.utils.clip_grad_norm_(clip, cfg.max_grad_norm)
            opt.step()

            mb_count += 1
            cf_sum += clipfrac
            kl_sum += approx_kl * n_act
            n_active_total += n_act
            running_kl = kl_sum / max(1, n_active_total)
            # Abort AFTER the step that produced this KL — the overshoot step is
            # applied, not rolled back (standard PPO early-stop). The actor LR is
            # adapted once after the loop from the final running KL.
            if cfg.target_kl is not None and running_kl > cfg.target_kl:
                aborted = True
                break
        if aborted:
            break
    # Decay (floored) on a large overshoot, restore (capped) on a healthy update.
    opt.param_groups[0]["lr"] = adapt_actor_lr(
        opt.param_groups[0]["lr"], running_kl, cfg)
    return {
        "policy_loss": float(pg.detach()),
        "value_loss": float(vl.detach()),
        "entropy": float(ent.detach()),
        "explained_variance": explained_variance,
        "approx_kl": kl_sum / max(1, n_active_total),
        "clipfrac": cf_sum / max(1, mb_count),
        "n_active": int(np.asarray(buf.actor_queried, bool).sum()),
        "minibatches": mb_count,
        "aborted": aborted,
        "actor_lr": float(opt.param_groups[0]["lr"]),
    }
