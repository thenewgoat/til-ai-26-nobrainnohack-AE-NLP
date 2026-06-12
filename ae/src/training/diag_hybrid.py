"""One-off diagnostic: why doesn't the hybrid post-handover actor improve?

Measures, with a fresh actor:
  1. gate-override rate  — fraction of actor-queried ticks where executed != proposed
                           (PPO trains `proposed` against the executed trajectory's
                           advantage; high override = miscredited gradient).
  2. advantage stats     — mean/std on actor-queried ticks (is the signal real or noise?).
  3. real policy drift   — actor action distribution on a FIXED held-out state batch,
                           before vs after N training updates (does the policy actually
                           move? — robust to the approx_kl pre-step measurement artifact).

Run:
  cd ae && PYTHONPATH=src:src/training ../til-26-ae/.venv/bin/python src/training/diag_hybrid.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from critic import CentralizedCritic
from hybrid_ppo import HybridPPOConfig
from hybrid_rollout import collect_hybrid_rollout
from policy import SymbolicTransformerActor
from scripted.handover import HandoverTrigger
from train_hybrid import GAMMA, GAE_LAMBDA, train_hybrid
from train_selfplay import compute_advantages, critic_values

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPPS = ["balanced", "balanced_extreme_opening", "adaptive", "forager", "defender"]
UPDATES = 30
K = 256                                   # held-out fixed-state batch size


def _fixed_batch(buf, idx, device):
    take = lambda a: torch.from_numpy(a[idx]).to(device)
    return dict(grid=take(buf.grid), base_feats=take(buf.base_feats),
                raw_agent=take(buf.raw_agent), raw_base=take(buf.raw_base),
                scalar=take(buf.scalar), mask=take(buf.masks))


def _dist(actor, b):
    actor.eval()
    with torch.no_grad():
        lg = actor.forward(b["grid"], b["base_feats"], b["raw_agent"],
                           b["raw_base"], b["scalar"])
        lg = torch.where(b["mask"], lg, torch.full_like(lg, -1e8))
        return F.softmax(lg, dim=-1)


def main():
    torch.manual_seed(0)
    actor = SymbolicTransformerActor(d_model=64, n_layers=4, n_heads=4, dropout=0.0).to(DEVICE)
    critic = CentralizedCritic(d_model=64, n_layers=4, n_heads=4, dropout=0.0).to(DEVICE)
    trig = HandoverTrigger(step_fallback=60)

    # ---- initial rollout: override rate + advantage stats ----
    buf = collect_hybrid_rollout(actor, ["agent_0"], 8, 0, trigger=trig,
                                 forward_bias=0.5, opponent_names=OPPS)
    aq = buf.actor_queried.astype(bool)
    override = (buf.executed_actions != buf.proposed_actions) & aq
    override_rate = override.sum() / max(1, aq.sum())

    vals = critic_values(critic, buf.gstate, buf.gscalar, DEVICE)
    adv, ret = compute_advantages(buf.rewards, vals, buf.dones, GAMMA, GAE_LAMBDA)
    adv_aq = np.asarray(adv)[aq]

    # per-action executed distribution on actor-queried ticks (what actually happened)
    exec_hist = np.bincount(buf.executed_actions[aq], minlength=6) / max(1, aq.sum())
    prop_hist = np.bincount(buf.proposed_actions[aq], minlength=6) / max(1, aq.sum())

    print("=== ROLLOUT DIAGNOSTICS (fresh actor, 8 episodes) ===")
    print(f"post-handover ticks      : {buf.size}")
    print(f"actor-queried ticks      : {int(aq.sum())} ({aq.mean()*100:.1f}%)")
    print(f"GATE-OVERRIDE RATE        : {override_rate*100:.1f}%  (executed != proposed | queried)")
    print(f"advantage mean/std (aq)  : {adv_aq.mean():+.4f} / {adv_aq.std():.4f}")
    print(f"reward mean/std          : {buf.rewards.mean():+.4f} / {buf.rewards.std():.4f}")
    print(f"proposed action hist     : {np.round(prop_hist,3)}  (0=fwd 1=back 2=left 3=right 4=stay 5=bomb)")
    print(f"executed action hist     : {np.round(exec_hist,3)}")

    # ---- fixed held-out batch + policy snapshot ----
    aq_idx = np.where(aq)[0][:K]
    batch = _fixed_batch(buf, aq_idx, DEVICE)
    d_before = _dist(actor, batch)
    ent_before = -(d_before * (d_before + 1e-9).log()).sum(-1).mean().item()

    # ---- train N updates with the moderate cold-start bump ----
    cfg = HybridPPOConfig(learning_rate=3e-4, update_epochs=2)
    print(f"\n=== TRAINING {UPDATES} updates (lr=3e-4, update_epochs=2, 4 workers) ===")
    actor, critic, hist = train_hybrid(
        total_updates=UPDATES, episodes_per_update=4, cfg=cfg,
        forward_bias_init=0.5, anti_idle_penalty=0.05, trigger=trig,
        opponent_names=OPPS, device=DEVICE, rollout_workers=4,
        actor=actor, critic=critic)

    # ---- policy drift on the SAME fixed batch ----
    d_after = _dist(actor.to(DEVICE), batch)
    ent_after = -(d_after * (d_after + 1e-9).log()).sum(-1).mean().item()
    kl_drift = (d_before * ((d_before + 1e-9).log() - (d_after + 1e-9).log())).sum(-1).mean().item()

    rets = [m.get("post_handover_return", 0.0) for m in hist if "post_handover_return" in m]
    print("\n=== POLICY DRIFT (fixed 256-state batch, before vs after) ===")
    print(f"entropy before/after     : {ent_before:.4f} -> {ent_after:.4f}  (max ln6={np.log(6):.4f})")
    print(f"KL(before||after)        : {kl_drift:.5f}   <- real policy movement")
    print(f"post_handover_return     : first5={np.round(rets[:5],1)}  last5={np.round(rets[-5:],1)}")


if __name__ == "__main__":
    main()
