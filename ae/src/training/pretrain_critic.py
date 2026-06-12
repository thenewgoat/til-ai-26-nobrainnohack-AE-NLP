"""Pretrain the centralized critic by MSE regression on reward-to-go.

A random critic at PPO start gives garbage advantages (advantage = return -
V(s)); paired with a competent BC actor it makes the policy gradient noisy or
wrong from update 1 — a likely contributor to the observed value_loss spikes
and collapse. This script fits CentralizedCritic to Monte-Carlo reward-to-go
targets from collect_critic_data.py rollouts.

Targets use the SAME raw env-reward scale and gamma PPO uses (see
train_selfplay Args.gamma / the unshaped GAE in compute_advantages), so the
warm-started critic drops into train_selfplay with no recalibration mismatch.
Track val_explained_var: a constant predictor scores 0, so PPO should not
start actor updates until it is comfortably positive.
"""
import argparse
import os
import sys

# Must precede `import torch` — the CUDA caching allocator reads this once at
# import time. `expandable_segments:True` mitigates fragmentation for the
# variable-batch critic-pretrain loop.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from critic import CentralizedCritic


def reward_to_go(env_rewards, episode_len, gamma):
    """Monte-Carlo discounted return-to-go per fixed-length episode run.

    The collect buffer is concatenated contiguous runs of `episode_len`
    transitions (one episode-slot each — see train_selfplay.collect_rollout's
    slot_base layout). G_t = r_t + gamma * G_{t+1}, reset at each run boundary.
    """
    runs = env_rewards.reshape(-1, episode_len).astype(np.float64)
    rtg = np.zeros_like(runs)
    g = np.zeros(runs.shape[0])
    for t in reversed(range(episode_len)):
        g = runs[:, t] + gamma * g
        rtg[:, t] = g
    return rtg.reshape(-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain CentralizedCritic on reward-to-go")
    parser.add_argument("--data", default="logs/critic_pretrain_data.pt",
                        help="collect_critic_data.py output, rel. to ae/training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount — MUST match train_selfplay Args.gamma")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="env-reward multiplier before computing returns; "
                             "keep 1.0 to match PPO's raw-reward GAE")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="critic_pretrained.pt",
                        help="output path, rel. to ae/training (kept out of "
                             "ae/src so it is never shipped in the container)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    here = os.path.dirname(os.path.abspath(__file__))

    data = torch.load(os.path.join(here, args.data), weights_only=False)
    gstate = np.asarray(data["gstate"], np.float32)
    gscalar = np.asarray(data["gscalar"], np.float32)
    env_rewards = np.asarray(data["env_rewards"], np.float32) * args.reward_scale
    episode_len = int(data["episode_len"])

    targets = reward_to_go(env_rewards, episode_len, args.gamma)
    n = gstate.shape[0]
    print(f"{n} transitions  target mean {targets.mean():.2f} "
          f"std {targets.std():.2f} "
          f"min {targets.min():.2f} max {targets.max():.2f}")

    perm = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = CentralizedCritic().to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

    gstate_t = torch.from_numpy(gstate)
    gscalar_t = torch.from_numpy(gscalar)
    targets_t = torch.from_numpy(targets)

    def evaluate(idx):
        """MSE and explained variance over `idx`. EV: 1 - MSE/Var(target);
        a constant predictor scores 0, a perfect critic scores 1."""
        critic.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), args.batch_size):
                mb = idx[s:s + args.batch_size]
                v = critic(gstate_t[mb].to(device), gscalar_t[mb].to(device))
                preds.append(v.cpu())
        pred = torch.cat(preds)
        tgt = targets_t[idx]
        mse = ((pred - tgt) ** 2).mean().item()
        var = tgt.var(unbiased=False).item()
        ev = 1.0 - mse / var if var > 1e-8 else 0.0
        return mse, ev

    epochs = tqdm(range(1, args.epochs + 1), desc="critic pretrain", unit="ep")
    for epoch in epochs:
        critic.train()
        np.random.shuffle(train_idx)
        tot = 0.0
        starts = range(0, len(train_idx), args.batch_size)
        mb_bar = tqdm(starts, desc=f"epoch {epoch}/{args.epochs}",
                      unit="mb", leave=False)
        for s in mb_bar:
            mb = train_idx[s:s + args.batch_size]
            v = critic(gstate_t[mb].to(device), gscalar_t[mb].to(device))
            loss = ((v - targets_t[mb].to(device)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            opt.step()
            tot += loss.item() * len(mb)
            mb_bar.set_postfix(mse=f"{loss.item():.1f}")
        mb_bar.close()
        train_mse = tot / len(train_idx)
        val_mse, val_ev = evaluate(val_idx)
        epochs.set_postfix(train_mse=f"{train_mse:.1f}",
                           val_mse=f"{val_mse:.1f}", ev=f"{val_ev:+.3f}")
        tqdm.write(f"epoch {epoch:3d}  train_mse {train_mse:9.2f}  "
                   f"val_mse {val_mse:9.2f}  val_explained_var {val_ev:+.3f}")

    out_path = os.path.join(here, args.out)
    torch.save({
        "cfg": critic.cfg,
        "state_dict": {k: v.cpu() for k, v in critic.state_dict().items()},
        "meta": {"gamma": args.gamma, "reward_scale": args.reward_scale,
                 "data": args.data, "transitions": int(n)},
    }, out_path)
    print(f"saved pretrained critic -> {out_path}")


if __name__ == "__main__":
    main()
