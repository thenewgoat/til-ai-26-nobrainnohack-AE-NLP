# AE Self-Play Training Workflow

End-to-end pipeline: scripted teacher → BC clone → critic pretrain → PPO self-play → ONNX export.

All commands are run from `/home/jupyter/til-ai-26/ae`. The training virtualenv lives at `src/training/.venv` and has all dependencies (torch, til_environment, onnxruntime, tqdm).

## One-time setup per shell

```bash
cd /home/jupyter/til-ai-26/ae
export PYTHONPATH="$PWD/src:$PWD/src/training:${PYTHONPATH:-}"
PY=./src/training/.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p src/training/logs
```

The training scripts auto-set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing torch — this lets the CUDA caching allocator grow / shrink segments and avoids fragmentation OOMs when the BC dataset is materialised on GPU (~3.5 GB after round 1, ~12 GB by round 3) and when slot-rotation changes the PPO batch size. To override (e.g. add `max_split_size_mb:128`), export it yourself before invoking — `setdefault` respects an existing value.

## Stage 1 — BC clone (~15-25 min on T4)

Trains the actor against the scripted teacher via DAgger (3 rounds, 400 episodes total). Output: `ae/src/policy_transformer_bc.pt`.

```bash
BC_TEACHER=balanced_extreme_opening BC_WORKERS=6 $PY -m run_bc_transformer \
  2>&1 | tee src/training/logs/bc_$STAMP.log
```

The teacher is configurable via env var; default is `balanced_extreme_opening`. Other strong choices: `adaptive`, `balanced_extreme`.

Model size knobs (also env vars — defaults match the production checkpoint):

| Env var | Default | What |
|---|---|---|
| `BC_TEACHER` | `balanced_extreme_opening` | scripted strategy used as labeler |
| `BC_WORKERS` | `1` | parallel rollout workers for DAgger collection |
| `BC_EPOCHS` | `10` | gradient epochs per DAgger round (lowered from 20 to cut late-epoch overfitting spikes) |
| `BC_LR` | `3e-4` | Adam learning rate for BC training (lowered from 1e-3 — typical transformer LR) |
| `TF_D_MODEL` | 64 | transformer width |
| `TF_N_LAYERS` | 4 | transformer depth |
| `TF_N_HEADS` | 4 | attention heads |
| `TF_FFN_DIM` | 4×d_model | FFN width |
| `TF_DROPOUT` | 0.1 | dropout |

With `BC_WORKERS=4` a 400-episode BC run drops from ~30 min to ~8 min on the
T4 workbench (CPU-bound env stepping parallelizes ~linearly until the GPU
training step becomes the bottleneck around 6-8 workers).

The run prints a banner with the teacher + config, then a tqdm bar per DAgger round (`DAgger β=1.0` for round 1, `DAgger β=0.5` for rounds 2-3), then per-epoch BC training progress. **If you don't see those progress bars within ~30s of the pygame warning, something is wrong — kill and retry.**

The script ends with `BC GATE PASS:` or `BC GATE FAIL:` showing the clone's score vs the teacher (gate tolerance ±5%).

Verify:

```bash
ls -la src/policy_transformer_bc.pt
```

## Stage 2a — Collect critic-pretraining data (~5-10 min)

Frozen BC actor vs scripted-anchor opponents. Default 200 episodes × 4 workers. Output: `ae/src/training/logs/critic_pretrain_data.pt`.

```bash
$PY -m collect_critic_data \
  --actor policy_transformer_bc.pt \
  --episodes 200 \
  --workers 6 \
  --out logs/critic_pretrain_data.pt \
  2>&1 | tee src/training/logs/collect_critic_$STAMP.log
```

Note `--actor` matches the file Stage 1 produced (`policy_transformer_bc.pt`, not `policy_bc.pt`).

## Stage 2b — Pretrain critic (~2-3 min)

30 epochs of MSE on reward-to-go at LR=3e-4 (the critic's natural learning rate). Output: `ae/src/training/critic_pretrained.pt`.

```bash
$PY -m pretrain_critic \
  --data logs/critic_pretrain_data.pt \
  --epochs 30 \
  --out critic_pretrained.pt \
  2>&1 | tee src/training/logs/pretrain_critic_$STAMP.log
```

Look for `val_explained_var` to be > 0.3 by the last epoch. If it's near zero, the BC actor is producing trajectories the critic can't fit — investigate before moving to PPO.

## Stage 3 — PPO self-play (hours)

1000 updates × 4 episodes per update is the spec default. Output: `ae/src/policy_final.pt` (overwritten each snapshot) plus `policy_rung*_u*.pt` checkpoints and `policy_best_rung*.pt`. Replays / metrics PNGs land in `ae/src/training/viz/`.

Critic uses LR=1e-4 (10× actor LR) via the split-optimizer change; bump to 3e-4 via the CLI flag if `explained_variance` (visible in the tqdm postfix as `ev=...`) stalls below 0.3.

```bash
$PY -m train_selfplay \
  --total-updates 1000 \
  --episodes-per-update 4 \
  --rollout-workers 4 \
  --bc-init policy_transformer_bc.pt \
  --critic-init critic_pretrained.pt \
  --seed 1 \
  2>&1 | tee src/training/logs/ppo_$STAMP.log
```

**Quick smoke first** (~10 min, verifies wiring before the long run):

```bash
$PY -m train_selfplay \
  --total-updates 20 \
  --episodes-per-update 2 \
  --rollout-workers 2 \
  --bc-init policy_transformer_bc.pt \
  --critic-init critic_pretrained.pt \
  --eval-every 10 --snapshot-every 10 --viz-every 10 \
  --seed 1
```

### What to watch in the tqdm bar / metrics

The training loop's `pbar` shows: `rung`, mean episode return (`ret`), policy loss, `ev` (explained variance — critic quality), and entropy coefficient.

Metrics CSV/PNG dumps in `src/training/viz/` track every PPO update:

| Metric | Healthy range | What it tells you |
|---|---|---|
| `mean_return` | gradually rising, sometimes noisy | true env-default return per learner episode |
| `explained_variance` | > 0.3 after 100 updates, ideally > 0.7 | critic fits well; below 0.3 → bump `--critic-learning-rate` |
| `value_loss` | spiky early, settles by ~100 updates | if persistently > 100, critic underfits |
| `policy_loss` | small (1e-3 to 1e-1) | very large means the clip is constantly biting |
| `entropy` | starts ~1.5, drifts down with ent_coef anneal | rising mid-training = exploration spike (bad sign) |
| `approx_kl` | < 0.02 between epochs | if > 0.05, lower actor LR or fewer update_epochs |
| `clipfrac` | < 0.3 | how often the ratio gets clipped — high = PPO clip is active |
| `eval_winrate` | rising rung-by-rung | the true performance signal |

### Opponent diversification (random curriculum + new edge strategies)

The PPO trainer now uses a **linear curriculum** on the random-opponent share: starts high (0.40) for robustness, tapers to low (0.15) as the league fills with frozen-checkpoint self-play opponents. The current value is visible in the tqdm postfix as `rnd=0.NN`.

| Knob | Default | What |
|---|---|---|
| `--opp-mix-random-start` | `0.40` | random share at update 1 |
| `--opp-mix-random-end` | `0.15` | random share at the final update |
| `--opp-mix-noisy-scripted` | `0.15` | static fraction across the run |
| `--opp-mix-raw-scripted` | `0.10` | static fraction across the run |

The league share absorbs the slack — by the end of training it's ~0.60 (mostly your own past checkpoints + scripted anchors).

Three new edge-case scripted opponents were added to the registry; the `League` picks them up automatically because anchors come from `STRATEGIES.keys()`:

| Strategy | Layers | Purpose |
|---|---|---|
| `glass_cannon` | `(strike, default)` | pure rush — no survive check. Tests punishing reckless aggression. |
| `pacifist` | `(survive, forage, sweep, hold)` | never attacks, just farms. Tests against passive grinders. |
| `hunter_killer` | `(hunt, survive, default)` | aggressive enemy-agent bombing. Tests against opponents that prioritize kills over base damage. |

### Knobs you can pass to Stage 3

These all map to `Args` fields in `train_selfplay.py` (tyro converts `slot_rotation_min` → `--slot-rotation-min` etc.):

| Flag | Default | What |
|---|---|---|
| `--learner-slots-mode` | `rotate` | `rotate` (random K∈[min,max] slots per ep) or `fixed` |
| `--slot-rotation-min` / `--slot-rotation-max` | `1` / `3` | learner-subset size range |
| `--opp-eps-noise` | `0.08` | ε on scripted opponents |
| `--opp-neural-temperature` | `1.2` | T on neural opponents (eval/BC use 0.001) |
| `--opp-mix-random-start` / `--opp-mix-random-end` | `0.40` / `0.15` | random-share curriculum endpoints (league absorbs the slack) |
| `--opp-mix-noisy-scripted` / `--opp-mix-raw-scripted` | `0.15` / `0.10` | static fractions; sum with random must be ≤ 1 |
| `--anti-idle-penalty` | `0.05` | STAY penalty (annealed to 0) |
| `--no-cuda` | (CUDA on) | force CPU |

Critic-LR knobs are inside `PPOConfig` — edit `train_selfplay.py::PPOConfig.critic_learning_rate` directly if you want to bump from 1e-4 to 3e-4.

## Stage 4 — Export to ONNX

`export_onnx.py::main()` is hardcoded to read `policy_transformer_bc.pt` → write `policy.onnx`. After PPO, point it at the trained policy:

```bash
$PY -c "
import os
from export_onnx import export_transformer
src = 'src'
export_transformer(os.path.join(src, 'policy_final.pt'),
                   os.path.join(src, 'policy.onnx'))
print('exported policy_final.pt -> policy.onnx')
"
```

(Or export the BC clone alone for a baseline:)

```bash
$PY -m export_onnx
```

## Re-running stages

Each stage's output is the next stage's input. To re-run a single stage, just delete its output and re-invoke. E.g., re-run only BC:

```bash
rm src/policy_transformer_bc.pt
BC_TEACHER=balanced_extreme_opening $PY -m run_bc_transformer ...
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No output after pygame warning, no progress bar | tqdm not flushing in your terminal — log redirect to file | Pipe through `tee` (as in the example commands) or pass `2>&1 | cat` |
| `BC gate FAIL: ... rel_gap > 0.05` | clone underfit | Increase `epochs` in `run_bc_transformer.py`'s `train_bc` calls from 20 to 30, or use a stronger teacher |
| PPO `explained_variance` stalls < 0.3 after 100 updates | critic undertrained | Bump `PPOConfig.critic_learning_rate` from 1e-4 to 3e-4 |
| PPO returns flat zero (idle collapse) | actor lost the BC behavior | Verify `--bc-init` path; check `anti_idle_penalty` is non-zero |
| `RuntimeError: shape '[N, 256, 85]' is invalid` | stale `policy_*.pt` from before the K=5 stack change | Re-run Stage 1 to regenerate the checkpoint |
| Worker rollouts use wrong opponent mix | older bug (now fixed) where worker `OpponentRegistry` ignored Args knobs | Pull latest `train_selfplay.py` — `registry_kwargs` is threaded through |
| `OSError: [Errno 24] Too many open files` during rollout-Pool startup | torch's default `file_descriptor` tensor-sharing opens one socket per tensor storage; high worker × episode counts exhaust fds | Fixed: `train_selfplay.py` / `collect_critic_data.py` now call `torch.multiprocessing.set_sharing_strategy("file_system")` at import. If you still hit it on a different entry point, add the same line, or `ulimit -n 65536` in your shell |

## Memory artifacts (per-run logs)

| Path | Contents |
|---|---|
| `src/training/logs/bc_$STAMP.log` | BC stdout (DAgger rounds + per-epoch loss) |
| `src/training/logs/collect_critic_$STAMP.log` | rollout collection stdout |
| `src/training/logs/pretrain_critic_$STAMP.log` | critic-pretrain per-epoch MSE + explained-var |
| `src/training/logs/ppo_$STAMP.log` | PPO update-by-update metrics |
| `src/training/viz/metrics_u*.png` | per-update metric panel (auto every `viz_every` updates) |
| `src/training/viz/replay_u*.mp4` | 200-step episode replay against the current opponents |
| `src/training/viz/leaderboard_u*.png` | league win-rate snapshot |

## Future extension — self-exploitation league (AlphaStar pattern)

If the trained actor scores well in eval but you suspect it has blind spots versus opponents not represented in the current league, the next step is **adversarial league diversification**:

1. At update ~500 (or after a desired-performance checkpoint), freeze the current actor as `policy_exploit_target.pt`.
2. Spin up a SEPARATE PPO run with reward inverted (`-opponent_reward`) against the frozen target. Train for ~100 updates. This produces an "exploiter" that specifically counters the main agent's weaknesses.
3. Add the exploiter to the main league pool. Resume main training; the actor must now beat the exploiter, forcing it to patch the blind spot.

This is the AlphaStar / OpenAI Five recipe. Cost: ~3-4 hours additional engineering + the exploiter PPO run time. Worth doing only if you have a known-strong baseline actor first — adversarial league diversification has nothing to bite on if the main actor is still BC-equivalent. Defer until after a successful end-to-end training run.

Implementation sketch (not yet built): would need a new `train_exploiter.py` that loads the frozen target as a fixed opponent in every rollout slot, runs PPO with negated rewards, and saves to a versioned checkpoint added to the league via `League.snapshot`.
