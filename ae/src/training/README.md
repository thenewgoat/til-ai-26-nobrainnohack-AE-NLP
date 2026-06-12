# AE PPO training

PPO baseline for the AE (Bomberman) challenge. Trains `agent_0` against five random opponents — see `docs/superpowers/specs/2026-05-16-ae-ppo-baseline-design.md`.

## Setup

```bash
cd ae/training
uv sync
uv add --editable ../../til-26-ae   # first time only
```

## Train

```bash
uv run python train_ppo.py                       # full run (2M steps)
uv run python train_ppo.py --total-timesteps 800 --episodes-per-update 2  # smoke
```

Weights are written to `ae/src/policy.pt`, baked into the container by `COPY src .` in `ae/Dockerfile`.

## Monitor

```bash
uv run tensorboard --logdir runs
```

## Test

```bash
uv run pytest
```
