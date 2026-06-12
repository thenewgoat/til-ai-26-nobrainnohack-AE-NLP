# AE submission cheat sheet

All flips happen via `Dockerfile` build args. `NeuralAEManager` always loads
`policy.onnx` from `/workspace`, so for neural builds you copy the chosen
ONNX onto that name before building.

Build args:
- `AE_MODE` — `scripted` (default) or `neural`.
- `AE_STRATEGY` — name from `ae/src/scripted/strategies.STRATEGIES`, only
  honoured when `AE_MODE=scripted`. Per the
  [strategy-arg memory](/home/jupyter/.claude/projects/-home-jupyter-til-ai-26/memory/til-docker-strategy-arg.md)
  always set this explicitly — don't trust the Dockerfile default.

Per the
[leaderboard-retains-highest memory](/home/jupyter/.claude/projects/-home-jupyter-til-ai-26/memory/til-leaderboard-retains-highest.md),
every submission needs a unique tag; you can't re-submit an old tag to
"revert".

All commands run from `ae/`:

```bash
cd /home/jupyter/til-ai-26/ae
```

## A. Scripted — strongest right now (BC 158 / smoke 135 / scripted ~245 in head-to-head)

```bash
docker build \
    --build-arg AE_MODE=scripted \
    --build-arg AE_STRATEGY=balanced_extreme_opening \
    -t til-ae:scripted-bxo-$(date +%Y%m%d-%H%M) .
```

Other scripted strategies you can pass to `AE_STRATEGY`:
`balanced`, `balanced_extreme`, `balanced_extreme_opening`,
`base_rusher`, `base_rusher_extreme`, `collector`, `camper`, `forager`,
`lean_rush`, `defender`, `adaptive`, plus the standalone `greedy`.

## B. Neural — BC clone (`policy_transformer_bc.pt`)

```bash
cp src/policy_bc.onnx src/policy.onnx
docker build \
    --build-arg AE_MODE=neural \
    -t til-ae:neural-bc-$(date +%Y%m%d-%H%M) .
```

## C. Neural — smoke-trained model (`policy_final.pt`, weaker than BC)

```bash
cp src/policy_smoke.onnx src/policy.onnx
docker build \
    --build-arg AE_MODE=neural \
    -t til-ae:neural-smoke-$(date +%Y%m%d-%H%M) .
```

## D. Smoke the built image locally before submitting

```bash
docker run --rm -p 5005:5005 til-ae:<tag>
# In another shell — health-check that uvicorn is up:
curl -s http://localhost:5005/openapi.json | head -1
# Stop with Ctrl-C in the docker run terminal.
```

A full `/ae` round-trip needs a real observation payload (matches the
`{"instances": [{"observation": {...}}]}` shape the contest sends).
`ae/tests/` has fixtures you can adapt.

## E. Refresh the ONNX exports after a new training run

The two ONNX files in `ae/src/` were exported from the matching `.pt`
checkpoints. After training a new model, re-export with:

```bash
cd /home/jupyter/til-ai-26/ae/src/training
.venv/bin/python -c "
import sys, os; sys.path.insert(0, os.path.abspath('..'))
from policy import SymbolicTransformerActor
from export_onnx import export_actor
actor = SymbolicTransformerActor.from_checkpoint(
    '/home/jupyter/til-ai-26/ae/src/policy_<NAME>.pt')
export_actor(actor, '/home/jupyter/til-ai-26/ae/src/policy_<NAME>.onnx')
"
```

Then `cp src/policy_<NAME>.onnx src/policy.onnx` before building.

## Status of artifacts in the repo right now

| file | meaning | head-to-head score vs 5× `balanced_extreme_opening` (seed=0) |
|---|---|---|
| `ae/src/policy_transformer_bc.pt` / `policy_bc.onnx` | BC clone of the scripted experts | 158, 4th of 6 |
| `ae/src/policy_final.pt` / `policy_smoke.onnx` | 20-update PPO smoke run from BC init | 135, 5th of 6 |
| (Dockerfile default `AE_STRATEGY=balanced_extreme_opening`) | scripted | ~245 median, opponents themselves |

The scripted path is the strongest of the three until you do a longer
training run that beats BC.
