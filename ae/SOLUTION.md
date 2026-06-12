# AE — Solution Description

## What this folder is

`ae/` holds the complete entry for the **AE (Autonomous Entity / Bomberman)**
challenge of TIL-26. AE is a 6-agent grid game: each agent navigates a 16×16
maze, collects items, places bombs, and attacks enemy team bases. The graded
interface is an HTTP server — TIL POSTs an observation to `/ae` on port 5005
and expects an action integer (0–5) back.

The folder contains two deliverables that share one codebase:

- **`src/`** — the deployed inference container (server + agents + weights).
- **`training/`** — the offline training pipeline that produces the neural
  agent. Never shipped.
- **`tools/`** — one-off offline utilities (e.g. `dump_map.py`).

There are **two interchangeable agents**, selected at container build/run time:

| Agent | `AE_MODE` | What it is |
| --- | --- | --- |
| **Scripted** (default) | `scripted` | Deterministic rule cascade. No neural net. The qualifier submission. |
| **Neural** | `neural` | Transformer policy, BC-cloned then PPO-trained, served via ONNX. The finals robustness play. |

---

## The central exploit — the map is deterministic

The eval runs the env with `cfg.env.novice=True`, which **hard-seeds** the
maze (seed 19) and the episode scatter (seed 88). Verified by dumping the env
three times: the maze layout, all agent/base spawns, and all 218 collectibles
are **fully deterministic**. The finals use the *same map* as the qualifier —
only the opponents change.

**Consequence:** the agent never has to *learn navigation*. The map is
precomputed offline and shipped as static priors:

- `src/arena_map.json` — wall layout + base/spawn positions.
- `src/respawn_map.json` — per-tile collectible respawn delay (10–40 steps;
  Perlin-seeded, deterministic). A collected tile refills at the *same* tile
  after `respawn_map[x,y]` steps.

This single fact drives the whole design: the scripted agent is viable at all,
and the neural agent gets a symbolic feature set instead of raw pixels.

---

## Agent 1 — the scripted agent (`src/scripted/`)

A deterministic rule cascade exploiting the fixed map. The container for this
mode is CPU-only and **drops torch entirely**.

### Pipeline (per step)

1. **`MapPrior`** — loads the static map; `identify_team()` at step 0 figures
   out which team this agent is on from `base_location`.
2. **`Belief`** — per-episode world model: tracks agent/enemy/bomb positions,
   monotonic wall discovery, collected tiles, own bombs in flight.
3. **`DangerMap`** — projects every known bomb's blast footprint into a
   per-tile "imminence" field (line-of-sight occluded).
4. **`pathfind`** — bomb-aware A* / Dijkstra planner that routes around danger.
5. **`decide.act()`** — runs the strategy's **layer cascade** and returns the
   first legal action a layer proposes.

### Key decision — strategy-pluggable architecture

The agent was refactored so a **Strategy = (ordered layer tuple +
`StrategyParams`)**. `layers.py` holds every behavior layer (`survive`,
`hunt`, `strike`, `forage`, `sweep`, `default`, `camp`, `hold`, …);
`strategies.py` composes them into named strategies; `decide.py` is just the
cascade runner. Registered strategies: `balanced` (default / qualifier),
`balanced_extreme`, `base_rusher`, `base_rusher_extreme`, `collector`,
`camper`.

This refactor pays double duty: it serves the scripted agent **and** supplies
the neural pipeline with a roster of BC teachers and self-play opponents.

The served strategy is chosen per-image via the `AE_STRATEGY` build arg
(default `balanced`).

---

## Agent 2 — the neural agent (training pipeline)

The neural track exists for **finals robustness**: the scripted agent is
strong but rigid against novel opponents. The plan ("spec C") trains a network
to imitate then surpass the scripted roster.

### Key decision — symbolic features, not raw pixels

Because the map is known, the network does not consume raw viewcones alone.
`training/features.py`'s **`FeatureBuilder`** renders the scripted modules'
world model into a **frozen five-tensor contract**:

| Tensor | Shape | Content |
| --- | --- | --- |
| `grid` | `[17,16,16]` | abstraction tile grid (walls, danger, pathing, …) |
| `base_feats` | `[5,11]` | per-enemy-base abstraction matrix |
| `raw_agent` | `[7,5,25]` | raw agent viewcone, kept whole (residual branch) |
| `raw_base` | `[7,7,25]` | raw base viewcone, kept whole (residual branch) |
| `scalar` | `[10]` | scalar token (step, health, resources, …) |

The builder is **stateful** — it owns a per-slot `Belief`, reset when it sees
`step == 0`. It is identical in training and serving. Layouts are frozen here;
`policy.py` mirrors them and `test_*_contract.py` asserts they agree.

### Key decision — one transformer, two branches

`src/policy.py`'s **`SymbolicTransformerActor`** turns the five tensors into a
347-token sequence (`[CLS] + 256 tiles + 5 bases + 84 raw + 1 scalar`), each
group with a learned type embedding. **Attention is hand-rolled** (explicit
matmul + softmax) so ONNX export uses only standard ops. The architecture
lives in `src/policy.py` — shared by training and serving so it *cannot drift*.

### Training stages

```
1. BC (behavior cloning)   — clone the actor from scripted-teacher actions
2. Critic pretraining      — fit the value function on empirical returns
3. PPO self-play ladder    — improve beyond the teachers
4. ONNX export             — ship the actor
```

#### Stage 1 — BC with DAgger (`training/bc.py`)

DAgger-style dataset aggregation: roll out a mix of a teacher strategy and the
in-training actor (`beta = P[teacher acts]`); **every** visited state is
labeled with the teacher's action regardless of who acted, widening coverage
beyond the teacher's near-deterministic trajectory. Slots and opponents are
varied so the parameter-shared actor sees all 6 positions.
`collect_family_winner.py` collects only *winning* balanced-family trajectories
as a higher-quality BC source.

#### Stage 2 — critic pretraining (`collect_critic_data.py` + `pretrain_critic.py`)

**Why:** PPO advantage `= return − V(s)`. A *random* critic paired with a
competent BC actor produces garbage advantages from update 1 — a likely
contributor to the observed spiky `value_loss` and policy collapse.

**Key decision — the critic is a separate trunk, pretrained separately.**
`training/critic.py`'s `CentralizedCritic` is its own transformer with its own
input (`encode_global_state` — a 9-plane global state from the entity
registry, **not** the FeatureBuilder five-tensor). So:

- It cannot share a value head bolted onto the BC model.
- There is no shared-trunk corruption risk; no gradient detaching needed.
- It is pretrained as a standalone supervised job.

The recipe: roll out the frozen BC actor vs the scripted panel, compute
Monte-Carlo **reward-to-go**, and MSE-fit the critic. Targets use the **raw
env-reward scale and `gamma = 0.99`** — identical to PPO's GAE — so there is
**no recalibration mismatch** at PPO start (no target normalization). Track
`val_explained_var`; do not start PPO actor updates until it is comfortably
positive. `train_selfplay.py` warm-starts the critic from
`critic_pretrained.pt` (kept in `ae/training/`, never shipped).

#### Stage 3 — PPO self-play ladder (`training/train_selfplay.py`)

CleanRL-style single-file PPO (chosen over PufferLib — uninstalled, AEC-env
integration too risky for a hackathon). Decisions:

- **CTDE / MAPPO** — decentralized actor, centralized critic.
- **Multi-slot collection** — transitions collected from *every* learner slot
  each step (~3–6× data gain). Buffer is laid out *slot-major, run-contiguous*
  because GAE assumes contiguous episode runs.
- **3-rung ladder** (`RungLadder`): rung 1 = the 6 scripted strategy anchors;
  rung 2 = PFSP-sampled frozen checkpoints; rung 3 = live self-play.
- **League + PFSP** (`league.py`) — opponent pool with `(1−winrate)²` "hard"
  sampling weights; 6 permanent scripted anchors + capped checkpoint pool.
- **Anti-idle reward shaping** (`AntiIdleShaper`) — annealed STAY penalty, a
  backstop against the original PPO's flat-0 idle collapse. Training-only; eval
  always scores the unshaped env reward.
- Centralized critic is **training-only** — never exported, zero serving cost.

#### Stage 4 — ONNX export (`training/export_onnx.py`)

The trained actor is exported to `src/policy.onnx`. The container ships
`onnxruntime` instead of torch.

### Observability (`training/visualize.py`, `metrics.py`)

`MetricsLogger` writes CSV + TensorBoard + multi-panel PNGs + a league
leaderboard. `visualize.py` renders labeled episode-replay MP4s. `train_selfplay`
has a `viz_every`-gated hook, try/except-isolated so a viz failure cannot kill
a training run. Output → `ae/training/viz/` (gitignored).

---

## Serving (`src/ae_server.py`, `src/ae_manager.py`)

- FastAPI app; `POST /ae` and `GET /health`.
- `AE_MODE` env var picks the manager: `AEManager` (scripted) or
  `NeuralAEManager` (ONNX).
- **No `/reset`** — the eval never calls it. Both managers detect a new round
  internally on `step == 0` and reset their belief/feature state.
- `NeuralAEManager` builds features, runs the ONNX actor, masks illegal
  actions with `-1e8`, and argmaxes.
- The raw request body is logged on parse failure for debuggability.

### Container (`Dockerfile`, `requirements.txt`)

- Base `python:3.11-slim` — CPU-only; AE does not need a GPU at serve time.
- `COPY src .` plus `COPY training/features.py .` — the neural path needs
  `features.py`, which lives in `training/`.
- Build args: `AE_MODE` (`scripted`/`neural`), `AE_STRATEGY` (which scripted
  strategy). Both also overridable per `docker run` via `-e`.
- Dependencies: `fastapi`, `uvicorn`, `numpy`, `onnxruntime`. **No torch.**

---

## Commands

All training commands run from `ae/training/` with the `uv`-managed env.

### Setup (first time)

```bash
cd ae/training
uv sync
uv add --editable ../../til-26-ae   # the til_environment package
```

### Train the neural agent (full pipeline)

```bash
# 1. BC — collect winning balanced-family trajectories, then clone the actor
uv run python collect_family_winner.py --episodes 400   # -> logs/family_winner.pt
uv run python run_bc_family_winner.py                   # -> ../src/policy_family_winner_bc.pt

# 2. Critic pretraining — frozen-actor rollouts -> reward-to-go MSE fit
uv run python collect_critic_data.py --episodes 200     # -> logs/critic_pretrain_data.pt
uv run python pretrain_critic.py --epochs 30            # -> critic_pretrained.pt
#   watch val_explained_var; it must be comfortably positive before PPO

# 3. PPO self-play ladder (hours on a T4; auto-loads critic_pretrained.pt)
uv run python train_selfplay.py                         # -> ../src/policy_final.pt + rung checkpoints

# 4. Export the trained actor to ONNX for serving
#    NOTE: export_onnx.py is hardcoded to ../src/policy_transformer_bc.pt -> ../src/policy.onnx
#    rename/copy the chosen checkpoint to that path first
uv run python export_onnx.py                            # -> ../src/policy.onnx
```

Useful flags: `collect_critic_data.py --workers N` (CPU rollout processes — raise
to parallelise), `pretrain_critic.py --gamma` (must match `train_selfplay`
`Args.gamma`), `train_selfplay.py --critic-init ""` (disable the pretrained
critic and use a random one).

### Profile a rollout (find the collection bottleneck)

```bash
uv run python -m cProfile -s cumtime collect_critic_data.py --episodes 1 --workers 1 | head -40
```

### Tests

```bash
uv run pytest                       # full suite (~250 tests)
uv run pytest tests/test_critic.py  # a single file
```

### Monitor training

```bash
uv run tensorboard --logdir viz     # plus PNG replays + leaderboard under ae/training/viz/
```

### Build & run the container

```bash
cd ae
# scripted agent (default mode is set in the Dockerfile's AE_MODE ARG)
docker build --build-arg AE_MODE=scripted --build-arg AE_STRATEGY=balanced -t ae .
# neural agent (serves src/policy.onnx)
docker build --build-arg AE_MODE=neural -t ae .
docker run -p 5005:5005 ae          # POST observations to http://localhost:5005/ae
```

---

## Environment facts (verified)

- 16×16 grid, 6 AEC agents, 200-step episodes, 6 actions
  (FORWARD/BACKWARD/LEFT/RIGHT/STAY/PLACE_BOMB), 25 viewcone channels.
- Agents **freeze and respawn** on death — they do not leave the board. Only
  **base destruction** ends an episode (and triggers the team loss).
- A destroyed base does **not** shrink its viewcone: the env still emits a full
  `[7,7,25]` "ghost" `base_viewcone` centred on the dead base
  (`base_health == 0`). `features.py._build_raw` zeroes `raw_base` in that case.
- Collectible respawn is fully precomputable (`respawn_map`, deterministic
  given seed 19).

---

## Conventions

- **Package manager: `uv`.** Training env: `cd ae/training && uv sync`.
- **Train on T4 GPU, serve on CPU.**
- **TDD throughout** — `ae/training/tests/` holds ~250 tests covering scripted
  modules, features, policy contracts, BC, PPO, league, metrics, ONNX parity.
- Frozen contracts (feature layouts, token counts) are asserted by
  `test_feature_contract.py` / `test_policy_contract.py` so training and
  serving can never silently diverge.

---

## Current artifacts (`src/`)

- `policy_family_winner_bc.pt` — BC actor cloned from winning balanced-family
  trajectories (the critic-pretraining and PPO starting point).
- `policy_rung*_u*.pt`, `policy_best_rung1.pt`, `policy_final.pt` — PPO
  self-play checkpoints across the ladder.
- `policy_bc.onnx` — exported actor for the neural serving path.
- `arena_map.json`, `respawn_map.json` — static deterministic map priors.

---

## Status summary

- **Scripted agent: complete and serving.** `balanced` is the qualifier
  submission; e2e episode score +636 vs −72 random.
- **Neural pipeline: code complete and tested.** BC → critic-pretrain → PPO →
  ONNX stages all exist. The original PPO (random opponents, no idle penalty,
  random critic) collapsed to a flat-0 idle policy; the BC warm-start,
  anti-idle shaper, and critic pretraining are the three defenses against that.
- The finals agent is produced by a real training run on the T4.
