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

> **Full module-by-module + behavior reference:** see the
> [Scripted agent reference](#appendix--scripted-agent-reference) appendix at
> the end of this document.

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

---

# Appendix — Scripted agent reference

A thorough reference for `ae/src/scripted/` (16 modules, ~3,000 LOC) and the
JSON priors in `ae/src/`. This is the deployed qualifier agent and the teacher
pool the neural pipeline clones from. It is heavily test-covered
(`ae/tests/test_scripted_*`).

## 1. The model — Strategy = layers + params + gates

The agent is **not** a monolith. A behaviour is a `Strategy`
(`scripted/strategies.py`):

```python
@dataclass(frozen=True)
class Strategy:
    name:   str
    layers: tuple          # ordered cascade of layer functions
    params: StrategyParams # the tuning-knob bag (frozen dataclass)
    gates:  tuple = ()     # post-decision overrides, run in order
```

- A **layer** is a pure function `f(belief, danger, planner, params) -> int | None`.
  It proposes an action or abstains (`None`).
- A **gate** is `g(belief, danger, planner, params, action) -> int | None`. It
  runs *after* the cascade and may override the chosen action or pass through.
- **`StrategyParams`** is a single frozen dataclass of ~30 knobs (thresholds,
  weights, radii). Strategies differ by *which layers in what order* and *which
  knobs*, not by bespoke code. This is what lets one codebase serve the
  qualifier agent **and** supply a roster of BC teachers / self-play opponents.

## 2. The per-step decision flow (`scripted/decide.py::act`)

```
observation ──► Belief.update(obs)        # fold viewcones, age bombs, detect stuck
                     │
   frozen_ticks>0 ──►└─► STAY (short-circuit; no planning while frozen)
                     │
                     ├─► DangerMap(enemy_bombs, belief)      # LOS-gated blast imminence
                     ├─► build_planner(belief, danger)       # time-aware Dijkstra
                     │
                     ├─► for layer in strategy.layers:       # CASCADE
                     │       a = layer(belief, danger, planner, params)
                     │       if a is legal: chosen = a; break # first legal wins
                     │   else: chosen = first-legal fallback
                     │
                     ├─► belief.last_layer = source          # published BEFORE gates
                     ├─► for gate in strategy.gates:          # OVERRIDES
                     │       o = gate(..., chosen)
                     │       if o is legal: chosen = o
                     │
                     └─► record_final_action(...)             # bookkeeping, return int
```

`record_final_action` sets `belief.expected_location` (next-tick stuck
detection), logs a `PLACE_BOMB` to `belief.own_bombs`, and records
`belief.last_layer` (read by the replay visualizer; never affects behaviour).
The hybrid controller bypasses the cascade runner but calls the same
`record_final_action`, so state stays consistent.

## 3. Module reference

| Module | LOC | Role / key API |
| --- | --- | --- |
| `decide.py` | 93 | Cascade runner. `act(belief, mask, strategy)`, `record_final_action`. |
| `strategies.py` | 155 | `Strategy`, `StrategyParams`, the `STRATEGIES` registry. |
| `layers.py` | 878 | Behaviour layers (§5) + their private helpers. |
| `adaptive_layers.py` | 372 | Artifact-driven layers: `forage_loop`, `rush_roi`, `defend_intercept`, `trap`. |
| `gates.py` | 229 | Post-decision gates (§6). |
| `belief.py` | 399 | `Belief` — per-episode world model (§7) + `_trace_*` debug facility. |
| `map_prior.py` | 69 | `MapPrior.load()`; `identify_team(base_location)`; static walls/bases/spawns/collectibles. |
| `danger.py` | 59 | `DangerMap` — per-tile soonest-detonation / overlap / lethal-at-tick. |
| `blast.py` | 188 | Faithful `til_environment` blast geometry: `bomb_reaches`, `walls_destroyed_by`, `replay_blasts`, `breach_bombs_needed`. |
| `pathfind.py` | 274 | `build_planner` → `Planner` over `(tile, facing, phase)` states (§8). |
| `geometry.py` | 37 | Numpy-free constants: action/direction ints, `MOVE`, `chebyshev`, `view_to_world`. |
| `greedy.py` | 139 | Standalone `greedy` agent (A* to nearest live base); independent of the cascade. |
| `escape.py` | 65 | Forced-escape floor for the hybrid controller (`must_force_escape`, `escape_selector`). |
| `handover.py` | 20 | `HandoverTrigger` — latching scripted→RL handover predicate (hybrid only). |
| `variants.py` | 54 | `build_training_anchor_pool` — finetuned strategy variants for self-play diversity. |
| `__init__.py` | 0 | package marker. |

## 4. The layer cascade

Layers are tried in order; the **first one that returns a legal action wins**.
Lower layers are fallbacks. All live in `layers.py` unless marked *(adaptive)*.

| Layer | Fires when… | Behaviour |
| --- | --- | --- |
| `survive` | a known bomb's blast can reach the agent's cell in time | Tier 1: route to a fully-safe cell reachable before detonation (biased toward open cells via `openness_*`); may drop a surplus bomb if escape still completes, or take a strike window first (`_strike_caveat`). Tier 2: least-bad reachable cell (fewest overlaps, latest detonation, nearest). Never returns STAY. |
| `hunt` | a live enemy is inside blast range and `team_bombs ≥ hunt_bomb_floor` (floor ignored once all bases dead) | Bomb a visible live enemy (one bomb per enemy, friendly-fire safe). |
| `strike` | `team_bombs > 0`, fewer than `strike_dead_bases_cap` bases dead, and a live non-doomed base is targetable | Pick the best base (`bombs_needed + target_travel_weight·arrival`), compute hit-tiles allowing up to `los_breach_max` wall-openers; bomb underfoot, breach a wall, or navigate to a hit-tile (prefers walking to a direct tile within `direct_walk_max` over dumping openers). |
| `forage_chain` | endgame (or `forage_requires_endgame=False`) | Rate-maximizing collector: BFS over `(cell, facing)` (turns cost ticks), greedily chains collectibles while each candidate beats the running average rate. |
| `forage_loop` *(adaptive)* | all enemy bases destroyed | Patrols precomputed `forage_loops.json` waypoints; picks the best `yield_endgame` loop; switches on falling realised yield (`loop_commit_ticks`, `switch_factor`). |
| `sweep` | a reachable collectible exists | Head to the best-value reachable collectible. Score = value × centre/visibility/enemy factors + drift toward enemy base (`sweep_base_gradient`); respects `camp_leash` / phase-B radius. |
| `rush_roi` *(adaptive)* | a live base clears the ROI gate | ROI-gated base attacker: scores bases (arrival + bomb-regen cost vs +50), sticky within `roi_gate_margin`. |
| `defend_intercept` *(adaptive)* | a live enemy is within `defend_radius` of our base | Intercept/bomb the attacker if arrival beats the base's time-to-destruction and the ROI swing beats foraging. |
| `trap` *(adaptive)* | `(our_cell, visible_enemy)` ∈ `killboxes.json` | Drop a deterministic guaranteed-kill bomb (offline-proved the enemy can't escape the fuse). Self-disables if `trap_enabled=False`. |
| `camp` | (camper) | Bomb enemies inside `camp_leash` of our base; return home if outside; else abstain so forage/sweep collect. |
| `default` | always (terminal) | Advance toward the chosen target base, or abstain if none. |
| `hold` | always (terminal, camper/defender) | Stay on station. |
| `forage` | — | Two-move-lookahead collector. **Defined but not wired into any current strategy** (superseded by `forage_chain`/`forage_loop`); kept for reference. |

## 5. The gates (`gates.py`)

Run after the cascade; first legal override wins. Only three are wired into
strategies today:

| Gate | Wired into | Effect |
| --- | --- | --- |
| `scripted_opening` | `balanced`, `balanced_extreme_opening` | Per-slot opening book: walk a fixed waypoint list / drop seed bombs in the deterministic opening, with a latched abort (danger underfoot, body-block stall, unreachable waypoint, no bombs, timeout). |
| `body_block_resolve` | most balanced strategies | Stuck-agent escape: when the agent expected to move but didn't (`stuck_ticks ≥ stuck_trigger_ticks`), blacklist the tile (`stuck_blacklist_ttl`) and route around it. |
| `strike_gate` | balanced family | After `strike` yields on the dead-bases cap, still drop a zero-detour bomb on an alive base in passing; guards against stale post-strike bomb drops. |
| `force_turn0_bomb` | *(none)* | Unconditional step-0 seed breach. Available, not currently wired. |
| `sweep_while_own_bomb` | *(none)* | Keep sweeping while own bomb is in flight. Available, not currently wired. |

## 6. Registered strategies (`STRATEGIES`)

15 strategies. `balanced_extreme_opening` is the marked default for the served
agent (`# USE THIS`); `balanced` is the `decide.act` fallback when no strategy
is passed.

| Strategy | Layer cascade | Non-default params | Gates |
| --- | --- | --- | --- |
| **`balanced_extreme_opening`** ★ | hunt, strike, survive, forage_chain, sweep, default | `centre_value_weight=0`, `enemy_avoid_factor=0.75` | opening, body_block, strike_gate |
| `balanced` | survive, hunt, strike, forage_chain, sweep, default | — | opening, body_block, strike_gate |
| `balanced_extreme` | hunt, strike, survive, forage_loop, sweep, default | — | body_block, strike_gate |
| `balanced_opening` | survive, hunt, strike, forage_chain, sweep, default | — | *(gates disabled)* |
| `base_rusher` | survive, strike, default | — | — |
| `base_rusher_extreme` | strike, survive, default | — | — |
| `collector` | survive, forage_chain, sweep, default | `forage_requires_endgame=False` | — |
| `camper` | survive, camp, forage_chain, sweep, hold | `camp_leash=4`, `forage_requires_endgame=False` | — |
| `forager` | survive, forage_loop, sweep, default | `centre_value_weight=0`, `enemy_avoid_factor=0.75`, `unseen_value_factor=0.5` | — |
| `lean_rush` | survive, hunt, rush_roi, sweep, default | — | — |
| `defender` | survive, defend_intercept, forage_loop, sweep, hold | — | — |
| `adaptive` | survive, rush_roi, trap, forage_loop, sweep, default | — | — |
| `glass_cannon` | strike, default | — | — |
| `pacifist` | survive, forage_chain, sweep, hold | `forage_requires_endgame=False` | — |
| `hunter_killer` | hunt, survive, default | — | — |

The layer **order** encodes priority. `balanced` leads with `survive`
(defensive); the `*_extreme*` variants lead with `hunt`/`strike` (offence
before self-preservation). `variants.py` derives further finetuned variants
(e.g. `camper_tight/loose`, `defender_*_perimeter`) for opponent diversity in
self-play without mutating `STRATEGIES`.

## 7. World model — `Belief`

One mutable instance, reused across episodes; reset when `update()` sees a new
episode. Tracks:

- **Walls**: monotonic discovery; `is_wall(a,b)` / `is_destructible(a,b)`,
  net of `destroyed_walls`.
- **Collectibles**: `collected` set, `remaining_collectibles()` (prior minus
  collected), trailing `realised_yield(window)`.
- **Bombs**: `enemy_bombs` / `ally_bombs` / `own_bombs` (cell→timer), aged each
  tick; detonated ally bombs credit `destroyed_walls`.
- **Enemies & bases**: `enemies`, `frozen_enemies`, `live_enemies()`,
  `dead_bases`, `enemy_base_health`, `live_enemy_bases()`, `base_alive()`.
- **Agent state**: `location`, `facing`, `step`, `health`, `base_health`,
  `frozen_ticks`, `team_bombs`, `team_resources`.
- **Stuck detection**: `expected_location`, `stuck_ticks`, `stuck_blacklist`
  (tile→expiry; planner treats blacklisted tiles as high-cost).
- **Adaptive state**: `adaptive_state` dict — per-slot cascade memory
  (opening index, active forage loop, rush target, …).

**Debug facility (gated, off by default):** `belief.trace_decisions = True`
records every layer/gate decision via `_trace_decision`; `trace_observations`
and `trace_wall_destruction` similarly. All are no-ops unless explicitly
enabled, so they cost nothing in serving. `belief.last_layer` always carries
the name of the layer/gate that produced the current action (the replay
visualizer reads it).

## 8. Perception & planning

- **`MapPrior`** (`map_prior.py`) — the deterministic novice map from
  `arena_map.json`: `wall_between` (frozenset pair → destructible bool),
  `bases`/`spawns` per team, `collectibles` (cell → value), `resource_cells`.
  `identify_team(base_location)` runs at step 0 to set `our_base`/`enemy_bases`.
- **`DangerMap`** (`danger.py`) — for each known bomb, projects its LOS-gated
  blast footprint into per-tile `ticks_to_danger` / `overlap` /
  `is_lethal_at(tick)`.
- **`blast.py`** — vendored `til_environment` blast geometry (supercover line +
  wall occlusion): `bomb_reaches`, `walls_destroyed_by`, `replay_blasts` (sim a
  bomb sequence vs a base), `breach_bombs_needed`.
- **`Planner`** (`pathfind.py`) — time-aware Dijkstra over
  `(tile, facing, phase)`: indestructible walls always block, destructible walls
  block until their detonation tick, danger cells are forbidden at the dangerous
  ticks, `BACKWARD` costs 1.4×, blacklisted tiles +1000. Exposes `dist_to`,
  `first_action`, `route_to`, `has_safe_continuation`, `survival_score(danger)`.

## 9. Data artifacts (`ae/src/*.json`)

| File | Shape | Content |
| --- | --- | --- |
| `arena_map.json` | keys `grid_size, bases, spawns, walls, collectibles` | The fixed novice board (seed 19). Loaded by `MapPrior`. |
| `respawn_map.json` | 16×16 list-of-lists | Per-tile collectible respawn delay (steps), Perlin-seeded & deterministic. |
| `forage_loops.json` | keys `loops, teams` | Offline-computed endgame patrol loops (waypoints + `yield_attack`/`yield_endgame`); per-team ordering. Used by `forage_loop`. |
| `killboxes.json` | key `killboxes` | Offline-proved `(agent_tile, enemy_tile)` guaranteed-kill configs. Used by `trap`. |

`arena_map.json` and `respawn_map.json` are generated offline by
`ae/tools/dump_map.py`; `forage_loops.json` by `tools/build_forage_loops.py`;
`killboxes.json` by `tools/build_killboxes.py`.

## 10. Extending the agent

- **New layer:** write `f(belief, danger, planner, params) -> int | None` in
  `layers.py` (or `adaptive_layers.py` if it needs an artifact / adaptive
  state). Return `None` to abstain. Add a test under `ae/tests/test_scripted_*`.
- **New knob:** add a field to `StrategyParams` with a default that preserves
  current behaviour (the dataclass is frozen; defaults keep existing strategies
  unchanged).
- **New strategy:** add an entry to `STRATEGIES` = `(name, layer tuple, params,
  gates)`. No other code changes; it becomes selectable via `AE_STRATEGY` and
  available as a BC teacher / self-play opponent automatically.
- **New gate:** write `g(..., action) -> int | None` in `gates.py` and add it to
  a strategy's `gates` tuple.
