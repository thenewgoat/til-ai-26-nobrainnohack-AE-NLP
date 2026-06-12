# AE Training Pipeline — Architecture & Component Reference

> Complementary to [WORKFLOW.md](./WORKFLOW.md). This doc explains **what** each
> component does and **why**; WORKFLOW.md explains **how** to run them.
>
> Audience: an engineer who is about to modify the trainer or diagnose a
> training run that went wrong. Reading order is top-to-bottom; cross-links
> point at the implementing source line.

## 0. High-level data flow

```
                  +--------------------------------+
                  | scripted.strategies.STRATEGIES |  (15 named cascades)
                  +---------------+----------------+
                                  |
                                  | balanced_extreme_opening
                                  v
+----------------------+    +--------------+    +---------------------+
| bc.collect_dagger_   |--->| bc.train_bc  |--->| policy_transformer_ |
| dataset (3 rounds)   |    | (CE + mask)  |    | bc.pt   (actor θ)   |
+----------------------+    +--------------+    +----------+----------+
                                                            |
                                                            v
                                          +-----------------+-----------------+
                                          | collect_critic_data:              |
                                          | frozen actor vs scripted anchors  |
                                          | -> (gstate, gscalar, env_rewards) |
                                          +-----------------+-----------------+
                                                            |
                                                            v
                                          +-----------------+-----------------+
                                          | pretrain_critic: MSE on RTG       |
                                          | -> critic_pretrained.pt           |
                                          +-----------------+-----------------+
                                                            |
                  +-----------------------------------------+
                  |
                  v
  +------------------------------+        +----------------------+
  | train_selfplay (PPO)         |<------>| League (anchors +    |
  |  - CTDE centralized critic   |        | checkpoint pool)     |
  |  - slot rotation             |        +----------+-----------+
  |  - kind-mix curriculum       |                   ^
  |  - 3-rung promotion          |                   | snapshot every N
  +-------------+----------------+                   |
                |                                    |
                | every snapshot_every updates       |
                +------------------------------------+
                |
                v
  +-------------------------+      +--------------------------+
  | policy_final.pt /       |----->| export_onnx -> policy.   |
  | policy_best_rung*.pt    |      | onnx (served container)  |
  +-------------------------+      +--------------------------+
```

The orchestration loop in `train_selfplay.main` ties this together: rollouts
collect transitions, GAE produces advantages from a CTDE critic, PPO updates
both networks, the league absorbs checkpoints, and an eval gate moves the
agent through the rung ladder.

## 1. Observation contract

The actor consumes a five-tensor obs built by `features.FeatureBuilder.build`.
The grid and scalar tensors are **K=5 frame-stacked** (newest-first along the
channel axis) so the policy sees short-horizon dynamics; the symbolic per-base
matrix and the raw viewcones are single-frame.

| Tensor       | Shape              | Source                                                                  | Stacked? |
| ------------ | ------------------ | ----------------------------------------------------------------------- | -------- |
| `grid`       | `[N, 85, 16, 16]`  | `FeatureBuilder._build_abstraction_grid` (Belief + planner + danger map)| K=5      |
| `base_feats` | `[N, 5, 11]`       | `scripted.layers._effective_hp` / `_target_base` (+ planner arrival)    | no       |
| `raw_agent`  | `[N, 7, 5, 25]`    | env's agent viewcone (kept whole)                                       | no       |
| `raw_base`   | `[N, 7, 7, 25]`    | env's base viewcone (kept whole)                                        | no       |
| `scalar`     | `[N, 50]`          | env scalar fields (step / health / bombs / resources / ...)             | K=5      |

The stack order is newest-first: `_grid_history.append(grid)` then
`reversed(self._grid_history)`. Zero-pads the past for the first 4 steps of an
episode. The frozen channel-index contract lives in `features.py` (CH_WALL_R
... CH_BASE_MARK, then SC_STEP ... SC_LIVE_ENEMY_BASES). Mutating either of
those breaks every saved checkpoint.

Per-slot statefulness: one `FeatureBuilder` instance per controlled slot
(it carries a `Belief`, a `MapPrior`, the K=5 deques, and a per-tile
respawn-countdown grid). Reset is implicit on `step == 0`.

## 2. Actor — `SymbolicTransformerActor`

`policy.SymbolicTransformerActor` — a token-set transformer over the four
symbolic + raw branches. Architecture summary:

| Token group | Count | Source                            | Type id |
| ----------- | ----- | --------------------------------- | ------- |
| CLS         |   1   | learned parameter                 |    —    |
| tile        | 256   | `tile_embed(grid_perm)` + spatial_pos | 0 |
| base        |   5   | `base_embed(base_feats)` + base_pos  | 1 |
| raw         |  84   | `raw_embed(cat(raw_agent, raw_base))` + raw_pos | 2 |
| scalar      |   1   | `scalar_embed(scalar)`              | 3 |
| **total**   | **347** |                                 |         |

Each group adds a learned type embedding; CLS (token 0) reads the post-LN head
to produce `[N, 6]` action logits. The attention is hand-rolled (`_MHA`) so
that ONNX export uses only standard ops.

`actor.act(...)` masks the logits to legal actions before sampling. Eval &
viz set `temperature=0.001` (argmax-like); opponent neural members use
`temperature=1.2` (more exploratory, league-diversity sliver).

Checkpoint format: `torch.save({"state_dict": ..., "cfg": ...})`. The cfg
records `(d_model, n_layers, n_heads, ffn_dim, dropout)` so a saved actor can
be rebuilt without knowing the env-var-driven scale that trained it. Used by
`SymbolicTransformerActor.from_checkpoint`.

## 3. Critic — `CentralizedCritic` (CTDE)

`critic.CentralizedCritic` — a separate transformer over a privileged global
state. **Centralized Training, Decentralized Execution**: only the critic
sees the env's entity registry; the actor is exported and served on the
agent-local observation alone.

Privileged input is built by `critic.encode_global_state(dynamics, slot,
step)`:

- 9 planes × 16×16: walls, self/enemy positions, agent HP, bomb timer, blast
  footprint (via the env's own LOS function), self/enemy base, collectibles.
- 6 scalars: self health, frozen ticks, team resources, team bombs, step
  fraction, team id.

Crucially the encoding is **agent-relative**: the querying slot's own
location/health/base appear on dedicated planes. This lets a single critic
value the several learner slots that share one env step (each has its own
reward stream, gstate, and gscalar tuple).

**Single-frame, not K=5 stacked**. The actor's frame-stack exists to give the
policy short-horizon dynamics from a partially-observable viewcone; the
critic already has the full entity registry, so stacking would only multiply
memory without adding information.

Architecture parity: the critic re-uses the actor's `_TransformerBlock` so
attention scaling stays in sync. d_model / n_layers / n_heads default to
`(64, 4, 4)` — same width and depth as the actor. Output: a single scalar
value head `[N]`.

Why a separate trunk (not a shared-encoder dual-head)? Decoupled tuning
(different LRs — see §6.1), and the actor must export cleanly to ONNX
without dragging value-head weights into the served graph.

## 4. Stage 1 — Behavior Cloning (BC)

Cold-starting PPO from a random actor against the scripted strategies
collapsed in earlier runs (the PPO regime can't outpace the scripted's
near-zero entropy with random gradient signal). BC fixes this by warming
the actor to a copy of the strongest scripted strategy
(`balanced_extreme_opening`).

### 4.1 DAgger structure

`run_bc_transformer.main` drives three rounds of dataset aggregation:

1. **R1 (β = 1.0)**: 200 episodes, teacher controls the learner slot
   exclusively. Pure offline imitation.
2. **R2 (β = 0.5)**: 100 episodes; with prob 0.5 the partially-trained
   actor drives the learner slot — but *every visited state still gets a
   teacher action label*. This is the DAgger correction: states the
   in-training actor visits but the teacher would never enter must still be
   labelled, or the clone never sees its own state distribution.
3. **R3 (β = 0.5)**: 100 episodes, dataset aggregated; same recipe.

Each round runs `bc.train_bc` for `BC_EPOCHS` (default 10) epochs on the
accumulated dataset.

### 4.2 Opponent variety

`bc._default_opponent_pool` puts **RandomAgent ×4** in the opponent pool
plus every non-teacher scripted strategy. The 4× random ratio prevents the
common failure where six scripted agents in mutual contention deadlock in
their starting tiles and never bomb — which starves the dataset of
PLACE_BOMB labels and produces a structurally non-bombing clone.

### 4.3 Multi-worker parallelism

`BC_WORKERS=N` uses `multiprocessing.Pool` to chunk seeds across workers.
The pool's tasks are picklable: state_dict + cfg dict + seed list + opponent
pool **meta** (list of `(kind, *args)` tuples; lambdas can't pickle).
Each worker reseeds torch/np/random from a chunk-unique `base_seed` so
trajectories never coincide. With `BC_WORKERS=4` the 400-episode pipeline
drops from ~30 min to ~8 min on a T4.

### 4.4 Training loop

`bc.train_bc` — vanilla cross-entropy. The entire dataset is stacked on the
**actor's device** (GPU when available). Illegal actions are masked via
`torch.where(mask, logits, -1e8)` *before* the softmax, so the clone never
learns to favour an env-forbidden move.

Dataset materialisation on GPU is large: ~3.5 GB after R1, ~12 GB by R3 at
K=5 stack. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is forced
before `import torch` to let the allocator grow/shrink segments.

### 4.5 BC gate

`bc.bc_gate(actor, teacher_strategy, tolerance=0.05)` runs `evaluate_policy`
on 16 fixed seeds (random opponents) twice: once with the clone, once with
the teacher itself. The clone passes if `(teacher - clone) / |teacher|
<= 0.05`. The teacher's score is the reference scale — only the *relative*
gap is gated, since the absolute env-default score is noisy.

The actor checkpoint is saved **before** the gate runs so that a gate-side
crash never loses the trained model.

Output: `ae/src/policy_transformer_bc.pt`.

## 5. Stage 2 — Critic Pretraining

A random critic at PPO start produces garbage advantages (advantage =
return − V(s)). Paired with a competent BC actor this manufactures spurious
policy gradient signal from update 1 — a documented contributor to the
historical "flat zero" PPO collapse. This two-step stage gives PPO a
critic that already explains a non-trivial fraction of the BC actor's
return distribution.

### 5.1 `collect_critic_data.py`

Frozen BC actor vs the league's scripted anchors (15 strategies), 200
episodes × 4 workers by default. Reuses `train_selfplay.collect_rollout_
parallel` with `reward_shaper=None` so `env_rewards` carries the unshaped
env-default reward on the same scale PPO's GAE consumes. Only the critic's
inputs are saved (`gstate, gscalar, env_rewards, dones, episode_len`) — the
actor's observation tensors are dropped to keep the dump small.

When `--workers=1` the script bypasses multiprocessing entirely (a per-
episode tqdm bar gives one tick every ~12 s), and moves the actor to CUDA
for faster learner-slot inference. With `--workers>1` it uses the spawn
context and `file_system` tensor-sharing.

### 5.2 `pretrain_critic.py`

Reads the dump, computes Monte-Carlo discounted return-to-go per
episode-slot run with γ=0.99 (matching `train_selfplay Args.gamma`), and
fits `CentralizedCritic` by MSE for 30 epochs at LR=3e-4.

Diagnostics:
- `train_mse`, `val_mse`: should drop monotonically.
- `val_explained_variance = 1 - MSE/Var(target)`: a constant predictor scores
  0.0. **Look for ≥ 0.3 by the final epoch** before promoting to PPO. If it's
  near zero, the BC actor is producing trajectories the critic structurally
  can't fit — investigate before moving on.

Output: `ae/src/training/critic_pretrained.pt` (kept out of `ae/src/` so the
container ship doesn't bundle the unused critic).

## 6. Stage 3 — PPO Self-Play

The substantive stage. CleanRL-style single-file PPO with a centralized
critic. The orchestration loop is in `train_selfplay.main` (line 808+).

### 6.1 The PPO update

`ppo_update(actor, critic, opt, buf, advantages, returns, cfg, device)`:

1. Move the rollout buffer to device.
2. Compute `explained_variance` from the **pre-update** values
   (`values = returns − advantages`).
3. Shuffle indices, iterate `cfg.update_epochs` epochs of `cfg.
   num_minibatches` minibatches each.
4. Per minibatch: re-evaluate actor + critic, compute
   - clipped surrogate: `max(-A*ratio, -A*clip(ratio, 1±ε))`,
   - critic MSE,
   - entropy bonus (annealed coefficient).
5. Backprop the combined loss; clip global gradient norm to
   `cfg.max_grad_norm`.
6. `opt.step()` updates both param groups (actor LR, critic LR — see
   `make_optimizer`).

Standard "approx_kl" diagnostic: `(-log_ratio).mean()`. Clipfrac:
`(|ratio − 1| > clip_coef).float().mean()`.

**Important**: the advantage is normalised **per minibatch**
(`(a - a.mean()) / (a.std() + 1e-8)`), not globally. This is the CleanRL
convention; it slightly reduces variance but trades off against fidelity to
the global advantage distribution.

Critical config note: actor LR (3e-6) and critic LR (1e-4) live in
`PPOConfig`, with the actor LR *much* lower than typical CleanRL Atari
defaults (2.5e-4) because the actor is BC-warm-started and we want PPO as
a gentle refiner, not an explorer. See §10 for what happens when this is
too high.

### 6.2 Centralized critic flow

Per learner-slot transition the rollout calls
`encode_global_state(dynamics, slot, step)` and writes `gstate, gscalar`
into the buffer alongside the actor's local obs. At update time the critic
consumes only those globals — never the actor's tensors. The asymmetry is
the whole point of CTDE: the critic doesn't lose by being privileged
because we never serve it.

### 6.3 Rollout collection

`collect_rollout` walks `env.agent_iter()` (the AEC turn order). For each
slot turn:

- If the slot is a learner: build features, run `actor.act`, write all
  buffer fields at `slot_base[slot] + slot_cursor[slot]`. The previous
  transition of this slot is back-filled with the reward and done that just
  arrived from the env (rewards are observed *one turn after the action*).
- If the slot is an opponent: dispatch the per-slot agent's `action(obs)`.

**Multi-slot data gain**: with K=3 random learner slots per episode the
trainer collects 3× transitions per env step — at the cost of intra-episode
correlation. (We mitigate this by mini-batching shuffling).

**Run-contiguous buffer layout**: each (episode, slot) pair is laid out as
one contiguous 200-step block in the buffer, so reversed-time GAE within
each block is correct. The cursor `total_written` advances globally — only
*completed* transitions consume buffer cells.

**Worst-case pre-alloc + slice**: with slot-rotation the actual K per
episode varies in [lo, hi]. The buffer is allocated at `hi * EPISODE_LEN *
num_episodes` (worst case), then sliced to `total_written` at the end via
`_slice_buffer`. This avoids per-episode resizing and trades a bit of peak
RAM for code simplicity.

### 6.4 Opponent registry & kind-mix

`OpponentRegistry.sample_slot_opponent(rung_opponents)` returns
`(Member, agent)` for one non-learner slot, drawn from a 4-way categorical:

| Kind             | Sample from                  | Agent class           |
| ---------------- | ---------------------------- | --------------------- |
| `league`         | `rung_opponents`             | `NoisyScriptedAgent` (if scripted) or `NeuralAgent(T=opp_neural_temperature)` |
| `random`         | —                            | `RandomAgent`         |
| `noisy_scripted` | `league.anchors()` uniformly | `NoisyScriptedAgent`  |
| `raw_scripted`   | `league.anchors()` uniformly | clean `ScriptedAgent` |

`NoisyScriptedAgent` (in `evaluate.py`) wraps a scripted strategy with
ε-noise: with prob ε replace the strategy's action with a uniform random
**legal** action. Default ε = 0.08.

`NeuralAgent` for league checkpoints uses `temperature=opp_neural_
temperature` (default 1.2) for more exploratory opponents. Eval and viz
use `temperature=0.001` (argmax-like).

Note: at rung 1 the "league" kind only produces *noisy* scripted opponents
because every league member is scripted; the only way to face a clean
scripted opponent at rung 1 is via the "raw_scripted" kind. This is by
design (the league should not artificially fight clean scripteds — that's
what eval is for).

**Random-share curriculum**. `_interp_random_share(update, total_updates,
start, end)` linearly interpolates the random share from `start`
(default 0.40) at update 1 to `end` (default 0.15) at the final update.
The league share absorbs the slack. The `_compute_mix` helper clamps the
league share to ≥ 1e-6 so `random.choices(..., weights=...)` never sees a
zero-weight bucket.

Rationale: early in training the policy needs maximum coverage so it
doesn't overfit to a narrow scripted-anchor regime; as the league fills
with self-play checkpoints the random share can taper.

### 6.5 League & 3-rung ladder

`League` (in `league.py`) holds:

- **Anchors**: one `Member` per `scripted.strategies.STRATEGIES` key.
  Never evicted. Auto-grow: adding a new strategy to STRATEGIES adds an
  anchor on next League construction.
- **Checkpoint pool**: FIFO-evicted at `max_checkpoints` (default 10).
  Each `snapshot(ckpt_path, update)` appends `Member(name=f"ckpt:{update}",
  kind="checkpoint", ref=ckpt_path)`.

PFSP sampling: `_pfsp_weight(winrate) = (1 − p)² + 1e-3`. Members the
learner loses to weight high; the epsilon keeps all members reachable.
An untested member's `learner_winrate()` defaults to 0.5 (so weight ≈
0.25), giving new checkpoints a fair sample rate immediately.

`RungLadder` (in `train_selfplay.py`):

| Rung | `current_opponents()` returns                  | `sample_opponents()` returns |
| ---- | ---------------------------------------------- | ---------------------------- |
| 1    | `league.anchors()` (15 scripted)               | the full anchor list         |
| 2    | `league.checkpoints()` (PFSP)                  | 5 PFSP-sampled members       |
| 3    | `league.checkpoints()` (PFSP, freshest in)     | 5 PFSP-sampled members       |

Promotion: `try_promote(win_rate)` advances if `win_rate >= 0.7` and the
current rung is < 3. There is no demotion.

**Bookkeeping gotcha (handled)**: `outcomes` returned from worker
processes carry *pickled copies* of `Member` objects. Recording wins on
those copies would never affect the main league's Members and the PFSP
weights would stay flat. `main` re-maps each returned member to the
canonical league Member by name (`scripted:<strat>` or `ckpt:<update>`
— unique). See line 915.

### 6.6 Reward shaping (training only)

`AntiIdleShaper(penalty=0.05, total_updates)`:
- On the STAY action only, subtract `penalty * frac`.
- `frac = max(0, 1 - update / total_updates)` — full strength at update 1,
  zero by the end.

Rationale: an earlier PPO run collapsed to a flat-zero idle policy. BC
warm-start is the primary defence; the shaper is a small backstop in case
PPO drifts back toward STAY early. Annealed to zero so it can't bias the
final policy.

**Eval never shapes**. `evaluate.evaluate_policy` runs the env with default
reward config, period. This is critical because every comparison across
training runs depends on the eval score being on a fixed scale.

### 6.7 Parallel rollout

`collect_rollout_parallel` chunks `num_episodes` across `num_workers` Pool
workers, sends each worker a CPU `state_dict` + cfg + episode count + seed,
and concatenates the returned RolloutBuffers.

Key configuration:

- **Spawn context**: `multiprocessing.get_context("spawn")`. Fork after
  `import torch` and `import tqdm` deadlocks on Linux (the parent already
  has torch's threadpool + tqdm's monitor thread running, and fork doesn't
  carry over thread state cleanly). Spawn re-imports modules in each
  worker (~3-5 s startup) but is bulletproof.
- **`set_sharing_strategy("file_system")`**: the default `file_descriptor`
  tensor-sharing opens one Unix socket per tensor storage; a state_dict
  with ~50 tensors × N workers × M tasks exhausts per-process fd caps
  (`OSError: [Errno 24]`). Filesystem-backed sharing uses named tempfiles
  instead.
- **One task per worker chunk**, NOT one task per episode. Earlier code
  dispatched per-episode and shipped the state_dict `num_episodes` times,
  which fd-exploded at high episode counts.
- **Per-worker progress**: each worker prints `[worker N] X/Y eps` every
  `log_every=5` episodes. No outer tqdm bar (would interleave badly with
  worker stdout).
- **Per-worker RNG**: each worker reseeds torch/np/random from `seed0`
  (= `args.seed + ep_offset`). The OpponentRegistry's `rng` is rebuilt
  from `seed0 ^ rng_seed` so per-update opponent picks are fresh — XORing
  with `args.seed ^ update` (see line 908) makes the rotation per-update
  rather than fixed across the run.
- **`registry_kwargs` threaded per-update**: the worker rebuilds its
  `OpponentRegistry` each update with the live `current_mix` weights,
  so the random-share curriculum reaches each worker correctly. A historical
  bug (before this thread-through) silently used class defaults inside
  workers — see WORKFLOW.md troubleshooting.

### 6.8 Evaluation & promotion

Every `eval_every` updates (default 25):

1. `opp_specs = ladder.current_opponents()`.
2. Pick **5** random opp_specs with replacement → instantiate via
   `registry.make`.
3. Deepcopy the actor to CPU (`copy.deepcopy(actor).to("cpu")`); never
   move the training actor — its Adam state is pinned to `device`.
4. `evaluate_policy(NeuralAgent(eval_actor, T=0.001), opp_agents,
   range(eval_seeds))` runs 16 episodes (default) and returns mean score +
   win-rate.
5. If `win_rate > best_winrate`, save `policy_best_rung{rung}.pt`.
6. `ladder.try_promote(win_rate)` advances if ≥ 0.7.

**Single-slot eval**: `evaluate_policy` puts the learner in `agent_0` only;
the other 5 slots are opponents. This differs from training rollouts
(which use K∈[1,3] learner slots). Don't compare eval `mean_return` to
in-rollout `mean_return` directly — the opponent pressure differs.

### 6.9 Snapshotting & artifacts

Every `snapshot_every` updates (default 25):

- Save `policy_rung{rung}_u{update}.pt` to `ae/src/`.
- `league.snapshot(...)` adds it to the checkpoint pool (FIFO-evicted at
  `max_checkpoints=10`).
- Overwrite `policy_final.pt`.

On a new max winrate: save `policy_best_rung{rung}.pt`.

End-of-training: save `policy_final.pt` one more time.

### 6.10 Metrics

`MetricsLogger` (in `metrics.py`):

- One CSV row per update (full rewrite each `log()` so late-appearing
  columns back-fill earlier rows with blanks).
- TensorBoard scalars: one `add_scalar` per metric per update.
- `plot(png)`: a 6-panel PNG (losses / entropy / mean_return / eval / rung
  / pool_size) at every `viz_every` updates.
- `leaderboard(league, update, csv, png)`: per-update leaderboard CSV
  (append) + bar-chart PNG (overwrite).

Per-update metric keys:

| Key                  | Source                                 |
| -------------------- | -------------------------------------- |
| `policy_loss`        | `ppo_update` return                    |
| `value_loss`         | `ppo_update`                           |
| `entropy`            | `ppo_update`                           |
| `explained_variance` | `ppo_update` (pre-update values)       |
| `approx_kl`          | `ppo_update`                           |
| `clipfrac`           | `ppo_update`                           |
| `advantage_mean`     | `ppo_update`                           |
| `advantage_std`      | `ppo_update`                           |
| `mean_return`        | `buf.env_rewards.sum() / n_runs`       |
| `rung`               | `ladder.rung`                          |
| `pool_size`          | `len(league.checkpoints())`            |
| `anti_idle_coef`     | `shaper.penalty * shaper._frac`        |
| `ent_coef`           | annealed; `cfg.ent_coef`               |
| `actor_lr`           | `cfg.learning_rate`                    |
| `critic_lr`          | `cfg.critic_learning_rate`             |
| `opp_mix_random`     | `current_mix[1]`                       |
| `opp_mix_league`     | `current_mix[0]`                       |

On eval updates the additional keys `eval_winrate` and `eval_score` are
appended.

Also rendered at `viz_every`: `replay_u{update}.mp4` — a 200-step labelled
episode replay (see §8).

## 7. Stage 4 — ONNX Export

`export_onnx.export_transformer(pt_path, onnx_path)`:

- Load `SymbolicTransformerActor.from_checkpoint(pt_path)`.
- Trace with dummy zeros of the five-tensor input shapes.
- `torch.onnx.export(..., opset_version=17, dynamo=False)` with the input
  names `(grid, base_feats, raw_agent, raw_base, scalar)` and the output
  name `logits`. Batch axis is dynamic on every tensor.

The served container drops torch and uses onnxruntime (smaller image,
faster cold start). The critic is never exported.

The CLI default `python -m export_onnx` exports
`policy_transformer_bc.pt → policy.onnx` (i.e. the BC clone). After PPO,
point it at `policy_final.pt` (or one of the `policy_best_rung*.pt`) — see
WORKFLOW.md §Stage 4.

## 8. Visualization

`visualize.render_episode(slot_agents, out, fps=5, max_steps=200, seed=0)`:

- Runs one env episode in `rgb_array` render mode.
- For each frame: env's rgb image, a per-slot color/label legend strip
  (top), and an optional belief side-panel (right) for `agent_0` when that
  slot is a scripted agent.
- `imageio.mimwrite(..., codec='libx264')` -> MP4.

The belief panel renders `Belief.last_visible_cells`, `enemy_bases`,
`enemy_bombs`, `dead_bases`, `live_enemies`, `frozen_enemies`,
remaining collectibles, and the wall graph — invaluable for diagnosing
scripted-strategy bugs.

`train_selfplay._build_viz_slot_agents(actor, learner_slots, opponents,
update)` builds the slot_agents dict:

- Learner (CPU deepcopy of actor, `NeuralAgent T=0.001`) at
  `learner_slots`.
- League members from the current `opponents` list fill the rest.

**Note**: as of this writing, the viz path always uses
`args.learner_slots_fixed` even when `args.learner_slots_mode == "rotate"`
is active — see Findings.

## 9. Knob inventory

Comprehensive list across all stages. Defaults match the code at this
revision.

### 9.1 BC stage (env vars + `bc.train_bc` args)

| Knob          | Default                     | Source                | Effect                                                  |
| ------------- | --------------------------- | --------------------- | ------------------------------------------------------- |
| `BC_TEACHER`  | `balanced_extreme_opening`  | env var (run_bc_*)    | scripted strategy used as labeler                       |
| `BC_WORKERS`  | `1`                         | env var               | parallel rollout workers for DAgger                     |
| `BC_EPOCHS`   | `10`                        | env var               | gradient epochs per DAgger round                        |
| `BC_LR`       | `3e-4`                      | env var               | Adam LR for BC training                                 |
| `TF_D_MODEL`  | `64`                        | env var               | transformer width                                       |
| `TF_N_LAYERS` | `4`                         | env var               | transformer depth                                       |
| `TF_N_HEADS`  | `4`                         | env var               | attention heads                                         |
| `TF_FFN_DIM`  | `4 × d_model`               | env var               | FFN width                                               |
| `TF_DROPOUT`  | `0.1`                       | env var               | dropout                                                 |
| `batch_size`  | `256`                       | `train_bc` arg        | BC minibatch size                                       |
| `tolerance`   | `0.05`                      | `bc_gate` arg         | relative score gap clone vs teacher                     |
| `beta`        | `1.0 / 0.5 / 0.5`           | per-round in run_bc_* | DAgger β (P[teacher acts])                              |

### 9.2 Critic stage (CLI flags on `collect_critic_data` + `pretrain_critic`)

| Knob              | Default                          | Source                | Effect                                          |
| ----------------- | -------------------------------- | --------------------- | ----------------------------------------------- |
| `--actor`         | `policy_family_winner_bc.pt`     | collect_critic_data   | Frozen actor; WORKFLOW.md overrides to `policy_transformer_bc.pt` |
| `--episodes`      | `200`                            | collect_critic_data   | rollout episodes                                |
| `--learner-slots` | `[agent_0, agent_1, agent_2]`    | collect_critic_data   | which slots the frozen actor controls           |
| `--workers`       | `4`                              | collect_critic_data   | parallel rollout workers                        |
| `--epochs`        | `30`                             | pretrain_critic       | critic-fit epochs                               |
| `--lr`            | `3e-4`                           | pretrain_critic       | critic Adam LR                                  |
| `--batch-size`    | `256`                            | pretrain_critic       | minibatch size                                  |
| `--val-frac`      | `0.10`                           | pretrain_critic       | validation split fraction                       |
| `--gamma`         | `0.99`                           | pretrain_critic       | discount; **must match `train_selfplay Args.gamma`** |
| `--reward-scale`  | `1.0`                            | pretrain_critic       | reward multiplier (keep 1.0)                    |

### 9.3 PPO stage (`Args` + `PPOConfig`)

| Knob                       | Default                       | Source     | Effect                                                       |
| -------------------------- | ----------------------------- | ---------- | ------------------------------------------------------------ |
| `--total-updates`          | `1000`                        | Args       | PPO update count                                             |
| `--episodes-per-update`    | `4`                           | Args       | rollout episodes per update                                  |
| `--rollout-workers`        | `4`                           | Args       | Pool workers                                                 |
| `--learner-slots-mode`     | `rotate`                      | Args       | `rotate` or `fixed`                                          |
| `--slot-rotation-min/max`  | `1 / 3`                       | Args       | K∈[min,max] learner slots per episode                        |
| `--learner-slots-fixed`    | `(agent_0, agent_1, agent_2)` | Args       | used when mode=fixed (and by viz)                            |
| `--gamma`                  | `0.99`                        | Args       | GAE discount                                                 |
| `--gae-lambda`             | `0.95`                        | Args       | GAE λ                                                        |
| `--snapshot-every`         | `25`                          | Args       | save+pool every N updates                                    |
| `--eval-every`             | `25`                          | Args       | eval cadence                                                 |
| `--eval-seeds`             | `16`                          | Args       | eval episodes                                                |
| `--viz-every`              | `25`                          | Args       | replay/metrics PNG cadence                                   |
| `--anti-idle-penalty`      | `0.05`                        | Args       | STAY penalty (annealed)                                      |
| `--bc-init`                | `policy_bc.pt`                | Args       | **CAUTION**: default does not match `run_bc_transformer.py`'s output `policy_transformer_bc.pt`; override on CLI |
| `--critic-init`            | `critic_pretrained.pt`        | Args       | pretrained critic                                            |
| `--seed`                   | `1`                           | Args       | top-level RNG seed                                           |
| `--cuda` / `--no-cuda`     | `True`                        | Args       | use CUDA when available                                      |
| `--opp-eps-noise`          | `0.08`                        | Args       | ε on noisy scripted opponents                                |
| `--opp-neural-temperature` | `1.2`                         | Args       | T on neural opponents                                        |
| `--opp-mix-random-start`   | `0.40`                        | Args       | random share at update 1                                     |
| `--opp-mix-random-end`     | `0.15`                        | Args       | random share at final update                                 |
| `--opp-mix-noisy-scripted` | `0.15`                        | Args       | constant fraction                                            |
| `--opp-mix-raw-scripted`   | `0.10`                        | Args       | constant fraction                                            |
| `learning_rate`            | `3e-6`                        | PPOConfig  | actor Adam LR                                                |
| `critic_learning_rate`     | `1e-4`                        | PPOConfig  | critic Adam LR (10–33× actor)                                |
| `num_minibatches`          | `8`                           | PPOConfig  | per-epoch shuffle splits                                     |
| `update_epochs`            | `2`                           | PPOConfig  | minibatch epochs per update                                  |
| `clip_coef`                | `0.2`                         | PPOConfig  | PPO ratio clip                                               |
| `ent_coef`                 | `0.05 → 0.005`                | PPOConfig  | annealed linearly across the run                             |
| `vf_coef`                  | `0.5`                         | PPOConfig  | value-loss weight                                            |
| `max_grad_norm`            | `0.5`                         | PPOConfig  | global grad-norm clip                                        |
| `League.max_checkpoints`   | `10`                          | League ctor| FIFO cap on checkpoint pool                                  |
| `RungLadder.promote_winrate`| `0.7`                        | ctor       | promotion gate                                               |

## 10. Recent learnings (post-mortem of the 2026-05-23 PPO run)

**Honest summary** — what happened, what we learned.

### What ran

- Smoke (20 updates, 2 episodes/update): `approx_kl ≈ 0.10`, `clipfrac ≈ 0.18`,
  `explained_variance ≈ 0.48`, `entropy ≈ 0.16`. Looked stable enough to
  promote to the full run.
- Full run (200 updates of the planned 1000): `mean_return` climbed roughly
  monotonically from update 1 to update ~100, then collapsed
  by update ~200. Eval winrate (vs rung-1 scripted anchors) was **0 throughout**.

### Why we trusted the smoke

Smoke gauged the trainer's *plumbing* (workers don't deadlock, GAE compiles,
checkpoints save). It did NOT gauge whether the PPO config could refine an
actor without destroying the BC basin. Two signals foreshadowed the
collapse but were not gated on:

1. `approx_kl ≈ 0.10` is **5× the conventional ceiling** for stable PPO
   (most papers target < 0.02). We accepted it because the surrogate clip
   and grad-norm clip "should" have been protecting the policy. They
   didn't — they bounded *per-step* drift but couldn't undo the cumulative
   move off the BC basin.
2. The early `mean_return` climb came from rollouts against an opponent
   mix that included random + noisy_scripted (~55% of slots at update 1).
   Beating those is much easier than beating clean scripted anchors at
   eval — so the rollout return rose while eval winrate stayed at 0.

### Root cause analysis

The collapse was driven by *config*, not by a code bug. The actor had a
BC-warm-started near-deterministic policy; the PPO regime as configured
treated it as if it needed exploration:

- `ent_coef=0.05` (start) is large for a BC clone whose entropy is
  already ~0.16. The entropy bonus *added* drift away from the BC mode.
- `actor learning_rate=3e-6` was ALREADY the cut-down value (the previous
  iteration used 1e-5). Even at 3e-6 the cumulative drift across `update_
  epochs=2 * num_minibatches=8 = 16 gradient steps per update * 200
  updates = 3200 gradient steps` was enough to walk off the BC basin.
- `clip_coef=0.2` is the default — but with `approx_kl=0.10`, the clip
  fires only on the trailing tail of ratios, not on the bulk. The clip
  protects against single-step explosions, not slow drift.
- The critic actually behaved well: `explained_variance` rose toward
  0.7+ during the climb. The policy was the problem.

### Lessons (what to change in the next attempt)

1. **Treat PPO as a refiner, not an explorer**, when the actor is BC-warm.
   Start `ent_coef ≤ 0.005` (or even 0) — exploration is not the bottleneck.
2. **Actor LR ≤ 5e-7**, possibly with linear warmup. The 3e-6 default is
   too aggressive for a BC basin. KL should hold under 0.02 the entire run.
3. **Add a KL early-abort or LR-scale**. If `approx_kl > 2 * kl_target`
   for K consecutive updates, halve the actor LR or skip the update.
   CleanRL has this; we don't.
4. **Eval is the ground truth**. The leaderboard against rung-1 scripted
   anchors was 0 throughout — we should have aborted by update 50 the
   moment in-rollout return diverged from eval winrate.
5. **Smoke must include a long-run extrapolation**. 20 updates isn't long
   enough to surface cumulative KL drift; a 100-update smoke with eval at
   25/50/75/100 would have caught this in 30 min.

### Outcome

Ship the BC clone (or the `policy_rung1_u100` snapshot if it tested
higher in eval) for the competition deadline. The PPO refinement is
deferred to a re-attempt with the config in §12.

## 11. Code review findings

### 11.1 Critical (block correctness — fix before next training)

**C1. `train_selfplay.Args.bc_init` default does not match `run_bc_transformer.py` output**
(`train_selfplay.py:790`)

```python
bc_init: str = "policy_bc.pt"
```

`run_bc_transformer.py` writes `policy_transformer_bc.pt`. A user who runs
`python -m train_selfplay ...` without `--bc-init` will silently get the
"WARNING: bc_init not found — actor trained from SCRATCH (random init)"
path (line 838), which will collapse. WORKFLOW.md correctly tells the user
to pass `--bc-init policy_transformer_bc.pt`, but the silent fallback is
dangerous (a user who misses the warning line in the log will burn a
multi-hour training run on garbage). **Suggested fix**: default to
`policy_transformer_bc.pt`, and **raise** rather than warn when the file
is missing.

**C2. `collect_critic_data.py` default `--actor` is stale**
(`collect_critic_data.py:48`)

```python
parser.add_argument("--actor", default="policy_family_winner_bc.pt", ...)
```

The "family winner" pipeline is a previous-iteration artifact; the current
BC entrypoint is `run_bc_transformer.py` producing `policy_transformer_
bc.pt`. Same risk as C1: silent stale default leads to a misleading load
failure or, worse, training the critic on a different actor than the user
intended. **Suggested fix**: default to `policy_transformer_bc.pt` to
match the current pipeline; or require the flag.

**C3. Final-step reward back-fill** (`train_selfplay.py:265-271`)

The reward/done back-fill at line 265 fires when a slot's next turn returns
`prev_idx[slot]` non-None. In normal AEC operation each slot gets one
follow-up turn (with the env-truncation flag) at episode end, so the back-
fill *does* run. But if the AEC iter loop terminates before that final
back-fill turn (rare in `bomberman_env`, but it's defensive depth-in-
defence territory), the last transition of each slot-run would carry
`reward=0, done=0` — which would silently corrupt the GAE bootstrap for
that episode. **Suggested fix**: after `env.agent_iter()` returns, walk
`prev_idx` and force `dones[idx] = 1.0` for any slot whose final cell was
never closed; back-fill 0 reward is acceptable (no information about
post-truncation reward exists).

### 11.2 Important (latent bugs, design issues)

**I1. Viz uses `learner_slots_fixed` regardless of mode**
(`train_selfplay.py:998`)

```python
slot_agents = _build_viz_slot_agents(
    actor, args.learner_slots_fixed,
    opponents, update)
```

When `args.learner_slots_mode == "rotate"` (the default), training uses
random K∈[1,3] slots per episode. But the replay MP4 always renders the
learner in `(agent_0, agent_1, agent_2)` — which never happens in
training. The replays look unrepresentative. **Suggested fix**: sample
`(rotate, lo, hi)` once per viz frame, OR document that viz uses fixed
slots intentionally.

**I2. PPO advantage is normalised per minibatch**
(`train_selfplay.py:612`)

```python
a = (a - a.mean()) / (a.std() + 1e-8)
```

CleanRL convention, but at `num_minibatches=8` and `episodes_per_update=4`
the per-minibatch sample size is small (~150-450 transitions). At rung 1
with mostly-positive returns the per-minibatch mean can swing
significantly, making the effective policy gradient direction unstable.
**Suggested fix**: at minimum, log `advantage_mean` and `advantage_std` of
the **global** buffer (already in `losses`) AND the per-minibatch ones; or
move to global advantage normalisation, which is what most contemporary
PPO recipes do.

**I3. KL is reported but not gated** (`train_selfplay.py:624`)

`approx_kl` is computed and logged per update but never used for early-
abort or LR-scaling. CleanRL has an optional `target_kl` arg that aborts
the inner update loop if KL exceeds the target. The May-23 collapse
post-mortem identifies this as a critical missing defense.

**Suggested fix**: add `target_kl` and inside `ppo_update`'s inner loop:

```python
if kl_sum / mb_count > 1.5 * target_kl:
    return ...  # early exit
```

**I4. Eval picks 5 opponents *with replacement* by `random.choice`**
(`train_selfplay.py:967`)

```python
opp_agents = [registry.make(random.choice(opp_specs))
              for _ in range(5)]
```

At rung 2/3 with `len(pool) < 5` you'll pick the same checkpoint multiple
times. At rung 1 with 15 anchors you'll routinely miss some. Either way
the eval set composition varies update-to-update, so `eval_winrate` is
noisier than it should be. **Suggested fix**: use `random.sample(opp_
specs, min(5, len(opp_specs)))` (no replacement); pad with random.choice
only if needed.

**I5. `OpponentRegistry._actor_cache` is unbounded**
(`train_selfplay.py:79, 93`)

```python
self._actor_cache = {}
...
if member.ref not in self._actor_cache:
    actor = SymbolicTransformerActor.from_checkpoint(member.ref)
```

When `max_checkpoints=10` the cache is bounded in steady state, but
across thousands of updates the cache holds entries for evicted
checkpoints (the dict never evicts; only the League evicts). Each entry
is ~10 MB of actor weights. Over a 1000-update run with snapshot_every=25
that's 40 evicted entries; bounded but not GC'd until process exit.
**Suggested fix**: subscribe to `League.snapshot`'s eviction and pop the
cache entry, OR use an LRU cache with `maxsize=max_checkpoints+5`.

**I6. `NoisyScriptedAgent` `rng=random.Random(self._rng.random())`**
(`train_selfplay.py:112, 121`)

Passing a float in [0,1) as a `Random` seed works but it's unusual and
the entropy is < 53 bits — at 1000s of NoisyScriptedAgent constructions
across a run, collisions become more likely than a clean integer seed
would produce. **Suggested fix**: `random.Random(self._rng.randint(0,
2**31 − 1))`.

**I7. `make_optimizer` uses a single Adam with two param groups, but Adam
state lives inside the optimizer instance** (`train_selfplay.py:547`)

This is fine as written, but a subtle landmine: re-creating
`make_optimizer` mid-run (e.g. to change LRs) drops the second-moment
estimates, which can cause a short-term gradient explosion. **Suggested
fix**: if a future revision adds adaptive LR scaling, mutate
`param_group['lr']` in place rather than re-building the optimizer.

### 11.3 Minor (style, naming, cleanup)

**M1. Duplicate `import copy`** (`train_selfplay.py:11` and `:749`)

`copy` is imported at module top *and* re-imported inside
`_build_viz_slot_agents`. Harmless but noisy. **Fix**: remove the inner
re-import.

**M2. `RolloutBuffer` import ordering across files**

`collect_critic_data.py` imports `OpponentRegistry, collect_rollout,
collect_rollout_parallel` from `train_selfplay`, but the module-top
docstring describes the function as a "stage between BC and PPO" — fine,
but the dependency is fragile (any refactor of `train_selfplay` that
moves these names breaks the critic-data step silently). **Fix**: factor
`OpponentRegistry` and the rollout helpers into a separate `rollout.py`.

**M3. `_default_opponent_pool` is unused after `_opponent_pool_meta`**
(`bc.py:56`)

`_default_opponent_pool` is left in place for tests and the single-worker
path uses it. But the parallel path uses `_opponent_pool_meta` /
`_opponent_pool_from_meta` since lambdas don't pickle. Slight duplication;
both must change together if the pool composition changes. **Fix**:
have `_default_opponent_pool` delegate to `_opponent_pool_from_meta`.

**M4. `slot_rotation_min/max` are inclusive on both sides**
(`train_selfplay.py:226`)

`ep_rng.randint(lo, hi)` is inclusive on both ends (Python convention).
WORKFLOW.md says "K∈[min,max]" which is correct. Naming
`slot_rotation_max=3` looks exclusive to a Python reader (off-by-one
trap). **Fix**: rename to `slot_rotation_max_inclusive` or document in
the docstring.

**M5. `n_returns = max(1, buf.size // EPISODE_LEN)` overcounts under
rotate** (`train_selfplay.py:935`)

When `episodes_per_update=4` and K=3, `buf.size = 12 * 200 = 2400`, so
`n_returns = 12`. But `buf.env_rewards.sum()` sums across 12 (episode,
slot) runs. So `mean_return = total / 12` is the **per-slot-run** mean,
not the per-episode mean. The metric is consistent across updates but
misnamed. **Fix**: rename to `mean_run_return` or document.

**M6. `metrics` dict initialised in a closure under `if logger is not
None`** (`train_selfplay.py:937`)

If `args.viz_every` is set to 0 (intentionally disable logging), the
later `if logger is not None: metrics["eval_winrate"] = ...` would
NameError — but the outer `if logger is not None:` guards it correctly.
Still confusing. **Fix**: define `metrics = {}` unconditionally and gate
its content build on `logger`.

**M7. `League` PFSP weight epsilon**

`_pfsp_weight` adds `+ 1e-3` to keep weights non-zero. With 15+ members
all at the default 0.5 winrate, the relative weight ratio with epsilon
is `(0.25 + 1e-3) / 0.25 ≈ 1.004` — essentially uniform, but the epsilon
becomes load-bearing when a strong member's winrate approaches 1
(weight → 1e-3 vs others at 0.25). **Suggested**: document that
`learner_winrate ≥ 0.97` effectively retires a member from sampling.

### 11.4 Strengths (what's done well)

**S1. Architectural separation**. Actor / critic / league / registry /
metrics each live in their own file with crisp contracts. PPO orchestrates
them but doesn't reach into their internals (except for the league-member
re-map, which is documented).

**S2. CTDE is implemented cleanly**. The critic input is built per-transition
from the env's entity registry at the moment the actor takes its action;
the actor never sees those tensors. ONNX export is unaffected. This is
the right MAPPO/CTDE pattern.

**S3. Defensive multiprocessing**. The spawn context + `file_system` tensor
sharing + per-worker `torch.set_num_threads(1)` reflect hard-won lessons
(documented in the file headers). The fact that the registry knobs are
threaded through per-update via `registry_kwargs` shows the team
discovered and fixed the earlier silent-default bug.

**S4. The reward shaper is annealed and eval-disabled by construction**.
The shaper class lives on the trainer and is not even constructed in
`evaluate.py`. Impossible to accidentally apply shaping to eval.

**S5. Checkpoint format carries cfg**. `{"state_dict": ..., "cfg": ...}`
means `from_checkpoint` reconstructs the right architecture without a
sidecar metadata file. Survives env-var-driven scale changes.

**S6. League re-map after worker outcomes** (line 915). A subtle bug
(stale pickled Members not affecting league bookkeeping) was found and
fixed with a clear comment. Good defensive engineering.

**S7. `expandable_segments:True` set before import torch** in every entry
point. Easy to forget; consistent here.

**S8. Buffer pre-alloc + slice**. `_slice_buffer` keeps the worst-case
pre-alloc cheap and the final buffer minimal; no resizing per episode.

**S9. The viz path includes a belief side panel**. For a scripted agent in
agent_0, seeing the live Belief + cascade layer alongside the env render
is a debugging force multiplier.

### 11.5 Test coverage gaps

- `compute_advantages` has no unit test. Given how subtle the slot-run
  contiguity + `T-1` boundary handling is, a synthetic-trajectory test
  (constant reward, alternating done flags) would lock in the math.
- `_compute_mix` + `_interp_random_share`: easy to add tests for
  endpoint behaviour, sum-to-1, clamping.
- `OpponentRegistry.sample_slot_opponent`: kind-mix distribution is
  testable by Monte-Carlo with a seeded RNG.
- `RungLadder.try_promote`: a 5-line test would lock in the no-demotion
  + ceiling-at-3 contract.
- `League.snapshot` eviction (FIFO at max_checkpoints).

## 12. Suggested next-attempt training config

A conservative re-attempt informed by §10. The intent: a *gentle refiner*
that won't lose the BC basin. If you change one thing, change actor LR —
that was the largest contributor to the collapse.

### PPOConfig

```python
@dataclass
class PPOConfig:
    learning_rate: float = 5e-7         # down from 3e-6
    critic_learning_rate: float = 1e-4  # unchanged
    num_minibatches: int = 8            # unchanged
    update_epochs: int = 1              # down from 2
    clip_coef: float = 0.1              # tighter clip (down from 0.2)
    ent_coef: float = 0.0               # OFF — BC is already informative
    ent_coef_final: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.3          # tighter (down from 0.5)
```

### Args overrides via CLI

```bash
$PY -m train_selfplay \
  --total-updates 500 \
  --episodes-per-update 4 \
  --rollout-workers 4 \
  --bc-init policy_transformer_bc.pt \
  --critic-init critic_pretrained.pt \
  --eval-every 10 --snapshot-every 25 --viz-every 25 \
  --eval-seeds 32 \
  --opp-mix-random-start 0.10 \
  --opp-mix-random-end 0.05 \
  --opp-mix-noisy-scripted 0.05 \
  --opp-mix-raw-scripted 0.30 \
  --anti-idle-penalty 0.0 \
  --seed 7
```

Rationale:

- **`eval-every=10`** (instead of 25) lets you abort by update 30-50 if
  `eval_winrate` doesn't track `mean_return`. The May-23 run could have
  saved 4 hours of wall time with this.
- **`eval-seeds=32`** halves the eval winrate variance.
- **`opp-mix-raw-scripted=0.30`** + low random share: train against
  *clean* scripted opponents most of the time. The robustness/coverage
  trade-off in the original mix was the wrong direction for a BC clone
  that's already broadly trained.
- **`anti-idle-penalty=0.0`**: the BC clone has bombed enough — the
  shaper is no longer needed.
- **First-step KL gate (not yet implemented)**: add a `target_kl=0.02`
  early-abort in `ppo_update`. Without it the actor LR cut alone isn't
  sufficient. Track approx_kl in the tqdm postfix (replace `ret=` with
  `kl=`) so an operator sees the divergence as it happens.

### Smoke check (mandatory before the long run)

```bash
$PY -m train_selfplay \
  --total-updates 50 --episodes-per-update 2 \
  --rollout-workers 2 \
  --bc-init policy_transformer_bc.pt \
  --critic-init critic_pretrained.pt \
  --eval-every 5 --snapshot-every 10 --viz-every 10 \
  --seed 7
```

Acceptance gates for the smoke:

- `approx_kl < 0.02` on every update past warmup.
- `eval_winrate ≥ 0.10` by update 25 against rung-1 scripted anchors.
- `explained_variance ≥ 0.3` by update 25.

If any of those fail, drop actor LR by 2× and re-smoke. Do **not** start
the long run with red gates.
