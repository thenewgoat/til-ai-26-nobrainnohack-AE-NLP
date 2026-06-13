# AE neural-agent training pipeline

Offline pipeline that produces the neural AE agent: **behavior-cloning → critic
pretraining → PPO self-play → ONNX export**. Never shipped in the container
(`ae/.dockerignore` excludes `src/training/`). The scripted agent in
`ae/src/scripted/` needs none of this.

For the *why* behind each stage see `ae/SOLUTION.md`; for the architecture
reference see `PIPELINE.md`; for the step-by-step run guide (commands, env vars,
troubleshooting) see `WORKFLOW.md`.

## Setup

```bash
cd ae/src/training
uv sync
uv add --editable ../../../til-26-ae   # the til_environment package, first time only
```

All scripts use flat imports resolved against `ae/src` + `ae/src/training`. Run
them with that on the path:

```bash
cd ae && PYTHONPATH=src:src/training python src/training/<script>.py
```

## File map

### Pipeline core (library modules, imported by the rest)
| File | Role |
| --- | --- |
| `bc.py` | DAgger behavior-cloning dataset collection + `train_bc` + `bc_gate`. |
| `critic.py` | `CentralizedCritic` + `encode_global_state` (9-plane privileged global state). |
| `train_selfplay.py` | PPO self-play core loop: rollout, GAE, `ppo_update`, league snapshots, rung ladder. |
| `train_hybrid.py` | PPO loop for the hybrid post-opener actor. |
| `hybrid_ppo.py` | Hybrid PPO math primitives (masked surrogate, advantage norm, KL-adaptive LR). |
| `hybrid_rollout.py` | Rollout collection for the hybrid post-opener (records post-handover ticks). |
| `league.py` | Opponent pool + PFSP `(1−winrate)²` sampling. |
| `evaluate.py` | Agent wrappers (`RandomAgent`, `GreedyAgent`, `ScriptedAgent`, `NoisyScriptedAgent`, `NeuralAgent`) + `evaluate_policy`. The most-imported module. |
| `metrics.py` | `MetricsLogger` — CSV + TensorBoard + multi-panel PNGs. |
| `export_onnx.py` | Export `SymbolicTransformerActor` → ONNX for serving. |

### Entry-point runners (CLI; one stage each)
| File | Stage |
| --- | --- |
| `run_bc_transformer.py` | BC DAgger of the scripted teacher into the transformer actor. |
| `run_bc_family_winner.py` | BC from winning balanced-family trajectories. |
| `collect_family_winner.py` | Collect the winning-family BC dataset. |
| `collect_critic_data.py` | Frozen-actor rollouts → critic-pretrain data. |
| `pretrain_critic.py` | MSE-fit the critic on Monte-Carlo reward-to-go. |
| `run_hybrid.py` | Hybrid post-opener PPO run / eval-only. |

### Evaluation
| File | Role |
| --- | --- |
| `hybrid_eval.py` | `HybridAgent` + paired-continuation A/B acceptance eval. |
| `measure_tournament.py` | Read-only N-way scripted tournament (strategy dominance). |

### Diagnostics (debug one-offs, no test coverage)
| File | Role |
| --- | --- |
| `diag_belief.py` | Belief-vs-ground-truth foraging cost. |
| `diag_hybrid.py` | Hybrid post-handover improvement (gate-override rate, advantage stats). |
| `diag_oscillation.py` | Post-handover oscillation / ONNX-vs-torch agreement. |
| `trace_belief.py` | Single-episode belief trace (proves foraging is belief-driven). |
| `probe_horizon.py` | `forage_chain` planning-depth probe. |

### Visualization
| File | Role |
| --- | --- |
| `visualize.py` | Labeled episode-replay MP4 renderer (library; used by the runs). |
| `viz_loss.py` | Per-event reward-breakdown PNG (where hybrid loses vs scripted). |
| `viz_stages.py` | Hybrid stage-transition overlay (OPENER / RL / ESCAPE / GATE). |
| `viz_qual.py` | Qualitative vectorized episode render. |

### Docs
`PIPELINE.md` (architecture reference) · `WORKFLOW.md` (run guide) · this file (index).

## Run the pipeline

See `WORKFLOW.md` for the full recipe with flags. In brief, from `ae/src/training`:

```bash
python run_bc_family_winner.py        # 1. BC clone        -> ../policy_family_winner_bc.pt
python collect_critic_data.py         # 2a. critic data    -> logs/critic_pretrain_data.pt
python pretrain_critic.py             # 2b. critic fit     -> critic_pretrained.pt
python train_selfplay.py              # 3. PPO self-play   -> ../policy_final.pt + rung ckpts
python export_onnx.py                 # 4. export          -> ../policy.onnx
```

## Test & monitor

```bash
uv run pytest                      # full suite (lives in ae/tests/)
uv run tensorboard --logdir viz    # plus PNG replays + leaderboard under ae/training/viz/
```
