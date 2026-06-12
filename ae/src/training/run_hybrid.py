"""CLI runner for hybrid post-opener PPO (Plans 5.1-5.5).

Trains `train_hybrid` with checkpointing + incremental jsonl logging + periodic
paired-continuation eval, then exports the final actor to ONNX and runs an
acceptance eval. Also supports `--eval-only` to score a saved checkpoint.

Run from `ae/` with the training package on the path:
    cd ae && PYTHONPATH=src:src/training ../til-26-ae/.venv/bin/python \
        src/training/run_hybrid.py [args]

Examples
--------
Pilot (one seed; run twice with --seed0 0 and --seed0 1000 for the 2-seed gate):
    ... run_hybrid.py --updates 400 --episodes-per-update 4 --device cuda \
        --forward-bias 0.5 --anti-idle 0.05 --critic-warmup 5 --step-fallback 60 \
        --opponents balanced balanced_extreme_opening adaptive forager defender \
        --checkpoint-dir runs/pilot_s0 --checkpoint-every 50 \
        --eval-every 100 --eval-seeds 8 --log runs/pilot_s0/metrics.jsonl --seed0 0

Full run:
    ... run_hybrid.py --updates 3000 --episodes-per-update 4 --device cuda \
        --forward-bias 0.5 --anti-idle 0.10 --critic-warmup 10 --step-fallback 60 \
        --opponents balanced balanced_extreme_opening adaptive forager defender lean_rush \
        --checkpoint-dir runs/full --checkpoint-every 100 \
        --eval-every 250 --eval-seeds 16 --log runs/full/metrics.jsonl \
        --export-onnx runs/full/rl_actor.onnx --seed0 0

Acceptance eval of a checkpoint (deploy-faithful argmax via ONNX):
    ... run_hybrid.py --eval-only --checkpoint runs/full/actor_3000.pt \
        --eval-seeds 64 --step-fallback 60
"""
import argparse
import copy
import json
import os
import tempfile
from dataclasses import replace


def _make_eval_fn(args, opener_name):
    """Build an eval callback: export the actor to ONNX (deploy-faithful argmax)
    and run the paired-continuation A/B over `eval_seeds` seeds."""
    from export_onnx import export_actor
    from hybrid_controller import OnnxActorRuntime
    from hybrid_eval import (HybridAgent, paired_continuation_eval,
                             paired_grid_eval_parallel)
    from scripted.handover import HandoverTrigger
    seeds = list(range(args.eval_seeds))

    def eval_fn(update, actor):
        a_cpu = copy.deepcopy(actor).cpu().eval()        # avoid CUDA/CPU export mismatch
        onnx_path = os.path.join(tempfile.mkdtemp(), f"eval_{update}.onnx")
        export_actor(a_cpu, onnx_path)
        agent = HybridAgent(OnnxActorRuntime.from_path(onnx_path),
                            trigger=HandoverTrigger(
                                min_destroyed_enemy_bases=args.min_destroyed_bases,
                                step_fallback=args.step_fallback))
        if args.eval_configs_per_slot > 0:
            # structured aggregate: 6 slots x N configs x A/B, parallelized.
            # Arm B rebuilt in workers from the exported ONNX (deploy-faithful).
            out = paired_grid_eval_parallel(
                ("onnx", onnx_path, args.min_destroyed_bases, args.step_fallback),
                opener_name=opener_name, opponent_bank=args.opponents,
                n_slots=6, configs_per_slot=args.eval_configs_per_slot,
                num_workers=args.rollout_workers)
            per_slot = " ".join(f"s{k}={v:+.0f}" for k, v in sorted(out["per_slot_mean"].items()))
            print(f"[eval @ {update}] grid n={out['n']} Δ mean={out['mean']:.2f} "
                  f"CI[{out['ci_lo']:.2f},{out['ci_hi']:.2f}]  "
                  f"(A={out['mean_a']:.1f} B={out['mean_b']:.1f})  per-slot Δ: {per_slot}",
                  flush=True)
        else:
            out = paired_continuation_eval(
                agent, seeds, opener_name=opener_name,
                randomize_slot=args.randomize_slot,
                opponent_bank=(args.opponents if args.randomize_slot else None))
            print(f"[eval @ {update}] paired Δ mean={out['mean']:.3f} "
                  f"CI[{out['ci_lo']:.3f},{out['ci_hi']:.3f}]  "
                  f"(A={out['mean_a']:.2f} B={out['mean_b']:.2f})", flush=True)
        return {k: out[k] for k in ("n", "mean", "ci_lo", "ci_hi", "mean_a", "mean_b")}

    return eval_fn


def _build_cfg(args):
    from hybrid_ppo import HybridPPOConfig
    cfg = HybridPPOConfig()
    over = {}
    if args.lr is not None:
        over["learning_rate"] = args.lr
    if args.update_epochs is not None:
        over["update_epochs"] = args.update_epochs
    if args.clip is not None:
        over["clip_coef"] = args.clip
    if args.target_kl is not None:
        over["target_kl"] = args.target_kl
    if args.ent_coef is not None:
        over["ent_coef"] = args.ent_coef
    if args.num_minibatches is not None:
        over["num_minibatches"] = args.num_minibatches
    return replace(cfg, **over) if over else cfg


def main():
    ap = argparse.ArgumentParser(description="Hybrid post-opener PPO runner")
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--episodes-per-update", type=int, default=4)
    ap.add_argument("--rollout-workers", type=int, default=1,
                    help="parallel rollout processes; effective parallelism is "
                         "min(rollout-workers, episodes-per-update), so raise "
                         "--episodes-per-update to feed more than that many workers")
    ap.add_argument("--randomize-slot", action="store_true",
                    help="learner plays a random slot 0-5 each episode and (in eval) "
                         "opponents are drawn from --opponents; board stays seed-88")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="freeze the actor every N updates into the self-play pool")
    ap.add_argument("--selfplay-prob", type=float, default=0.0,
                    help="per opponent slot, probability of a frozen self-play "
                         "opponent (vs a scripted one) once the pool is non-empty")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--learner-slots", nargs="+", default=["agent_0"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    # schedule / shaping / handover
    ap.add_argument("--forward-bias", type=float, default=0.0)
    ap.add_argument("--anti-idle", type=float, default=0.0)
    ap.add_argument("--critic-warmup", type=int, default=0)
    ap.add_argument("--step-fallback", type=int, default=100)
    ap.add_argument("--min-destroyed-bases", type=int, default=3,
                    help="handover after this many enemy bases destroyed OR "
                         "--step-fallback, whichever comes first")
    ap.add_argument("--opponents", nargs="*", default=None,
                    help="scripted strategy names; omit for RandomAgent opponents")
    ap.add_argument("--opener", default="balanced_extreme_opening")
    # PPO knobs (override HybridPPOConfig defaults)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--update-epochs", type=int, default=None)
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--target-kl", type=float, default=None)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--num-minibatches", type=int, default=None)
    # io / eval
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-seeds", type=int, default=16)
    ap.add_argument("--eval-configs-per-slot", type=int, default=0,
                    help="structured grid eval: 6 slots x this many opponent "
                         "configs x A/B (e.g. 30 -> 360 episodes). 0 = legacy "
                         "seed-based eval over --eval-seeds")
    ap.add_argument("--log", default=None, help="jsonl metrics path (appended)")
    ap.add_argument("--export-onnx", default=None, help="final actor .onnx path")
    # eval-only mode
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="actor .pt for --eval-only")
    args = ap.parse_args()

    if args.eval_only:
        from policy import SymbolicTransformerActor
        if not args.checkpoint:
            ap.error("--eval-only requires --checkpoint")
        actor = SymbolicTransformerActor.from_checkpoint(args.checkpoint)
        result = _make_eval_fn(args, args.opener)(0, actor)
        print(json.dumps(result))
        return

    from scripted.handover import HandoverTrigger
    from train_hybrid import train_hybrid
    cfg = _build_cfg(args)
    eval_fn = _make_eval_fn(args, args.opener) if args.eval_every else None
    actor, critic, hist = train_hybrid(
        total_updates=args.updates, episodes_per_update=args.episodes_per_update,
        learner_slots=tuple(args.learner_slots), seed0=args.seed0, cfg=cfg,
        forward_bias_init=args.forward_bias, anti_idle_penalty=args.anti_idle,
        critic_warmup=args.critic_warmup,
        trigger=HandoverTrigger(
            min_destroyed_enemy_bases=args.min_destroyed_bases,
            step_fallback=args.step_fallback),
        opponent_names=args.opponents, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads, device=args.device,
        checkpoint_dir=args.checkpoint_dir, checkpoint_every=args.checkpoint_every,
        log_path=args.log, eval_every=args.eval_every, eval_fn=eval_fn,
        rollout_workers=args.rollout_workers, randomize_slot=args.randomize_slot,
        snapshot_every=args.snapshot_every, selfplay_prob=args.selfplay_prob)

    if args.export_onnx:
        from export_onnx import export_actor
        a_cpu = copy.deepcopy(actor).cpu().eval()
        export_actor(a_cpu, args.export_onnx)
        print(f"exported final actor -> {args.export_onnx}", flush=True)

    print("[final acceptance eval]", flush=True)
    _make_eval_fn(args, args.opener)(args.updates, actor)


if __name__ == "__main__":
    main()
