"""Qualitative eval of the trained hybrid agent across varied matchups.

Renders N episodes (hybrid in agent_0 so its belief panel is shown; opponents
drawn from the bank per seed) to MP4, and prints each episode's score + reward-by-
component breakdown + handover step. Board stays novice/seed-88; variety is the
opponent draw (matches the randomized benchmark's opponent axis).

Run:
  cd ae && PYTHONPATH=src:src/training ../til-26-ae/.venv/bin/python src/training/viz_qual.py
"""
import os
import random
from collections import defaultdict

import numpy as np

from til_environment import bomberman_env
from til_environment.config import default_config
from evaluate import ScriptedAgent
from hybrid_controller import OnnxActorRuntime
from hybrid_eval import HybridAgent
from scripted.handover import HandoverTrigger
from visualize import render_episode

OUT = "runs/viz_qual"
ONNX = "runs/pilot_rand_s0/rl_actor.onnx"
BANK = ["balanced", "balanced_extreme_opening", "adaptive", "forager", "defender"]
SEEDS = [0, 1, 2, 3]
FALLBACK = 60


def _mk_hybrid():
    return HybridAgent(OnnxActorRuntime.from_path(ONNX),
                       trigger=HandoverTrigger(step_fallback=FALLBACK))


def _opps(seed):
    r = random.Random(seed)
    return [r.choice(BANK) for _ in range(5)]


def capture(seed, opp_names):
    """Reward-by-component for agent_0 (the hybrid) + handover step on this matchup."""
    cfg = default_config(); cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    dyn = env.unwrapped.dynamics
    orig = dyn.rewards.award
    events = []

    def wrapped(rid, ev, multiplier=1.0):
        v = orig(rid, ev, multiplier)
        if rid == "agent_0" and v != 0.0:
            events.append((ev, float(v)))
        return v
    dyn.rewards.award = wrapped

    h = _mk_hybrid()
    by = {"agent_0": h, **{f"agent_{i+1}": ScriptedAgent(opp_names[i]) for i in range(5)}}
    random.seed(seed); env.reset(seed=seed)
    for a in by.values():
        a.reset()
    total, handover, step = 0.0, None, 0
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if slot == "agent_0":
            total += float(reward)
            step = int(np.asarray(obs["step"]).flat[0])
        if term or trunc:
            env.step(None); continue
        if slot == "agent_0":
            act, dec = h.controller.step(obs)
            if dec is not None and handover is None:
                handover = step
            env.step(act)
        else:
            env.step(by[slot].action(obs))
    env.close()
    by_comp = defaultdict(float)
    for ev, v in events:
        by_comp[ev] += v
    return total, dict(by_comp), handover


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'seed':>4} {'opponents':<46} {'score':>7} {'hand':>5}  reward by component")
    for seed in SEEDS:
        opp = _opps(seed)
        total, comp, hb = capture(seed, opp)
        comp_str = " ".join(f"{k}={v:+.0f}" for k, v in sorted(comp.items(), key=lambda kv: kv[1]))
        print(f"{seed:>4} {','.join(opp):<46} {total:>7.0f} {str(hb):>5}  {comp_str}")
        render_episode(
            {"agent_0": (_mk_hybrid(), "HYBRID(RL)"),
             **{f"agent_{i+1}": (ScriptedAgent(opp[i]), opp[i][:8]) for i in range(5)}},
            f"{OUT}/ep_seed{seed}.mp4", fps=6, seed=seed)
        print(f"      -> {OUT}/ep_seed{seed}.mp4")


if __name__ == "__main__":
    main()
