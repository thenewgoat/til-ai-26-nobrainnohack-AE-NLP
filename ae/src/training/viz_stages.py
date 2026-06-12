"""Qualitative viz with the hybrid on DIFFERENT bases + an explicit stage banner.

For each focus slot (different starting base), renders one episode to MP4 with a
top banner showing, every step: which slot is the hybrid, the step index, the
current STAGE (OPENER / RL / ESCAPE / GATE), the handover step once it fires, and
the running score. Also prints a per-episode stage timeline for tuning the
handover trigger.

Run:
  cd ae && PYTHONPATH=src:src/training ../til-26-ae/.venv/bin/python src/training/viz_stages.py
"""
import glob
import os
import random
from collections import Counter

import numpy as np
import imageio
from PIL import Image, ImageDraw
from til_environment import bomberman_env
from til_environment.config import default_config
from evaluate import ScriptedAgent
from hybrid_controller import ActorRuntime, HybridController
from policy import SymbolicTransformerActor
from scripted.handover import HandoverTrigger

OUT = "runs/viz_stages"
# newest save across the active run (snapshots + checkpoints), by mtime
_SAVES = glob.glob("runs/full_sp/snap_*.pt") + glob.glob("runs/full_sp/actor_*.pt") \
    + glob.glob("runs/pilot_rand_s0/actor_*.pt")
CKPT = max(_SAVES, key=os.path.getmtime)
BANK = ["balanced", "balanced_extreme_opening", "adaptive", "forager", "defender"]
SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]
FOCUS_SLOTS = [0, 2, 4]          # different starting bases to visualise
FALLBACK = 100                   # min_destroyed_enemy_bases defaults to 3

# stage -> (label, banner RGB)
STAGE = {"opener": ("OPENER (scripted)", (90, 90, 90)),
         "rl_layer": ("RL (network)", (30, 140, 40)),
         "forced_escape": ("ESCAPE (scripted floor)", (170, 30, 30)),
         "gate": ("GATE (scripted override)", (200, 120, 20))}


def _stage_key(source):
    if source is None:
        return "opener"
    return "gate" if source.startswith("gate") else source


def _banner(frame, lines, rgb):
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    h = 14 * len(lines) + 8
    d.rectangle([0, 0, img.width, h], fill=rgb)
    for i, ln in enumerate(lines):
        d.text((6, 4 + 14 * i), ln, fill=(255, 255, 255))
    return np.asarray(img)


def render_focus(actor, focus_idx, seed):
    cfg = default_config(); cfg.env.novice = True
    cfg.env.render_mode = "rgb_array"
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    fslot = SLOTS[focus_idx]
    ctrl = HybridController(ActorRuntime(actor), HandoverTrigger(step_fallback=FALLBACK))
    r = random.Random(seed)
    opp_names = [r.choice(BANK) for _ in range(5)]
    opps, oi = {}, 0
    for s in SLOTS:
        if s == fslot:
            continue
        opps[s] = ScriptedAgent(opp_names[oi]); oi += 1
    by = {fslot: ctrl, **opps}

    random.seed(seed); env.reset(seed=seed)
    for a in opps.values():          # controller is fresh per call; only opponents reset
        a.reset()
    legend = f"agent_{focus_idx}=HYBRID  |  opp: " + ",".join(opp_names)

    frames, stages = [], []
    handover, step, score, cur_stage = None, 0, 0.0, "opener"
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if slot == fslot:
            score += float(reward)
            step = int(np.asarray(obs["step"]).flat[0])
        if term or trunc:
            env.step(None); continue
        if slot == fslot:
            act, dec = ctrl.step(obs)
            cur_stage = _stage_key(dec.source if dec is not None else None)
            if dec is not None and handover is None:
                handover = step
            stages.append(cur_stage)
            env.step(act)
            label, rgb = STAGE[cur_stage]
            frame = env.render()
            if frame is not None:
                frames.append(_banner(
                    np.asarray(frame, np.uint8),
                    [f"step {step:3d}  STAGE: {label}   handover@{handover if handover is not None else '-'}   score {score:+.0f}",
                     legend], rgb))
        else:
            env.step(by[slot].action(obs))
    env.close()
    imageio.mimwrite(f"{OUT}/base{focus_idx}_seed{seed}.mp4", frames, fps=6, codec="libx264")
    return fslot, handover, score, Counter(stages)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"actor: {CKPT}")
    actor = SymbolicTransformerActor.from_checkpoint(CKPT).eval()
    print(f"{'base':>10} {'handover':>9} {'score':>7}   stage counts (post-handover incl.)")
    for fi in FOCUS_SLOTS:
        fslot, hb, score, counts = render_focus(actor, fi, seed=1)
        cc = " ".join(f"{STAGE[k][0].split()[0]}={counts[k]}" for k in counts)
        print(f"{fslot:>10} {str(hb):>9} {score:>7.0f}   {cc}")
        print(f"           -> {OUT}/base{fi}_seed1.mp4")


if __name__ == "__main__":
    main()
