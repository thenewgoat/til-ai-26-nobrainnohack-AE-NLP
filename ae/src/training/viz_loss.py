"""Visualize WHERE the hybrid loses ~300 pts vs scripted on the fixed novice board.

Produces, in runs/viz/:
  arm_A_scripted.mp4  — full scripted agent_0 (scores ~+219)
  arm_B_hybrid.mp4    — scripted opener + RL after handover (scores ~-101)
  reward_breakdown.png — cumulative agent_0 reward over the game (A vs B) +
                         per-component totals (what earns / loses the points)
and prints a per-component table + the biggest single point-loss events.

Run:
  cd ae && PYTHONPATH=src:src/training ../til-26-ae/.venv/bin/python src/training/viz_loss.py
"""
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from til_environment import bomberman_env
from til_environment.config import default_config
from evaluate import ScriptedAgent
from hybrid_controller import HybridController, OnnxActorRuntime
from hybrid_eval import HybridAgent
from scripted.handover import HandoverTrigger
from visualize import render_episode

OUT = "runs/viz"
ONNX = "runs/pilot_s0_orig_failed/rl_actor.onnx"
OPENER = "balanced_extreme_opening"
STEP_FALLBACK = 60


def _mk_A():
    return ScriptedAgent(OPENER)


def _mk_B():
    return HybridAgent(OnnxActorRuntime.from_path(ONNX),
                       trigger=HandoverTrigger(step_fallback=STEP_FALLBACK))


def capture(make_a0):
    """Run one fixed-board episode; capture per-event reward for agent_0 + handover."""
    cfg = default_config(); cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    dyn = env.unwrapped.dynamics
    orig = dyn.rewards.award
    events = []                         # (step, event, value)
    st = {"step": 0}

    def wrapped(recipient_id, event, multiplier=1.0):
        v = orig(recipient_id, event, multiplier)
        if recipient_id == "agent_0" and v != 0.0:
            events.append((st["step"], event, float(v)))
        return v
    dyn.rewards.award = wrapped

    a0 = make_a0()
    opps = [ScriptedAgent("balanced") for _ in range(5)]
    by = {"agent_0": a0, **{f"agent_{i+1}": opps[i] for i in range(5)}}
    env.reset(seed=0)                    # novice -> seed 88 regardless
    for a in by.values():
        a.reset()

    total, handover = 0.0, None
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if slot == "agent_0":
            total += float(reward)
            st["step"] = int(np.asarray(obs["step"]).flat[0])
        if term or trunc:
            env.step(None); continue
        if slot == "agent_0" and isinstance(a0, HybridAgent):
            # peek the controller to learn the handover step (when RL takes over)
            act, dec = a0.controller.step(obs)
            if dec is not None and handover is None:
                handover = st["step"]
            env.step(act)
        else:
            env.step(by[slot].action(obs))
    env.close()
    return total, events, handover


def summarize(tag, total, events):
    by_comp = defaultdict(float)
    for _, e, v in events:
        by_comp[e] += v
    print(f"\n=== {tag}  whole-game agent_0 total = {total:+.1f} ===")
    for e, v in sorted(by_comp.items(), key=lambda kv: kv[1]):
        n = sum(1 for _, ev, _ in events if ev == e)
        print(f"  {e:<22} {v:+8.1f}   ({n} events)")
    worst = sorted(events, key=lambda x: x[2])[:6]
    print("  biggest single losses:", [(s, e, round(v, 1)) for s, e, v in worst])
    return by_comp


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- reward capture (deterministic board) ----
    tA, evA, _ = capture(_mk_A)
    tB, evB, hb = capture(_mk_B)
    cA = summarize("ARM A  (full scripted)", tA, evA)
    cB = summarize("ARM B  (opener + RL)", tB, evB)
    print(f"\nhandover step (arm B): {hb}")

    # ---- cumulative-reward timeline + component bars ----
    def cum(events):
        m = defaultdict(float)
        for s, _, v in events:
            m[s] += v
        xs = sorted(m)
        ys, run = [], 0.0
        for x in xs:
            run += m[x]; ys.append(run)
        return xs, ys

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    xA, yA = cum(evA); xB, yB = cum(evB)
    ax1.plot(xA, yA, label=f"A scripted ({tA:+.0f})", color="green", lw=2)
    ax1.plot(xB, yB, label=f"B opener+RL ({tB:+.0f})", color="red", lw=2)
    if hb is not None:
        ax1.axvline(hb, ls="--", color="gray", label=f"handover (step {hb})")
    ax1.set_xlabel("env step"); ax1.set_ylabel("cumulative agent_0 reward")
    ax1.set_title("Where the points go"); ax1.legend(); ax1.grid(alpha=0.3)

    comps = sorted(set(cA) | set(cB))
    x = np.arange(len(comps)); w = 0.4
    ax2.bar(x - w/2, [cA.get(c, 0) for c in comps], w, label="A scripted", color="green")
    ax2.bar(x + w/2, [cB.get(c, 0) for c in comps], w, label="B opener+RL", color="red")
    ax2.set_xticks(x); ax2.set_xticklabels(comps, rotation=30, ha="right")
    ax2.set_ylabel("total reward"); ax2.set_title("Reward by component"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/reward_breakdown.png", dpi=110)
    print(f"\nsaved {OUT}/reward_breakdown.png")

    # ---- MP4 replays ----
    rA = render_episode({"agent_0": (_mk_A(), "scripted"),
                         **{f"agent_{i+1}": (ScriptedAgent("balanced"), f"balanced{i+1}") for i in range(5)}},
                        f"{OUT}/arm_A_scripted.mp4", fps=6)
    rB = render_episode({"agent_0": (_mk_B(), "hybrid(RL)"),
                         **{f"agent_{i+1}": (ScriptedAgent("balanced"), f"balanced{i+1}") for i in range(5)}},
                        f"{OUT}/arm_B_hybrid.mp4", fps=6)
    print(f"saved {rA['path']}  (agent_0 score {rA['scores']['agent_0']:+.0f}, {rA['steps']} steps)")
    print(f"saved {rB['path']}  (agent_0 score {rB['scores']['agent_0']:+.0f}, {rB['steps']} steps)")


if __name__ == "__main__":
    main()
