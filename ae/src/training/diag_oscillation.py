"""Why does the post-handover RL actor oscillate in 2 cells?

Runs the hybrid (torch actor) on the fixed novice board and logs, for each
post-handover tick: agent position, the actor's masked action probabilities,
argmax vs sampled action, executed action, and source. Then reports whether the
policy is (a) a deterministic 2-cycle, (b) unresponsive to the observation
(near-constant probs), and (c) whether the ONNX deploy argmax matches torch.

Run:
  cd ae && PYTHONPATH=src:src/training ../til-26-ae/.venv/bin/python src/training/diag_oscillation.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from til_environment import bomberman_env
from til_environment.config import default_config
from til_environment.entities.dynamic import Agent
from til_environment.entities.base import EntityStatus
from evaluate import ScriptedAgent
from hybrid_controller import ActorRuntime, HybridController, OnnxActorRuntime
from policy import SymbolicTransformerActor
from scripted.handover import HandoverTrigger

CKPT = "runs/pilot_s0_orig_failed/actor_400.pt"
ONNX = "runs/pilot_s0_orig_failed/rl_actor.onnx"
A = ["FWD", "BACK", "LEFT", "RIGHT", "STAY", "BOMB"]


def main():
    actor = SymbolicTransformerActor.from_checkpoint(CKPT).eval()
    onnx_rt = OnnxActorRuntime.from_path(ONNX)

    cfg = default_config(); cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    dyn = env.unwrapped.dynamics
    ctrl = HybridController(ActorRuntime(actor), HandoverTrigger(step_fallback=60))
    opps = [ScriptedAgent("balanced") for _ in range(5)]
    by = {"agent_0": ctrl, **{f"agent_{i+1}": opps[i] for i in range(5)}}
    env.reset(seed=0)
    [o.reset() for o in opps]

    def my_pos():
        for a in dyn.registry.query().type(Agent).status(EntityStatus.ACTIVE).all():
            if a.entity_id == "agent_0":
                return tuple(int(x) for x in a.position)
        return None

    ticks = []
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None); continue
        if slot == "agent_0":
            action, dec = ctrl.step(obs)
            if dec is not None:                      # post-handover
                g, bf, ra, rb, sc = [torch.from_numpy(np.asarray(x)).unsqueeze(0).float()
                                     for x in dec.features]
                m = torch.from_numpy(np.asarray(dec.action_mask, bool)).reshape(1, -1)
                with torch.no_grad():
                    lg = actor.forward(g, bf, ra, rb, sc)
                    lg = torch.where(m, lg, torch.full_like(lg, -1e8))
                    probs = F.softmax(lg, -1)[0].numpy()
                onnx_a = onnx_rt.act(dec.features, np.asarray(dec.action_mask, bool)) \
                    if hasattr(onnx_rt, "act") else -1
                ticks.append(dict(pos=my_pos(), probs=probs,
                                  argmax=int(probs.argmax()),
                                  executed=dec.executed_action, onnx=onnx_a,
                                  src=dec.source))
            env.step(action)
        else:
            env.step(by[slot].action(obs))
    env.close()

    print(f"post-handover ticks: {len(ticks)}")
    print(f"distinct positions visited: {len(set(t['pos'] for t in ticks))}")
    print(f"distinct executed actions : {sorted(set(t['executed'] for t in ticks))} "
          f"-> {[A[a] for a in sorted(set(t['executed'] for t in ticks))]}")
    print("\nfirst 30 post-handover ticks:")
    print(f"{'pos':>10} {'exec':>5} {'argmax':>6} {'onnx':>5}  probs[FWD BACK LEFT RIGHT STAY BOMB]")
    for t in ticks[:30]:
        pr = " ".join(f"{p:.2f}" for p in t["probs"])
        print(f"{str(t['pos']):>10} {A[t['executed']]:>5} {A[t['argmax']]:>6} "
              f"{(A[t['onnx']] if t['onnx']>=0 else '-'):>5}  [{pr}]")

    # responsiveness: std of each action's prob across ticks (≈0 => ignores obs)
    P = np.array([t["probs"] for t in ticks])
    print(f"\nper-action prob std across ticks: {np.round(P.std(0), 3)}  (≈0 => policy ignores the observation)")
    onnx_match = np.mean([t["onnx"] == t["argmax"] for t in ticks if t["onnx"] >= 0]) if ticks else 0
    print(f"onnx-argmax == torch-argmax: {onnx_match*100:.0f}%")


if __name__ == "__main__":
    main()
