"""Hybrid evaluation: a frozen-hybrid opponent/eval agent, the paired-continuation
A/B acceptance harness (the spec's north-star metric), and intervention
diagnostics. Reuses evaluate.evaluate_policy unchanged. train_selfplay.py and
evaluate.py are untouched.
"""
import numpy as np

from features import FeatureBuilder
from hybrid_controller import ActorRuntime, HybridController
from scripted.handover import HandoverTrigger
from scripted.strategies import STRATEGIES


class HybridAgent:
    """A frozen trained hybrid (scripted opener -> RL post-opener) as a self-play
    opponent or eval agent. Runtime-agnostic: pass a torch `ActorRuntime` (samples)
    or an `OnnxActorRuntime` (deterministic argmax — use this for acceptance eval).
    A raw `SymbolicTransformerActor` is wrapped in a torch `ActorRuntime`."""

    def __init__(self, actor, opener=None, trigger=None, post_params=None,
                 name="hybrid"):
        self._runtime = actor if hasattr(actor, "query") else ActorRuntime(actor)
        self._opener = opener
        self._trigger = trigger
        self._post = post_params
        self.name = name
        self.reset()

    def reset(self):
        self.controller = HybridController(
            self._runtime, self._trigger or HandoverTrigger(),
            opener=self._opener, post_params=self._post,
            feature_builder=FeatureBuilder(), forward_bias=0.0)

    def action(self, observation):
        action, _decision = self.controller.step(observation)
        return int(action)


def intervention_rates(buf):
    """Summarize a HybridRolloutBuffer's controller behavior (for diagnostics):
    how often the actor was queried vs forced-escaped, and how often a gate
    overrode the actor's proposal."""
    n = int(buf.size)
    if n == 0:
        return {"n": 0, "actor_query_rate": 0.0, "forced_escape_rate": 0.0,
                "gate_override_rate": 0.0, "proposal_executed_disagreement": 0.0}
    aq = np.asarray(buf.actor_queried, bool)
    disagree = np.asarray(buf.proposed_actions) != np.asarray(buf.executed_actions)
    return {
        "n": n,
        "actor_query_rate": float(aq.mean()),
        "forced_escape_rate": float((~aq).mean()),
        "gate_override_rate": float((aq & disagree).mean()),
        "proposal_executed_disagreement": float(disagree.mean()),
    }


def summarize_paired_deltas(deltas, n_boot=2000, ci=0.95, seed=0):
    """Mean + bootstrap CI over paired (B - A) deltas. CI excluding 0 on the low
    side is the acceptance signal."""
    d = np.asarray(deltas, np.float64)
    if d.size == 0:
        return {"n": 0, "mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=d.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"n": int(d.size), "mean": float(d.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_continuation_eval(hybrid_agent, seeds,
                             opener_name="balanced_extreme_opening",
                             opponents=None, randomize_slot=False,
                             opponent_bank=None):
    """Paired A/B acceptance (spec north star). Arm A: the scripted opener runs the
    whole episode; arm B: the SAME opener until handover, then `hybrid_agent`. Same
    seeds + opponents, so pre-handover is identical and the per-seed env-return
    delta (B - A) isolates the post-handover difference.

    `randomize_slot`/`opponent_bank` are forwarded to evaluate_policy; their setup
    is seed-derived, so both arms get the SAME slot + opponents per seed (pairing
    holds) while the benchmark gains spawn-slot + opponent-mix variance.

    For deploy-faithful acceptance, build `hybrid_agent` with an OnnxActorRuntime
    (argmax). Returns {deltas, per_seed_a, per_seed_b, mean_a, mean_b, **summary}.
    """
    from evaluate import ScriptedAgent, evaluate_policy   # local: avoid import cycle
    a_agent = ScriptedAgent(opener_name)
    if opponents is None and not opponent_bank:
        opponents = [ScriptedAgent("balanced") for _ in range(5)]
    kw = dict(randomize_slot=randomize_slot, opponent_bank=opponent_bank)
    res_a = evaluate_policy(a_agent, opponents, seeds, **kw)
    res_b = evaluate_policy(hybrid_agent, opponents, seeds, **kw)
    deltas = [float(b) - float(a)
              for a, b in zip(res_a.per_seed_scores, res_b.per_seed_scores)]
    out = {"deltas": deltas,
           "per_seed_a": list(res_a.per_seed_scores),
           "per_seed_b": list(res_b.per_seed_scores),
           "mean_a": float(res_a.mean_score),
           "mean_b": float(res_b.mean_score)}
    out.update(summarize_paired_deltas(deltas))
    return out


def paired_grid_eval(hybrid_agent, opener_name="balanced_extreme_opening",
                     opponent_bank=None, n_slots=6, configs_per_slot=30, base_seed=0):
    """Structured aggregate eval over a (starting-slot x opponent-config) grid.

    For each of `n_slots` slots and `configs_per_slot` opponent draws, run arm A
    (full scripted opener) and arm B (`hybrid_agent`) in the SAME slot vs the SAME
    seed-derived opponents (paired). Total episodes = 2 * n_slots * configs_per_slot.
    Returns the aggregate {n, deltas, mean, ci_lo, ci_hi, mean_a, mean_b} plus
    `per_slot_mean` (mean delta by starting slot) for per-spawn diagnosis."""
    from evaluate import ScriptedAgent, evaluate_policy   # local: avoid import cycle
    if opponent_bank is None:
        opponent_bank = ["balanced"]
    setups = [(s, base_seed + s * 100000 + c)
              for s in range(n_slots) for c in range(configs_per_slot)]
    kw = dict(randomize_slot=False, opponent_bank=opponent_bank, setups=setups)
    res_a = evaluate_policy(ScriptedAgent(opener_name), [], None, **kw)
    res_b = evaluate_policy(hybrid_agent, [], None, **kw)
    deltas = [float(b) - float(a)
              for a, b in zip(res_a.per_seed_scores, res_b.per_seed_scores)]
    per_slot = {}
    for (s, _), d in zip(setups, deltas):
        per_slot.setdefault(s, []).append(d)
    out = {"n": len(deltas), "deltas": deltas,
           "mean_a": float(res_a.mean_score), "mean_b": float(res_b.mean_score),
           "per_slot_mean": {s: float(np.mean(v)) for s, v in per_slot.items()}}
    out.update(summarize_paired_deltas(deltas))
    return out


def _grid_eval_worker(spec, setups_chunk, opponent_bank):
    """Pool worker: build the arm's agent from `spec` and score its setups chunk.
    spec is ("scripted", name) or ("onnx", path, min_bases, step_fallback)."""
    import torch
    torch.set_num_threads(1)
    from evaluate import ScriptedAgent, evaluate_policy
    if spec[0] == "scripted":
        agent = ScriptedAgent(spec[1])
    else:                                    # ("onnx", path, min_bases, step_fallback)
        from hybrid_controller import OnnxActorRuntime
        agent = HybridAgent(OnnxActorRuntime.from_path(spec[1]),
                            trigger=HandoverTrigger(min_destroyed_enemy_bases=spec[2],
                                                    step_fallback=spec[3]))
    return evaluate_policy(agent, [], None, opponent_bank=opponent_bank,
                           setups=setups_chunk).per_seed_scores


def _run_grid(pool, spec, setups, opponent_bank, n_workers):
    n = max(1, min(n_workers, len(setups)))
    size = (len(setups) + n - 1) // n
    chunks = [setups[i:i + size] for i in range(0, len(setups), size)]   # contiguous
    results = pool.starmap(_grid_eval_worker,
                           [(spec, ch, opponent_bank) for ch in chunks])
    return [s for chunk in results for s in chunk]                       # flatten in order


def paired_grid_eval_parallel(arm_b_spec, opener_name="balanced_extreme_opening",
                              opponent_bank=None, n_slots=6, configs_per_slot=10,
                              base_seed=0, num_workers=4):
    """Parallel `paired_grid_eval`. arm A = scripted opener; arm B is rebuilt in
    workers from `arm_b_spec` (e.g. ("onnx", path, min_bases, step_fallback)).
    Episodes are independent and split across a spawn pool. Same setups/results as
    the serial version, just faster. Returns the same dict (+ per_slot_mean)."""
    import multiprocessing
    if opponent_bank is None:
        opponent_bank = ["balanced"]
    setups = [(s, base_seed + s * 100000 + c)
              for s in range(n_slots) for c in range(configs_per_slot)]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(max(1, min(num_workers, len(setups)))) as pool:
        a = _run_grid(pool, ("scripted", opener_name), setups, opponent_bank, num_workers)
        b = _run_grid(pool, arm_b_spec, setups, opponent_bank, num_workers)
    deltas = [float(bi) - float(ai) for ai, bi in zip(a, b)]
    per_slot = {}
    for (s, _), d in zip(setups, deltas):
        per_slot.setdefault(s, []).append(d)
    out = {"n": len(deltas), "deltas": deltas,
           "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
           "per_slot_mean": {s: float(np.mean(v)) for s, v in per_slot.items()}}
    out.update(summarize_paired_deltas(deltas))
    return out
