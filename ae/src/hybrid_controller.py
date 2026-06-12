"""Runtime composition of the hybrid post-opener agent: the scripted opener, the
forced-escape floor, the RL actor, and the post-decision gates, behind one
per-slot `HybridController`. The actor and FeatureBuilder are runtime state — they
live here, not in `StrategyParams`/`Belief`.
"""
from dataclasses import dataclass

import numpy as np

from features import FeatureBuilder
from scripted.danger import DangerMap
from scripted.decide import act, record_final_action
from scripted.escape import escape_selector, must_force_escape
from scripted.gates import body_block_resolve, strike_gate
from scripted.geometry import FORWARD
from scripted.pathfind import build_planner
from scripted.strategies import STRATEGIES, StrategyParams


@dataclass
class RLDecision:
    """Per-tick record of a post-handover decision, consumed by the trainer
    (Plan 5). The PPO ratio uses `proposed_action`/`old_proposal_logp`; rewards
    and critic targets use `executed_action`. No critic value here — the
    CentralizedCritic is trainer-only and consumes privileged global state."""
    features: tuple                  # the 5 actor-input arrays the actor saw
    action_mask: np.ndarray          # exact mask at collection (for PPO re-eval)
    proposed_action: int | None      # actor's sample (None on forced-escape ticks)
    executed_action: int             # action actually sent to the env
    old_proposal_logp: float         # log mu(proposed) at collection
    entropy: float
    actor_queried: bool              # True iff the actor was queried this tick
    intervened: bool                 # True iff a gate/escape changed the proposal
    source: str                      # "rl_layer" | "forced_escape" | "gate:<name>"
    forward_bias: float


class ActorRuntime:
    """Torch actor wrapper for TRAINING/eval — samples a masked action and applies
    the forward bias to the PRE-mask logits. Torch is imported lazily so the
    serving path (`OnnxActorRuntime`, torch-free container) can import this module
    without torch installed. No critic / value head."""

    def __init__(self, actor, device="cpu"):
        self.actor = actor.to(device).eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from policy import SymbolicTransformerActor
        return cls(SymbolicTransformerActor.from_checkpoint(path), device)

    def query(self, features, mask, forward_bias=0.0):
        import torch
        grid, base_feats, raw_agent, raw_base, scalar = features
        mask_arr = np.asarray(mask).reshape(-1)

        def bt(x, dtype=torch.float32):
            return torch.as_tensor(np.asarray(x), dtype=dtype,
                                   device=self.device).unsqueeze(0)

        logit_bias = None
        if forward_bias:
            logit_bias = torch.zeros(mask_arr.shape[0], device=self.device)
            logit_bias[FORWARD] = float(forward_bias)
        with torch.no_grad():
            action, logp, entropy = self.actor.act(
                bt(grid), bt(base_feats), bt(raw_agent), bt(raw_base), bt(scalar),
                bt(mask_arr, torch.bool), logit_bias=logit_bias)
        return int(action.item()), float(logp.item()), float(entropy.item())


class OnnxActorRuntime:
    """ONNX actor wrapper for SERVING — deterministic masked ARGMAX. Uses
    onnxruntime + numpy only (no torch). Matches `ActorRuntime.query`'s signature
    so `HybridController` works with either backend. At serving `forward_bias=0`."""

    def __init__(self, session):
        self.session = session

    @classmethod
    def from_path(cls, onnx_path):
        import onnxruntime as ort
        return cls(ort.InferenceSession(onnx_path))

    def query(self, features, mask, forward_bias=0.0):
        grid, base_feats, raw_agent, raw_base, scalar = features
        logits = self.session.run(["logits"], {
            "grid": np.asarray(grid, np.float32)[None],
            "base_feats": np.asarray(base_feats, np.float32)[None],
            "raw_agent": np.asarray(raw_agent, np.float32)[None],
            "raw_base": np.asarray(raw_base, np.float32)[None],
            "scalar": np.asarray(scalar, np.float32)[None],
        })[0][0]
        if forward_bias:
            logits = logits.copy()
            logits[FORWARD] += float(forward_bias)
        m = np.asarray(mask, bool).reshape(-1)
        masked = np.where(m, logits, -1e8)
        action = int(np.argmax(masked))
        z = masked - masked.max()
        p = np.exp(z)
        p = p / p.sum()
        logp = float(np.log(p[action] + 1e-12))
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        return action, logp, entropy


# Post-decision gates, run on the actor path only (never after a forced escape).
_POST_GATES = (body_block_resolve, strike_gate)


def post_handover_decision(belief, danger, planner, mask, actor, features,
                           post_params, forward_bias):
    """One post-handover decision: forced-escape floor -> actor -> gates.

    The floor (if it fires) preempts the actor entirely (`actor_queried=False`).
    Otherwise the actor proposes; gates may transform the proposal into the
    executed action (override re-validated against the mask, dropped if illegal).
    The actor's PROPOSAL is what gets recorded/trained — gate-overridden ticks
    still have `actor_queried=True`. Always calls `record_final_action` so
    body_block_resolve's stuck state stays correct."""
    # 1) Forced-escape floor -- actor not queried.
    if must_force_escape(belief, danger, mask):
        a = escape_selector(belief, danger, planner, mask)
        record_final_action(belief, a, "forced_escape")
        return a, RLDecision(features, mask, None, a, 0.0, 0.0, False, True,
                             "forced_escape", forward_bias)
    # 2) Actor proposes (always recorded, even if a gate overrides it).
    # NB: we trust actor.query to return a mask-legal action (ActorRuntime masks
    # illegal logits before sampling). Frozen ticks are safe via the env mask:
    # when the agent is frozen the env exposes a STAY-only mask, so the actor can
    # only emit STAY and the floor's escape set collapses to {STAY} — no special
    # frozen short-circuit is needed here. (If a non-env mask is ever fed in,
    # this coupling is what keeps a frozen agent from emitting a phantom move.)
    proposed, logp, entropy = actor.query(features, mask, forward_bias)
    executed, source = proposed, "rl_layer"
    # 3) Gates may override; re-validate against the mask (drop illegal overrides).
    for gate in _POST_GATES:
        override = gate(belief, danger, planner, post_params, executed)
        if override is not None and 0 <= override < len(mask) and mask[override]:
            executed, source = override, f"gate:{gate.__name__}"
            break
    record_final_action(belief, executed, source)
    return executed, RLDecision(features, mask, proposed, executed, logp, entropy,
                                True, source != "rl_layer", source, forward_bias)


class HybridController:
    """Per-slot controller: scripted opener until handover, then the post-handover
    cascade. Shares the FeatureBuilder's belief (single source of truth) and warms
    the builder every tick so the actor sees a warm frame-stack at handover."""

    def __init__(self, actor, trigger, opener=None, post_params=None,
                 feature_builder=None, forward_bias=0.0):
        self.actor = actor
        self.trigger = trigger
        self.opener = opener if opener is not None else STRATEGIES["balanced_extreme_opening"]
        self.post_params = post_params if post_params is not None else StrategyParams()
        self.fb = feature_builder if feature_builder is not None else FeatureBuilder()
        self.forward_bias = forward_bias
        self.handover_fired = False

    @property
    def belief(self):
        return self.fb.belief

    def step(self, observation):
        """Returns (action, RLDecision | None). The decision is None pre-handover."""
        features = self.fb.build(observation)          # warms the builder + updates belief
        belief = self.fb.belief
        mask = np.asarray(observation["action_mask"])
        if not self.handover_fired and self.trigger(belief):
            self.handover_fired = True
        if not self.handover_fired:
            return act(belief, mask, self.opener), None   # decide.act does its own record
        danger = DangerMap(belief.enemy_bombs, belief)
        planner = build_planner(belief, danger)
        return post_handover_decision(belief, danger, planner, mask, self.actor,
                                      features, self.post_params, self.forward_bias)
