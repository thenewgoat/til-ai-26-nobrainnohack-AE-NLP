"""Manages the AE model — deterministic scripted agent (no neural network)."""

import os

import numpy as np

from scripted import greedy
from scripted.belief import Belief, _scalar
from scripted.decide import act
from scripted.strategies import STRATEGIES
from scripted.map_prior import MapPrior


class AEManager:
    """Serves the scripted AE agent. One instance per server process."""

    GREEDY = "greedy"

    def __init__(self):
        self.prior = MapPrior.load()
        self.belief = Belief()
        self._episode_started = False   # True once a step-0 (round start) has been processed
        # Which scripted strategy to serve; set per-image via the AE_STRATEGY
        # env var (Docker build-arg). Defaults to the qualifier agent.
        name = os.environ.get("AE_STRATEGY", "balanced")
        self._greedy = name == self.GREEDY
        if self._greedy:
            self.strategy = None
        elif name in STRATEGIES:
            self.strategy = STRATEGIES[name]
        else:
            raise ValueError(
                f"AE_STRATEGY={name!r} is not a known strategy; "
                f"choose one of {sorted(list(STRATEGIES) + [self.GREEDY])}")

    def ae(self, observation: dict) -> int:
        """Return the next action for the agent.

        Args:
            observation: environment observation; see `ae/README.md`.

        Returns:
            An integer action in [0, 6).
        """
        # _scalar handles int / list / numpy-array forms uniformly.
        step = _scalar(observation["step"])

        # step == 0 marks a new round; the eval never calls /reset.
        if step == 0 or not self._episode_started:
            self.prior.identify_team(observation["base_location"])
            self.belief.reset(self.prior)
            self._episode_started = True

        self.belief.update(observation)
        if self._greedy:
            return greedy.act(self.belief, observation["action_mask"])
        return act(self.belief, observation["action_mask"], self.strategy)


class NeuralAEManager:
    """Serves the spec C trained network via ONNX.

    Each step: fold the observation into the per-agent FeatureBuilder belief
    (reset on step==0), build the symbolic feature tensors, run the ONNX actor,
    mask illegal actions, argmax. The container ships onnxruntime instead of
    torch (spec C §8).
    """

    def __init__(self, onnx_path=None):
        import onnxruntime as ort
        from features import FeatureBuilder
        if onnx_path is None:
            # Baked submission path: the smoke-trained model exported from
            # policy_final.pt (20-update PPO from BC init). Swap this filename
            # to ship a different model (e.g. "policy_bc.onnx" for the BC
            # clone, or a freshly exported name after a longer training run).
            onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "policy_smoke.onnx")
        self.session = ort.InferenceSession(onnx_path)
        self.fb = FeatureBuilder()

    def ae(self, observation: dict) -> int:
        """Return the next action for the agent."""
        # FeatureBuilder.build() handles the step==0 belief reset internally.
        grid, base_feats, raw_agent, raw_base, scalar = self.fb.build(
            observation)
        logits = self.session.run(
            ["logits"],
            {"grid": grid[None].astype(np.float32),
             "base_feats": base_feats[None].astype(np.float32),
             "raw_agent": raw_agent[None].astype(np.float32),
             "raw_base": raw_base[None].astype(np.float32),
             "scalar": scalar[None].astype(np.float32)},
        )[0][0]
        mask = np.asarray(observation["action_mask"], dtype=bool).reshape(-1)
        logits = np.where(mask, logits, -1e8)
        return int(np.argmax(logits))


class HybridAEManager:
    """Serves the hybrid post-opener agent: scripted opener until handover, then
    the RL actor (ONNX) under the forced-escape floor + gates.

    Loads the trained actor from AE_RL_ACTOR_PATH (a baked .onnx). A missing path
    is a FATAL startup error — never silently degrade to scripted/default.
    """

    def __init__(self):
        from features import FeatureBuilder
        from hybrid_controller import HybridController, OnnxActorRuntime
        from scripted.handover import HandoverTrigger
        path = os.environ.get("AE_RL_ACTOR_PATH")
        if not path or not os.path.exists(path):
            raise RuntimeError(
                "AE_MODE=hybrid requires AE_RL_ACTOR_PATH to point at an exported "
                f"actor .onnx; got {path!r}")
        actor = OnnxActorRuntime.from_path(path)
        self.controller = HybridController(
            actor, HandoverTrigger(), feature_builder=FeatureBuilder(),
            forward_bias=0.0)

    def ae(self, observation: dict) -> int:
        action, _decision = self.controller.step(observation)
        return action
