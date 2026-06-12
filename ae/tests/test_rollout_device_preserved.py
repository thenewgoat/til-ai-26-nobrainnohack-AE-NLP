"""Regression: collect_hybrid_rollout must not change the actor's device.

ActorRuntime moves the actor to CPU in place for per-step inference. Before the
fix this silently left the *training* actor on CPU, so the subsequent
ppo_update_hybrid (which moves minibatches to the training device and assumes the
actor is already there) crashed with a cuda/cpu mismatch in tile_embed.
"""
import pytest
import torch

from hybrid_rollout import collect_hybrid_rollout
from policy import SymbolicTransformerActor
from scripted.handover import HandoverTrigger


def _run_and_get_device(device):
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    actor.to(device)
    collect_hybrid_rollout(
        actor, learner_slots=["agent_0"], num_episodes=1, seed0=0,
        trigger=HandoverTrigger(step_fallback=5))
    return next(actor.parameters()).device


def test_rollout_preserves_cpu_actor_device():
    assert _run_and_get_device("cpu").type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rollout_preserves_cuda_actor_device():
    # This is the exact scenario from the crashing run (--device cuda).
    assert _run_and_get_device("cuda").type == "cuda"
