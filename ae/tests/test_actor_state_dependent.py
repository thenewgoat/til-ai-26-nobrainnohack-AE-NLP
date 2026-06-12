"""The actor's output must depend on its observation.

With the old action-head init (std=0.01), the head was ~100x too small: it crushed
the (present) CLS signal to ~0.002 logit variation, so the policy was a near-constant
function of the input — at argmax-deploy it oscillated in 2 cells and never learned.
A usable policy head must produce clearly input-dependent logits.
"""
import torch

from policy import SymbolicTransformerActor
from features import (STACKED_GRID_CHANNELS, NUM_BASES, BASE_FIELDS,
                      RAW_AGENT_SHAPE, RAW_BASE_SHAPE, STACKED_SCALARS)


def _random_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, STACKED_GRID_CHANNELS, 16, 16, generator=g),
            torch.randn(n, NUM_BASES, BASE_FIELDS, generator=g),
            torch.randn(n, *RAW_AGENT_SHAPE, generator=g),
            torch.randn(n, *RAW_BASE_SHAPE, generator=g),
            torch.randn(n, STACKED_SCALARS, generator=g)]


def test_fresh_actor_logits_depend_on_observation():
    torch.manual_seed(0)
    actor = SymbolicTransformerActor(d_model=64, n_layers=4, n_heads=4, dropout=0.0).eval()
    with torch.no_grad():
        logits = actor.forward(*_random_batch(16, seed=1)).numpy()
    # per-action std across distinct inputs; the broken head gave ~0.002.
    assert logits.std(0).mean() > 0.05, (
        f"actor output barely varies with input (std={logits.std(0).mean():.4f}) "
        "— policy is ~constant; the action head is too small")
