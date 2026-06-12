"""SymbolicTransformerActor accepts stacked grid + scalar dims."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import torch

import policy as P
from policy import SymbolicTransformerActor, NUM_ACTIONS, NUM_TOKENS


def _stacked_dummy(n=3):
    return (torch.zeros(n, P.STACKED_GRID_CHANNELS, 16, 16),
            torch.zeros(n, 5, 11),
            torch.zeros(n, 7, 5, 25), torch.zeros(n, 7, 7, 25),
            torch.zeros(n, P.STACKED_SCALARS))


def test_forward_accepts_stacked_inputs():
    net = SymbolicTransformerActor()
    logits = net(*_stacked_dummy(4))
    assert logits.shape == (4, NUM_ACTIONS)


def test_token_count_unchanged():
    assert NUM_TOKENS == 347


def test_tile_embed_input_dim_is_stacked():
    net = SymbolicTransformerActor()
    assert net.tile_embed.in_features == P.STACKED_GRID_CHANNELS == 85


def test_scalar_embed_input_dim_is_stacked():
    net = SymbolicTransformerActor()
    # scalar_embed is nn.Sequential(nn.Linear, nn.GELU)
    assert net.scalar_embed[0].in_features == P.STACKED_SCALARS == 50
