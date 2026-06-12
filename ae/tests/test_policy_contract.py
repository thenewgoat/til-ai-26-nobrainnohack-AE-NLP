"""SymbolicTransformerActor: two-branch forward, 347 tokens, masked sampling."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import torch

from policy import SymbolicTransformerActor, NUM_ACTIONS, NUM_TOKENS


def _dummy(n=3):
    return (torch.zeros(n, 85, 16, 16), torch.zeros(n, 5, 11),
            torch.zeros(n, 7, 5, 25), torch.zeros(n, 7, 7, 25),
            torch.zeros(n, 50))


def test_forward_accepts_five_tensors_returns_logits():
    net = SymbolicTransformerActor()
    logits = net(*_dummy(4))
    assert logits.shape == (4, NUM_ACTIONS)


def test_token_count_is_347():
    assert NUM_TOKENS == 1 + 256 + 5 + 84 + 1 == 347


def test_act_masks_illegal_actions():
    net = SymbolicTransformerActor()
    grid, base_feats, raw_agent, raw_base, scalar = _dummy(2)
    mask = torch.zeros(2, NUM_ACTIONS, dtype=torch.bool)
    mask[:, 0] = True
    action, logp, ent = net.act(grid, base_feats, raw_agent, raw_base,
                                scalar, mask)
    assert (action == 0).all()


def test_type_embeddings_are_distinct_per_group():
    net = SymbolicTransformerActor()
    te = net.type_embed.detach()
    assert te.shape[0] == 4
    assert not torch.allclose(te[0], te[1])


def test_checkpoint_round_trip(tmp_path):
    net = SymbolicTransformerActor(d_model=32, n_layers=2, n_heads=4)
    path = tmp_path / "actor.pt"
    net.save_checkpoint(str(path))
    loaded = SymbolicTransformerActor.from_checkpoint(str(path))
    net.eval()
    loaded.eval()
    out_a = net(*_dummy(2))
    out_b = loaded(*_dummy(2))
    torch.testing.assert_close(out_a, out_b)


def test_policy_dims_match_feature_contract():
    """policy.py mirrors features.py's frozen contract."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "src", "training"))
    import features as Fe
    import policy as P
    assert P.GRID_CHANNELS == Fe.GRID_CHANNELS
    assert P.NUM_BASES == Fe.NUM_BASES
    assert P.BASE_FIELDS == Fe.BASE_FIELDS
    assert P.SYMBOLIC_SCALARS == Fe.FEATURE_SCALARS
    assert P.RAW_CHANNELS == Fe.RAW_AGENT_SHAPE[2] == Fe.RAW_BASE_SHAPE[2]
    assert P.NUM_RAW_AGENT == Fe.RAW_AGENT_SHAPE[0] * Fe.RAW_AGENT_SHAPE[1]
    assert P.NUM_RAW_BASE == Fe.RAW_BASE_SHAPE[0] * Fe.RAW_BASE_SHAPE[1]
    assert P.NUM_TOKENS == 347
    # frame-stack contract agreement
    assert P.STACK == Fe.STACK
    assert P.STACKED_GRID_CHANNELS == Fe.STACKED_GRID_CHANNELS
    assert P.STACKED_SCALARS == Fe.STACKED_SCALARS
