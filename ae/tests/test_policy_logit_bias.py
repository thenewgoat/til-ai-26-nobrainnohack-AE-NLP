import torch

from policy import SymbolicTransformerActor
from scripted.geometry import FORWARD, STAY


def _inputs(n=1):
    return (torch.zeros(n, 85, 16, 16), torch.zeros(n, 5, 11),
            torch.zeros(n, 7, 5, 25), torch.zeros(n, 7, 7, 25), torch.zeros(n, 50))


def _actor():
    a = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    a.eval()                                          # deterministic (no dropout)
    return a


def test_logit_bias_none_is_inert():
    a = _actor()
    g, bf, ra, rb, sc = _inputs()
    mask = torch.ones(1, 6, dtype=torch.bool)
    act = torch.tensor([FORWARD])
    _, lp_default, _ = a.act(g, bf, ra, rb, sc, mask, action=act)
    _, lp_none, _ = a.act(g, bf, ra, rb, sc, mask, action=act, logit_bias=None)
    assert torch.allclose(lp_default, lp_none)


def test_logit_bias_does_not_unmask_an_illegal_action():
    a = _actor()
    g, bf, ra, rb, sc = _inputs()
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, FORWARD] = False                          # FORWARD illegal
    bias = torch.zeros(6)
    bias[FORWARD] = 100.0                             # huge bias on the illegal action
    action, _, _ = a.act(g, bf, ra, rb, sc, mask, logit_bias=bias)
    assert action.item() != FORWARD                   # mask applied AFTER bias


def test_logit_bias_shifts_probability_mass_toward_forward():
    a = _actor()
    g, bf, ra, rb, sc = _inputs()
    mask = torch.ones(1, 6, dtype=torch.bool)
    act = torch.tensor([FORWARD])
    _, lp0, _ = a.act(g, bf, ra, rb, sc, mask, action=act)
    bias = torch.zeros(6)
    bias[FORWARD] = 2.0
    _, lp_biased, _ = a.act(g, bf, ra, rb, sc, mask, action=act, logit_bias=bias)
    assert lp_biased.item() > lp0.item()              # FORWARD more probable under bias
