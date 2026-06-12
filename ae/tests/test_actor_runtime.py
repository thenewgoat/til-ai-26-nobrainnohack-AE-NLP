import numpy as np

from hybrid_controller import ActorRuntime, RLDecision
from policy import SymbolicTransformerActor
from scripted.geometry import FORWARD, PLACE_BOMB, STAY


def _features():
    return (np.zeros((85, 16, 16), np.float32), np.zeros((5, 11), np.float32),
            np.zeros((7, 5, 25), np.float32), np.zeros((7, 7, 25), np.float32),
            np.zeros(50, np.float32))


def _runtime():
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    return ActorRuntime(actor)


def test_query_returns_a_legal_action_and_floats():
    rt = _runtime()
    mask = np.zeros(6, bool)
    mask[FORWARD] = True
    mask[STAY] = True
    action, logp, entropy = rt.query(_features(), mask)
    assert action in (FORWARD, STAY)              # never a masked-illegal action
    assert isinstance(logp, float) and isinstance(entropy, float)


def test_query_respects_mask_under_forward_bias():
    rt = _runtime()
    mask = np.ones(6, bool)
    mask[FORWARD] = False                          # FORWARD illegal
    for _ in range(20):
        action, _, _ = rt.query(_features(), mask, forward_bias=50.0)
        assert action != FORWARD                   # huge bias must not unmask FORWARD


def test_rl_decision_is_a_plain_record():
    d = RLDecision(features=_features(), action_mask=np.ones(6, bool),
                   proposed_action=FORWARD, executed_action=PLACE_BOMB,
                   old_proposal_logp=-0.5, entropy=1.2, actor_queried=True,
                   intervened=True, source="gate:strike_gate", forward_bias=0.0)
    assert d.proposed_action == FORWARD and d.executed_action == PLACE_BOMB
    assert d.actor_queried and d.intervened
