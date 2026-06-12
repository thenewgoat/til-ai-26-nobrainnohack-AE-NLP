import numpy as np

from hybrid_controller import HybridController, post_handover_decision
from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.handover import HandoverTrigger
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner
from scripted.strategies import Strategy, StrategyParams
import scripted.gates as gates
from scripted.geometry import FORWARD, PLACE_BOMB, STAY


_DUMMY_FEATS = (None,) * 5


class _Prior:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.wall_between = {}
        self.collectibles = {}


def _open_belief(loc, facing, grid_size=5):
    b = Belief()
    b.prior = _Prior(grid_size)
    b.location = loc
    b.facing = facing
    b.team_bombs = 0
    return b


def _real_belief(loc=(8, 8), facing=0):
    prior = MapPrior.load()
    prior.identify_team(prior.bases[0])
    b = Belief()
    b.reset(prior)
    b.location = loc
    b.facing = facing
    return b


class _StubActor:
    def __init__(self, proposed=FORWARD, logp=-0.5, entropy=1.0):
        self.proposed, self.logp, self.entropy, self.calls = proposed, logp, entropy, 0

    def query(self, features, mask, forward_bias=0.0):
        self.calls += 1
        return self.proposed, self.logp, self.entropy


class _StubFB:
    def __init__(self, belief, features=_DUMMY_FEATS):
        self.belief = belief
        self._features = features

    def build(self, observation):
        return self._features


def _forward_layer(belief, danger, planner, params):
    return FORWARD


_TEST_OPENER = Strategy("test_opener", (_forward_layer,), StrategyParams())


# ---- post_handover_decision (unit) ----

def test_actor_path_no_intervention():
    b = _real_belief()
    b.team_bombs = 0                         # strike_gate yields; stuck_ticks 0 -> body_block yields
    danger = DangerMap({}, b)                # no danger -> floor does not fire
    planner = build_planner(b, danger)
    actor = _StubActor(proposed=FORWARD)
    action, dec = post_handover_decision(
        b, danger, planner, np.ones(6, bool), actor, _DUMMY_FEATS, StrategyParams(), 0.0)
    assert action == FORWARD
    assert dec.actor_queried is True and dec.intervened is False
    assert dec.source == "rl_layer" and dec.proposed_action == FORWARD
    assert dec.executed_action == FORWARD
    assert b.last_layer == "rl_layer"        # record_final_action ran


def test_strike_gate_overrides_proposal(monkeypatch):
    b = _real_belief()
    b.team_bombs = 2
    monkeypatch.setattr(gates, "bomb_reaches", lambda loc, base, belief: True)
    monkeypatch.setattr(gates, "_base_doomed", lambda belief, base: False)
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    actor = _StubActor(proposed=FORWARD)
    action, dec = post_handover_decision(
        b, danger, planner, np.ones(6, bool), actor, _DUMMY_FEATS, StrategyParams(), 0.0)
    assert action == PLACE_BOMB
    assert dec.proposed_action == FORWARD and dec.executed_action == PLACE_BOMB
    assert dec.actor_queried is True and dec.intervened is True
    assert dec.source == "gate:strike_gate"


def test_gate_override_dropped_when_illegal_in_mask(monkeypatch):
    b = _real_belief()
    b.team_bombs = 2
    monkeypatch.setattr(gates, "bomb_reaches", lambda loc, base, belief: True)
    monkeypatch.setattr(gates, "_base_doomed", lambda belief, base: False)
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    mask = np.ones(6, bool)
    mask[PLACE_BOMB] = False                  # strike_gate WOULD fire, but PLACE_BOMB illegal
    actor = _StubActor(proposed=FORWARD)
    action, dec = post_handover_decision(
        b, danger, planner, mask, actor, _DUMMY_FEATS, StrategyParams(), 0.0)
    assert action == FORWARD                   # override dropped
    assert dec.source == "rl_layer" and dec.intervened is False


def test_forced_escape_preempts_actor():
    # trapped geometry from Plan 2: agent (1,0), bomb (0,0) timer 2 -> floor fires
    b = _open_belief((1, 0), 0)
    danger = DangerMap({(0, 0): 2}, b)
    planner = build_planner(b, danger)
    actor = _StubActor(proposed=FORWARD)
    action, dec = post_handover_decision(
        b, danger, planner, np.ones(6, bool), actor, _DUMMY_FEATS, StrategyParams(), 0.0)
    assert dec.actor_queried is False and dec.source == "forced_escape"
    assert dec.proposed_action is None
    assert actor.calls == 0                    # actor not queried when the floor fires
    assert action in (FORWARD, STAY) and action != PLACE_BOMB


# ---- HybridController.step (dispatch / latch) ----

def test_pre_handover_runs_opener_and_returns_no_decision():
    b = _open_belief((2, 2), 0)
    b.step = 5                                 # before fallback, no dead bases
    b.dead_bases = set()
    ctl = HybridController(_StubActor(), HandoverTrigger(),
                           opener=_TEST_OPENER, feature_builder=_StubFB(b))
    action, decision = ctl.step({"action_mask": np.ones(6, bool)})
    assert decision is None
    assert action == FORWARD                    # opener's _forward_layer
    assert ctl.handover_fired is False
    assert ctl.actor.calls == 0


def test_handover_latches_and_stays_post_handover():
    b = _real_belief()
    b.step = 60                                 # step fallback fires
    b.dead_bases = set()
    ctl = HybridController(_StubActor(proposed=STAY), HandoverTrigger(),
                           opener=_TEST_OPENER, feature_builder=_StubFB(b))
    _, dec1 = ctl.step({"action_mask": np.ones(6, bool)})
    assert ctl.handover_fired is True and dec1 is not None
    b.step = 1                                  # even if step rewinds, latch holds
    _, dec2 = ctl.step({"action_mask": np.ones(6, bool)})
    assert ctl.handover_fired is True and dec2 is not None


def _real_features():
    return (np.zeros((85, 16, 16), np.float32), np.zeros((5, 11), np.float32),
            np.zeros((7, 5, 25), np.float32), np.zeros((7, 7, 25), np.float32),
            np.zeros(50, np.float32))


def test_step_calls_fb_build_once_and_shares_its_belief():
    # The core contract: step() warms the FeatureBuilder every tick and operates
    # on fb.belief as the single source of truth.
    b = _real_belief()
    b.step = 60                                   # force handover so the post path runs
    fb = _StubFB(b)
    calls = []
    orig = fb.build
    fb.build = lambda obs: (calls.append(obs) or orig(obs))
    ctl = HybridController(_StubActor(proposed=STAY), HandoverTrigger(),
                           opener=_TEST_OPENER, feature_builder=fb)
    assert ctl.belief is b                        # property exposes the FB's belief
    ctl.step({"action_mask": np.ones(6, bool)})
    assert len(calls) == 1                        # build called exactly once this tick
    # record_final_action ran on the SHARED belief (last_layer set on b):
    assert b.last_layer in ("rl_layer", "forced_escape") or b.last_layer.startswith("gate:")


def test_post_handover_respects_stay_only_mask_like_frozen():
    # Frozen → env mask is STAY-only. With the REAL ActorRuntime the actor can
    # only emit STAY; the floor's escape set is {STAY} too. Executed action = STAY.
    from hybrid_controller import ActorRuntime
    from policy import SymbolicTransformerActor
    b = _real_belief()
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    actor = ActorRuntime(SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2))
    mask = np.zeros(6, bool)
    mask[STAY] = True                             # only STAY legal (frozen)
    action, dec = post_handover_decision(
        b, danger, planner, mask, actor, _real_features(), StrategyParams(), 0.0)
    assert action == STAY


def test_body_block_resolve_fires_before_strike_gate(monkeypatch):
    # body_block_resolve is first in _POST_GATES; when it fires its PLACE_BOMB wins
    # (loop breaks on the first firing gate).
    b = _real_belief()
    b.team_bombs = 2
    b.stuck_ticks = 5                             # >= stuck_trigger_ticks
    b.last_layer = "rl_layer"                     # not "survive"
    monkeypatch.setattr(gates, "_has_escape", lambda belief, danger: True)
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    actor = _StubActor(proposed=FORWARD)
    action, dec = post_handover_decision(
        b, danger, planner, np.ones(6, bool), actor, _DUMMY_FEATS, StrategyParams(), 0.0)
    assert action == PLACE_BOMB
    assert dec.source == "gate:body_block_resolve"
    assert dec.intervened is True
