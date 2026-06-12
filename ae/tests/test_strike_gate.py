import scripted.gates as gates
from scripted.belief import Belief
from scripted.geometry import FORWARD, PLACE_BOMB
from scripted.map_prior import MapPrior


def _fresh_belief():
    prior = MapPrior.load()
    prior.identify_team(prior.bases[0])
    b = Belief()
    b.reset(prior)
    return b


def _patch(monkeypatch, reaches=True, doomed=False):
    monkeypatch.setattr(gates, "bomb_reaches", lambda loc, base, belief: reaches)
    monkeypatch.setattr(gates, "_base_doomed", lambda belief, base: doomed)


def test_places_bomb_when_in_range_of_live_nondoomed_base(monkeypatch):
    b = _fresh_belief()
    b.team_bombs = 2
    _patch(monkeypatch, reaches=True, doomed=False)
    assert gates.strike_gate(b, None, None, None, FORWARD) == PLACE_BOMB


def test_yields_when_actor_already_proposed_bomb(monkeypatch):
    b = _fresh_belief()
    b.team_bombs = 2
    _patch(monkeypatch, reaches=True, doomed=False)
    assert gates.strike_gate(b, None, None, None, PLACE_BOMB) is None


def test_yields_when_no_bombs_in_hand(monkeypatch):
    b = _fresh_belief()
    b.team_bombs = 0
    _patch(monkeypatch, reaches=True, doomed=False)
    assert gates.strike_gate(b, None, None, None, FORWARD) is None


def test_yields_when_no_base_in_range(monkeypatch):
    b = _fresh_belief()
    b.team_bombs = 2
    _patch(monkeypatch, reaches=False, doomed=False)
    assert gates.strike_gate(b, None, None, None, FORWARD) is None


def test_yields_when_only_reachable_base_is_doomed(monkeypatch):
    b = _fresh_belief()
    b.team_bombs = 2
    _patch(monkeypatch, reaches=True, doomed=True)
    assert gates.strike_gate(b, None, None, None, FORWARD) is None


def test_defers_to_survive_pick(monkeypatch):
    # survive's _strike_caveat already weighed bombing against the escape
    # deadline; if it still chose a flee move, the gate must not burn the tick.
    b = _fresh_belief()
    b.team_bombs = 2
    b.last_layer = "survive"
    _patch(monkeypatch, reaches=True, doomed=False)
    assert gates.strike_gate(b, None, None, None, FORWARD) is None


def test_fires_after_strike_dead_bases_cap_give_up(monkeypatch):
    # The cap silences the strike layer, not the gate: walking past an alive
    # base with bombs in hand still drops one (zero detour, still scores).
    b = _fresh_belief()
    b.team_bombs = 2
    b.dead_bases = set(b.live_enemy_bases()[1:])  # two enemy bases down
    b.base_health = 0.0                           # ours too -> dead == cap (3)
    b.last_layer = "forage_chain"
    _patch(monkeypatch, reaches=True, doomed=False)
    assert gates.strike_gate(b, None, None, None, FORWARD) == PLACE_BOMB


def test_filters_per_base_keeps_only_reachable_nondoomed(monkeypatch):
    b = _fresh_belief()
    b.team_bombs = 2
    bases = b.live_enemy_bases()
    assert len(bases) >= 2            # map has multiple enemy bases
    keep = bases[0]
    monkeypatch.setattr(gates, "bomb_reaches", lambda loc, base, belief: True)
    # every base reachable, but all-but-one are doomed → only `keep` survives the filter
    monkeypatch.setattr(gates, "_base_doomed", lambda belief, base: base != keep)
    assert gates.strike_gate(b, None, None, None, FORWARD) == PLACE_BOMB
