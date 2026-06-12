from scripted.strategies import StrategyParams


def test_strategy_params_defaults():
    p = StrategyParams()
    assert p.breach_min_bombs == 5
    assert p.sweep_base_gradient == 0.5
    assert p.forage_requires_endgame is True
    assert p.camp_leash is None


def test_strategy_params_is_frozen():
    p = StrategyParams()
    try:
        p.camp_leash = 4
        assert False, "StrategyParams must be frozen"
    except AttributeError:
        pass


def test_strategy_params_override():
    p = StrategyParams(camp_leash=4, forage_requires_endgame=False)
    assert p.camp_leash == 4
    assert p.forage_requires_endgame is False
    assert p.sweep_base_gradient == 0.5  # untouched default


def test_strategy_registry_has_layer_only_strategies():
    from scripted.strategies import STRATEGIES
    for name in ("balanced", "balanced_extreme", "base_rusher",
                 "base_rusher_extreme", "collector"):
        s = STRATEGIES[name]
        assert s.name == name
        assert len(s.layers) >= 3
        assert all(callable(layer) for layer in s.layers)


def test_balanced_strike_order():
    from scripted.strategies import STRATEGIES
    from scripted.layers import hunt, survive, strike
    bal = STRATEGIES["balanced"].layers
    ext = STRATEGIES["balanced_extreme"].layers
    # balanced: survive first, then hunt, then strike.
    assert bal.index(survive) < bal.index(hunt) < bal.index(strike)
    # balanced_extreme: hunt first, then strike, with strike still before
    # survive (the "extreme" identity — bombs a base from a dangerous cell).
    assert ext.index(hunt) < ext.index(strike) < ext.index(survive)


def test_base_rusher_uses_default_params():
    from scripted.strategies import STRATEGIES
    assert STRATEGIES["base_rusher"].params == StrategyParams()
    assert STRATEGIES["base_rusher_extreme"].params == StrategyParams()


def test_act_runs_with_explicit_strategy():
    from scripted.belief import Belief
    from scripted.decide import act
    from scripted.map_prior import MapPrior
    from scripted.strategies import STRATEGIES
    prior = MapPrior.load()
    prior.identify_team(prior.bases[0])
    b = Belief()
    b.reset(prior)
    b.location = prior.spawns[0]["pos"]
    b.facing = prior.spawns[0]["facing"]
    a = act(b, [1, 1, 1, 1, 1, 1], STRATEGIES["base_rusher"])
    assert 0 <= a <= 5


def test_registry_has_all_strategies():
    from scripted.strategies import STRATEGIES
    assert set(STRATEGIES) == {
        "balanced", "balanced_extreme", "base_rusher", "base_rusher_extreme",
        "collector", "camper", "forager", "lean_rush",
        "defender", "adaptive", "balanced_extreme_opening",
        "glass_cannon", "pacifist", "hunter_killer",
    }
    for name, s in STRATEGIES.items():
        assert s.name == name
        assert len(s.layers) >= 2
        assert all(callable(layer) for layer in s.layers)


def test_camper_params_and_layers():
    from scripted.strategies import STRATEGIES
    from scripted.layers import camp, hold
    camper = STRATEGIES["camper"]
    assert camper.params.camp_leash == 4
    assert camper.params.forage_requires_endgame is False
    assert camp in camper.layers
    assert camper.layers[-1] is hold  # hold is the final fallback


def test_balanced_includes_hunt():
    from scripted.strategies import STRATEGIES
    from scripted.layers import hunt
    assert hunt in STRATEGIES["balanced"].layers
    assert hunt in STRATEGIES["balanced_extreme"].layers


def test_strategy_params_has_offense_knobs():
    p = StrategyParams()
    assert p.target_travel_weight == 0.05
    assert p.soften_floor == 60.0


def test_forager_strategy_is_registered():
    from scripted.strategies import STRATEGIES
    from scripted.adaptive_layers import forage_loop
    from scripted.layers import survive, sweep, default
    s = STRATEGIES["forager"]
    assert s.name == "forager"
    assert s.layers == (survive, forage_loop, sweep, default)


def test_strategyparams_has_roi_offense_knobs():
    from scripted.strategies import StrategyParams
    p = StrategyParams()
    assert p.roi_gate_margin == 0.15
    assert p.vulture_hp_boost == 2.0


def test_lean_rush_strategy_is_registered():
    from scripted.strategies import STRATEGIES
    from scripted.adaptive_layers import rush_roi
    from scripted.layers import survive, hunt, sweep, default
    s = STRATEGIES["lean_rush"]
    assert s.name == "lean_rush"
    assert s.layers == (survive, hunt, rush_roi, sweep, default)


def test_strategyparams_has_defend_radius():
    from scripted.strategies import StrategyParams
    assert StrategyParams().defend_radius == 4


def test_defender_strategy_is_registered():
    from scripted.strategies import STRATEGIES
    from scripted.adaptive_layers import defend_intercept, forage_loop
    from scripted.layers import survive, sweep, hold
    s = STRATEGIES["defender"]
    assert s.name == "defender"
    assert s.layers == (survive, defend_intercept, forage_loop, sweep, hold)


def test_strategyparams_has_trap_enabled_knob():
    from scripted.strategies import StrategyParams
    p = StrategyParams()
    assert p.trap_enabled is True


def test_adaptive_strategy_is_registered():
    from scripted.strategies import STRATEGIES
    from scripted.adaptive_layers import forage_loop, rush_roi, trap
    from scripted.layers import default, survive, sweep
    s = STRATEGIES["adaptive"]
    assert s.name == "adaptive"
    assert s.layers == (survive, rush_roi, trap, forage_loop, sweep, default)


def test_strategyparams_has_stuck_knobs():
    from scripted.strategies import StrategyParams
    p = StrategyParams()
    assert p.stuck_trigger_ticks == 2
    assert p.stuck_blacklist_ttl == 10


def test_balanced_extreme_has_body_block_gate():
    from scripted.strategies import STRATEGIES
    from scripted.gates import body_block_resolve
    s = STRATEGIES["balanced_extreme"]
    assert body_block_resolve in s.gates


def test_balanced_extreme_opening_has_body_block_gate():
    from scripted.strategies import STRATEGIES
    from scripted.gates import body_block_resolve, scripted_opening
    s = STRATEGIES["balanced_extreme_opening"]
    assert body_block_resolve in s.gates
    assert scripted_opening in s.gates
    # scripted_opening MUST run last so the opening book wins ticks 0-5.
    assert s.gates.index(body_block_resolve) < s.gates.index(scripted_opening)


def test_new_edge_case_strategies_registered():
    """glass_cannon, pacifist, hunter_killer must be present and constructable."""
    from scripted.strategies import STRATEGIES
    for name in ("glass_cannon", "pacifist", "hunter_killer"):
        assert name in STRATEGIES, f"missing strategy: {name}"
        strat = STRATEGIES[name]
        assert strat.name == name
        assert len(strat.layers) > 0


def test_glass_cannon_has_no_survive_layer():
    """glass_cannon's defining trait: skips the survive layer entirely."""
    from scripted.strategies import STRATEGIES
    from scripted.layers import survive
    assert survive not in STRATEGIES["glass_cannon"].layers


def test_pacifist_has_no_strike_or_hunt():
    """pacifist's defining trait: never attacks (no strike, no hunt)."""
    from scripted.strategies import STRATEGIES
    from scripted.layers import strike, hunt
    assert strike not in STRATEGIES["pacifist"].layers
    assert hunt not in STRATEGIES["pacifist"].layers


def test_hunter_killer_hunts_first():
    """hunter_killer's defining trait: hunt is the FIRST layer in the cascade."""
    from scripted.strategies import STRATEGIES
    from scripted.layers import hunt
    assert STRATEGIES["hunter_killer"].layers[0] is hunt


def test_collector_uses_forage_chain():
    from scripted.strategies import STRATEGIES
    from scripted.layers import forage, forage_chain, survive, sweep, default
    s = STRATEGIES["collector"]
    assert s.name == "collector"
    assert s.layers == (survive, forage_chain, sweep, default)
    assert forage not in s.layers
    assert s.params.forage_requires_endgame is False
