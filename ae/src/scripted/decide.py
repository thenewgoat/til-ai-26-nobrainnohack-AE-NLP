"""The cascade runner: act() executes a Strategy's layer sequence."""
from scripted.danger import DangerMap
from scripted.geometry import BACKWARD, FORWARD, LEFT, MOVE, PLACE_BOMB, RIGHT, STAY
from scripted.pathfind import build_planner
from scripted.strategies import STRATEGIES


def _first_legal(mask, preference):
    """First action in `preference` allowed by `mask`; if none match, any legal
    action in the mask; STAY only as a last resort."""
    for a in preference:
        if 0 <= a < len(mask) and mask[a] == 1:
            return a
    for a in range(len(mask)):
        if mask[a] == 1:
            return a
    return STAY


def _record(belief, action, layer):
    """Record the layer/source that produced `action` on the belief (read by
    the visualizer overlay; never affects behaviour), then return the action
    unchanged. A `PLACE_BOMB` is also logged to `belief.own_bombs` — the action
    mask guarantees the env will actually place it. `layer` is a cascade layer
    function's `__name__` (e.g. "survive", "sweep") or one of the literal
    strings "first_legal", "frozen".
    """
    if action == PLACE_BOMB:
        belief.record_own_bomb()
    belief.last_layer = layer
    return action


def record_final_action(belief, action, source):
    """Per-tick bookkeeping that decide.act applies around the chosen action,
    extracted so the hybrid controller (which bypasses the cascade runner) keeps
    the same state. Sets `belief.expected_location` (stuck-detection plumbing read
    by body_block_resolve next tick), logs a PLACE_BOMB to `belief.own_bombs`, and
    records `belief.last_layer = source`. Returns the action unchanged."""
    if action == FORWARD:
        dx, dy = MOVE[belief.facing]
        belief.expected_location = (belief.location[0] + dx, belief.location[1] + dy)
    elif action == BACKWARD:
        dx, dy = MOVE[(belief.facing + 2) % 4]
        belief.expected_location = (belief.location[0] + dx, belief.location[1] + dy)
    else:
        belief.expected_location = belief.location
    return _record(belief, action, source)


def _legal(action, mask):
    return action is not None and 0 <= action < len(mask) and mask[action] == 1


def act(belief, action_mask, strategy=None):
    """Run a Strategy's layer cascade and return a legal action int.

    `belief` must already be updated with the current observation.
    `strategy` defaults to the balanced strategy (the qualifier agent).

    After the cascade picks an action, the strategy's post-decision `gates`
    run in order: each gate may return an int to override (silently dropped
    if illegal) or None to pass through. Gates can encode opening rules,
    forced overrides, etc.
    """
    if strategy is None:
        strategy = STRATEGIES["balanced"]
    mask = list(action_mask)
    if belief.frozen_ticks > 0:
        return _record(belief, _first_legal(mask, [STAY]), "frozen")

    danger = DangerMap(belief.enemy_bombs, belief)
    planner = build_planner(belief, danger)

    chosen, source = None, None
    for layer in strategy.layers:
        action = layer(belief, danger, planner, strategy.params)
        if _legal(action, mask):
            chosen, source = action, layer.__name__
            break
    if chosen is None:
        chosen = _first_legal(mask, [FORWARD, BACKWARD, LEFT, RIGHT, STAY])
        source = "first_legal"

    # Publish the cascade's pick before gates run, so survive-deference checks
    # (body_block_resolve, strike_gate) read THIS tick's layer, not last tick's.
    belief.last_layer = source
    for gate in getattr(strategy, "gates", ()):
        override = gate(belief, danger, planner, strategy.params, chosen)
        if _legal(override, mask):
            chosen, source = override, f"gate:{gate.__name__}"

    return record_final_action(belief, chosen, source)
