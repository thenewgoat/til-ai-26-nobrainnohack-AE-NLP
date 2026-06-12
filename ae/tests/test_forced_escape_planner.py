from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.geometry import BACKWARD, FORWARD, LEFT, STAY
from scripted.pathfind import INF, build_planner


class _Prior:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.wall_between = {}
        self.collectibles = {}


def _belief(loc, facing, grid_size=5):
    b = Belief()
    b.prior = _Prior(grid_size)
    b.location = loc
    b.facing = facing
    b.team_bombs = 3
    return b


def test_forced_first_action_constrains_the_first_move():
    b = _belief((0, 0), 0)                         # facing RIGHT
    free = build_planner(b, DangerMap({}, b))
    assert free.dist_to((1, 0)) == 1               # one FORWARD east
    forced = build_planner(b, DangerMap({}, b), forced_first_action=LEFT)
    assert forced.dist_to((1, 0)) > 1


def test_default_forced_action_is_inert():
    b1, b2 = _belief((0, 0), 0), _belief((0, 0), 0)
    a = build_planner(b1, DangerMap({}, b1))
    c = build_planner(b2, DangerMap({}, b2), forced_first_action=None)
    for t in [(0, 0), (1, 0), (4, 0), (2, 2), (4, 4)]:
        assert a.dist_to(t) == c.dist_to(t)


def test_has_safe_continuation_true_without_danger():
    b = _belief((0, 0), 0)
    p = build_planner(b, DangerMap({}, b), forced_first_action=STAY)
    assert p.has_safe_continuation() is True
    assert p.survival_score(DangerMap({}, b))[0] == 1


def test_has_safe_continuation_false_when_trapped_on_a_lethal_cell():
    b = _belief((0, 0), 0, grid_size=1)            # single cell, nowhere to move
    danger = DangerMap({(0, 0): 1}, b)             # bomb detonates here at tick 1
    assert danger.is_lethal_at((0, 0), 1) is True
    p = build_planner(b, danger, forced_first_action=STAY)
    assert p.has_safe_continuation() is False
    assert p.survival_score(danger)[0] == 0


def test_survival_score_survivor_beats_nonsurvivor():
    surv = _belief((0, 0), 0, grid_size=1)
    d_safe = DangerMap({}, surv)
    s_surv = build_planner(surv, d_safe, forced_first_action=STAY).survival_score(d_safe)
    trap = _belief((0, 0), 0, grid_size=1)
    d_trap = DangerMap({(0, 0): 1}, trap)
    s_trap = build_planner(trap, d_trap, forced_first_action=STAY).survival_score(d_trap)
    assert s_surv > s_trap
    assert s_surv[0] == 1 and s_trap[0] == 0


def test_forced_move_blocked_by_wall_has_no_continuation():
    # Forcing FORWARD into an indestructible wall yields no successor — the
    # planner scores it as "dies in place". This is conservatively safe: STAY
    # (same physical effect, scored correctly) dominates it, so the floor never
    # picks a blocked move over STAY.
    b = _belief((0, 0), 0)                                        # facing RIGHT
    b.prior.wall_between = {frozenset({(0, 0), (1, 0)}): False}   # wall to the east
    danger = DangerMap({}, b)
    blocked = build_planner(b, danger, forced_first_action=FORWARD)
    assert blocked.has_safe_continuation() is False
    assert blocked.dist_to((1, 0)) == INF                        # never reached
    stay = build_planner(b, danger, forced_first_action=STAY)
    assert stay.survival_score(danger) > blocked.survival_score(danger)


def test_forced_backward_steps_behind_at_backward_cost():
    b = _belief((2, 0), 0)                                        # facing RIGHT; BACKWARD = west
    p = build_planner(b, DangerMap({}, b), forced_first_action=BACKWARD)
    assert p.dist_to((1, 0)) == 1.4                              # one BACKWARD step (BACKWARD_COST)


def test_survival_score_prefers_later_death_among_nonsurvivors():
    # Single cell, nowhere to move. A timer-1 bomb kills at phase 1 (max_phase 0);
    # a timer-3 bomb lets the agent stay safely through phase 2 before dying.
    b1 = _belief((0, 0), 0, grid_size=1)
    d1 = DangerMap({(0, 0): 1}, b1)
    s_early = build_planner(b1, d1, forced_first_action=STAY).survival_score(d1)
    b2 = _belief((0, 0), 0, grid_size=1)
    d2 = DangerMap({(0, 0): 3}, b2)
    s_late = build_planner(b2, d2, forced_first_action=STAY).survival_score(d2)
    assert s_early[0] == 0 and s_late[0] == 0                    # both doomed
    assert s_late > s_early                                       # later death ranks higher
