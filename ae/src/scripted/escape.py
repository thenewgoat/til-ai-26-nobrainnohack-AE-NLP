"""Forced-escape floor for the hybrid post-opener controller.

Fires only when STAYing this tick is STRICTLY worse for SURVIVAL than some other
escape action — never to chase a merely-faster escape (that efficiency is the RL
policy's job). Enemy bombs only (the DangerMap it receives is built from enemy
bombs; own bombs are harmless). PLACE_BOMB is never a candidate — the floor is
bomb-free.

The fire decision (`must_force_escape`) compares the survival PREFIX `[:3]` of
each action's `survival_score`; the action choice (`escape_selector`) compares
the full score, so once we must move we pick the safest-then-cheapest escape.
"""
from scripted.geometry import BACKWARD, FORWARD, LEFT, RIGHT, STAY
from scripted.pathfind import BOMB_TIMER, build_planner

ESCAPE_HORIZON = BOMB_TIMER                  # matches layers.ESCAPE_HORIZON (= bomb fuse)
# Escape-only candidate actions — PLACE_BOMB (=5) is deliberately excluded.
ESCAPE_ACTIONS = (FORWARD, BACKWARD, LEFT, RIGHT, STAY)


def legal_escape_actions(mask):
    """The ESCAPE_ACTIONS that are legal under the env action mask."""
    return [a for a in ESCAPE_ACTIONS if bool(mask[a])]


def survival_score_after_first_action(belief, danger, first_action):
    """survival_score of committing `first_action` now, then planning freely."""
    return build_planner(belief, danger,
                         forced_first_action=first_action).survival_score(danger)


def _scores(belief, danger, mask):
    """{action: survival_score} over legal escape actions — one planner each.
    Built fresh; only called on dangerous ticks (a minority), so the cost is
    acceptable (a per-tick cache is a possible later optimisation)."""
    return {a: survival_score_after_first_action(belief, danger, a)
            for a in legal_escape_actions(mask)}


def must_force_escape(belief, danger, mask):
    """True iff a non-STAY escape action is STRICTLY safer than STAY on the
    survival prefix `[:3]`. False when the current cell is not in near-term
    danger, or when STAY survives (no survivor beats another on `[:3]`)."""
    if not danger.is_dangerous(belief.location, within=ESCAPE_HORIZON):
        return False
    scores = _scores(belief, danger, mask)
    stay = scores.get(STAY, (0, 0, 0, 0.0))[:3]
    best_alt = max((s[:3] for a, s in scores.items() if a != STAY), default=stay)
    return best_alt > stay


def escape_selector(belief, danger, planner, mask):
    """The legal escape action with the best FULL survival_score (STAY included,
    PLACE_BOMB excluded). When `must_force_escape` fired, the best action is
    necessarily a non-STAY; when every action is doomed it may return STAY
    (least-bad). `planner` is unused — kept for the controller's uniform call
    signature."""
    scores = _scores(belief, danger, mask)
    if not scores:
        return STAY
    # Invariant: when must_force_escape fired, some non-STAY action strictly beat
    # STAY on the survival prefix [:3] (slots 0-2). The full-tuple max weights
    # those same slots before the cost tie-break (slot 3), so it necessarily
    # returns a non-STAY action here — the two functions never disagree on a fire.
    return max(scores, key=lambda a: scores[a])
