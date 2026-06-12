"""Eval diversity: randomize the learner's slot (0-5) and draw opponents from the
scripted bank, so the benchmark is no longer a single pinned scenario.

The board stays novice/seed-88 (intentionally not un-pinned); variance comes purely
from spawn-slot + opponent-mix. Randomization must be seed-derived so the paired
A/B arms see the same setup per seed.
"""
from evaluate import ScriptedAgent, evaluate_policy

BANK = ["balanced", "adaptive", "defender", "forager", "camper", "hunter_killer"]


def test_pinned_eval_is_constant():
    # baseline: no randomization -> identical score every seed (novice pins the board)
    res = evaluate_policy(ScriptedAgent("balanced_extreme_opening"),
                          [ScriptedAgent("balanced") for _ in range(5)],
                          list(range(4)))
    assert len(set(res.per_seed_scores)) == 1


def test_randomized_eval_has_variance():
    res = evaluate_policy(ScriptedAgent("balanced_extreme_opening"), [],
                          list(range(8)), randomize_slot=True, opponent_bank=BANK)
    assert len(set(res.per_seed_scores)) > 1, "randomized eval should not be constant"


def test_randomized_setup_is_seed_reproducible():
    # same seeds -> same per-seed setup -> same scores (required for paired A/B)
    common = dict(randomize_slot=True, opponent_bank=BANK)
    r1 = evaluate_policy(ScriptedAgent("balanced_extreme_opening"), [], [3, 4, 5], **common)
    r2 = evaluate_policy(ScriptedAgent("balanced_extreme_opening"), [], [3, 4, 5], **common)
    assert r1.per_seed_scores == r2.per_seed_scores
