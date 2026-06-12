"""evaluate.py reports mean score / win-rate over deterministic seeds."""
from evaluate import evaluate_policy, ScriptedAgent, RandomAgent


def test_scripted_beats_random_winrate():
    """A scripted balanced agent should clobber random opponents on novice."""
    result = evaluate_policy(
        agent=ScriptedAgent("balanced"),
        opponents=[RandomAgent()] * 5,
        seeds=list(range(8)),
    )
    assert result.episodes == 8
    assert 0.0 <= result.win_rate <= 1.0
    assert result.win_rate >= 0.5      # scripted dominates random on the known map


def test_same_seeds_are_deterministic():
    a = evaluate_policy(ScriptedAgent("balanced"), [RandomAgent()] * 5,
                        seeds=[1, 2, 3])
    b = evaluate_policy(ScriptedAgent("balanced"), [RandomAgent()] * 5,
                        seeds=[1, 2, 3])
    assert a.mean_score == b.mean_score


def test_result_fields_present():
    r = evaluate_policy(ScriptedAgent("balanced"), [RandomAgent()] * 5, seeds=[0])
    assert hasattr(r, "mean_score") and hasattr(r, "win_rate")
    assert hasattr(r, "episodes")
