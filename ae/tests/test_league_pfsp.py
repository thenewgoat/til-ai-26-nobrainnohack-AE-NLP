"""PFSP samples hard opponents (those that beat the learner) more often."""
from collections import Counter

from league import League


def test_pfsp_weights_favor_winning_opponents():
    lg = League()
    members = lg.members()
    easy, hard = members[0], members[1]
    # learner beats `easy` every game, loses to `hard` every game
    for _ in range(20):
        lg.record_result(easy, learner_won=True)
        lg.record_result(hard, learner_won=False)
    weights = lg.pfsp_weights(members)
    i_easy, i_hard = members.index(easy), members.index(hard)
    assert weights[i_hard] > weights[i_easy]


def test_pfsp_sample_is_in_pool():
    lg = League()
    for _ in range(50):
        m = lg.sample_opponent()
        assert m in lg.members()


def test_pfsp_sample_distribution_biases_hard(tmp_path):
    lg = League()
    members = lg.members()
    # make member[3] very hard (learner loses) and member[0] trivial
    for _ in range(30):
        lg.record_result(members[0], learner_won=True)
        lg.record_result(members[3], learner_won=False)
    counts = Counter(lg.sample_opponent().name for _ in range(2000))
    assert counts[members[3].name] > counts[members[0].name]
