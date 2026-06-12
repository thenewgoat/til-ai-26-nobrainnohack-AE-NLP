"""Opponent-pool manager for the self-play ladder.

Holds two kinds of opponents:
  - PERMANENT scripted-strategy anchors (never evicted) — one per registered
    strategy in scripted.strategies.STRATEGIES; guarantees the learner cannot
    forget how to beat any scripted opponent.
  - frozen learner CHECKPOINTS, capped at max_checkpoints (oldest evicted).

Tracks per-member win-rate bookkeeping for PFSP sampling (Task 12).
"""
from dataclasses import dataclass, field

from scripted.strategies import STRATEGIES

@dataclass(eq=False)
class Member:
    name: str
    kind: str                      # "scripted" | "checkpoint"
    ref: str                       # strategy name, or checkpoint .pt path
    update: int = 0                # training update at which it was snapshotted
    # win-rate bookkeeping: games where the LEARNER beat this member
    learner_wins: int = 0
    games: int = 0

    def learner_winrate(self):
        return self.learner_wins / self.games if self.games else 0.5

    def record(self, learner_won):
        self.games += 1
        if learner_won:
            self.learner_wins += 1


class League:
    """The opponent pool."""

    def __init__(self, max_checkpoints=10):
        self.max_checkpoints = max_checkpoints
        self._anchors = [
            Member(name=f"scripted:{s}", kind="scripted", ref=s)
            for s in STRATEGIES
        ]
        self._checkpoints = []

    def members(self):
        return self._anchors + self._checkpoints

    def checkpoints(self):
        return list(self._checkpoints)

    def anchors(self):
        return list(self._anchors)

    def snapshot(self, ckpt_path, update):
        """Freeze the current learner into the pool; evict the oldest if full."""
        self._checkpoints.append(Member(
            name=f"ckpt:{update}", kind="checkpoint", ref=ckpt_path,
            update=update,
        ))
        if len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)

    def record_result(self, member, learner_won):
        member.record(learner_won)


import random


def _pfsp_weight(learner_winrate):
    """PFSP 'hard' curve: weight high when the learner LOSES to a member.

    f_hard(p) = (1 - p)^2 where p = P[learner beats member]. An untested
    member (p defaults to 0.5) gets a moderate weight so it still gets played.
    """
    return (1.0 - learner_winrate) ** 2 + 1e-3   # epsilon keeps every member reachable


# attach as methods of League ------------------------------------------- #
def _league_pfsp_weights(self, members=None):
    members = members if members is not None else self.members()
    return [_pfsp_weight(m.learner_winrate()) for m in members]


def _league_sample_opponent(self):
    members = self.members()
    weights = self.pfsp_weights(members)
    return random.choices(members, weights=weights, k=1)[0]


League.pfsp_weights = _league_pfsp_weights
League.sample_opponent = _league_sample_opponent
