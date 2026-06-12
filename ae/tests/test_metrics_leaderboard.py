"""MetricsLogger.leaderboard snapshots league standings to CSV + PNG."""
import csv
import os

from metrics import MetricsLogger
from league import League


def test_leaderboard_writes_csv_and_png(tmp_path):
    league = League()
    # give a few members non-default win-rate data
    anchors = league.anchors()
    for _ in range(4):
        league.record_result(anchors[0], learner_won=True)
    for _ in range(4):
        league.record_result(anchors[1], learner_won=False)
    league.snapshot(os.path.join(str(tmp_path), "ckpt_u25.pt"), update=25)
    for _ in range(2):
        league.record_result(league.checkpoints()[0], learner_won=True)

    logger = MetricsLogger(str(tmp_path))
    csv_path = os.path.join(str(tmp_path), "leaderboard.csv")
    png_path = os.path.join(str(tmp_path), "leaderboard_u25.png")
    logger.leaderboard(league, 25, csv_path, png_path)
    logger.close()

    assert os.path.exists(png_path) and os.path.getsize(png_path) > 0
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # one row per member (6 anchors + 1 checkpoint)
    assert len(rows) == len(league.members())
    expected_cols = {"update", "name", "kind", "ref", "snapshot_update",
                     "learner_wins", "games", "learner_winrate",
                     "pfsp_weight"}
    assert expected_cols.issubset(rows[0].keys())
    assert rows[0]["update"] == "25"


def test_leaderboard_csv_is_appended(tmp_path):
    """A second leaderboard() call appends rows, not overwrites."""
    league = League()
    logger = MetricsLogger(str(tmp_path))
    csv_path = os.path.join(str(tmp_path), "leaderboard.csv")
    logger.leaderboard(league, 25, csv_path,
                       os.path.join(str(tmp_path), "lb_u25.png"))
    logger.leaderboard(league, 50, csv_path,
                       os.path.join(str(tmp_path), "lb_u50.png"))
    logger.close()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * len(league.members())
