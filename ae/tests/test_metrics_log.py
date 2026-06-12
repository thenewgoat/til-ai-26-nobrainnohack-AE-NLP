"""MetricsLogger.log writes CSV rows and TensorBoard scalars."""
import csv
import glob
import os

from metrics import MetricsLogger


def test_log_writes_csv_and_tb(tmp_path):
    logger = MetricsLogger(str(tmp_path))
    logger.log(1, {"policy_loss": 0.5, "value_loss": 1.2, "entropy": 1.7,
                   "mean_return": -3.0, "rung": 1, "pool_size": 0,
                   "anti_idle_coef": 0.05})
    # update 2 also has eval-only metrics -> new columns appear
    logger.log(2, {"policy_loss": 0.4, "value_loss": 1.1, "entropy": 1.6,
                   "mean_return": 2.0, "rung": 1, "pool_size": 0,
                   "anti_idle_coef": 0.04,
                   "eval_winrate": 0.5, "eval_score": 10.0})
    logger.close()

    csv_path = os.path.join(str(tmp_path), "metrics.csv")
    assert os.path.exists(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["update"] == "1"
    assert "policy_loss" in rows[0]
    # eval-only column exists; update 1's cell is blank
    assert "eval_winrate" in rows[0]
    assert rows[0]["eval_winrate"] == ""
    assert rows[1]["eval_winrate"] == "0.5"

    # a TensorBoard event file was created under tb/
    assert glob.glob(os.path.join(str(tmp_path), "tb", "events.out.*"))
