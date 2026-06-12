"""MetricsLogger.plot renders a non-empty multi-panel PNG."""
import os

from metrics import MetricsLogger


def test_plot_writes_png(tmp_path):
    logger = MetricsLogger(str(tmp_path))
    for u in range(1, 6):
        m = {"policy_loss": 0.5 / u, "value_loss": 1.0 / u,
             "entropy": 1.7 - 0.1 * u, "mean_return": -3.0 + u,
             "rung": 1 + (u >= 3), "pool_size": u // 2,
             "anti_idle_coef": 0.05 - 0.005 * u}
        if u % 2 == 0:               # eval-only metrics on even updates
            m["eval_winrate"] = 0.4 + 0.05 * u
            m["eval_score"] = 5.0 * u
        logger.log(u, m)
    png = os.path.join(str(tmp_path), "metrics_u5.png")
    logger.plot(png)
    logger.close()
    assert os.path.exists(png) and os.path.getsize(png) > 0
