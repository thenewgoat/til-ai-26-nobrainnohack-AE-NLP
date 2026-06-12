"""Metrics logging + plotting for AE self-play training.

MetricsLogger accumulates per-update training metrics, mirrors them to a CSV
and to TensorBoard, renders multi-panel PNG snapshots, and renders the league
leaderboard (CSV + bar-chart PNG). All output goes under a single out_dir.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")          # headless: no display on the training box
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    """CSV + TensorBoard logging and PNG plotting for a training run."""

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.tb = SummaryWriter(os.path.join(out_dir, "tb"))
        # accumulated history: list of {"update": int, **metrics}
        self._rows = []
        # ordered union of all metric column names seen so far
        self._columns = ["update"]

    def log(self, update, metrics):
        """Record one update's metrics: append a CSV row + write TB scalars.

        Columns are the union of all keys ever seen; the CSV is rewritten in
        full each call so a late-appearing column (eval-only metrics) back-
        fills earlier rows with blank cells.
        """
        row = {"update": update}
        for key, value in metrics.items():
            row[key] = value
            if key not in self._columns:
                self._columns.append(key)
            self.tb.add_scalar(key, float(value), update)
        self._rows.append(row)
        self._rewrite_csv()

    def _rewrite_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns,
                                    restval="")
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def _series(self, key):
        """Return (updates, values) for one metric, skipping missing cells.

        Eval-only metrics are absent on most updates; those rows are dropped
        so the panel plots only the updates where the metric exists.
        """
        xs, ys = [], []
        for row in self._rows:
            if key in row and row[key] != "":
                xs.append(row["update"])
                ys.append(float(row[key]))
        return xs, ys

    def plot(self, png_path):
        """Render the accumulated history to a 6-panel PNG (spec §5.3)."""
        fig, axes = plt.subplots(3, 2, figsize=(11, 9))
        ax = axes.flatten()

        # panel 1: policy loss & value loss
        xs, ys = self._series("policy_loss")
        ax[0].plot(xs, ys, label="policy_loss", color="tab:blue")
        xs, ys = self._series("value_loss")
        ax[0].plot(xs, ys, label="value_loss", color="tab:orange")
        ax[0].set_title("losses")
        ax[0].legend()

        # panel 2: entropy
        xs, ys = self._series("entropy")
        ax[1].plot(xs, ys, color="tab:green")
        ax[1].set_title("entropy")

        # panel 3: mean return
        xs, ys = self._series("mean_return")
        ax[2].plot(xs, ys, color="tab:purple")
        ax[2].set_title("mean return")

        # panel 4: eval win-rate & eval score (only at eval updates)
        xs, ys = self._series("eval_winrate")
        ax[3].plot(xs, ys, "o-", label="eval_winrate", color="tab:red")
        ax3b = ax[3].twinx()
        xs, ys = self._series("eval_score")
        ax3b.plot(xs, ys, "s--", label="eval_score", color="tab:gray")
        ax[3].set_title("eval win-rate / score")
        ax[3].legend(loc="upper left")
        ax3b.legend(loc="upper right")

        # panel 5: rung (step line, 1->2->3)
        xs, ys = self._series("rung")
        ax[4].step(xs, ys, where="post", color="tab:brown")
        ax[4].set_title("rung")
        ax[4].set_yticks([1, 2, 3])

        # panel 6: checkpoint-pool size
        xs, ys = self._series("pool_size")
        ax[5].plot(xs, ys, color="tab:cyan")
        ax[5].set_title("checkpoint-pool size")

        for a in ax:
            a.set_xlabel("update")
            a.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=80)
        plt.close(fig)

    _LEADERBOARD_COLS = ["update", "name", "kind", "ref", "snapshot_update",
                         "learner_wins", "games", "learner_winrate",
                         "pfsp_weight"]

    def leaderboard(self, league, update, csv_path, png_path):
        """Snapshot league standings: append CSV rows + render a bar PNG.

        csv_path is a growing log (one block of rows per call); png_path is a
        fresh per-update snapshot.
        """
        members = league.members()
        weights = league.pfsp_weights(members)
        records = []
        for member, weight in zip(members, weights):
            records.append({
                "update": update,
                "name": member.name,
                "kind": member.kind,
                "ref": member.ref,
                "snapshot_update": member.update,
                "learner_wins": member.learner_wins,
                "games": member.games,
                "learner_winrate": member.learner_winrate(),
                "pfsp_weight": weight,
            })

        # append to the growing CSV (write the header only when new)
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._LEADERBOARD_COLS)
            if new_file:
                writer.writeheader()
            for rec in records:
                writer.writerow(rec)

        # render the bar chart: sorted by win-rate, scripted vs checkpoint
        ordered = sorted(records, key=lambda r: r["learner_winrate"])
        names = [r["name"] for r in ordered]
        winrates = [r["learner_winrate"] for r in ordered]
        colors = ["tab:blue" if r["kind"] == "scripted" else "tab:orange"
                  for r in ordered]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(ordered))))
        bars = ax.barh(names, winrates, color=colors)
        for bar, rec in zip(bars, ordered):
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"g={rec['games']} w={rec['pfsp_weight']:.3f}",
                    va="center", fontsize=7)
        ax.set_xlim(0, 1.25)
        ax.set_xlabel("learner win-rate vs member")
        ax.set_title(f"league leaderboard @ update {update}")
        # legend for the two kinds
        ax.barh([], [], color="tab:blue", label="scripted anchor")
        ax.barh([], [], color="tab:orange", label="checkpoint")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(png_path, dpi=80)
        plt.close(fig)

    def close(self):
        """Flush and close the TB writer. The CSV is already on disk."""
        self.tb.flush()
        self.tb.close()
