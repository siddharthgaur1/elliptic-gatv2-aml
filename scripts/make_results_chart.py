"""Regenerate results/figures/model_comparison.png from the committed results.

Reads results/table.md's source of truth (results/*_run.json), so the chart
cannot disagree with the Results table - rerun after any retrain:

    python scripts/make_results_chart.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = RESULTS / "figures" / "model_comparison.png"

LABELS = {"rf": "Random Forest", "mlp": "MLP\n(features only)", "gatv2": "GATv2", "gcn": "GCN"}
# Graph models in one colour, feature-only models in another: the whole point of
# the chart is that the split runs along that line, not along model complexity.
COLOURS = {"rf": "#2a9d8f", "mlp": "#2a9d8f", "gatv2": "#c1666b", "gcn": "#c1666b"}


def main() -> None:
    rows = []
    for key in ("rf", "mlp", "gatv2", "gcn"):
        d = json.loads((RESULTS / f"{key}_run.json").read_text())
        t = d["test_metrics"]
        rows.append((key, t["illicit_precision"], t["illicit_recall"], t["illicit_f1"]))
    rows.sort(key=lambda r: r[3], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    names = [LABELS[r[0]] for r in rows]
    f1s = [r[3] for r in rows]
    bars = ax1.bar(names, f1s, color=[COLOURS[r[0]] for r in rows])
    ax1.set_ylabel("Illicit-F1 (test)")
    ax1.set_title("Feature-only models beat the graph models")
    ax1.set_ylim(0, 1)
    for b, v in zip(bars, f1s):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=10, fontweight="bold")

    # precision/recall scatter: where the graph models actually go wrong
    for key, p, r, _f in rows:
        ax2.scatter(r, p, s=140, color=COLOURS[key], zorder=3)
        ax2.annotate(LABELS[key].replace("\n", " "), (r, p),
                     textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax2.set_xlabel("Illicit recall")
    ax2.set_ylabel("Illicit precision")
    ax2.set_title("GATv2 trades precision for recall")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.25, zorder=0)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Elliptic Bitcoin AML — test set (time steps 35–49, seed 0)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
