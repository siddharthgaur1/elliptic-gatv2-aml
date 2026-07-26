"""Train GATv2, GCN, MLP, RF; write results/metrics.json, results/table.md,
confusion matrices and the per-time-step F1 curve figure.

Usage: python -m scripts.run_all [--epochs N] [--seed N]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.baselines import train_rf
from src.data import load_data
from src.evaluate import (
    comparison_table,
    confusion,
    nn_predict,
    per_timestep_f1,
    write_table_md,
)
from src.train import train_model

SEED = 0


def main(epochs=100, seed=SEED, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_infos = {}
    trained = {}

    for name in ["gatv2", "gcn", "mlp"]:
        model, data, info = train_model(name, epochs=epochs, seed=seed, out_dir=str(out_dir))
        run_infos[name] = info
        trained[name] = model

    _, _rf_data, rf_info = train_rf(seed=seed, out_dir=str(out_dir))
    run_infos["rf"] = rf_info

    # ---- comparison table ----
    rows = comparison_table(run_infos)
    write_table_md(rows, out_dir / "table.md")

    # ---- confusion matrices + assemble metrics.json ----
    _, data = load_data(seed=seed)
    test_mask = data.test_mask
    y_true = data.y[test_mask].numpy()

    confusions = {}
    predictions = {}
    for name, model in trained.items():
        pred = nn_predict(model, data, test_mask)
        confusions[name] = confusion(y_true, pred)
        predictions[name] = pred

    from src.evaluate import load_trained_rf

    rf_clf = load_trained_rf(str(out_dir))
    rf_pred = rf_clf.predict(data.x[test_mask].numpy())
    confusions["rf"] = confusion(y_true, rf_pred)
    predictions["rf"] = rf_pred

    metrics = {
        "seed": seed,
        "comparison_table": rows,
        "confusion_matrices": confusions,
        "run_info": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in run_infos.items()},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- per-time-step F1 curve ----
    time_values = data.x[test_mask, 0].numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    curve_data = {}
    for name, pred in predictions.items():
        f1s = per_timestep_f1(y_true, pred, time_values)
        curve_data[name] = f1s
        xs = list(range(35, 35 + len(f1s)))
        ys = [f1 if f1 is not None else np.nan for f1 in f1s]
        ax.plot(xs, ys, marker="o", label=name)
    ax.axvline(43, color="gray", linestyle="--", alpha=0.6, label="step 43 (dark-market shutdown)")
    ax.set_xlabel("approx. time step (chronological quantile bin, canonical steps 35-49)")
    ax.set_ylabel("illicit F1")
    ax.set_title("Per-time-step illicit F1 on test set")
    ax.legend()
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "per_timestep_f1.png", dpi=120)
    plt.close(fig)

    with open(out_dir / "per_timestep_f1.json", "w") as f:
        json.dump(curve_data, f, indent=2)

    print("\n=== Comparison table ===")
    print((out_dir / "table.md").read_text())
    print("Wrote results/metrics.json, results/table.md, results/figures/per_timestep_f1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(epochs=args.epochs, seed=args.seed)
