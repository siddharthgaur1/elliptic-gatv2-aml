"""Evaluation utilities: comparison table, confusion matrices, per-time-step F1 curve.

PyG's EllipticBitcoinDataset exposes time step only as a standardized
(z-scored) float (feature column 0), not the raw integer 1..49 used in the
original paper. To plot a "per-time-step" curve we rank-order test nodes by
that value and bin them into 15 equal-count quantile bins -- one bin per
canonical step 35..49, in chronological order. Bin boundaries are therefore
approximate, not exact integer time-step membership, and this is called out
in the README.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.data import ILLICIT
from src.models import build_model
import joblib


N_TEST_STEPS = 15  # canonical steps 35..49


@torch.no_grad()
def nn_predict(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    return logits[mask].argmax(dim=1).numpy()


def load_trained_nn(model_name, data, hidden=64, out_dir="results"):
    model = build_model(model_name, in_channels=data.num_features, hidden=hidden)
    # weights_only=True: state_dict is plain tensors, and this is our own
    # committed artifact -- no untrusted pickle content.
    state = torch.load(Path(out_dir) / "models" / f"{model_name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_trained_rf(out_dir="results"):
    # joblib.load unpickles -- fine here, this is our own committed artifact.
    return joblib.load(Path(out_dir) / "models" / "rf.joblib")


def comparison_table(run_infos):
    """run_infos: dict[model_name] -> run_info dict (from *_run.json)."""
    rows = []
    for name, info in run_infos.items():
        m = info["test_metrics"]
        rows.append(
            {
                "model": name,
                "illicit_precision": round(m["illicit_precision"], 4),
                "illicit_recall": round(m["illicit_recall"], 4),
                "illicit_f1": round(m["illicit_f1"], 4),
                "macro_f1": round(m["macro_f1"], 4),
            }
        )
    return rows


def write_table_md(rows, path):
    header = "| Model | Illicit-P | Illicit-R | Illicit-F1 | Macro-F1 |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in sorted(rows, key=lambda r: -r["illicit_f1"]):
        lines.append(
            f"| {r['model']} | {r['illicit_precision']:.4f} | {r['illicit_recall']:.4f} | "
            f"{r['illicit_f1']:.4f} | {r['macro_f1']:.4f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}


def per_timestep_f1(y_true, y_pred, time_values, n_bins=N_TEST_STEPS):
    """Bin test nodes into n_bins chronological quantile bins, compute illicit-F1 per bin."""
    order = np.argsort(time_values)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    bins = np.array_split(np.arange(len(order)), n_bins)
    f1s = []
    for b in bins:
        if len(b) == 0:
            f1s.append(None)
            continue
        yt, yp = y_true_sorted[b], y_pred_sorted[b]
        if (yt == ILLICIT).sum() == 0:
            f1s.append(None)  # no illicit nodes in this bin -- undefined
        else:
            f1s.append(f1_score(yt, yp, pos_label=ILLICIT, zero_division=0))
    return f1s
