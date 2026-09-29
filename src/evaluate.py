"""Evaluation utilities: comparison table, confusion matrices, per-time-step F1 curve.

The per-time-step curve groups test nodes by their real 1..49 time step
(`data.time_step`, read from the raw CSV by src.data -- PyG's `x` does not
contain it), one point per canonical test step 35..49.
"""
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.data import ILLICIT
from src.models import build_model


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


def per_timestep_f1(y_true, y_pred, time_steps):
    """Illicit-F1 per distinct time step, in ascending step order.

    None for a step with no illicit nodes (F1 undefined).
    """
    f1s = []
    for step in np.unique(time_steps):
        in_step = time_steps == step
        yt, yp = y_true[in_step], y_pred[in_step]
        if (yt == ILLICIT).sum() == 0:
            f1s.append(None)
        else:
            f1s.append(f1_score(yt, yp, pos_label=ILLICIT, zero_division=0))
    return f1s
