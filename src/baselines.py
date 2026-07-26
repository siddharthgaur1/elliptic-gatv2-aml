"""Random Forest baseline -- the honest non-graph yardstick.

Same node features as the GNNs/MLP, no edges, class_weight="balanced".
"""
import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data import ILLICIT, load_data


def train_rf(seed=0, n_estimators=100, out_dir="results", data_root="data/elliptic", verbose=True):
    _, data = load_data(root=data_root, seed=seed)

    x = data.x.numpy()
    y = data.y.numpy()

    # RF has no separate early-stopping mechanism; use fit_mask + val_mask
    # together as its training set (val exists only to early-stop the NNs).
    fit_mask = (data.fit_mask | data.val_mask).numpy()
    test_mask = data.test_mask.numpy()

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(x[fit_mask], y[fit_mask])
    runtime = time.time() - t0

    pred = clf.predict(x[test_mask])
    y_true = y[test_mask]
    test_metrics = {
        "illicit_f1": f1_score(y_true, pred, pos_label=ILLICIT, zero_division=0),
        "illicit_precision": precision_score(y_true, pred, pos_label=ILLICIT, zero_division=0),
        "illicit_recall": recall_score(y_true, pred, pos_label=ILLICIT, zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(clf, model_dir / "rf.joblib")

    run_info = {
        "model": "rf",
        "seed": seed,
        "hyperparams": {"n_estimators": n_estimators, "class_weight": "balanced"},
        "test_metrics": test_metrics,
        "runtime_sec": runtime,
    }
    with open(out_dir / "rf_run.json", "w") as f:
        json.dump(run_info, f, indent=2)

    if verbose:
        print(f"[rf] TEST: {test_metrics} (runtime {runtime:.1f}s)")

    return clf, data, run_info


if __name__ == "__main__":
    train_rf()
