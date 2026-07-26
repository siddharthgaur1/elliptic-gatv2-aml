"""Shared training harness for GATv2 / GCN / MLP node classifiers.

Class-weighted cross-entropy (weight inversely proportional to class
frequency in the fit set), Adam + weight decay, early stopping on
validation illicit-F1. Fixed seed. Full-batch (Elliptic is small enough for
CPU full-batch training).
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data import ILLICIT, LICIT, load_data
from src.models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y, mask):
    y_m = y[mask]
    n_licit = max(int((y_m == LICIT).sum()), 1)
    n_illicit = max(int((y_m == ILLICIT).sum()), 1)
    total = n_licit + n_illicit
    w_licit = total / (2 * n_licit)
    w_illicit = total / (2 * n_illicit)
    return torch.tensor([w_licit, w_illicit], dtype=torch.float)


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[mask].argmax(dim=1)
    y_true = data.y[mask]
    f1 = f1_score(y_true, pred, pos_label=ILLICIT, zero_division=0)
    prec = precision_score(y_true, pred, pos_label=ILLICIT, zero_division=0)
    rec = recall_score(y_true, pred, pos_label=ILLICIT, zero_division=0)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    return {"illicit_f1": f1, "illicit_precision": prec, "illicit_recall": rec, "macro_f1": macro_f1}


def train_model(
    model_name,
    epochs=100,
    lr=0.01,
    weight_decay=5e-4,
    patience=15,
    hidden=64,
    seed=0,
    out_dir="results",
    data_root="data/elliptic",
    verbose=True,
):
    set_seed(seed)
    dataset, data = load_data(root=data_root, seed=seed)

    model = build_model(model_name, in_channels=data.num_features, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = class_weights(data.y, data.fit_mask)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.fit_mask], data.y[data.fit_mask], weight=weights)
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, data, data.val_mask)
        history.append({"epoch": epoch, "loss": loss.item(), **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["illicit_f1"] > best_val_f1:
            best_val_f1 = val_metrics["illicit_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"[{model_name}] epoch {epoch:03d} loss {loss.item():.4f} val_illicit_f1 {val_metrics['illicit_f1']:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"[{model_name}] early stopping at epoch {epoch} (best epoch {best_epoch}, val_f1={best_val_f1:.4f})")
            break

    runtime = time.time() - t0
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, data, data.test_mask)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / f"{model_name}.pt")

    run_info = {
        "model": model_name,
        "seed": seed,
        "hyperparams": {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "patience": patience, "hidden": hidden},
        "best_epoch": best_epoch,
        "best_val_illicit_f1": best_val_f1,
        "test_metrics": test_metrics,
        "runtime_sec": runtime,
        "history": history,
    }
    with open(out_dir / f"{model_name}_run.json", "w") as f:
        json.dump(run_info, f, indent=2)

    if verbose:
        print(f"[{model_name}] TEST: {test_metrics} (runtime {runtime:.1f}s)")

    return model, data, run_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gatv2", choices=["gatv2", "gcn", "mlp"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_model(args.model, epochs=args.epochs, seed=args.seed)
