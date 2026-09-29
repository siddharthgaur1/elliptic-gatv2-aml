"""Load the Elliptic Bitcoin dataset and build train/val/test masks.

Empirically confirmed by scripts/inspect_data.py (run once, see its output):
  - y == 0: licit    (42,019 nodes)
  - y == 1: illicit   (4,545 nodes)
  - y == 2: unknown  (157,205 nodes) -- excluded from supervision, still
    present for message passing.
  - data.train_mask / data.test_mask already implement the canonical Weber
    et al. temporal split (time steps 1-34 train, 35-49 test): 29,894 /
    16,670 labeled nodes respectively.
  - PyG's `data.x` does NOT contain the time step. PyG 2.8's
    EllipticBitcoinDataset builds `x = feat_df.loc[:, 2:]`, dropping raw
    column 0 (txId) and column 1 (time step, 1..49). `x[:, 0]` is an
    anonymised transaction feature (corr -0.026 with time). We therefore read
    the time step back from the raw features CSV into `data.time_step`
    (PyG node i == CSV row i; checked against PyG's own masks on load).
  - Validation for early stopping = the latest time steps of the training
    range holding >= 15% of the labeled training nodes: steps 29-34, 4,687
    nodes (15.7%). The cut is rounded up to a whole time step so no step is
    split between fit and val. Test set is left untouched as PyG's canonical
    test_mask.
"""
import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

LICIT, ILLICIT, UNKNOWN = 0, 1, 2
VAL_FRACTION = 0.15


def temporal_val_split(train_mask, time_col, val_fraction=VAL_FRACTION):
    """Carve the temporally-latest `val_fraction` of `train_mask` off as validation.

    Split out of `load_data` so the anti-leakage guarantee is testable without
    downloading the dataset.

    The guarantee: every validation node's time step is strictly later than
    every fit node's. The cut sits on a time-step boundary: if the
    `val_fraction` quantile lands inside a step, that whole step goes to
    validation (so val can be slightly larger than the fraction).

    `time_col` must be the real time step (`data.time_step`), not a feature.
    """
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    n_val = int(len(train_idx) * val_fraction)

    if n_val == 0:  # too few training nodes to hold any back
        return train_mask.clone(), torch.zeros_like(train_mask)

    cutoff = torch.sort(time_col[train_idx]).values[-n_val]
    val_mask = train_mask & (time_col >= cutoff)
    fit_mask = train_mask & (time_col < cutoff)
    return fit_mask, val_mask


def read_time_step(dataset, data):
    """Real 1..49 time step per node, from raw features CSV column 1.

    PyG maps txId -> node index by enumerating the features CSV rows in order,
    so node i is CSV row i. Asserted, not assumed: the time steps must
    reproduce PyG's own train/test masks exactly.
    """
    raw = pd.read_csv(dataset.raw_paths[0], header=None, usecols=[1])
    time_step = torch.tensor(raw[1].to_numpy(), dtype=torch.long)
    labeled = data.y != UNKNOWN
    assert time_step.numel() == data.num_nodes
    assert torch.equal((time_step < 35) & labeled, data.train_mask)
    assert torch.equal((time_step >= 35) & labeled, data.test_mask)
    return time_step


def load_data(root="data/elliptic", val_fraction=VAL_FRACTION, seed=0):
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]
    data.time_step = read_time_step(dataset, data)

    fit_mask, val_mask = temporal_val_split(
        data.train_mask, data.time_step, val_fraction=val_fraction
    )
    data.fit_mask = fit_mask
    data.val_mask = val_mask
    data.test_mask = data.test_mask.clone()

    return dataset, data


if __name__ == "__main__":
    _, data = load_data()
    for name in ["fit_mask", "val_mask", "test_mask"]:
        mask = getattr(data, name)
        y = data.y[mask]
        ts = data.time_step[mask]
        print(
            f"{name}: n={mask.sum().item()} illicit={int((y==ILLICIT).sum())} "
            f"licit={int((y==LICIT).sum())} steps={int(ts.min())}-{int(ts.max())}"
        )
