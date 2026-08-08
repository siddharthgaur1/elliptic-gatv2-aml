"""Load the Elliptic Bitcoin dataset and build train/val/test masks.

Empirically confirmed by scripts/inspect_data.py (run once, see its output):
  - y == 0: licit    (42,019 nodes)
  - y == 1: illicit   (4,545 nodes)
  - y == 2: unknown  (157,205 nodes) -- excluded from supervision, still
    present for message passing.
  - data.train_mask / data.test_mask already implement the canonical Weber
    et al. temporal split (time steps 1-34 train, 35-49 test): 29,894 /
    16,670 labeled nodes respectively.
  - Feature column 0 is a standardized (z-scored) time step, not a raw
    1..49 integer, but it is monotonic with real time order. We rank-order
    the train_mask nodes by this value and carve the temporally-latest 15%
    off as a validation slice (early stopping), instead of the raw
    "35-39/40-49" split suggested in the spec, because PyG does not expose
    the raw integer time step -- only a scaled version of it. Test set is
    left untouched as PyG's canonical test_mask.
"""
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

LICIT, ILLICIT, UNKNOWN = 0, 1, 2
VAL_FRACTION = 0.15



def temporal_val_split(train_mask, time_col, val_fraction=VAL_FRACTION):
    """Carve the temporally-latest `val_fraction` of `train_mask` off as validation.

    Split out of `load_data` so the anti-leakage guarantee is testable without
    downloading the dataset: this is the claim the whole benchmark rests on, and
    it was previously reachable only through a network call.

    The guarantee: every validation node's time step is >= every fit node's.
    Early stopping therefore never sees a node from before the nodes it trains
    on, which is what makes the reported illicit-F1 a forward-looking number
    rather than an interpolated one.

    `time_col` is the standardized (z-scored) time step -- monotonic with real
    time, which is all the ordering needs.
    """
    train_mask = train_mask.clone()
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    order = torch.argsort(time_col[train_idx])  # ascending: earliest -> latest
    n_val = int(len(order) * val_fraction)

    if n_val == 0:  # too few training nodes to hold any back
        fit_idx, val_idx = train_idx[order], train_idx[:0]
    else:
        val_idx = train_idx[order[-n_val:]]   # temporally latest slice
        fit_idx = train_idx[order[:-n_val]]   # remainder gets gradient updates

    fit_mask = torch.zeros_like(train_mask)
    fit_mask[fit_idx] = True
    val_mask = torch.zeros_like(train_mask)
    val_mask[val_idx] = True
    return fit_mask, val_mask


def load_data(root="data/elliptic", val_fraction=VAL_FRACTION, seed=0):
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    fit_mask, val_mask = temporal_val_split(
        data.train_mask, data.x[:, 0], val_fraction=val_fraction
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
        print(f"{name}: n={mask.sum().item()} illicit={int((y==ILLICIT).sum())} licit={int((y==LICIT).sum())}")
