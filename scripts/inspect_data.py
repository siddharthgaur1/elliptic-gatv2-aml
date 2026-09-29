"""Inspect the PyG EllipticBitcoinDataset: shapes, label encoding, temporal split.

Run this BEFORE writing any code that assumes a label encoding or split —
the numbers below are read straight off the tensors, not from memory.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset


def main():
    dataset = EllipticBitcoinDataset(root="data/elliptic")
    data = dataset[0]

    print("=== Shapes ===")
    print(f"num_nodes:    {data.num_nodes}")
    print(f"num_edges:    {data.num_edges}")
    print(f"num_features: {data.num_features}")

    print("\n=== y label values ===")
    uniq, counts = torch.unique(data.y, return_counts=True)
    for u, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  y == {u}: {c} nodes")

    print("\n=== train_mask / test_mask present? ===")
    has_train = hasattr(data, "train_mask")
    has_test = hasattr(data, "test_mask")
    print(f"train_mask: {has_train}, test_mask: {has_test}")

    # PyG builds x = feat_df.loc[:, 2:], dropping raw CSV column 0 (txId) and
    # column 1 (time step). So x[:, 0] is NOT time -- show that, then read the
    # real time step back from the raw CSV the way src.data does.
    raw = pd.read_csv(dataset.raw_paths[0], header=None)
    print("\n=== raw CSV vs PyG x ===")
    print(f"raw CSV columns: {raw.shape[1]}, PyG x columns: {data.num_features}")
    same = np.allclose(data.x.numpy(), raw.loc[:, 2:].to_numpy(dtype=np.float32), atol=1e-5)
    print(f"x == raw CSV columns 2.. (row-aligned): {same}")
    time_step = torch.tensor(raw[1].to_numpy())
    x0 = data.x[:, 0]
    corr = np.corrcoef(x0.numpy(), time_step.numpy())[0, 1]
    print(f"x[:, 0]: {torch.unique(x0).numel()} unique values, corr with time step {corr:.3f}")
    print(f"raw time step: min {int(time_step.min())}, max {int(time_step.max())}, "
          f"{torch.unique(time_step).numel()} unique")

    if has_train and has_test:
        print("\n=== train_mask/test_mask stats ===")
        print(f"train_mask sum: {data.train_mask.sum().item()}")
        print(f"test_mask sum: {data.test_mask.sum().item()}")
        train_ts = time_step[data.train_mask]
        test_ts = time_step[data.test_mask]
        print(f"train time steps: {int(train_ts.min())}..{int(train_ts.max())}")
        print(f"test time steps:  {int(test_ts.min())}..{int(test_ts.max())}")

        # Label distribution within masks
        for name, mask in [("train", data.train_mask), ("test", data.test_mask)]:
            y_m = data.y[mask]
            u, c = torch.unique(y_m, return_counts=True)
            print(f"{name} label dist: " + ", ".join(f"{uu.item()}:{cc.item()}" for uu, cc in zip(u, c)))

    print("\n=== dataset docstring / class info ===")
    print(dataset)
    print(data)


if __name__ == "__main__":
    main()
