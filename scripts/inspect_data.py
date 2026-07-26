"""Inspect the PyG EllipticBitcoinDataset: shapes, label encoding, temporal split.

Run this BEFORE writing any code that assumes a label encoding or split —
the numbers below are read straight off the tensors, not from memory.
"""
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

    # First feature column is the time step (per Elliptic paper / PyG docs).
    time_step = data.x[:, 0]
    print("\n=== time step (feature col 0) range ===")
    print(f"min: {time_step.min().item()}, max: {time_step.max().item()}")
    # time step is stored normalized-looking or raw int 1..49 -- print raw uniques
    uniq_ts = torch.unique(time_step)
    print(f"num unique time steps: {uniq_ts.numel()}")
    print(f"unique values (sorted): {sorted(uniq_ts.tolist())}")

    if has_train and has_test:
        print("\n=== train_mask/test_mask stats ===")
        print(f"train_mask sum: {data.train_mask.sum().item()}")
        print(f"test_mask sum: {data.test_mask.sum().item()}")
        # Check what time steps fall in train vs test mask
        train_ts = time_step[data.train_mask]
        test_ts = time_step[data.test_mask]
        print(f"train time steps range: {train_ts.min().item()}..{train_ts.max().item()}")
        print(f"test time steps range: {test_ts.min().item()}..{test_ts.max().item()}")
        print(f"train unique ts: {sorted(torch.unique(train_ts).tolist())}")
        print(f"test unique ts: {sorted(torch.unique(test_ts).tolist())}")

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
