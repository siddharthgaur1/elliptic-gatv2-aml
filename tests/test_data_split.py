"""The anti-leakage guarantee the benchmark's headline numbers rest on.

The early-stopping slice must be temporally after the nodes the model fits on.
That needs two things: the split orders by the column it is given, and the
column it is given is the real time step (PyG's x[:, 0] is not -- see the
read_time_step tests at the bottom).

No network, no dataset: `temporal_val_split` takes a mask and a time column.
"""

from __future__ import annotations

import torch

from src.data import temporal_val_split


def _masks(n=100, n_train=80):
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_train] = True
    return train_mask


def test_every_validation_node_is_at_or_after_every_fit_node():
    """The guarantee itself: no fit node may postdate a validation node."""
    train_mask = _masks()
    time_col = torch.arange(100, dtype=torch.float)  # already chronological

    fit, val = temporal_val_split(train_mask, time_col, val_fraction=0.15)

    assert fit.sum() > 0 and val.sum() > 0
    assert time_col[fit].max() <= time_col[val].min()


def test_the_guarantee_holds_when_time_is_shuffled_relative_to_index():
    """Ordering must come from the time column, not from array position.

    If the split ever silently used index order, this is the test that fails.
    """
    train_mask = _masks()
    g = torch.Generator().manual_seed(0)
    time_col = torch.randperm(100, generator=g).float()

    fit, val = temporal_val_split(train_mask, time_col, val_fraction=0.15)

    assert time_col[fit].max() <= time_col[val].min()


def test_fit_and_val_partition_train_with_no_overlap_and_no_loss():
    train_mask = _masks(n=100, n_train=80)
    time_col = torch.arange(100, dtype=torch.float)

    fit, val = temporal_val_split(train_mask, time_col, val_fraction=0.15)

    assert not (fit & val).any(), "a node cannot be both fitted on and validated on"
    assert torch.equal(fit | val, train_mask), "every train node lands in exactly one"


def test_nothing_outside_the_train_mask_is_ever_selected():
    """Test nodes must not leak into fit or val."""
    train_mask = _masks(n=100, n_train=80)
    time_col = torch.arange(100, dtype=torch.float)

    fit, val = temporal_val_split(train_mask, time_col, val_fraction=0.15)

    held_out = ~train_mask
    assert not (fit & held_out).any()
    assert not (val & held_out).any()


def test_validation_size_follows_the_fraction():
    train_mask = _masks(n=100, n_train=80)
    time_col = torch.arange(100, dtype=torch.float)

    _, val = temporal_val_split(train_mask, time_col, val_fraction=0.25)

    assert int(val.sum()) == 20  # 25% of 80


def test_a_fraction_too_small_to_hold_a_node_keeps_everything_fittable():
    """Degenerate case: rather than an empty fit set or a crash, val is empty."""
    train_mask = _masks(n=10, n_train=4)
    time_col = torch.arange(10, dtype=torch.float)

    fit, val = temporal_val_split(train_mask, time_col, val_fraction=0.01)

    assert int(val.sum()) == 0
    assert torch.equal(fit, train_mask)


def test_ties_never_split_a_time_step_between_fit_and_val():
    """Real time steps are discrete (1..49), so many nodes share one.

    The old argsort-based split could put part of a step in fit and the rest in
    val; the cut must land on a step boundary so val is strictly later.
    """
    train_mask = _masks(n=100, n_train=80)
    time_col = torch.arange(100) // 10  # 10 nodes per step; 15% of 80 = 12 -> mid-step

    fit, val = temporal_val_split(train_mask, time_col, val_fraction=0.15)

    assert time_col[fit].max() < time_col[val].min()
    assert int(val.sum()) == 20  # rounded up to two whole steps (6 and 7)


class _FakeDataset:
    def __init__(self, raw_path):
        self.raw_paths = [str(raw_path)]


def _fake_graph(tmp_path, time_steps, feature0):
    """Mimic PyG: x drops raw CSV columns 0 (txId) and 1 (time step)."""
    import pandas as pd
    from torch_geometric.data import Data

    n = len(time_steps)
    pd.DataFrame({0: range(1000, 1000 + n), 1: time_steps, 2: feature0}).to_csv(
        tmp_path / "feat.csv", header=False, index=False
    )
    ts = torch.tensor(time_steps)
    y = torch.zeros(n, dtype=torch.long)
    data = Data(x=torch.tensor(feature0, dtype=torch.float).unsqueeze(1), y=y,
                train_mask=ts < 35, test_mask=ts >= 35)
    return _FakeDataset(tmp_path / "feat.csv"), data


def test_time_step_comes_from_raw_csv_not_feature_column_0(tmp_path):
    """The original bug: x[:, 0] was assumed to be time. It is not."""
    from src.data import read_time_step

    steps = [1, 5, 20, 34, 35, 49]
    feature0 = [0.9, -0.3, 0.1, -2.0, 0.5, 0.0]  # anonymised, unrelated to time
    dataset, data = _fake_graph(tmp_path, steps, feature0)

    assert torch.equal(read_time_step(dataset, data), torch.tensor(steps))


def test_misaligned_csv_is_rejected(tmp_path):
    import pytest

    from src.data import read_time_step

    dataset, data = _fake_graph(tmp_path, [1, 5, 35, 49], [0.0] * 4)
    data.train_mask = ~data.train_mask  # PyG masks no longer agree with the CSV
    with pytest.raises(AssertionError):
        read_time_step(dataset, data)
