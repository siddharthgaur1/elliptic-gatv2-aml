"""The anti-leakage guarantee the benchmark's headline numbers rest on.

This repo's result -- Random Forest illicit-F1 0.8085 beating GATv2's 0.4266 --
is only meaningful if validation never sees nodes from before the ones the
model fits on. That property was implemented but never tested, and it was
unreachable without downloading the dataset.

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
