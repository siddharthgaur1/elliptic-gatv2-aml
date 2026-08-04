import numpy as np
import pytest

from src.data import ILLICIT, LICIT
from src.evaluate import comparison_table, confusion, per_timestep_f1, write_table_md


def test_confusion_matches_hand_count():
    y_true = np.array([LICIT, LICIT, ILLICIT, ILLICIT, ILLICIT])
    y_pred = np.array([LICIT, ILLICIT, ILLICIT, ILLICIT, LICIT])
    cm = confusion(y_true, y_pred)
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}


def test_confusion_all_correct_has_no_off_diagonal():
    y_true = np.array([LICIT, ILLICIT, LICIT, ILLICIT])
    cm = confusion(y_true, y_true)
    assert cm["fp"] == 0
    assert cm["fn"] == 0
    assert cm["tn"] == 2
    assert cm["tp"] == 2


def test_per_timestep_f1_perfect_predictions_score_1():
    n = 30
    y_true = np.array([ILLICIT if i % 3 == 0 else LICIT for i in range(n)])
    time_values = np.arange(n)  # already sorted, so bins == chronological chunks
    f1s = per_timestep_f1(y_true, y_true, time_values, n_bins=3)
    assert len(f1s) == 3
    assert all(f1 == 1.0 for f1 in f1s)


def test_per_timestep_f1_bin_with_no_illicit_nodes_is_none():
    y_true = np.array([LICIT] * 10)
    y_pred = np.array([LICIT] * 10)
    time_values = np.arange(10)
    f1s = per_timestep_f1(y_true, y_pred, time_values, n_bins=2)
    assert f1s == [None, None]


def test_per_timestep_f1_respects_chronological_order_not_array_order():
    # Node order in the arrays is scrambled; time_values re-sorts it before binning.
    y_true = np.array([ILLICIT, LICIT, ILLICIT, LICIT])
    y_pred = np.array([ILLICIT, LICIT, LICIT, LICIT])  # miss on the illicit node at time=2
    time_values = np.array([3, 1, 2, 0])  # array index 0 is chronologically last
    f1s = per_timestep_f1(y_true, y_pred, time_values, n_bins=2)
    # chronological order by time_values: idx3(t0,LICIT) idx1(t1,LICIT) | idx2(t2,ILLICIT,miss) idx0(t3,ILLICIT,hit)
    assert f1s[0] is None  # first bin: no illicit nodes
    assert f1s[1] == pytest.approx(2 / 3)  # bin1: precision=1 (no FP), recall=0.5 (1 of 2 illicit caught)


def test_comparison_table_rounds_and_preserves_all_models():
    run_infos = {
        "rf": {"test_metrics": {"illicit_precision": 0.91987, "illicit_recall": 0.72111,
                                 "illicit_f1": 0.80850, "macro_f1": 0.89841}},
        "gatv2": {"test_metrics": {"illicit_precision": 0.30111, "illicit_recall": 0.73128,
                                    "illicit_f1": 0.42661, "macro_f1": 0.67742}},
    }
    rows = comparison_table(run_infos)
    assert len(rows) == 2
    rf_row = next(r for r in rows if r["model"] == "rf")
    assert rf_row["illicit_f1"] == 0.8085


def test_write_table_md_sorts_by_illicit_f1_descending(tmp_path):
    rows = [
        {"model": "gcn", "illicit_precision": 0.39, "illicit_recall": 0.43, "illicit_f1": 0.41, "macro_f1": 0.68},
        {"model": "rf", "illicit_precision": 0.92, "illicit_recall": 0.72, "illicit_f1": 0.81, "macro_f1": 0.90},
    ]
    out = tmp_path / "table.md"
    write_table_md(rows, out)
    text = out.read_text()
    lines = [line for line in text.splitlines() if line.startswith("| ")][1:]  # skip header
    assert lines[0].startswith("| rf")   # higher illicit_f1 sorts first
    assert lines[1].startswith("| gcn")
