"""scripts.aggregate_seeds: a mislabelled or half-finished seed must never reach the mean."""

import json
import statistics

import pytest

from scripts.aggregate_seeds import MODELS, load_runs, main


def _write_seed(dirpath, seed, f1_by_model):
    dirpath.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        (dirpath / f"{model}_run.json").write_text(json.dumps({
            "seed": seed,
            "best_epoch": 10 + seed,
            "test_metrics": {
                "illicit_precision": 0.5,
                "illicit_recall": 0.5,
                "illicit_f1": f1_by_model[model],
                "macro_f1": 0.7,
            },
        }))


def _f1(value):
    return {m: value for m in MODELS}


def test_mean_and_sample_std_over_complete_seeds(tmp_path):
    _write_seed(tmp_path, 0, _f1(0.40))
    _write_seed(tmp_path / "seeds" / "seed1", 1, _f1(0.20))
    _write_seed(tmp_path / "seeds" / "seed2", 2, _f1(0.30))

    summary = main(str(tmp_path))

    gat = summary["models"]["gatv2"]["illicit_f1"]
    assert summary["seeds"] == [0, 1, 2]
    assert gat["mean"] == pytest.approx(0.30)
    assert gat["std"] == pytest.approx(statistics.stdev([0.40, 0.20, 0.30]))
    assert (tmp_path / "seeds" / "table.md").read_text().count("| gatv2 |") == 1


def test_seed_still_training_is_skipped_not_half_counted(tmp_path):
    _write_seed(tmp_path, 0, _f1(0.40))
    partial = tmp_path / "seeds" / "seed3"
    _write_seed(partial, 3, _f1(0.10))
    (partial / "rf_run.json").unlink()  # rf not written yet

    assert sorted(load_runs(tmp_path)) == [0]


def test_file_whose_seed_disagrees_with_its_folder_fails(tmp_path):
    _write_seed(tmp_path, 0, _f1(0.40))
    _write_seed(tmp_path / "seeds" / "seed1", 2, _f1(0.20))  # copied from seed 2 by mistake

    with pytest.raises(ValueError, match="seed-1 slot"):
        load_runs(tmp_path)
