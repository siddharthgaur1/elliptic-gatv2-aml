# Changelog

## [Unreleased]

Baseline snapshot as of the portfolio hygiene pass (2026-08-04):

- GATv2/GCN/MLP/Random Forest benchmark on the Elliptic Bitcoin dataset, with committed results (`results/metrics.json`, `results/table.md`, per-model run logs and figures).
- Random Forest wins on illicit-F1 (0.8085) against every graph model — documented honestly rather than cherry-picked.
- Added `tests/test_evaluate.py` (previously zero test coverage): confusion matrix, per-time-step F1 binning, comparison table rounding, and Markdown table sort order — the functions that produce the Results table's numbers.
- Wired `pytest` into CI (previously ruff + training smoke tests only; new tests were never exercised).
