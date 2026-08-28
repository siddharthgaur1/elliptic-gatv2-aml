# Changelog

## [Unreleased]

Portfolio nav link (2026-08-22):

- Added a link back to the portfolio index from the README.

Test coverage and docs pass (2026-08-08):

- Added `tests/test_data_split.py`, extracting the anti-leakage invariant (`temporal_val_split`) into a pure, network-free function so it could be tested directly.
- **Caught a real bug during extraction**: `val_fraction=0` triggered a numpy negative-slice inversion (`order[-0:]` is the whole array, `order[:-0]` is empty) that silently sent every training node to validation. Unreachable at the committed `val_fraction=0.15`, so published results are unaffected — now covered by a regression test.
- README documents the invariant and the bug explicitly ("The anti-leakage invariant, and the bug testing it caught").
- Fixed a README mechanism claim ("aggregation dilutes the illicit signal") that contradicted its own numbers — GATv2's recall is the *highest* of any model, which rules out dilution; the two GNNs fail for different reasons (GATv2 over-propagates, GCN dilutes), now documented separately.
- Added hero visual / social preview card.

Convergence rerun (2026-08-06):

- Retrained all models with `--epochs 800` and early stopping (patience 15). Every run now ends on early stopping rather than the epoch budget: GATv2 at 114 (best 99), GCN at 110 (best 95), MLP at 123 (best 108).
- **MLP improved**: illicit-F1 0.6417 → 0.6558, macro-F1 0.8091 → 0.8164. It was the only model the previous 100-epoch budget actually truncated.
- **GATv2 and GCN reproduce exactly** (0.4266 and 0.4088) — their previous best epochs, 99 and 95, were genuine optima rather than the budget running out. `results/models/gatv2.pt` is byte-identical across the two runs.
- Ranking unchanged: Random Forest 0.8085 > MLP 0.6558 > GATv2 0.4266 > GCN 0.4088. Fixing the truncation raised the *baseline*, so the graph models sit slightly further behind than before.
- Reproduce command updated to `--epochs 800`; the old `--epochs 100` no longer reproduces the committed numbers.

Baseline snapshot as of the portfolio hygiene pass (2026-08-04):

- GATv2/GCN/MLP/Random Forest benchmark on the Elliptic Bitcoin dataset, with committed results (`results/metrics.json`, `results/table.md`, per-model run logs and figures).
- Random Forest wins on illicit-F1 (0.8085) against every graph model — documented honestly rather than cherry-picked.
- Added `tests/test_evaluate.py` (previously zero test coverage): confusion matrix, per-time-step F1 binning, comparison table rounding, and Markdown table sort order — the functions that produce the Results table's numbers.
- Wired `pytest` into CI (previously ruff + training smoke tests only; new tests were never exercised).
