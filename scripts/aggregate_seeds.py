"""Aggregate test metrics across seeds: mean, sample std, and every per-seed value.

Seed 0 is the canonical run in results/; further seeds live in results/seeds/seed<N>/,
produced with ``main(epochs=800, seed=N, out_dir="results/seeds/seed<N>")``. Every
number comes from a committed *_run.json, and each file's own "seed" field must match
the folder it sits in, so a mislabelled run fails loudly instead of skewing the mean.

Usage: python -m scripts.aggregate_seeds
Writes results/seeds/summary.json and results/seeds/table.md.
"""

import json
import statistics
from pathlib import Path

MODELS = ["rf", "mlp", "gatv2", "gcn"]
METRICS = ["illicit_precision", "illicit_recall", "illicit_f1", "macro_f1"]


def seed_dirs(root: Path) -> dict[int, Path]:
    dirs = {0: root}
    for d in sorted((root / "seeds").glob("seed*")):
        if d.is_dir() and d.name[4:].isdigit():
            dirs[int(d.name[4:])] = d
    return dirs


def load_runs(root: Path) -> dict[int, dict[str, dict]]:
    runs: dict[int, dict[str, dict]] = {}
    for seed, d in seed_dirs(root).items():
        files = {m: d / f"{m}_run.json" for m in MODELS}
        if not all(f.is_file() for f in files.values()):
            continue  # a seed still training is skipped, never half-counted
        runs[seed] = {}
        for model, f in files.items():
            run = json.loads(f.read_text())
            if run["seed"] != seed:
                raise ValueError(f"{f} says seed {run['seed']} but sits in the seed-{seed} slot")
            runs[seed][model] = run
    return runs


def summarize(runs: dict[int, dict[str, dict]]) -> dict:
    seeds = sorted(runs)
    out = {"seeds": seeds, "n": len(seeds), "models": {}}
    for model in MODELS:
        entry = {}
        for metric in METRICS:
            values = [runs[s][model]["test_metrics"][metric] for s in seeds]
            entry[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "per_seed": dict(zip(map(str, seeds), values)),
            }
        entry["best_epoch"] = {str(s): runs[s][model].get("best_epoch") for s in seeds}
        out["models"][model] = entry
    return out


def table_md(summary: dict) -> str:
    seeds = summary["seeds"]
    lines = [
        f"Test set (time steps 35-49), {summary['n']} seeds ({', '.join(map(str, seeds))}). "
        "Mean ± sample std; per-seed illicit-F1 in the last column.",
        "",
        "| Model | Illicit-P | Illicit-R | Illicit-F1 | Macro-F1 | Illicit-F1 per seed |",
        "|---|---|---|---|---|---|",
    ]
    order = sorted(MODELS, key=lambda m: -summary["models"][m]["illicit_f1"]["mean"])
    for model in order:
        e = summary["models"][model]
        cells = [f"{e[m]['mean']:.4f} ± {e[m]['std']:.4f}" for m in METRICS]
        per_seed = ", ".join(f"{e['illicit_f1']['per_seed'][str(s)]:.4f}" for s in seeds)
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {per_seed} |")
    return "\n".join(lines) + "\n"


def main(root: str = "results") -> dict:
    root_path = Path(root)
    summary = summarize(load_runs(root_path))
    seeds_dir = root_path / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    (seeds_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (seeds_dir / "table.md").write_text(table_md(summary))
    print(table_md(summary))
    return summary


if __name__ == "__main__":
    main()
