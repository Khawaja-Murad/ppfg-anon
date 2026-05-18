#!/usr/bin/env python
"""Aggregate metrics across runs in `results/` and produce comparison tables + plots.

Usage:
    python scripts/eval.py --runs_dir results/
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd


def collect_metrics(runs_dir: Path) -> pd.DataFrame:
    """Walk `runs_dir/*/metrics.json` and assemble a long-format DataFrame."""
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        with metrics_path.open() as f:
            metrics = json.load(f)
        with config_path.open() as f:
            cfg = json.load(f)
        for k, val in metrics["pass_at_k"].items():
            rows.append({
                "run": run_dir.name,
                "method": metrics["method"],
                "task": cfg["task"]["name"],
                "n_problems": metrics["n_problems"],
                "n_chains": cfg["population"]["n_chains"],
                "seed": cfg["experiment"]["seed"],
                "tokens_per_problem": metrics["tokens_per_problem"],
                "k": int(k),
                "pass_at_k": val,
                "answer_mode_rate": metrics["diversity"].get("answer_mode_rate"),
                "semantic_diversity": metrics["diversity"].get("semantic_diversity"),
                "distinct_4": metrics["diversity"].get("distinct_4"),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/aggregate.csv")
    args = parser.parse_args()

    df = collect_metrics(Path(args.runs_dir))
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

    # Summary table: mean Pass@k by method, task, k (averaged over seeds)
    summary = (
        df.groupby(["task", "method", "k"], as_index=False)["pass_at_k"]
        .agg(["mean", "std", "count"])
    )
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Wrote summary to {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
