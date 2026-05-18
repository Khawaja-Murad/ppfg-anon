#!/usr/bin/env python
"""Produce the headline plots: Pass@k vs k by method, and diversity vs tokens.

Usage:
    python scripts/plot_results.py --aggregate results/aggregate.csv --out figures/
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures/")
    args = parser.parse_args()

    df = pd.read_csv(args.aggregate)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    # Plot 1: Pass@k vs k, faceted by task, hue=method, error bars over seeds
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        plt.figure(figsize=(5.5, 3.5))
        sns.lineplot(
            data=sub, x="k", y="pass_at_k", hue="method", marker="o",
            errorbar=("ci", 95),
        )
        plt.xscale("log", base=2)
        plt.xlabel("k (number of samples drawn)")
        plt.ylabel("Pass@k")
        plt.title(f"Pass@k on {task}")
        plt.legend(title="Method", loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / f"pass_at_k_{task}.pdf")
        plt.savefig(out_dir / f"pass_at_k_{task}.png", dpi=150)
        plt.close()
        print(f"Saved {out_dir / f'pass_at_k_{task}.pdf'}")

    # Plot 2: answer-mode-rate (diversity collapse) vs method
    if "answer_mode_rate" in df.columns:
        agg = (
            df[df["k"] == df["k"].max()]
            .groupby(["task", "method"], as_index=False)["answer_mode_rate"].mean()
        )
        plt.figure(figsize=(5.5, 3.5))
        sns.barplot(data=agg, x="method", y="answer_mode_rate", hue="task")
        plt.ylabel("Answer mode-collapse rate")
        plt.title("Diversity collapse (lower = more diverse)")
        plt.tight_layout()
        plt.savefig(out_dir / "mode_rate.pdf")
        plt.savefig(out_dir / "mode_rate.png", dpi=150)
        plt.close()
        print(f"Saved {out_dir / 'mode_rate.pdf'}")

    # Plot 3: Pass@k vs compute (tokens_per_problem), at k=4
    sub = df[df["k"] == 4]
    if not sub.empty:
        plt.figure(figsize=(5.5, 3.5))
        sns.scatterplot(
            data=sub, x="tokens_per_problem", y="pass_at_k", hue="method",
            style="task", s=80,
        )
        plt.xscale("log")
        plt.xlabel("Tokens per problem (log)")
        plt.ylabel("Pass@4")
        plt.title("Pass@4 vs compute budget")
        plt.tight_layout()
        plt.savefig(out_dir / "pass_at_4_vs_compute.pdf")
        plt.savefig(out_dir / "pass_at_4_vs_compute.png", dpi=150)
        plt.close()
        print(f"Saved {out_dir / 'pass_at_4_vs_compute.pdf'}")


if __name__ == "__main__":
    main()
