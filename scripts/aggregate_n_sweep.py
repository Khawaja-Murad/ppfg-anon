"""Reproduce the population-size sweep (paper Appendix: Table `tab:n-sweep`).

Matched independent and PPFG-stag populations at N in {16,32,64,128,256}
on the first-100 MATH500 slice, 3 seeds per cell. Reports PPFG-stag minus
independent in percentage points for Pass@{1,2,4,8} and the answer-mode rate.

Both arms of every cell are read from `metrics.json`, so each delta compares
two runs scored under the same comparator. Deltas are the interpretable
quantity; absolute levels depend on comparator availability at scoring time.

Usage:
  PYTHONPATH=src python scripts/aggregate_n_sweep.py \
      [--results_dir results] [--out results/n_sweep_summary.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

N_VALUES = (16, 32, 64, 128, 256)
SEEDS = (42, 43, 44)
KS = ("1", "2", "4", "8")


def cell(results_dir: Path, experiment_name: str) -> dict | None:
    """Aggregate metrics.json across seeds for one experiment name."""
    found: dict[int, dict] = {}
    for cfg_path in sorted(results_dir.glob("*/config.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        exp = cfg.get("experiment", {})
        if exp.get("name") != experiment_name:
            continue
        seed = exp.get("seed")
        if seed not in SEEDS or seed in found:
            continue
        metrics_path = cfg_path.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        found[seed] = json.loads(metrics_path.read_text())
    if len(found) != len(SEEDS):
        return None
    runs = [found[s] for s in SEEDS]
    out = {k: st.mean(r["pass_at_k"][k] for r in runs) for k in KS}
    out["sd_1"] = st.stdev([r["pass_at_k"]["1"] for r in runs])
    out["mode"] = st.mean(r["diversity"]["answer_mode_rate"] for r in runs)
    out["n_seeds"] = len(runs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/n_sweep_summary.json")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    summary: dict = {"seeds": list(SEEDS), "n_problems": 100, "cells": {}}
    rows: dict[str, dict[int, float]] = {k: {} for k in (*KS, "mode")}

    for N in N_VALUES:
        indep = cell(results_dir, f"qwen_n100_indep_N{N}")
        ppfg = cell(results_dir, f"qwen_n100_ppfg_stag_N{N}")
        if not indep or not ppfg:
            print(f"  !! N={N}: incomplete ({'indep' if not indep else ''}"
                  f"{' ppfg' if not ppfg else ''}) -- skipping")
            continue
        summary["cells"][str(N)] = {"indep": indep, "ppfg_stag": ppfg}
        for metric in (*KS, "mode"):
            rows[metric][N] = 100 * (ppfg[metric] - indep[metric])

    summary["deltas_pp"] = {m: {str(n): v for n, v in d.items()}
                            for m, d in rows.items()}

    present = [N for N in N_VALUES if N in rows["1"]]
    print("\n  delta (pp) | " + " | ".join(f"N={N:>4d}" for N in present))
    for metric in (*KS, "mode"):
        label = f"Pass@{metric}" if metric != "mode" else "mode rate"
        print(f"  {label:<10s} | "
              + " | ".join(f"{rows[metric][N]:+6.2f}" for N in present))

    if present:
        biggest = max(((abs(rows[k][N]), k, N) for k in KS for N in present))
        print(f"\n  binding Pass@k cell: N={biggest[2]} Pass@{biggest[1]} "
              f"at {rows[biggest[1]][biggest[2]]:+.2f} pp")
        any_positive = any(rows[k][N] > 0 for k in KS for N in present)
        print(f"  any positive Pass@k delta at any N: {any_positive}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
