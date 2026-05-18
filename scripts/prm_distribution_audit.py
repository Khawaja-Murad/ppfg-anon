#!/usr/bin/env python
"""PRM-distribution audit for §5.1.5 reviewer defense (Day-10 R1).

Reads independent-baseline trajectories for each n=500 architecture cell
(Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, DeepSeek-R1-Distill-Qwen-7B)
and reports the PRM-score distribution Math-Shepherd-Mistral-7B places on
each architecture's decoded steps. Defensive against reviewer concern that
the cross-arch PPFG-parity finding partially measures PRM mis-calibration
on out-of-Mistral-family decoders.

Output:
  - results/prm_distribution_summary.json (machine-readable)
  - console table of mean ± std, quantiles, prune-trigger rate, extract-quality rate
  - pairwise KS-test p-values vs Qwen anchor

Usage:
  PYTHONPATH=src python scripts/prm_distribution_audit.py
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

try:
    from scipy.stats import ks_2samp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

TAU_KILL = 0.4
TAU_EXTRACT = 0.6

INDEP_CELLS = {
    "Qwen2.5-7B-Instruct": [
        "results/independent-math500-20260514-161757-j60966078",
        "results/independent-math500-20260514-161757-j60966131",
        "results/independent-math500-20260514-161757-j60966148",
    ],
    "LLaMA-3.1-8B-Instruct": [
        "results/independent-math500-20260515-004010-j60993304",
        "results/independent-math500-20260515-004009-j60993314",
        "results/independent-math500-20260515-004424-j60993330",
    ],
    "DeepSeek-R1-Distill-Qwen-7B": [
        "results/independent-math500-20260515-004009-j60993307",
        "results/independent-math500-20260515-004010-j60993315",
        "results/independent-math500-20260515-004424-j60993331",
    ],
}


def quantile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def stats(xs):
    n = len(xs)
    if n == 0:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = var ** 0.5
    return {
        "n_steps": n,
        "mean": mean,
        "std": std,
        "q10": quantile(xs, 0.10),
        "q25": quantile(xs, 0.25),
        "q50": quantile(xs, 0.50),
        "q75": quantile(xs, 0.75),
        "q90": quantile(xs, 0.90),
        "prune_trigger_rate": sum(1 for x in xs if x < TAU_KILL) / n,
        "extract_quality_rate": sum(1 for x in xs if x >= TAU_EXTRACT) / n,
    }


def aggregate_prm_scores(cell_dirs):
    """Return flat list of every per-step PRM score across all chains in all cells."""
    xs = []
    n_chains = 0
    n_problems = 0
    for d in cell_dirs:
        traj_path = Path(d) / "trajectories.json"
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)
        with traj_path.open() as f:
            recs = json.load(f)
        n_problems += len(recs)
        for rec in recs:
            for ch in rec["population"]["chains"]:
                n_chains += 1
                for s in (ch.get("prm_scores") or []):
                    if s is None:
                        continue
                    xs.append(float(s))
    return xs, n_chains, n_problems


def main():
    out = {
        "tau_kill": TAU_KILL,
        "tau_extract": TAU_EXTRACT,
        "scipy_available": HAVE_SCIPY,
        "architectures": {},
    }
    all_xs = {}
    for arch, dirs in INDEP_CELLS.items():
        xs, n_chains, n_problems = aggregate_prm_scores(dirs)
        s = stats(xs)
        s["n_chains"] = n_chains
        s["n_problems"] = n_problems
        s["n_cells"] = len(dirs)
        out["architectures"][arch] = s
        all_xs[arch] = xs

    if HAVE_SCIPY:
        anchor = "Qwen2.5-7B-Instruct"
        out["ks_test_vs_anchor"] = {}
        for arch in INDEP_CELLS:
            if arch == anchor:
                continue
            stat, p = ks_2samp(all_xs[anchor], all_xs[arch])
            out["ks_test_vs_anchor"][arch] = {"ks_statistic": float(stat), "p_value": float(p)}

    Path("results/prm_distribution_summary.json").write_text(json.dumps(out, indent=2))

    # Pretty console table.
    print(f"\nPRM-distribution audit on independent-baseline n=500 trajectories")
    print(f"  τ_kill = {TAU_KILL}    τ_extract = {TAU_EXTRACT}")
    print()
    cols = ["Architecture", "n_steps", "mean", "std", "q10", "q25", "q50", "q75", "q90",
            "prune-trigger", "extract-quality"]
    fmt = "  {:<30} {:>8} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6} {:>6} {:>14} {:>15}"
    print(fmt.format(*cols))
    print("  " + "-" * 130)
    for arch in INDEP_CELLS:
        s = out["architectures"][arch]
        print(fmt.format(
            arch, s["n_steps"],
            f"{s['mean']:.3f}", f"{s['std']:.3f}",
            f"{s['q10']:.3f}", f"{s['q25']:.3f}", f"{s['q50']:.3f}", f"{s['q75']:.3f}", f"{s['q90']:.3f}",
            f"{s['prune_trigger_rate']:.3f}", f"{s['extract_quality_rate']:.3f}",
        ))

    if HAVE_SCIPY:
        print(f"\nKS-test vs Qwen2.5-7B-Instruct anchor:")
        for arch, r in out["ks_test_vs_anchor"].items():
            print(f"  {arch:<30}  KS={r['ks_statistic']:.4f}  p={r['p_value']:.3e}")

    print(f"\nSaved: results/prm_distribution_summary.json")


if __name__ == "__main__":
    main()
