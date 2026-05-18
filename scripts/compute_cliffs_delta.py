#!/usr/bin/env python3
"""V14.A.3 — Cliff's delta on the §5.8 per-event PRM-delta arrays.

Cliff's delta measures the probability that a randomly drawn observation
from one distribution exceeds a randomly drawn observation from another,
on the [-1, 1] scale. Here we compare each per-event delta array against
the null of \Delta = 0 (a one-sample shift estimator), implemented as:

    delta_cliff = (n_neg - n_pos) / n_total   (for null = 0)

This is the standard non-parametric effect-size companion to the Wilcoxon
signed-rank test in §5.8 (which already gives p-values). Thresholds:
|delta| < 0.147 negligible, 0.147-0.33 small, 0.33-0.474 medium,
> 0.474 large.

Outputs: results/cliffs_delta_prm_injection.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_injection_prm_delta import collect_deltas, PPFG_STAG_DIRS

EMPTY_BRACKET_DIRS = [
    "results/ppfg-math500-20260516-161338-j61088009",  # seed 42
    "results/ppfg-math500-20260516-161339-j61088010",  # seed 43
    "results/ppfg-math500-20260516-053543-j61065172",  # seed 44
]
def find_empty_bracket_dirs():
    return [d for d in EMPTY_BRACKET_DIRS if Path(d).exists()]


def cliffs_delta_one_sample(values, ref: float = 0.0):
    """Cliff's delta vs a null reference value (default 0).

    For each event, the comparison is sign(value - ref). Returns
    (n_gt - n_lt) / n_total, range [-1, 1]:
      +1 = all events above ref
      -1 = all events below ref
       0 = balanced
    """
    n = len(values)
    if n == 0:
        return 0.0, {"n": 0, "n_gt": 0, "n_lt": 0, "n_eq": 0}
    arr = np.asarray(values, dtype=float)
    n_gt = int((arr > ref).sum())
    n_lt = int((arr < ref).sum())
    n_eq = int((arr == ref).sum())
    delta = (n_gt - n_lt) / n
    return delta, {"n": n, "n_gt": n_gt, "n_lt": n_lt, "n_eq": n_eq}


def magnitude(delta: float) -> str:
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def main():
    repo_root = Path(__file__).resolve().parent.parent
    summary = {"_thresholds": {
        "negligible": "|d| < 0.147",
        "small": "0.147 <= |d| < 0.33",
        "medium": "0.33 <= |d| < 0.474",
        "large": "|d| >= 0.474",
    }, "cells": {}}

    # Real PPFG-stag fragments
    deltas_real = []
    for d in PPFG_STAG_DIRS:
        evs, _, _ = collect_deltas(str(repo_root / d))
        deltas_real.extend(e["delta"] for e in evs)
    d_real, counts_real = cliffs_delta_one_sample(deltas_real, ref=0.0)
    summary["cells"]["ppfg_stag_full_fragment"] = {
        "n_events": counts_real["n"],
        "cliffs_delta_vs_zero": d_real,
        "magnitude": magnitude(d_real),
        "counts": counts_real,
        "delta_mean": float(np.mean(deltas_real)),
        "delta_median": float(np.median(deltas_real)),
    }

    # Empty-bracket control
    eb_dirs = find_empty_bracket_dirs()
    if eb_dirs:
        deltas_eb = []
        for d in eb_dirs:
            evs, _, _ = collect_deltas(str(repo_root / d))
            deltas_eb.extend(e["delta"] for e in evs)
        d_eb, counts_eb = cliffs_delta_one_sample(deltas_eb, ref=0.0)
        summary["cells"]["ppfg_stag_empty_bracket"] = {
            "n_events": counts_eb["n"],
            "cliffs_delta_vs_zero": d_eb,
            "magnitude": magnitude(d_eb),
            "counts": counts_eb,
            "_sources": eb_dirs,
            "delta_mean": float(np.mean(deltas_eb)),
            "delta_median": float(np.median(deltas_eb)),
        }
    else:
        summary["cells"]["ppfg_stag_empty_bracket"] = {
            "_error": "no empty-bracket result dirs found",
            "_searched": EMPTY_BRACKET_DIRS,
        }

    out = repo_root / "results" / "cliffs_delta_prm_injection.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
