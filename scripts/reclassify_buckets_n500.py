#!/usr/bin/env python
"""Recompute the four-bucket targeting classification on the FULL n=500
PPFG-stag corpus (322 events across 3 Qwen seeds), not just the first-100
slice (55 events) used in §5.3.1 Table 2.

Reuses the canonical first-annotator classifier and event-gathering logic
from scripts/inter_annotator_classify.py so the bucket definitions are
byte-identical to the n=100 analysis. Buckets are deterministic functions
of saved trajectory data; no GPU, no new generation.

Usage:
  PYTHONPATH=src python scripts/reclassify_buckets_n500.py

Output:
  results/bucket_classification_n500.json
"""
from __future__ import annotations
import importlib.util, json
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "iac", str(Path(__file__).with_name("inter_annotator_classify.py")))
iac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iac)

# Canonical n=500 PPFG-stag cells, seeds 42/43/44 (Qwen2.5-7B / Math-Shepherd).
DIRS = {
    42: "results/ppfg-math500-20260514-161757-j60966183",
    43: "results/ppfg-math500-20260514-161757-j60966186",
    44: "results/ppfg-math500-20260514-161757-j60966195",
}
BUCKETS = ["high-flat-PRM", "near-completion", "target-succeeded", "plausibly-helped"]


def main():
    events = []
    for seed, d in DIRS.items():
        events += iac._gather_events(Path(d) / "trajectories.json", seed)
    n = len(events)
    counts = Counter()
    helped = 0
    for ev in events:
        tags = iac._classify_a1(ev)
        for t in tags:
            counts[t] += 1
        if tags == frozenset({"plausibly-helped"}):
            helped += 1
    out = {
        "n_events": n,
        "bucket_counts": {b: counts.get(b, 0) for b in BUCKETS},
        "bucket_pct": {b: round(100 * counts.get(b, 0) / n, 1) for b in BUCKETS},
        "well_targeted": helped,
        "well_targeted_pct": round(100 * helped / n, 1),
        "wasted": n - helped,
        "wasted_pct": round(100 * (n - helped) / n, 1),
        "seeds": list(DIRS.keys()),
        "note": "Buckets non-mutually-exclusive; plausibly-helped is the complement of their union.",
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/bucket_classification_n500.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
