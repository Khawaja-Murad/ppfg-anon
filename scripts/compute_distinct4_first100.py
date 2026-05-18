#!/usr/bin/env python
"""Compute distinct_4 over the first-100 slice of n=500 trajectories,
matching the diversity.distinct_n implementation in src/hyp_forest/metrics/diversity.py.

Used to populate Table 1's distinct_4 column for indep / smc / BoN-N=16 cells
whose first-100 numbers in §5.1 are sliced from full-500 trajectories already on disk.

Usage:
  PYTHONPATH=src python scripts/compute_distinct4_first100.py <dir1> <dir2> ...
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def distinct_4(records):
    """Re-implementation of src/hyp_forest/metrics/diversity.py::distinct_n with n=4,
    operating on the raw trajectories.json record list (skipping the Population dataclass)."""
    n = 4
    total = 0
    unique = set()
    for rec in records:
        chains = rec.get("population", rec).get("chains", [])
        for c in chains:
            text = " ".join(c.get("steps") or [])
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                total += 1
                unique.add(gram)
    return (len(unique) / total) if total else 0.0


def main():
    if len(sys.argv) < 2:
        print("usage: compute_distinct4_first100.py <results/dir> [...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        traj_path = Path(d) / "trajectories.json"
        if not traj_path.exists():
            print(f"  {d}: MISSING trajectories.json")
            continue
        with traj_path.open() as f:
            recs = json.load(f)
        full_d4 = distinct_4(recs)
        first100_d4 = distinct_4(recs[:100])
        print(f"  {d}: full={full_d4:.4f}  first100={first100_d4:.4f}  n_recs={len(recs)}")


if __name__ == "__main__":
    main()
