"""Compute continuous raw modal share per cell from saved trajectories.

The paper's `answer_mode_rate` (per `src/hyp_forest/metrics/diversity.py`) is a
binary-thresholded metric: fraction of problems where the modal non-null answer's
share exceeds 0.5. DeepSeek R2 reviewer asked for a continuous supplement.

Raw modal share (per cell):
  - For each problem, compute max(count(answer)) / count(non-null answers) over chains
  - Average across problems
  - Mean ± std across seeds

Also reports Spearman ρ between raw modal share and `answer_mode_rate` across
all cells found, to support the §4.5 footnote claim that the binary version
preserves ranking.

Usage:
  PYTHONPATH=src python scripts/compute_raw_modal_share.py
"""

from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import statistics


def normalize_answer(s):
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"^\$+", "", s)
    s = re.sub(r"\$+$", "", s)
    s = s.strip()
    s = s.replace(r"\dfrac", r"\frac").replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s if s else None


def per_problem_modal_share(chains):
    answers = [normalize_answer(c.get("final_answer")) for c in chains]
    nn = [a for a in answers if a]
    if not nn:
        return 0.0
    return Counter(nn).most_common(1)[0][1] / len(nn)


def cell_raw_modal_share(run_dir):
    traj_path = run_dir / "trajectories.json"
    if not traj_path.exists():
        return None
    traj = json.load(open(traj_path))
    shares = []
    for p in traj:
        chains = p.get("population", p).get("chains", [])
        shares.append(per_problem_modal_share(chains))
    if not shares:
        return None
    return sum(shares) / len(shares)


def spearman_rho(xs, ys):
    """Pure-stdlib Spearman ρ (avoids scipy dependency)."""
    def rank(vs):
        # Average rank for ties
        idx = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and vs[idx[j + 1]] == vs[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks
    rx = rank(xs)
    ry = rank(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = (sum((r - mx) ** 2 for r in rx)) ** 0.5
    dy = (sum((r - my) ** 2 for r in ry)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main():
    results = Path("results")
    dirs = sorted([d for d in results.glob("*-math500-*") if d.is_dir() and "_archive" not in str(d)])
    # Group by (method, n_problems, base_model, injection_rule) — pull from config.json
    cells = defaultdict(list)
    rows = []
    for d in dirs:
        cfg_path = d / "config.json"
        m_path = d / "metrics.json"
        if not cfg_path.exists() or not m_path.exists():
            continue
        cfg = json.load(open(cfg_path))
        m = json.load(open(m_path))
        method = m.get("method")
        n_problems = m.get("n_problems")
        base_model = cfg.get("model", {}).get("base_model", "?").split("/")[-1]
        injection_rule = cfg.get("ppfg", {}).get("injection_rule") if method == "ppfg" else None
        seed = cfg.get("experiment", {}).get("seed")
        raw = cell_raw_modal_share(d)
        amr = m.get("diversity", {}).get("answer_mode_rate")
        if raw is None or amr is None:
            continue
        key = (base_model, method, injection_rule, n_problems)
        cells[key].append({"seed": seed, "raw_modal_share": raw, "answer_mode_rate": amr})
        rows.append({"dir": d.name, "base_model": base_model, "method": method, "rule": injection_rule, "n": n_problems, "seed": seed, "raw_modal_share": raw, "answer_mode_rate": amr})

    summary = {}
    raw_means = []
    amr_means = []
    for key, seeds in sorted(cells.items()):
        if len(seeds) < 2:
            continue
        raw_vals = [s["raw_modal_share"] for s in seeds]
        amr_vals = [s["answer_mode_rate"] for s in seeds]
        raw_mean = sum(raw_vals) / len(raw_vals)
        raw_std = statistics.stdev(raw_vals) if len(raw_vals) > 1 else 0.0
        amr_mean = sum(amr_vals) / len(amr_vals)
        amr_std = statistics.stdev(amr_vals) if len(amr_vals) > 1 else 0.0
        label = f"{key[0]} | {key[1]}{'+'+key[2] if key[2] else ''} | n={key[3]}"
        summary[label] = {
            "raw_modal_share_mean": raw_mean,
            "raw_modal_share_std": raw_std,
            "answer_mode_rate_mean": amr_mean,
            "answer_mode_rate_std": amr_std,
            "n_seeds": len(seeds),
        }
        raw_means.append(raw_mean)
        amr_means.append(amr_mean)

    rho = spearman_rho(raw_means, amr_means) if len(raw_means) >= 3 else float("nan")

    out = {
        "spearman_rho_raw_vs_amr": rho,
        "n_cells": len(summary),
        "cells": summary,
        "per_run": rows,
    }
    output_path = Path("results/raw_modal_share_summary.json")
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {output_path}")
    print(f"Spearman ρ (raw_modal_share vs answer_mode_rate) = {rho:.4f} over {len(raw_means)} cells")
    print()
    print(f"{'Cell':<70s} {'raw_ms':>10s} {'amr':>10s}")
    for label, s in sorted(summary.items()):
        print(f"{label:<70s} {s['raw_modal_share_mean']:>6.4f}±{s['raw_modal_share_std']:>0.4f} {s['answer_mode_rate_mean']:>6.4f}±{s['answer_mode_rate_std']:>0.4f}")


if __name__ == "__main__":
    main()
