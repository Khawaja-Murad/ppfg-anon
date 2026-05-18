#!/usr/bin/env python
"""Compute minimum-detectable-effect (MDE) for the headline metrics in §5.

Two-sample t-test power formula:

    MDE = (t_{alpha/2, df} + t_{beta, df}) * sigma_pooled * sqrt(2 / n_per_arm)

with alpha=0.05 (two-sided), beta=0.20 (80% power), n_per_arm=3 (three seeds),
df=2*(n-1)=4 (equal-variance two-sample test).

Per-arm seed standard deviations:
- PPFG-stag stds are read directly from each run's metrics.json
  (results/ppfg-math500-20260514-{012146-j60934133,012146-j60934134,012346-j60934135}).
  metrics.json uses the runner's task.is_correct path (math_verify-backed),
  which is the canonical correctness function for the paper.
- Indep stds at n=100 are taken from §5.1 Table 1 of paper_outline.md
  (verified at 2026-05-15 against agent_transfer.md §3.6.4). These are
  computed by subsetting the indep-math500-seed{42,43,44} 500-problem
  trajectories to the first 100 problems and running the same task.is_correct
  path the runner uses.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)

# Two-sided t critical at alpha/2 = 0.025, df = 4. Matches scipy.stats.t.ppf(0.975, 4).
T_CRIT_ALPHA_2 = 2.776
# One-sided t critical at beta = 0.20, df = 4. Matches scipy.stats.t.ppf(0.80, 4).
T_CRIT_BETA = 0.941
N_PER_ARM = 3
DF = 2 * (N_PER_ARM - 1)
assert DF == 4, "df must be 4 for two-sample t with n_per_arm=3"
MULTIPLIER = (T_CRIT_ALPHA_2 + T_CRIT_BETA) * math.sqrt(2 / N_PER_ARM)


def compute_mde(sigma_pooled: float) -> float:
    return MULTIPLIER * sigma_pooled


# Published Table 1 indep stds (n=100 first-100 subset of indep-math500-seed{42,43,44}).
INDEP_TABLE1_STDS = {
    "mode_rate": 0.0346,
    "pass@8": 0.0115,
    "pass@1": 0.0038,
}

# PPFG-stag run dirs (Day-5 n=100 cells, j60934133/4/5).
PPFG_STAG_DIRS = {
    42: REPO_ROOT / "results" / "ppfg-math500-20260514-012146-j60934133",
    43: REPO_ROOT / "results" / "ppfg-math500-20260514-012146-j60934134",
    44: REPO_ROOT / "results" / "ppfg-math500-20260514-012346-j60934135",
}


def _ppfg_stag_seed_std(metric_key: str) -> float:
    """Pull per-seed value from metrics.json and compute seed-level std."""
    vals = []
    for s, d in PPFG_STAG_DIRS.items():
        m = json.load((d / "metrics.json").open())
        if metric_key == "mode_rate":
            v = m["diversity"]["answer_mode_rate"]
        elif metric_key.startswith("pass@"):
            k = metric_key.split("@", 1)[1]
            v = m["pass_at_k"][k]
        else:
            raise KeyError(metric_key)
        vals.append(v)
    return statistics.stdev(vals)


def main():
    print(f"Power-analysis multiplier (t_a/2 + t_b) * sqrt(2/n): {MULTIPLIER:.3f}")
    print(f"  (alpha=0.05 two-sided, power=0.80, n_per_arm={N_PER_ARM}, df={DF})\n")

    results = {}
    for label in ("mode_rate", "pass@8", "pass@1"):
        s_indep = INDEP_TABLE1_STDS[label]
        s_ppfg = _ppfg_stag_seed_std(label)
        sigma_pooled = math.sqrt((s_indep ** 2 + s_ppfg ** 2) / 2)
        mde = compute_mde(sigma_pooled)
        results[label] = {
            "indep_seed_std": s_indep,
            "ppfg_stag_seed_std": s_ppfg,
            "sigma_pooled": sigma_pooled,
            "mde": mde,
            "mde_pct_pts": mde * 100,
        }
        print(
            f"  {label:>10}: "
            f"s_indep={s_indep:.4f}  s_ppfg={s_ppfg:.4f}  "
            f"sigma_pooled={sigma_pooled:.4f}  "
            f"MDE={mde:.4f}  (~{mde*100:.1f} pts)"
        )

    out_path = REPO_ROOT / "results" / "mde_summary.json"
    out_path.write_text(json.dumps({
        "alpha_two_sided": 0.05,
        "power": 0.80,
        "n_per_arm": N_PER_ARM,
        "df": DF,
        "t_crit_alpha_2": T_CRIT_ALPHA_2,
        "t_crit_beta": T_CRIT_BETA,
        "multiplier": MULTIPLIER,
        "indep_seed_std_source": "paper_outline.md §5.1 Table 1; agent_transfer.md §3.6.4 (first-100 subset of indep-math500-seed{42,43,44}/trajectories.json via runner task.is_correct)",
        "ppfg_stag_seed_std_source": "metrics.json of {j60934133,j60934134,j60934135}",
        "metrics": results,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
