"""Reproduce the prune-immunity content control (paper Appendix: Table
`tab:prune-immunity`).

Three arms on the FIRST-100 MATH500 slice, Qwen2.5-7B-Instruct +
Math-Shepherd, N=8, 3 seeds:

  (a) independent, no immunity          <- sliced from the n=500 indep runs
  (b) immunity + null graft             <- configs/..._prune_immune_nullctrl.yaml
  (c) immunity + real graft             <- configs/..._prune_immune.yaml

(c)-(a) measures what suppressing the PRM mis-pruning channel buys.
(c)-(b) isolates graft CONTENT with that channel held closed -- the
load-bearing contrast, since (b) and (c) differ in exactly one config key.

TWO TRAPS THIS SCRIPT EXISTS TO AVOID
-------------------------------------
1. Arm (a) must be the FIRST-100 problems, matching the head-slice that
   `problem_subset: null` gives arms (b)/(c) (see runner.py). Comparing
   against a run over problems 400-499 is not a matched baseline; doing so
   is what produced the incorrect "+6.96 pp" in the ARR response period.
2. Every arm must be scored by ONE comparator in ONE environment. A
   metrics.json written when `math_verify` was importable is not
   comparable to one recomputed when it was not. This script re-scores
   all arms itself and warns if the comparator is degraded.

Usage:
  PYTHONPATH=src python scripts/prune_immunity_analysis.py \
      [--results_dir results] [--out results/prune_immunity_analysis.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

from hyp_forest.comparator_guard import warn_if_degraded, math_verify_available

# Reuse the exact scoring path used elsewhere in the repo.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slice_first100 import first_n  # noqa: E402

N_PROBLEMS = 100
KS = (1, 2, 4, 8)

# Arms are identified by experiment name in config.json, so run-dir
# timestamps/job-ids do not have to be hard-coded.
ARMS = {
    "a_indep_no_immunity":  "qwen_n500_indep_math500",           # sliced to first 100
    "b_immunity_null":      "qwen_n100_ppfg_stag_prune_immune_nullctrl",
    "c_immunity_graft":     "qwen_n100_ppfg_stag_prune_immune",
    "ref_ppfg_stag_canon":  "ppfg_stag_math500",
}
SEEDS = (42, 43, 44)


def find_runs(results_dir: Path, experiment_name: str) -> list[Path]:
    """All run dirs whose config.json names this experiment, seeds 42-44."""
    out = {}
    for cfg in sorted(results_dir.glob("*/config.json")):
        try:
            c = json.loads(cfg.read_text())
        except Exception:
            continue
        exp = c.get("experiment", {})
        if exp.get("name") != experiment_name:
            continue
        seed = exp.get("seed")
        if seed in SEEDS and seed not in out:
            out[seed] = cfg.parent
    return [out[s] for s in SEEDS if s in out]


def pruned_fraction(run_dir: Path, n_problems: int = N_PROBLEMS) -> float:
    """Manipulation check: fraction of chains ending in `pruned` status."""
    traj = json.loads((run_dir / "trajectories.json").read_text())[:n_problems]
    counts: Counter = Counter()
    for problem in traj:
        for chain in problem["population"]["chains"]:
            counts[chain["status"]] += 1
    total = sum(counts.values())
    return counts["pruned"] / total if total else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/prune_immunity_analysis.json")
    args = ap.parse_args()

    degraded = warn_if_degraded()
    results_dir = Path(args.results_dir)

    summary: dict = {
        "n_problems": N_PROBLEMS,
        "seeds": list(SEEDS),
        "math_verify_available": math_verify_available(),
        "_note": (
            "All arms re-scored in one environment on the first-%d problems. "
            "Absolute levels shift with comparator availability; the "
            "within-analysis deltas are the interpretable quantities."
            % N_PROBLEMS
        ),
        "arms": {},
    }

    for arm, exp_name in ARMS.items():
        runs = find_runs(results_dir, exp_name)
        if len(runs) != len(SEEDS):
            print(f"  !! {arm}: found {len(runs)}/{len(SEEDS)} runs for "
                  f"'{exp_name}' -- skipping")
            continue
        per_seed = [first_n(r.name, N_PROBLEMS)[0] for r in runs]
        entry = {
            "experiment_name": exp_name,
            "run_dirs": [r.name for r in runs],
            "pass_at_k": {
                str(k): {
                    "mean": st.mean(p[k] for p in per_seed),
                    "sd": st.stdev([p[k] for p in per_seed]),
                }
                for k in KS
            },
        }
        pf = [pruned_fraction(r) for r in runs]
        entry["pruned_fraction"] = {"mean": st.mean(pf), "sd": st.stdev(pf)}
        summary["arms"][arm] = entry
        print(f"  {arm:24s} Pass@1 {entry['pass_at_k']['1']['mean']:.4f} "
              f"+/- {entry['pass_at_k']['1']['sd']:.4f}   "
              f"pruned {entry['pruned_fraction']['mean']:.4f}")

    a, b, c = (summary["arms"].get(k) for k in
               ("a_indep_no_immunity", "b_immunity_null", "c_immunity_graft"))
    if a and b and c:
        summary["deltas_pp"] = {
            "immunity_lift_c_minus_a": {
                str(k): 100 * (c["pass_at_k"][str(k)]["mean"]
                               - a["pass_at_k"][str(k)]["mean"]) for k in KS},
            "graft_effect_c_minus_b": {
                str(k): 100 * (c["pass_at_k"][str(k)]["mean"]
                               - b["pass_at_k"][str(k)]["mean"]) for k in KS},
        }
        print("\n  immunity lift (c-a), pp:",
              {k: round(v, 2) for k, v in
               summary["deltas_pp"]["immunity_lift_c_minus_a"].items()})
        print("  graft effect  (c-b), pp:",
              {k: round(v, 2) for k, v in
               summary["deltas_pp"]["graft_effect_c_minus_b"].items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(f"\nWrote {args.out}")
    if degraded:
        print("NOTE: comparator degraded -- deltas are valid, absolute "
              "levels are depressed relative to the paper.")


if __name__ == "__main__":
    main()
