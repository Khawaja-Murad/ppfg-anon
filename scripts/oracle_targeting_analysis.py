#!/usr/bin/env python3
"""
Reviewer-#2 E3 (Day-20) — Oracle Targeting upper bound.

Post-hoc analysis on existing trajectories. Adjudicates the reviewer's hypothesis:
``If even an Oracle selector cannot improve Pass@k over the independent baseline,
you prove that verbatim in-context grafting is a completely dead mechanism for
LLM reasoning. If the Oracle does unlock significant gains, you effectively
re-center the paper's value, proving that the grafting mechanism is highly
potent but demands non-heuristic, learned cross-chain value estimators to
execute safely.''

Two answers are computed:

A) **Oracle Pass@k upper bound** (loosest defensible bound). For each problem-seed
   pair, the oracle is allowed to choose between the PPFG-stag population and the
   independent baseline population on a per-problem basis. Oracle correctness on
   a problem = OR(PPFG correct, indep correct). Pass@k via Chen-et-al-2021 unbiased
   estimator over the union per problem-seed, then averaged across problems.
   If oracle Pass@k <= max(PPFG, indep) Pass@k + seed-sigma, the union of methods
   does not exceed either method individually -> the mechanism has nothing
   architecturally absent from indep, even under perfect choice.

B) **Per-injection success conditioning** (tighter bound). For each of the 322
   stagnation-rule injection events across the 3 Qwen seeds, identify whether the
   *receiving* chain went on to produce the correct boxed answer (PROMOTED + match
   to gold). Compare to two baselines:
   (i)  the success rate of non-injected siblings in the same population,
   (ii) the success rate of chains in the matched indep population.
   If injection-success rate is statistically indistinguishable from the
   non-injected siblings AND from indep, the targeting heuristic is no worse than
   random — but also no better. The reviewer's E3 hypothesis (oracle targeting
   unlocks gains) requires the per-injection success rate to exceed indep by an
   effect size larger than the seed-to-seed noise floor.

Inputs (Qwen2.5-7B MATH500 n=500, seeds 42-44):
  PPFG-stag:
    results/ppfg-math500-20260514-161757-j60966183 (seed 42)
    results/ppfg-math500-20260514-161757-j60966186 (seed 43)
    results/ppfg-math500-20260514-161757-j60966195 (seed 44)
  Indep matched:
    results/independent-math500-20260514-161757-j60966078 (seed 42)
    results/independent-math500-20260514-161757-j60966131 (seed 43)
    results/independent-math500-20260514-161757-j60966148 (seed 44)

Outputs:
  results/oracle_targeting_analysis.json    (summary deltas + per-seed metrics)
  results/oracle_targeting_per_problem.csv  (per-problem oracle vs PPFG vs indep)
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path


RESULTS = Path("results")
OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PPFG_STAG_PATHS = {
    42: RESULTS / "ppfg-math500-20260514-161757-j60966183",
    43: RESULTS / "ppfg-math500-20260514-161757-j60966186",
    44: RESULTS / "ppfg-math500-20260514-161757-j60966195",
}
INDEP_PATHS = {
    42: RESULTS / "independent-math500-20260514-161757-j60966078",
    43: RESULTS / "independent-math500-20260514-161757-j60966131",
    44: RESULTS / "independent-math500-20260514-161757-j60966148",
}

K_VALUES = [1, 2, 4, 8]
N_BOOT = 5000
BOOT_SEED = 20260520


def load_traj(path: Path):
    with (path / "trajectories.json").open() as f:
        return json.load(f)


def answer_is_correct(chain, gold) -> bool:
    """String-comparison fallback consistent with surviving_sibling_counterfactual.
    Conservative — may miss math-equivalent surface forms — but applied identically
    across PPFG and indep so the delta is unbiased."""
    fa = chain.get("final_answer")
    if fa is None or gold is None:
        return False
    a = str(fa).strip().rstrip(".").replace(" ", "")
    g = str(gold).strip().rstrip(".").replace(" ", "")
    return a == g


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Chen et al. 2021 unbiased Pass@k estimator.

    Returns the probability that at least one of k uniformly-sampled-without-replacement
    chains is correct, given c correct out of n total chains.
    """
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    num = 1.0
    den = 1.0
    for i in range(k):
        num *= (n - c - i)
        den *= (n - i)
    return 1.0 - num / den


def correct_indicators(pop, gold) -> list[bool]:
    """For each chain in population, return whether it produced the correct answer."""
    return [answer_is_correct(c, gold) for c in pop["chains"]]


def pass_at_k_from_indicators(inds: list[bool], k: int) -> float:
    n = len(inds)
    c = sum(inds)
    return pass_at_k_unbiased(n, c, k)


def oracle_indicators(ppfg_inds: list[bool], indep_inds: list[bool]) -> list[bool]:
    """Oracle population: under the perfect-choice assumption, the oracle gets to pick
    chain i from whichever method's chain i is correct.

    Realization: oracle_inds[i] = ppfg_inds[i] OR indep_inds[i].

    This is the LOOSEST defensible upper bound: any chain that succeeded under either
    method counts. The oracle does not get to mix-and-match across methods on a finer
    grain than the chain level (which would require running unobserved counterfactuals).
    """
    n = min(len(ppfg_inds), len(indep_inds))
    return [ppfg_inds[i] or indep_inds[i] for i in range(n)]


def compute_per_problem_metrics(ppfg_traj, indep_traj):
    """Per-problem metrics dict keyed by problem_id.

    Each entry: dict with ppfg_inds, indep_inds, oracle_inds, gold, n_injections_recv.
    """
    indep_by_pid = {p["problem_id"]: p for p in indep_traj}
    out = {}
    for ppfg_p in ppfg_traj:
        pid = ppfg_p["problem_id"]
        if pid not in indep_by_pid:
            continue
        indep_p = indep_by_pid[pid]
        gold = ppfg_p.get("answer")
        if gold is None:
            gold = indep_p.get("answer")
        ppfg_inds = correct_indicators(ppfg_p["population"], gold)
        indep_inds = correct_indicators(indep_p["population"], gold)
        oracle_inds = oracle_indicators(ppfg_inds, indep_inds)
        n_inj = sum(
            1
            for c in ppfg_p["population"]["chains"]
            if c.get("injected_fragments")
        )
        out[pid] = {
            "gold": gold,
            "ppfg_inds": ppfg_inds,
            "indep_inds": indep_inds,
            "oracle_inds": oracle_inds,
            "n_chains_ppfg": len(ppfg_inds),
            "n_chains_indep": len(indep_inds),
            "n_chains_injected": n_inj,
        }
    return out


def per_injection_success_indicators(ppfg_traj) -> tuple[list[bool], list[bool]]:
    """For each injection event in the PPFG-stag trajectories, return:
    (a) did the *receiving* chain ultimately produce the correct boxed answer?
    (b) for the same problem, did the non-injected siblings produce correct answers?
        (returned as a flat list — one entry per non-injected sibling per problem).
    """
    inj_chain_success: list[bool] = []
    noninj_sibling_success: list[bool] = []
    for prob in ppfg_traj:
        gold = prob.get("answer")
        for c in prob["population"]["chains"]:
            if c.get("injected_fragments"):
                inj_chain_success.append(answer_is_correct(c, gold))
            else:
                noninj_sibling_success.append(answer_is_correct(c, gold))
    return inj_chain_success, noninj_sibling_success


def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (mean, lb, ub)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return sum(values) / n, lo, hi


def main():
    print(f"Loading {len(PPFG_STAG_PATHS)} PPFG-stag and {len(INDEP_PATHS)} indep runs...")
    summary = {
        "K_VALUES": K_VALUES,
        "seeds": sorted(PPFG_STAG_PATHS.keys()),
        "per_seed": {},
        "aggregate": {},
    }

    # Per-seed Pass@k for PPFG, indep, oracle
    per_seed_passk_ppfg = {k: [] for k in K_VALUES}
    per_seed_passk_indep = {k: [] for k in K_VALUES}
    per_seed_passk_oracle = {k: [] for k in K_VALUES}

    # Per-injection / non-injection success aggregation
    all_inj_success = []
    all_noninj_success = []

    # CSV per-problem rows
    csv_rows = []

    for seed in sorted(PPFG_STAG_PATHS.keys()):
        ppfg_traj = load_traj(PPFG_STAG_PATHS[seed])
        indep_traj = load_traj(INDEP_PATHS[seed])
        per_prob = compute_per_problem_metrics(ppfg_traj, indep_traj)
        print(f"  seed {seed}: {len(per_prob)} matched problems")

        passk_ppfg = {k: 0.0 for k in K_VALUES}
        passk_indep = {k: 0.0 for k in K_VALUES}
        passk_oracle = {k: 0.0 for k in K_VALUES}
        n_prob = len(per_prob)

        for pid, m in per_prob.items():
            for k in K_VALUES:
                passk_ppfg[k] += pass_at_k_from_indicators(m["ppfg_inds"], k)
                passk_indep[k] += pass_at_k_from_indicators(m["indep_inds"], k)
                passk_oracle[k] += pass_at_k_from_indicators(m["oracle_inds"], k)
            csv_rows.append({
                "seed": seed,
                "problem_id": pid,
                "n_chains_injected": m["n_chains_injected"],
                "ppfg_correct_chains": sum(m["ppfg_inds"]),
                "indep_correct_chains": sum(m["indep_inds"]),
                "oracle_correct_chains": sum(m["oracle_inds"]),
            })
        for k in K_VALUES:
            passk_ppfg[k] /= max(n_prob, 1)
            passk_indep[k] /= max(n_prob, 1)
            passk_oracle[k] /= max(n_prob, 1)
            per_seed_passk_ppfg[k].append(passk_ppfg[k])
            per_seed_passk_indep[k].append(passk_indep[k])
            per_seed_passk_oracle[k].append(passk_oracle[k])

        # Per-injection
        inj_succ, noninj_succ = per_injection_success_indicators(ppfg_traj)
        all_inj_success.extend(inj_succ)
        all_noninj_success.extend(noninj_succ)

        summary["per_seed"][seed] = {
            "n_problems": n_prob,
            "pass_at_k": {
                "ppfg":   {str(k): passk_ppfg[k]   for k in K_VALUES},
                "indep":  {str(k): passk_indep[k]  for k in K_VALUES},
                "oracle": {str(k): passk_oracle[k] for k in K_VALUES},
            },
            "n_injection_events": len(inj_succ),
            "injection_success_rate": (sum(inj_succ) / len(inj_succ)) if inj_succ else float("nan"),
            "n_noninjected_sibling_chains": len(noninj_succ),
            "noninjected_success_rate": (sum(noninj_succ) / len(noninj_succ)) if noninj_succ else float("nan"),
        }

    # Aggregate (mean +/- std across 3 seeds)
    def mean_std(xs):
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, 0.0
        s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
        return m, s

    for k in K_VALUES:
        m_ppfg, s_ppfg     = mean_std(per_seed_passk_ppfg[k])
        m_indep, s_indep   = mean_std(per_seed_passk_indep[k])
        m_oracle, s_oracle = mean_std(per_seed_passk_oracle[k])
        summary["aggregate"][f"pass_at_{k}"] = {
            "ppfg":   {"mean": m_ppfg,   "std": s_ppfg},
            "indep":  {"mean": m_indep,  "std": s_indep},
            "oracle": {"mean": m_oracle, "std": s_oracle},
            "oracle_vs_indep_delta":         m_oracle - m_indep,
            "oracle_vs_indep_delta_pp_pts":  (m_oracle - m_indep) * 100,
            "ppfg_vs_indep_delta":           m_ppfg - m_indep,
            "ppfg_vs_indep_delta_pp_pts":    (m_ppfg - m_indep) * 100,
            "oracle_exceeds_indep_seed_sigma": (m_oracle - m_indep) > s_indep,
        }

    # Per-injection aggregate
    n_inj = len(all_inj_success)
    n_noninj = len(all_noninj_success)
    inj_rate = (sum(all_inj_success) / n_inj) if n_inj else float("nan")
    noninj_rate = (sum(all_noninj_success) / n_noninj) if n_noninj else float("nan")

    # Bootstrap CI for the inj - noninj delta
    rng = random.Random(BOOT_SEED + 1)
    inj_arr = [1.0 if b else 0.0 for b in all_inj_success]
    noninj_arr = [1.0 if b else 0.0 for b in all_noninj_success]
    delta_boots = []
    for _ in range(N_BOOT):
        a = sum(inj_arr[rng.randrange(n_inj)] for _ in range(n_inj)) / max(n_inj, 1)
        b = sum(noninj_arr[rng.randrange(n_noninj)] for _ in range(n_noninj)) / max(n_noninj, 1)
        delta_boots.append(a - b)
    delta_boots.sort()
    delta_mean = inj_rate - noninj_rate
    delta_lb = delta_boots[int(0.025 * len(delta_boots))]
    delta_ub = delta_boots[int(0.975 * len(delta_boots))]

    summary["per_injection"] = {
        "n_injection_events": n_inj,
        "n_noninjected_siblings": n_noninj,
        "injection_success_rate": inj_rate,
        "noninjected_success_rate": noninj_rate,
        "delta_inj_minus_noninj": {
            "mean": delta_mean,
            "ci95_lb": delta_lb,
            "ci95_ub": delta_ub,
            "ci95_excludes_zero": (delta_lb > 0) or (delta_ub < 0),
        },
    }

    # Save outputs
    out_json = OUT_DIR / "oracle_targeting_analysis.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    out_csv = OUT_DIR / "oracle_targeting_per_problem.csv"
    with out_csv.open("w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Wrote {out_csv}")

    # Summary print
    print("\n=== Oracle Pass@k upper bound vs PPFG-stag vs indep ===")
    print(f"{'k':>4} {'PPFG':>14} {'indep':>14} {'Oracle':>14} {'Oracle-indep':>14}")
    for k in K_VALUES:
        a = summary["aggregate"][f"pass_at_{k}"]
        print(
            f"{k:>4} "
            f"{a['ppfg']['mean']:.4f} +/- {a['ppfg']['std']:.4f}  "
            f"{a['indep']['mean']:.4f} +/- {a['indep']['std']:.4f}  "
            f"{a['oracle']['mean']:.4f} +/- {a['oracle']['std']:.4f}  "
            f"{a['oracle_vs_indep_delta_pp_pts']:+.2f} pp"
        )

    print("\n=== Per-injection success rate vs non-injected siblings ===")
    print(f"  Injection success rate: {inj_rate:.4f} (n={n_inj})")
    print(f"  Non-injected sibling success rate: {noninj_rate:.4f} (n={n_noninj})")
    print(f"  Delta (inj - noninj): {delta_mean:+.4f} CI95 [{delta_lb:+.4f}, {delta_ub:+.4f}]")
    print(f"  CI excludes zero: {summary['per_injection']['delta_inj_minus_noninj']['ci95_excludes_zero']}")


if __name__ == "__main__":
    main()
