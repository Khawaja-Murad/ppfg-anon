#!/usr/bin/env python3
"""
V15.B — Surviving-sibling counterfactual.

Adjudicates the §5.4 framing between two readings of the
$2.75\\times$ post-injection pruning hazard:

  (alpha) compensation: when PPFG kills the injected chain, the
          surviving siblings take up the slack (higher promotion
          rate / more tokens / matched correctness vs indep)
  (beta)  selection-property: injected chain was on a doomed
          trajectory anyway; PPFG just pulls its termination
          forward; the rest of the population behaves as in indep
          (no detectable change in surviving siblings)
  (gamma) something else: population correctness deviates from
          indep — major finding, surface

Method:
  Define treated unit = (problem_id, seed) where at least one
  injected chain in the PPFG-stag population reached PRUNED
  status within k+3 steps of receiving the injection. For each
  treated unit, compute outcomes on the NON-INJECTED siblings
  (the chains in that population that received no injection).

  Match to a control unit = same (problem_id, seed) in the
  independent baseline (all 8 chains are non-injected by
  construction).

  Three outcomes per unit:
    (a) pop_correctness:    did any chain produce the correct
                            boxed answer? (1 or 0)
    (b) per_chain_promotion: fraction of (non-injected) chains
                             reaching PROMOTED status
    (c) per_chain_tokens:    mean(n_tokens_generated) over the
                             same chains

  Bootstrap 95% CIs on deltas (5000 resamples over matched
  problem-seed pairs).

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
  results/surviving_sibling_counterfactual.json  (summary)
  results/surviving_sibling_per_problem.csv      (per-unit detail)
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


RESULTS = Path("./results")
OUT_DIR = Path("./results")
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

K_PLUS_DELTA = 3   # "within 3 steps of receiving the injection"
N_BOOT = 5000
BOOT_SEED = 20260518


def load_traj(path: Path):
    with (path / "trajectories.json").open() as f:
        return json.load(f)


def chain_terminal_step(chain) -> int:
    return len(chain["steps"]) - 1


def answer_is_correct(chain, gold) -> bool:
    """String-comparison fallback consistent with the existing
    post_injection_prune_analysis.py approach. Conservative — may
    miss math-equivalent surface forms — but applied identically
    to A and B so the delta is unbiased."""
    fa = chain.get("final_answer")
    if fa is None or gold is None:
        return False
    a = str(fa).strip().rstrip(".").replace(" ", "")
    g = str(gold).strip().rstrip(".").replace(" ", "")
    return a == g


def has_injection_pruned_within_k3(pop) -> bool:
    """True if any chain in this population received an injection
    AND reached PRUNED status within K_PLUS_DELTA steps of it."""
    for chain in pop["chains"]:
        fragments = chain.get("injected_fragments") or []
        if not fragments:
            continue
        if chain["status"] != "pruned":
            continue
        terminal = chain_terminal_step(chain)
        # the chain might have multiple injection events; take the
        # latest as the "trigger" because that's the one closest to
        # its terminal step.
        k_latest = max(int(f["injected_at_step"]) for f in fragments)
        if terminal - k_latest <= K_PLUS_DELTA:
            return True
    return False


def non_injected_chains(pop):
    return [c for c in pop["chains"] if not (c.get("injected_fragments") or [])]


def compute_outcomes_for_chains(chains, gold):
    """Three per-unit outcomes computed over a list of chains:
      - pop_correctness: 1 if any of these chains is promoted-correct
      - per_chain_promotion: mean(status == 'promoted')
      - per_chain_tokens:    mean(n_tokens_generated)
    """
    n = len(chains)
    if n == 0:
        return None
    any_correct = any(
        c["status"] == "promoted" and answer_is_correct(c, gold)
        for c in chains
    )
    promoted = sum(1 for c in chains if c["status"] == "promoted") / n
    tokens = sum(c.get("n_tokens_generated", 0) for c in chains) / n
    return {
        "pop_correctness": int(any_correct),
        "per_chain_promotion": promoted,
        "per_chain_tokens": tokens,
    }


def pop_correctness_full(pop, gold) -> int:
    """For control: same definition as the (A) pop_correctness but
    over all 8 chains (none are injected in indep)."""
    return int(any(
        c["status"] == "promoted" and answer_is_correct(c, gold)
        for c in pop["chains"]
    ))


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05, seed=BOOT_SEED):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _i in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def main():
    # Index indep by (seed, problem_id) for fast match
    indep_index = {}
    for seed, path in INDEP_PATHS.items():
        for prob in load_traj(path):
            indep_index[(seed, prob["problem_id"])] = prob

    paired = []   # list of dicts (one per matched problem-seed)

    for seed, path in PPFG_STAG_PATHS.items():
        probs_ppfg = load_traj(path)
        for prob in probs_ppfg:
            pop = prob["population"]
            if not has_injection_pruned_within_k3(pop):
                continue
            gold = prob.get("answer")

            non_inj = non_injected_chains(pop)
            outcomes_A = compute_outcomes_for_chains(non_inj, gold)
            if outcomes_A is None:
                continue

            ctrl = indep_index.get((seed, prob["problem_id"]))
            if ctrl is None:
                continue
            outcomes_B = compute_outcomes_for_chains(ctrl["population"]["chains"], gold)

            paired.append({
                "seed": seed,
                "problem_id": prob["problem_id"],
                "n_non_injected_A": len(non_inj),
                "n_chains_B": len(ctrl["population"]["chains"]),
                "A_pop_correctness": outcomes_A["pop_correctness"],
                "B_pop_correctness": outcomes_B["pop_correctness"],
                "A_per_chain_promotion": outcomes_A["per_chain_promotion"],
                "B_per_chain_promotion": outcomes_B["per_chain_promotion"],
                "A_per_chain_tokens": outcomes_A["per_chain_tokens"],
                "B_per_chain_tokens": outcomes_B["per_chain_tokens"],
            })

    if not paired:
        raise RuntimeError("Zero matched problem-seed pairs — check input paths.")

    # Deltas
    dcorr   = [p["A_pop_correctness"]      - p["B_pop_correctness"]      for p in paired]
    dprom   = [p["A_per_chain_promotion"]  - p["B_per_chain_promotion"]  for p in paired]
    dtokens = [p["A_per_chain_tokens"]     - p["B_per_chain_tokens"]     for p in paired]

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0

    mean_dcorr   = mean(dcorr)
    mean_dprom   = mean(dprom)
    mean_dtokens = mean(dtokens)

    ci_dcorr   = bootstrap_ci(dcorr, seed=BOOT_SEED + 1)
    ci_dprom   = bootstrap_ci(dprom, seed=BOOT_SEED + 2)
    ci_dtokens = bootstrap_ci(dtokens, seed=BOOT_SEED + 3)

    # Per-seed-σ for context (helps interpret "within seed-σ of 0")
    per_seed_dcorr = defaultdict(list)
    per_seed_dprom = defaultdict(list)
    per_seed_dtokens = defaultdict(list)
    for p in paired:
        per_seed_dcorr[p["seed"]].append(p["A_pop_correctness"]      - p["B_pop_correctness"])
        per_seed_dprom[p["seed"]].append(p["A_per_chain_promotion"]  - p["B_per_chain_promotion"])
        per_seed_dtokens[p["seed"]].append(p["A_per_chain_tokens"]   - p["B_per_chain_tokens"])

    def per_seed_means(d):
        return {s: mean(v) for s, v in sorted(d.items())}

    seed_means_corr = per_seed_means(per_seed_dcorr)
    seed_means_prom = per_seed_means(per_seed_dprom)
    seed_means_tok  = per_seed_means(per_seed_dtokens)

    def seed_sigma(d):
        ms = list(per_seed_means(d).values())
        if len(ms) < 2:
            return 0.0
        return statistics.stdev(ms)

    sigma_corr = seed_sigma(per_seed_dcorr)
    sigma_prom = seed_sigma(per_seed_dprom)
    sigma_tok  = seed_sigma(per_seed_dtokens)

    # ------- compute the matched 1.8x value -------------------------
    # For the Phase A.9 caveat: a "matched-PRM-trajectory" baseline
    # restricted to non-injected chains in the SAME population whose
    # latest-3-step PRM range is below tau_flat (the stagnation gate).
    # Approximates the prune rate of "comparable" chains. The
    # corresponding ratio between matched-trajectory baseline and
    # naive (all non-injected) baseline gives the [matched, naive]
    # interval the Phase A.9 caveat references.
    matched_stat = compute_matched_prm_baseline_ratio()

    # Apply the interpretation rubric (alpha/beta/gamma).
    # The rule from the plan:
    #   alpha: |dcorr|/sigma small AND (dprom > 0, CI excl 0) OR (dtokens > 0, CI excl 0)
    #   beta:  |dcorr|/sigma small AND |dprom|/sigma small AND |dtokens|/sigma small
    #   gamma: |dcorr|/sigma large (CI excludes 0)

    def excludes_zero(ci):
        return ci[0] > 0 or ci[1] < 0

    case = None
    if excludes_zero(ci_dcorr):
        case = "gamma"
    else:
        positive_compensation = (
            (ci_dprom[0] > 0) or (ci_dtokens[0] > 0)
        )
        # selection-property: nothing detectable on either axis
        if positive_compensation:
            case = "alpha"
        elif (not excludes_zero(ci_dprom)) and (not excludes_zero(ci_dtokens)):
            case = "beta"
        else:
            # something detectable but in the wrong direction
            case = "beta_with_caveat"

    summary = {
        "n_matched_pairs": len(paired),
        "n_per_seed": {s: sum(1 for p in paired if p["seed"] == s)
                       for s in sorted(set(p["seed"] for p in paired))},
        "delta_means": {
            "pop_correctness":   mean_dcorr,
            "per_chain_promotion": mean_dprom,
            "per_chain_tokens":  mean_dtokens,
        },
        "delta_bootstrap_ci95": {
            "pop_correctness":   list(ci_dcorr),
            "per_chain_promotion": list(ci_dprom),
            "per_chain_tokens":  list(ci_dtokens),
        },
        "per_seed_means": {
            "pop_correctness":   seed_means_corr,
            "per_chain_promotion": seed_means_prom,
            "per_chain_tokens":  seed_means_tok,
        },
        "seed_sigma": {
            "pop_correctness":   sigma_corr,
            "per_chain_promotion": sigma_prom,
            "per_chain_tokens":  sigma_tok,
        },
        "interpretation_case": case,
        "matched_prm_baseline": matched_stat,
        "config": {
            "k_plus_delta": K_PLUS_DELTA,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "ppfg_stag_paths": {s: str(p) for s, p in PPFG_STAG_PATHS.items()},
            "indep_paths":     {s: str(p) for s, p in INDEP_PATHS.items()},
        },
    }

    out_json = OUT_DIR / "surviving_sibling_counterfactual.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    out_csv = OUT_DIR / "surviving_sibling_per_problem.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        for row in paired:
            w.writerow(row)

    # Console summary
    print(f"=== Surviving-sibling counterfactual ===")
    print(f"matched pairs: {len(paired)}")
    print(f"per-seed pair counts: {summary['n_per_seed']}")
    print(f"")
    print(f"Δ pop_correctness     mean = {mean_dcorr:+.4f}  CI95 = [{ci_dcorr[0]:+.4f}, {ci_dcorr[1]:+.4f}]")
    print(f"Δ per_chain_promotion mean = {mean_dprom:+.4f}  CI95 = [{ci_dprom[0]:+.4f}, {ci_dprom[1]:+.4f}]")
    print(f"Δ per_chain_tokens    mean = {mean_dtokens:+.2f}     CI95 = [{ci_dtokens[0]:+.2f}, {ci_dtokens[1]:+.2f}]")
    print(f"")
    print(f"per-seed σ:  corr={sigma_corr:.4f}  prom={sigma_prom:.4f}  tok={sigma_tok:.2f}")
    print(f"interpretation: {case}")
    print(f"")
    print(f"matched-PRM baseline: {matched_stat}")
    print(f"")
    print(f"wrote: {out_json}")
    print(f"wrote: {out_csv}")


def compute_matched_prm_baseline_ratio():
    """
    For the Phase A.9 caveat about the 2.75× ratio.

    The naive matched-step baseline computes the empirical prune rate
    of non-injected chains at the same step indices as the injection
    events; this averages over all non-injected siblings.

    A more honest counterfactual restricts the baseline to non-injected
    chains whose pre-injection PRM trajectory resembles a stagnation-
    rule trigger candidate (latest-3-step PRM range below tau_flat).
    That subset has higher baseline prune rate, so the ratio
    (post-injection / matched-PRM-baseline) shrinks.

    We compute both for the 322-event PPFG-stag corpus.
    """
    TAU_FLAT = 0.10  # ppfg.flatness_threshold default
    K_DELTA = K_PLUS_DELTA

    # Collect injection events from all 3 PPFG-stag seeds
    inject_events = []  # (seed, problem_id, k_inject, chain pruned-within-3?)
    matched_pool = []   # candidate non-injected sibling steps (their post-k3 status)

    for seed, path in PPFG_STAG_PATHS.items():
        for prob in load_traj(path):
            pop = prob["population"]
            inj_chains = []
            non_inj_chains = []
            for c in pop["chains"]:
                if c.get("injected_fragments"):
                    inj_chains.append(c)
                else:
                    non_inj_chains.append(c)

            for ic in inj_chains:
                # latest injection step on this chain
                fragments = ic.get("injected_fragments") or []
                if not fragments:
                    continue
                k_inj = max(int(f["injected_at_step"]) for f in fragments)
                terminal = chain_terminal_step(ic)
                pruned_within = (ic["status"] == "pruned" and terminal - k_inj <= K_DELTA)
                inject_events.append({
                    "seed": seed, "problem_id": prob["problem_id"],
                    "k_inj": k_inj, "pruned_within": int(pruned_within),
                })

                # Find matched-trajectory non-injected siblings
                # whose PRM range over [k_inj-2, k_inj] (latest-3 window
                # ending at k_inj-1, i.e. the steps the stagnation rule
                # would have seen) is below TAU_FLAT, AND who have
                # produced at least k_inj steps (to be a valid match).
                for sc in non_inj_chains:
                    if len(sc["prm_scores"]) < k_inj + 1:
                        continue
                    window = sc["prm_scores"][max(0, k_inj - 2): k_inj + 1]
                    if not window:
                        continue
                    rng = max(window) - min(window)
                    if rng > TAU_FLAT:
                        continue
                    # this sibling matches the trajectory profile.
                    # Did it become pruned within K_DELTA more steps?
                    sib_terminal = chain_terminal_step(sc)
                    if sib_terminal < k_inj:
                        continue
                    sib_pruned_window = (
                        sc["status"] == "pruned"
                        and (sib_terminal - k_inj) <= K_DELTA
                    )
                    matched_pool.append(int(sib_pruned_window))

    if not inject_events or not matched_pool:
        return {"note": "insufficient data for matched-PRM baseline"}

    p_post = sum(e["pruned_within"] for e in inject_events) / len(inject_events)
    p_matched = sum(matched_pool) / len(matched_pool)

    # Naive baseline (recompute from existing analysis: 9.4%)
    p_naive = 0.094  # documented in §5.4 main body; we don't recompute

    ratio_matched = (p_post / p_matched) if p_matched > 0 else None
    ratio_naive   = (p_post / p_naive)   if p_naive   > 0 else None

    return {
        "n_inject_events": len(inject_events),
        "n_matched_pool":  len(matched_pool),
        "p_post":          p_post,
        "p_matched":       p_matched,
        "p_naive_documented": p_naive,
        "ratio_post_over_matched": ratio_matched,
        "ratio_post_over_naive":   ratio_naive,
        "interval_low_high": [
            ratio_matched if ratio_matched else 1.0,
            ratio_naive   if ratio_naive   else 1.0,
        ],
    }


if __name__ == "__main__":
    main()
