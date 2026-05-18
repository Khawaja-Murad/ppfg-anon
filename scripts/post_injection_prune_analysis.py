#!/usr/bin/env python3
"""
V14.A.1 — chain-dynamics post-injection analysis.

For every PPFG-stag injection event on Qwen MATH500 (3 seeds, 322 events),
look at the target chain (the chain that RECEIVED the injection) at +1/+2/+3
steps after the injection and bucket the outcome. Compare to a matched-step
baseline computed from non-injected chains in the same population.

Buckets:
  (a) pruned within 3 steps         — terminal status PRUNED, terminated at or before k+3
  (b) promoted-correct within 3     — terminal PROMOTED, gold-matched, terminated at or before k+3
  (c) promoted-incorrect within 3   — terminal PROMOTED, not gold-matched, terminated at or before k+3
  (d) still active past k+3         — terminated AFTER k+3, or status is HIT_MAX past k+3

Baseline: at every injection event step k, examine the sibling chains in the
same population that did NOT receive any injection, and compute the
per-step prune probability from k+1 to k+3.

Outputs results/post_injection_prune_analysis.json.
"""
import json
import math
import random
from collections import Counter
from pathlib import Path


RESULTS = Path("results")
PPFG_STAG_PATHS = [
    RESULTS / "ppfg-math500-20260514-161757-j60966183",
    RESULTS / "ppfg-math500-20260514-161757-j60966186",
    RESULTS / "ppfg-math500-20260514-161757-j60966195",
]
OUT_PATH = RESULTS / "post_injection_prune_analysis.json"


def load_trajectories(path: Path):
    with (path / "trajectories.json").open() as f:
        return json.load(f)


def chain_terminal_step(chain) -> int:
    """The step index at which the chain reached its terminal status.
    A pruned chain has its last step at len(steps)-1 with PRM < tkill.
    A promoted chain has its last step at len(steps)-1 with a detected answer.
    A hit_max chain reached T_max steps."""
    return len(chain["steps"]) - 1


def answer_is_correct(chain, gold) -> bool:
    """Cheap string comparison fallback. Matches detect_final_answer +
    math_verify behavior at the level we need (binary correct)."""
    fa = chain.get("final_answer")
    if fa is None or gold is None:
        return False
    # Normalize whitespace and surface artifacts
    a = str(fa).strip().rstrip(".").replace(" ", "")
    g = str(gold).strip().rstrip(".").replace(" ", "")
    return a == g


def categorize_event(chain, k_inject: int, gold) -> str:
    """Bucket the chain outcome relative to the injection step k_inject."""
    terminal_step = chain_terminal_step(chain)
    status = chain["status"]
    steps_after = terminal_step - k_inject
    if steps_after > 3 or status == "active":
        return "d_still_active_past_k3"
    # terminated at or before k_inject + 3
    if status == "pruned":
        return "a_pruned_within_3"
    if status == "promoted":
        if answer_is_correct(chain, gold):
            return "b_promoted_correct_within_3"
        else:
            return "c_promoted_incorrect_within_3"
    if status == "hit_max":
        return "d_still_active_past_k3"
    return "d_still_active_past_k3"


def find_injected_chain_and_step(pop, fragment_event):
    """The injection_at_step field is on the fragment record stored on the
    TARGET chain's injected_fragments list. So the chain containing this
    fragment IS the target chain; k_inject = fragment_event['injected_at_step']."""
    return fragment_event["injected_at_step"]


def collect_events(traj_path: Path):
    """Yield (chain, k_inject, gold) for every injection event in this seed."""
    probs = load_trajectories(traj_path)
    for prob in probs:
        gold = prob.get("answer")
        pop = prob["population"]
        for chain in pop["chains"]:
            for fragment in chain.get("injected_fragments", []) or []:
                k = find_injected_chain_and_step(pop, fragment)
                if k is None:
                    continue
                yield prob, chain, int(k), gold


def baseline_prune_rate(traj_path: Path):
    """Per-step prune probability on chains that received NO injection.
    Returns dict step_idx -> (n_pruned_at_this_step, n_active_at_this_step)."""
    probs = load_trajectories(traj_path)
    by_step = Counter()  # (step_idx, "active" | "pruned_here")
    for prob in probs:
        pop = prob["population"]
        for chain in pop["chains"]:
            if chain.get("injected_fragments"):
                continue
            terminal = chain_terminal_step(chain)
            status = chain["status"]
            # At each step s, the chain was active. Mark whether it
            # was pruned at step s.
            for s in range(terminal + 1):
                by_step[(s, "active")] += 1
                if s == terminal and status == "pruned":
                    by_step[(s, "pruned_here")] += 1
    rate = {}
    for (s, kind), v in by_step.items():
        if kind == "active":
            denom = v
            num = by_step.get((s, "pruned_here"), 0)
            rate[s] = {"n_active": denom, "n_pruned_here": num, "p": num / denom if denom else 0.0}
    return rate


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for the mean of a binary indicator list."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def main():
    all_events = []  # list of dicts
    per_seed_buckets = {}
    per_seed_baseline = {}

    for path in PPFG_STAG_PATHS:
        seed_tag = path.name
        events_this_seed = []
        for prob, chain, k_inject, gold in collect_events(path):
            bucket = categorize_event(chain, k_inject, gold)
            terminal = chain_terminal_step(chain)
            events_this_seed.append({
                "problem_id": prob["problem_id"],
                "chain_id": chain["chain_id"],
                "k_inject": k_inject,
                "terminal_step": terminal,
                "status": chain["status"],
                "steps_after_inject": terminal - k_inject,
                "bucket": bucket,
            })
        per_seed_buckets[seed_tag] = Counter(e["bucket"] for e in events_this_seed)
        per_seed_baseline[seed_tag] = baseline_prune_rate(path)
        all_events.extend(events_this_seed)

    # Aggregate buckets across seeds
    agg_buckets = Counter(e["bucket"] for e in all_events)
    n_total = len(all_events)

    # Compute matched-step post-injection prune rate vs baseline
    # For each event with k_inject = k, examine the next 3 steps:
    #   - did the chain become pruned at step k+1, k+2, or k+3?
    # Compare to the baseline prune rate at those same step indices.

    # Per-event prune-within-3 indicator
    pruned_within_3 = [1 if e["bucket"] == "a_pruned_within_3" else 0 for e in all_events]
    p_post = sum(pruned_within_3) / n_total

    # Build a single baseline rate at step indices that match the injection
    # events' k_inject+1..k_inject+3 schedule.
    # We average across the three seeds' baselines.
    merged_baseline = {}
    for seed_tag, rate in per_seed_baseline.items():
        for s, rec in rate.items():
            agg = merged_baseline.setdefault(s, {"n_active": 0, "n_pruned_here": 0})
            agg["n_active"] += rec["n_active"]
            agg["n_pruned_here"] += rec["n_pruned_here"]
    for s, agg in merged_baseline.items():
        agg["p"] = agg["n_pruned_here"] / agg["n_active"] if agg["n_active"] else 0.0

    # For each event, sum the baseline prune probabilities at k+1, k+2, k+3
    # treating them as independent — the matched-window "any pruned in 3 steps"
    # approximation is 1 - prod_{d=1..3} (1 - p_baseline[k+d]).
    baseline_within_3 = []
    for e in all_events:
        k = e["k_inject"]
        any_pruned = 1.0
        for d in (1, 2, 3):
            p_d = merged_baseline.get(k + d, {"p": 0.0})["p"]
            any_pruned *= (1.0 - p_d)
        baseline_within_3.append(1.0 - any_pruned)
    p_base_mean = sum(baseline_within_3) / len(baseline_within_3)

    ci_post = bootstrap_ci(pruned_within_3, n_boot=2000)
    # baseline is a per-event probability (continuous); use bootstrap of mean
    ci_base = bootstrap_ci(baseline_within_3, n_boot=2000)

    ratio = p_post / p_base_mean if p_base_mean > 0 else float("inf")

    # Bucket fractions
    bucket_frac = {b: agg_buckets[b] / n_total for b in [
        "a_pruned_within_3",
        "b_promoted_correct_within_3",
        "c_promoted_incorrect_within_3",
        "d_still_active_past_k3",
    ]}

    summary = {
        "n_events": n_total,
        "bucket_counts": dict(agg_buckets),
        "bucket_fractions": bucket_frac,
        "post_injection_prune_rate_within_3steps": {
            "p": p_post,
            "ci_95": list(ci_post),
        },
        "matched_step_baseline_prune_rate_within_3steps": {
            "p_mean": p_base_mean,
            "ci_95": list(ci_base),
            "_note": "Per-event baseline computed as 1 - prod_{d=1..3} (1 - p_baseline[k+d]) where p_baseline[s] is the empirical prune-at-step-s rate over chains in the same seed that received NO injection. Averaged across all 322 events.",
        },
        "ratio_post_over_baseline": ratio,
        "per_seed_buckets": {k: dict(v) for k, v in per_seed_buckets.items()},
        "case_classification": classify_case(p_post, p_base_mean),
        "_inputs": [str(p) for p in PPFG_STAG_PATHS],
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def classify_case(p_post: float, p_base: float) -> str:
    if p_base <= 0:
        return "undefined (baseline zero)"
    ratio = p_post / p_base
    if 1 / 1.5 <= ratio <= 1.5:
        return "case_i_local_PRM_penalty_not_cascading"
    if ratio > 2.0:
        return "case_ii_elevated_post_inj_pruning"
    if ratio < 0.5:
        return "case_iii_suppressed_post_inj_pruning"
    return f"intermediate (ratio={ratio:.3f})"


if __name__ == "__main__":
    main()
