#!/usr/bin/env python3
"""Cross-architecture replication of the post-injection pruning hazard
analysis (originally scripts/post_injection_prune_analysis.py, Qwen-only,
322 events, 2.75x post/baseline ratio -> Case ii elevated post-injection
pruning).

Motivated by ARR review (Reviewer 7Diq): "Replicate the per-event hazard
analysis (Sec 5.4) beyond Qwen so the strongest mechanistic claim isn't on
single-model." Limitations item 10 already discloses this as Qwen-only;
this script closes the gap using already-collected trajectories (no new
GPU compute) for LLaMA-3.1-8B-Instruct and DeepSeek-R1-Distill-Qwen-7B,
5 seeds each (42-46) -- more seeds than the original Qwen 3-seed analysis.

Reuses categorize_event/collect_events/baseline_prune_rate/bootstrap_ci/
classify_case UNCHANGED from post_injection_prune_analysis.py via
importlib, so the bucket definitions and ratio computation are
byte-identical to the published Qwen analysis.

Usage:
  PYTHONPATH=src python scripts/post_injection_prune_analysis_crossarch.py

Output:
  results/post_injection_prune_analysis_llama.json
  results/post_injection_prune_analysis_deepseek.json
"""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "pipa", str(Path(__file__).with_name("post_injection_prune_analysis.py")))
pipa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipa)

RESULTS = Path("results")

ARCH_PATHS = {
    "llama-3.1-8b-instruct": [
        RESULTS / "ppfg-math500-20260515-004424-j60993376",   # seed 42
        RESULTS / "ppfg-math500-20260515-004838-j60993385",   # seed 43
        RESULTS / "ppfg-math500-20260515-005243-j60993390",   # seed 44
        RESULTS / "ppfg-math500-20260516-052440-j61065161",   # seed 45
        RESULTS / "ppfg-math500-20260516-052820-j61065167",   # seed 46
    ],
    "deepseek-r1-distill-qwen-7b": [
        RESULTS / "ppfg-math500-20260515-004424-j60993377",   # seed 42
        RESULTS / "ppfg-math500-20260515-004838-j60993386",   # seed 43
        RESULTS / "ppfg-math500-20260515-005243-j60993391",   # seed 44
        RESULTS / "ppfg-math500-20260516-052819-j61065164",   # seed 45
        RESULTS / "ppfg-math500-20260516-053159-j61065170",   # seed 46
    ],
}


def run_one(arch: str, paths: list[Path]) -> dict:
    all_events = []
    per_seed_buckets = {}
    per_seed_baseline = {}

    for path in paths:
        seed_tag = path.name
        events_this_seed = []
        for prob, chain, k_inject, gold in pipa.collect_events(path):
            bucket = pipa.categorize_event(chain, k_inject, gold)
            terminal = pipa.chain_terminal_step(chain)
            events_this_seed.append({
                "problem_id": prob["problem_id"],
                "chain_id": chain["chain_id"],
                "k_inject": k_inject,
                "terminal_step": terminal,
                "status": chain["status"],
                "steps_after_inject": terminal - k_inject,
                "bucket": bucket,
            })
        per_seed_buckets[seed_tag] = dict(__import__("collections").Counter(
            e["bucket"] for e in events_this_seed))
        per_seed_baseline[seed_tag] = pipa.baseline_prune_rate(path)
        all_events.extend(events_this_seed)

    n_total = len(all_events)
    if n_total == 0:
        return {"arch": arch, "n_events": 0, "note": "no injection events found"}

    pruned_within_3 = [1 if e["bucket"] == "a_pruned_within_3" else 0 for e in all_events]
    p_post = sum(pruned_within_3) / n_total

    merged_baseline = {}
    for seed_tag, rate in per_seed_baseline.items():
        for s, rec in rate.items():
            agg = merged_baseline.setdefault(s, {"n_active": 0, "n_pruned_here": 0})
            agg["n_active"] += rec["n_active"]
            agg["n_pruned_here"] += rec["n_pruned_here"]
    for s, agg in merged_baseline.items():
        agg["p"] = agg["n_pruned_here"] / agg["n_active"] if agg["n_active"] else 0.0

    baseline_within_3 = []
    for e in all_events:
        k = e["k_inject"]
        any_pruned = 1.0
        for d in (1, 2, 3):
            p_d = merged_baseline.get(k + d, {"p": 0.0})["p"]
            any_pruned *= (1.0 - p_d)
        baseline_within_3.append(1.0 - any_pruned)
    p_base_mean = sum(baseline_within_3) / len(baseline_within_3)

    ci_post = pipa.bootstrap_ci(pruned_within_3, n_boot=2000)
    ci_base = pipa.bootstrap_ci(baseline_within_3, n_boot=2000)
    ratio = p_post / p_base_mean if p_base_mean > 0 else float("inf")

    bucket_frac = {b: dict(__import__("collections").Counter(e["bucket"] for e in all_events)).get(b, 0) / n_total
                   for b in ["a_pruned_within_3", "b_promoted_correct_within_3",
                             "c_promoted_incorrect_within_3", "d_still_active_past_k3"]}

    return {
        "arch": arch,
        "n_events": n_total,
        "n_seeds": len(paths),
        "bucket_fractions": bucket_frac,
        "post_injection_prune_rate_within_3steps": {"p": p_post, "ci_95": list(ci_post)},
        "matched_step_baseline_prune_rate_within_3steps": {"p_mean": p_base_mean, "ci_95": list(ci_base)},
        "ratio_post_over_baseline": ratio,
        "case_classification": pipa.classify_case(p_post, p_base_mean),
        "_inputs": [str(p) for p in paths],
    }


def main():
    for arch, paths in ARCH_PATHS.items():
        missing = [p for p in paths if not (p / "trajectories.json").exists()]
        if missing:
            print(f"WARNING: {arch} missing trajectory files: {missing}")
            continue
        out = run_one(arch, paths)
        slug = arch.split("-")[0]
        out_path = RESULTS / f"post_injection_prune_analysis_{slug}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"=== {arch} ===")
        print(json.dumps(out, indent=2))
        print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    main()
