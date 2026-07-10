#!/usr/bin/env python
"""Stratify the four-bucket targeting classification (Appendix B, 322 events,
n=500 Qwen PPFG-stag corpus) by problem base-rate difficulty, to separate the
"model already succeeds most of the time" (base-rate) explanation from the
"PRM plateau" (landscape) explanation for why 89% of injections don't target
a chain with room to be rescued.

Motivated by ARR review (Reviewer 7Diq): "the 'well-targeted' share barely
improves (11% vs 14%) between the harder first-100 slice and the full
n=500 corpus, suggesting PRM flatness is the binding constraint — but this
separation is not explicitly made in the paper."

Method:
  1. Per-problem difficulty = fraction of independent-baseline chains correct,
     pooled across the 3 canonical Qwen n=500 independent seeds (42/43/44),
     using the FIXED nested-brace extractor (chain.py::detect_final_answer)
     and the same answers_equivalent() comparator as
     scripts/recompute_metrics_post_fix.py, so difficulty is computed with
     the same validated correctness pipeline as all headline Pass@k numbers.
  2. Injection events + bucket tags are gathered via the UNCHANGED
     scripts/inter_annotator_classify.py::_gather_events / _classify_a1 —
     byte-identical to the officially reported 322-event / 14.0% analysis in
     results/bucket_classification_n500.json — so this script adds a
     difficulty join key without altering the underlying classification.
  3. Problems are split into difficulty terciles by pooled independent
     Pass@1 (24 chains/problem: 3 seeds x 8 chains); events are joined to
     their problem's tercile; bucket rates are reported per tercile.

No GPU compute. Pure re-extraction + join + arithmetic over saved trajectories.

Usage:
  PYTHONPATH=src python scripts/targeting_by_difficulty_stratum.py

Output:
  results/targeting_by_difficulty_stratum.json
"""
from __future__ import annotations
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

from hyp_forest.chains.chain import detect_final_answer

spec = importlib.util.spec_from_file_location(
    "iac", str(Path(__file__).with_name("inter_annotator_classify.py")))
iac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iac)

spec2 = importlib.util.spec_from_file_location(
    "rmpf", str(Path(__file__).with_name("recompute_metrics_post_fix.py")))
rmpf = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(rmpf)

INDEP_DIRS = {
    42: "results/independent-math500-20260514-161757-j60966078",
    43: "results/independent-math500-20260514-161757-j60966131",
    44: "results/independent-math500-20260514-161757-j60966148",
}
PPFG_STAG_DIRS = {
    42: "results/ppfg-math500-20260514-161757-j60966183",
    43: "results/ppfg-math500-20260514-161757-j60966186",
    44: "results/ppfg-math500-20260514-161757-j60966195",
}
BUCKETS = ["high-flat-PRM", "near-completion", "target-succeeded", "plausibly-helped"]


def extract_chain_answer(chain: dict):
    steps = chain.get("steps", [])
    if not steps:
        return None
    a = detect_final_answer(steps[-1])
    if a is not None:
        return a
    for s in reversed(steps[:-1]):
        a = detect_final_answer(s)
        if a is not None:
            return a
    return None


def per_problem_difficulty():
    """Return {problem_id: (n_correct, n_total)} pooled across 3 indep seeds,
    using the fixed extractor + answers_equivalent comparator."""
    counts = defaultdict(lambda: [0, 0])
    for seed, d in INDEP_DIRS.items():
        with (Path(d) / "trajectories.json").open() as f:
            records = json.load(f)
        for rec in records:
            gold = rec["answer"]
            pid = rec["problem_id"]
            for ch in rec["population"]["chains"]:
                pred = extract_chain_answer(ch)
                correct = rmpf.answers_equivalent(pred, gold)
                counts[pid][1] += 1
                if correct:
                    counts[pid][0] += 1
    return {pid: (c, n) for pid, (c, n) in counts.items()}


def main():
    difficulty = per_problem_difficulty()
    n_problems = len(difficulty)
    print(f"Pooled independent-baseline difficulty computed for {n_problems} problems "
          f"(3 seeds x 8 chains = up to {3*8}/problem)")

    # Rank problems by pooled Pass@1 (n_correct/n_total) into terciles.
    scored = sorted(difficulty.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
    n = len(scored)
    tercile_bounds = [n // 3, 2 * n // 3]
    pid_to_tercile = {}
    for i, (pid, _) in enumerate(scored):
        if i < tercile_bounds[0]:
            pid_to_tercile[pid] = "hard"
        elif i < tercile_bounds[1]:
            pid_to_tercile[pid] = "medium"
        else:
            pid_to_tercile[pid] = "easy"

    tercile_pass1 = defaultdict(list)
    for pid, (c, tot) in difficulty.items():
        tercile_pass1[pid_to_tercile[pid]].append(c / max(tot, 1))
    tercile_mean_pass1 = {
        t: round(sum(v) / len(v), 3) for t, v in tercile_pass1.items()
    }

    # Gather the OFFICIAL 322-event corpus + bucket tags, unmodified.
    events = []
    for seed, d in PPFG_STAG_DIRS.items():
        events += iac._gather_events(Path(d) / "trajectories.json", seed)
    print(f"Gathered {len(events)} official injection events (should match Appendix B: 322)")

    per_tercile_counts = {t: Counter() for t in ["hard", "medium", "easy"]}
    per_tercile_helped = {t: 0 for t in ["hard", "medium", "easy"]}
    per_tercile_n = {t: 0 for t in ["hard", "medium", "easy"]}
    unmatched = 0
    for ev in events:
        pid = ev["problem_id"]
        if pid not in pid_to_tercile:
            unmatched += 1
            continue
        t = pid_to_tercile[pid]
        tags = iac._classify_a1(ev)
        per_tercile_n[t] += 1
        for tag in tags:
            per_tercile_counts[t][tag] += 1
        if tags == frozenset({"plausibly-helped"}):
            per_tercile_helped[t] += 1

    out = {
        "n_problems": n_problems,
        "n_events_total": len(events),
        "n_events_unmatched_to_difficulty": unmatched,
        "tercile_mean_indep_pass1": tercile_mean_pass1,
        "tercile_n_events": per_tercile_n,
        "tercile_bucket_pct": {
            t: {b: round(100 * per_tercile_counts[t].get(b, 0) / max(per_tercile_n[t], 1), 1)
                for b in BUCKETS}
            for t in ["hard", "medium", "easy"]
        },
        "tercile_well_targeted_pct": {
            t: round(100 * per_tercile_helped[t] / max(per_tercile_n[t], 1), 1)
            for t in ["hard", "medium", "easy"]
        },
        "note": (
            "Difficulty terciles computed from pooled independent-baseline Pass@1 "
            "(fixed extractor, 24 chains/problem, seeds 42-44), independent of the "
            "PPFG-stag population being classified. Event gathering and bucket "
            "classification (_gather_events/_classify_a1) are unmodified from "
            "scripts/inter_annotator_classify.py, so totals reconcile against "
            "results/bucket_classification_n500.json (322 events, 14.0% well-targeted)."
        ),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/targeting_by_difficulty_stratum.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
