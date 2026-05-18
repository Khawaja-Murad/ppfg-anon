#!/usr/bin/env python
"""Hand-inspection: find PPFG-stag injection events where the target was flagged
'stagnating' but didn't actually need help.

Heuristics for 'didn't need help':
  (a) target chain ended in PROMOTED (got a final answer) and was_correct=True
      → injection landed on a chain that was succeeding.
  (b) target's PRM trend over the last 3 pre-injection scores was HIGH-flat
      (mean >= 0.6 with range < 0.05) → 'cosmetic plateau'.
  (c) target had <= 2 steps remaining (i.e., produced final answer within 2 steps
      after injection) → near-completion false positive.

Usage:
  PYTHONPATH=src python scripts/day5_inspect_stagnation.py
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAVE_MV = True
except Exception:
    HAVE_MV = False


def norm(s):
    return s.strip().replace("$", "").replace(" ", "").lower() if s else ""


def is_correct(cand, gold):
    if cand is None:
        return False
    if HAVE_MV:
        try:
            return bool(mv_verify(mv_parse(gold), mv_parse(cand)))
        except Exception:
            return norm(cand) == norm(gold)
    return norm(cand) == norm(gold)


def find_ppfg_stag_dirs(root: Path):
    out = {}
    for d in sorted(root.glob("ppfg-math500-20260514-*-j*")):
        cfg = d / "config.json"
        if not cfg.exists():
            continue
        try:
            with cfg.open() as f:
                c = json.load(f)
            if c["ppfg"]["injection_rule"] == "stagnation":
                seed = c["experiment"]["seed"]
                out[seed] = d
        except Exception:
            continue
    return out


def inspect(traj_path: Path, seed: int):
    """Walk all injection events, classify each."""
    with traj_path.open() as f:
        records = json.load(f)
    events = []
    for rec in records:
        gold = rec["answer"]
        problem_id = rec["problem_id"]
        for ch in rec["population"]["chains"]:
            fragments = ch.get("injected_fragments", [])
            if not fragments:
                continue
            target_id = ch["chain_id"]
            final_status = str(ch.get("status", ""))
            final_answer = ch.get("final_answer")
            correct = is_correct(final_answer, gold)
            prm_seq = ch.get("prm_scores", []) or []
            n_steps_total = len(ch.get("steps", []))
            for frag in fragments:
                inj_at = frag.get("injected_at_step", None)
                if inj_at is None:
                    inj_at = frag.get("target_step_idx", None)
                # PRM trend at the moment of injection (window of 3 before)
                trend = prm_seq[max(0, (inj_at or 0) - 3):(inj_at or 0)] if inj_at is not None else []
                trend_mean = statistics.mean(trend) if trend else None
                trend_range = (max(trend) - min(trend)) if trend else None
                steps_after_inj = (n_steps_total - (inj_at or 0)) if inj_at is not None else None
                events.append({
                    "seed": seed,
                    "problem_id": problem_id,
                    "target_chain": target_id,
                    "inj_at_step": inj_at,
                    "trend_pre_inj": trend,
                    "trend_mean": trend_mean,
                    "trend_range": trend_range,
                    "steps_after_inj": steps_after_inj,
                    "final_status": final_status,
                    "final_answer": final_answer,
                    "correct": correct,
                    "fragment_q": frag.get("quality", frag.get("source_prm_mean")),
                    "fragment_len": frag.get("length", frag.get("n_steps")),
                    "compat": frag.get("compat_score"),
                })
    return events


def classify(ev):
    """Tag each event with which false-positive bucket(s) it falls into."""
    tags = []
    if ev["correct"]:
        tags.append("target-succeeded")  # injection on a chain that got right answer anyway
    if ev["trend_mean"] is not None and ev["trend_mean"] >= 0.6 and (ev["trend_range"] or 1) < 0.05:
        tags.append("high-flat-PRM")  # cosmetic plateau
    if ev["steps_after_inj"] is not None and ev["steps_after_inj"] <= 2:
        tags.append("near-completion")
    if not tags:
        tags.append("plausibly-helped")
    return tags


def main():
    root = Path("results")
    dirs = find_ppfg_stag_dirs(root)
    all_events = []
    for seed, d in sorted(dirs.items()):
        evs = inspect(d / "trajectories.json", seed)
        all_events.extend(evs)
    print(f"Total injection events across 3 seeds: {len(all_events)}")
    if not all_events:
        return
    # Classify and bucket
    from collections import Counter
    bucket_counts = Counter()
    for ev in all_events:
        for t in classify(ev):
            bucket_counts[t] += 1
    print()
    print("Per-bucket counts (one event can land in multiple buckets):")
    for t, n in bucket_counts.most_common():
        print(f"  {t}: {n}  ({n / len(all_events) * 100:.1f}%)")
    # Show 10 representative "didn't need help" cases
    print()
    print("Sample 10 events flagged as false-positive (target didn't need help):")
    false_positives = [
        ev for ev in all_events
        if any(t in ("target-succeeded", "high-flat-PRM", "near-completion") for t in classify(ev))
    ]
    for ev in false_positives[:10]:
        tags = classify(ev)
        trend_str = "[" + ", ".join(f"{x:.2f}" for x in (ev["trend_pre_inj"] or [])) + "]"
        print(f"  seed={ev['seed']}  prob={ev['problem_id']}  chain={ev['target_chain']}  "
              f"inj_step={ev['inj_at_step']}  trend={trend_str}  "
              f"after={ev['steps_after_inj']}  final={ev['final_status']}  correct={ev['correct']}  "
              f"tags={tags}")


if __name__ == "__main__":
    main()
