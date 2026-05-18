#!/usr/bin/env python
"""Second-annotator classification of the 55 Day-5 PPFG-stag injection events
for inter-annotator agreement (Cohen's κ) on §5.3.1's four-bucket scheme.

The first-annotator labels are captured by re-running scripts/day5_inspect_stagnation.py
and dumping its per-event tags. This script re-implements the four-bucket
classification from the §5.3.1 prose definitions, independently of that script's
classify() function. Output buckets are deterministic from the saved trajectories;
three are fully mechanical, and the fourth is the complement of the union of the
other three.

Bucket definitions (re-derived from §5.3.1 of paper_outline.md):
  - target-succeeded: target chain has final status `promoted` AND is_correct
  - high-flat-PRM: mean of pre-injection 3-step PRM window >= 0.6 AND range < 0.05
  - near-completion: target produced final answer within <=2 steps after injection
  - plausibly-helped: complement of the union of the three above

Output:
  - results/inter_annotator_labels.json: per-event labels under each annotator
  - results/inter_annotator_kappa.json: per-bucket and overall κ, exact-match rate
  - Console summary

Usage:
  PYTHONPATH=src python scripts/inter_annotator_classify.py
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAVE_MV = True
except Exception:
    HAVE_MV = False


BUCKETS = ["target-succeeded", "high-flat-PRM", "near-completion", "plausibly-helped"]


def _norm(s):
    return (s or "").strip().replace("$", "").replace(" ", "").lower()


def _is_correct(candidate, gold):
    if candidate is None:
        return False
    if HAVE_MV:
        try:
            return bool(mv_verify(mv_parse(gold), mv_parse(candidate)))
        except Exception:
            return _norm(candidate) == _norm(gold)
    return _norm(candidate) == _norm(gold)


def _find_ppfg_stag_n100_dirs(root: Path):
    """Day-5 PPFG-stag n=100 cells, seeds 42/43/44, identified by config.json."""
    out = {}
    for d in sorted(root.glob("ppfg-math500-20260514-*")):
        cfg = d / "config.json"
        if not cfg.exists():
            continue
        try:
            with cfg.open() as f:
                c = json.load(f)
            ppfg_section = c.get("ppfg") or {}
            if ppfg_section.get("injection_rule") != "stagnation":
                continue
            exp_section = c.get("experiment") or {}
            seed = exp_section.get("seed")
            # The Day-5 n=100 PPFG-stag cells use experiment.name = 'ppfg_stag_math500';
            # the Day-8 n=500 replication cells use 'qwen_n500_ppfg_stag_math500' etc.
            # §5.3.1's 55-event classification is on the Day-5 n=100 cells, so anchor
            # the second-annotator κ against that exact event set.
            if exp_section.get("name") != "ppfg_stag_math500":
                continue
            base = (c.get("model") or {}).get("base_model", "")
            if "Qwen2.5-7B-Instruct" not in base:
                continue
            tk = (c.get("population") or {}).get("prune_threshold")
            if tk != 0.4:
                continue
            # Tie-break by directory name (alphabetical first wins on duplicate seed).
            if seed in out:
                continue
            out[seed] = d
        except Exception:
            continue
    return out


def _gather_events(traj_path: Path, seed: int):
    """Yield one event dict per (chain, injected_fragment) pair on the PPFG-stag cell."""
    with traj_path.open() as f:
        records = json.load(f)
    events = []
    for rec in records:
        gold = rec["answer"]
        problem_id = rec["problem_id"]
        for ch in rec["population"]["chains"]:
            frags = ch.get("injected_fragments") or []
            if not frags:
                continue
            final_status = str(ch.get("status", ""))
            final_answer = ch.get("final_answer")
            correct = _is_correct(final_answer, gold)
            prm_seq = ch.get("prm_scores") or []
            n_steps_total = len(ch.get("steps") or [])
            target_chain_id = ch.get("chain_id")
            for frag in frags:
                inj_at = frag.get("injected_at_step")
                if inj_at is None:
                    inj_at = frag.get("target_step_idx")
                window_lo = max(0, (inj_at or 0) - 3)
                window_hi = inj_at or 0
                window = prm_seq[window_lo:window_hi]
                if window:
                    w_mean = sum(window) / len(window)
                    w_range = max(window) - min(window)
                else:
                    w_mean = None
                    w_range = None
                steps_after = (n_steps_total - (inj_at or 0)) if inj_at is not None else None
                events.append({
                    "seed": seed,
                    "problem_id": problem_id,
                    "target_chain": target_chain_id,
                    "inj_at_step": inj_at,
                    "window_pre_inj": window,
                    "window_mean": w_mean,
                    "window_range": w_range,
                    "steps_after_inj": steps_after,
                    "final_status": final_status,
                    "final_answer": final_answer,
                    "correct": correct,
                })
    return events


def _classify_a2(ev):
    """Second-annotator classifier — re-implemented from §5.3.1 prose, in a different
    coding style than scripts/day5_inspect_stagnation.py::classify().
    Returns a frozenset of bucket labels (each event can land in multiple buckets);
    plausibly-helped is exclusive of the other three."""
    tags = set()
    # target-succeeded: final status was promoted AND answer was right
    status_str = (ev.get("final_status") or "").lower()
    promoted = ("promoted" in status_str)
    if promoted and ev.get("correct"):
        tags.add("target-succeeded")
    # high-flat-PRM cosmetic plateau
    wm, wr = ev.get("window_mean"), ev.get("window_range")
    if wm is not None and wr is not None and wm >= 0.6 and wr < 0.05:
        tags.add("high-flat-PRM")
    # near-completion: produced final answer within 2 steps after injection
    sa = ev.get("steps_after_inj")
    if sa is not None and sa <= 2:
        tags.add("near-completion")
    if not tags:
        tags.add("plausibly-helped")
    return frozenset(tags)


def _classify_a1(ev):
    """First-annotator classifier — replicates the classify() function in
    scripts/day5_inspect_stagnation.py (the canonical hand-classification used
    in §5.3.1 Table 2). Imported here so we can capture per-event labels without
    modifying the original script. Inputs come from _gather_events above, which
    matches the original script's inspect() output schema."""
    tags = set()
    if ev.get("correct"):
        tags.add("target-succeeded")
    wm = ev.get("window_mean")
    wr = ev.get("window_range")
    if wm is not None and wm >= 0.6 and (wr if wr is not None else 1) < 0.05:
        tags.add("high-flat-PRM")
    sa = ev.get("steps_after_inj")
    if sa is not None and sa <= 2:
        tags.add("near-completion")
    if not tags:
        tags.add("plausibly-helped")
    return frozenset(tags)


def _cohens_kappa_binary(labels_a, labels_b):
    """Cohen's κ for a binary classification.
    labels_a, labels_b are lists of {0, 1} of the same length."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float("nan"), 0.0
    # observed agreement
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    # expected agreement under chance
    p1a = sum(labels_a) / n
    p1b = sum(labels_b) / n
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    if pe >= 1.0:
        return float("nan"), po
    kappa = (po - pe) / (1.0 - pe)
    return kappa, po


def main():
    root = Path("results")
    dirs = _find_ppfg_stag_n100_dirs(root)
    if len(dirs) != 3:
        print(f"WARNING: expected 3 PPFG-stag n=100 cells, found {len(dirs)} — proceeding anyway")
        for s, d in sorted(dirs.items()):
            print(f"  seed={s}: {d}")

    all_events = []
    for seed, d in sorted(dirs.items()):
        evs = _gather_events(d / "trajectories.json", seed)
        all_events.extend(evs)
    print(f"Gathered {len(all_events)} injection events across {len(dirs)} PPFG-stag cells")

    # Tag each event under both annotators.
    per_event = []
    for ev in all_events:
        a1 = _classify_a1(ev)
        a2 = _classify_a2(ev)
        per_event.append({
            "seed": ev["seed"],
            "problem_id": ev["problem_id"],
            "target_chain": ev["target_chain"],
            "inj_at_step": ev["inj_at_step"],
            "window_pre_inj": ev["window_pre_inj"],
            "window_mean": ev["window_mean"],
            "window_range": ev["window_range"],
            "steps_after_inj": ev["steps_after_inj"],
            "final_status": ev["final_status"],
            "correct": ev["correct"],
            "a1_buckets": sorted(a1),
            "a2_buckets": sorted(a2),
        })

    # Per-bucket Cohen's κ.
    kappa_by_bucket = {}
    a1_counts = defaultdict(int)
    a2_counts = defaultdict(int)
    for b in BUCKETS:
        la = [1 if b in p["a1_buckets"] else 0 for p in per_event]
        lb = [1 if b in p["a2_buckets"] else 0 for p in per_event]
        for v in la:
            a1_counts[b] += v
        for v in lb:
            a2_counts[b] += v
        kappa, po = _cohens_kappa_binary(la, lb)
        # Per-event exact-match rate for this bucket
        kappa_by_bucket[b] = {
            "kappa": kappa,
            "exact_match_rate": po,
            "a1_count": sum(la),
            "a2_count": sum(lb),
            "n_disagreements": sum(1 for x, y in zip(la, lb) if x != y),
        }

    # Overall exact-match rate across the four-bucket label set.
    exact_full = sum(1 for p in per_event
                     if frozenset(p["a1_buckets"]) == frozenset(p["a2_buckets"])) / len(per_event)

    # Macro-averaged κ.
    macro_kappa = sum(v["kappa"] for v in kappa_by_bucket.values()) / len(kappa_by_bucket)

    out = {
        "n_events": len(per_event),
        "buckets": kappa_by_bucket,
        "overall_set_exact_match_rate": exact_full,
        "macro_averaged_kappa": macro_kappa,
        "a1_aggregate_counts": dict(a1_counts),
        "a2_aggregate_counts": dict(a2_counts),
    }

    Path("results/inter_annotator_kappa.json").write_text(json.dumps(out, indent=2))
    Path("results/inter_annotator_labels.json").write_text(json.dumps(per_event, indent=2))

    print()
    print(f"Per-bucket Cohen's κ (A1=day5_inspect_stagnation.py, A2=this script):")
    print(f"  {'Bucket':<22} {'κ':>7} {'EM-rate':>9} {'A1 count':>9} {'A2 count':>9} {'disagree':>9}")
    for b in BUCKETS:
        v = kappa_by_bucket[b]
        k_str = f"{v['kappa']:.3f}" if v['kappa'] == v['kappa'] else "nan"
        print(f"  {b:<22} {k_str:>7} {v['exact_match_rate']:>9.3f} "
              f"{v['a1_count']:>9} {v['a2_count']:>9} {v['n_disagreements']:>9}")
    print()
    print(f"Macro-averaged κ:            {macro_kappa:.3f}")
    print(f"Overall set-exact-match rate: {exact_full:.3f}")
    print()
    print("Saved: results/inter_annotator_kappa.json")
    print("Saved: results/inter_annotator_labels.json")


if __name__ == "__main__":
    main()
