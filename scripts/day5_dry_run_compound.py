#!/usr/bin/env python
"""Dry-run the compound stagnation gate against EXISTING Day-5 trajectories.

For each of the 55 historical injection events (which used the OLD `is_stagnating`
rule), compute whether the proposed COMPOUND gate would still have allowed that
injection. Bucket the survivors by the original 4-category false-positive scheme
to verify the new criterion is directionally sane before any GPU spend.

Compound gate (ALL 5 required):
  1. PRM-low gate:    trend_mean(prm[-3:]) < 0.6
  2. PRM-flat gate:   max(prm[-3:]) - min(prm[-3:]) < 0.05  (original criterion)
  3. Headroom gate:   step_idx < max_steps - 4              (default max_steps=16, so < 12)
  4. Population gate: chain.latest_prm < median(latest_prm over active peers at this step)
  5. Liveness gate:   n_other_active_peers >= 3

Active-at-step-t reconstruction (per problem):
  A chain is "active at step t" iff:
    - it has produced at least t+1 steps (i.e., len(prm_scores) > t)
    - none of its prm_scores[0..t] were below prune_threshold (0.4 by default)
    - it has not yet emitted a final answer at step <= t
  This mirrors Population.advance()'s prune/promote logic.
"""
from __future__ import annotations
import json
import statistics
from collections import Counter
from pathlib import Path

# Compound-gate thresholds (parametrize for re-evaluation)
PRM_LOW_T = 0.70
PRM_FLAT_T = 0.05
HEADROOM_BUDGET = 4   # injection at step_idx < max_steps - HEADROOM_BUDGET
LIVENESS_MIN = 3
PRUNE_T = 0.4
MAX_STEPS = 16

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAVE_MV = True
except Exception:
    HAVE_MV = False


def norm(s): return s.strip().replace("$","").replace(" ","").lower() if s else ""

def is_correct(c, g):
    if c is None: return False
    if HAVE_MV:
        try: return bool(mv_verify(mv_parse(g), mv_parse(c)))
        except Exception: return norm(c) == norm(g)
    return norm(c) == norm(g)


def chain_step_status(chain: dict, t: int, prune_t: float):
    """Return one of 'active' | 'pruned' | 'promoted' | 'not_yet_produced'
    for the chain at step index t (0-based)."""
    prms = chain.get("prm_scores", []) or []
    steps = chain.get("steps", []) or []
    if len(prms) <= t or len(steps) <= t:
        return "not_yet_produced"
    # Check if pruned at any point [0..t]
    for i in range(t + 1):
        if prms[i] < prune_t:
            if i == t:
                return "pruned"  # just got pruned this step
            else:
                return "pruned"  # already pruned by step i < t
    # Check if completed (final answer detected) at step <= t.
    # We approximate by: chain.final_answer is not None AND chain ended at <= t+1 steps.
    # If chain has exactly t+1 steps and a final answer, it completed at this step.
    if chain.get("final_answer") and len(steps) == t + 1:
        return "promoted"
    return "active"


def get_active_peers_latest_prm(chains: list[dict], target_chain_id: int, t: int, prune_t: float):
    """Return list of latest_prm values for chains other than target that are
    active at step t."""
    out = []
    for c in chains:
        if c["chain_id"] == target_chain_id:
            continue
        status = chain_step_status(c, t, prune_t)
        if status == "active":
            prms = c.get("prm_scores", []) or []
            if prms and len(prms) > t:
                out.append(prms[t])
    return out


def evaluate_compound(target_chain: dict, all_chains: list[dict], inj_step: int):
    """Run the 5 gates. Return (compound_pass: bool, gates: dict)."""
    prms = target_chain.get("prm_scores", []) or []
    # Window: 3 PRM scores ending at inj_step-1 (the chain's history BEFORE the injection-step's PRM)
    # is_stagnating uses prm_scores[-window:] at the moment is_stagnating is called, which is
    # AFTER the new step's PRM is added. So `inj_step` (the step the fragment is injected
    # BEFORE) corresponds to PRM history prm[:inj_step].
    history = prms[max(0, inj_step - 3):inj_step]
    if len(history) < 3:
        # Old rule wouldn't fire either; mark all gates failed
        return False, {
            "gate1_prm_low": False, "gate2_prm_flat": False,
            "gate3_headroom": False, "gate4_population": False,
            "gate5_liveness": False,
            "history": history, "trend_mean": None,
        }
    trend_mean = statistics.mean(history)
    trend_range = max(history) - min(history)
    g1 = trend_mean < PRM_LOW_T
    g2 = trend_range < PRM_FLAT_T
    g3 = inj_step < (MAX_STEPS - HEADROOM_BUDGET)
    # Population: latest_prm of target at inj_step-1 vs median of active peers at same step
    target_latest = history[-1] if history else None
    peer_latest = get_active_peers_latest_prm(all_chains, target_chain["chain_id"], inj_step - 1, PRUNE_T)
    if peer_latest:
        peer_median = statistics.median(peer_latest)
        g4 = target_latest is not None and target_latest < peer_median
    else:
        g4 = False  # no peers to compare against
        peer_median = None
    g5 = len(peer_latest) >= LIVENESS_MIN
    gates = {
        "gate1_prm_low": g1, "gate2_prm_flat": g2,
        "gate3_headroom": g3, "gate4_population": g4, "gate5_liveness": g5,
        "history": history, "trend_mean": trend_mean,
        "target_latest": target_latest, "peer_median": peer_median,
        "n_peers": len(peer_latest),
    }
    return (g1 and g2 and g3 and g4 and g5), gates


def find_ppfg_stag_dirs(root: Path):
    out = {}
    for d in sorted(root.glob("ppfg-math500-20260514-*-j*")):
        c = d / "config.json"
        if not c.exists(): continue
        with c.open() as f: cfg = json.load(f)
        if cfg["ppfg"]["injection_rule"] == "stagnation":
            out[cfg["experiment"]["seed"]] = d
    return out


def classify_event(target_chain: dict, gold: str, inj_step: int):
    """Re-derive the 4 buckets from day5_inspect_stagnation.py."""
    tags = []
    final_answer = target_chain.get("final_answer")
    correct = is_correct(final_answer, gold)
    prms = target_chain.get("prm_scores", []) or []
    history = prms[max(0, inj_step - 3):inj_step]
    trend_mean = statistics.mean(history) if len(history) >= 3 else None
    trend_range = (max(history) - min(history)) if len(history) >= 3 else None
    n_steps_total = len(target_chain.get("steps", []) or [])
    steps_after_inj = n_steps_total - inj_step if inj_step is not None else None

    if correct:
        tags.append("target-succeeded")
    if trend_mean is not None and trend_mean >= 0.6 and (trend_range or 1) < 0.05:
        tags.append("high-flat-PRM")
    if steps_after_inj is not None and steps_after_inj <= 2:
        tags.append("near-completion")
    if not tags:
        tags.append("plausibly-helped")
    return tags


def main():
    root = Path("results")
    dirs = find_ppfg_stag_dirs(root)
    all_events = []
    for seed, d in sorted(dirs.items()):
        with (d / "trajectories.json").open() as f:
            recs = json.load(f)
        for rec in recs:
            gold = rec["answer"]
            chains = rec["population"]["chains"]
            chain_by_id = {c["chain_id"]: c for c in chains}
            for c in chains:
                for frag in c.get("injected_fragments", []):
                    inj_step = frag.get("injected_at_step")
                    if inj_step is None:
                        inj_step = frag.get("target_step_idx", 0)
                    compound_pass, gates = evaluate_compound(c, chains, inj_step)
                    tags = classify_event(c, gold, inj_step)
                    all_events.append({
                        "seed": seed, "problem_id": rec["problem_id"],
                        "target_id": c["chain_id"], "inj_step": inj_step,
                        "tags": tags, "compound_pass": compound_pass,
                        "gates": gates,
                    })

    n_events = len(all_events)
    n_pass = sum(1 for e in all_events if e["compound_pass"])
    n_reject = n_events - n_pass

    print(f"Total historical injection events:    {n_events}")
    print(f"Compound-gate would ALLOW:            {n_pass}  ({n_pass/n_events*100:.1f}%)")
    print(f"Compound-gate would REJECT:           {n_reject}  ({n_reject/n_events*100:.1f}%)")
    print()
    print("Gate-by-gate rejection breakdown (events the compound gate rejected, by which gate fired):")
    gate_names = ["gate1_prm_low", "gate2_prm_flat", "gate3_headroom", "gate4_population", "gate5_liveness"]
    for g in gate_names:
        n_fail = sum(1 for e in all_events if not e["gates"][g])
        n_alone = sum(1 for e in all_events
                      if not e["gates"][g] and all(e["gates"][o] for o in gate_names if o != g))
        print(f"  {g}: {n_fail} events fail this gate ({n_fail/n_events*100:.1f}%); "
              f"{n_alone} events fail ONLY this gate")
    print()
    # Bucket breakdown for events the compound gate ALLOWS
    print("=== Four-bucket breakdown on COMPOUND-ALLOWED events (the directional check) ===")
    if n_pass == 0:
        print("  Compound gate rejects ALL events — gate is too strict; criterion needs relaxation.")
    else:
        allowed = [e for e in all_events if e["compound_pass"]]
        bucket_counts = Counter()
        for e in allowed:
            for t in e["tags"]:
                bucket_counts[t] += 1
        for t, n in bucket_counts.most_common():
            print(f"  {t}: {n}  ({n/n_pass*100:.1f}%)")
    print()
    print("=== For reference, original Day-5 four-bucket counts (all 55 events) ===")
    orig_counts = Counter()
    for e in all_events:
        for t in e["tags"]:
            orig_counts[t] += 1
    for t, n in orig_counts.most_common():
        print(f"  {t}: {n}  ({n/n_events*100:.1f}%)")

    # Sample 5 representative events the compound gate allows
    print()
    print("=== 5 sample events the compound gate ALLOWS (target chains the new rule would still pick) ===")
    allowed = [e for e in all_events if e["compound_pass"]]
    for e in allowed[:5]:
        g = e["gates"]
        print(f"  s{e['seed']} prob={e['problem_id']} chain={e['target_id']} step={e['inj_step']}")
        print(f"    history={['{:.2f}'.format(x) for x in g['history']]} trend_mean={g['trend_mean']:.3f} "
              f"target_latest={g['target_latest']:.3f} peer_median={g['peer_median']:.3f} n_peers={g['n_peers']}")
        print(f"    tags={e['tags']}")


if __name__ == "__main__":
    main()
