"""Parse PPFG injection events from trajectories.json (per-chain injected_fragments).

For PPFG-stag and PPFG-random at n=500 (3 seeds each), aggregate:
  - n_injection_events (total across seeds)
  - per_problem_injection_rate (events / total problem-instances)
  - frac_problems_with_at_least_one_injection
  - mean / std fragment length (k* = len(fragment_steps))
  - mean / std fragment quality (per-event 'quality')
  - mean / std compatibility score
  - mean / median target-chain step-index at injection

Writes results/firing_stats_n500.json.
"""

from __future__ import annotations
import json
import statistics
from pathlib import Path

DIRS = {
    "ppfg_stag": [
        "results/ppfg-math500-20260514-161757-j60966183",
        "results/ppfg-math500-20260514-161757-j60966186",
        "results/ppfg-math500-20260514-161757-j60966195",
    ],
    "ppfg_random": [
        "results/ppfg-math500-20260514-161757-j60966200",
        "results/ppfg-math500-20260514-161757-j60966204",
        "results/ppfg-math500-20260514-161757-j60966213",
    ],
}


def parse_one(dir_path):
    t = json.load(open(f"{dir_path}/trajectories.json"))
    events = []
    n_problems = len(t)
    problems_with_inj = 0
    for p in t:
        chains = p.get("population", p).get("chains", [])
        any_inj = False
        for c in chains:
            for ev in c.get("injected_fragments", []) or []:
                events.append(ev)
                any_inj = True
        if any_inj:
            problems_with_inj += 1
    return events, n_problems, problems_with_inj


def aggregate(events_per_seed, n_problems_per_seed, frac_per_seed):
    all_events = [e for seed_evs in events_per_seed for e in seed_evs]
    n_total = sum(n_problems_per_seed)
    n_events = len(all_events)

    def m_s(vals):
        if not vals:
            return None, None
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mu, sd

    k_stars = [len(e["fragment_steps"]) for e in all_events]
    qualities = [e["quality"] for e in all_events]
    compats = [e["compat_score"] for e in all_events]
    inj_steps = [e["injected_at_step"] for e in all_events]

    return {
        "n_seeds": len(events_per_seed),
        "n_total_problem_instances": n_total,
        "n_injection_events": n_events,
        "per_problem_injection_rate": n_events / n_total if n_total else 0.0,
        "frac_problems_with_injection_mean": statistics.mean(frac_per_seed) if frac_per_seed else 0.0,
        "fragment_length_mean": m_s(k_stars)[0],
        "fragment_length_std": m_s(k_stars)[1],
        "fragment_quality_mean": m_s(qualities)[0],
        "fragment_quality_std": m_s(qualities)[1],
        "compat_score_mean": m_s(compats)[0],
        "compat_score_std": m_s(compats)[1],
        "target_step_idx_mean": m_s(inj_steps)[0],
        "target_step_idx_median": statistics.median(inj_steps) if inj_steps else None,
    }


def main():
    summary = {}
    for cell_name, dirs in DIRS.items():
        events_per_seed, n_per_seed, frac_per_seed = [], [], []
        for d in dirs:
            evs, n_probs, n_inj = parse_one(d)
            events_per_seed.append(evs)
            n_per_seed.append(n_probs)
            frac_per_seed.append(n_inj / n_probs if n_probs else 0.0)
        summary[cell_name] = aggregate(events_per_seed, n_per_seed, frac_per_seed)

    out_path = Path("results/firing_stats_n500.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")
    print()
    for cell, s in summary.items():
        print(f"=== {cell} ===")
        print(f"  events: {s['n_injection_events']} / {s['n_total_problem_instances']} problem-instances "
              f"= {s['per_problem_injection_rate']*100:.1f}% per-problem rate")
        print(f"  frac problems with ≥1 injection: {s['frac_problems_with_injection_mean']*100:.1f}%")
        print(f"  fragment length: {s['fragment_length_mean']:.2f} ± {s['fragment_length_std']:.2f} steps")
        print(f"  fragment quality: {s['fragment_quality_mean']:.3f} ± {s['fragment_quality_std']:.3f}")
        print(f"  compatibility: {s['compat_score_mean']:.3f} ± {s['compat_score_std']:.3f}")
        print(f"  target step idx at injection: {s['target_step_idx_mean']:.2f} (median {s['target_step_idx_median']})")
        print()


if __name__ == "__main__":
    main()
