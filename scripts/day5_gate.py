#!/usr/bin/env python
"""Day-5 diversity-rescue gate: compute matched-N comparison between
independent (existing 500-prob runs, sliced to first n_subset) and PPFG-stag
(new 100-prob runs, 3 seeds), then apply the gate from the Day-5 directive.

Gate:
  - mode_rate Δ (indep - ppfg_stag) >= 3 absolute points on 3-seed mean
  - Pass@k regression at every k bounded by max(1σ, 1 absolute point)
    where σ is the indep seed std at that k.

Usage:
  python scripts/day5_gate.py [--n_subset 100]
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAVE_MV = True
except Exception:
    HAVE_MV = False


def norm(s: str | None) -> str:
    return s.strip().replace("$", "").replace(" ", "").lower() if s else ""


def is_correct(cand: str | None, gold: str) -> bool:
    if cand is None:
        return False
    if HAVE_MV:
        try:
            return bool(mv_verify(mv_parse(gold), mv_parse(cand)))
        except Exception:
            return norm(cand) == norm(gold)
    return norm(cand) == norm(gold)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_metrics(traj_path: Path, n_subset: int = 100, k_values=(1, 2, 4, 8), mode_threshold: float = 0.5):
    """Compute Pass@k and the runner-style binary answer_mode_rate.

    answer_mode_rate matches src/hyp_forest/metrics/diversity.py::answer_mode_rate:
    fraction of problems where the most-common final answer's share of non-null
    answers exceeds `mode_threshold` (default 0.5).
    """
    with traj_path.open() as f:
        records = json.load(f)
    records = records[:n_subset]
    pass_buckets = {k: [] for k in k_values}
    collapsed = 0
    n_with_answers = 0
    n_chains_per = []
    for rec in records:
        gold = rec["answer"]
        chains = rec["population"]["chains"]
        finals = [c.get("final_answer") for c in chains]
        n_total = len(finals)
        n_chains_per.append(n_total)
        corrects = sum(1 for fa in finals if is_correct(fa, gold))
        for k in k_values:
            if n_total < k:
                continue
            pass_buckets[k].append(pass_at_k(n_total, corrects, k))
        # answer_mode_rate (binary, runner-style): >threshold of non-null finals same
        non_null = [fa for fa in finals if fa is not None]
        if non_null:
            mode_count = Counter(non_null).most_common(1)[0][1]
            if mode_count / len(non_null) > mode_threshold:
                collapsed += 1
            n_with_answers += 1
    out = {f"pass@{k}": (sum(v) / len(v) if v else None) for k, v in pass_buckets.items()}
    out["mode_rate"] = (collapsed / n_with_answers) if n_with_answers else None
    out["n_problems"] = len(records)
    out["n_chains"] = statistics.mode(n_chains_per) if n_chains_per else None
    return out


def aggregate(per_seed: dict[int, dict]) -> dict:
    """Mean ± std across seeds for each metric."""
    metrics = list(next(iter(per_seed.values())).keys())
    agg = {}
    for m in metrics:
        if m in ("n_problems", "n_chains"):
            continue
        vals = [per_seed[s][m] for s in per_seed if per_seed[s][m] is not None]
        if len(vals) >= 2:
            agg[m] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals), "n_seeds": len(vals)}
        elif len(vals) == 1:
            agg[m] = {"mean": vals[0], "std": 0.0, "n_seeds": 1}
        else:
            agg[m] = {"mean": None, "std": None, "n_seeds": 0}
    return agg


def find_ppfg_stag_dirs(results_root: Path, seeds=(42, 43, 44)) -> dict[int, Path]:
    """Find the latest ppfg-math500-20260514-*-j<jobid> dir for each seed,
    by inspecting saved config.json's experiment.seed."""
    candidates = sorted(results_root.glob("ppfg-math500-20260514-*-j*"))
    out: dict[int, Path] = {}
    for d in candidates:
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open() as f:
                cfg = json.load(f)
            seed = cfg["experiment"]["seed"]
            rule = cfg["ppfg"]["injection_rule"]
            if rule == "stagnation" and seed in seeds:
                # Take the latest dir per seed (timestamp in name is monotonic)
                if seed not in out or d.name > out[seed].name:
                    out[seed] = d
        except Exception:
            continue
    return out


def fmt(v, fmt_str=".4f"):
    return f"{v:{fmt_str}}" if v is not None else "---"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_subset", type=int, default=100)
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()

    root = Path(args.results_root)
    seeds = [42, 43, 44]

    # Indep anchor (subset of full 500-prob trajectories)
    indep_per_seed = {}
    for s in seeds:
        p = root / f"independent-math500-seed{s}" / "trajectories.json"
        indep_per_seed[s] = compute_metrics(p, n_subset=args.n_subset)
    indep_agg = aggregate(indep_per_seed)

    # PPFG-stag (full traj from new 100-prob runs)
    ppfg_dirs = find_ppfg_stag_dirs(root)
    if not ppfg_dirs:
        print("WARNING: no PPFG-stag run dirs found yet — only indep anchor will print.")
        ppfg_per_seed: dict[int, dict] = {}
    else:
        ppfg_per_seed = {}
        for s in seeds:
            if s in ppfg_dirs:
                p = ppfg_dirs[s] / "trajectories.json"
                if p.exists():
                    ppfg_per_seed[s] = compute_metrics(p, n_subset=args.n_subset)
                else:
                    print(f"NOTE: seed {s} dir found but no trajectories.json yet")
        ppfg_agg = aggregate(ppfg_per_seed) if ppfg_per_seed else {}

    # Print
    print(f"\n=== Matched-N comparison (n_subset={args.n_subset}) ===\n")
    print(f"  INDEP  ({len(indep_per_seed)} seeds):")
    for s in seeds:
        m = indep_per_seed[s]
        print(f"    seed {s}: pass@1={fmt(m['pass@1'])} pass@2={fmt(m['pass@2'])} pass@4={fmt(m['pass@4'])} pass@8={fmt(m['pass@8'])} mode={fmt(m['mode_rate'])}")
    print(f"    aggregate:")
    for k in ("pass@1", "pass@2", "pass@4", "pass@8", "mode_rate"):
        a = indep_agg[k]
        print(f"      {k}: {fmt(a['mean'])} ± {fmt(a['std'])}  (n={a['n_seeds']})")

    if not ppfg_per_seed:
        print("\n  PPFG-STAG: not yet available\n")
        return

    print(f"\n  PPFG-STAG ({len(ppfg_per_seed)} seeds):")
    for s in seeds:
        if s in ppfg_per_seed:
            m = ppfg_per_seed[s]
            print(f"    seed {s}: pass@1={fmt(m['pass@1'])} pass@2={fmt(m['pass@2'])} pass@4={fmt(m['pass@4'])} pass@8={fmt(m['pass@8'])} mode={fmt(m['mode_rate'])}")
    print(f"    aggregate:")
    for k in ("pass@1", "pass@2", "pass@4", "pass@8", "mode_rate"):
        a = ppfg_agg[k]
        print(f"      {k}: {fmt(a['mean'])} ± {fmt(a['std'])}  (n={a['n_seeds']})")

    # Gate
    print("\n=== Diversity-rescue gate ===\n")
    indep_mode = indep_agg["mode_rate"]["mean"]
    ppfg_mode = ppfg_agg["mode_rate"]["mean"]
    mode_delta = indep_mode - ppfg_mode
    print(f"  mode_rate Δ = indep ({indep_mode:.4f}) - ppfg_stag ({ppfg_mode:.4f}) = {mode_delta:+.4f}  ({mode_delta*100:+.2f} pts)")
    mode_pass = mode_delta >= 0.03
    print(f"  ≥ 3 absolute points? {'PASS' if mode_pass else 'FAIL'}")

    print("\n  Pass@k regression bounds (regression bounded by max(1σ, 1 abs pt) per k):")
    pass_k_ok = True
    pass_k_reports = []
    for k in ("pass@1", "pass@2", "pass@4", "pass@8"):
        ind_m = indep_agg[k]["mean"]
        ind_s = indep_agg[k]["std"]
        ppfg_m = ppfg_agg[k]["mean"]
        if ind_m is None or ppfg_m is None:
            print(f"    {k}: SKIP (missing)")
            continue
        delta = ppfg_m - ind_m  # positive = improvement
        bound = max(ind_s, 0.01)  # 1σ or 1 abs point
        ok = (delta >= -bound)
        pass_k_ok = pass_k_ok and ok
        verdict = "OK" if ok else "REGRESS"
        pass_k_reports.append((k, ind_m, ppfg_m, delta, bound, verdict))
        print(f"    {k}: indep={ind_m:.4f} ppfg={ppfg_m:.4f} Δ={delta:+.4f} bound=±{bound:.4f} -> {verdict}")

    print(f"\n  Pass@k all OK? {'YES' if pass_k_ok else 'NO'}")
    print(f"\n  OUTCOME: ", end="")
    if mode_pass and pass_k_ok:
        print("A — gate cleanly passed (proceed to Day 6, do NOT start today)")
    elif pass_k_ok and not mode_pass:
        print(f"B — Pass@k bounded but mode_rate Δ = {mode_delta*100:+.2f} pts < 3 (surface to the lead author, hand-inspect 10 problems)")
    else:
        print("C — Pass@k regresses > 1σ at some k (Contingency A trigger; surface to the lead author with 3-seed table)")


if __name__ == "__main__":
    main()
