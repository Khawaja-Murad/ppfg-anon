"""Recompute per-run metrics.json using the FIXED detect_final_answer.

Day-8 fixed `chain.py::detect_final_answer` to handle nested-brace LaTeX in
\\boxed{}. This script re-reads each results/<run>/trajectories.json, applies
the fixed extractor to each chain's last (and earlier) steps, recomputes
Pass@k + answer_mode_rate, and writes a new metrics.json. Original metrics
backed up to metrics_pre_fix.json.

Usage:
  PYTHONPATH=src python scripts/recompute_metrics_post_fix.py [glob ...]

Args (optional): one or more glob patterns under results/. Defaults to
'results/*-math500-*' (all math500 runs). Skips runs without trajectories.json.

No GPU compute. Pure re-extraction + arithmetic.
"""

from __future__ import annotations
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# Use the production extractor (now fixed).
from hyp_forest.chains.chain import detect_final_answer


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s


def answers_equivalent(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    if normalize_answer(pred) == normalize_answer(gold):
        return True
    try:
        from math_verify import parse, verify  # type: ignore
        return verify(parse(gold), parse(pred))
    except ImportError:
        from hyp_forest.comparator_guard import warn_if_degraded
        warn_if_degraded()
        return False
    except Exception:
        return False


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def extract_chain_answer(chain: dict) -> str | None:
    """Apply fixed detector to chain's last step, fallback to earlier steps."""
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


def recompute_one(run_dir: Path) -> dict | None:
    traj_path = run_dir / "trajectories.json"
    metrics_path = run_dir / "metrics.json"
    if not traj_path.exists() or not metrics_path.exists():
        return None
    try:
        traj = json.load(open(traj_path))
        old_metrics = json.load(open(metrics_path))
    except Exception as e:
        print(f"  SKIP {run_dir.name}: {e}")
        return None

    n_chains = None
    per_problem_correct: list[int] = []
    per_problem_modal_share: list[float] = []
    for p in traj:
        chains = p.get("population", p).get("chains", [])
        if n_chains is None:
            n_chains = len(chains)
        gold = p.get("answer", p.get("gold_answer"))
        answers = [extract_chain_answer(c) for c in chains]
        # Correctness count
        n_correct = sum(1 for a in answers if answers_equivalent(a, gold))
        per_problem_correct.append(n_correct)
        # Modal share among non-null answers
        non_null = [normalize_answer(a) for a in answers if a]
        if non_null:
            most_common = Counter(non_null).most_common(1)[0]
            per_problem_modal_share.append(most_common[1] / len(non_null))
        else:
            per_problem_modal_share.append(0.0)

    n = n_chains or 8
    k_values = old_metrics.get("pass_at_k", {"1": 0, "2": 0, "4": 0, "8": 0, "16": 0})
    new_pk = {}
    for k_str in k_values.keys():
        k = int(k_str)
        if k > n:
            new_pk[k_str] = float("nan")
        else:
            new_pk[k_str] = sum(pass_at_k_estimator(n, c, k) for c in per_problem_correct) / max(1, len(per_problem_correct))

    # answer_mode_rate (binary def): fraction of problems with modal_share > 0.5
    n_problems = len(per_problem_modal_share)
    new_mode_rate = sum(1 for x in per_problem_modal_share if x > 0.5) / max(1, n_problems)

    new_metrics = dict(old_metrics)
    new_metrics["pass_at_k"] = new_pk
    new_metrics["diversity"] = dict(old_metrics.get("diversity", {}))
    new_metrics["diversity"]["answer_mode_rate"] = new_mode_rate
    new_metrics["_recomputed_post_fix"] = True

    # Backup original then overwrite
    backup_path = run_dir / "metrics_pre_fix.json"
    if not backup_path.exists():
        shutil.copy(metrics_path, backup_path)
    with open(metrics_path, "w") as f:
        json.dump(new_metrics, f, indent=2)

    return {
        "dir": run_dir.name,
        "old_pass_at_1": old_metrics.get("pass_at_k", {}).get("1"),
        "new_pass_at_1": new_pk.get("1"),
        "old_mode_rate": old_metrics.get("diversity", {}).get("answer_mode_rate"),
        "new_mode_rate": new_mode_rate,
        "n_problems": n_problems,
    }


def main():
    patterns = sys.argv[1:] if len(sys.argv) > 1 else ["results/*-math500-*"]
    dirs = []
    for pat in patterns:
        dirs.extend(Path(".").glob(pat))
    dirs = [d for d in dirs if d.is_dir() and "_archive" not in str(d)]
    dirs.sort()
    print(f"Found {len(dirs)} candidate dirs.")
    print(f"{'dir':<60s} {'old@1':>7s} {'new@1':>7s} {'Δ':>7s} {'old_mr':>7s} {'new_mr':>7s}")
    for d in dirs:
        r = recompute_one(d)
        if r is None:
            continue
        old1 = r["old_pass_at_1"]
        new1 = r["new_pass_at_1"]
        delta = (new1 - old1) if (old1 is not None and new1 is not None) else None
        oldmr = r["old_mode_rate"]
        newmr = r["new_mode_rate"]
        delta_s = f"{delta:+.3f}" if delta is not None else "?"
        print(f"{r['dir']:<60s} {old1:>7.4f} {new1:>7.4f} {delta_s:>7s} {oldmr:>7.4f} {newmr:>7.4f}")


if __name__ == "__main__":
    main()
