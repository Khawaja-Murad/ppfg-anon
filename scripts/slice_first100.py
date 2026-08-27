"""Re-extract Pass@k for first-100 problems from a results dir using fixed extractor."""
import json
import math
import re
import sys
from collections import Counter

from hyp_forest.chains.chain import detect_final_answer


def normalize_answer(s):
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"^\$+", "", s)
    s = re.sub(r"\$+$", "", s)
    s = s.strip()
    s = s.replace(r"\dfrac", r"\frac").replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s


def answers_equivalent(pred, gold):
    if pred is None or gold is None:
        return False
    if normalize_answer(pred) == normalize_answer(gold):
        return True
    try:
        from math_verify import parse, verify
        return verify(parse(gold), parse(pred))
    except Exception:
        from hyp_forest.comparator_guard import warn_if_degraded
        warn_if_degraded()
        return False


def pass_at_k_estimator(n, c, k):
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def extract(chain):
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


def first_n(dir, n_problems=100):
    traj = json.load(open(f"results/{dir}/trajectories.json"))
    traj = traj[:n_problems]
    per_correct, per_mode = [], []
    n_chains = None
    for p in traj:
        chains = p.get("population", p).get("chains", [])
        if n_chains is None:
            n_chains = len(chains)
        gold = p.get("answer", p.get("gold_answer"))
        answers = [extract(c) for c in chains]
        per_correct.append(sum(1 for a in answers if answers_equivalent(a, gold)))
        nn = [normalize_answer(a) for a in answers if a]
        per_mode.append(Counter(nn).most_common(1)[0][1] / len(nn) if nn else 0)
    n = n_chains or 8
    k_values = [1, 2, 4, 8]
    if n >= 16:
        k_values.append(16)
    pk = {k: sum(pass_at_k_estimator(n, c, k) for c in per_correct) / max(1, len(per_correct)) for k in k_values}
    mr = sum(1 for x in per_mode if x > 0.5) / max(1, len(per_mode))
    return pk, mr


if __name__ == "__main__":
    dirs = sys.argv[1:]
    for d in dirs:
        pk, mr = first_n(d)
        cols = " ".join(f"Pass@{k}={pk[k]:.4f}" for k in pk)
        print(f"{d[-15:]}: {cols} mode_rate={mr:.4f}")
