"""Recompute Table 1 at n=500 with all 6 methods.

Methods:
  - indep (N=8)
  - smc (N=8)
  - bon_n16 (N=16, 2× compute)
  - bon_n8 (DERIVED from indep N=8 by argmax(final_step_PRM) per problem per seed)
  - ppfg_stag (N=8)
  - ppfg_random (N=8)

Each cell's metrics.json was already recomputed Day-9 with the fixed extractor;
this script aggregates across 3 seeds for Pass@1/2/4/8/16, mode_rate, distinct_4,
tokens/prob.

For BoN-N=8 we derive from indep trajectories using argmax(final-step PRM).
"""

from __future__ import annotations
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


def normalize_answer(s):
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"^\$+", "", s)
    s = re.sub(r"\$+$", "", s)
    s = s.strip()
    s = s.replace(r"\dfrac", r"\frac").replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s if s else None


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


def load_metrics(d):
    return json.load(open(f"{d}/metrics.json"))


def aggregate_native(dirs):
    rows = [load_metrics(d) for d in dirs]
    pk_keys = ["1", "2", "4", "8", "16"]
    out = {}
    for k in pk_keys:
        vals = [r["pass_at_k"].get(k) for r in rows]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals:
            out[f"pass_at_{k}_mean"] = statistics.mean(vals)
            out[f"pass_at_{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        else:
            out[f"pass_at_{k}_mean"] = None
            out[f"pass_at_{k}_std"] = None
    for fld, src_path in [
        ("mode_rate", lambda r: r["diversity"]["answer_mode_rate"]),
        ("distinct_4", lambda r: r["diversity"].get("distinct_4")),
        ("tpp", lambda r: r["tokens_per_problem"]),
    ]:
        vals = [src_path(r) for r in rows if src_path(r) is not None]
        if vals:
            out[f"{fld}_mean"] = statistics.mean(vals)
            out[f"{fld}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        else:
            out[f"{fld}_mean"] = None
            out[f"{fld}_std"] = None
    return out


def derive_bon_n8_from_indep(indep_dirs):
    """Derive BoN-N=8 matched-compute metrics from indep N=8 trajectories.

    Convention (matches Day-6 native BoN-N=8 cells, j60944109/110/111):
      - Pass@1 = argmax(final-step PRM) chain is correct (BoN-specific Pass@1)
      - Pass@k for k≥2 = Chen-2021 estimator over the full N=8 population (same as indep)
      - mode_rate, distinct_4, tpp = from full N=8 population (same as indep)

    Re-extracts final_answer from chain steps using the fixed
    chain.py::detect_final_answer (saved final_answer fields in trajectories
    are from the buggy era pre Day-9 fix).
    """
    from hyp_forest.chains.chain import detect_final_answer

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

    per_seed_metrics = []
    for d in indep_dirs:
        t = json.load(open(f"{d}/trajectories.json"))
        n_problems = len(t)
        n_chains = None
        per_problem_correct_argmax = 0  # BoN Pass@1
        per_problem_correct_counts = []  # for Chen estimator at k≥2
        per_problem_modal_share = []
        tokens_per_problem = []
        # distinct_4 across all chains in run
        all_4grams_unique = set()
        all_4grams_count = 0
        for p in t:
            chains = p.get("population", p).get("chains", [])
            if n_chains is None:
                n_chains = len(chains)
            gold = p.get("answer", p.get("gold_answer"))
            # Re-extract answers with fixed extractor
            answers = [extract(c) for c in chains]
            corrects = [answers_equivalent(a, gold) for a in answers]
            # BoN Pass@1 = argmax-PRM chain correct?
            best_idx = max(
                range(len(chains)),
                key=lambda i: chains[i]["prm_scores"][-1] if chains[i].get("prm_scores") else -1.0,
            )
            if corrects[best_idx]:
                per_problem_correct_argmax += 1
            # Full-population correct count for Chen Pass@k≥2
            per_problem_correct_counts.append(sum(corrects))
            # mode_rate / tpp / distinct_4 over full population
            nn = [normalize_answer(a) for a in answers if a]
            if nn:
                modal = Counter(nn).most_common(1)[0][1] / len(nn)
            else:
                modal = 0.0
            per_problem_modal_share.append(modal)
            tokens_per_problem.append(sum(c.get("n_tokens_generated", 0) for c in chains))
            full_text = " ".join(step for c in chains for step in c.get("steps", []))
            words = full_text.split()
            if len(words) >= 4:
                grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
                all_4grams_unique.update(grams)
                all_4grams_count += len(grams)
        N = n_chains or 8
        pass_at_1 = per_problem_correct_argmax / n_problems
        pass_at_k = {}
        for k in [2, 4, 8]:
            if k > N:
                pass_at_k[k] = None
                continue
            pass_at_k[k] = sum(pass_at_k_estimator(N, c, k) for c in per_problem_correct_counts) / n_problems
        binary_mode = sum(1 for x in per_problem_modal_share if x > 0.5) / n_problems
        d4 = len(all_4grams_unique) / all_4grams_count if all_4grams_count else 0.0
        tpp = statistics.mean(tokens_per_problem) if tokens_per_problem else 0.0
        per_seed_metrics.append({
            "pass_at_1": pass_at_1,
            "pass_at_2": pass_at_k.get(2),
            "pass_at_4": pass_at_k.get(4),
            "pass_at_8": pass_at_k.get(8),
            "mode_rate": binary_mode,
            "distinct_4": d4,
            "tpp": tpp,
        })

    def m_s(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    out = {}
    for fld in ["pass_at_1", "pass_at_2", "pass_at_4", "pass_at_8", "mode_rate", "distinct_4", "tpp"]:
        mean, std = m_s([m[fld] for m in per_seed_metrics])
        out[f"{fld}_mean"], out[f"{fld}_std"] = mean, std
    out["pass_at_16_mean"] = None
    out["pass_at_16_std"] = None
    return out


CELLS = {
    "indep_n500": [
        "results/independent-math500-seed42",
        "results/independent-math500-seed43",
        "results/independent-math500-seed44",
    ],
    "smc_n500": [
        "results/smc-math500-seed42",
        "results/smc-math500-seed43",
        "results/smc-math500-seed44",
    ],
    "bon_n500_N16": [
        "results/best_of_n-math500-seed42",
        "results/best_of_n-math500-seed43",
        "results/best_of_n-math500-seed44",
    ],
    "ppfg_stag_n500": [
        "results/ppfg-math500-20260514-161757-j60966183",
        "results/ppfg-math500-20260514-161757-j60966186",
        "results/ppfg-math500-20260514-161757-j60966195",
    ],
    "ppfg_random_n500": [
        "results/ppfg-math500-20260514-161757-j60966200",
        "results/ppfg-math500-20260514-161757-j60966204",
        "results/ppfg-math500-20260514-161757-j60966213",
    ],
}


def main():
    summary = {}
    for cell, dirs in CELLS.items():
        summary[cell] = aggregate_native(dirs)
    # Derived BoN-N=8
    summary["bon_n500_N8_derived"] = derive_bon_n8_from_indep(CELLS["indep_n500"])

    out_path = Path("results/table1_n500.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown rendering
    md_lines = [
        "**Table 1 (n=500): Pass@k, `answer_mode_rate`, `distinct_4` on full MATH500, mean ± std over 3 seeds.**",
        "",
        "| Method | Pass@1 | Pass@2 | Pass@4 | Pass@8 | Pass@16 | mode_rate ↓ | distinct_4 ↑ | tpp |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    method_order = ["indep_n500", "smc_n500", "bon_n500_N16", "bon_n500_N8_derived", "ppfg_stag_n500", "ppfg_random_n500"]
    display = {
        "indep_n500": "independent (N=8)",
        "smc_n500": "smc (N=8)",
        "bon_n500_N16": "best-of-N (N=16, 2× compute)",
        "bon_n500_N8_derived": "best-of-N (N=8, matched compute; derived)",
        "ppfg_stag_n500": "**PPFG-stag (N=8)**",
        "ppfg_random_n500": "**PPFG-random (N=8)**",
    }

    def fmt(mean, std):
        if mean is None:
            return "—"
        if std is None or std == 0:
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"

    for m in method_order:
        s = summary[m]
        row = (
            f"| {display[m]} | "
            f"{fmt(s.get('pass_at_1_mean'), s.get('pass_at_1_std'))} | "
            f"{fmt(s.get('pass_at_2_mean'), s.get('pass_at_2_std'))} | "
            f"{fmt(s.get('pass_at_4_mean'), s.get('pass_at_4_std'))} | "
            f"{fmt(s.get('pass_at_8_mean'), s.get('pass_at_8_std'))} | "
            f"{fmt(s.get('pass_at_16_mean'), s.get('pass_at_16_std'))} | "
            f"{fmt(s.get('mode_rate_mean'), s.get('mode_rate_std'))} | "
            f"{fmt(s.get('distinct_4_mean'), s.get('distinct_4_std'))} | "
            f"{s.get('tpp_mean', 0):.0f} |"
        )
        md_lines.append(row)

    md_path = Path("results/table1_n500.md")
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")
    print()
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
