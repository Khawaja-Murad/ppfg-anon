#!/usr/bin/env python3
"""
Aggregate the xi_min sweep (Phase 1) into a single table.

5 rows:
  - xi=0.00 (anchor PPFG-stag, seeds 42/43/44)
  - xi=0.10, 0.30, 0.50, 0.70 (3 seeds each, Phase 1 jobs 61020468-61020479)

Writes:
  ./results/table_xi_min_sweep.json
  ./results/table_xi_min_sweep.md
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

RESULTS_DIR = Path("./results")
OUT_JSON = RESULTS_DIR / "table_xi_min_sweep.json"
OUT_MD = RESULTS_DIR / "table_xi_min_sweep.md"

# Cell layout: (xi_min, [run_dirs])
CELLS = [
    (
        0.00,
        [
            "ppfg-math500-20260514-161757-j60966183",
            "ppfg-math500-20260514-161757-j60966186",
            "ppfg-math500-20260514-161757-j60966195",
        ],
    ),
    (
        0.10,
        [
            "ppfg-math500-20260515-150727-j61020468",
            "ppfg-math500-20260515-151143-j61020469",
            "ppfg-math500-20260515-151143-j61020470",
        ],
    ),
    (
        0.30,
        [
            "ppfg-math500-20260515-151143-j61020471",
            "ppfg-math500-20260515-151143-j61020472",
            "ppfg-math500-20260515-151552-j61020473",
        ],
    ),
    (
        0.50,
        [
            "ppfg-math500-20260515-151552-j61020474",
            "ppfg-math500-20260515-151552-j61020475",
            "ppfg-math500-20260515-151552-j61020476",
        ],
    ),
    (
        0.70,
        [
            "ppfg-math500-20260515-151552-j61020477",
            "ppfg-math500-20260515-151552-j61020478",
            "ppfg-math500-20260515-151552-j61020479",
        ],
    ),
]


def _stat(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    mean = statistics.mean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return mean, sd


def _inj_rate(run_dir: Path) -> float:
    """Fraction of problems where at least one chain has a non-empty injected_fragments list."""
    traj_path = run_dir / "trajectories.json"
    trajs = json.load(open(traj_path))
    n = len(trajs)
    if n == 0:
        return float("nan")
    n_inj = 0
    for p in trajs:
        chains = p["population"]["chains"]
        if any(len(c.get("injected_fragments", [])) > 0 for c in chains):
            n_inj += 1
    return n_inj / n


def _load_cell(xi: float, dirs: list[str]) -> dict:
    p1, p4, p8, mr, d4, tpp, inj = [], [], [], [], [], [], []
    for d in dirs:
        run_dir = RESULTS_DIR / d
        m = json.load(open(run_dir / "metrics.json"))
        pak = m["pass_at_k"]
        # pass_at_k keys may be str or int
        def _g(k):
            return pak.get(str(k), pak.get(k))
        p1.append(_g(1))
        p4.append(_g(4))
        p8.append(_g(8))
        div = m.get("diversity", {})
        mr.append(div.get("answer_mode_rate"))
        d4.append(div.get("distinct_4"))
        tpp.append(m.get("tokens_per_problem"))
        inj.append(_inj_rate(run_dir))

    row = {
        "xi_min": xi,
        "n_seeds": len(dirs),
        "run_dirs": dirs,
        "pass_at_1": _stat(p1),
        "pass_at_4": _stat(p4),
        "pass_at_8": _stat(p8),
        "answer_mode_rate": _stat(mr),
        "distinct_4": _stat(d4),
        "tokens_per_problem": _stat(tpp),
        "inj_rate": _stat(inj),
        "per_seed": {
            "pass_at_1": p1,
            "pass_at_4": p4,
            "pass_at_8": p8,
            "answer_mode_rate": mr,
            "distinct_4": d4,
            "tokens_per_problem": tpp,
            "inj_rate": inj,
        },
    }
    return row


def _fmt_pct(mean: float, sd: float, decimals: int = 1) -> str:
    if math.isnan(mean):
        return "n/a"
    return f"{100*mean:.{decimals}f}±{100*sd:.{decimals}f}"


def _fmt_int(mean: float, sd: float) -> str:
    if math.isnan(mean):
        return "n/a"
    return f"{round(mean):d}±{round(sd):d}"


def main():
    rows = [_load_cell(xi, dirs) for xi, dirs in CELLS]
    out = {
        "_meta": {
            "task": "math500",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prm": "peiyi9979/math-shepherd-mistral-7b-prm",
            "method": "ppfg",
            "injection_rule": "stagnation",
            "n_problems": 500,
            "n_chains": 8,
            "seeds": [42, 43, 44],
            "note": "Phase 1 xi_min sweep + anchor at xi_min=0.0. inj_rate = fraction of problems with >=1 non-empty injected_fragments list across the 8 chains.",
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # Markdown
    lines = []
    lines.append("# xi_min sweep (Phase 1)\n")
    lines.append("Qwen2.5-7B + Math-Shepherd PRM + MATH500 (n=500), PPFG-stag, 3 seeds per cell.\n")
    lines.append("| xi_min | inj_rate | Pass@1 | Pass@4 | Pass@8 | mode_rate | distinct_4 | tpp |")
    lines.append("|--------|----------|--------|--------|--------|-----------|-----------|-----|")
    for r in rows:
        lines.append(
            "| {xi:.2f} | {inj} | {p1} | {p4} | {p8} | {mr} | {d4} | {tpp} |".format(
                xi=r["xi_min"],
                inj=_fmt_pct(*r["inj_rate"]),
                p1=_fmt_pct(*r["pass_at_1"]),
                p4=_fmt_pct(*r["pass_at_4"]),
                p8=_fmt_pct(*r["pass_at_8"]),
                mr=_fmt_pct(*r["answer_mode_rate"]),
                d4=_fmt_pct(*r["distinct_4"]),
                tpp=_fmt_int(*r["tokens_per_problem"]),
            )
        )
    OUT_MD.write_text("\n".join(lines) + "\n")

    # Echo to stdout
    print("\n".join(lines))
    print(f"\nWrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
