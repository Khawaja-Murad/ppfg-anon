"""TOST equivalence analysis for PPFG vs independent baseline.

Pre-registered bounds (see results/tost_bounds_specification.md):
  pass_at_1:        +- 0.018
  pass_at_8:        +- 0.031
  answer_mode_rate: +- 0.022

Two one-sided Welch t-tests at alpha = 0.05.  Verdict:
  - both H0 rejected     -> equivalent
  - upper-side rejected  -> one-sided-upper (mean_diff bounded above)
  - lower-side rejected  -> one-sided-lower (mean_diff bounded below)
  - neither              -> inconclusive

Inputs are per-seed metrics.json from results/<run>/.  Cells are looked
up by experiment.name in each run's config.json (canonical cross-arch
tagging set on Day 8/9, see inventory_n500.json).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

RESULTS_DIR = Path("./results")
OUT_JSON = RESULTS_DIR / "tost_results.json"
OUT_MD = RESULTS_DIR / "tost_results.md"

BOUNDS = {
    "pass_at_1": 0.018,
    "pass_at_8": 0.031,
    "answer_mode_rate": 0.022,
}

# Map experiment.name -> (arch, method).  Anything else is ignored.
NAME_MAP = {
    "qwen_n500_indep_math500":         ("qwen",    "indep"),
    "qwen_n500_ppfg_stag_math500":     ("qwen",    "ppfg_stag"),
    "qwen_n500_ppfg_random_math500":   ("qwen",    "ppfg_random"),
    "llama31_n500_indep_math500":      ("llama31", "indep"),
    "llama31_n500_ppfg_stag_math500":  ("llama31", "ppfg_stag"),
    "llama31_n500_ppfg_random_math500":("llama31", "ppfg_random"),
    "dsr1_n500_indep_math500":         ("dsr1",    "indep"),
    "dsr1_n500_ppfg_stag_math500":     ("dsr1",    "ppfg_stag"),
    "dsr1_n500_ppfg_random_math500":   ("dsr1",    "ppfg_random"),
}


def load_cells() -> Dict[Tuple[str, str], Dict[int, dict]]:
    """Return {(arch, method) -> {seed -> metrics_dict}}."""
    cells: Dict[Tuple[str, str], Dict[int, dict]] = defaultdict(dict)
    for cfg_path in glob.glob(str(RESULTS_DIR / "*" / "config.json")):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        name = cfg.get("experiment", {}).get("name", "")
        if name not in NAME_MAP:
            continue
        seed = cfg.get("experiment", {}).get("seed")
        run_dir = Path(cfg_path).parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        # metrics.json may contain literal NaN (json5-ish); parse permissively
        with open(metrics_path) as f:
            txt = f.read()
        try:
            m = json.loads(txt)
        except json.JSONDecodeError:
            # cope with bare NaN tokens from Python json.dump(allow_nan=True)
            m = json.loads(txt.replace("NaN", "null"))
        arch, method = NAME_MAP[name]
        # Some seeds may have replicate runs (failed jobs leaving P@1=0
        # metrics, or deterministic re-runs). Prefer the first valid run
        # (P@1 > 0 indicates the run actually evaluated); fall back to
        # whatever is present.
        p1 = m.get("pass_at_k", {}).get("1", 0.0) or 0.0
        existing = cells[(arch, method)].get(seed)
        if existing is None:
            cells[(arch, method)][seed] = m
        else:
            ex_p1 = existing.get("pass_at_k", {}).get("1", 0.0) or 0.0
            if ex_p1 == 0.0 and p1 > 0.0:
                cells[(arch, method)][seed] = m
    return cells


def extract_metric(metrics: dict, metric: str) -> float | None:
    if metric == "answer_mode_rate":
        v = metrics.get("diversity", {}).get("answer_mode_rate")
    elif metric == "pass_at_1":
        v = metrics.get("pass_at_k", {}).get("1")
    elif metric == "pass_at_8":
        v = metrics.get("pass_at_k", {}).get("8")
    else:
        raise KeyError(metric)
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv):
        return None
    return fv


def welch_one_sided(treatment: np.ndarray, control: np.ndarray, delta: float,
                    side: str) -> Tuple[float, float]:
    """Welch one-sided t-test for the hypothesis on mean(treatment) - mean(control).

    side='less'    -> H1: diff <  +delta   (H0: diff >= +delta)
    side='greater' -> H1: diff >  -delta   (H0: diff <= -delta)

    Returns (t_statistic, one_sided_p_value).
    """
    n1, n2 = len(treatment), len(control)
    m1, m2 = float(np.mean(treatment)), float(np.mean(control))
    # sample variances (unbiased)
    v1 = float(np.var(treatment, ddof=1)) if n1 > 1 else 0.0
    v2 = float(np.var(control,   ddof=1)) if n2 > 1 else 0.0
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0.0:
        # Degenerate: all values identical in both arms.  p = 0 or 1 by sign.
        diff = m1 - m2
        if side == "less":
            return (float("inf") if diff < delta else float("-inf"),
                    0.0 if diff < delta else 1.0)
        else:
            return (float("inf") if diff > -delta else float("-inf"),
                    0.0 if diff > -delta else 1.0)
    se = math.sqrt(se2)
    # df via Welch-Satterthwaite
    if n1 > 1 and n2 > 1:
        num = se2 ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else (n1 + n2 - 2)
    else:
        df = max(n1 + n2 - 2, 1)
    if side == "less":
        t = (m1 - m2 - delta) / se
        p = float(stats.t.cdf(t, df))
    elif side == "greater":
        t = (m1 - m2 + delta) / se
        p = float(stats.t.sf(t, df))
    else:
        raise ValueError(side)
    return float(t), p


def tost(treatment: np.ndarray, control: np.ndarray, bound: float) -> dict:
    # `bound` is the magnitude of the equivalence margin (>0).  `welch_one_sided`
    # already encodes the sign:  side='less' tests H0: diff >= +delta,
    # side='greater' tests H0: diff <= -delta.  Pass the positive magnitude to
    # both sides; passing -bound to the 'greater' side double-negates the sign
    # and silently flips the test direction (this was a bug in the v11 code).
    t_upper, p_upper = welch_one_sided(treatment, control, bound, "less")
    t_lower, p_lower = welch_one_sided(treatment, control, bound, "greater")
    if p_upper < 0.05 and p_lower < 0.05:
        verdict = "equivalent"
    elif p_upper < 0.05:
        verdict = "one-sided-upper"
    elif p_lower < 0.05:
        verdict = "one-sided-lower"
    else:
        verdict = "inconclusive"
    return dict(
        n_treatment=int(len(treatment)),
        n_control=int(len(control)),
        mean_treatment=float(np.mean(treatment)),
        mean_control=float(np.mean(control)),
        mean_diff=float(np.mean(treatment) - np.mean(control)),
        sd_treatment=float(np.std(treatment, ddof=1)) if len(treatment) > 1 else 0.0,
        sd_control=float(np.std(control, ddof=1)) if len(control) > 1 else 0.0,
        bound=float(bound),
        t_upper=float(t_upper),
        p_upper=float(p_upper),
        t_lower=float(t_lower),
        p_lower=float(p_lower),
        verdict=verdict,
    )


def _filter_seeds(cell: Dict[int, dict], arch: str,
                  per_arch_seeds: Dict[str, Optional[Set[int]]]) -> Dict[int, dict]:
    """Restrict cell to the seeds allowed for this architecture."""
    allowed = per_arch_seeds.get(arch)
    if allowed is None:
        return cell
    return {s: v for s, v in cell.items() if s in allowed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=None,
                        help=("Comma-separated seed list to use for ALL archs, "
                              "OR per-arch override of the form "
                              "'qwen=42,43,44,45,46;llama31=42,43,44'. "
                              "If omitted, every discovered seed is used."))
    parser.add_argument("--out-json", type=str, default=str(OUT_JSON),
                        help="Output JSON path (default tost_results.json).")
    parser.add_argument("--out-md", type=str, default=str(OUT_MD),
                        help="Output Markdown path (default tost_results.md).")
    parser.add_argument("--label", type=str, default="n=3 seeds per arm",
                        help="Header label describing the seed configuration.")
    args = parser.parse_args()

    archs = ["qwen", "llama31", "dsr1"]
    per_arch_seeds: Dict[str, Optional[Set[int]]] = {a: None for a in archs}
    if args.seeds:
        if "=" in args.seeds:
            for chunk in args.seeds.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                arch, seeds_csv = chunk.split("=", 1)
                arch = arch.strip()
                seeds = {int(s) for s in seeds_csv.split(",") if s.strip()}
                per_arch_seeds[arch] = seeds
        else:
            seeds = {int(s) for s in args.seeds.split(",") if s.strip()}
            for a in archs:
                per_arch_seeds[a] = seeds

    cells = load_cells()
    print(f"Loaded {len(cells)} cells (raw)")
    for k, v in sorted(cells.items()):
        print(f"  {k}: seeds={sorted(v.keys())}")
    if any(per_arch_seeds[a] is not None for a in archs):
        print("Per-arch seed filter:")
        for a in archs:
            print(f"  {a}: {sorted(per_arch_seeds[a]) if per_arch_seeds[a] else 'ALL'}")

    treatments = ["ppfg_stag", "ppfg_random"]
    metrics = ["pass_at_1", "pass_at_8", "answer_mode_rate"]

    rows = []
    for arch in archs:
        ctrl_cell = _filter_seeds(cells.get((arch, "indep"), {}), arch, per_arch_seeds)
        for tx in treatments:
            tx_cell = _filter_seeds(cells.get((arch, tx), {}), arch, per_arch_seeds)
            for metric in metrics:
                ctrl_vals = [extract_metric(ctrl_cell[s], metric) for s in sorted(ctrl_cell)]
                ctrl_vals = [x for x in ctrl_vals if x is not None]
                tx_vals = [extract_metric(tx_cell[s], metric) for s in sorted(tx_cell)]
                tx_vals = [x for x in tx_vals if x is not None]
                if len(ctrl_vals) < 2 or len(tx_vals) < 2:
                    rows.append(dict(
                        method=tx, arch=arch, metric=metric,
                        bound=BOUNDS[metric],
                        n_treatment=len(tx_vals), n_control=len(ctrl_vals),
                        verdict="data-incomplete",
                        mean_treatment=float(np.mean(tx_vals)) if tx_vals else None,
                        mean_control=float(np.mean(ctrl_vals)) if ctrl_vals else None,
                        mean_diff=(float(np.mean(tx_vals)) - float(np.mean(ctrl_vals)))
                                  if tx_vals and ctrl_vals else None,
                    ))
                    continue
                r = tost(np.array(tx_vals), np.array(ctrl_vals), BOUNDS[metric])
                r["method"] = tx
                r["arch"] = arch
                r["metric"] = metric
                rows.append(r)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    with open(out_json, "w") as f:
        json.dump(dict(bounds=BOUNDS, rows=rows,
                       seed_filter={a: sorted(per_arch_seeds[a]) if per_arch_seeds[a] else None
                                    for a in archs}),
                  f, indent=2)

    md = ["# TOST Equivalence Results\n",
          "Pre-registered bounds (results/tost_bounds_specification.md):",
          f"- pass_at_1: +-{BOUNDS['pass_at_1']}",
          f"- pass_at_8: +-{BOUNDS['pass_at_8']}",
          f"- answer_mode_rate: +-{BOUNDS['answer_mode_rate']}",
          "",
          f"Welch two-sample one-sided t-tests, alpha = 0.05. {args.label}.",
          ""]
    if any(per_arch_seeds[a] is not None for a in archs):
        md.append("Seed configuration (per architecture):")
        for a in archs:
            sds = sorted(per_arch_seeds[a]) if per_arch_seeds[a] else "ALL"
            md.append(f"- {a}: {sds}")
        md.append("")
    md.extend([
          "| method | arch | metric | mean_tx | mean_ctrl | mean_diff | bound | p_upper | p_lower | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"])
    for r in rows:
        if r["verdict"] == "data-incomplete":
            md.append(f"| {r['method']} | {r['arch']} | {r['metric']} | "
                      f"{r.get('mean_treatment','-')} | {r.get('mean_control','-')} | "
                      f"{r.get('mean_diff','-')} | {r['bound']} | - | - | "
                      f"data-incomplete (n_tx={r['n_treatment']}, n_ctrl={r['n_control']}) |")
            continue
        md.append(
            f"| {r['method']} | {r['arch']} | {r['metric']} | "
            f"{r['mean_treatment']:.4f} | {r['mean_control']:.4f} | "
            f"{r['mean_diff']:+.4f} | {r['bound']:.3f} | "
            f"{r['p_upper']:.4f} | {r['p_lower']:.4f} | {r['verdict']} |"
        )
    md.append("")
    md.append("Reading guide: 'equivalent' means we statistically rule out a")
    md.append("difference outside +-bound at alpha=0.05 on both sides; ")
    md.append("'one-sided-X' means we rule out only one side; ")
    md.append("'inconclusive' means we cannot reject at this n / variance.")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")

    # Summary to stdout
    print("\n=== TOST results ===")
    for r in rows:
        verdict = r["verdict"]
        if verdict == "data-incomplete":
            print(f"  {r['method']:12s} {r['arch']:7s} {r['metric']:16s}  {verdict}")
            continue
        print(f"  {r['method']:12s} {r['arch']:7s} {r['metric']:16s}  "
              f"diff={r['mean_diff']:+.4f}  p_up={r['p_upper']:.3f}  "
              f"p_lo={r['p_lower']:.3f}  -> {verdict}")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
