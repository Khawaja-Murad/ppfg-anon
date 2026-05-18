"""Mechanistic spot-check: per-event post-injection PRM minus pre-injection PRM.

For each PPFG-stag injection event across the 3 n=500 Qwen seeds, compute:
  delta = mean(prm_scores[inject_at : inject_at+3]) - mean(prm_scores[inject_at-3 : inject_at])

The target chain's `injected_at_step` is the chain's step count at injection,
so chain steps with index >= injected_at_step are "post-injection" (decoded
under the influence of the spliced fragment) and steps with index <
injected_at_step are "pre-injection".

Reports aggregate statistics, Wilcoxon signed-rank test (H0: median delta = 0),
and histogram bins, written to results/injection_delta_prm.json. Also generates
paper/figures/fig_e_injection_delta.pdf.

Run from repo root (cwd = .):
    PYTHONPATH=src python scripts/compute_injection_prm_delta.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PPFG_STAG_DIRS = [
    "results/ppfg-math500-20260514-161757-j60966183",
    "results/ppfg-math500-20260514-161757-j60966186",
    "results/ppfg-math500-20260514-161757-j60966195",
]

PRE_WINDOW = 3
POST_WINDOW = 3


def collect_deltas(dir_path: str):
    """Return list of (delta, pre_mean, post_mean, inject_at, n_steps) per event."""
    t = json.load(open(f"{dir_path}/trajectories.json"))
    out = []
    skipped_no_pre = 0
    skipped_no_post = 0
    for prob in t:
        pop = prob.get("population", prob)
        for chain in pop.get("chains", []):
            prm = chain.get("prm_scores", []) or []
            for inj in chain.get("injected_fragments", []) or []:
                k = int(inj["injected_at_step"])
                # pre: scores at indices [k-PRE_WINDOW, k)
                pre_lo = max(0, k - PRE_WINDOW)
                pre = prm[pre_lo:k]
                # post: scores at indices [k, k+POST_WINDOW)
                post = prm[k:k + POST_WINDOW]
                if len(pre) < 1:
                    skipped_no_pre += 1
                    continue
                if len(post) < 1:
                    skipped_no_post += 1
                    continue
                pre_mean = float(np.mean(pre))
                post_mean = float(np.mean(post))
                out.append({
                    "delta": post_mean - pre_mean,
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "pre_window_len": len(pre),
                    "post_window_len": len(post),
                    "injected_at_step": k,
                    "chain_n_steps": len(prm),
                    "problem_id": prob.get("problem_id"),
                    "chain_id": chain.get("chain_id"),
                })
    return out, skipped_no_pre, skipped_no_post


def main():
    repo_root = Path(__file__).resolve().parent.parent
    all_events = []
    total_skipped_pre = 0
    total_skipped_post = 0
    for d in PPFG_STAG_DIRS:
        evs, sp, sq = collect_deltas(str(repo_root / d))
        all_events.extend(evs)
        total_skipped_pre += sp
        total_skipped_post += sq

    deltas = np.array([e["delta"] for e in all_events], dtype=float)
    n = len(deltas)
    n_pos = int((deltas > 0).sum())
    n_neg = int((deltas < 0).sum())
    n_zero = int((deltas == 0).sum())

    mean_d = float(np.mean(deltas)) if n else float("nan")
    median_d = float(np.median(deltas)) if n else float("nan")
    std_d = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    q1 = float(np.percentile(deltas, 25)) if n else float("nan")
    q3 = float(np.percentile(deltas, 75)) if n else float("nan")
    iqr = q3 - q1

    # Wilcoxon signed-rank test (drop zeros via 'wilcox' method or use default).
    if n >= 1 and not np.all(deltas == 0):
        try:
            w_stat, p_val = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w_stat)
            p_val = float(p_val)
        except ValueError as e:
            w_stat = None
            p_val = None
            wilcox_err = str(e)
        else:
            wilcox_err = None
    else:
        w_stat = None
        p_val = None
        wilcox_err = "all-zero or empty"

    # Histogram bins: width 0.05 in [-1, 1]
    bin_edges = np.arange(-1.0, 1.0001, 0.05)
    hist, _ = np.histogram(deltas, bins=bin_edges)

    out = {
        "source_dirs": PPFG_STAG_DIRS,
        "pre_window": PRE_WINDOW,
        "post_window": POST_WINDOW,
        "n_events_analyzed": n,
        "n_skipped_no_pre_window": total_skipped_pre,
        "n_skipped_no_post_window": total_skipped_post,
        "delta_mean": mean_d,
        "delta_median": median_d,
        "delta_std": std_d,
        "delta_q1": q1,
        "delta_q3": q3,
        "delta_iqr": iqr,
        "delta_min": float(np.min(deltas)) if n else None,
        "delta_max": float(np.max(deltas)) if n else None,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "frac_positive": n_pos / n if n else None,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": p_val,
        "wilcoxon_error": wilcox_err,
        "histogram_bin_edges": bin_edges.tolist(),
        "histogram_counts": hist.tolist(),
    }

    out_path = repo_root / "results" / "injection_delta_prm.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

    # --- Figure ---
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.hist(deltas, bins=bin_edges, color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(mean_d, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"mean = {mean_d:+.3f}")
    ax.axvline(median_d, color="#55A868", linestyle="-.", linewidth=1.2,
               label=f"median = {median_d:+.3f}")
    ax.set_xlabel("Post-injection PRM − Pre-injection 3-step mean PRM")
    ax.set_ylabel("Count")
    ax.set_xlim(-1.0, 1.0)
    p_str = f"{p_val:.3f}" if p_val is not None else "n/a"
    ax.text(0.02, 0.97,
            f"n={n} events; Wilcoxon p={p_str}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()

    fig_path = repo_root.parent / "paper" / "figures" / "fig_e_injection_delta.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, format="pdf")
    print(f"Wrote {fig_path}")

    # --- Stdout summary ---
    print()
    print(f"=== PPFG-stag injection ΔPRM (n=500, 3 seeds) ===")
    print(f"  events analyzed:    {n}")
    print(f"  skipped (no pre):   {total_skipped_pre}")
    print(f"  skipped (no post):  {total_skipped_post}")
    print(f"  mean   Δ:           {mean_d:+.4f}")
    print(f"  median Δ:           {median_d:+.4f}")
    print(f"  std    Δ:           {std_d:.4f}")
    print(f"  IQR    Δ:           [{q1:+.4f}, {q3:+.4f}] (width {iqr:.4f})")
    print(f"  positive:           {n_pos} ({100*n_pos/max(n,1):.1f}%)")
    print(f"  negative:           {n_neg} ({100*n_neg/max(n,1):.1f}%)")
    print(f"  zero:               {n_zero}")
    if p_val is not None:
        print(f"  Wilcoxon stat:      {w_stat:.2f}")
        print(f"  Wilcoxon p:         {p_val:.4f}")
        verdict = ("significantly shifted" if p_val < 0.05 else "centered on zero")
        print(f"  verdict (α=0.05):    {verdict}")
    else:
        print(f"  Wilcoxon:           {wilcox_err}")


if __name__ == "__main__":
    main()
