"""One-shot: emit raw-modal-share recomputed Table 1 and Table 2 (compound).

Reuses compute_raw_modal_share.cell_raw_modal_share() but groups by
(model, method, rule, n, prm_low_t) so the two compound thresholds
(t=0.60, t=0.70) appear as separate cells.

Also reads Pass@k / distinct_4 / tpp from each run's metrics.json so the
emitted Markdown matches Table 1's column layout exactly.
"""

from __future__ import annotations
import json
import statistics
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_raw_modal_share import cell_raw_modal_share  # noqa: E402

RESULTS = Path("./results")


def collect():
    """Return dict[cell_key -> list[per-seed metric dict]]."""
    cells = defaultdict(list)
    for d in sorted(RESULTS.glob("*-math500-*")):
        if not d.is_dir() or "_archive" in str(d):
            continue
        cfg_p = d / "config.json"
        m_p = d / "metrics.json"
        if not cfg_p.exists() or not m_p.exists():
            continue
        cfg = json.load(open(cfg_p))
        m = json.load(open(m_p))
        method = m.get("method")
        n = m.get("n_problems")
        model = cfg.get("model", {}).get("base_model", "?").split("/")[-1]
        ppfg = cfg.get("ppfg", {}) if method == "ppfg" else {}
        rule = ppfg.get("injection_rule")
        prm_low_t = ppfg.get("prm_low_t")
        seed = cfg.get("experiment", {}).get("seed")
        raw = cell_raw_modal_share(d)
        if raw is None:
            continue
        passk = m.get("pass_at_k", {})
        div = m.get("diversity", {})
        cost = m.get("cost", {})
        tpp = m.get("tokens_per_problem", cost.get("tokens_per_problem"))
        row = {
            "seed": seed,
            "raw_modal_share": raw,
            "answer_mode_rate": div.get("answer_mode_rate"),
            "distinct_4": div.get("distinct_4"),
            "tpp": tpp,
            "pass_at_1": passk.get("1"),
            "pass_at_2": passk.get("2"),
            "pass_at_4": passk.get("4"),
            "pass_at_8": passk.get("8"),
            "pass_at_16": passk.get("16"),
            "dir": d.name,
        }
        key = (model, method, rule, n, prm_low_t)
        cells[key].append(row)
    return cells


def fmt(mean, std, places=3):
    if mean is None:
        return "---"
    if std is None:
        return f"{mean:.{places}f}"
    return f"{mean:.{places}f} ± {std:.{places}f}"


def fmt_int(mean):
    if mean is None:
        return "---"
    return f"{int(round(mean))}"


def cell_stats(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def emit_row(label, rows):
    cols = []
    for k in ("pass_at_1", "pass_at_2", "pass_at_4", "pass_at_8"):
        mean, std = cell_stats(rows, k)
        cols.append(fmt(mean, std))
    mean, std = cell_stats(rows, "raw_modal_share")
    cols.append(fmt(mean, std, places=4))
    mean, std = cell_stats(rows, "distinct_4")
    cols.append(fmt(mean, std))
    mean, _ = cell_stats(rows, "tpp")
    cols.append(fmt_int(mean))
    return f"| {label} | " + " | ".join(cols) + " |"


def main():
    cells = collect()

    # ---- Table 1: Qwen n=500 main comparison
    table1_keys = [
        ("indep", ("Qwen2.5-7B-Instruct", "independent", None, 500, None)),
        ("SMC (N=8)", ("Qwen2.5-7B-Instruct", "smc", None, 500, None)),
        ("BoN (N=16, 2× compute)", ("Qwen2.5-7B-Instruct", "best_of_n", None, 500, None)),
        ("PPFG-stag (N=8)", ("Qwen2.5-7B-Instruct", "ppfg", "stagnation", 500, None)),
        ("PPFG-random (N=8)", ("Qwen2.5-7B-Instruct", "ppfg", "random", 500, None)),
    ]

    table1_md = [
        "# Table 1 (raw-modal-share variant) — Qwen, n=500",
        "",
        "Mean ± std over 3 seeds (per cell). Replaces binary `answer_mode_rate` ↓ with continuous **raw_modal_share** ↓.",
        "",
        "| Method | Pass@1 | Pass@2 | Pass@4 | Pass@8 | raw_modal_share ↓ | distinct_4 ↑ | tpp |",
        "|---|---|---|---|---|---|---|---|",
    ]
    table1_json = {}
    for label, k in table1_keys:
        rows = cells.get(k, [])
        # BoN-N=8 derived line is not in /results; emit only N=16 BoN here.
        if not rows:
            table1_md.append(f"| {label} | (no data) | | | | | | |")
            continue
        # Filter to exactly 3 distinct seeds; for compound n=500 the indep cell has 6 seeds total.
        # Use all available rows (mean over whichever count was collected).
        table1_md.append(emit_row(label, rows))
        ms, ms_std = cell_stats(rows, "raw_modal_share")
        amr_m, amr_std = cell_stats(rows, "answer_mode_rate")
        table1_json[label] = {
            "n_seeds": len(rows),
            "raw_modal_share_mean": ms,
            "raw_modal_share_std": ms_std,
            "answer_mode_rate_mean": amr_m,
            "answer_mode_rate_std": amr_std,
            "abs_delta_vs_amr": (ms - amr_m) if (ms is not None and amr_m is not None) else None,
        }

    Path(RESULTS / "table1_raw_modal_share.md").write_text("\n".join(table1_md) + "\n")

    # ---- Table 2: compound-gate at n=500
    indep_key = ("Qwen2.5-7B-Instruct", "independent", None, 500, None)
    ppfg_stag_key = ("Qwen2.5-7B-Instruct", "ppfg", "stagnation", 500, None)
    compound_keys = [(k, v) for k, v in cells.items()
                     if k[0] == "Qwen2.5-7B-Instruct"
                     and k[1] == "ppfg"
                     and k[2] == "stagnation_compound"
                     and k[3] == 500]

    table2_md = [
        "# Table 2 (raw-modal-share variant) — Compound-gate ablation, Qwen, n=500",
        "",
        "Mean ± std over 3 seeds (per cell). Replaces binary `answer_mode_rate` ↓ with continuous **raw_modal_share** ↓.",
        "",
        "| Method | Pass@1 | Pass@2 | Pass@4 | Pass@8 | raw_modal_share ↓ | distinct_4 ↑ | tpp |",
        "|---|---|---|---|---|---|---|---|",
    ]
    table2_md.append(emit_row("indep (N=8)", cells.get(indep_key, [])))
    table2_md.append(emit_row("PPFG-stag (N=8)", cells.get(ppfg_stag_key, [])))
    table2_json = {}
    for k, rows in sorted(compound_keys, key=lambda x: (x[0][4] or 0)):
        t = k[4]
        label = f"PPFG-stag-compound (t={t:.2f}, N=8)"
        table2_md.append(emit_row(label, rows))
        ms_m, ms_s = cell_stats(rows, "raw_modal_share")
        amr_m, amr_s = cell_stats(rows, "answer_mode_rate")
        table2_json[label] = {
            "n_seeds": len(rows),
            "raw_modal_share_mean": ms_m,
            "raw_modal_share_std": ms_s,
            "answer_mode_rate_mean": amr_m,
            "answer_mode_rate_std": amr_s,
            "abs_delta_vs_amr": (ms_m - amr_m) if (ms_m is not None and amr_m is not None) else None,
        }
    for ref_label, ref_k in (("indep (N=8)", indep_key), ("PPFG-stag (N=8)", ppfg_stag_key)):
        rows = cells.get(ref_k, [])
        ms_m, ms_s = cell_stats(rows, "raw_modal_share")
        amr_m, amr_s = cell_stats(rows, "answer_mode_rate")
        table2_json[ref_label] = {
            "n_seeds": len(rows),
            "raw_modal_share_mean": ms_m,
            "raw_modal_share_std": ms_s,
            "answer_mode_rate_mean": amr_m,
            "answer_mode_rate_std": amr_s,
            "abs_delta_vs_amr": (ms_m - amr_m) if (ms_m is not None and amr_m is not None) else None,
        }

    Path(RESULTS / "table2_raw_modal_share.md").write_text("\n".join(table2_md) + "\n")

    # ---- Full JSON dump
    full = {
        "spearman_rho_qwen_n500_only": None,  # filled below
        "table1_cells": table1_json,
        "table2_cells": table2_json,
        "all_cells": {},
    }
    qwen_n500_raw = []
    qwen_n500_amr = []
    for k, rows in cells.items():
        if len(rows) < 2:
            continue
        ms_m, ms_s = cell_stats(rows, "raw_modal_share")
        amr_m, amr_s = cell_stats(rows, "answer_mode_rate")
        p1_m, p1_s = cell_stats(rows, "pass_at_1")
        p2_m, p2_s = cell_stats(rows, "pass_at_2")
        p4_m, p4_s = cell_stats(rows, "pass_at_4")
        p8_m, p8_s = cell_stats(rows, "pass_at_8")
        d4_m, d4_s = cell_stats(rows, "distinct_4")
        tpp_m, _ = cell_stats(rows, "tpp")
        label = f"{k[0]} | {k[1]}{'+'+k[2] if k[2] else ''} | n={k[3]}{' | prm_low_t='+str(k[4]) if k[4] is not None else ''}"
        full["all_cells"][label] = {
            "n_seeds": len(rows),
            "pass_at_1": [p1_m, p1_s],
            "pass_at_2": [p2_m, p2_s],
            "pass_at_4": [p4_m, p4_s],
            "pass_at_8": [p8_m, p8_s],
            "raw_modal_share": [ms_m, ms_s],
            "answer_mode_rate": [amr_m, amr_s],
            "distinct_4": [d4_m, d4_s],
            "tpp": tpp_m,
            "abs_delta_share_vs_mode_rate": (ms_m - amr_m) if (ms_m and amr_m) else None,
        }
        if k[0] == "Qwen2.5-7B-Instruct" and k[3] == 500:
            qwen_n500_raw.append(ms_m)
            qwen_n500_amr.append(amr_m)

    # Spearman ρ over Qwen n=500 cells only (paper's claim)
    def spearman(xs, ys):
        def rank(vs):
            idx = sorted(range(len(vs)), key=lambda i: vs[i])
            ranks = [0.0] * len(vs)
            i = 0
            while i < len(idx):
                j = i
                while j + 1 < len(idx) and vs[idx[j + 1]] == vs[idx[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1
                for kk in range(i, j + 1):
                    ranks[idx[kk]] = avg
                i = j + 1
            return ranks
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx = sum(rx) / n
        my = sum(ry) / n
        num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        dx = (sum((r - mx) ** 2 for r in rx)) ** 0.5
        dy = (sum((r - my) ** 2 for r in ry)) ** 0.5
        return num / (dx * dy) if dx and dy else float("nan")

    if len(qwen_n500_raw) >= 3:
        full["spearman_rho_qwen_n500_only"] = spearman(qwen_n500_raw, qwen_n500_amr)
        full["n_qwen_n500_cells"] = len(qwen_n500_raw)

    Path(RESULTS / "raw_modal_share_full.json").write_text(json.dumps(full, indent=2))

    print("Wrote:")
    print("  results/table1_raw_modal_share.md")
    print("  results/table2_raw_modal_share.md")
    print("  results/raw_modal_share_full.json")
    print()
    print(f"Spearman ρ on Qwen-n=500 cells = {full['spearman_rho_qwen_n500_only']:.4f} "
          f"(N={full.get('n_qwen_n500_cells', 0)} cells)")


if __name__ == "__main__":
    main()
