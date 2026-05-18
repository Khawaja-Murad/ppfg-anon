#!/usr/bin/env python
"""VersaPRM loading + scoring smoke test.

Day-13 V13.0 sprint: verifies VersaPRM (UW-Madison-Lee-Lab/VersaPRM PEFT adapter on
UW-Madison-Lee-Lab/Llama-PRM800K base) loads and produces plausible per-step scores on
hardcoded MATH500-style and MMLU-Pro Law-style chains.

This is an isolation test — no SLURM, no problem-set loading, no PPFG plumbing. It only
exercises the new `prm_kind="versaprm"` code path in src/hyp_forest/models/prm.py.

Calibration intuition (NOT a hard pass/fail — VersaPRM is a noisy scorer):
    - Chains with arithmetic / reasoning errors should score lower than correct chains
      *on average* over the steps in that chain.
    - MMLU-Pro Law chains exercise a non-math domain — we report scores but do not assert
      anything specific. Their purpose is to confirm the scorer doesn't crash on prose.

Output: results/versaprm_smoke.json with per-chain step scores and per-chain mean,
plus a printed summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

# Make src/ importable when run directly (mirrors the SLURM launcher).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyp_forest.models.prm import ProcessRewardModel  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded test inputs. Two domains, five chains each. Within each domain the
# first three are "correct" reasoning, the last two have a deliberate error.
# ---------------------------------------------------------------------------

MATH_PROBLEM = (
    "Compute the value of x in the equation 3x + 7 = 22."
)
MATH_CHAINS: list[tuple[str, list[str]]] = [
    # --- 3 correct ---
    ("math_correct_1", [
        "Subtract 7 from both sides: 3x = 22 - 7 = 15.",
        "Divide both sides by 3: x = 15 / 3 = 5.",
        "Therefore x = 5.",
    ]),
    ("math_correct_2", [
        "We isolate the term containing x by subtracting 7 from both sides, giving 3x = 15.",
        "Then divide by 3 to obtain x = 5.",
        "Check: 3(5) + 7 = 15 + 7 = 22, which matches the right-hand side. So x = 5.",
    ]),
    ("math_correct_3", [
        "Rewrite as 3x = 22 - 7.",
        "So 3x = 15, hence x = 5.",
    ]),
    # --- 2 incorrect (arithmetic error in step 1 propagates) ---
    ("math_incorrect_1", [
        "Subtract 7 from both sides: 3x = 22 - 7 = 16.",  # wrong: 22-7=15
        "Divide both sides by 3: x = 16 / 3.",
        "Therefore x = 16/3.",
    ]),
    ("math_incorrect_2", [
        "Add 7 to both sides: 3x = 22 + 7 = 29.",  # wrong operation
        "Divide both sides by 3: x = 29 / 3.",
        "Therefore x ≈ 9.67.",
    ]),
]

LAW_PROBLEM = (
    "A homeowner orally agrees with a neighbor to sell a parcel of land for $50,000. "
    "The neighbor pays a $5,000 deposit but no written contract is signed. The homeowner "
    "later refuses to sell. Under the Statute of Frauds in most US jurisdictions, is the "
    "oral agreement enforceable as a contract for the sale of real property?"
)
LAW_CHAINS: list[tuple[str, list[str]]] = [
    # --- 3 correct (correctly invokes Statute of Frauds) ---
    ("law_correct_1", [
        "The Statute of Frauds requires that contracts for the sale of real property be evidenced by a writing signed by the party against whom enforcement is sought.",
        "Here the agreement is purely oral and there is no signed writing.",
        "Therefore the oral agreement is not enforceable as a contract for the sale of real property, absent an applicable exception such as part performance.",
    ]),
    ("law_correct_2", [
        "Real estate sales fall within the Statute of Frauds in virtually every US jurisdiction.",
        "A $5,000 deposit alone is generally not enough to satisfy the writing requirement; courts typically require a signed memorandum identifying the parties, the land, and the price.",
        "So the homeowner can raise the Statute of Frauds as a defense and the oral agreement is unenforceable.",
    ]),
    ("law_correct_3", [
        "The Statute of Frauds bars enforcement of oral contracts for the sale of land.",
        "Hence the neighbor cannot compel specific performance on this oral promise.",
    ]),
    # --- 2 incorrect (misapplies doctrine) ---
    ("law_incorrect_1", [
        "Oral contracts are always enforceable as long as some money has been paid.",  # wrong general rule
        "Therefore the $5,000 deposit makes the contract binding regardless of the Statute of Frauds.",
        "So the homeowner must sell.",
    ]),
    ("law_incorrect_2", [
        "The Statute of Frauds only applies to contracts that cannot be performed within one year.",  # wrong — also applies to land
        "A real estate sale can usually be closed in a few weeks, so the Statute of Frauds does not apply.",
        "Therefore the oral agreement is fully enforceable.",
    ]),
]


def score_chains(prm: ProcessRewardModel, problem: str, chains: list[tuple[str, list[str]]]) -> list[dict]:
    """Score each chain and return a per-chain summary."""
    out: list[dict] = []
    for cid, steps in chains:
        scores = prm.score_steps(problem, steps)
        out.append({
            "chain_id": cid,
            "n_steps": len(steps),
            "step_scores": [round(float(s), 4) for s in scores],
            "mean_score": round(float(mean(scores)), 4) if scores else None,
        })
    return out


def main() -> int:
    print("[versaprm_smoke] Constructing ProcessRewardModel(prm_kind='versaprm')...")
    prm = ProcessRewardModel(prm_kind="versaprm")
    print("[versaprm_smoke] Loaded. Scoring 5 math + 5 law chains...")

    math_results = score_chains(prm, MATH_PROBLEM, MATH_CHAINS)
    law_results = score_chains(prm, LAW_PROBLEM, LAW_CHAINS)

    # Aggregate sanity stats: mean of (correct chain means) vs mean of (incorrect chain means).
    def split_means(results: list[dict], correct_prefix: str, incorrect_prefix: str) -> tuple[float, float]:
        c = [r["mean_score"] for r in results if r["chain_id"].startswith(correct_prefix)]
        ic = [r["mean_score"] for r in results if r["chain_id"].startswith(incorrect_prefix)]
        return (mean(c) if c else float("nan"), mean(ic) if ic else float("nan"))

    math_c_mean, math_ic_mean = split_means(math_results, "math_correct", "math_incorrect")
    law_c_mean, law_ic_mean = split_means(law_results, "law_correct", "law_incorrect")

    summary = {
        "prm_kind": "versaprm",
        "base_model": "UW-Madison-Lee-Lab/Llama-PRM800K",
        "adapter": "UW-Madison-Lee-Lab/VersaPRM",
        "math": {
            "problem": MATH_PROBLEM,
            "chains": math_results,
            "mean_correct": round(float(math_c_mean), 4),
            "mean_incorrect": round(float(math_ic_mean), 4),
            "calibration_gap_correct_minus_incorrect": round(float(math_c_mean - math_ic_mean), 4),
        },
        "law": {
            "problem": LAW_PROBLEM,
            "chains": law_results,
            "mean_correct": round(float(law_c_mean), 4),
            "mean_incorrect": round(float(law_ic_mean), 4),
            "calibration_gap_correct_minus_incorrect": round(float(law_c_mean - law_ic_mean), 4),
        },
    }

    # Print human-readable summary.
    print("\n========== VersaPRM smoke summary ==========")
    for domain in ("math", "law"):
        d = summary[domain]
        print(f"\n[{domain}] mean(correct) = {d['mean_correct']:.4f}  "
              f"mean(incorrect) = {d['mean_incorrect']:.4f}  "
              f"gap = {d['calibration_gap_correct_minus_incorrect']:+.4f}")
        for r in d["chains"]:
            print(f"  {r['chain_id']:>22s}  mean={r['mean_score']:.4f}  "
                  f"steps={r['step_scores']}")
    print("\n============================================\n")

    out_path = ROOT / "results" / "versaprm_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[versaprm_smoke] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
