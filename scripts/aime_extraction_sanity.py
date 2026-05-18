#!/usr/bin/env python3
"""V14.D.2 — AIME extraction sanity.

Sample 10 random chains from each architecture's INDEP AIME trajectory.
For each: print gold, model's last 200 chars, extractor output, and
flag any mismatch where the gold integer is visible in the model
output but the extractor missed it.
"""
import json
import random
import re
from pathlib import Path

ARCH_DIRS = {
    "Qwen":     "results/independent-math500-20260516-055038-j61065730",     # seed 42
    "LLaMA":    "results/independent-math500-20260516-165609-j61090306",     # seed 42 (has trajectories)
    "DeepSeek": "results/independent-math500-20260516-163909-j61088919",     # seed 43 (has trajectories)
}

N_CHAINS_PER_ARCH = 10
SEED = 17

OUT = Path("results/aime_extraction_sanity_check.json")


def visible_gold_in_text(text: str, gold) -> bool:
    """Does the gold integer appear as a free-standing number in text?"""
    if gold is None:
        return False
    g = str(gold).strip()
    if not g.isdigit():
        return False
    # match at word boundary or in boxed
    pat = re.compile(rf'(?<!\d){g}(?!\d)')
    return bool(pat.search(text))


def main():
    rng = random.Random(SEED)
    report = {"arch_results": {}, "_mismatch_flags": []}

    for arch, d in ARCH_DIRS.items():
        traj = json.load(open(Path(d) / "trajectories.json"))
        # AIME has 120 problems; pick 10 random problems, take chain 0 from each
        chosen_problems = rng.sample(range(len(traj)), min(N_CHAINS_PER_ARCH, len(traj)))
        rows = []
        for pidx in chosen_problems:
            prob = traj[pidx]
            gold = prob.get("answer")
            chain = prob["population"]["chains"][0]
            last_text = "\n".join(chain.get("steps", [])[-2:])[-300:]
            extracted = chain.get("final_answer")
            visible = visible_gold_in_text(last_text, gold)
            mismatch = False
            if extracted is None and visible:
                mismatch = True
            elif extracted is not None and gold is not None:
                # both have values, but they differ
                a = str(extracted).strip().rstrip(".").replace(" ", "")
                g = str(gold).strip().rstrip(".").replace(" ", "")
                if a != g and visible:
                    mismatch = True
            rows.append({
                "problem_id": prob.get("problem_id"),
                "gold": gold,
                "extracted": extracted,
                "chain_status": chain.get("status"),
                "gold_visible_in_last_text": visible,
                "mismatch_flag": mismatch,
                "last_text_300": last_text,
            })
            if mismatch:
                report["_mismatch_flags"].append({
                    "arch": arch,
                    "problem_id": prob.get("problem_id"),
                    "gold": gold,
                    "extracted": extracted,
                })
        report["arch_results"][arch] = {
            "source_dir": d,
            "n_chains_examined": len(rows),
            "n_mismatch": sum(1 for r in rows if r["mismatch_flag"]),
            "n_extractor_found": sum(1 for r in rows if r["extracted"] is not None),
            "n_chain_promoted": sum(1 for r in rows if r["chain_status"] == "promoted"),
            "rows": rows,
        }

    OUT.write_text(json.dumps(report, indent=2))

    print(f"=== AIME extraction sanity check ===")
    for arch, r in report["arch_results"].items():
        print(f"  {arch}: n={r['n_chains_examined']}, promoted={r['n_chain_promoted']},"
              f" extracted={r['n_extractor_found']}, mismatch={r['n_mismatch']}")
    print(f"\nTotal mismatch flags: {len(report['_mismatch_flags'])}")
    if report["_mismatch_flags"]:
        print("Case (ii) — extraction bugs found. See", OUT)
    else:
        print("Case (i) — extraction consistent. No paper edit needed.")


if __name__ == "__main__":
    main()
