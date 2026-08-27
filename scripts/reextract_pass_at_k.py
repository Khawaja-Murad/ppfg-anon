"""Post-hoc re-extraction of final answers from saved trajectories.json.

Day-8 found that `src/hyp_forest/chains/chain.py::detect_final_answer` uses
`r"\\boxed\{([^{}]+)\}"` which does NOT match nested braces in LaTeX. When
the regex fails on `\boxed{\left(3,\frac{\pi}{2}\right)}`, the function
falls through to a "the answer is" regex that greedily captures trailing
prose ("is indeed \\boxed{X}"), causing math_verify mismatch.

This script:
  1. Reads a results/<dir>/trajectories.json
  2. For each chain's last step, applies a *balanced-brace* `\\boxed{}` parser
  3. Falls back to a tightened "the answer is" matcher that strips wrapper text
  4. Recomputes pass_at_k against the gold answer from the task loader
  5. Prints {old, new} Pass@1 / Pass@k for comparison

Usage:
  PYTHONPATH=src python scripts/reextract_pass_at_k.py <results_dir>

NOTE: Does NOT modify any production code. Does NOT modify saved trajectories.
Read-only post-hoc analysis. If results look promising on the failed smokes,
the lead author can decide to fix detect_final_answer in chain.py and re-run eval.py.
"""

from __future__ import annotations
import json
import re
import sys
import math
from pathlib import Path


def extract_boxed_balanced(text: str) -> str | None:
    """Find the LAST \\boxed{...} in text with balanced-brace matching."""
    needle = r"\boxed{"
    idx = text.rfind(needle)
    if idx < 0:
        return None
    start = idx + len(needle)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def detect_final_answer_fixed(step_text: str) -> str | None:
    """Improved version of chain.py::detect_final_answer.

    Changes from the original:
      1. Balanced-brace matcher for \\boxed{...} — handles nested braces.
      2. Uses LAST \\boxed{} occurrence (in case the model boxes a partial
         answer earlier in the step and re-boxes the final answer at the end).
      3. Final-answer fallback regex is stricter: requires the captured text
         to NOT contain "\\boxed" (to prevent capturing wrapper prose).
    """
    boxed = extract_boxed_balanced(step_text)
    if boxed is not None:
        return boxed
    m = re.search(
        r"(?:final answer|the answer is)\s*[:\-]?\s*([^\n.]+?)(?:\.|$|\n)",
        step_text,
        re.IGNORECASE,
    )
    if m:
        captured = m.group(1).strip().rstrip(".")
        # If the "the answer is X" capture itself contains a \\boxed{} wrapper,
        # extract the inner content (defensive — the balanced matcher above
        # should have caught it, but the model may produce edge cases).
        if r"\boxed" in captured:
            inner = extract_boxed_balanced(captured)
            return inner if inner is not None else None
        return captured
    return None


def normalize_answer(s: str) -> str:
    """Loose normalization for fallback string comparison."""
    if s is None:
        return ""
    s = s.strip()
    # Strip outer $...$ math delimiters and outer braces
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # Common LaTeX cleanups
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\s+", "", s)
    # Strip trailing units like "\\text{}"
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s


def answers_equivalent(pred: str, gold: str) -> bool:
    """Try math_verify if available; else fall back to normalized string match."""
    if pred is None or gold is None:
        return False
    if normalize_answer(pred) == normalize_answer(gold):
        return True
    # Try math_verify for symbolic equivalence
    try:
        from math_verify import parse, verify
        return verify(parse(gold), parse(pred))
    except ImportError:
        from hyp_forest.comparator_guard import warn_if_degraded
        warn_if_degraded()
        return False
    except Exception:
        return False


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k from Chen et al. 2021."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/reextract_pass_at_k.py <results_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    traj_path = run_dir / "trajectories.json"
    metrics_path = run_dir / "metrics.json"
    if not traj_path.exists():
        print(f"NO TRAJECTORIES at {traj_path}")
        sys.exit(1)
    traj = json.load(open(traj_path))
    old_metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}

    # Need to grab gold answers — they're not in trajectories.json (only in the
    # task object at runtime). Re-load the math500 task to get gold answers
    # for the problems this cell ran.
    # Trajectory schema: per-problem dict has 'problem_id', 'answer', 'metadata',
    # 'population'. The gold answer is directly in t['answer']. No external load.
    pass

    n_chains = None
    per_problem_correct: list[int] = []
    total_chains = 0
    for p in traj:
        chains = p.get("population", p).get("chains", [])
        if n_chains is None:
            n_chains = len(chains)
        gold = p.get("answer", p.get("gold_answer"))
        n_correct = 0
        for c in chains:
            steps = c.get("steps", [])
            if not steps:
                continue
            # Fixed extraction from last step
            new_ans = detect_final_answer_fixed(steps[-1])
            # Also try preceding steps if last step didn't have a final answer
            if new_ans is None:
                for s in reversed(steps[:-1]):
                    new_ans = detect_final_answer_fixed(s)
                    if new_ans is not None:
                        break
            if new_ans is not None and gold is not None:
                if answers_equivalent(new_ans, gold):
                    n_correct += 1
        per_problem_correct.append(n_correct)
        total_chains += len(chains)

    n = n_chains or 8
    k_values = [1, 2, 4, 8]
    print(f"=== {run_dir.name} ===")
    print(f"N chains/problem: {n}, N problems: {len(per_problem_correct)}")
    if old_metrics:
        old_pk = old_metrics.get("pass_at_k", {})
        old_mr = old_metrics.get("diversity", {}).get("answer_mode_rate", -1)
        print(f"OLD Pass@k: {{1: {old_pk.get('1', '?'):.4f}, 2: {old_pk.get('2', '?'):.4f}, 4: {old_pk.get('4', '?'):.4f}, 8: {old_pk.get('8', '?'):.4f}}}")
        print(f"OLD mode_rate: {old_mr:.4f}")
    new_pk = {k: sum(pass_at_k_estimator(n, c, k) for c in per_problem_correct) / max(1, len(per_problem_correct)) for k in k_values}
    print(f"NEW Pass@k: {{1: {new_pk[1]:.4f}, 2: {new_pk[2]:.4f}, 4: {new_pk[4]:.4f}, 8: {new_pk[8]:.4f}}}")
    print(f"  Pass@1 Δ: {new_pk[1] - old_metrics.get('pass_at_k', {}).get('1', 0):+.4f}")


if __name__ == "__main__":
    main()
