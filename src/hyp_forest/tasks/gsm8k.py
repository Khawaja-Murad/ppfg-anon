"""GSM8K task — grade-school math word problems.

Loaded from HuggingFace `gsm8k`, config 'main', split 'test' (1319 problems).

Gold answers in GSM8K live in the `answer` field as a full worked solution
terminated by '#### N' where N is the numeric final answer. We extract the
numeric part with a regex on load. Chain-side, we reuse the existing
`detect_final_answer` (\\boxed{...} / 'Final answer: ...') because the
prompt instructs the model to box its final answer, mirroring MATH500.
A numeric fallback (last integer/float in the candidate string) is provided
for `is_correct` robustness.
"""

from __future__ import annotations
import logging
import re
from typing import Any

from hyp_forest.tasks.base import Problem, Task
from hyp_forest.utils.prompts import build_math_cot_prompt

logger = logging.getLogger(__name__)

_GOLD_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_gold(answer_text: str) -> str | None:
    m = _GOLD_RE.search(answer_text)
    return m.group(1) if m else None


def _canon_num(s: str) -> str | None:
    """Normalize a numeric string: strip commas, $, %, whitespace; collapse to
    plain float-or-int form. Returns None if no number found."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    # First try a direct float parse
    try:
        f = float(s)
    except (ValueError, TypeError):
        m = _NUM_RE.search(s)
        if m is None:
            return None
        try:
            f = float(m.group(0))
        except (ValueError, TypeError):
            return None
    # Render as integer when exact, else as float with up to 6 decimals
    if f.is_integer():
        return str(int(f))
    return f"{f:.6f}".rstrip("0").rstrip(".")


class GSM8K(Task):
    name = "gsm8k"

    def load(self, n: int = -1, subset: list[int] | None = None) -> list[Problem]:
        """Load GSM8K problems. If `subset` is provided (list of dataset
        indices), return only those problems and ignore `n`. Otherwise return
        the first `n` problems (head-slice; n=-1 means all)."""
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("gsm8k", "main", split="test")
        if subset is not None:
            subset_set = set(subset)
            problems = []
            for i, row in enumerate(ds):
                if i not in subset_set:
                    continue
                gold = _extract_gold(row["answer"])
                if gold is None:
                    logger.warning(f"gsm8k-{i}: gold answer regex miss, skipping")
                    continue
                problems.append(
                    Problem(
                        problem_id=f"gsm8k-{i}",
                        question=row["question"],
                        answer=gold,
                        metadata={"raw_solution": row["answer"]},
                    )
                )
            logger.info(f"Loaded {len(problems)} problems from GSM8K (subset of indices {min(subset)}..{max(subset)})")
            return problems
        problems = []
        for i, row in enumerate(ds):
            if 0 <= n <= i:
                break
            gold = _extract_gold(row["answer"])
            if gold is None:
                logger.warning(f"gsm8k-{i}: gold answer regex miss, skipping")
                continue
            problems.append(
                Problem(
                    problem_id=f"gsm8k-{i}",
                    question=row["question"],
                    answer=gold,
                    metadata={"raw_solution": row["answer"]},
                )
            )
        logger.info(f"Loaded {len(problems)} problems from GSM8K")
        return problems

    def build_prompt(self, problem: Problem, lm) -> str:
        # Reuse the MATH chain-of-thought system prompt — it instructs the
        # model to step-number and \boxed{} the final answer, which matches
        # how detect_final_answer picks up the chain-side answer.
        return build_math_cot_prompt(problem.question, lm)

    def is_correct(self, candidate: str | None, problem: Problem) -> bool:
        if candidate is None:
            return False
        cand_canon = _canon_num(candidate)
        gold_canon = _canon_num(problem.answer)
        if cand_canon is None or gold_canon is None:
            return False
        # Direct canonical match
        if cand_canon == gold_canon:
            return True
        # Tolerate trivial float equality (e.g., "5" vs "5.0")
        try:
            return abs(float(cand_canon) - float(gold_canon)) < 1e-6
        except (ValueError, TypeError):
            return False
