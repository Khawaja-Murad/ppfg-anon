"""MATH500 task — 500-problem subset of the MATH dataset.

Loaded from HuggingFace datasets: HuggingFaceH4/MATH-500 (created by Lightman et al.
for 'Let's Verify Step by Step', the canonical PRM benchmark).

Answer comparison uses math_verify (sympy-based) for robust comparison; falls back to
string-equality if math_verify isn't installed.
"""

from __future__ import annotations
import logging
from typing import Any

from hyp_forest.tasks.base import Problem, Task
from hyp_forest.utils.prompts import build_math_cot_prompt

logger = logging.getLogger(__name__)


class MATH500(Task):
    name = "math500"

    def load(self, n: int = -1, subset: list[int] | None = None) -> list[Problem]:
        """Load MATH500 problems. If `subset` is provided (list of dataset
        indices), return only those problems and ignore `n`. Otherwise return
        the first `n` problems (head-slice; n=-1 means all)."""
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        if subset is not None:
            subset_set = set(subset)
            problems = [
                Problem(
                    problem_id=f"math500-{i}",
                    question=row["problem"],
                    answer=row["answer"],
                    metadata={
                        "subject": row.get("subject"),
                        "level": row.get("level"),
                    },
                )
                for i, row in enumerate(ds) if i in subset_set
            ]
            logger.info(f"Loaded {len(problems)} problems from MATH500 (subset of indices {min(subset)}..{max(subset)})")
            return problems
        problems = []
        for i, row in enumerate(ds):
            if 0 <= n <= i:
                break
            problems.append(
                Problem(
                    problem_id=f"math500-{i}",
                    question=row["problem"],
                    answer=row["answer"],
                    metadata={
                        "subject": row.get("subject"),
                        "level": row.get("level"),
                    },
                )
            )
        logger.info(f"Loaded {len(problems)} problems from MATH500")
        return problems

    def build_prompt(self, problem: Problem, lm) -> str:
        return build_math_cot_prompt(problem.question, lm)

    def is_correct(self, candidate: str | None, problem: Problem) -> bool:
        if candidate is None:
            return False
        # Try math_verify first (handles \boxed{}, fractions, equivalent forms)
        try:
            from math_verify import parse, verify  # type: ignore
            gold = parse(problem.answer)
            cand = parse(candidate)
            return bool(verify(gold, cand))
        except ImportError:
            return _fallback_compare(candidate, problem.answer)
        except Exception as e:
            logger.debug(f"math_verify error on '{candidate}' vs '{problem.answer}': {e}")
            return _fallback_compare(candidate, problem.answer)


def _fallback_compare(cand: str, gold: str) -> bool:
    """Conservative string-equality fallback: strip whitespace, dollar signs, lowercase."""
    def norm(s: str) -> str:
        return s.strip().replace("$", "").replace(" ", "").lower()
    return norm(cand) == norm(gold)
