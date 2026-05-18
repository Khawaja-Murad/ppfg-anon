"""AIME task — American Invitational Mathematics Examination problems.

Loaded from HuggingFace:
    - AIME 2024 (30 problems): AI-MO/aimo-validation-aime
    - AIME 2025 (30 problems): yentinglin/aime_2025 (optional; skipped if unavailable)

Answers are integers in [0, 999]. Extraction tries `\\boxed{N}` first, then
'the answer is N', then the last integer in the string. `is_correct` parses
both candidate and ground-truth as int.
"""

from __future__ import annotations
import logging
import re
from typing import Any

from hyp_forest.tasks.base import Problem, Task
from hyp_forest.utils.prompts import build_math_cot_prompt

logger = logging.getLogger(__name__)

_BOXED_INT_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_ANSWER_IS_RE = re.compile(r"(?:the\s+answer\s+is|final\s+answer\s*[:=]?)\s*(-?\d+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def _extract_int(text: str) -> int | None:
    """Pull an integer out of an answer string."""
    if text is None:
        return None
    m = _BOXED_INT_RE.search(text)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, TypeError):
            pass
    m = _ANSWER_IS_RE.search(text)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, TypeError):
            pass
    matches = _INT_RE.findall(text)
    if matches:
        try:
            return int(matches[-1])
        except (ValueError, TypeError):
            return None
    return None


class AIME(Task):
    name = "aime"

    def load(self, n: int = -1, subset: list[int] | None = None) -> list[Problem]:
        """Load AIME 2024 (+ 2025 if available) problems. Reads parquet files
        directly from the HF cache because the AIME datasets are gated behind
        load_dataset's online metadata check even when fully cached on a CC
        compute node with HF_HUB_OFFLINE=1; the direct-parquet path bypasses
        that. (Verified Day-13 fix after first AIME submission produced
        n_problems=0 results.)"""
        import glob, os
        import pandas as pd  # type: ignore

        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        problems: list[Problem] = []
        years_loaded: list[str] = []

        # AIME 2024
        fs = sorted(glob.glob(
            f"{hf_home}/hub/datasets--AI-MO--aimo-validation-aime/snapshots/*/data/*.parquet"
        ))
        if fs:
            df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            for i, row in df.iterrows():
                q = row.get("problem") or row.get("question") or row.get("Problem")
                a = row.get("answer") or row.get("Answer")
                if q is None or a is None:
                    logger.warning(f"aime-2024-{i}: missing question or answer, skipping")
                    continue
                problems.append(
                    Problem(
                        problem_id=f"aime-2024-{i}",
                        question=str(q),
                        answer=str(a).strip(),
                        metadata={"year": 2024},
                    )
                )
            years_loaded.append("2024")
            logger.info(f"Loaded {len(problems)} AIME 2024 problems from parquet ({len(fs)} files)")
        else:
            logger.warning(
                f"AIME 2024 parquet not found under {hf_home}/hub/datasets--AI-MO--aimo-validation-aime/snapshots/*/data"
            )

        # AIME 2025 (optional)
        n_after_2024 = len(problems)
        fs = sorted(glob.glob(
            f"{hf_home}/hub/datasets--yentinglin--aime_2025/snapshots/*/data/*.parquet"
        ))
        if fs:
            df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            for i, row in df.iterrows():
                q = row.get("problem") or row.get("question") or row.get("Problem")
                a = row.get("answer") or row.get("Answer")
                if q is None or a is None:
                    logger.warning(f"aime-2025-{i}: missing question or answer, skipping")
                    continue
                problems.append(
                    Problem(
                        problem_id=f"aime-2025-{i}",
                        question=str(q),
                        answer=str(a).strip(),
                        metadata={"year": 2025},
                    )
                )
            years_loaded.append("2025")
            logger.info(f"Loaded {len(problems) - n_after_2024} AIME 2025 problems from parquet")
        else:
            logger.warning(
                f"AIME 2025 parquet not found under {hf_home}/hub/datasets--yentinglin--aime_2025/snapshots/*/data; "
                f"proceeding with AIME 2024 only"
            )

        logger.info(f"AIME years loaded: {years_loaded}; total problems: {len(problems)}")

        if subset is not None:
            subset_set = set(subset)
            problems = [p for i, p in enumerate(problems) if i in subset_set]
            logger.info(
                f"Returning {len(problems)} AIME problems (subset of indices "
                f"{min(subset) if subset else '-'}..{max(subset) if subset else '-'})"
            )
            return problems

        if 0 <= n < len(problems):
            problems = problems[:n]
        logger.info(f"Returning {len(problems)} AIME problems")
        return problems

    def build_prompt(self, problem: Problem, lm) -> str:
        return build_math_cot_prompt(problem.question, lm)

    def is_correct(self, candidate: str | None, problem: Problem) -> bool:
        if candidate is None:
            return False
        cand_int = _extract_int(candidate)
        gold_int = _extract_int(problem.answer)
        if cand_int is None or gold_int is None:
            return False
        return cand_int == gold_int
