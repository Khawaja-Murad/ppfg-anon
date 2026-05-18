"""NuminaMath-CoT task — replacement for OlympiadBench (Day-13 v13 pivot).

Loaded from HuggingFace: AI-MO/NuminaMath-CoT, test split (100 problems).
Sources include olympiads, aops_forum, amc_aime, synthetic_math, orca_math,
cn_k12, gsm8k, synthetic_amc, and math. Mixed difficulty with substantial
olympiad-tier representation — fills the OlympiadBench role at a smaller n.

Loader bypasses HF `load_dataset` (which requires online metadata in offline
mode) and reads the cached parquet directly. Answer extraction mirrors MATH500:
`math_verify` primary + normalized-string fallback.

Note: the NuminaMath-CoT `solution` field is the full CoT solution text, often
with the final answer in `\\boxed{...}`. The `answer` field is NOT present —
we extract gold answer from the `\\boxed{...}` of the solution at load time.
"""

from __future__ import annotations
import logging
import re
from typing import Any

from hyp_forest.tasks.base import Problem, Task
from hyp_forest.utils.prompts import build_math_cot_prompt

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_boxed_answer(text: str) -> str | None:
    """Pull the contents of the last `\\boxed{...}` in a string. Handles one
    level of nested braces (covers most math answers like `\\boxed{\\frac{1}{2}}`)."""
    if text is None:
        return None
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_for_comparison(s: str) -> str:
    """Light normalization for the math-verify fallback."""
    if s is None:
        return ""
    s = s.strip().replace(" ", "").replace("\\,", "").replace("$", "")
    return s


class NuminaMath(Task):
    name = "numina_math"

    def load(self, n: int = -1, subset: list[int] | None = None) -> list[Problem]:
        """Load NuminaMath-CoT test problems. Reads parquet directly from
        the HF cache (offline-safe)."""
        import glob, os
        import pandas as pd  # type: ignore

        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        fs = sorted(glob.glob(
            f"{hf_home}/hub/datasets--AI-MO--NuminaMath-CoT/snapshots/*/data/test-*.parquet"
        ))
        if not fs:
            raise RuntimeError(
                f"NuminaMath-CoT test parquet not found under "
                f"{hf_home}/hub/datasets--AI-MO--NuminaMath-CoT/snapshots/*/data; "
                f"run `huggingface-cli download AI-MO/NuminaMath-CoT --repo-type dataset --include 'data/test*'` "
                f"on the login node"
            )
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        logger.info(f"NuminaMath-CoT test split: {len(df)} rows from {len(fs)} parquet file(s)")

        subset_set = set(subset) if subset is not None else None
        problems: list[Problem] = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            if subset_set is not None:
                if i not in subset_set:
                    continue
            elif 0 <= n <= i:
                break

            question = row.get("problem")
            solution = row.get("solution", "")
            gold = _extract_boxed_answer(solution)
            if question is None or gold is None:
                logger.warning(
                    f"numina-{i}: missing question or no \\boxed{{}} in solution "
                    f"(source={row.get('source')}); skipping"
                )
                continue
            problems.append(
                Problem(
                    problem_id=f"numina-{i}",
                    question=str(question),
                    answer=str(gold),
                    metadata={"source": str(row.get("source", ""))},
                )
            )
        logger.info(f"Loaded {len(problems)} NuminaMath-CoT problems")
        return problems

    def build_prompt(self, problem: Problem, lm) -> str:
        return build_math_cot_prompt(problem.question, lm)

    def is_correct(self, candidate: str | None, problem: Problem) -> bool:
        if candidate is None:
            return False
        # Try math_verify first (best for math expressions)
        try:
            from math_verify import parse, verify  # type: ignore
            cand_boxed = _extract_boxed_answer(candidate)
            cand = cand_boxed if cand_boxed is not None else candidate
            try:
                if verify(parse(problem.answer), parse(cand)):
                    return True
            except Exception:
                pass
        except ImportError:
            pass
        # Fallback: normalized string comparison
        cand_boxed = _extract_boxed_answer(candidate)
        if cand_boxed is None:
            cand_boxed = candidate
        return _normalize_for_comparison(cand_boxed) == _normalize_for_comparison(problem.answer)
