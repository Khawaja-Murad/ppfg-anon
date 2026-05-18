"""OlympiadBench-math (English text-only) task.

Loaded from HuggingFace: Hothan/OlympiadBench. We try config
`OE_TO_maths_en_COMP` first (Open-Ended, Text-Only, English Math,
Competition), then fall back to `en_text_only_math` if the first name
isn't recognized.

We use the first 500 problems (head-slice). Answer comparison mirrors
MATH500: math_verify first, normalized-string fallback if math_verify
isn't available.
"""

from __future__ import annotations
import logging
from typing import Any

from hyp_forest.tasks.base import Problem, Task
from hyp_forest.utils.prompts import build_math_cot_prompt

logger = logging.getLogger(__name__)


def _normalize_answer(ans: Any) -> str:
    """OlympiadBench `final_answer` is sometimes a list (multi-part). We
    coerce to a single string by joining with ' ; ' so the verifier sees
    something deterministic. Scalars pass through."""
    if ans is None:
        return ""
    if isinstance(ans, list):
        return " ; ".join(str(x).strip() for x in ans if x is not None)
    return str(ans).strip()


class OlympiadBenchMath(Task):
    name = "olympiadbench_math"

    def load(self, n: int = -1, subset: list[int] | None = None) -> list[Problem]:
        """Load OlympiadBench-math problems (head-slice n, default 500). If
        `subset` is provided, return only those indices and ignore `n`."""
        from datasets import load_dataset  # type: ignore

        config_attempts = ["OE_TO_maths_en_COMP", "en_text_only_math"]
        ds = None
        used_config = None
        last_err: Exception | None = None
        for cfg in config_attempts:
            try:
                ds = load_dataset("Hothan/OlympiadBench", cfg, split="train")
                used_config = cfg
                break
            except Exception as e:
                last_err = e
                logger.warning(f"OlympiadBench config '{cfg}' failed: {e}")
        if ds is None:
            raise RuntimeError(
                f"Could not load Hothan/OlympiadBench with any of {config_attempts}; "
                f"last error: {last_err}"
            )
        logger.info(f"Loaded OlympiadBench with config '{used_config}'")

        # Cap to first 500 (head-slice) before applying subset/n.
        head_cap = 500
        rows = []
        for i, row in enumerate(ds):
            if i >= head_cap:
                break
            rows.append((i, row))

        subset_set = set(subset) if subset is not None else None
        problems: list[Problem] = []
        for i, row in rows:
            if subset_set is not None:
                if i not in subset_set:
                    continue
            elif 0 <= n <= i:
                break

            # OlympiadBench schema: 'question' (str), 'final_answer' (list[str] or str).
            question = (
                row.get("question")
                or row.get("problem")
                or row.get("Problem")
            )
            answer_raw = (
                row.get("final_answer")
                if "final_answer" in row
                else row.get("answer")
            )
            if question is None or answer_raw is None:
                logger.warning(f"olympiad-math-{i}: missing question or answer, skipping")
                continue
            answer = _normalize_answer(answer_raw)
            problems.append(
                Problem(
                    problem_id=f"olympiad-math-{i}",
                    question=str(question),
                    answer=answer,
                    metadata={
                        "subject": row.get("subject"),
                        "level": row.get("level"),
                        "config": used_config,
                        "raw_answer": answer_raw,
                    },
                )
            )
        logger.info(f"Loaded {len(problems)} problems from OlympiadBench-math")
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
