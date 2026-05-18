"""MMLU-Pro (Law + Biology + Philosophy) task.

Loaded from HuggingFace: TIGER-Lab/MMLU-Pro, split 'test'. We filter to
the three categories {law, biology, philosophy} and stratified-sample
~167 per subset (or 200/150/150 when one is smaller) to a total of 500.
Stratification uses seed=42 for reproducibility.

MMLU-Pro is a 10-choice MCQ (some questions have fewer, but the label
space is A..J). Answer extraction tries 'the answer is X' first, then
the last A..J letter in the candidate string.
"""

from __future__ import annotations
import logging
import random
import re
from typing import Any

from hyp_forest.tasks.base import Problem, Task
from hyp_forest.utils.prompts import build_mmlu_pro_prompt

logger = logging.getLogger(__name__)

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
_LETTER_SET = set(_LETTERS)

_ANSWER_IS_RE = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*[:=]?)\s*\(?([A-J])\b",
    re.IGNORECASE,
)
_LETTER_TOKEN_RE = re.compile(r"\b([A-J])\b")
_LEADING_LETTER_RE = re.compile(r"^\s*\(?([A-J])[).:]?", re.IGNORECASE)


def _extract_letter(text: str) -> str | None:
    """Pull A..J out of an answer string. Order:
    1. 'the answer is X' / 'final answer: X' regex,
    2. lone leading 'A)' / '(A)' / 'A.' style,
    3. last A..J token in the string.
    """
    if text is None:
        return None
    s = text.strip()
    m = _ANSWER_IS_RE.search(s)
    if m:
        c = m.group(1).upper()
        if c in _LETTER_SET:
            return c
    su = s.upper()
    if su in _LETTER_SET:
        return su
    m = _LEADING_LETTER_RE.match(su)
    if m:
        c = m.group(1).upper()
        if c in _LETTER_SET:
            return c
    matches = _LETTER_TOKEN_RE.findall(su)
    if matches:
        return matches[-1]
    return None


class MMLUPro(Task):
    name = "mmlu_pro"

    def __init__(self, stratify_seed: int = 42, total_n: int = 500):
        self._stratify_seed = stratify_seed
        self._total_n = total_n
        self._categories = ("law", "biology", "philosophy")

    def load(self, n: int = -1, subset: list[int] | None = None) -> list[Problem]:
        """Load a stratified subset of MMLU-Pro across law/biology/philosophy
        (default 500 total, ~167 per subset; or 200/150/150 when one is
        smaller). If `subset` is provided, it indexes into the stratified
        problem list and `n` is ignored. Otherwise `n` head-slices the
        stratified list (n=-1 = full stratified list)."""
        # Direct parquet read (load_dataset's offline path fails on a CC compute
        # node even when the parquet is fully cached; verified Day-13 fix).
        import glob, os
        import pandas as pd  # type: ignore

        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        fs = sorted(glob.glob(
            f"{hf_home}/hub/datasets--TIGER-Lab--MMLU-Pro/snapshots/*/data/test-*.parquet"
        ))
        if not fs:
            raise RuntimeError(
                f"MMLU-Pro test parquet not found under "
                f"{hf_home}/hub/datasets--TIGER-Lab--MMLU-Pro/snapshots/*/data; "
                f"run `huggingface-cli download TIGER-Lab/MMLU-Pro --repo-type dataset` on the login node"
            )
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        logger.info(f"MMLU-Pro raw rows (test split): {len(df)} from {len(fs)} parquet file(s)")

        # Bucket by category (lowercased; MMLU-Pro stores 'category' as proper-case).
        buckets: dict[str, list[tuple[int, dict]]] = {c: [] for c in self._categories}
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            cat = str(row.get("category", "")).strip().lower()
            if cat in buckets:
                buckets[cat].append((i, row))

        for cat in self._categories:
            logger.info(f"MMLU-Pro bucket '{cat}': {len(buckets[cat])} rows")

        # Stratified sampling. Target ~equal counts. If a bucket is short,
        # take all of it and reallocate the deficit across the others.
        target = self._total_n
        n_cats = len(self._categories)
        base_per = target // n_cats
        sizes = {cat: len(buckets[cat]) for cat in self._categories}

        # Two-pass allocation: shorts get capped to their size, remainder is
        # split as evenly as possible across the remaining (longer) buckets.
        allocations: dict[str, int] = {}
        remaining = target
        long_cats: list[str] = []
        for cat in self._categories:
            if sizes[cat] <= base_per:
                allocations[cat] = sizes[cat]
                remaining -= sizes[cat]
            else:
                long_cats.append(cat)
        if long_cats:
            per_long = remaining // len(long_cats)
            leftover = remaining - per_long * len(long_cats)
            for j, cat in enumerate(long_cats):
                take = per_long + (1 if j < leftover else 0)
                allocations[cat] = min(sizes[cat], take)
        # Final sanity: clamp to bucket sizes.
        for cat in self._categories:
            allocations[cat] = min(allocations[cat], sizes[cat])

        logger.info(f"MMLU-Pro allocations (target total {target}): {allocations}")

        rng = random.Random(self._stratify_seed)
        sampled: list[tuple[int, Any, str]] = []  # (orig_idx, row, category)
        for cat in self._categories:
            pool = buckets[cat]
            k = allocations[cat]
            if k >= len(pool):
                picks = pool
            else:
                picks = rng.sample(pool, k)
            for orig_idx, row in picks:
                sampled.append((orig_idx, row, cat))

        # Deterministic sort by (category, orig_idx) so problem_id is stable
        # across runs with the same seed.
        sampled.sort(key=lambda t: (t[2], t[0]))

        subset_set = set(subset) if subset is not None else None
        problems: list[Problem] = []
        for i, (orig_idx, row, cat) in enumerate(sampled):
            if subset_set is not None:
                if i not in subset_set:
                    continue
            elif 0 <= n <= i:
                break

            # MMLU-Pro fields: 'question' (str), 'options' (list[str] / np.ndarray, up to 10),
            # 'answer' (letter A..J), 'answer_index' (int), 'category' (str).
            # Note: pandas iloc.to_dict() returns numpy arrays for list-typed columns;
            # use `is None` / explicit length checks rather than truthiness `or`.
            q_val = row.get("question")
            if q_val is None:
                q_val = row.get("problem")
            question = q_val
            opt_val = row.get("options")
            choices = opt_val if opt_val is not None else row.get("choices")
            if choices is None:
                choices = []
            gold_letter = row.get("answer")
            if (gold_letter is None or gold_letter == "") and "answer_index" in row:
                ai = row["answer_index"]
                if isinstance(ai, (int,)) and 0 <= int(ai) < len(_LETTERS):
                    gold_letter = _LETTERS[int(ai)]
            # numpy-array-safe length check
            try:
                n_choices = len(choices)
            except TypeError:
                n_choices = 0
            if question is None or n_choices == 0 or gold_letter is None:
                logger.warning(
                    f"mmlu-pro-{i}: missing question/options/answer "
                    f"(cat={cat}, orig_idx={orig_idx}), skipping"
                )
                continue
            gold_letter = str(gold_letter).strip().upper()
            if gold_letter not in _LETTER_SET:
                logger.warning(
                    f"mmlu-pro-{i}: gold letter '{gold_letter}' out of A-J "
                    f"(cat={cat}, orig_idx={orig_idx}), skipping"
                )
                continue
            choices_list = [str(c) for c in choices]
            problems.append(
                Problem(
                    problem_id=f"mmlu-pro-{i}",
                    question=str(question),
                    answer=gold_letter,
                    metadata={
                        "choices": choices_list,
                        "category": cat,
                        "orig_idx": orig_idx,
                        # NOTE: do NOT carry the raw pandas row here — `options`
                        # is a numpy ndarray that breaks json.dump at trajectory
                        # serialization time. Lost trajectories.json on 18 cells
                        # Day-13 before this fix.
                    },
                )
            )
        logger.info(
            f"Loaded {len(problems)} problems from MMLU-Pro "
            f"(categories={list(self._categories)}, seed={self._stratify_seed})"
        )
        return problems

    def build_prompt(self, problem: Problem, lm) -> str:
        choices = problem.metadata["choices"]
        return build_mmlu_pro_prompt(problem.question, choices, lm)

    def is_correct(self, candidate: str | None, problem: Problem) -> bool:
        if candidate is None:
            return False
        cand = _extract_letter(candidate)
        return cand == problem.answer
