"""GPQA-Diamond — graduate-level science multiple-choice benchmark.

Each question has 4 choices (A/B/C/D); the model must select one. The PRM (Math-Shepherd)
is math-tuned but Skywork-PRM-Qwen handles general reasoning better — switch via config
if you find Math-Shepherd's GPQA scores degenerate.

Loaded from HF: Idavidrein/gpqa, config 'gpqa_diamond'.
"""

from __future__ import annotations
import logging
import random
import re
from typing import Any

from hyp_forest.tasks.base import Problem, Task
from hyp_forest.utils.prompts import build_gpqa_prompt

logger = logging.getLogger(__name__)


class GPQADiamond(Task):
    name = "gpqa_diamond"

    def __init__(self, shuffle_choices_seed: int = 42):
        self._shuffle_seed = shuffle_choices_seed

    def load(self, n: int = -1, subset: list[int] | None = None) -> list[Problem]:
        """If `subset` is provided, return only those dataset indices (and
        ignore `n`); otherwise head-slice `n`.

        Direct CSV read: Idavidrein/gpqa ships gpqa_diamond.csv in the snapshot
        root. load_dataset's offline path fails to discover this layout, so we
        read the CSV directly (Day-13 fix; verified offline)."""
        import glob, os
        import pandas as pd  # type: ignore

        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        csv_paths = glob.glob(
            f"{hf_home}/hub/datasets--Idavidrein--gpqa/snapshots/*/gpqa_diamond.csv"
        )
        if not csv_paths:
            raise RuntimeError(
                f"GPQA-Diamond CSV not found under "
                f"{hf_home}/hub/datasets--Idavidrein--gpqa/snapshots/*/gpqa_diamond.csv; "
                f"run `huggingface-cli download Idavidrein/gpqa --repo-type dataset` on the login node "
                f"after requesting access at https://huggingface.co/datasets/Idavidrein/gpqa"
            )
        df = pd.read_csv(csv_paths[0])
        rng = random.Random(self._shuffle_seed)
        subset_set = set(subset) if subset is not None else None
        problems = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            if subset_set is not None:
                if i not in subset_set:
                    continue
            elif 0 <= n <= i:
                break
            # GPQA stores correct + 3 incorrect; we need to assemble + shuffle to A/B/C/D
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            indices = list(range(4))
            rng.shuffle(indices)
            shuffled_choices = [choices[i] for i in indices]
            correct_letter = "ABCD"[indices.index(0)]
            question = row["Question"]
            problems.append(
                Problem(
                    problem_id=f"gpqa-d-{i}",
                    question=question,
                    answer=correct_letter,
                    metadata={
                        "choices": [str(c) for c in shuffled_choices],
                        "subject": str(row.get("Subdomain", "")),
                        # No "raw" pandas row — keeps metadata JSON-clean and
                        # tightens trajectories.json size.
                    },
                )
            )
        logger.info(f"Loaded {len(problems)} problems from GPQA-Diamond")
        return problems

    def build_prompt(self, problem: Problem, lm) -> str:
        choices = problem.metadata["choices"]
        return build_gpqa_prompt(problem.question, choices, lm)

    def is_correct(self, candidate: str | None, problem: Problem) -> bool:
        if candidate is None:
            return False
        cand = self._extract_letter(candidate)
        return cand == problem.answer

    @staticmethod
    def _extract_letter(text: str) -> str | None:
        """Pull A/B/C/D out of an answer string."""
        text = text.strip().upper()
        # Bare letter
        if text in {"A", "B", "C", "D"}:
            return text
        # 'A)' or '(A)' or 'A.' or 'A:'
        m = re.match(r"\(?([ABCD])[).:]?", text)
        if m:
            return m.group(1)
        # First A/B/C/D in the string
        m = re.search(r"\b([ABCD])\b", text)
        if m:
            return m.group(1)
        return None
