"""Base task abstraction.

A Task knows how to:
    1. Load problems (load() -> list[Problem])
    2. Build the prompt for the LM (build_prompt(problem) -> str)
    3. Decide whether a candidate answer is correct (is_correct(candidate, problem) -> bool)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Problem:
    """One problem: question text + ground-truth answer + optional metadata."""
    problem_id: str
    question: str
    answer: str            # canonical ground-truth answer
    metadata: dict | None = None


class Task(ABC):
    name: str

    @abstractmethod
    def load(self, n: int = -1, subset: list[int] | None = None) -> list[Problem]:
        """Load problems. If `subset` is provided, return only those indices
        (and ignore `n`). Otherwise return the first `n` problems; n=-1 means all."""
        ...

    @abstractmethod
    def build_prompt(self, problem: Problem, lm) -> str:
        """Build the chat-formatted prompt the LM will see."""
        ...

    @abstractmethod
    def is_correct(self, candidate: str | None, problem: Problem) -> bool:
        """True if the candidate answer matches the problem's ground truth."""
        ...
