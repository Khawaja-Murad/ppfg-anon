"""Diversity metrics — the headline observable.

Three metrics aligned with the diversity-collapse literature:
    - semantic_diversity: mean pairwise (1 - cos) over final-step embeddings of completed chains
    - answer_mode_rate: fraction of problems where >50% of chains share the same final answer
    - distinct_n: ratio of unique n-grams to total n-grams across all chains' full reasoning
"""

from __future__ import annotations
from collections import Counter
from typing import Sequence

import numpy as np

from hyp_forest.chains.population import Population
from hyp_forest.tasks.base import Problem, Task


def semantic_diversity(populations: list[Population], encoder) -> float:
    """Mean over problems of mean pairwise (1 - cos) between final-step embeddings."""
    per_problem = []
    for pop in populations:
        finals = [c.steps[-1] for c in pop.chains if c.steps]
        if len(finals) < 2:
            continue
        embeddings = encoder.encode(finals, normalize_embeddings=True, show_progress_bar=False)
        sims = np.dot(embeddings, embeddings.T)
        # Mean off-diagonal cosine, converted to distance
        n = sims.shape[0]
        off_diag = sims[~np.eye(n, dtype=bool)]
        per_problem.append(float(1.0 - off_diag.mean()))
    return float(np.mean(per_problem)) if per_problem else 0.0


def answer_mode_rate(
    populations: list[Population],
    problems: list[Problem],
    task: Task,
    threshold: float = 0.5,
) -> float:
    """Fraction of problems where >threshold of chains' final answers are the same.

    High mode rate == diversity collapse.
    """
    collapsed = 0
    n = 0
    for pop, prob in zip(populations, problems):
        answers = [a for _, a in pop.all_final_answers() if a is not None]
        if not answers:
            continue
        # Normalize answers to canonical form using task-specific logic where possible
        # For MATH500 use raw string; for GPQA use letter
        counts = Counter(answers)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(answers) > threshold:
            collapsed += 1
        n += 1
    return collapsed / n if n else 0.0


def distinct_n(populations: list[Population], n: int = 4) -> float:
    """Distinct-n: ratio of unique n-grams to total n-grams across all chains."""
    total = 0
    unique: set[tuple[str, ...]] = set()
    for pop in populations:
        for c in pop.chains:
            text = " ".join(c.steps)
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                total += 1
                unique.add(gram)
    return len(unique) / total if total else 0.0


def all_diversity_metrics(
    populations: list[Population],
    problems: list[Problem],
    task: Task,
    encoder=None,
) -> dict[str, float]:
    """Compute all diversity metrics. encoder must be provided for semantic_diversity."""
    out = {
        "answer_mode_rate": answer_mode_rate(populations, problems, task),
        "distinct_4": distinct_n(populations, 4),
    }
    if encoder is not None:
        out["semantic_diversity"] = semantic_diversity(populations, encoder)
    return out
