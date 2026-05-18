"""Pass@k evaluation.

Standard unbiased Pass@k estimator from Chen et al. 'Evaluating Large Language Models
Trained on Code' (2021):

    pass@k = 1 - C(n - c, k) / C(n, k)

where n = total samples, c = number correct.

Returns a curve over k values. Compute-matched comparison: aggregate over all problems'
chains using a fixed total token budget to ensure baselines and PPFG are compared on equal
compute footing.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

from hyp_forest.chains.population import Population
from hyp_forest.tasks.base import Problem, Task


@dataclass
class PassAtKResult:
    k: int
    pass_at_k: float
    n_problems: int
    n_correct_per_problem: list[int]
    n_samples_per_problem: list[int]


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k for n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def evaluate_pass_at_k(
    populations: list[Population],
    problems: list[Problem],
    task: Task,
    k_values: list[int],
) -> dict[int, PassAtKResult]:
    """Evaluate Pass@k across a list of completed populations.

    Each population corresponds to one problem; chains in the population are the n samples.
    """
    assert len(populations) == len(problems), "populations and problems must align"
    results: dict[int, PassAtKResult] = {}

    n_correct_per_problem = []
    n_samples_per_problem = []
    for pop, prob in zip(populations, problems):
        candidates = [ans for _, ans in pop.all_final_answers()]
        n = len(candidates)
        c = sum(1 for cand in candidates if task.is_correct(cand, prob))
        n_correct_per_problem.append(c)
        n_samples_per_problem.append(n)

    for k in k_values:
        per_problem = []
        for n, c in zip(n_samples_per_problem, n_correct_per_problem):
            if n < k:
                # Pad: if we have fewer than k samples, this is undefined. Skip.
                continue
            per_problem.append(_pass_at_k_unbiased(n, c, k))
        if not per_problem:
            results[k] = PassAtKResult(
                k=k, pass_at_k=float("nan"),
                n_problems=0,
                n_correct_per_problem=n_correct_per_problem,
                n_samples_per_problem=n_samples_per_problem,
            )
        else:
            results[k] = PassAtKResult(
                k=k, pass_at_k=float(np.mean(per_problem)),
                n_problems=len(per_problem),
                n_correct_per_problem=n_correct_per_problem,
                n_samples_per_problem=n_samples_per_problem,
            )
    return results


def total_tokens(populations: list[Population]) -> int:
    """Total tokens generated across all populations — used for compute-matched comparison."""
    return sum(c.n_tokens_generated for pop in populations for c in pop.chains)
