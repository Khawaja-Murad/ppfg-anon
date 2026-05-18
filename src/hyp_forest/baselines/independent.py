"""Independent parallel CoT baseline.

This is just a Population with no PPFG hook — N chains generate fully independently,
get pruned at threshold, no fragment exchange. Pass@k evaluated over all chains.

This is the natural baseline to demonstrate diversity collapse at the population level
under standard sampling.
"""

from __future__ import annotations
from hyp_forest.chains.population import Population, PopulationConfig
from hyp_forest.models.base_lm import BaseLM
from hyp_forest.models.prm import ProcessRewardModel


def run_independent(
    problem: str,
    base_prompt: str,
    lm: BaseLM,
    prm: ProcessRewardModel,
    config: PopulationConfig,
    seeds: list[int] | None = None,
) -> Population:
    """Run independent parallel CoT and return the completed population."""
    pop = Population(problem=problem, base_prompt=base_prompt, config=config, seeds=seeds)
    pop.run_to_completion(lm, prm)  # default no-op hook
    return pop
