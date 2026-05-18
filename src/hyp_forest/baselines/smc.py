"""Sequential Monte Carlo (SMC) / particle-filter baseline.

At each step:
    1. Generate next step for all N chains (same as Population)
    2. Score each chain's latest step with PRM
    3. Compute weights w_i = exp(score_i / temperature) / sum_j exp(score_j / temperature)
    4. Resample N chains with replacement according to weights — chains with low PRM
       die and chains with high PRM duplicate.

This is the canonical adaptive-allocation baseline. Compared with PPFG, it does NOT do
fragment-level extraction; it only duplicates whole successful chains.

References:
- Reject, Resample, Repeat (arXiv 2603.07887)
- Particle filter for LLM reasoning (arXiv 2502.01618)
"""

from __future__ import annotations
import copy
import logging
import random
from dataclasses import dataclass

import numpy as np

from hyp_forest.chains.chain import Chain, ChainStatus, detect_final_answer
from hyp_forest.chains.population import Population, PopulationConfig
from hyp_forest.models.base_lm import BaseLM
from hyp_forest.models.prm import ProcessRewardModel

logger = logging.getLogger(__name__)


@dataclass
class SMCConfig:
    resample_temperature: float = 0.5  # softmax temperature on PRM scores for resampling
    seed: int = 42


def _resample(chains: list[Chain], weights: np.ndarray, rng: random.Random) -> list[Chain]:
    """Multinomial resampling with replacement. Returns deep-copied chains with new IDs."""
    n = len(chains)
    indices = rng.choices(range(n), weights=weights.tolist(), k=n)
    new_chains = []
    for new_id, src_idx in enumerate(indices):
        c = copy.deepcopy(chains[src_idx])
        c.chain_id = new_id
        # Reseed for diverging next steps even if we duplicated this chain
        c.seed = rng.randint(0, 2**31 - 1)
        new_chains.append(c)
    return new_chains


def run_smc(
    problem: str,
    base_prompt: str,
    lm: BaseLM,
    prm: ProcessRewardModel,
    pop_config: PopulationConfig,
    smc_config: SMCConfig,
) -> Population:
    """Run SMC with PRM weights at each step. Returns final population."""
    rng = random.Random(smc_config.seed)
    pop = Population(
        problem=problem, base_prompt=base_prompt, config=pop_config,
        seeds=[rng.randint(0, 2**31 - 1) for _ in range(pop_config.n_chains)],
    )

    while not pop.is_done:
        active = pop.active_chains
        if not active:
            break

        # Generate one step for all active chains
        prompts = [c.build_prompt(pop.config.step_separator) for c in active]
        seeds = [c.seed for c in active]
        outputs = lm.generate_steps(
            prompts,
            temperature=pop.config.temperature, top_p=pop.config.top_p,
            max_step_tokens=pop.config.max_step_tokens,
            step_separator=pop.config.step_separator,
            seeds=seeds,
        )
        for c, o in zip(active, outputs):
            c.steps.append(o.text)
            c.n_tokens_generated += o.n_tokens

        # Score all steps
        problems_list = [c.problem for c in active]
        all_steps = [c.steps for c in active]
        all_scores = prm.score_steps_batched(problems_list, all_steps)
        for c, scores in zip(active, all_scores):
            c.prm_scores = list(scores)

        # Promote any chain that produced a final answer
        survivors: list[Chain] = []
        for c in active:
            if c.steps:
                ans = detect_final_answer(c.steps[-1])
                if ans is not None:
                    c.final_answer = ans
                    c.status = ChainStatus.PROMOTED
                    continue
            if c.n_steps >= pop.config.max_steps:
                c.status = ChainStatus.HIT_MAX_STEPS
                continue
            survivors.append(c)

        if not survivors:
            pop.step_idx += 1
            continue

        # Resample survivors by PRM weight
        latest_scores = np.array([c.latest_prm_score or 0.0 for c in survivors])
        # softmax over PRM scores
        logits = latest_scores / max(smc_config.resample_temperature, 1e-6)
        weights = np.exp(logits - logits.max())
        weights = weights / weights.sum()
        resampled = _resample(survivors, weights, rng)

        # Replace the active subset of pop.chains with resampled survivors;
        # promoted/hit-max chains stay terminal
        terminal = [c for c in pop.chains if not c.is_active]
        pop.chains = terminal + resampled
        # Ensure unique chain_ids globally
        for new_id, c in enumerate(pop.chains):
            c.chain_id = new_id

        pop.step_idx += 1

    return pop
