"""Best-of-N baseline.

Generates N full CoT completions, scores each one's final-step PRM (or mean-step PRM),
and returns the highest-scoring completion. No early pruning, no interaction between chains.

Used to compare against compute-matched PPFG: same total tokens generated, but no
adaptive allocation or fragment grafting.
"""

from __future__ import annotations
import logging

from hyp_forest.chains.chain import Chain, ChainStatus, detect_final_answer
from hyp_forest.chains.population import Population, PopulationConfig
from hyp_forest.models.base_lm import BaseLM
from hyp_forest.models.prm import ProcessRewardModel

logger = logging.getLogger(__name__)


def run_best_of_n(
    problem: str,
    base_prompt: str,
    lm: BaseLM,
    prm: ProcessRewardModel,
    config: PopulationConfig,
    bon_n: int = 16,
    max_tokens_per_chain: int = 2048,
    seeds: list[int] | None = None,
) -> Population:
    """Generate `bon_n` independent full completions and store as a Population."""
    if seeds is None:
        seeds = list(range(bon_n))

    # Build N copies of the base prompt
    prompts = [base_prompt + "Step 1:" for _ in range(bon_n)]
    completions = lm.generate_complete(
        prompts,
        temperature=config.temperature, top_p=config.top_p,
        max_tokens=max_tokens_per_chain, seeds=seeds,
    )

    # Build a Population with bon_n chains, each holding one full completion split by step separator
    pop_cfg = PopulationConfig(
        n_chains=bon_n, prune_threshold=config.prune_threshold,
        promote_on_complete=config.promote_on_complete, max_steps=config.max_steps,
        step_separator=config.step_separator, temperature=config.temperature,
        top_p=config.top_p, max_step_tokens=config.max_step_tokens,
    )
    pop = Population(problem=problem, base_prompt=base_prompt, config=pop_cfg, seeds=seeds)

    for c, completion in zip(pop.chains, completions):
        # Split on step separator and clean
        raw_steps = [s.strip() for s in completion.split(config.step_separator) if s.strip()]
        # The first piece will likely include the "Step 1:" prefix we added; trim
        cleaned = []
        for i, s in enumerate(raw_steps):
            if s.lower().startswith(f"step {i + 1}:"):
                s = s[len(f"step {i + 1}:"):].strip()
            cleaned.append(s)
            if len(cleaned) >= config.max_steps:
                break
        c.steps = cleaned
        c.n_tokens_generated = len(lm.tokenizer.encode(completion))
        # Detect final answer
        for s in c.steps:
            ans = detect_final_answer(s)
            if ans:
                c.final_answer = ans
                break
        c.status = ChainStatus.PROMOTED if c.final_answer else ChainStatus.HIT_MAX_STEPS

    # Score all chains' steps with PRM
    problems_list = [c.problem for c in pop.chains]
    all_steps = [c.steps for c in pop.chains]
    all_scores = prm.score_steps_batched(problems_list, all_steps)
    for c, scores in zip(pop.chains, all_scores):
        c.prm_scores = list(scores)

    return pop
