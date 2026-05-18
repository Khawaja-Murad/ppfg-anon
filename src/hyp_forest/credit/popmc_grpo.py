"""PopMC-GRPO post-hoc value estimation.

This module is *not* a training algorithm. It is an analysis tool that takes saved
trajectories from a completed experiment and, for each population-level decision point
(specifically: the moment a chain was pruned, where PPFG would inject vs not), estimates
the counterfactual value of each action via K short Monte Carlo rollouts from a frozen
LM snapshot.

This gives us the credit-assignment ablation table the proposal §3.4 promises:

    A(s_t, a_t) = V(s_{t+1} | a_t) - V(s_t)

Where V is estimated by averaging K rollouts to terminal reward. We use this to argue
that PPFG injections receive systematically higher advantage than alternative actions
(PRUNE, DEEPEN) — empirically validating that the cross-chain credit assignment matters.

This is a strict subset of full PopMC-GRPO training: we only do the value-estimation half,
not the policy update. Training would take weeks; this analysis takes hours.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np

from hyp_forest.chains.chain import Chain
from hyp_forest.chains.population import Population, PopulationConfig
from hyp_forest.models.base_lm import BaseLM
from hyp_forest.models.prm import ProcessRewardModel
from hyp_forest.tasks.base import Problem, Task

logger = logging.getLogger(__name__)


@dataclass
class ValueEstimate:
    state_id: str                 # e.g., "problem-42:step-3:chain-0"
    action: str                   # 'inject' | 'prune' | 'deepen'
    value: float                  # estimated V via mean of K terminal rewards
    n_rollouts: int


@dataclass
class ActionAdvantage:
    """The advantage A(s_t, a_t) = V(s_{t+1}) - V(s_t) for one decision point."""
    problem_id: str
    decision_step: int
    chain_id: int
    action: str
    advantage: float
    v_before: float
    v_after: float


@dataclass
class PopMCGRPOConfig:
    k_rollouts: int = 8
    rollout_max_tokens: int = 512
    rollout_temperature: float = 0.8
    rollout_top_p: float = 0.95


class PopMCGRPOAnalyzer:
    """Post-hoc credit-assignment analyzer over saved populations."""

    def __init__(self, config: PopMCGRPOConfig):
        self.config = config

    def estimate_value(
        self,
        prompt: str,
        problem: Problem,
        task: Task,
        lm: BaseLM,
    ) -> float:
        """Estimate V(s) via K rollouts from the given prompt to terminal reward."""
        seeds = list(range(self.config.k_rollouts))
        prompts = [prompt] * self.config.k_rollouts
        completions = lm.generate_complete(
            prompts,
            temperature=self.config.rollout_temperature,
            top_p=self.config.rollout_top_p,
            max_tokens=self.config.rollout_max_tokens,
            seeds=seeds,
        )

        # Extract final answer from each rollout, score binary correctness
        from hyp_forest.chains.chain import detect_final_answer
        rewards = []
        for completion in completions:
            answer = None
            # Try to detect across all generated text
            for line in completion.split("\n"):
                a = detect_final_answer(line)
                if a is not None:
                    answer = a
                    break
            if answer is None:
                # Try the whole completion
                answer = detect_final_answer(completion)
            r = 1.0 if task.is_correct(answer, problem) else 0.0
            rewards.append(r)
        return float(np.mean(rewards))

    def compute_advantages_for_population(
        self,
        pop: Population,
        problem: Problem,
        task: Task,
        lm: BaseLM,
    ) -> list[ActionAdvantage]:
        """For each PPFG injection in this population, estimate its advantage vs counterfactual prune."""
        advantages: list[ActionAdvantage] = []
        for c in pop.chains:
            for inj in c.injected_fragments:
                # State BEFORE the injection: chain c's prompt at step `inj.injected_at_step`
                # without the fragment text spliced in.
                # Reconstruct by temporarily popping the injected fragment (we'll restore after).
                before_steps = list(c.steps[:inj.injected_at_step])
                after_steps = list(c.steps[:inj.injected_at_step])

                before_prompt = self._build_prompt_with_steps(c, before_steps)
                # AFTER state: same as before but with injection text inserted before next step
                after_prompt = before_prompt + inj.injection_text + f"\nStep {inj.injected_at_step + 1}:"
                before_prompt = before_prompt + f"\nStep {inj.injected_at_step + 1}:"

                v_before = self.estimate_value(before_prompt, problem, task, lm)
                v_after = self.estimate_value(after_prompt, problem, task, lm)
                adv = v_after - v_before
                advantages.append(
                    ActionAdvantage(
                        problem_id=problem.problem_id,
                        decision_step=inj.injected_at_step,
                        chain_id=c.chain_id,
                        action="inject",
                        advantage=adv,
                        v_before=v_before,
                        v_after=v_after,
                    )
                )
        return advantages

    @staticmethod
    def _build_prompt_with_steps(chain: Chain, steps: list[str]) -> str:
        """Reconstruct prompt for value estimation with a specific step prefix."""
        if not steps:
            return chain.base_prompt
        rendered = "\n\n".join(f"Step {i + 1}: {s.strip()}" for i, s in enumerate(steps))
        return chain.base_prompt + rendered + "\n\n"
