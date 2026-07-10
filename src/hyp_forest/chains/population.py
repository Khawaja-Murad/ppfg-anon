"""Population manager — the core orchestration loop.

A `Population` holds N parallel chains for one problem and advances them step-by-step.
Each step:
    1. Build prompts for all ACTIVE chains (Chain.build_prompt)
    2. Generate next step in batch via BaseLM.generate_steps
    3. Score new steps with the PRM
    4. Apply status updates (prune below threshold, promote on final answer, hit max steps)
    5. Run optional PPFG hook to extract fragments from newly-pruned chains and inject into surviving ones
    6. Repeat until all chains terminal or max_steps reached
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Callable, Protocol, TYPE_CHECKING

from hyp_forest.chains.chain import Chain, ChainStatus, detect_final_answer

if TYPE_CHECKING:
    # These imports pull in torch / vLLM / transformers — guarded so the chain logic
    # remains importable in CPU-only test environments.
    from hyp_forest.models.base_lm import BaseLM
    from hyp_forest.models.prm import ProcessRewardModel

logger = logging.getLogger(__name__)


class PPFGHook(Protocol):
    """Callable invoked after each generation step. Sees all chains, mutates them in place.

    Implementations: ppfg.policy.PPFGPolicy.run_hook (for PPFG runs), or a no-op for baselines.
    """
    def __call__(
        self,
        population: "Population",
        newly_pruned: list[Chain],
    ) -> None: ...


def _noop_hook(population: "Population", newly_pruned: list[Chain]) -> None:
    return


@dataclass
class PopulationConfig:
    n_chains: int = 8
    prune_threshold: float = 0.4       # PRM below this -> chain pruned
    promote_on_complete: bool = True   # detect_final_answer hit -> promoted
    max_steps: int = 16
    step_separator: str = "\n\n"
    temperature: float = 0.8
    top_p: float = 0.95
    max_step_tokens: int = 256
    post_injection_prune_immunity_steps: int = 0
    # If >0, a chain that received an injection is exempt from PRM-threshold
    # pruning for this many steps after the injection (still subject to
    # promote-on-complete and max-steps). 0 (default) reproduces prior
    # behavior exactly. Added for the ARR rebuttal PRM-decoupled check:
    # does suppressing the *pruning consequence* of the post-injection PRM
    # dip change the injected chain's eventual correctness, separating the
    # scorer's punitive reaction from the content's own effect.


class Population:
    """Manages N parallel CoT chains for one problem."""

    def __init__(
        self,
        problem: str,
        base_prompt: str,
        config: PopulationConfig,
        seeds: list[int] | None = None,
    ):
        self.problem = problem
        self.config = config
        if seeds is None:
            seeds = list(range(config.n_chains))
        assert len(seeds) == config.n_chains, "seeds length must match n_chains"
        self.chains: list[Chain] = [
            Chain(chain_id=i, problem=problem, base_prompt=base_prompt, seed=seeds[i])
            for i in range(config.n_chains)
        ]
        self.step_idx: int = 0

    @property
    def active_chains(self) -> list[Chain]:
        return [c for c in self.chains if c.is_active]

    @property
    def terminal_chains(self) -> list[Chain]:
        return [c for c in self.chains if not c.is_active]

    @property
    def is_done(self) -> bool:
        return len(self.active_chains) == 0 or self.step_idx >= self.config.max_steps

    def _is_prune_immune(self, c: Chain) -> bool:
        """True if c is within its post-injection prune-immunity window."""
        window = self.config.post_injection_prune_immunity_steps
        if window <= 0 or not c.injected_fragments:
            return False
        last_inj_step = max(f.injected_at_step for f in c.injected_fragments)
        steps_since_injection = c.n_steps - last_inj_step
        return 0 <= steps_since_injection <= window

    def advance(
        self,
        lm: "BaseLM",
        prm: "ProcessRewardModel",
        ppfg_hook: PPFGHook = _noop_hook,
    ) -> list[Chain]:
        """Advance the entire active population by one step. Returns newly-pruned chains."""
        active = self.active_chains
        if not active:
            return []

        # 1. Build prompts
        prompts = [c.build_prompt(self.config.step_separator) for c in active]
        seeds = [c.seed for c in active]

        # 2. Generate one step for each active chain in batch
        outputs = lm.generate_steps(
            prompts,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_step_tokens=self.config.max_step_tokens,
            step_separator=self.config.step_separator,
            seeds=seeds,
        )
        for c, out in zip(active, outputs):
            c.steps.append(out.text)
            c.n_tokens_generated += out.n_tokens

        # 3. Score new steps with PRM (batched per chain — each chain's full step list)
        problems = [c.problem for c in active]
        all_steps = [c.steps for c in active]
        all_scores = prm.score_steps_batched(problems, all_steps)
        for c, scores in zip(active, all_scores):
            # We only need the score for the newly-added step, but we accept the full
            # list and overwrite to keep things simple and ensure consistency.
            c.prm_scores = list(scores)

        # 4. Status updates
        newly_pruned: list[Chain] = []
        for c in active:
            latest = c.latest_prm_score
            # Final-answer detection takes priority over pruning
            answer = detect_final_answer(c.steps[-1])
            if answer is not None and self.config.promote_on_complete:
                c.final_answer = answer
                c.status = ChainStatus.PROMOTED
                continue
            # Prune on low PRM (unless within post-injection immunity window)
            if latest is not None and latest < self.config.prune_threshold:
                if self._is_prune_immune(c):
                    continue
                c.status = ChainStatus.PRUNED
                newly_pruned.append(c)
                continue
            # Hit max steps
            if c.n_steps >= self.config.max_steps:
                c.status = ChainStatus.HIT_MAX_STEPS
                continue

        # 5. Run PPFG hook (no-op for baselines; PPFG runs use ppfg.policy.PPFGPolicy)
        ppfg_hook(self, newly_pruned)

        self.step_idx += 1
        return newly_pruned

    def run_to_completion(
        self,
        lm: "BaseLM",
        prm: "ProcessRewardModel",
        ppfg_hook: PPFGHook = _noop_hook,
    ) -> None:
        """Drive the population until done."""
        while not self.is_done:
            self.advance(lm, prm, ppfg_hook)
        # Convert any leftover active chains to HIT_MAX_STEPS
        for c in self.active_chains:
            c.status = ChainStatus.HIT_MAX_STEPS

    def best_chain_by_prm(self) -> Chain:
        """Return chain with highest mean PRM among PROMOTED, falling back to all chains."""
        candidates = [c for c in self.chains if c.status == ChainStatus.PROMOTED]
        if not candidates:
            candidates = self.chains
        return max(candidates, key=lambda c: c.mean_prm_score)

    def all_final_answers(self) -> list[tuple[Chain, str | None]]:
        """For Pass@k evaluation: each chain's final answer (or None if it never produced one)."""
        results = []
        for c in self.chains:
            if c.final_answer is not None:
                results.append((c, c.final_answer))
            else:
                # Try to detect from the last step
                if c.steps:
                    answer = detect_final_answer(c.steps[-1])
                    results.append((c, answer))
                else:
                    results.append((c, None))
        return results
