"""When-to-deploy policy for PPFG.

Decision: when a chain is freshly pruned, do we just drop it (PRUNE) or extract a fragment
and inject it into a sibling chain (EXTRACT_PRUNE)?

Heuristic rules (per the proposal §3.3 and operationalized for V1):
    (a) Fragment quality — only extract if a qualifying fragment exists (k* >= k_min,
        mean PRM > τ_extract). Handled by FragmentExtractor.
    (b) Complementarity — only inject if compat score > compat_min (target gains genuine
        novelty, not redundant content). 'compat' rule.
    (c) Target stagnation — only inject if at least one target chain is stagnating.
        'stagnation' rule.

Four ablation modes are configurable:
    - 'compat': pick target with highest compat score, regardless of stagnation
    - 'stagnation': pick stagnating target with highest compat score (v1 stagnation rule
      keyed on PRM-range flatness in chain.is_stagnating)
    - 'stagnation_compound': five-gate compound stagnation rule (Day-6 spec); only
      candidates passing all five gates are eligible, then compat-scored among the survivors
    - 'random': pick random active target (ablation control)

A learned policy replacing this heuristic is future work.
"""

from __future__ import annotations
import logging
import random
import statistics
from dataclasses import dataclass

from hyp_forest.chains.chain import Chain
from hyp_forest.chains.population import Population
from hyp_forest.ppfg.extractor import Fragment, FragmentExtractor
from hyp_forest.ppfg.compatibility import CompatibilityScorer
from hyp_forest.ppfg.injector import FragmentInjector

logger = logging.getLogger(__name__)


@dataclass
class PPFGPolicyConfig:
    extract_threshold: float = 0.6
    k_min: int = 2
    k_max: int = 8
    compat_min: float = 0.1                 # require fragment offers genuine novelty
    injection_rule: str = "compat"          # 'compat' | 'stagnation' | 'stagnation_compound' | 'random'
    max_injections_per_chain: int = 2       # cap to avoid context bloat
    seed: int = 42
    # Smoke β: anti-redundancy gate — reject injection if best target's compat_score
    # is below this threshold (i.e., fragment is too similar to anything we could
    # graft it onto). Default 0.0 preserves prior behavior (only `compat_min` applies).
    min_compat_threshold: float = 0.0
    # Smoke γ: injection-text rendering mode. 'bracket_bullet' is the original
    # multi-line bracketed bullet block; 'last_step_inline' renders only the
    # highest-PRM (last) fragment step as a single-line inline note.
    injection_format: str = "bracket_bullet"
    # Compound-gate (Day-6 spec) parameters; used only when injection_rule == 'stagnation_compound'.
    prm_low_t: float = 0.60       # gate 1: mean PRM over last 3 steps must be below this
    prm_flat_t: float = 0.05      # gate 2: PRM range over last 3 steps must be below this
    headroom_budget: int = 4      # gate 3: at least this many steps remaining before max_steps
    liveness_min: int = 3         # gate 5: at least this many chains active including target
    # Reviewer-#2 E2 (Day-20 add-on): when True, every injection has a brief
    # self-check directive appended; the target's next decoded step is the
    # self-check, the one after is the actual continuation. Lightweight
    # version of separate-solver-pass re-attention (Prediction-1 in §6).
    reattention_pass: bool = False


class PPFGPolicy:
    """Glues extractor + compat scorer + injector with the when-to-deploy logic."""

    def __init__(
        self,
        config: PPFGPolicyConfig,
        compat_scorer: CompatibilityScorer,
    ):
        self.config = config
        self.extractor = FragmentExtractor(
            extract_threshold=config.extract_threshold,
            k_min=config.k_min,
            k_max=config.k_max,
        )
        self.compat = compat_scorer
        self.injector = FragmentInjector(
            injection_format=config.injection_format,
            reattention_pass=config.reattention_pass,
        )
        self._rng = random.Random(config.seed)

    def run_hook(self, population: Population, newly_pruned: list[Chain]) -> None:
        """Population.advance calls this after status updates. Mutates target chains in place."""
        if not newly_pruned:
            return

        candidate_targets = [
            c for c in population.active_chains
            if len(c.injected_fragments) < self.config.max_injections_per_chain
        ]
        if not candidate_targets:
            return

        # Extract a fragment from each freshly pruned chain
        for src in newly_pruned:
            fragment = self.extractor.extract(src)
            if fragment is None:
                continue

            # Pick target according to the configured rule
            target = self._select_target(fragment, candidate_targets, population)
            if target is None:
                continue

            # Compute compat for the chosen target (used in record-keeping and threshold gate)
            compat_score = self.compat.score(fragment, target)
            if compat_score < self.config.compat_min:
                logger.debug(
                    f"Skipping injection from chain {src.chain_id}: "
                    f"compat={compat_score:.3f} below min={self.config.compat_min}"
                )
                continue
            # Smoke β anti-redundancy gate: skip when the best available target is
            # still too similar to the fragment (compat_score below threshold).
            if compat_score < self.config.min_compat_threshold:
                logger.debug(
                    f"Skipping injection from chain {src.chain_id} (anti-redundancy): "
                    f"compat={compat_score:.3f} below min_compat_threshold="
                    f"{self.config.min_compat_threshold}"
                )
                continue

            # Inject
            record = self.injector.inject(target, fragment, compat_score)
            logger.debug(
                f"PPFG: injected from chain {src.chain_id} ({len(fragment.steps)} steps, "
                f"q={fragment.quality:.3f}) into chain {target.chain_id} "
                f"(compat={compat_score:.3f})"
            )

            # If target now has max injections, remove from candidates
            if len(target.injected_fragments) >= self.config.max_injections_per_chain:
                candidate_targets = [c for c in candidate_targets if c.chain_id != target.chain_id]
                if not candidate_targets:
                    break

    def _select_target(
        self,
        fragment: Fragment,
        candidates: list[Chain],
        population: Population,
    ) -> Chain | None:
        """Pick a target chain for this fragment based on the configured injection_rule."""
        if not candidates:
            return None

        if self.config.injection_rule == "random":
            return self._rng.choice(candidates)

        if self.config.injection_rule == "stagnation":
            stagnating = [c for c in candidates if c.is_stagnating()]
            if not stagnating:
                return None  # no stagnating target -> skip injection
            scores = self.compat.score_all(fragment, stagnating)
            return stagnating[max(range(len(scores)), key=lambda i: scores[i])]

        if self.config.injection_rule == "stagnation_compound":
            max_steps = population.config.max_steps
            active = [c for c in population.active_chains if c.prm_scores]
            survivors: list[Chain] = []
            for cand in candidates:
                peer_latest_prms = [
                    c.prm_scores[-1] for c in active if c.chain_id != cand.chain_id
                ]
                if _is_compound_stagnating(
                    cand,
                    peer_latest_prms,
                    max_steps,
                    prm_low_t=self.config.prm_low_t,
                    prm_flat_t=self.config.prm_flat_t,
                    headroom_budget=self.config.headroom_budget,
                    liveness_min=self.config.liveness_min,
                ):
                    survivors.append(cand)
            if not survivors:
                return None  # no candidate passes all five gates -> skip injection
            scores = self.compat.score_all(fragment, survivors)
            return survivors[max(range(len(scores)), key=lambda i: scores[i])]

        if self.config.injection_rule == "compat":
            scores = self.compat.score_all(fragment, candidates)
            return candidates[max(range(len(scores)), key=lambda i: scores[i])]

        raise ValueError(f"Unknown injection_rule: {self.config.injection_rule}")


def _is_compound_stagnating(
    target: Chain,
    peer_latest_prms: list[float],
    max_steps: int,
    prm_low_t: float = 0.60,
    prm_flat_t: float = 0.05,
    headroom_budget: int = 4,
    liveness_min: int = 3,
) -> bool:
    """Five-gate compound stagnation rule (Day-6 spec).

    All five gates must pass for the target to be eligible:
      1. target's last-3-step PRM mean < prm_low_t
      2. target's last-3-step PRM range < prm_flat_t
      3. max_steps - target.n_steps >= headroom_budget
      4. target.prm_scores[-1] < median(peer_latest_prms)
      5. len(peer_latest_prms) + 1 >= liveness_min

    `peer_latest_prms` should contain the latest PRM of every currently-active chain
    OTHER than `target` (i.e., excluding target itself).
    """
    if len(target.prm_scores) < 3:
        return False
    recent = target.prm_scores[-3:]
    # Gate 1
    if (sum(recent) / 3.0) >= prm_low_t:
        return False
    # Gate 2
    if (max(recent) - min(recent)) >= prm_flat_t:
        return False
    # Gate 3
    if (max_steps - target.n_steps) < headroom_budget:
        return False
    # Gate 5 — liveness check (must include peers; without peers, gate-4 is undefined)
    if (len(peer_latest_prms) + 1) < liveness_min:
        return False
    if not peer_latest_prms:
        # No peers means gate-4 (peer-relative) is undefined; conservatively reject.
        return False
    # Gate 4
    if target.prm_scores[-1] >= statistics.median(peer_latest_prms):
        return False
    return True
