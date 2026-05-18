"""Fragment extraction from pruned chains.

The DCA proposal defines fragments as the *longest contiguous prefix* of a pruned chain
whose PRM scores all exceed a threshold τ_extract. Quality is the mean PRM score over
the fragment.

Operational tightening (vs. the original proposal): we require fragment length k* >= k_min,
and we cap at k_max to avoid grafting massive blocks of context that distract the target chain.
"""

from __future__ import annotations
from dataclasses import dataclass

from hyp_forest.chains.chain import Chain


@dataclass
class Fragment:
    """An extracted fragment from a pruned chain."""
    source_chain_id: int
    steps: list[str]                  # the actual reasoning step text
    prm_scores: list[float]           # parallel PRM scores
    quality: float                    # mean PRM score over fragment
    start_step_idx: int = 0           # always 0 (we extract prefix); kept for future-proofing
    end_step_idx: int = 0             # exclusive; equal to len(steps)


class FragmentExtractor:
    """Extracts the longest contiguous high-PRM prefix from a pruned chain."""

    def __init__(
        self,
        extract_threshold: float = 0.6,   # τ_extract — only steps above this are eligible
        k_min: int = 2,                   # minimum fragment length in steps
        k_max: int = 8,                   # maximum fragment length in steps
    ):
        self.extract_threshold = extract_threshold
        self.k_min = k_min
        self.k_max = k_max

    def extract(self, chain: Chain) -> Fragment | None:
        """Return the best extractable fragment from this chain, or None if none qualifies."""
        if not chain.prm_scores:
            return None

        # Find longest contiguous prefix with all scores >= threshold
        k_star = 0
        for i, score in enumerate(chain.prm_scores):
            if score >= self.extract_threshold:
                k_star = i + 1
            else:
                break

        if k_star < self.k_min:
            return None

        k_star = min(k_star, self.k_max)
        steps = chain.steps[:k_star]
        scores = chain.prm_scores[:k_star]
        quality = sum(scores) / len(scores)

        return Fragment(
            source_chain_id=chain.chain_id,
            steps=list(steps),
            prm_scores=list(scores),
            quality=quality,
            start_step_idx=0,
            end_step_idx=k_star,
        )

    def extract_all(self, chains: list[Chain]) -> list[Fragment]:
        """Extract from every chain that qualifies."""
        out = []
        for c in chains:
            f = self.extract(c)
            if f is not None:
                out.append(f)
        return out
