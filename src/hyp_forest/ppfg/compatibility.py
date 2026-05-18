"""Compatibility scoring for fragment-to-target matching.

The original DCA proposal uses 1 - cos(embed(fragment_last), embed(target_last)) so that
*more diverse* fragments score higher (the rationale: a fragment offering reasoning the target
chain lacks is more useful than redundant content).

A subtle issue: pure diversity reward can mismatch reasoning that's actually irrelevant. We
keep the diversity formulation as the proposal specifies, but provide a `match` mode
(cos similarity directly) for ablation.
"""

from __future__ import annotations
import logging
from typing import Sequence

import numpy as np

from hyp_forest.chains.chain import Chain
from hyp_forest.ppfg.extractor import Fragment

logger = logging.getLogger(__name__)


class CompatibilityScorer:
    """Wraps a sentence embedder and computes fragment-to-chain compatibility."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        mode: str = "diversity",   # "diversity" (1 - cos, proposal default) or "match" (cos)
    ):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.mode = mode

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.encoder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    def score(self, fragment: Fragment, target: Chain) -> float:
        """Compatibility of fragment with target chain. Higher == better."""
        if not target.steps:
            # Empty target — any fragment is maximally compatible-by-novelty
            return 1.0
        frag_text = fragment.steps[-1]
        target_text = target.steps[-1]
        embeddings = self._embed([frag_text, target_text])
        cos_sim = float(np.dot(embeddings[0], embeddings[1]))
        if self.mode == "diversity":
            return 1.0 - cos_sim
        elif self.mode == "match":
            return cos_sim
        else:
            raise ValueError(f"Unknown compat mode: {self.mode}")

    def score_all(self, fragment: Fragment, targets: list[Chain]) -> list[float]:
        """Score one fragment against many targets — useful for picking the best target."""
        if not targets:
            return []
        frag_text = fragment.steps[-1]
        target_texts = [t.steps[-1] if t.steps else "" for t in targets]
        all_text = [frag_text] + target_texts
        embeddings = self._embed(all_text)
        frag_emb = embeddings[0]
        scores = []
        for i, t_emb in enumerate(embeddings[1:]):
            cos_sim = float(np.dot(frag_emb, t_emb))
            scores.append(1.0 - cos_sim if self.mode == "diversity" else cos_sim)
        return scores
