"""Chain — represents one parallel CoT reasoning chain.

A Chain accumulates `steps` (list of step text), `prm_scores` (one per step), and a `status`
that tracks whether it's still being generated, has been pruned, or has been promoted as
a final candidate. PPFG operates by examining pruned chains' steps, extracting fragments,
and injecting them into still-active chains via the `injected_fragments` field.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ChainStatus(str, Enum):
    ACTIVE = "active"          # still being extended
    PRUNED = "pruned"          # killed because PRM score dropped below threshold
    PROMOTED = "promoted"      # produced final answer; halted normally
    HIT_MAX_STEPS = "hit_max"  # ran out of step budget
    EXTRACTED = "extracted"    # was pruned, but a fragment was extracted before death


@dataclass
class InjectedFragment:
    """Record of a fragment that was grafted into this chain."""
    source_chain_id: int
    fragment_steps: list[str]
    fragment_prm_scores: list[float]
    quality: float                    # mean PRM of fragment
    compat_score: float               # 1 - cos(target_last, fragment_last)
    injected_at_step: int             # how many steps this chain had when injected
    injection_text: str               # the actual text spliced into the prompt


@dataclass
class Chain:
    """One parallel CoT reasoning chain."""
    chain_id: int
    problem: str
    base_prompt: str                  # full chat-formatted prompt for the LM
    steps: list[str] = field(default_factory=list)
    prm_scores: list[float] = field(default_factory=list)
    status: ChainStatus = ChainStatus.ACTIVE
    final_answer: str | None = None
    injected_fragments: list[InjectedFragment] = field(default_factory=list)
    n_tokens_generated: int = 0       # cumulative tokens — for compute-matched analysis
    seed: int | None = None

    @property
    def is_active(self) -> bool:
        return self.status == ChainStatus.ACTIVE

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def latest_prm_score(self) -> float | None:
        return self.prm_scores[-1] if self.prm_scores else None

    @property
    def mean_prm_score(self) -> float:
        return sum(self.prm_scores) / len(self.prm_scores) if self.prm_scores else 0.0

    def is_stagnating(self, window: int = 3, threshold: float = 0.05) -> bool:
        """True if the chain's PRM has plateaued — used by injection target selection."""
        if len(self.prm_scores) < window + 1:
            return False
        recent = self.prm_scores[-window:]
        return (max(recent) - min(recent)) < threshold

    def build_prompt(self, step_separator: str = "\n\n") -> str:
        """Build the prompt for the LM to continue this chain.

        Order: [base_prompt] + [any injected fragment markers] + [previously generated steps]
        Injected fragments are spliced in chronological order of injection.
        """
        parts = [self.base_prompt]

        # Splice fragments at the step indices where they were injected.
        # We render them as a delimited block before the next step the chain produces.
        # For simplicity, inject all fragments before the first step they targeted.
        steps_with_injections = list(self.steps)
        # Build a map: step_idx_at_injection -> list of injection texts to insert BEFORE that idx
        inject_before = {}
        for inj in self.injected_fragments:
            inject_before.setdefault(inj.injected_at_step, []).append(inj.injection_text)

        rendered: list[str] = []
        for i, s in enumerate(steps_with_injections):
            if i in inject_before:
                for t in inject_before[i]:
                    rendered.append(t)
            rendered.append(f"Step {i + 1}: {s.strip()}")

        # Catch any injections that happened after the last step (about to be the next step)
        if len(steps_with_injections) in inject_before:
            for t in inject_before[len(steps_with_injections)]:
                rendered.append(t)

        if rendered:
            parts.append(step_separator.join(rendered))
            parts.append(f"\nStep {len(steps_with_injections) + 1}:")
        else:
            parts.append("Step 1:")
        return "".join(parts) if not rendered else step_separator.join(parts)


def _extract_boxed_balanced(text: str) -> str | None:
    """Find the LAST \\boxed{...} in text with balanced-brace matching."""
    needle = r"\boxed{"
    idx = text.rfind(needle)
    if idx < 0:
        return None
    start = idx + len(needle)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return None


def detect_final_answer(step_text: str) -> str | None:
    """Heuristic: if a step looks like a final answer, extract it.

    Detects '\\boxed{...}' (with nested-brace support), 'Final answer: ...',
    or 'The answer is ...'. Returns None if no match.
    """
    import re
    boxed = _extract_boxed_balanced(step_text)
    if boxed is not None:
        return boxed
    m = re.search(
        r"(?:final answer|the answer is)\s*[:\-]?\s*([^\n.]+?)(?:\.|$|\n)",
        step_text,
        re.IGNORECASE,
    )
    if m:
        captured = m.group(1).strip().rstrip(".")
        if r"\boxed" in captured:
            inner = _extract_boxed_balanced(captured)
            return inner if inner is not None else None
        return captured
    return None
