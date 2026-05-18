"""Fragment injection into a still-decoding target chain.

The graft happens at the *next* step boundary of the target chain (i.e., the chain finishes
its current step normally, then on the next .build_prompt() call the injection text is
included right before the next 'Step N:' anchor).

Format of the injection text — chosen to be salient to the model without being overly
disruptive to its native reasoning style:

    [Insight from a related reasoning path:
     - <step 1 of fragment>
     - <step 2 of fragment>
     ... ]

We use bracket+bullet form rather than 'Step N:' to make it explicit that the fragment is
borrowed context rather than the chain's own reasoning. This avoids confusing the model
about its own step numbering.
"""

from __future__ import annotations
from hyp_forest.chains.chain import Chain, InjectedFragment
from hyp_forest.ppfg.extractor import Fragment


def format_fragment_as_injection(
    fragment: Fragment,
    injection_format: str = "bracket_bullet",
) -> str:
    """Render a fragment as the in-context block that gets spliced into the target's prompt.

    Modes:
      - 'bracket_bullet' (default): original multi-line bracketed bullet block.
      - 'last_step_inline' (Smoke γ): single-line inline note containing only the
        highest-PRM (last) fragment step. Avoids context-bloat / working-memory
        displacement from the bracket+bullet block.
      - 'empty_bracket' (V12.B confound control): same bracket-bullet wrapping
        as the default, but the bulleted content is replaced with the literal
        string '(empty)'. The fragment-quality scalar is still rendered. Used
        to disentangle whether the post-injection PRM depression reported in
        §5.8 is driven by foreign-style content or by the out-of-distribution
        bracket-format wrapping itself.
    """
    if injection_format == "last_step_inline":
        last_step = fragment.steps[-1].strip() if fragment.steps else ""
        return f"(Note from a parallel approach: {last_step})\n"
    if injection_format == "empty_bracket":
        return (
            "[Insight from a related reasoning path "
            f"(quality={fragment.quality:.2f}, length=0 steps):\n"
            "  (empty)\n"
            "]\n"
        )
    # default
    bullets = "\n".join(f"  - {s.strip()}" for s in fragment.steps)
    return (
        "[Insight from a related reasoning path "
        f"(quality={fragment.quality:.2f}, length={len(fragment.steps)} steps):\n"
        f"{bullets}\n"
        "]\n"
    )


class FragmentInjector:
    """Performs the in-place injection. Stateless; just records the InjectedFragment on the target."""

    def __init__(self, injection_format: str = "bracket_bullet"):
        self.injection_format = injection_format

    def inject(
        self,
        target: Chain,
        fragment: Fragment,
        compat_score: float,
    ) -> InjectedFragment:
        """Splice the fragment into the target's context. Returns the recorded InjectedFragment."""
        injection_text = format_fragment_as_injection(fragment, self.injection_format)
        record = InjectedFragment(
            source_chain_id=fragment.source_chain_id,
            fragment_steps=list(fragment.steps),
            fragment_prm_scores=list(fragment.prm_scores),
            quality=fragment.quality,
            compat_score=compat_score,
            injected_at_step=target.n_steps,  # injected before the next step the chain will produce
            injection_text=injection_text,
        )
        target.injected_fragments.append(record)
        return record
