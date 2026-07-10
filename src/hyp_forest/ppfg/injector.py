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
      - 'seamless' (Reviewer-#2 E1, Day-20 add-on): renders fragment steps as
        native 'Step N: <text>' lines with NO bracket wrapper, NO quality
        label, NO 'related reasoning path' marker. The fragment appears
        indistinguishable from the target chain's own prior reasoning, which
        neutralizes the format-pollution confound: any residual PRM
        depression or Pass@k change must be attributable to the spliced
        content itself rather than to bracketed-demonstration markup.
        Step numbering is left to chain.build_prompt(), which renumbers
        the full sequence; here we emit only the inner text joined by
        the step separator.
    """
    if injection_format == "null":
        # Rebuttal (2026-07-09) matched-immunity control: emit NO spliced text
        # at all. The InjectedFragment record is still created by the injector
        # (so the chain receives the identical post-injection prune-immunity
        # window and identical candidate/max-injection bookkeeping as a real
        # graft), but zero content is added to the prompt. Comparing a run in
        # this mode against the real-graft run — both with prune-immunity on —
        # isolates the causal effect of the grafted CONTENT from the effect of
        # the extra decoding budget the immunity window grants. build_prompt()
        # skips empty injection_text, so no stray separators are rendered.
        return ""
    if injection_format == "seamless":
        # Plain step text joined by the standard step separator. No header,
        # no quality scalar, no bracket. chain.build_prompt() will renumber
        # surrounding 'Step N:' anchors so the splice appears as native
        # continuation. The trailing newline keeps spacing consistent with
        # the other modes.
        steps_text = "\n\n".join(s.strip() for s in fragment.steps)
        return steps_text + "\n"
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


_REATTENTION_DIRECTIVE = (
    "Before continuing my own reasoning, I'll briefly evaluate the borrowed "
    "insight above: does it align with the approach I've been taking on this "
    "problem, and if so, what does it add? I'll answer that in one short step, "
    "then continue normally.\n"
)


class FragmentInjector:
    """Performs the in-place injection. Stateless; just records the InjectedFragment on the target.

    Re-attention variant (Reviewer-#2 E2, Day-20 add-on): when reattention_pass is
    True, the injection text is suffixed with a brief self-check directive that
    instructs the model to evaluate the incoming fragment before continuing its
    decoding sequence. The model's next decoded step becomes the self-check; the
    step after that is the actual continuation. This implements a lightweight
    version of Prediction-1 (separate-solver-pass re-attention) without requiring
    a full second AR pass. Compute cost is one extra step per injected chain.
    """

    def __init__(
        self,
        injection_format: str = "bracket_bullet",
        reattention_pass: bool = False,
    ):
        self.injection_format = injection_format
        self.reattention_pass = reattention_pass

    def inject(
        self,
        target: Chain,
        fragment: Fragment,
        compat_score: float,
    ) -> InjectedFragment:
        """Splice the fragment into the target's context. Returns the recorded InjectedFragment."""
        injection_text = format_fragment_as_injection(fragment, self.injection_format)
        if self.reattention_pass:
            # Append a self-check directive that gets rendered immediately
            # before the next 'Step N:' anchor. The directive lives inside
            # the injection_text so it's preserved across re-builds of the
            # target's prompt (see chain.build_prompt).
            injection_text = injection_text + _REATTENTION_DIRECTIVE
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
