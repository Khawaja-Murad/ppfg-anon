"""CPU-only tests for the core PPFG logic.

These tests do NOT require vLLM, A100s, or any model downloads. They verify the
data-structure and algorithmic logic of Chain, FragmentExtractor, FragmentInjector,
PPFGPolicy, and the metric computations. Run them locally before launching jobs:

    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import pytest

from hyp_forest.chains.chain import Chain, ChainStatus, detect_final_answer, InjectedFragment
from hyp_forest.metrics.pass_at_k import _pass_at_k_unbiased
from hyp_forest.ppfg.extractor import FragmentExtractor


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def test_chain_empty():
    c = Chain(chain_id=0, problem="What is 2+2?", base_prompt="solve:")
    assert c.is_active
    assert c.n_steps == 0
    assert c.latest_prm_score is None
    assert c.mean_prm_score == 0.0


def test_chain_with_steps():
    c = Chain(chain_id=0, problem="x", base_prompt="p")
    c.steps = ["First step.", "Second step.", "Final answer: 42"]
    c.prm_scores = [0.8, 0.6, 0.9]
    assert c.n_steps == 3
    assert c.latest_prm_score == 0.9
    assert abs(c.mean_prm_score - (0.8 + 0.6 + 0.9) / 3) < 1e-9


def test_is_stagnating():
    c = Chain(chain_id=0, problem="x", base_prompt="p")
    c.prm_scores = [0.5, 0.5, 0.5, 0.5]
    assert c.is_stagnating(window=3, threshold=0.05)
    c.prm_scores = [0.5, 0.6, 0.7, 0.8]
    assert not c.is_stagnating(window=3, threshold=0.05)


def test_detect_final_answer_boxed():
    assert detect_final_answer(r"So the answer is \boxed{42}") == "42"
    assert detect_final_answer(r"\boxed{x = 3}") == "x = 3"


def test_detect_final_answer_phrase():
    assert detect_final_answer("Final answer: 7") == "7"
    assert detect_final_answer("The answer is 99.") == "99"


def test_detect_final_answer_none():
    assert detect_final_answer("Just thinking out loud.") is None


def test_chain_build_prompt_no_steps():
    c = Chain(chain_id=0, problem="x", base_prompt="<system>solve:</system>")
    prompt = c.build_prompt()
    assert "<system>solve:</system>" in prompt
    assert "Step 1:" in prompt


def test_chain_build_prompt_with_injection():
    c = Chain(chain_id=0, problem="x", base_prompt="<sys>solve</sys>")
    c.steps = ["First step.", "Second step."]
    c.injected_fragments.append(InjectedFragment(
        source_chain_id=99, fragment_steps=["Insight A.", "Insight B."],
        fragment_prm_scores=[0.8, 0.9], quality=0.85, compat_score=0.4,
        injected_at_step=2,  # injected before the next (3rd) step
        injection_text="[Insight from a related reasoning path:\n  - Insight A.\n  - Insight B.\n]\n",
    ))
    prompt = c.build_prompt()
    assert "Insight A." in prompt
    assert "Insight B." in prompt
    assert "First step." in prompt
    assert "Step 3:" in prompt  # next step to be generated


# ---------------------------------------------------------------------------
# FragmentExtractor
# ---------------------------------------------------------------------------

def test_extractor_finds_longest_high_prefix():
    c = Chain(chain_id=1, problem="x", base_prompt="p")
    c.steps = ["a", "b", "c", "d", "e"]
    c.prm_scores = [0.9, 0.8, 0.7, 0.4, 0.9]   # prefix [0.9, 0.8, 0.7] all >= 0.6
    ext = FragmentExtractor(extract_threshold=0.6, k_min=2, k_max=8)
    frag = ext.extract(c)
    assert frag is not None
    assert frag.steps == ["a", "b", "c"]
    assert abs(frag.quality - 0.8) < 1e-9


def test_extractor_below_kmin_returns_none():
    c = Chain(chain_id=1, problem="x", base_prompt="p")
    c.steps = ["a", "b", "c"]
    c.prm_scores = [0.9, 0.3, 0.9]   # only first step qualifies, k* = 1 < k_min = 2
    ext = FragmentExtractor(extract_threshold=0.6, k_min=2)
    assert ext.extract(c) is None


def test_extractor_caps_at_kmax():
    c = Chain(chain_id=1, problem="x", base_prompt="p")
    c.steps = [f"step{i}" for i in range(10)]
    c.prm_scores = [0.9] * 10
    ext = FragmentExtractor(extract_threshold=0.5, k_min=2, k_max=4)
    frag = ext.extract(c)
    assert frag is not None
    assert len(frag.steps) == 4


def test_extractor_no_scores():
    c = Chain(chain_id=1, problem="x", base_prompt="p")
    ext = FragmentExtractor()
    assert ext.extract(c) is None


# ---------------------------------------------------------------------------
# Pass@k
# ---------------------------------------------------------------------------

def test_pass_at_k_zero_correct():
    assert _pass_at_k_unbiased(n=10, c=0, k=1) == 0.0
    assert _pass_at_k_unbiased(n=10, c=0, k=5) == 0.0


def test_pass_at_k_all_correct():
    assert _pass_at_k_unbiased(n=10, c=10, k=1) == 1.0
    assert _pass_at_k_unbiased(n=10, c=10, k=5) == 1.0


def test_pass_at_k_partial():
    # 5 correct out of 10, k=1 => P(any of k=1 sample is correct) = 5/10 = 0.5
    val = _pass_at_k_unbiased(n=10, c=5, k=1)
    assert abs(val - 0.5) < 1e-9


def test_pass_at_k_n_less_than_k():
    # If n - c < k we always pass
    assert _pass_at_k_unbiased(n=5, c=1, k=10) == 1.0


# ---------------------------------------------------------------------------
# PPFG injector formatting (no model dependency)
# ---------------------------------------------------------------------------

def test_format_injection():
    from hyp_forest.ppfg.extractor import Fragment
    from hyp_forest.ppfg.injector import format_fragment_as_injection
    frag = Fragment(
        source_chain_id=3,
        steps=["Try substitution u=x+1.", "Now the integral becomes ∫u du."],
        prm_scores=[0.85, 0.9], quality=0.875, end_step_idx=2,
    )
    text = format_fragment_as_injection(frag)
    assert "Try substitution u=x+1." in text
    assert "Now the integral becomes" in text
    assert "0.88" in text  # quality formatted to 2 decimals
    assert "2 steps" in text


# ---------------------------------------------------------------------------
# ChainStatus enum sanity
# ---------------------------------------------------------------------------

def test_chain_status_values():
    assert ChainStatus.ACTIVE.value == "active"
    assert ChainStatus.PRUNED.value == "pruned"
    assert ChainStatus.PROMOTED.value == "promoted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
