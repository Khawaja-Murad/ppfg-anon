"""vLLM wrapper for batched, step-by-step parallel CoT generation.

Design constraint: we need to generate N parallel chains *one reasoning step at a time*,
score each with a PRM between steps, and inject context into still-decoding chains.
HuggingFace's .generate() is too slow for this regime; vLLM's continuous batching is essential.

Approach: at each population step, we batch all active chains' next-step prompts into a
single LLM.generate() call with a stop string (default "\n\n") that separates reasoning steps.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class StepGenerationOutput:
    """Output of generating one reasoning step for one chain."""
    text: str               # the generated step text (without leading whitespace)
    finish_reason: str      # 'stop' (hit step separator), 'length' (hit max_step_tokens), 'eos'
    n_tokens: int           # number of tokens generated for this step
    cumulative_logprob: float  # sum of token logprobs (for SMC weight calculation)


class BaseLM:
    """Thin vLLM wrapper used by all population managers and baselines."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        seed: int = 42,
    ):
        # Lazy-import vLLM so the module can be imported in test environments without GPUs.
        from vllm import LLM, SamplingParams  # type: ignore
        self._SamplingParams = SamplingParams

        logger.info(f"Loading base LM: {model_name}")
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            enforce_eager=False,  # CUDA graphs are fine for inference-only
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.model_name = model_name

    def generate_steps(
        self,
        prompts: Sequence[str],
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_step_tokens: int = 256,
        step_separator: str = "\n\n",
        seeds: Sequence[int] | None = None,
    ) -> list[StepGenerationOutput]:
        """Generate exactly one reasoning step for each input prompt, in parallel.

        The step terminates when either (a) we generate `step_separator`, or
        (b) we hit `max_step_tokens`. The separator is NOT included in the returned text.

        Args:
            prompts: Each prompt is the full context up to (but not including) the next step.
            seeds: Optional per-prompt seeds for reproducibility across re-runs.

        Returns:
            One StepGenerationOutput per input prompt, in the same order.
        """
        if seeds is None:
            seeds = [None] * len(prompts)

        # vLLM SamplingParams accepts a `stop` list of strings. The model will produce up to
        # `max_tokens` tokens, then stop early if it generates any of the stop strings.
        sampling_params_list = [
            self._SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_step_tokens,
                stop=[step_separator],
                # logprobs=1 gives us cumulative_logprob via output.outputs[0].cumulative_logprob
                logprobs=1,
                seed=seed,
            )
            for seed in seeds
        ]

        # vLLM accepts either one SamplingParams or a list aligned with prompts.
        outputs = self.llm.generate(
            prompts=list(prompts),
            sampling_params=sampling_params_list,
            use_tqdm=False,
        )

        results: list[StepGenerationOutput] = []
        for out in outputs:
            assert len(out.outputs) == 1, "Expected n=1 sampling per prompt"
            o = out.outputs[0]
            results.append(
                StepGenerationOutput(
                    text=o.text.strip(),
                    finish_reason=o.finish_reason or "unknown",
                    n_tokens=len(o.token_ids),
                    cumulative_logprob=float(o.cumulative_logprob or 0.0),
                )
            )
        return results

    def generate_complete(
        self,
        prompts: Sequence[str],
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        seeds: Sequence[int] | None = None,
    ) -> list[str]:
        """Generate full completions in one shot — used for Best-of-N baseline."""
        if seeds is None:
            seeds = [None] * len(prompts)
        sampling_params_list = [
            self._SamplingParams(
                temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
            )
            for seed in seeds
        ]
        outputs = self.llm.generate(
            prompts=list(prompts),
            sampling_params=sampling_params_list,
            use_tqdm=False,
        )
        return [out.outputs[0].text for out in outputs]

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Convenience: build a chat-formatted prompt for instruct models."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
