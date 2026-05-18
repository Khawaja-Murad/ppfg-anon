"""Process Reward Model wrapper.

Defaults to Math-Shepherd (peiyi9979/math-shepherd-mistral-7b-prm), which scores
each reasoning step in [0, 1] given the problem and the partial solution up to that step.

Math-Shepherd protocol (from the paper / model card):
    - Format: "<problem> Step 1: <s1> ки\nStep 2: <s2> ки\n..." where 'ки' is a step-end marker.
    - At each ки position, the model emits a token whose logit ratio for ('+', '-') gives the score.
    - Specifically: score = sigmoid(logit['+'] - logit['-']) at each ки position.

This wrapper abstracts that detail so the rest of the codebase calls .score_steps(problem, steps)
and gets back a list[float] of per-step PRM scores.

Switching PRM: set `prm_kind` in the constructor:
    - 'math_shepherd' (default)
    - 'skywork' (Skywork-o1-Open-PRM-Qwen-2.5-7B; uses different protocol — see _score_skywork)
    - 'versaprm' (VersaPRM: UW-Madison-Lee-Lab/VersaPRM PEFT adapter on top of
       UW-Madison-Lee-Lab/Llama-PRM800K base — see _score_versaprm)
    - 'dummy' (returns 0.5 for all steps; for unit tests only)
"""

from __future__ import annotations
import logging
from typing import Sequence

import torch

logger = logging.getLogger(__name__)


# Math-Shepherd-specific tokens
MS_GOOD_TOKEN = "+"
MS_BAD_TOKEN = "-"
MS_STEP_TAG = "ки"

# VersaPRM-specific constants. The V13.0.2 spec said step_tag='И' and softmax
# index 0, but the actual UW-Madison-Lee-Lab/VersaPRM reference scoring code
# uses 4 newlines '\n\n\n\n' as the classification anchor (token 23535 in the
# Llama-PRM800K tokenizer) and takes softmax[..., 1] over the candidate-token
# logits. Day-13 smoke (61065051) with the V13.0.2 literal protocol produced
# inverted, degenerate-low scores (mean 0.03 on correct vs 0.06 on incorrect);
# this corrected protocol matches the reference impl. See:
#   https://github.com/UW-Madison-Lee-Lab/VersaPRM
VERSAPRM_BASE_MODEL = "UW-Madison-Lee-Lab/Llama-PRM800K"
VERSAPRM_ADAPTER = "UW-Madison-Lee-Lab/VersaPRM"
VERSAPRM_STEP_TAG = " \n\n\n\n"  # space + 4 newlines; tokenizes to single id 23535
# (the leading space prevents BPE-merging with the preceding step's trailing
# punctuation; without it, '.\\n\\n\\n\\n' merges into token 2055 and the
# anchor positions become inconsistent — verified via Day-13 smoke 61065277)
VERSAPRM_CANDIDATE_TOKEN_IDS = [12, 10]  # [token 12 = '+', token 10 = '-']
VERSAPRM_SCORE_INDEX = 1  # reference impl: softmax(...)[..., 1] is P(correct)


class ProcessRewardModel:
    """Wraps a PRM and exposes a uniform .score_steps(problem, steps) -> list[float] interface."""

    def __init__(
        self,
        model_name: str = "peiyi9979/math-shepherd-mistral-7b-prm",
        prm_kind: str = "math_shepherd",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.prm_kind = prm_kind
        self.device = device

        if prm_kind == "dummy":
            self.tokenizer = None
            self.model = None
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        torch_dtype = getattr(torch, dtype)

        if prm_kind == "versaprm":
            # VersaPRM = PEFT adapter on top of Llama-PRM800K. The `model_name`
            # arg is informational only here — the actual identifiers are fixed
            # (base + adapter) per V13.0.2 spec.
            try:
                import peft  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "VersaPRM requires the `peft` package (transformers.PreTrainedModel.load_adapter "
                    "is a thin wrapper over peft). Install it on the login node before submitting "
                    "any SLURM job: `pip install peft`."
                ) from e

            logger.info(
                f"Loading VersaPRM: base={VERSAPRM_BASE_MODEL} adapter={VERSAPRM_ADAPTER}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                VERSAPRM_BASE_MODEL,
                padding_side="left",
                truncation_side="left",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                VERSAPRM_BASE_MODEL, torch_dtype=torch_dtype, device_map=device
            )
            # PEFT adapter attached via the transformers integration.
            # Monkey-patch peft's TP-sharding pre-import: peft 0.15.2 eagerly imports
            # `transformers.integrations.tensor_parallel`, which transformers 4.51.3
            # does not ship. Since we run single-GPU (no `_hf_tp_plan`), the inner
            # loop body is a no-op anyway; replacing the whole function with a
            # no-op skips the bad import. (Verified Day-13 GPQA × VersaPRM jobs
            # 61066131/133/135/137/149/152/154/156/158 all hit this path.)
            try:
                from peft.utils import save_and_load as _peft_sal  # type: ignore
                _peft_sal._maybe_shard_state_dict_for_tp = lambda *a, **kw: None
            except Exception:
                pass
            self.model.load_adapter(VERSAPRM_ADAPTER)
            self.model.eval()

            # Literal candidate IDs per V13.0.2 spec (do NOT look up via tokenizer).
            self._versa_candidate_ids = list(VERSAPRM_CANDIDATE_TOKEN_IDS)
            # Step-tag ID is resolved via tokenizer (it's a real vocab token in Llama-PRM800K).
            self._versa_step_tag_id = self.tokenizer.encode(
                VERSAPRM_STEP_TAG, add_special_tokens=False
            )[-1]
            return

        logger.info(f"Loading PRM: {model_name} (kind={prm_kind})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device
        )
        self.model.eval()

        if prm_kind == "math_shepherd":
            self._good_id = self.tokenizer.encode(MS_GOOD_TOKEN)[-1]
            self._bad_id = self.tokenizer.encode(MS_BAD_TOKEN)[-1]
            self._step_tag_id = self.tokenizer.encode(MS_STEP_TAG)[-1]

    @torch.no_grad()
    def score_steps(self, problem: str, steps: Sequence[str]) -> list[float]:
        """Return per-step PRM scores for one chain. Length == len(steps)."""
        if self.prm_kind == "dummy":
            return [0.5] * len(steps)
        if not steps:
            return []
        if self.prm_kind == "math_shepherd":
            return self._score_math_shepherd(problem, steps)
        if self.prm_kind == "skywork":
            return self._score_skywork(problem, steps)
        if self.prm_kind == "versaprm":
            return self._score_versaprm(problem, steps)
        raise ValueError(f"Unknown prm_kind: {self.prm_kind}")

    @torch.no_grad()
    def score_steps_batched(
        self, problems: Sequence[str], steps_per_problem: Sequence[Sequence[str]]
    ) -> list[list[float]]:
        """Batched scoring over multiple chains. Same return shape as repeated score_steps."""
        # Simple loop for now; for V1 this is acceptable. A real batched impl pads then masks.
        return [self.score_steps(p, s) for p, s in zip(problems, steps_per_problem)]

    def _score_math_shepherd(self, problem: str, steps: Sequence[str]) -> list[float]:
        """Math-Shepherd format: insert ки after each step, take +/- logit ratio at each ки."""
        # Build the formatted prompt
        formatted_steps = []
        for i, s in enumerate(steps, 1):
            # Strip any pre-existing "Step N:" prefix from the step text
            cleaned = s.strip()
            if cleaned.lower().startswith(f"step {i}:"):
                cleaned = cleaned[len(f"step {i}:"):].strip()
            formatted_steps.append(f"Step {i}: {cleaned} {MS_STEP_TAG}")
        prompt = problem.strip() + "\n" + "\n".join(formatted_steps)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        logits = self.model(input_ids).logits  # [1, T, V]

        # Find positions of ки tokens
        step_tag_positions = (input_ids[0] == self._step_tag_id).nonzero(as_tuple=True)[0]
        if len(step_tag_positions) != len(steps):
            logger.warning(
                f"PRM tag mismatch: {len(step_tag_positions)} tags found vs {len(steps)} steps. "
                f"Returning uniform 0.5 fallback."
            )
            return [0.5] * len(steps)

        # At each ки position, score = sigmoid(logit[+] - logit[-]) at that position
        scores = []
        for pos in step_tag_positions:
            l_good = logits[0, pos, self._good_id].item()
            l_bad = logits[0, pos, self._bad_id].item()
            score = torch.sigmoid(torch.tensor(l_good - l_bad)).item()
            scores.append(float(score))
        return scores

    def _score_skywork(self, problem: str, steps: Sequence[str]) -> list[float]:
        """Skywork-PRM uses a regression head; score per step is in [0,1].

        See https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B for details.
        Implementation note: Skywork emits a step-end token at which a value head produces
        the score directly. If you switch to Skywork, verify the exact tokenization the
        model card prescribes — the format below is a sane default but may need adjustment.
        """
        # Minimal correct-ish implementation; verify against model card before relying on it.
        formatted = problem.strip() + "\n"
        per_step_inputs = []
        running = formatted
        for i, s in enumerate(steps, 1):
            running = running + f"Step {i}: {s.strip()}\n"
            per_step_inputs.append(running)

        scores = []
        for text in per_step_inputs:
            ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            out = self.model(ids)
            # Skywork uses last-token logit -> sigmoid as the value
            logit = out.logits[0, -1, 0].item()
            scores.append(float(torch.sigmoid(torch.tensor(logit)).item()))
        return scores

    def _score_versaprm(self, problem: str, steps: Sequence[str]) -> list[float]:
        """VersaPRM scoring (corrected to match UW-Madison-Lee-Lab reference impl).

        Protocol:
            1. Build context = problem + step_1 + '\\n\\n\\n\\n' + step_2 +
               '\\n\\n\\n\\n' + ... — the 4-newline block tokenizes to a single
               token id (23535 in the Llama-PRM800K tokenizer) which is the
               classification anchor the LoRA head was trained to score over.
            2. Single forward pass over the full sequence.
            3. At each classification-anchor position, take logits over the two
               candidate token IDs ([12, 10] = ['+', '-']) and softmax. The
               reference impl reports softmax[..., 1] as the P(correct) score.
        """
        # 1. Build the running context with embedded ' \\n\\n\\n\\n' anchors after every step.
        # The space prefix prevents the anchor from BPE-merging with the previous step's
        # trailing punctuation, keeping the anchor token id stable at 23535. We do NOT
        # add an extra "\\n\\n" separator between steps: the anchor's 4 trailing newlines
        # already separate adjacent steps, and any extra newline run merges with the anchor
        # into a different token id (verified empirically against the
        # UW-Madison-Lee-Lab/Llama-PRM800K tokenizer, Day-13 smoke 61065516).
        prompt = problem.strip()
        for s in steps:
            prompt = prompt + " " + s.strip().rstrip(".").rstrip() + VERSAPRM_STEP_TAG

        # 2. Tokenize + forward pass.
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        logits = self.model(input_ids).logits  # [1, T, V]

        # 3. Find all anchor positions.
        step_tag_positions = (input_ids[0] == self._versa_step_tag_id).nonzero(as_tuple=True)[0]
        if len(step_tag_positions) != len(steps):
            logger.warning(
                f"VersaPRM tag mismatch: {len(step_tag_positions)} anchor tokens found "
                f"vs {len(steps)} steps (prompt may have re-tokenized '\\n\\n\\n\\n' unexpectedly). "
                f"Returning uniform 0.5 fallback."
            )
            return [0.5] * len(steps)

        # 4. softmax over candidate IDs; reference impl uses index 1 as P(correct).
        cand_ids = torch.tensor(self._versa_candidate_ids, device=logits.device)
        scores: list[float] = []
        for pos in step_tag_positions:
            cand_logits = logits[0, pos, cand_ids]  # shape: [2]
            probs = torch.softmax(cand_logits.float(), dim=-1)
            scores.append(float(probs[VERSAPRM_SCORE_INDEX].item()))
        return scores
