"""Prompt construction.

We use the chat template of the base model (vLLM exposes the tokenizer) and let the model
emit numbered 'Step N:' steps separated by blank lines. The step separator is configured
in PopulationConfig.step_separator and used as the vLLM stop token.
"""

from __future__ import annotations


MATH_SYSTEM = (
    "You are a careful mathematical reasoner. Solve the following problem step by step. "
    "Number each reasoning step as 'Step 1:', 'Step 2:', etc., separated by blank lines. "
    "When you have the final answer, write it inside \\boxed{...} on a step of its own."
)

GPQA_SYSTEM = (
    "You are a careful scientific reasoner. Read the question and four choices. "
    "Reason step by step, numbering each step as 'Step 1:', 'Step 2:', etc., separated by "
    "blank lines. When you have selected an answer, write it on a step of its own as "
    "'Final answer: X' where X is one of A, B, C, D."
)

MMLU_PRO_SYSTEM = (
    "You are a careful reasoner. Read the question and ten choices. "
    "Reason step by step, numbering each step as 'Step 1:', 'Step 2:', etc., separated by "
    "blank lines. When you have selected an answer, write it on a step of its own as "
    "'Final answer: X' where X is one of A, B, C, D, E, F, G, H, I, J."
)


def build_math_cot_prompt(question: str, lm) -> str:
    """Build chat-formatted prompt for a math problem."""
    messages = [
        {"role": "system", "content": MATH_SYSTEM},
        {"role": "user", "content": question.strip()},
    ]
    return lm.apply_chat_template(messages)


def build_gpqa_prompt(question: str, choices: list[str], lm) -> str:
    letters = ["A", "B", "C", "D"]
    rendered_choices = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
    user = f"{question.strip()}\n\nChoices:\n{rendered_choices}"
    messages = [
        {"role": "system", "content": GPQA_SYSTEM},
        {"role": "user", "content": user},
    ]
    return lm.apply_chat_template(messages)


def build_mmlu_pro_prompt(question: str, choices: list[str], lm) -> str:
    """Build chat-formatted prompt for an MMLU-Pro 10-choice MCQ.

    Choices are letter-labeled A..J (up to 10). If fewer than 10 choices are
    supplied (rare in MMLU-Pro), only the first `len(choices)` letters are
    used; the system prompt still references A..J as the legal label set.
    """
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    rendered_choices = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
    user = f"{question.strip()}\n\nChoices:\n{rendered_choices}"
    messages = [
        {"role": "system", "content": MMLU_PRO_SYSTEM},
        {"role": "user", "content": user},
    ]
    return lm.apply_chat_template(messages)
