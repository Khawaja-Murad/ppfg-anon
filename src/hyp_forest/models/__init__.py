"""Models package — lazy imports so the rest of the codebase remains importable
on CPU-only machines for testing/inspection without torch/vLLM/transformers."""

# Lazy re-exports: don't pull torch/vLLM at import time.

__all__ = ["BaseLM", "StepGenerationOutput", "ProcessRewardModel"]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name in {"BaseLM", "StepGenerationOutput"}:
        from hyp_forest.models.base_lm import BaseLM, StepGenerationOutput
        return {"BaseLM": BaseLM, "StepGenerationOutput": StepGenerationOutput}[name]
    if name == "ProcessRewardModel":
        from hyp_forest.models.prm import ProcessRewardModel
        return ProcessRewardModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
