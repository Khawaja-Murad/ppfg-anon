"""Logging helpers — stdlib logging with optional wandb."""

from __future__ import annotations
import logging
import os
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet known-noisy loggers
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def init_wandb(project: str, name: str, config: dict, enabled: bool = True) -> Any:
    if not enabled or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb  # type: ignore
        return wandb.init(project=project, name=name, config=config, reinit=True)
    except ImportError:
        return None


def wandb_log(run, payload: dict) -> None:
    if run is None:
        return
    run.log(payload)


def wandb_finish(run) -> None:
    if run is None:
        return
    run.finish()
