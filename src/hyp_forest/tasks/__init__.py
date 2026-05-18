from hyp_forest.tasks.base import Task, Problem
from hyp_forest.tasks.math500 import MATH500
from hyp_forest.tasks.gpqa import GPQADiamond
from hyp_forest.tasks.gsm8k import GSM8K
from hyp_forest.tasks.aime import AIME
from hyp_forest.tasks.olympiadbench import OlympiadBenchMath
from hyp_forest.tasks.mmlu_pro import MMLUPro
from hyp_forest.tasks.numina_math import NuminaMath


def get_task(name: str) -> Task:
    name = name.lower().replace("-", "_")
    if name in {"math500", "math_500"}:
        return MATH500()
    if name in {"gpqa", "gpqa_diamond"}:
        return GPQADiamond()
    if name in {"gsm8k", "gsm_8k"}:
        return GSM8K()
    if name in {"aime"}:
        return AIME()
    if name in {"olympiadbench_math", "olympiad_math", "olympiadbench", "olympiad"}:
        return OlympiadBenchMath()
    if name in {"mmlu_pro", "mmlupro", "mmlu_pro_lbp"}:
        return MMLUPro()
    if name in {"numina_math", "numinamath", "numina"}:
        return NuminaMath()
    raise ValueError(f"Unknown task: {name}")


__all__ = [
    "Task",
    "Problem",
    "MATH500",
    "GPQADiamond",
    "GSM8K",
    "AIME",
    "OlympiadBenchMath",
    "MMLUPro",
    "NuminaMath",
    "get_task",
]
