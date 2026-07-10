"""Main experiment runner.

Config-driven orchestrator. Reads a YAML config, builds models + task + method, runs over
all problems, computes metrics, saves results to disk + wandb.

Usage:
    python -m hyp_forest.runner --config configs/ppfg.yaml [--n_problems 10] [--seed 1]
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf  # type: ignore

from hyp_forest.baselines import run_independent, run_smc, SMCConfig, run_best_of_n
from hyp_forest.chains.population import Population, PopulationConfig
from hyp_forest.metrics import (
    evaluate_pass_at_k, total_tokens, all_diversity_metrics,
)
from hyp_forest.models import BaseLM, ProcessRewardModel
from hyp_forest.ppfg import CompatibilityScorer, PPFGPolicy, PPFGPolicyConfig
from hyp_forest.tasks import get_task, Problem, Task
from hyp_forest.utils import seed_everything, setup_logging, init_wandb, wandb_log, wandb_finish

logger = logging.getLogger(__name__)


def _chain_seeds(run_seed: int, n_chains: int) -> list[int]:
    """Per-chain seeds derived from the run seed. vLLM's SamplingParams.seed
    is per-request and not affected by seed_everything(), so chain seeds MUST
    be mixed with the run seed here — otherwise different --seed values produce
    byte-identical chains."""
    return [run_seed * 10_000 + i for i in range(n_chains)]


def load_config(config_path: str) -> dict:
    """Load YAML config with simple `defaults: [base]` resolution."""
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        bases = cfg.pop("defaults")
        if isinstance(bases, str):
            bases = [bases]
        merged = OmegaConf.create({})
        config_dir = os.path.dirname(os.path.abspath(config_path))
        for base in bases:
            base_path = os.path.join(config_dir, f"{base}.yaml")
            base_cfg = load_config(base_path)
            merged = OmegaConf.merge(merged, base_cfg)
        cfg = OmegaConf.merge(merged, cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def run_one_problem(
    problem: Problem,
    task: Task,
    lm: BaseLM,
    prm: ProcessRewardModel,
    pop_config: PopulationConfig,
    method: str,
    cfg: dict,
    compat_scorer: CompatibilityScorer | None,
) -> Population:
    """Dispatch to the right baseline/PPFG runner for one problem."""
    base_prompt = task.build_prompt(problem, lm)

    if method == "independent":
        return run_independent(
            problem.question, base_prompt, lm, prm, pop_config,
            seeds=_chain_seeds(cfg["experiment"]["seed"], pop_config.n_chains),
        )

    if method == "smc":
        smc_cfg = SMCConfig(
            resample_temperature=cfg["baseline"]["smc_resample_temperature"],
            seed=cfg["experiment"]["seed"],
        )
        return run_smc(
            problem.question, base_prompt, lm, prm, pop_config, smc_cfg,
        )

    if method == "best_of_n":
        return run_best_of_n(
            problem.question, base_prompt, lm, prm, pop_config,
            bon_n=cfg["baseline"]["bon_n"],
            seeds=_chain_seeds(cfg["experiment"]["seed"], cfg["baseline"]["bon_n"]),
        )

    if method == "ppfg":
        assert compat_scorer is not None, "PPFG requires compatibility scorer"
        ppfg_cfg = PPFGPolicyConfig(
            extract_threshold=cfg["ppfg"]["extract_threshold"],
            k_min=cfg["ppfg"]["k_min"],
            k_max=cfg["ppfg"]["k_max"],
            injection_rule=cfg["ppfg"]["injection_rule"],
            max_injections_per_chain=cfg["ppfg"]["max_injections_per_chain"],
            seed=cfg["experiment"]["seed"],
            min_compat_threshold=cfg["ppfg"].get("min_compat_threshold", 0.0),
            injection_format=cfg["ppfg"].get("injection_format", "bracket_bullet"),
            prm_low_t=cfg["ppfg"].get("prm_low_t", 0.60),
            prm_flat_t=cfg["ppfg"].get("prm_flat_t", 0.05),
            headroom_budget=cfg["ppfg"].get("headroom_budget", 4),
            liveness_min=cfg["ppfg"].get("liveness_min", 3),
            reattention_pass=cfg["ppfg"].get("reattention_pass", False),
        )
        policy = PPFGPolicy(ppfg_cfg, compat_scorer)
        pop = Population(
            problem=problem.question, base_prompt=base_prompt, config=pop_config,
            seeds=_chain_seeds(cfg["experiment"]["seed"], pop_config.n_chains),
        )
        pop.run_to_completion(lm, prm, ppfg_hook=policy.run_hook)
        return pop

    raise ValueError(f"Unknown method: {method}")


def serialize_population(pop: Population) -> dict:
    """Convert a Population to a JSON-serializable dict for archival."""
    return {
        "problem": pop.problem,
        "step_idx": pop.step_idx,
        "chains": [
            {
                "chain_id": c.chain_id,
                "status": c.status.value,
                "steps": c.steps,
                "prm_scores": c.prm_scores,
                "final_answer": c.final_answer,
                "n_tokens_generated": c.n_tokens_generated,
                "seed": c.seed,
                "injected_fragments": [asdict(f) for f in c.injected_fragments],
            }
            for c in pop.chains
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_problems", type=int, default=None,
                        help="Override task.n_problems")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override experiment.seed")
    parser.add_argument("--method", type=str, default=None,
                        help="Override baseline.method")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override experiment.output_dir")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    if args.n_problems is not None:
        cfg["task"]["n_problems"] = args.n_problems
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.method is not None:
        cfg["baseline"]["method"] = args.method
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir

    seed_everything(cfg["experiment"]["seed"])

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with (output_dir / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    method = cfg["baseline"]["method"]
    run_name = f"{cfg['experiment']['name']}-{method}-seed{cfg['experiment']['seed']}"
    wandb_run = init_wandb(
        project=cfg["experiment"]["wandb_project"],
        name=run_name, config=cfg,
        enabled=cfg["experiment"]["log_to_wandb"],
    )

    # Load task + problems. If task.problem_subset is set (non-null, non-empty),
    # load that explicit list of dataset indices; otherwise head-slice n_problems.
    task = get_task(cfg["task"]["name"])
    _problem_subset = cfg["task"].get("problem_subset")
    if _problem_subset:
        problems = task.load(subset=list(_problem_subset))
    else:
        problems = task.load(n=cfg["task"]["n_problems"])
    logger.info(f"Running {method} on {len(problems)} problems from {task.name}")

    # Build model + PRM
    lm = BaseLM(
        model_name=cfg["model"]["base_model"],
        dtype=cfg["model"]["dtype"],
        tensor_parallel_size=cfg["model"]["tensor_parallel_size"],
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
        max_model_len=cfg["model"]["max_model_len"],
        seed=cfg["experiment"]["seed"],
    )
    prm = ProcessRewardModel(
        model_name=cfg["model"]["prm"],
        prm_kind=cfg["model"].get("prm_kind", "math_shepherd"),
        dtype=cfg["model"]["dtype"],
    )
    compat_scorer = (
        CompatibilityScorer(embedding_model=cfg["model"]["embedding_model"], mode="diversity")
        if method == "ppfg" else None
    )

    pop_config = PopulationConfig(
        n_chains=cfg["population"]["n_chains"],
        prune_threshold=cfg["population"]["prune_threshold"],
        promote_on_complete=cfg["population"]["promote_on_complete"],
        max_steps=cfg["generation"]["max_steps"],
        step_separator=cfg["generation"]["step_separator"],
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        max_step_tokens=cfg["generation"]["max_step_tokens"],
        post_injection_prune_immunity_steps=cfg["population"].get(
            "post_injection_prune_immunity_steps", 0),
    )

    # Run all problems
    populations: list[Population] = []
    t_start = time.time()
    for i, prob in enumerate(problems):
        logger.info(f"[{i + 1}/{len(problems)}] {prob.problem_id}")
        try:
            pop = run_one_problem(prob, task, lm, prm, pop_config, method, cfg, compat_scorer)
            populations.append(pop)
        except Exception as e:
            logger.exception(f"Error on {prob.problem_id}: {e}")
            # Append an empty placeholder so indices stay aligned
            populations.append(Population(
                problem=prob.question, base_prompt="", config=pop_config,
                seeds=list(range(pop_config.n_chains)),
            ))

        # Periodic save
        if (i + 1) % 25 == 0 or (i + 1) == len(problems):
            _save_run(output_dir, populations[: i + 1], problems[: i + 1])
            wandb_log(wandb_run, {
                "progress": (i + 1) / len(problems),
                "elapsed_min": (time.time() - t_start) / 60,
            })

    # Final metrics
    pass_at_k = evaluate_pass_at_k(populations, problems, task, cfg["eval"]["k_values"])
    diversity = (
        all_diversity_metrics(populations, problems, task, encoder=compat_scorer.encoder if compat_scorer else None)
        if compat_scorer is not None
        else all_diversity_metrics(populations, problems, task, encoder=None)
    )
    tokens = total_tokens(populations)

    final_metrics = {
        "method": method,
        "n_problems": len(problems),
        "total_tokens": tokens,
        "tokens_per_problem": tokens / max(len(problems), 1),
        "pass_at_k": {k: r.pass_at_k for k, r in pass_at_k.items()},
        "diversity": diversity,
        "elapsed_min": (time.time() - t_start) / 60,
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    wandb_log(wandb_run, final_metrics)
    wandb_finish(wandb_run)


def _save_run(output_dir: Path, populations: list[Population], problems: list[Problem]) -> None:
    """Persist all populations + problems for later analysis."""
    runs = []
    for pop, prob in zip(populations, problems):
        runs.append({
            "problem_id": prob.problem_id,
            "answer": prob.answer,
            "metadata": prob.metadata,
            "population": serialize_population(pop),
        })
    with (output_dir / "trajectories.json").open("w") as f:
        # default=str: defensive fallback for non-JSON-serializable objects
        # in problem.metadata (e.g., numpy arrays from pandas-derived loaders).
        json.dump(runs, f, indent=2, default=str)


if __name__ == "__main__":
    main()
