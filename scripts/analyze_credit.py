#!/usr/bin/env python
"""Post-hoc PopMC-GRPO credit-assignment analysis.

Walks a completed PPFG run's saved trajectories.json, finds every PPFG injection event,
and estimates the advantage A = V(s_after_injection) - V(s_before_injection) via K
Monte Carlo rollouts each.

Usage:
    python scripts/analyze_credit.py --run_dir results/ppfg_math500-ppfg-seed42/
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from hyp_forest.chains.chain import Chain, ChainStatus, InjectedFragment
from hyp_forest.chains.population import Population, PopulationConfig
from hyp_forest.credit import PopMCGRPOAnalyzer, PopMCGRPOConfig
from hyp_forest.models import BaseLM
from hyp_forest.tasks import get_task, Problem
from hyp_forest.utils import setup_logging, seed_everything

logger = logging.getLogger(__name__)


def reconstruct_population(record: dict, pop_config: PopulationConfig) -> Population:
    """Rebuild a Population object from a serialized record."""
    pop = Population(
        problem=record["population"]["problem"],
        base_prompt="",
        config=pop_config,
        seeds=[c["seed"] for c in record["population"]["chains"]],
    )
    pop.chains = []
    for c_data in record["population"]["chains"]:
        c = Chain(
            chain_id=c_data["chain_id"],
            problem=record["population"]["problem"],
            base_prompt="",
            steps=c_data["steps"],
            prm_scores=c_data["prm_scores"],
            status=ChainStatus(c_data["status"]),
            final_answer=c_data["final_answer"],
            n_tokens_generated=c_data["n_tokens_generated"],
            seed=c_data["seed"],
        )
        for f in c_data["injected_fragments"]:
            c.injected_fragments.append(InjectedFragment(**f))
        pop.chains.append(c)
    return pop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--k_rollouts", type=int, default=8)
    parser.add_argument("--rollout_max_tokens", type=int, default=512)
    parser.add_argument("--n_problems_to_analyze", type=int, default=20,
                        help="Cap problems analyzed; this is expensive")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    setup_logging("INFO")
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    trajectories = json.loads((run_dir / "trajectories.json").read_text())

    seed_everything(cfg["experiment"]["seed"])

    task = get_task(cfg["task"]["name"])

    lm = BaseLM(
        model_name=cfg["model"]["base_model"],
        dtype=cfg["model"]["dtype"],
        tensor_parallel_size=cfg["model"]["tensor_parallel_size"],
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
        max_model_len=cfg["model"]["max_model_len"],
        seed=cfg["experiment"]["seed"],
    )
    analyzer = PopMCGRPOAnalyzer(PopMCGRPOConfig(
        k_rollouts=args.k_rollouts, rollout_max_tokens=args.rollout_max_tokens,
        rollout_temperature=cfg["generation"]["temperature"],
        rollout_top_p=cfg["generation"]["top_p"],
    ))
    pop_config = PopulationConfig(
        n_chains=cfg["population"]["n_chains"],
        prune_threshold=cfg["population"]["prune_threshold"],
        promote_on_complete=cfg["population"]["promote_on_complete"],
        max_steps=cfg["generation"]["max_steps"],
        step_separator=cfg["generation"]["step_separator"],
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        max_step_tokens=cfg["generation"]["max_step_tokens"],
    )

    all_advantages = []
    for i, record in enumerate(trajectories[: args.n_problems_to_analyze]):
        pop = reconstruct_population(record, pop_config)
        prob = Problem(
            problem_id=record["problem_id"], question=pop.problem,
            answer=record["answer"], metadata=record.get("metadata"),
        )
        # Reconstruct base_prompt for value estimation
        base_prompt = task.build_prompt(prob, lm)
        for c in pop.chains:
            c.base_prompt = base_prompt

        advantages = analyzer.compute_advantages_for_population(pop, prob, task, lm)
        for a in advantages:
            all_advantages.append({
                "problem_id": a.problem_id,
                "decision_step": a.decision_step,
                "chain_id": a.chain_id,
                "action": a.action,
                "advantage": a.advantage,
                "v_before": a.v_before,
                "v_after": a.v_after,
            })
        logger.info(f"[{i + 1}/{args.n_problems_to_analyze}] {prob.problem_id}: "
                    f"{len(advantages)} injections, mean adv = "
                    f"{np.mean([a.advantage for a in advantages]) if advantages else 0:.3f}")

    out_path = Path(args.out) if args.out else (run_dir / "credit_advantages.json")
    out_path.write_text(json.dumps(all_advantages, indent=2))
    logger.info(f"Wrote {len(all_advantages)} advantage records to {out_path}")

    # Summary
    if all_advantages:
        advs = np.array([a["advantage"] for a in all_advantages])
        summary = {
            "n_injections_analyzed": len(advs),
            "mean_advantage": float(advs.mean()),
            "std_advantage": float(advs.std()),
            "frac_positive": float((advs > 0).mean()),
            "median_advantage": float(np.median(advs)),
        }
        (run_dir / "credit_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
