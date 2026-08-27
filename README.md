# PPFG: PRM-Pruned Fragment Grafting

Code release for
**"Characterizing a Configuration Where Inference-Time PRM-Pruned Fragment
Grafting Is Inert: Evidence from Three Reasoning LMs"**

Khawaja Murad ul Hassan (QLU.ai, Islamabad) and
Mehran Ebrahimi (Faculty of Science, Ontario Tech University, Oshawa, ON, Canada).

> arXiv: *link to be added on posting.*
> This repository was previously an anonymized ARR release; it is now public
> and de-anonymized. The repository name is retained so that links in the
> ARR May 2026 review thread keep resolving.

**What the paper reports.** PPFG extracts the high-PRM prefix of a chain at
the moment a process reward model prunes it, and grafts it verbatim into a
still-decoding sibling. Across three base LMs, six reasoning benchmarks, two
PRMs, several targeting rules, and populations from N=8 to N=256, the
mechanism is statistically indistinguishable from independent parallel CoT
on Pass@k, mode rate, and diversity. The result is a *mechanism null* at a
precisely specified operating point, established with pre-registered TOST
equivalence tests rather than a bare non-significant result.

This repository implements the PPFG mechanism, the three matched-compute
baselines (independent parallel CoT, SMC with PRM-weighted resampling,
Best-of-N), and the analysis scripts that produce every table and
figure in the paper.

---

## Layout

```
.
├── src/hyp_forest/        # PPFG mechanism, baselines, runner
│   ├── chains/            # Chain dataclass + Population loop
│   ├── models/            # vLLM wrapper + Math-Shepherd / VersaPRM PRMs
│   ├── ppfg/              # extract / target-select / gate / inject
│   ├── baselines/         # independent, SMC, Best-of-N
│   ├── credit/            # post-hoc value-estimation skeleton
│   └── tasks/             # MATH500, GSM8K, AIME, NuminaMath, GPQA, MMLU-Pro
├── configs/               # One YAML per (model × task × method) cell
├── scripts/               # run_experiment, eval, analysis
├── slurm/                 # SBATCH wrappers for HPC submission
└── tests/test_smoke.py    # CPU-only smoke tests
```

## Reproducing the headline numbers

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

vLLM (in `requirements.txt`) requires a CUDA-capable GPU with SM80+
(A100 or H100). On HPC clusters with module systems you may need
`module load cuda/12.2 python/3.11 arrow opencv` (or equivalents)
before `pip install`.

Set `HF_HOME` to a large cache directory and pre-download the models:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
huggingface-cli download peiyi9979/math-shepherd-mistral-7b-prm
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
# Optional cross-arch:
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# Optional cross-PRM:
huggingface-cli download UWNSL/VersaPRM
```

### Local smoke test (CPU-only, <1 minute)

```bash
PYTHONPATH=src pytest tests/test_smoke.py -v
```
19 tests; all must pass before launching any GPU run.

### Single-cell GPU run (~2-3 hours on a single 40GB A100)

```bash
# Sanity: 10 problems, no PPFG hook
PYTHONPATH=src python scripts/run_experiment.py \
    --config configs/sanity.yaml \
    --method independent \
    --output_dir results/sanity

# Full Qwen × MATH500 × PPFG-stag at n=500, seed 42
PYTHONPATH=src python scripts/run_experiment.py \
    --config configs/ppfg.yaml \
    --method ppfg \
    --seed 42 \
    --output_dir results/qwen-ppfg-stag-seed42
```

### Full 3-arch × 6-benchmark matrix on a SLURM cluster

```bash
# Edit slurm/run_experiment.sh and set --account=<YOUR_SLURM_ACCOUNT>
sbatch slurm/run_sweep.sh math500
sbatch slurm/run_ablations.sh math500
```

Each SLURM script accepts `SEED=<n>` as an environment override; per-seed
parallel runs use `for s in 42 43 44; do SEED=$s sbatch slurm/run_experiment.sh ...; done`.

## Analysis scripts

Each script is run from the repo root with `PYTHONPATH=src` and writes
to `results/`:

| Script | Purpose | Paper reference |
|---|---|---|
| `scripts/eval.py` | Aggregate `metrics.json` across all run dirs | All tables |
| `scripts/compute_tost.py` | Two-one-sided-tests equivalence verdicts | Appendix I |
| `scripts/compute_injection_prm_delta.py` | Per-event pre-vs-post PRM delta | §5.4 |
| `scripts/compute_cliffs_delta.py` | Cliff's δ effect size for the PRM delta | §5.4 |
| `scripts/post_injection_prune_analysis.py` | Chain-dynamics bucket counts | §5.4 |
| `scripts/prm_distribution_audit.py` | PRM-distribution audit by architecture | Appendix G |
| `scripts/compute_firing_stats.py` | Injection rate, fragment length, quality | §5.1 |
| `scripts/aggregate_xi_min.py` | ξ_min sensitivity sweep aggregation | Appendix H |
| `scripts/oracle_targeting_analysis.py` | Hindsight-oracle Pass@k upper bound (+0.13 pp) | §6, Appendix M |
| `scripts/targeting_by_difficulty_stratum.py` | Targeting quality by difficulty tercile | Appendix B |
| `scripts/post_injection_prune_analysis_crossarch.py` | Pruning hazard on LLaMA / DeepSeek | Appendix L |
| `scripts/surviving_sibling_counterfactual.py` | Matched-pair sibling counterfactual (N=65) | Appendix L |
| `scripts/prune_immunity_analysis.py` | Prune-immunity content control (see correction below) | Appendix O |
| `scripts/aggregate_n_sweep.py` | Population-size sweep, N=16..256 | Appendix P |
| `scripts/inter_annotator_classify.py` | Four-bucket targeting classifier | Appendix B |
| `scripts/recompute_metrics_post_fix.py` | Re-score a run with the balanced-brace extractor | All tables |

## Mechanism summary

PPFG is invoked once per population step on chains pruned that step:

1. **Extract.** Longest contiguous prefix of the pruned chain whose
   per-step PRM scores all exceed `extract_threshold`, kept if its
   length falls in `[k_min, k_max]`.
2. **Target-select.** Choose a still-decoding sibling under the rule
   `R ∈ {compat, stagnation, stagnation_compound, random}`.
3. **Gate.** Compute sentence-embedding compatibility `ξ(F, target)`;
   admit if `ξ ≥ xi_min`.
4. **Inject.** Splice the fragment verbatim into the target's prompt
   as a bracketed in-place demonstration.

All four operations are visible in `src/hyp_forest/ppfg/`.

## License

MIT — see `LICENSE`.

## Notes on reproducibility

- All experiments use seeds 42, 43, 44 (plus 45, 46 for the Qwen
  5-seed TOST cells).
- Per-chain sampling seeds are deterministic functions of the run seed
  and chain index, mixed to avoid byte-identical chains across seeds.
- vLLM does not respect `torch.manual_seed`; per-chain seeds are passed
  via vLLM `SamplingParams.seed`. See
  `src/hyp_forest/chains/population.py` for the canonical mixing.
- The balanced-brace `\boxed{·}` extractor handles nested LaTeX (e.g.,
  `\boxed{\left(3,\frac{\pi}{2}\right)}`). An earlier non-balanced
  regex undercounted Pass@k uniformly by ≈10 absolute points; the
  balanced parser is used throughout the paper.
- **Install `math-verify`** (`pip install math-verify`) before scoring
  anything you intend to compare against the paper. Without it the
  comparator silently degrades to string matching -- see the correction
  note below. Every scoring path now warns when this happens.

## A correction, and the trap that caused it

During the ARR May 2026 response period we reported a **+6.96 pp** Pass@1
"immunity lift" for the prune-immunity control. **That number is wrong. The
correct figure is +0.71 pp, which is inside seed-sigma.** The published
paper states the corrected value and discloses the correction; this note
records the cause so that nobody reproducing this work repeats it.

Two silent mismatches stacked:

1. **Unmatched problem subsets.** The baseline arm came from a run over
   MATH500 problems **400-499**, while the two immunity arms used
   `problem_subset: null`, which `runner.py` head-slices to problems
   **0-99**. Different problem sets, different difficulty.
2. **Unmatched comparators.** `math_verify` is an optional dependency and
   every scoring path falls back to normalized string matching when it is
   absent. The fallback under-counts correct answers by several absolute
   points, and it was **silent**. A `metrics.json` written when
   `math_verify` was importable is not comparable to one recomputed when it
   was not.

**What survives.** The load-bearing contrast of that experiment --
(immunity + real graft) vs. (immunity + null graft) -- is unaffected, because
those two arms always matched on both problems and comparator. It is
-0.04 pp at Pass@1. The experiment's conclusion is unchanged; only the
"the exemption is a large lever" framing does not survive. The correct
manipulation check is the pruned-chain fraction (0.1104 -> 0.0804), which
the analysis script now reports.

**What changed in this repository.** `src/hyp_forest/comparator_guard.py`
makes the degraded comparator warn loudly, once, from every scoring path,
and `scripts/prune_immunity_analysis.py` re-scores all arms itself on
matched problems rather than trusting stored `metrics.json`.

**The general rule:** before reporting any cross-run delta, confirm both
runs share (a) the same resolved problem subset and (b) the same comparator
state. Prefer contrasts between arms that differ in exactly one config key.

## Notes

- Saved trajectories are not in this release (size). Request
  `trajectories.tar.gz` (~1 GB) to re-run the analysis scripts without
  re-decoding.
- The architectural figure (Figure 1 in the paper) is generated by
  matplotlib; see `scripts/figures/` for the source.

## Citation

```bibtex
@misc{hassan2026ppfg,
  title  = {Characterizing a Configuration Where Inference-Time PRM-Pruned
            Fragment Grafting Is Inert: Evidence from Three Reasoning LMs},
  author = {Khawaja Murad ul Hassan and Mehran Ebrahimi},
  year   = {2026},
  note   = {arXiv preprint; identifier to be added on posting}
}
```
