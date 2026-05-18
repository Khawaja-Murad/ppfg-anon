# PPFG: PRM-Pruned Fragment Grafting

Anonymous code release for the ARR May 2026 submission
*Inference-Time Fragment Grafting Doesn't Move Strong-LM Reasoning*.

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
18 tests; all must pass before launching any GPU run.

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

## Caveats for reviewers

- Saved trajectories are not in this release (size). A separate
  `trajectories.tar.gz` (~1 GB) is uploaded alongside if reviewers
  want to re-run analysis without re-decoding.
- The architectural figure (Figure 1 in the paper) is generated by
  matplotlib; see `scripts/figures/` for the source.
- Author identity has been scrubbed from every file in this release;
  if any artifact appears non-anonymous, please notify the area chair.
