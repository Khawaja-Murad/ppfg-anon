#!/bin/bash
#SBATCH --job-name=hyp-forest-sweep
#SBATCH --account=<YOUR_SLURM_ACCOUNT>               # 
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-11                         # 4 methods × 3 seeds = 12 jobs

# Sweep across (method, seed) combinations as a SLURM job array.
#
# Submit:
#   sbatch slurm/run_sweep.sh math500
#   sbatch slurm/run_sweep.sh gpqa_diamond
#
# Each job pulls one (method, seed) from arrays below.

set -euo pipefail

TASK=${1:-math500}

METHODS=("independent" "smc" "best_of_n" "ppfg")
SEEDS=(42 43 44)

# Index decomposition
N_METHODS=${#METHODS[@]}
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
METHOD=${METHODS[$METHOD_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Pick config per method
if [[ "$METHOD" == "ppfg" ]]; then
    CONFIG=configs/ppfg.yaml
else
    CONFIG=configs/default.yaml
fi

# Inherit setup from run_experiment.sh
mkdir -p logs results

module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11 arrow opencv/4.12.0

VENV_DIR="${HOME}/scratch/hyp-forest-venv"
source "${VENV_DIR}/bin/activate"

export HF_HOME="${HOME}/scratch/hf_cache"
# Do NOT set TRANSFORMERS_CACHE — deprecated, overrides HF_HOME's hub/ subdir.
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# Compute nodes have no internet — force offline so HF Hub HEAD calls fail
# immediately instead of wasting ~3 min per file on retry+backoff.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="${HOME}/scratch/wandb"
mkdir -p "${WANDB_DIR}"

RUN_NAME="${METHOD}-${TASK}-seed${SEED}"
OUTPUT_DIR="results/${RUN_NAME}"

echo "=========================================="
echo "Sweep job ${SLURM_ARRAY_TASK_ID}: ${METHOD} on ${TASK} seed=${SEED}"
echo "=========================================="

PYTHONPATH=src python scripts/run_experiment.py \
    --config "${CONFIG}" \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --n_problems "${N_PROBLEMS:-500}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_level INFO
