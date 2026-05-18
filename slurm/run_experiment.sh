#!/bin/bash
#SBATCH --job-name=hyp-forest
#SBATCH --account=<YOUR_SLURM_ACCOUNT>               # 
#SBATCH --gres=gpu:a100:1                    # 1x A100 (40GB or 80GB depending on cluster)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00                      # 8 hours; bump to 24:00:00 for full sweeps
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# the HPC system launcher for one experiment run.
#
# Usage:
#   sbatch slurm/run_experiment.sh <method> <task> [config]
#
# Examples:
#   sbatch slurm/run_experiment.sh independent math500
#   sbatch slurm/run_experiment.sh ppfg math500 configs/ppfg.yaml
#   sbatch slurm/run_experiment.sh smc gpqa
#
# Notes on the HPC system clusters:
#   - HPC A100 80GB, recommended for vLLM tensor_parallel=1 with 7B models
#   - Cedar: V100/P100 — DO NOT USE; vLLM bf16 needs A100/H100 SM80+
#   - Beluga: V100 — same; avoid
#   - DRAC's Trillium / future: H100 — works fine

set -euo pipefail

METHOD=${1:-independent}
TASK=${2:-math500}
CONFIG=${3:-configs/default.yaml}

mkdir -p logs results

# --- Module loading ---
# the HPC system uses StdEnv/2023 by default with python/3.11. CUDA via cudacore.
module purge
module load StdEnv/2023
module load gcc/12.3 cuda/12.2 python/3.11
module load arrow                              # speeds up datasets
module load opencv/4.12.0                      # vLLM's mistral-common[opencv] needs cv2 from the module, not pip

# --- Virtual environment ---
# Best practice on CC: create venv on the COMPUTE NODE in $SLURM_TMPDIR for fast access,
# OR persist in $HOME/scratch. We persist for reuse across jobs.
VENV_DIR="${HOME}/scratch/hyp-forest-venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating venv at ${VENV_DIR}"
    python -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --no-index --upgrade pip
    # CC has wheels mirrored locally; --no-index uses them
    pip install --no-index -r requirements.txt || pip install -r requirements.txt
else
    source "${VENV_DIR}/bin/activate"
fi

# --- HuggingFace cache: put on scratch, not home (home is small/quota-limited) ---
export HF_HOME="${HOME}/scratch/hf_cache"
# Do NOT set TRANSFORMERS_CACHE — it's deprecated and overrides HF_HOME's hub/
# subdir, sending transformers/vLLM to look at the wrong path. HF_HOME alone is
# correct for recent transformers (4.x) and huggingface_hub.
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}"
# Compute nodes have no internet — force offline mode so missing cache is an
# immediate error rather than a hang + retry loop.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- vLLM tweaks for CC ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
# A100s on the HPC cluster are 40GB; if 80GB use 0.9
export VLLM_GPU_MEMORY_UTILIZATION=0.55

# --- Wandb (offline mode if no internet on compute node) ---
# CC compute nodes may lack internet — set offline. Sync after.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="${HOME}/scratch/wandb"
mkdir -p "${WANDB_DIR}"

# --- Run ---
RUN_NAME="${METHOD}-${TASK}-$(date +%Y%m%d-%H%M%S)-j${SLURM_JOB_ID:-local}"
OUTPUT_DIR="results/${RUN_NAME}"

echo "=========================================="
echo "Hypothesis Forest experiment"
echo "  Method:   ${METHOD}"
echo "  Task:     ${TASK}"
echo "  Config:   ${CONFIG}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

PYTHONPATH=src python scripts/run_experiment.py \
    --config "${CONFIG}" \
    --method "${METHOD}" \
    --output_dir "${OUTPUT_DIR}" \
    ${SEED:+--seed ${SEED}} \
    --log_level INFO

echo "=========================================="
echo "Done. Results in ${OUTPUT_DIR}"
echo "=========================================="
