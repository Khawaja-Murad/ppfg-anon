#!/bin/bash
#SBATCH --job-name=versaprm-smoke
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# VersaPRM loading + scoring smoke test (Day-13 V13.0 sprint).
#
# Usage:
#   sbatch slurm/run_versaprm_smoke.sh
#
# Loads UW-Madison-Lee-Lab/Llama-PRM800K base + UW-Madison-Lee-Lab/VersaPRM adapter
# and scores 5 math + 5 law chains hardcoded in scripts/versaprm_smoke.py.
# Writes results/versaprm_smoke.json.

set -euo pipefail

mkdir -p logs results

# --- Modules (identical pattern to slurm/run_experiment.sh) ---
module purge
module load StdEnv/2023
module load gcc/12.3 cuda/12.2 python/3.11
module load arrow
module load opencv/4.12.0

# --- Virtual environment ---
VENV_DIR="${HOME}/scratch/hyp-forest-venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: venv not found at ${VENV_DIR}; run setup_cc.sh on the login node first."
    exit 2
fi
source "${VENV_DIR}/bin/activate"

# --- HuggingFace cache: scratch, offline (compute nodes have no internet) ---
export HF_HOME="${HOME}/scratch/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Misc ---
export TOKENIZERS_PARALLELISM=false

# --- Run ---
echo "=========================================="
echo "VersaPRM smoke test"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  HF_HOME:  ${HF_HOME}"
echo "=========================================="

PYTHONPATH=src python scripts/versaprm_smoke.py

echo "=========================================="
echo "Done. Output: results/versaprm_smoke.json"
echo "=========================================="
