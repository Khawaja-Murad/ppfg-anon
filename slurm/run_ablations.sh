#!/bin/bash
#SBATCH --job-name=hyp-forest-ablations
#SBATCH --account=<YOUR_SLURM_ACCOUNT>               # 
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-26                         # 3*3*3 = 27 ablation cells

# PPFG ablations: τ_kill × k_min × injection_rule
# 3 × 3 × 3 = 27 cells, each on 100-problem MATH500 subset

set -euo pipefail

TASK=${1:-math500}

PRUNE_THRESHOLDS=(0.3 0.5 0.7)               # τ_kill
K_MINS=(2 4 6)                               # min fragment length
INJECTION_RULES=("compat" "stagnation" "random")

N_TH=${#PRUNE_THRESHOLDS[@]}
N_KM=${#K_MINS[@]}
N_IR=${#INJECTION_RULES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
TH_IDX=$((IDX / (N_KM * N_IR)))
REM=$((IDX % (N_KM * N_IR)))
KM_IDX=$((REM / N_IR))
IR_IDX=$((REM % N_IR))

PRUNE_THRESHOLD=${PRUNE_THRESHOLDS[$TH_IDX]}
K_MIN=${K_MINS[$KM_IDX]}
INJECTION_RULE=${INJECTION_RULES[$IR_IDX]}

# Build a temp config file by overriding base
TMP_CONFIG=$(mktemp --suffix=.yaml)
cat > "${TMP_CONFIG}" <<EOF
defaults:
  - default

experiment:
  name: "abl-prune${PRUNE_THRESHOLD}-kmin${K_MIN}-${INJECTION_RULE}"

baseline:
  method: "ppfg"

population:
  prune_threshold: ${PRUNE_THRESHOLD}

ppfg:
  enabled: true
  k_min: ${K_MIN}
  injection_rule: "${INJECTION_RULE}"

task:
  n_problems: 100   # ablation budget — keep small
EOF

# Need configs/ in the path the runner resolves `defaults` against; copy temp config there
ABLATION_CONFIG="configs/_abl_${IDX}.yaml"
cp "${TMP_CONFIG}" "${ABLATION_CONFIG}"

mkdir -p logs results

module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11 arrow opencv/4.12.0

VENV_DIR="${HOME}/scratch/hyp-forest-venv"
source "${VENV_DIR}/bin/activate"

export HF_HOME="${HOME}/scratch/hf_cache"
# Compute nodes have no internet — force offline so HF Hub HEAD calls fail
# immediately instead of wasting ~3 min per file on retry+backoff.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}

RUN_NAME="abl-prune${PRUNE_THRESHOLD}-kmin${K_MIN}-${INJECTION_RULE}"
OUTPUT_DIR="results/ablations/${RUN_NAME}"

echo "Ablation: prune=${PRUNE_THRESHOLD} k_min=${K_MIN} rule=${INJECTION_RULE}"

PYTHONPATH=src python scripts/run_experiment.py \
    --config "${ABLATION_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_level INFO

# Cleanup
rm -f "${ABLATION_CONFIG}"
