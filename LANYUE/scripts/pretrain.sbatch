#!/usr/bin/env bash
#SBATCH --job-name=lanyue_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=lanyue-%j.out
#SBATCH --error=lanyue-%j.err
set -euo pipefail
# Submit from the repository root or model directory after activating your environment.
# Select a partition/account/time limit according to your own cluster.
MODEL_DIR="${SLURM_SUBMIT_DIR:?Submit from the repository root or model directory}"
if [[ -f "${MODEL_DIR}/LANYUE/scripts/pretrain.sh" ]]; then
  MODEL_DIR="${MODEL_DIR}/LANYUE"
fi
if [[ ! -f "${MODEL_DIR}/scripts/pretrain.sh" ]]; then
  printf 'Submit from the LANYUE repository root or its model directory.\n' >&2
  exit 1
fi
cd "${MODEL_DIR}"
export NUM_GPUS="${NUM_GPUS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec bash scripts/pretrain.sh "$@"
