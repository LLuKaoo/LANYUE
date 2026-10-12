#!/usr/bin/env bash
#SBATCH --job-name=lanyue_nac_distill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=lanyue-nac-%j.out
#SBATCH --error=lanyue-nac-%j.err
set -euo pipefail
# Activate the Python environment and submit from the repository root.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export NUM_GPUS="${NUM_GPUS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec bash scripts/distill_nac.sh "$@"
