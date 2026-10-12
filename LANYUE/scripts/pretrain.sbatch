#!/usr/bin/env bash
#SBATCH --job-name=lanyue_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=lanyue-%j.out
#SBATCH --error=lanyue-%j.err
set -euo pipefail
# Run sbatch from the repository root after activating your Python environment.
# Select a partition/account/time limit according to your own cluster.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export NUM_GPUS="${NUM_GPUS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec bash scripts/pretrain.sh "$@"
