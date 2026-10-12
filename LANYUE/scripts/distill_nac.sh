#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
: "${DATA_ROOT:?Set DATA_ROOT to the CE2-anchor v2 retention dataset}"
: "${NAC_ROOT:?Set NAC_ROOT to the native NAC TAR/Zstd dataset}"
: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to a v2 five-modal checkpoint}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${CONFIG:?Set CONFIG to an existing NAC distillation JSON configuration}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lanyue_nac}"
NUM_GPUS="${NUM_GPUS:-1}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  -m lanyue.distill --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" --nac-root "${NAC_ROOT}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" --output-dir "${OUTPUT_DIR}" "$@"
