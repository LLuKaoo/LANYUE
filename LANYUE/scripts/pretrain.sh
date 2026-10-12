#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
: "${DATA_ROOT:?Set DATA_ROOT to the CE2-anchor v2 dataset directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${CONFIG:?Set CONFIG to an existing pretraining JSON configuration}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lanyue_base}"
NUM_GPUS="${NUM_GPUS:-1}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  -m lanyue.train --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}" "$@"
