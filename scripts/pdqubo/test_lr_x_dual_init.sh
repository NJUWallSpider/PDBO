#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" research/compare_lr_x_dual_init.py \
    --max-iters 20000 \
    --sample-every 500 \
    --lr-y 0.01 \
    --dual-patience-threshold 1e-4 \
    --dual-patience-every 100 \
    "$@"
