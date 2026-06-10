#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it with: python3 -m pip install --user uv" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_ENVS="${NUM_ENVS:-2048}"
SEED="${SEED:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"

uv sync --frozen --all-packages --extra rslrl

exec uv run --frozen python -u scripts/train.py \
    --env k1-getup \
    --rllib rslrl \
    --num-envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max-iterations "$MAX_ITERATIONS" \
    "$@"
