#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it with: python3 -m pip install --user uv" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv sync --frozen --all-packages --extra rslrl

exec uv run --frozen python -u scripts/train.py \
    --env k1-flat-terrain-walk \
    --rllib rslrl \
    --num-envs 4096 \
    --seed 1 \
    "$@"
