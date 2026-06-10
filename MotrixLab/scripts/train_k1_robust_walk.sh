#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_POLICY="${BASE_POLICY:-runs/k1-flat-terrain-walk/rslrl/26-06-08_21-37-09-_386985_PPO/model_1350.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"

if [[ ! -f "$BASE_POLICY" ]]; then
    echo "Base walk policy not found: $BASE_POLICY" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it with: python3 -m pip install --user uv" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv sync --frozen --all-packages --extra rslrl

exec uv run --frozen python -u scripts/train.py \
    --env k1-flat-terrain-walk \
    --rllib rslrl \
    --num-envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max-iterations "$MAX_ITERATIONS" \
    --resume-policy "$BASE_POLICY" \
    --resume-noise-std "${RESUME_NOISE_STD:-0.30}" \
    "$@"
