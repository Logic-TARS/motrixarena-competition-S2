#!/bin/bash
# K1 Speed Recovery Training
#
# Resumes from the walk checkpoint (model_1200) and fine-tunes with:
#   - straight-dominant command sampling (55% straight, 10% turn, 5% mixed)
#   - no forward-yaw envelope (real [vx=1, w≈0] commands in training)
#   - 10-second command hold time
#   - turn/direction-change/sprint rewards zeroed
#
# Goal: restore the old model_3600's long-range straight-line chasing
# capability while keeping the current vx>=0 forward-only convention.
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- base walk checkpoint (model_1200 from turn-robust run) ----
BASE_POLICY="${BASE_POLICY:-runs/k1-flat-terrain-walk/rslrl/26-06-11_10-54-03-_233224_PPO/model_1200.pt}"

# ---- trainable parameters ----
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-600}"
RESUME_NOISE_STD="${RESUME_NOISE_STD:-0.20}"
SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-8}"
SMOKE_ITERATIONS="${SMOKE_ITERATIONS:-2}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$BASE_POLICY" ]]; then
    echo "ERR  Base walk policy not found: $BASE_POLICY" >&2
    echo "     Set BASE_POLICY=/path/to/model_NNN.pt" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERR  uv is required. Install with: python3 -m pip install --user uv" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== K1 Speed Recovery Training ==="
echo "Base:    $BASE_POLICY"
echo "Envs:    $NUM_ENVS"
echo "Iters:   $MAX_ITERATIONS"
echo "Noise:   $RESUME_NOISE_STD"
echo ""

uv sync --frozen --all-packages --extra rslrl

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

run_cmd uv run --frozen python scripts/check_k1_speed_recovery.py \
    "$BASE_POLICY" \
    --env k1-flat-terrain-walk-speed

if [[ "$SKIP_SMOKE" != "1" ]]; then
    echo "=== Smoke training: ${SMOKE_NUM_ENVS} envs x ${SMOKE_ITERATIONS} iterations ==="
    run_cmd uv run --frozen python -u scripts/train.py \
        --env k1-flat-terrain-walk-speed \
        --rllib rslrl \
        --num-envs "$SMOKE_NUM_ENVS" \
        --seed "$SEED" \
        --max-iterations "$SMOKE_ITERATIONS" \
        --resume-policy "$BASE_POLICY" \
        --resume-noise-std "$RESUME_NOISE_STD"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    run_cmd uv run --frozen python -u scripts/train.py \
        --env k1-flat-terrain-walk-speed \
        --rllib rslrl \
        --num-envs "$NUM_ENVS" \
        --seed "$SEED" \
        --max-iterations "$MAX_ITERATIONS" \
        --resume-policy "$BASE_POLICY" \
        --resume-noise-std "$RESUME_NOISE_STD" \
        "$@"
    exit 0
fi

exec uv run --frozen python -u scripts/train.py \
    --env k1-flat-terrain-walk-speed \
    --rllib rslrl \
    --num-envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max-iterations "$MAX_ITERATIONS" \
    --resume-policy "$BASE_POLICY" \
    --resume-noise-std "$RESUME_NOISE_STD" \
    "$@"
