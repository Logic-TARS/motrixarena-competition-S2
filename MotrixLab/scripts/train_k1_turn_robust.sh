#!/bin/bash
# K1 Sustained-Turn Robustness Training
#
# Resumes from the best walk checkpoint and fine-tunes with:
#   - mixed_turn command mode (vx > 0 + high yaw simultaneously)
#   - sustained-turn curriculum (hold high-yaw commands for ≥ 6 s)
#   - turn_stability reward (penalise tilt during turns)
#   - turn_survival bonus (extra alive reward during turns)
#
# This trains the 47→12 legged locomotion policy to maintain balance
# through the long-duration high-yaw / low-vx turns that soccer
# navigation requires (ESCAPE_FACE, orbit, side recovery).
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- base walk checkpoint (latest from prior run) ----
BASE_POLICY="${BASE_POLICY:-runs/k1-flat-terrain-walk/rslrl/26-06-10_23-59-14-_973548_PPO/model_650.pt}"

# ---- trainable parameters ----
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
RESUME_NOISE_STD="${RESUME_NOISE_STD:-0.30}"

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

echo "=== K1 Turn-Robust Training ==="
echo "Base:    $BASE_POLICY"
echo "Envs:    $NUM_ENVS"
echo "Iters:   $MAX_ITERATIONS"
echo "Noise:   $RESUME_NOISE_STD"
echo ""

uv sync --frozen --all-packages --extra rslrl

exec uv run --frozen python -u scripts/train.py \
    --env k1-flat-terrain-walk \
    --rllib rslrl \
    --num-envs "$NUM_ENVS" \
    --seed "$SEED" \
    --max-iterations "$MAX_ITERATIONS" \
    --resume-policy "$BASE_POLICY" \
    --resume-noise-std "$RESUME_NOISE_STD" \
    "$@"
