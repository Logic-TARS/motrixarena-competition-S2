#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec conda run -n sim_soccer_rl python -u scripts/train.py \
    --env k1-flat-terrain-walk \
    --rllib rslrl \
    --num-envs 2048 \
    "$@"
