#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-1 2 3}"
CONTINUOUS="${CONTINUOUS:-1}"
round=0
while true; do
    for seed in $SEEDS; do
        effective_seed=$((seed + round * 1000))
        echo "=== K1 get-up training seed=$effective_seed round=$round ==="
        SEED="$effective_seed" ./scripts/train_k1_getup.sh "$@"
    done
    if [[ "$CONTINUOUS" != "1" ]]; then
        break
    fi
    round=$((round + 1))
done
