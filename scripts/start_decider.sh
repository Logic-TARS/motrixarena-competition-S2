#!/bin/bash
# Start a K1 decider (robot brain) connecting to simulation
# Usage: ./scripts/start_decider.sh [--color red|blue] [--id N] [--port PORT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COLOR="${COLOR:-red}"
ROBOT_ID="${ROBOT_ID:-0}"
PORT="${PORT:-5555}"
IP="${IP:-127.0.0.1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --color) COLOR="$2"; shift 2 ;;
        --id) ROBOT_ID="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ip) IP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! command -v uv &>/dev/null; then
    echo "uv is required. Install: python3 -m pip install --user uv" >&2
    exit 1
fi

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/decider${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Starting Decider ==="
echo "Color:  $COLOR"
echo "ID:     $ROBOT_ID"
echo "Target: tcp://$IP:$PORT"
echo ""

exec uv run --directory "$REPO_ROOT/MotrixLab" \
    --with pyyaml --with pyzmq --with transitions --with matplotlib \
    python -u "$REPO_ROOT/decider/decider.py" \
    --simulation \
    --ip "$IP" \
    --port "$PORT" \
    --color "$COLOR" \
    --id "$ROBOT_ID"
