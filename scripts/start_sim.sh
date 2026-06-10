#!/bin/bash
# Start K1 soccer simulation with MotrixLab-trained policy
# Usage: ./scripts/start_sim.sh [--team-size N] [--no-real-time] [--policy PATH] [--blue-policy PATH]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEAM_SIZE="${TEAM_SIZE:-1}"
REAL_TIME="--real-time"
# Use exported TorchScript model (with built-in EmpiricalNormalization from RSLRL training)
POLICY_ARG="--policy $REPO_ROOT/MotrixLab/exported/model_1350_torchscript.pt"
BLUE_POLICY_ARG=""
RECORD_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --team-size) TEAM_SIZE="$2"; shift 2 ;;
        --no-real-time) REAL_TIME="--no-real-time"; shift ;;
        --policy) POLICY_ARG="--policy $2"; shift 2 ;;
        --blue-policy) BLUE_POLICY_ARG="--blue-policy $2 --blue-policy-flavor legged_gym"; shift 2 ;;
        --record-video) RECORD_ARG="--record-video $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check uv is available
if ! command -v uv &>/dev/null; then
    echo "uv is required. Install: python3 -m pip install --user uv" >&2
    exit 1
fi

# Kill any stale sim on the same ports
fuser -k 5555/tcp 2>/dev/null || true
fuser -k 5811/tcp 2>/dev/null || true
sleep 0.5

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/simulation/motrixsim${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Starting K1 Simulation ==="
echo "Team size: $TEAM_SIZE"
echo "WebView:   http://localhost:5811"
echo "ZMQ:       tcp://*:5555"
echo ""

exec uv run --directory "$REPO_ROOT/MotrixLab" python -u -m app.runner \
    --team-size "$TEAM_SIZE" \
    $REAL_TIME \
    $POLICY_ARG \
    $BLUE_POLICY_ARG \
    $RECORD_ARG
