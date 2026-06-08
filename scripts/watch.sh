#!/bin/bash
# One-click: launch simulation + decider and open WebView for real-time watching.
# Usage: ./scripts/watch.sh [--team-size N] [--color red|blue] [--id N]
#        ./scripts/watch.sh --sim-fixed-cmd "0.5,0,0"  # straight-line test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- defaults ---
TEAM_SIZE="${TEAM_SIZE:-1}"
REAL_TIME="--real-time"
POLICY_ARG=""
BLUE_POLICY_ARG=""
COLOR="${COLOR:-red}"
ROBOT_ID="${ROBOT_ID:-0}"
FIXED_CMD=""
WEBVIEW_URL="http://localhost:5811"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --team-size)       TEAM_SIZE="$2"; shift 2 ;;
        --no-real-time)    REAL_TIME="--no-real-time"; shift ;;
        --policy)          POLICY_ARG="--policy $2"; shift 2 ;;
        --blue-policy)     BLUE_POLICY_ARG="--blue-policy $2 --blue-policy-flavor legged_gym"; shift 2 ;;
        --color)           COLOR="$2"; shift 2 ;;
        --id)              ROBOT_ID="$2"; shift 2 ;;
        --sim-fixed-cmd)   FIXED_CMD="--sim-fixed-cmd $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! command -v uv &>/dev/null; then
    echo "uv is required. Install: python3 -m pip install --user uv" >&2
    exit 1
fi

# --- cleanup on exit ---
cleanup() {
    echo ""
    echo "=== Stopping ==="
    kill $SIM_PID $DECIDER_PID 2>/dev/null || true
    wait $SIM_PID $DECIDER_PID 2>/dev/null || true
    echo "=== Done ==="
}
trap cleanup EXIT

# --- kill stale processes ---
fuser -k 5555/tcp 2>/dev/null || true
fuser -k 5811/tcp 2>/dev/null || true
sleep 0.5

# --- launch simulation (background) ---
echo "=== Starting Simulation ==="
echo "Team size: $TEAM_SIZE"
echo "WebView:   $WEBVIEW_URL"
echo ""

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/simulation/motrixsim${PYTHONPATH:+:$PYTHONPATH}"
uv run --directory "$REPO_ROOT/MotrixLab" python -u -m app.runner \
    --team-size "$TEAM_SIZE" \
    $REAL_TIME \
    $POLICY_ARG \
    $BLUE_POLICY_ARG &
SIM_PID=$!
echo "Sim PID: $SIM_PID"

# --- wait for sim ZMQ to be ready ---
sleep 2

# --- launch decider (background) ---
echo "=== Starting Decider ==="
echo "Color: $COLOR  ID: $ROBOT_ID"
echo ""

export PYTHONPATH="$REPO_ROOT/decider${PYTHONPATH:+:$PYTHONPATH}"
uv run --directory "$REPO_ROOT/MotrixLab" python -u "$REPO_ROOT/decider/decider.py" \
    --simulation --ip 127.0.0.1 --port 5555 \
    --color "$COLOR" --id "$ROBOT_ID" \
    $FIXED_CMD &
DECIDER_PID=$!
echo "Decider PID: $DECIDER_PID"

# --- open web viewer ---
echo ""
echo "=== Opening WebView: $WEBVIEW_URL ==="
if command -v xdg-open &>/dev/null; then
    xdg-open "$WEBVIEW_URL" 2>/dev/null || true
elif command -v open &>/dev/null; then
    open "$WEBVIEW_URL" 2>/dev/null || true
fi

echo ""
echo "Press Ctrl+C to stop."
echo ""

# --- wait forever (Ctrl+C triggers cleanup) ---
wait
