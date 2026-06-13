#!/bin/bash
# Start K1 soccer simulation and one decider together.
# Usage: ./scripts/start_all.sh [--team-size N] [--no-real-time] [--policy PATH] [--color red|blue] [--id N] [--ip IP] [--port PORT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEAM_SIZE="${TEAM_SIZE:-1}"
REAL_TIME_ARG=()
POLICY_ARG=()
COLOR="${COLOR:-red}"
ROBOT_ID="${ROBOT_ID:-0}"
IP="${IP:-127.0.0.1}"
PORT="${PORT:-5555}"

SIM_PID=""
DECIDER_PID=""
CLEANED_UP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --team-size)
            TEAM_SIZE="$2"
            shift 2
            ;;
        --no-real-time)
            REAL_TIME_ARG=(--no-real-time)
            shift
            ;;
        --policy)
            POLICY_ARG=(--policy "$2")
            shift 2
            ;;
        --color)
            COLOR="$2"
            shift 2
            ;;
        --id)
            ROBOT_ID="$2"
            shift 2
            ;;
        --ip)
            IP="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

cleanup() {
    if [[ "$CLEANED_UP" -eq 1 ]]; then
        return
    fi
    CLEANED_UP=1
    trap - INT TERM EXIT

    echo ""
    echo "=== Stopping ==="
    if [[ -n "$DECIDER_PID" ]] && kill -0 "$DECIDER_PID" 2>/dev/null; then
        kill -INT "$DECIDER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
        kill -INT "$SIM_PID" 2>/dev/null || true
    fi
    if [[ -n "$DECIDER_PID" ]]; then
        wait "$DECIDER_PID" 2>/dev/null || true
    fi
    if [[ -n "$SIM_PID" ]]; then
        wait "$SIM_PID" 2>/dev/null || true
    fi
}

trap cleanup INT TERM EXIT

echo "=== Starting K1 Simulation + Decider ==="
echo "Team size: $TEAM_SIZE"
echo "Decider:   $COLOR id=$ROBOT_ID"
echo "Target:    tcp://$IP:$PORT"
echo "WebView:   http://localhost:5811"
echo ""

"$SCRIPT_DIR/start_sim.sh" \
    --team-size "$TEAM_SIZE" \
    "${REAL_TIME_ARG[@]}" \
    "${POLICY_ARG[@]}" &
SIM_PID=$!

echo "Simulation PID: $SIM_PID"
sleep 2

if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "Simulation exited before Decider startup." >&2
    exit 1
fi

"$SCRIPT_DIR/start_decider.sh" \
    --color "$COLOR" \
    --id "$ROBOT_ID" \
    --ip "$IP" \
    --port "$PORT" &
DECIDER_PID=$!

echo "Decider PID:    $DECIDER_PID"
echo ""
echo "Running. Press Ctrl+C to stop both processes."

while true; do
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        echo "Simulation process exited."
        exit 1
    fi
    if ! kill -0 "$DECIDER_PID" 2>/dev/null; then
        echo "Decider process exited."
        exit 1
    fi
    sleep 1
done
