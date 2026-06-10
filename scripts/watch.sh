#!/bin/bash
# One-click: launch simulation + decider and open WebView for real-time watching.
# Default: straight-line walk test.  Edit match_config.sh to change defaults.
# Usage: ./scripts/watch.sh                  # walk test
#        ./scripts/watch.sh --play           # full game (DeciderFSM)
#        ./scripts/watch.sh --team-size 3    # 3v3
#        ./scripts/watch.sh --play --trajectory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/match_config.sh"
parse_args "$@"
check_uv

if [[ "$TRAJECTORY_ENABLED" -eq 1 ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TRAJECTORY_DIR="${TRAJECTORY_DIR:-$REPO_ROOT/debug_logs/trajectory_${TIMESTAMP}}"
    mkdir -p "$TRAJECTORY_DIR"
fi

# --- cleanup ---
cleanup() {
    echo ""
    echo "=== Stopping ==="
    if [[ -n "${DECIDER_PID:-}" ]]; then
        kill "$DECIDER_PID" 2>/dev/null || true
        wait "$DECIDER_PID" 2>/dev/null || true
    fi
    if [[ -n "${SIM_PID:-}" ]]; then
        kill "$SIM_PID" 2>/dev/null || true
        wait "$SIM_PID" 2>/dev/null || true
    fi
    if [[ "$TRAJECTORY_ENABLED" -eq 1 ]]; then
        echo "Trajectory: $TRAJECTORY_DIR"
    fi
    echo "=== Done ==="
}
trap cleanup EXIT

kill_stale

launch_sim
sleep 2
launch_decider

# --- open WebView ---
URL="$(webview_url)"
echo "=== Opening WebView: $URL ==="
if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" 2>/dev/null || true
elif command -v open &>/dev/null; then
    open "$URL" 2>/dev/null || true
fi

echo "Press Ctrl+C to stop."
echo ""
wait
