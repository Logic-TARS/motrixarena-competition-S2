#!/bin/bash
# One-click: record a match and produce a video.
# Default: continuous PushToGoal in STATE_PLAYING.
# Usage: ./scripts/record_match.sh                  # PushToGoal, 60 s
#        ./scripts/record_match.sh --play           # compatibility alias
#        ./scripts/record_match.sh --d 30 --output /tmp/my_match.mp4
#        ./scripts/record_match.sh --trajectory --d 30
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/match_config.sh"
parse_args "$@"
check_uv

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${OUTPUT:-$REPO_ROOT/video/match_${TIMESTAMP}.mp4}"
if [[ "$TRAJECTORY_ENABLED" -eq 1 ]]; then
    TRAJECTORY_DIR="${TRAJECTORY_DIR:-$REPO_ROOT/video/trajectory_${TIMESTAMP}}"
    mkdir -p "$TRAJECTORY_DIR"
fi
FRAME_DIR="$(mktemp -d /tmp/sim_frames_XXXXX)"
SIM_EXTRA="$SIM_EXTRA --record-video $FRAME_DIR"

# --- cleanup ---
cleanup() {
    echo "=== Stopping ==="
    if [[ -n "${DECIDER_PID:-}" ]]; then
        kill "$DECIDER_PID" 2>/dev/null || true
        wait "$DECIDER_PID" 2>/dev/null || true
    fi
    if [[ -n "${SIM_PID:-}" ]]; then
        kill "$SIM_PID" 2>/dev/null || true
        wait "$SIM_PID" 2>/dev/null || true
    fi
    echo "=== Encoding video ==="
    cd "$REPO_ROOT"
    mkdir -p "$(dirname "$OUTPUT")"
    ffmpeg -y -framerate 30 -i "$FRAME_DIR/frame_%06d.png" \
        -c:v libx264 -pix_fmt yuv420p -loglevel warning "$OUTPUT" && \
        echo "Done: $OUTPUT" || echo "FFmpeg failed. Frames in: $FRAME_DIR"
    if [[ "$TRAJECTORY_ENABLED" -eq 1 ]]; then
        echo "Trajectory: $TRAJECTORY_DIR"
    fi
    rm -rf "$FRAME_DIR"
}
trap cleanup EXIT

kill_stale

launch_sim
sleep 2
launch_decider

echo "Recording ${DURATION}s → $OUTPUT"
echo ""
sleep "$DURATION"
echo "=== Time's up ==="
