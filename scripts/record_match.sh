#!/bin/bash
# One-click: record a match and produce a video.
# Default: straight-line walk test.  Edit match_config.sh to change defaults.
# Usage: ./scripts/record_match.sh                  # walk test, 20 s
#        ./scripts/record_match.sh --play           # full game (DeciderFSM)
#        ./scripts/record_match.sh --d 30 --output /tmp/my_match.mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/match_config.sh"
parse_args "$@"
check_uv

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${OUTPUT:-$REPO_ROOT/video/match_${TIMESTAMP}.mp4}"
FRAME_DIR="$(mktemp -d /tmp/sim_frames_XXXXX)"
SIM_EXTRA="--record-video $FRAME_DIR"

# --- cleanup ---
cleanup() {
    echo "=== Stopping ==="
    kill $SIM_PID $DECIDER_PID 2>/dev/null || true
    wait $SIM_PID $DECIDER_PID 2>/dev/null || true
    echo "=== Encoding video ==="
    cd "$REPO_ROOT"
    mkdir -p "$(dirname "$OUTPUT")"
    ffmpeg -y -framerate 30 -i "$FRAME_DIR/frame_%06d.png" \
        -c:v libx264 -pix_fmt yuv420p -loglevel warning "$OUTPUT" && \
        echo "Done: $OUTPUT" || echo "FFmpeg failed. Frames in: $FRAME_DIR"
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
