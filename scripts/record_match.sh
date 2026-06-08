#!/bin/bash
# One-click: record a match and produce a video.
# Default: straight-line walk test (--sim-fixed-cmd "0.5,0,0")
# Usage: ./scripts/record_match.sh                          # walk test
#        ./scripts/record_match.sh --play                   # full game (DeciderFSM)
#        ./scripts/record_match.sh --duration 30 --output /tmp/match.mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DURATION="${DURATION:-20}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${OUTPUT:-$REPO_ROOT/video/match_${TIMESTAMP}.mp4}"
FIXED_CMD="--sim-fixed-cmd 0.5,0,0"   # default: straight-line walk test

while [[ $# -gt 0 ]]; do
    case "$1" in
        --d) DURATION="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --play) FIXED_CMD=""; shift ;;                      # switch to full game mode
        --sim-fixed-cmd) FIXED_CMD="--sim-fixed-cmd $2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

FRAME_DIR="$(mktemp -d /tmp/sim_frames_XXXXX)"
echo "=== Recording $DURATION s → $OUTPUT ==="

# Cleanup on exit
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

# Kill stale processes
fuser -k 5555/tcp 2>/dev/null || true
fuser -k 5811/tcp 2>/dev/null || true
sleep 0.5

# Launch simulation (background)
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/simulation/motrixsim${PYTHONPATH:+:$PYTHONPATH}"
uv run --directory "$REPO_ROOT/MotrixLab" python -u -m app.runner \
    --team-size 1 --record-video "$FRAME_DIR" &
SIM_PID=$!
echo "Sim PID: $SIM_PID"

# Wait for sim ZMQ to be ready
sleep 2

# Launch decider (background)
export PYTHONPATH="$REPO_ROOT/decider${PYTHONPATH:+:$PYTHONPATH}"
uv run --directory "$REPO_ROOT/MotrixLab" python -u "$REPO_ROOT/decider/decider.py" \
    --simulation --ip 127.0.0.1 --port 5555 --color red --id 0 \
    $FIXED_CMD &
DECIDER_PID=$!
echo "Decider PID: $DECIDER_PID"

# Run
echo "Recording for $DURATION seconds..."
sleep "$DURATION"

# Script exit triggers cleanup → ffmpeg
echo "=== Time's up ==="
