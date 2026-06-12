#!/bin/bash
# One-click recording for the continuous push strategy worktree.
# Default: this worktree's simulator policy plus the simulation decision path:
# find_ball + ContinuousPushController.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/match_config.sh"
parse_args "$@"
check_uv

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${OUTPUT:-$REPO_ROOT/video/match_${TIMESTAMP}.mp4}"
if [[ "$TRAJECTORY_ENABLED" -eq 1 && -z "$TRAJECTORY_DIR" ]]; then
    TRAJECTORY_DIR="$REPO_ROOT/video/trajectory_${TIMESTAMP}"
fi
FRAME_DIR="$(mktemp -d /tmp/sim_frames_XXXXX)"
SIM_EXTRA="$SIM_EXTRA --record-video $FRAME_DIR"
HAVE_FRAMES=0

wait_for_first_frame() {
    local timeout_s="${1:-30}"
    local start_s
    start_s="$(date +%s)"
    echo "Waiting for simulation frames..."
    while true; do
        if compgen -G "$FRAME_DIR/frame_*.png" >/dev/null; then
            HAVE_FRAMES=1
            return 0
        fi
        if [[ -n "${SIM_PID:-}" ]] && ! kill -0 "$SIM_PID" 2>/dev/null; then
            echo "Simulation exited before writing frames." >&2
            return 1
        fi
        if (( $(date +%s) - start_s >= timeout_s )); then
            echo "Timed out waiting for frames in $FRAME_DIR" >&2
            return 1
        fi
        sleep 0.5
    done
}

record_for_duration() {
    local duration_s="$1"
    local start_s
    start_s="$(date +%s)"
    while (( $(date +%s) - start_s < duration_s )); do
        if [[ -n "${SIM_PID:-}" ]] && ! kill -0 "$SIM_PID" 2>/dev/null; then
            echo "Simulation exited during recording." >&2
            return 1
        fi
        if [[ "$DEMO_3V3" -eq 1 ]]; then
            for pid in "${RED_DECIDER_PIDS[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "A decider (PID $pid) exited during recording." >&2
                    return 1
                fi
            done
        elif [[ -n "${DECIDER_PID:-}" ]] && ! kill -0 "$DECIDER_PID" 2>/dev/null; then
            echo "Decider exited during recording." >&2
            return 1
        fi
        sleep 0.5
    done
}

cleanup() {
    echo "=== Stopping ==="
    if [[ "$DEMO_3V3" -eq 1 && -n "${RED_DECIDER_PIDS+x}" ]]; then
        for pid in "${RED_DECIDER_PIDS[@]}"; do
            kill -INT "$pid" 2>/dev/null || true
        done
        for pid in "${RED_DECIDER_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    elif [[ -n "${DECIDER_PID:-}" ]]; then
        kill -INT "$DECIDER_PID" 2>/dev/null || true
        wait "$DECIDER_PID" 2>/dev/null || true
    fi
    if [[ -n "${SIM_PID:-}" ]]; then
        kill "$SIM_PID" 2>/dev/null || true
        wait "$SIM_PID" 2>/dev/null || true
    fi
    echo "=== Encoding video ==="
    cd "$REPO_ROOT"
    mkdir -p "$(dirname "$OUTPUT")"
    if compgen -G "$FRAME_DIR/frame_*.png" >/dev/null; then
        ffmpeg -y -framerate 30 -i "$FRAME_DIR/frame_%06d.png" \
            -c:v libx264 -pix_fmt yuv420p -loglevel warning "$OUTPUT" && \
            echo "Done: $OUTPUT" || echo "FFmpeg failed. Frames in: $FRAME_DIR"
    else
        echo "No frames were recorded; skipping ffmpeg. Frames dir: $FRAME_DIR"
    fi
    rm -rf "$FRAME_DIR"
}
trap cleanup EXIT

kill_stale

launch_sim
sleep 2
if [[ "$DEMO_3V3" -eq 1 ]]; then
    launch_deciders_3v3
else
    launch_decider
fi
wait_for_first_frame 30
if [[ "$DEMO_3V3" -eq 1 ]]; then
    for pid in "${RED_DECIDER_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "A decider (PID $pid) exited before recording started." >&2
            exit 1
        fi
    done
elif [[ -n "${DECIDER_PID:-}" ]] && ! kill -0 "$DECIDER_PID" 2>/dev/null; then
    echo "Decider exited before recording started." >&2
    exit 1
fi

echo "Recording ${DURATION}s -> $OUTPUT"
echo ""
record_for_duration "$DURATION"
echo "=== Time's up ==="
