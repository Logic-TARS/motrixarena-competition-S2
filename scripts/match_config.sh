#!/bin/bash
# match_config.sh — single source of truth for watch.sh and record_match.sh.
# Sourced by both scripts; defines defaults, CLI parsing, and launch helpers.
# To change defaults (team size, fixed cmd, etc.), edit THIS file only.

# ============================================================================
# Defaults — override via env or CLI
# ============================================================================
TEAM_SIZE="${TEAM_SIZE:-1}"
REAL_TIME="--real-time"
POLICY_ARG="--policy $REPO_ROOT/MotrixLab/exported/model_1200_turn_robust_torchscript.pt --k1-policy-flavor motrixlab"
BLUE_POLICY_ARG=""
COLOR="${COLOR:-red}"
ROBOT_ID="${ROBOT_ID:-0}"
FIXED_CMD="--sim-fixed-cmd 0.5,0,0"   # default: straight-line walk test
REFEREE_ARG=""
REFEREE_STATE_ARG=""
DURATION="${DURATION:-60}"
OUTPUT=""
SIM_EXTRA=""       # extra flags passed to sim launcher (e.g. --record-video)
TRAJECTORY_ENABLED=0
TRAJECTORY_DIR=""

# ============================================================================
# CLI parsing — handles args for BOTH watch.sh and record_match.sh
# ============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --team-size)       TEAM_SIZE="$2"; shift 2 ;;
            --real-time)       REAL_TIME="--real-time"; shift ;;
            --no-real-time)    REAL_TIME="--no-real-time"; shift ;;
            --policy)          POLICY_ARG="--policy $2"; shift 2 ;;
            --blue-policy)     BLUE_POLICY_ARG="--blue-policy $2 --blue-policy-flavor legged_gym"; shift 2 ;;
            --color)           COLOR="$2"; shift 2 ;;
            --id)              ROBOT_ID="$2"; shift 2 ;;
            --play)            FIXED_CMD="";
                               [[ -n "$REFEREE_ARG" ]] || REFEREE_ARG="--use-referee";
                               [[ -n "$REFEREE_STATE_ARG" ]] || REFEREE_STATE_ARG="--referee-state playing";
                               shift ;;
            --sim-fixed-cmd)   FIXED_CMD="--sim-fixed-cmd $2"; shift 2 ;;
            --use-referee)     REFEREE_ARG="--use-referee"; shift ;;
            --no-use-referee)  REFEREE_ARG="--no-use-referee"; shift ;;
            --referee-state)   REFEREE_STATE_ARG="--referee-state $2"; shift 2 ;;
            --policy-debug)    SIM_EXTRA="$SIM_EXTRA --policy-debug"; shift ;;
            --policy-debug-interval) SIM_EXTRA="$SIM_EXTRA --policy-debug-interval $2"; shift 2 ;;
            --t)       TRAJECTORY_ENABLED=1; shift ;;
            --trajectory-dir)   TRAJECTORY_ENABLED=1; TRAJECTORY_DIR="$2"; shift 2 ;;
            --d)               DURATION="$2"; shift 2 ;;
            --output)          OUTPUT="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

# ============================================================================
# Shared helpers
# ============================================================================
check_uv() {
    if ! command -v uv &>/dev/null; then
        echo "uv is required. Install: python3 -m pip install --user uv" >&2
        exit 1
    fi
}

kill_stale() {
    fuser -k 5555/tcp 2>/dev/null || true
    fuser -k 5811/tcp 2>/dev/null || true
    sleep 0.5
}

launch_sim() {
    echo "=== Starting Simulation ==="
    echo "Team size: $TEAM_SIZE"
    echo ""
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$REPO_ROOT/simulation/motrixsim${PYTHONPATH:+:$PYTHONPATH}"
    uv run --directory "$REPO_ROOT/MotrixLab" python -u -m app.runner \
        --team-size "$TEAM_SIZE" \
        $REAL_TIME \
        $REFEREE_ARG \
        $REFEREE_STATE_ARG \
        $POLICY_ARG \
        $BLUE_POLICY_ARG \
        $SIM_EXTRA &
    SIM_PID=$!
    echo "Sim PID: $SIM_PID"
}

launch_decider() {
    local trajectory_args=()
    if [[ "$TRAJECTORY_ENABLED" -eq 1 ]]; then
        trajectory_args+=(--record-trajectory)
        if [[ -n "$TRAJECTORY_DIR" ]]; then
            trajectory_args+=(--trajectory-dir "$TRAJECTORY_DIR")
        fi
    fi

    echo "=== Starting Decider ==="
    echo "Color: $COLOR  ID: $ROBOT_ID"
    echo "Cmd:   ${FIXED_CMD:-(game FSM)}"
    if [[ "$TRAJECTORY_ENABLED" -eq 1 ]]; then
        echo "Trajectory: ${TRAJECTORY_DIR:-(Decider default directory)}"
    fi
    echo ""
    export PYTHONPATH="$REPO_ROOT/decider${PYTHONPATH:+:$PYTHONPATH}"
    uv run --directory "$REPO_ROOT/MotrixLab" python -u "$REPO_ROOT/decider/decider.py" \
        --simulation --ip 127.0.0.1 --port 5555 \
        --color "$COLOR" --id "$ROBOT_ID" \
        $FIXED_CMD \
        "${trajectory_args[@]}" &
    DECIDER_PID=$!
    echo "Decider PID: $DECIDER_PID"
}

webview_url() {
    echo "http://localhost:5811"
}
