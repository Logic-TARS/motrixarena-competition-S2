#!/bin/bash
# Recording helpers for the continuous push strategy worktree.
# This worktree runs find_ball + ContinuousPushController in simulation;
# --trajectory enables local diagnostics.

TEAM_SIZE="${TEAM_SIZE:-1}"
REAL_TIME="--real-time"
COLOR="${COLOR:-red}"
ROBOT_ID="${ROBOT_ID:-0}"
FIXED_CMD=""
REFEREE_ARG="--use-referee"
REFEREE_STATE_ARG=""
DURATION="${DURATION:-60}"
OUTPUT=""
SIM_EXTRA=""
TRAJECTORY_ENABLED=0
TRAJECTORY_DIR=""
DEMO_3V3=0
RED_DECIDER_PIDS=()

# Default to this worktree's simulator policy selection.  In f922da9 that means
# simulation/motrixsim/assets/policies/k1_walk_model_3600_motrixlab.pt.
# Pass --policy PATH only when intentionally testing an external model.
POLICY_ARG=""

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --team-size)       TEAM_SIZE="$2"; shift 2 ;;
            --real-time)       REAL_TIME="--real-time"; shift ;;
            --no-real-time)    REAL_TIME="--no-real-time"; shift ;;
            --policy)          POLICY_ARG="--policy $2 --k1-policy-flavor motrixlab"; shift 2 ;;
            --no-policy)       POLICY_ARG=""; shift ;;
            --color)           COLOR="$2"; shift 2 ;;
            --id)              ROBOT_ID="$2"; shift 2 ;;
            --play)            FIXED_CMD=""; shift ;;
            --sim-fixed-cmd)   FIXED_CMD="--sim-fixed-cmd $2"; shift 2 ;;
            --use-referee)     REFEREE_ARG="--use-referee"; shift ;;
            --no-use-referee)  REFEREE_ARG="--no-use-referee"; shift ;;
            --referee-state)   echo "--referee-state is not supported in this f922da9 worktree; ignoring '$2'."; shift 2 ;;
            --policy-debug)    SIM_EXTRA="$SIM_EXTRA --policy-debug"; shift ;;
            --policy-debug-interval) SIM_EXTRA="$SIM_EXTRA --policy-debug-interval $2"; shift 2 ;;
            --t|--trajectory)  TRAJECTORY_ENABLED=1; shift ;;
            --trajectory-dir)  TRAJECTORY_ENABLED=1; TRAJECTORY_DIR="$2"; shift 2 ;;
            --d)               DURATION="$2"; shift 2 ;;
            --output)          OUTPUT="$2"; shift 2 ;;
            --demo-3v3)        DEMO_3V3=1; TEAM_SIZE=3; shift ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

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
    echo "Policy: ${POLICY_ARG:-(old simulator default)}"
    echo ""
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$REPO_ROOT/simulation/motrixsim${PYTHONPATH:+:$PYTHONPATH}"
    uv run --directory "$REPO_ROOT/MotrixLab" \
        --with absl-py --with pyzmq --with flask --with flask-socketio --with pillow \
        python -u -m app.runner \
        --team-size "$TEAM_SIZE" \
        $REAL_TIME \
        $REFEREE_ARG \
        $POLICY_ARG \
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
    echo "Strategy: continuous push"
    echo "Cmd: ${FIXED_CMD:-(find_ball + ContinuousPushController)}"
    if [[ "$TRAJECTORY_ENABLED" -eq 1 ]]; then
        echo "Trajectory: ${TRAJECTORY_DIR:-(Decider default directory)}"
    fi
    echo ""
    export PYTHONPATH="$REPO_ROOT/decider${PYTHONPATH:+:$PYTHONPATH}"
    uv run --directory "$REPO_ROOT/MotrixLab" --with pyyaml --with pyzmq --with transitions --with matplotlib python -u "$REPO_ROOT/decider/decider.py" \
        --simulation --ip 127.0.0.1 --port 5555 \
        --color "$COLOR" --id "$ROBOT_ID" \
        $FIXED_CMD \
        "${trajectory_args[@]}" &
    DECIDER_PID=$!
    echo "Decider PID: $DECIDER_PID"
}

launch_deciders_3v3() {
    local base_traj_dir="$TRAJECTORY_DIR"
    local ids
    ids=(0 1 2)
    echo "=== Starting 3 Red Deciders (0=attacker, 1=support, 2=defender) ==="
    echo "Color: red  IDs: 0 1 2"
    echo "Strategy: continuous_push 3v3 demo"
    echo "Red deciders: 0 attacker, 1 support, 2 defender"
    echo "Policy: ${POLICY_ARG:-(old simulator default)}"
    echo ""
    RED_DECIDER_PIDS=()
    for rid in "${ids[@]}"; do
        local trajectory_args=()
        if [[ "$TRAJECTORY_ENABLED" -eq 1 ]]; then
            trajectory_args+=(--record-trajectory)
            if [[ -n "$base_traj_dir" ]]; then
                trajectory_args+=(--trajectory-dir "$base_traj_dir/robot_$rid")
            fi
        fi
        export PYTHONPATH="$REPO_ROOT/decider${PYTHONPATH:+:$PYTHONPATH}"
        uv run --directory "$REPO_ROOT/MotrixLab" --with pyyaml --with pyzmq --with transitions --with matplotlib \
            python -u "$REPO_ROOT/decider/decider.py" \
            --simulation --ip 127.0.0.1 --port 5555 \
            --color red --id "$rid" \
            $FIXED_CMD \
            "${trajectory_args[@]}" &
        local pid=$!
        RED_DECIDER_PIDS+=("$pid")
        echo "  Decider id=$rid PID: $pid"
        sleep 0.3
    done
    echo ""
}

webview_url() {
    echo "http://localhost:5811"
}
