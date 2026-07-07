#!/bin/bash
# ===========================================================================
# loco_baseline.sh -- Locomotion model baseline test runner (T0-T9)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO]  $*"; }

# --- source for kill_stale / check_uv helpers -----------------------------
source "$SCRIPT_DIR/match_config.sh"

# --- defaults -------------------------------------------------------------
SUBSET=""
MODEL_TAG="default"
POLICY_ARG=""
LOCO_CONFIG="$REPO_ROOT/decider/scripts/loco_config.json"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_loco_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --subset)       SUBSET="$2"; shift 2 ;;
            --policy)       POLICY_ARG="--policy $2 --k1-policy-flavor motrixlab"
                            MODEL_TAG=$(basename "$2" .pt | sed 's/[^a-zA-Z0-9_-]//g'); shift 2 ;;
            --no-policy)    POLICY_ARG=""; shift ;;
            --help|-h)
                echo "Usage: $0 [--subset T0,T1,...] [--policy PATH]"
                echo "  --subset T0,T1,T4   Run only specified tests"
                echo "  --policy PATH       Use alternative policy checkpoint"
                exit 0 ;;
            *) shift ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Environment-specific launch functions (override match_config.sh versions)
# ---------------------------------------------------------------------------

launch_sim_loco() {
    info "  Launching sim (conda motrixsim0508) ..."
    export PYTHONUNBUFFERED=1
    PYTHONPATH="$REPO_ROOT/simulation/motrixsim${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n motrixsim0508 --no-capture-output \
        python -u -m app.runner \
        --team-size 1 \
        --no-real-time \
        --no-use-referee \
        --no-webview \
        $POLICY_ARG \
        > /tmp/loco_sim_$$.log 2>&1 &
    SIM_PID=$!
    info "  Sim PID: $SIM_PID"
}

launch_decider_loco() {
    local traj_dir="$1"
    local fixed_cmd_arg="$2"    # e.g. "--sim-fixed-cmd 0,0,0" or "--sim-fixed-cmd-seq [...]"
    shift 2 || true

    info "  Launching decider (uv venv) ..."
    uv run --directory "$REPO_ROOT/MotrixLab" \
        python -u "$REPO_ROOT/decider/decider.py" \
        --simulation --ip 127.0.0.1 --port 5555 \
        --color red --id 0 \
        $fixed_cmd_arg \
        --record-trajectory \
        --trajectory-dir "$traj_dir" \
        > /tmp/loco_decider_$$.log 2>&1 &
    DECIDER_PID=$!
    info "  Decider PID: $DECIDER_PID"
}

# ---------------------------------------------------------------------------
# Test matrix helpers
# ---------------------------------------------------------------------------

get_test_ids() {
    if [[ -n "$SUBSET" ]]; then
        IFS=',' read -ra IDS <<< "$SUBSET"
        for id in "${IDS[@]}"; do echo "$id"; done
    else
        python3 -c "import json; cfg=json.load(open('$LOCO_CONFIG')); [print(t) for t in cfg['tests']]"
    fi
}

# ---------------------------------------------------------------------------
# Run a single constant-command case
# ---------------------------------------------------------------------------
run_constant_case() {
    local test_id="$1"
    local cmd_str="$2"
    local duration="$3"
    local run_dir="$4"

    mkdir -p "$run_dir"
    info "  cmd=$cmd_str  duration=${duration}s"

    kill_stale
    launch_sim_loco
    sleep 4

    launch_decider_loco "$run_dir" "--sim-fixed-cmd $cmd_str"
    sleep 2

    info "  recording ${duration}s ..."
    sleep "$duration"

    # SIGINT for graceful shutdown (_finish_trajectory)
    kill -INT "$DECIDER_PID" 2>/dev/null || true
    sleep 2
    wait "$DECIDER_PID" 2>/dev/null || true
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
    kill_stale
    sleep 1

    info "  analyzing ..."
    cd "$REPO_ROOT"
    python3 -u "$REPO_ROOT/decider/scripts/analyze_loco_baseline.py" \
        "$run_dir/trajectory.csv" \
        --test-id "$test_id" \
        --output-dir "$run_dir" \
        --config "$LOCO_CONFIG" \
        || info "  (analysis completed with warnings)"
}

# ---------------------------------------------------------------------------
# T8 ball push - needs ball teleported after decider starts
# ---------------------------------------------------------------------------
run_ball_push_case() {
    local test_id="$1"
    local cmd_str="$2"
    local duration="$3"
    local run_dir="$4"
    local ball_x="$5"
    local ball_y="$6"
    local ball_z="${7:-0.0917}"

    mkdir -p "$run_dir"
    info "  cmd=$cmd_str  duration=${duration}s  ball=($ball_x,$ball_y)"

    kill_stale
    launch_sim_loco
    sleep 4

    launch_decider_loco "$run_dir" "--sim-fixed-cmd $cmd_str"
    sleep 3

    # Teleport ball to position in front of robot
    info "  teleporting ball ..."
    python3 -c "
import zmq, json, time
ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.connect('tcp://127.0.0.1:5555')
s.setsockopt(zmq.RCVTIMEO, 2000)
try:
    s.send_json({'cmd': [0.0,0.0,0.0], 'id': 0, 'timestamp': time.time(),
                 'teleport': {'name': 'ball', 'x': $ball_x, 'y': $ball_y, 'z': $ball_z}})
    s.recv_json()
    print('  ball teleported OK')
except Exception as e:
    print(f'  ball teleport failed: {e}')
finally:
    s.close()
" 2>&1 || true

    info "  recording ${duration}s ..."
    sleep "$duration"

    kill -INT "$DECIDER_PID" 2>/dev/null || true
    sleep 2
    wait "$DECIDER_PID" 2>/dev/null || true
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
    kill_stale
    sleep 1

    info "  analyzing ..."
    cd "$REPO_ROOT"
    python3 -u "$REPO_ROOT/decider/scripts/analyze_loco_baseline.py" \
        "$run_dir/trajectory.csv" \
        --test-id "$test_id" \
        --output-dir "$run_dir" \
        --config "$LOCO_CONFIG" \
        || info "  (analysis completed with warnings)"
}

# ---------------------------------------------------------------------------
# T6 sequence case
# ---------------------------------------------------------------------------
run_sequence_case() {
    local test_id="$1"
    local seq_json_file="$2"
    local total_duration="$3"
    local run_dir="$4"

    mkdir -p "$run_dir"
    info "  seq duration=${total_duration}s"

    kill_stale
    launch_sim_loco
    sleep 4

    launch_decider_loco "$run_dir" "--sim-fixed-cmd-seq @$seq_json_file"
    sleep 2

    info "  recording ${total_duration}s ..."
    sleep "$total_duration"

    kill -INT "$DECIDER_PID" 2>/dev/null || true
    sleep 2
    wait "$DECIDER_PID" 2>/dev/null || true
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
    kill_stale
    sleep 1

    info "  analyzing ..."
    cd "$REPO_ROOT"
    python3 -u "$REPO_ROOT/decider/scripts/analyze_loco_baseline.py" \
        "$run_dir/trajectory.csv" \
        --test-id "$test_id" \
        --output-dir "$run_dir" \
        --config "$LOCO_CONFIG" \
        || info "  (analysis completed with warnings)"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_loco_args "$@"
    check_uv

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BASELINE_DIR="$REPO_ROOT/video/loco_baseline/${TIMESTAMP}_${MODEL_TAG}"
    mkdir -p "$BASELINE_DIR"

    info "============================================"
    info "Locomotion Baseline Test Suite"
    info "Model:  ${POLICY_ARG:-(default K1 policy)}"
    info "Output: $BASELINE_DIR"
    info "============================================"
    echo ""

    TEST_IDS=$(get_test_ids)

    for test_id in $TEST_IDS; do
        info "--- $test_id ---"

        test_type=$(python3 -c "
import json
cfg=json.load(open('$LOCO_CONFIG'))
t=cfg['tests'].get('$test_id',{})
print(t.get('type','constant'))
")

        if [[ "$test_type" == "sequence" ]]; then
            # T6: command sequences
            seq_count=$(python3 -c "
import json
cfg=json.load(open('$LOCO_CONFIG'))
seqs=cfg['tests']['$test_id']['sequences']
print(len(seqs))
")
            for ((si=0; si<seq_count; si++)); do
                seq_data=$(python3 -c "
import json
cfg=json.load(open('$LOCO_CONFIG'))
s=cfg['tests']['$test_id']['sequences'][$si]
total=sum(x['duration_s'] for x in s['steps'])
print(json.dumps({'id': s['id'], 'steps': s['steps'], 'total': total}))
")
                seq_id=$(echo "$seq_data" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
                total_d=$(echo "$seq_data" | python3 -c "import json,sys; print(json.load(sys.stdin)['total'])")
                # Write seq JSON to temp file to avoid shell escaping issues
                seq_tmp=$(mktemp /tmp/loco_seq_XXXXX.json)
                echo "$seq_data" | python3 -c "import json,sys; json.dump(json.load(sys.stdin)['steps'], open('$seq_tmp','w'))"

                run_dir="$BASELINE_DIR/${test_id}/${seq_id}"
                run_sequence_case "$test_id" "$seq_tmp" "$total_d" "$run_dir"
                rm -f "$seq_tmp"
            done

        else
            # Constant command cases (T0-T5, T7-T9)
            case_count=$(python3 -c "
import json
cfg=json.load(open('$LOCO_CONFIG'))
print(len(cfg['tests']['$test_id'].get('cases',[])))
")
            for ((ci=0; ci<case_count; ci++)); do
                case_data=$(python3 -c "
import json
cfg=json.load(open('$LOCO_CONFIG'))
c=cfg['tests']['$test_id']['cases'][$ci]
print(json.dumps(c))
")
                case_id=$(echo "$case_data" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
                cmd_vx=$(echo "$case_data"  | python3 -c "import json,sys; print(json.load(sys.stdin)['cmd'][0])")
                cmd_vy=$(echo "$case_data"  | python3 -c "import json,sys; print(json.load(sys.stdin)['cmd'][1])")
                cmd_w=$(echo "$case_data"   | python3 -c "import json,sys; print(json.load(sys.stdin)['cmd'][2])")
                duration=$(echo "$case_data" | python3 -c "import json,sys; print(json.load(sys.stdin)['duration_s'])")
                repeats=$(echo "$case_data"  | python3 -c "import json,sys; print(json.load(sys.stdin)['repeats'])")
                cmd_str="${cmd_vx},${cmd_vy},${cmd_w}"

                # Check for ball teleport (T8)
                ball_x=$(echo "$case_data" | python3 -c "import json,sys; c=json.load(sys.stdin); tp=c.get('ball_teleport',{}); print(tp.get('x',''))" 2>/dev/null || echo "")
                ball_y=$(echo "$case_data" | python3 -c "import json,sys; c=json.load(sys.stdin); tp=c.get('ball_teleport',{}); print(tp.get('y',''))" 2>/dev/null || echo "")
                ball_z=$(echo "$case_data" | python3 -c "import json,sys; c=json.load(sys.stdin); tp=c.get('ball_teleport',{}); print(tp.get('z','0.0917'))" 2>/dev/null || echo "0.0917")

                for ((ri=1; ri<=repeats; ri++)); do
                    run_label=$(printf "run_%03d" "$ri")
                    run_dir="$BASELINE_DIR/${test_id}/${case_id}/${run_label}"

                    if [[ -n "$ball_x" && -n "$ball_y" ]]; then
                        run_ball_push_case "$test_id" "$cmd_str" "$duration" "$run_dir" "$ball_x" "$ball_y" "$ball_z"
                    else
                        run_constant_case "$test_id" "$cmd_str" "$duration" "$run_dir"
                    fi
                    sleep 1
                done
            done
        fi
        echo ""
    done

    # --- Aggregated report ---
    info "============================================"
    info "Generating aggregated report ..."
    cd "$REPO_ROOT"
    python3 -u "$REPO_ROOT/decider/scripts/loco_report.py" \
        --baseline-dir "$BASELINE_DIR" \
        --config "$LOCO_CONFIG" \
        --output-md "$BASELINE_DIR/loco_baseline_report.md" \
        --output-json "$BASELINE_DIR/loco_baseline_report.json"

    info "Done. Report: $BASELINE_DIR/loco_baseline_report.md"
    info "        JSON:  $BASELINE_DIR/loco_baseline_report.json"
}

main "$@"
