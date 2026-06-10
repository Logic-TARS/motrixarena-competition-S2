#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_NAME="${ENV_NAME:-k1-getup}"
RUN_ROOT="${RUN_ROOT:-runs}"
SEED="${SEED:-1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
GETUP_EPISODES="${GETUP_EPISODES:-200}"
WALK_SECONDS="${WALK_SECONDS:-10}"

case "$ENV_NAME" in
    k1-getup)
        experiment="k1_full_body_getup"
        config_file="motrix_envs/src/motrix_envs/locomotion/k1/getup_cfg.py"
        ;;
    k1-flat-terrain-walk)
        experiment="k1_g1_style_walk"
        config_file="motrix_envs/src/motrix_envs/locomotion/k1/cfg.py"
        ;;
    *)
        echo "Unsupported ENV_NAME: $ENV_NAME" >&2
        exit 2
        ;;
esac

candidate_root="$RUN_ROOT/candidates/$ENV_NAME"
mkdir -p "$candidate_root"

echo "Monitoring $RUN_ROOT/$ENV_NAME/rslrl for every $EVAL_INTERVAL iterations"
while true; do
    while IFS= read -r checkpoint; do
        name="$(basename "$checkpoint" .pt)"
        iteration="${name#model_}"
        if ! [[ "$iteration" =~ ^[0-9]+$ ]] || (( iteration == 0 || iteration % EVAL_INTERVAL != 0 )); then
            continue
        fi
        run_name="$(basename "$(dirname "$checkpoint")")"
        output_dir="$candidate_root/$run_name/$name"
        manifest="$output_dir/model.pt.manifest.json"
        if [[ -f "$manifest" ]]; then
            continue
        fi

        mkdir -p "$output_dir"
        cp "$checkpoint" "$output_dir/model.pt"
        cp "$config_file" "$output_dir/$(basename "$config_file")"
        if [[ "$ENV_NAME" == "k1-getup" ]]; then
            uv run --frozen python scripts/eval_k1_getup.py \
                --policy "$output_dir/model.pt" \
                --episodes "$GETUP_EPISODES" \
                --seed "$SEED" \
                --output "$output_dir/evaluation.json"
        else
            uv run --frozen python scripts/eval_k1_walk_grid.py \
                --policy "$output_dir/model.pt" \
                --seconds "$WALK_SECONDS" \
                --seed "$SEED" \
                --output "$output_dir/evaluation.json"
        fi
        uv run --frozen python scripts/model_manifest.py "$output_dir/model.pt" \
            --seed "$SEED" \
            --iteration "$iteration" \
            --env "$ENV_NAME" \
            --evaluation "$output_dir/evaluation.json" \
            --config "$output_dir/$(basename "$config_file")" \
            --output "$manifest"
        if uv run --frozen python -c \
            'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["eligible"] else 1)' \
            "$output_dir/evaluation.json"; then
            touch "$output_dir/PROMOTED"
            echo "Candidate promoted: $output_dir"
        else
            echo "Candidate rejected: $output_dir"
        fi
    done < <(find "$RUN_ROOT/$ENV_NAME/rslrl" -type f -name 'model_*.pt' 2>/dev/null | sort)
    sleep "$POLL_SECONDS"
done
