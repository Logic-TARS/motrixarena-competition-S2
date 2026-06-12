#!/usr/bin/env python3
"""Generate plots and a JSON summary from a Decider trajectory CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


STATE_COLORS = {
    "SEARCH_BALL": "#777777",
    "APPROACH_BALL": "#1f77b4",
    "ALIGN_BEHIND_BALL": "#ff7f0e",
    "SIDE_RECOVERY": "#17becf",
    "DRIBBLE": "#2ca02c",
    "KICK": "#d62728",
    "RECOVER": "#9467bd",
    "RETURN_TO_FIELD": "#8c564b",
    "STOP": "#111111",
}


def _float(row: dict[str, str], key: str):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in ("1", "true", "yes")


def load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _state_segments(rows: list[dict[str, str]]):
    if not rows:
        return []
    segments = []
    start = 0
    current = rows[0].get("fsm_state", "")
    for index in range(1, len(rows)):
        state = rows[index].get("fsm_state", "")
        if state != current:
            segments.append((start, index - 1, current))
            start = index
            current = state
    segments.append((start, len(rows) - 1, current))
    return segments


def _category_durations(rows: list[dict[str, str]], key: str):
    durations = defaultdict(float)
    for index, row in enumerate(rows):
        category = row.get(key, "")
        if not category:
            continue
        start = _float(row, "elapsed_s")
        if start is None:
            continue
        end = start
        if index + 1 < len(rows):
            next_time = _float(rows[index + 1], "elapsed_s")
            if next_time is not None:
                end = next_time
        durations[category] += max(0.0, end - start)
    return dict(sorted(durations.items()))


def build_summary(
    rows: list[dict[str, str]],
    displacement_epsilon: float = 0.005,
) -> dict[str, Any]:
    if not rows:
        return {
            "run_mode": "unknown",
            "frame_count": 0,
            "duration_s": 0.0,
            "average_hz": None,
            "fsm_enabled": False,
            "state_durations_s": {},
            "align_mode_durations_s": {},
            "side_recovery_phase_durations_s": {},
            "state_transitions": [],
            "kick_count": 0,
            "kicks": [],
            "push_to_goal": {},
        }

    start_t = _float(rows[0], "elapsed_s") or 0.0
    end_t = _float(rows[-1], "elapsed_s") or start_t
    duration = max(0.0, end_t - start_t)
    average_hz = (len(rows) - 1) / duration if duration > 0 and len(rows) > 1 else None
    run_mode = rows[0].get("run_mode", "") or "unknown"

    state_durations = defaultdict(float)
    transitions = []
    for start, end, state in _state_segments(rows):
        if not state:
            continue
        segment_start = _float(rows[start], "elapsed_s") or 0.0
        segment_end = _float(rows[end], "elapsed_s") or segment_start
        if end + 1 < len(rows):
            next_time = _float(rows[end + 1], "elapsed_s")
            if next_time is not None:
                segment_end = next_time
        state_durations[state] += max(0.0, segment_end - segment_start)
        if start > 0:
            transitions.append(
                {
                    "time_s": segment_start,
                    "from": rows[start - 1].get("fsm_state", ""),
                    "to": state,
                }
            )

    kick_entries = [
        index
        for index in range(len(rows))
        if rows[index].get("fsm_state") == "KICK"
        and (index == 0 or rows[index - 1].get("fsm_state") != "KICK")
    ]
    kicks = []
    for number, entry in enumerate(kick_entries, start=1):
        end = entry
        while end + 1 < len(rows) and rows[end + 1].get("fsm_state") == "KICK":
            end += 1
        push_indices = [
            index for index in range(entry, end + 1) if _bool(rows[index], "kick_push")
        ]
        before = rows[entry - 1] if entry > 0 else rows[entry]
        kick_info: dict[str, Any] = {
            "index": number,
            "entry_time_s": _float(rows[entry], "elapsed_s"),
            "pre_kick": {
                "ball_local_x": _float(before, "ball_local_x"),
                "ball_local_y": _float(before, "ball_local_y"),
                "behind_depth": _float(before, "behind_depth"),
                "lateral_err": _float(before, "lateral_err"),
                "ball_to_goal_yaw_err_deg": _float(
                    before, "ball_to_goal_yaw_err_deg"
                ),
                "distance_to_goal": _float(before, "distance_to_goal"),
                "can_kick": _bool(before, "can_kick"),
                "can_kick_reason": before.get("can_kick_reason", ""),
            },
            "push_observed": bool(push_indices),
            "ball_displacement": None,
        }
        if push_indices:
            first_push = push_indices[0]
            start_x = _float(rows[first_push], "ball_x")
            start_y = _float(rows[first_push], "ball_y")
            end_x = _float(rows[end], "ball_x")
            end_y = _float(rows[end], "ball_y")
            field_length = _float(rows[first_push], "field_length")
            if None not in (start_x, start_y, end_x, end_y, field_length):
                dx = end_x - start_x
                dy = end_y - start_y
                magnitude = math.hypot(dx, dy)
                goal_dx = field_length / 2.0 - start_x
                goal_dy = -start_y
                goal_norm = math.hypot(goal_dx, goal_dy)
                projection = None
                direction_error = None
                if goal_norm > 1e-9:
                    ux = goal_dx / goal_norm
                    uy = goal_dy / goal_norm
                    projection = dx * ux + dy * uy
                    if magnitude >= displacement_epsilon:
                        cosine = max(
                            -1.0,
                            min(1.0, projection / magnitude),
                        )
                        direction_error = math.degrees(math.acos(cosine))
                kick_info["ball_displacement"] = {
                    "dx": dx,
                    "dy": dy,
                    "distance": magnitude,
                    "toward_goal_projection": projection,
                    "direction_error_deg": direction_error,
                }
        kicks.append(kick_info)

    # --- push_to_goal analysis ---
    push_to_goal = {}
    if run_mode in ("push_to_goal", "simple_push_to_goal", "continuous_push"):
        ball_dists = [
            _float(row, "ball_distance")
            for row in rows
            if _float(row, "ball_distance") is not None
        ]
        if ball_dists:
            push_to_goal["ball_dist_initial"] = ball_dists[0]
            push_to_goal["ball_dist_min"] = min(ball_dists)
            push_to_goal["ball_dist_final"] = ball_dists[-1]
            enter_05 = None
            for row in rows:
                bd = _float(row, "ball_distance")
                if bd is not None and bd <= 0.5:
                    t = _float(row, "elapsed_s")
                    if t is not None:
                        enter_05 = t
                        break
            push_to_goal["time_to_enter_0_5m"] = enter_05

        robot_positions = [
            (_float(row, "elapsed_s"), _float(row, "robot_x"), _float(row, "robot_y"))
            for row in rows
        ]
        robot_positions = [
            (t, x, y) for t, x, y in robot_positions
            if None not in (t, x, y)
        ]
        if len(robot_positions) >= 2:
            dx = robot_positions[-1][1] - robot_positions[0][1]
            dy = robot_positions[-1][2] - robot_positions[0][2]
            push_to_goal["robot_displacement"] = math.hypot(dx, dy)

        ball_movements = [
            (_float(row, "elapsed_s"), _float(row, "ball_x"), _float(row, "ball_y"))
            for row in rows
        ]
        ball_movements = [
            (t, x, y) for t, x, y in ball_movements
            if None not in (t, x, y)
        ]
        if len(ball_movements) >= 2 and rows:
            field_len = _float(rows[0], "field_length") or 9.0
            team = str(rows[0].get("team", "red")).strip().lower()
            goal_sign = -1.0 if team == "blue" else 1.0
            goal_x = goal_sign * field_len / 2.0
            ball_dx = ball_movements[-1][1] - ball_movements[0][1]
            ball_dy = ball_movements[-1][2] - ball_movements[0][2]
            goal_dx = goal_x - ball_movements[0][1]
            goal_dy = -ball_movements[0][2]
            goal_norm = math.hypot(goal_dx, goal_dy)
            if goal_norm > 1e-9:
                ux = goal_dx / goal_norm
                uy = goal_dy / goal_norm
                push_to_goal["goal_direction_ball_progress"] = ball_dx * ux + ball_dy * uy

            crossed_goal_time = None
            for t, x, _ in ball_movements:
                if goal_sign * x >= field_len / 2.0:
                    crossed_goal_time = t
                    break
            push_to_goal["time_to_cross_goal_line"] = crossed_goal_time

        cmd_vx = [
            _float(row, "cmd_vx") for row in rows
            if _float(row, "cmd_vx") is not None
        ]
        cmd_vy = [
            _float(row, "cmd_vy") for row in rows
            if _float(row, "cmd_vy") is not None
        ]
        cmd_w = [
            _float(row, "cmd_w") for row in rows
            if _float(row, "cmd_w") is not None
        ]
        push_to_goal["cmd_statistics"] = {
            "vx_mean": sum(cmd_vx) / len(cmd_vx) if cmd_vx else None,
            "vx_max": max(cmd_vx) if cmd_vx else None,
            "vy_mean": sum(cmd_vy) / len(cmd_vy) if cmd_vy else None,
            "vy_abs_max": max(map(abs, cmd_vy)) if cmd_vy else None,
            "w_mean": sum(cmd_w) / len(cmd_w) if cmd_w else None,
            "w_abs_max": max(map(abs, cmd_w)) if cmd_w else None,
        }
        near_vx = []
        for row in rows:
            distance = _float(row, "ball_distance")
            vx = _float(row, "cmd_vx")
            if distance is not None and distance <= 0.5 and vx is not None:
                near_vx.append(vx)
        push_to_goal["near_ball_vx_mean"] = (
            sum(near_vx) / len(near_vx) if near_vx else None
        )

    summary: dict[str, Any] = {
        "run_mode": run_mode,
        "fsm_enabled": any(row.get("fsm_state", "") for row in rows),
        "frame_count": len(rows),
        "duration_s": duration,
        "average_hz": average_hz,
        "state_durations_s": dict(sorted(state_durations.items())),
        "align_mode_durations_s": _category_durations(rows, "align_mode"),
        "side_recovery_phase_durations_s": _category_durations(
            rows, "side_recovery_phase"
        ),
        "state_transitions": transitions,
        "kick_count": len(kick_entries),
        "kicks": kicks,
        "push_to_goal": push_to_goal,
    }
    return summary


def _shade_states(ax, rows: list[dict[str, str]]) -> None:
    for start, end, state in _state_segments(rows):
        if not state:
            continue
        x0 = _float(rows[start], "elapsed_s")
        x1 = _float(rows[end], "elapsed_s")
        if x0 is None or x1 is None:
            continue
        ax.axvspan(
            x0,
            x1,
            color=STATE_COLORS.get(state, "#cccccc"),
            alpha=0.08,
            linewidth=0,
        )


def _series(rows: list[dict[str, str]], key: str):
    return [_float(row, key) for row in rows]


def generate_plots(rows: list[dict[str, str]], output_dir: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Trajectory CSV and summary were saved, but matplotlib is unavailable; "
            "PNG plots were skipped.",
            file=sys.stderr,
        )
        return False

    valid_robot = [
        (index, _float(row, "robot_x"), _float(row, "robot_y"))
        for index, row in enumerate(rows)
    ]
    valid_robot = [(i, x, y) for i, x, y in valid_robot if x is not None and y is not None]
    valid_ball = [
        (_float(row, "ball_x"), _float(row, "ball_y"))
        for row in rows
    ]
    valid_ball = [(x, y) for x, y in valid_ball if x is not None and y is not None]

    fig, ax = plt.subplots(figsize=(10, 7))
    field_length = _float(rows[0], "field_length") if rows else None
    field_width = _float(rows[0], "field_width") if rows else None
    if field_length and field_width:
        ax.add_patch(
            plt.Rectangle(
                (-field_length / 2.0, -field_width / 2.0),
                field_length,
                field_width,
                fill=False,
                color="#555555",
                linewidth=1.2,
            )
        )
        ax.axvline(0.0, color="#aaaaaa", linewidth=0.7)
    for state, color in STATE_COLORS.items():
        points = [
            (x, y)
            for index, x, y in valid_robot
            if rows[index].get("fsm_state", "") == state
        ]
        if points:
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                ".",
                color=color,
                markersize=3,
                label=state,
            )
    if valid_robot and not any(row.get("fsm_state", "") for row in rows):
        ax.plot(
            [point[1] for point in valid_robot],
            [point[2] for point in valid_robot],
            color="#1f77b4",
            linewidth=1.2,
            label="robot",
        )
    if valid_ball:
        ax.plot(
            [point[0] for point in valid_ball],
            [point[1] for point in valid_ball],
            color="#111111",
            linewidth=1.2,
            label="ball",
        )
    if valid_robot:
        ax.scatter(valid_robot[0][1], valid_robot[0][2], c="#2ca02c", s=45, marker="o")
        ax.scatter(valid_robot[-1][1], valid_robot[-1][2], c="#d62728", s=45, marker="x")
    for index in range(1, len(rows)):
        state = rows[index].get("fsm_state", "")
        previous_state = rows[index - 1].get("fsm_state", "")
        if state and state != previous_state:
            x = _float(rows[index], "robot_x")
            y = _float(rows[index], "robot_y")
            if x is not None and y is not None:
                if state == "KICK":
                    ax.scatter(x, y, c="#d62728", marker="*", s=100, zorder=5)
                else:
                    ax.scatter(
                        x,
                        y,
                        facecolors="none",
                        edgecolors=STATE_COLORS.get(state, "#555555"),
                        marker="o",
                        s=45,
                        zorder=5,
                    )
    ax.set_title("Robot and Ball Trajectory")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_xy.png", dpi=160)
    plt.close(fig)

    t = _series(rows, "elapsed_s")
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    plots = [
        (
            axes[0],
            [
                ("ball_local_x", "ball local X"),
                ("ball_local_y", "ball local Y"),
            ],
            "Ball relative position (m)",
        ),
        (
            axes[1],
            [("cmd_vx", "cmd vx"), ("cmd_vy", "cmd vy"), ("cmd_w", "cmd w")],
            "Final command",
        ),
        (
            axes[2],
            [
                ("behind_depth", "behind depth"),
                ("lateral_err", "lateral error"),
            ],
            "Behind-ball geometry (m)",
        ),
        (
            axes[3],
            [("ball_to_goal_yaw_err_deg", "yaw error")],
            "Ball-to-goal yaw error (deg)",
        ),
    ]
    for ax, lines, ylabel in plots:
        _shade_states(ax, rows)
        for key, label in lines:
            ax.plot(t, _series(rows, key), linewidth=1.0, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_timeseries.png", dpi=160)
    plt.close(fig)
    return True


def analyze_trajectory(csv_path: str | Path, output_dir: str | Path | None = None):
    csv_path = Path(csv_path).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else csv_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(csv_path)
    summary = build_summary(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if rows:
        generate_plots(rows, output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to trajectory.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run trajectory failure diagnosis after analysis",
    )
    args = parser.parse_args()
    summary = analyze_trajectory(args.csv_path, args.output_dir)
    if args.diagnose:
        from diagnose_trajectory import diagnose_trajectory

        diagnose_trajectory(args.csv_path, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
