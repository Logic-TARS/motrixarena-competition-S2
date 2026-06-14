"""Trajectory recording helpers for simulation diagnostics."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any


TRAJECTORY_FIELDS = [
    "frame",
    "wall_time",
    "elapsed_s",
    "sim_time",
    "dt_s",
    "run_mode",
    "team",
    "robot_id",
    "field_length",
    "field_width",
    "robot_x",
    "robot_y",
    "robot_yaw_deg",
    "is_fallen",
    "ball_x",
    "ball_y",
    "ball_z",
    "ball_local_x",
    "ball_local_y",
    "ball_distance",
    "cmd_vx",
    "cmd_vy",
    "cmd_w",
    "game_state",
    "fsm_state",
    "align_mode",
    "side_recovery_phase",
    "state_duration_s",
    "behind_depth",
    "depth_err",
    "lateral_err",
    "ball_to_goal_yaw_err_deg",
    "distance_to_goal",
    "can_kick",
    "can_kick_reason",
    "kick_push",
    # Alignment-pipeline diagnostic fields
    "is_behind_ball",
    "is_laterally_aligned",
    "is_facing_goal",
    "robot_speed",
    "can_kick_candidate",
    "ball_angle_deg",
    "approach_guard_input_vx",
    "approach_guard_input_vy",
    "approach_guard_input_w",
    "approach_guard_applied",
]


def _finite_or_blank(value: Any):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else ""


class TrajectoryRecorder:
    """Stream simulation trajectory rows to CSV with bounded data loss."""

    def __init__(self, output_dir: str | Path, flush_interval: int = 20):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "trajectory.csv"
        self.flush_interval = max(1, int(flush_interval))
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=TRAJECTORY_FIELDS,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._file.flush()
        self._rows_since_flush = 0
        self._closed = False

    def write(self, row: dict[str, Any]) -> None:
        if self._closed:
            return
        normalized = {
            field: _finite_or_blank(row.get(field, ""))
            for field in TRAJECTORY_FIELDS
        }
        self._writer.writerow(normalized)
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if self._closed:
            return
        self._file.flush()
        self._rows_since_flush = 0

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._file.close()
        self._closed = True


def default_trajectory_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return repo_root / "debug_logs" / f"trajectory_{timestamp}"
