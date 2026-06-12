#!/usr/bin/env python3
"""Diagnose why a MiniPushFSM (fsm_mvp) trajectory failed to push the ball.

Three diagnostic layers:
  1. State layer — which FSM states were entered, durations, time-to-0.5m.
  2. Condition layer — per-frame check of the 5 PUSH_FORWARD entry conditions
     (from MiniPushFSM._ready_to_push), producing per-condition stats and a
     primary-blocker verdict.
  3. Execution layer — if PUSH_FORWARD was entered, check whether the ball
     actually progressed toward the goal.

Outputs diagnosis.json (structured) and diagnosis.txt (Chinese report).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers (duplicated from analyze_trajectory.py to keep this script
# self-contained / independently runnable)
# ---------------------------------------------------------------------------


def _float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(row: Dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in ("1", "true", "yes")


def _state_segments(rows: List[Dict[str, str]]) -> List[Tuple[int, int, str]]:
    """Return [(start_index, end_index, fsm_state), ...] contiguous segments."""
    if not rows:
        return []
    segments: List[Tuple[int, int, str]] = []
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


def _category_durations(
    rows: List[Dict[str, str]], key: str
) -> Dict[str, float]:
    """Sum elapsed time per distinct value of *key* column."""
    durations: Dict[str, float] = defaultdict(float)
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


# ---------------------------------------------------------------------------
# Config / thresholds
# ---------------------------------------------------------------------------

# Hardcoded defaults matching MiniPushFSMConfig dataclass
_DEFAULT_THRESHOLDS = {
    "push_ball_x_min": 0.14,
    "push_ball_x_max": 0.42,
    "push_ball_y_max": 0.08,
    "push_facing_error_deg": 15.0,
    "push_lateral_tol": 0.10,
    "push_min_behind_depth": 0.16,
}


def _load_thresholds(
    use_external_config: bool = True,
) -> Tuple[Dict[str, float], str]:
    """Load push-entry thresholds from fsm_mvp config section.

    When *use_external_config* is True, tries ``configuration.load_config()``
    first; falls back to hardcoded MiniPushFSMConfig defaults on any failure.

    Returns
    -------
    (thresholds, source_label)
        *thresholds* is the resolved threshold dict.
        *source_label* describes where the values came from
        (e.g. ``"config.yaml (fsm_mvp)"`` or ``"hardcoded_defaults"``).
    """
    thresholds = dict(_DEFAULT_THRESHOLDS)
    if not use_external_config:
        return thresholds, "hardcoded_defaults"
    try:
        from decider.configuration import load_config

        config = load_config()
        fsm_mvp = config.get("fsm_mvp", {})
        if fsm_mvp:
            for key in list(thresholds.keys()):
                if key in fsm_mvp:
                    thresholds[key] = float(fsm_mvp[key])
            return thresholds, "config.yaml (fsm_mvp)"
    except Exception:
        pass  # graceful fallback to hardcoded defaults
    return thresholds, "hardcoded_defaults"


# ---------------------------------------------------------------------------
# Condition specifications — one entry per _ready_to_push() check
# ---------------------------------------------------------------------------

CONDITION_SPECS = [
    {
        "name": "ball_local_x_in_window",
        "description": "球前后距离在窗口内",
        "kind": "range",
        "column": "ball_local_x",
        "threshold_keys": ("push_ball_x_min", "push_ball_x_max"),
    },
    {
        "name": "ball_local_y_centered",
        "description": "球横向居中",
        "kind": "abs_lt",
        "column": "ball_local_y",
        "threshold_keys": ("push_ball_y_max",),
    },
    {
        "name": "yaw_error_small",
        "description": "偏航角对准球-门线",
        "kind": "abs_lt",
        "column": "ball_to_goal_yaw_err_deg",
        "threshold_keys": ("push_facing_error_deg",),
    },
    {
        "name": "lateral_err_small",
        "description": "横向对齐误差小",
        "kind": "abs_lt",
        "column": "lateral_err",
        "threshold_keys": ("push_lateral_tol",),
    },
    {
        "name": "behind_depth_sufficient",
        "description": "球后深度足够",
        "kind": "gt",
        "column": "behind_depth",
        "threshold_keys": ("push_min_behind_depth",),
    },
]


def _check_condition(
    spec: Dict[str, Any], row: Dict[str, str], thresholds: Dict[str, float]
) -> Optional[bool]:
    """Return True/False if the condition passes, or None when data is missing."""
    column = spec["column"]
    value = _float(row, column)
    if value is None:
        return None
    kind = spec["kind"]
    keys = spec["threshold_keys"]
    if kind == "abs_lt":
        t = thresholds[keys[0]]
        return abs(value) < t
    elif kind == "gt":
        t = thresholds[keys[0]]
        return value > t
    elif kind == "range":
        lo = thresholds[keys[0]]
        hi = thresholds[keys[1]]
        return lo <= value <= hi
    return None


def _condition_distance(
    spec: Dict[str, Any], value: float, thresholds: Dict[str, float]
) -> float:
    """Positive distance from threshold boundary; 0.0 means 'passes'."""
    kind = spec["kind"]
    keys = spec["threshold_keys"]
    if kind == "abs_lt":
        t = thresholds[keys[0]]
        return max(0.0, abs(value) - t)
    elif kind == "gt":
        t = thresholds[keys[0]]
        return max(0.0, t - value)
    elif kind == "range":
        lo = thresholds[keys[0]]
        hi = thresholds[keys[1]]
        return max(0.0, lo - value, value - hi)
    return 0.0


# ---------------------------------------------------------------------------
# Layer 1 — State Diagnosis
# ---------------------------------------------------------------------------


def _analyze_state_layer(
    rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Extract FSM state overview from the trajectory CSV."""
    state_durations: Dict[str, float] = defaultdict(float)
    transitions: List[Dict[str, Any]] = []
    for start, end, state in _state_segments(rows):
        if not state:
            continue
        seg_start = _float(rows[start], "elapsed_s") or 0.0
        seg_end = _float(rows[end], "elapsed_s") or seg_start
        if end + 1 < len(rows):
            nxt = _float(rows[end + 1], "elapsed_s")
            if nxt is not None:
                seg_end = nxt
        state_durations[state] += max(0.0, seg_end - seg_start)
        if start > 0:
            transitions.append(
                {
                    "time_s": seg_start,
                    "from": rows[start - 1].get("fsm_state", ""),
                    "to": state,
                }
            )

    align_durations = _category_durations(rows, "align_mode")

    # Ball distance statistics
    ball_dists = [
        d
        for d in (_float(r, "ball_distance") for r in rows)
        if d is not None
    ]
    ball_dist_initial = ball_dists[0] if ball_dists else None
    ball_dist_min = min(ball_dists) if ball_dists else None
    ball_dist_final = ball_dists[-1] if ball_dists else None

    # Time to enter 0.5 m
    time_05 = None
    for row in rows:
        bd = _float(row, "ball_distance")
        if bd is not None and bd <= 0.5:
            t = _float(row, "elapsed_s")
            if t is not None:
                time_05 = t
                break

    states_entered = []
    for start, end, state in _state_segments(rows):
        if state and state not in states_entered:
            states_entered.append(state)

    return {
        "states_entered": states_entered,
        "state_durations_s": dict(sorted(state_durations.items())),
        "align_mode_durations_s": align_durations,
        "did_reach_chase": "CHASE" in state_durations,
        "did_reach_align_behind": "ALIGN_BEHIND" in state_durations,
        "did_reach_push_forward": "PUSH_FORWARD" in state_durations,
        "last_state": states_entered[-1] if states_entered else None,
        "transitions": transitions,
        "time_to_enter_0_5m": time_05,
        "ball_dist_initial": ball_dist_initial,
        "ball_dist_min": ball_dist_min,
        "ball_dist_final": ball_dist_final,
    }


# ---------------------------------------------------------------------------
# Layer 2 — Condition Diagnosis
# ---------------------------------------------------------------------------


def _get_analysis_frame_range(
    rows: List[Dict[str, str]], state_layer: Dict[str, Any]
) -> Tuple[int, int, str]:
    """Determine which frame range to analyze for condition checking.

    Returns (start_index, end_index, state_name).  Prefers ALIGN_BEHIND
    frames when available, otherwise falls back to CHASE.
    """
    if state_layer["did_reach_align_behind"]:
        for i, row in enumerate(rows):
            if row.get("fsm_state", "") == "ALIGN_BEHIND":
                # Find the last ALIGN_BEHIND frame
                end = i
                while end + 1 < len(rows) and rows[end + 1].get("fsm_state", "") == "ALIGN_BEHIND":
                    end += 1
                return i, end, "ALIGN_BEHIND"
    if state_layer["did_reach_chase"]:
        for i, row in enumerate(rows):
            if row.get("fsm_state", "") == "CHASE":
                end = i
                while end + 1 < len(rows) and rows[end + 1].get("fsm_state", "") == "CHASE":
                    end += 1
                return i, end, "CHASE"
    return 0, len(rows) - 1, state_layer.get("last_state", "")


def _analyze_condition_layer(
    rows: List[Dict[str, str]],
    state_layer: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Per-frame check of all PUSH_FORWARD entry conditions."""
    start_idx, end_idx, analyzed_state = _get_analysis_frame_range(
        rows, state_layer
    )
    frame_count = max(1, end_idx - start_idx + 1)

    # Initialize accumulators per condition
    stats: Dict[str, Dict[str, Any]] = {}
    for spec in CONDITION_SPECS:
        stats[spec["name"]] = {
            "condition": spec["name"],
            "description": spec["description"],
            "pass_count": 0,
            "fail_count": 0,
            "missing_count": 0,
            "first_pass_time": None,
            "last_pass_time": None,
            "worst_value": None,
            "worst_distance": -1.0,
            "closest_value": None,
            "closest_distance": float("inf"),
            "failing_seconds": 0.0,
            "latest_value": None,
        }

    # Single pass over analysis window
    for i in range(start_idx, end_idx + 1):
        row = rows[i]
        elapsed = _float(row, "elapsed_s") or 0.0
        # dt for this row
        dt = 0.0
        if i + 1 < len(rows):
            next_elapsed = _float(rows[i + 1], "elapsed_s")
            if next_elapsed is not None:
                dt = max(0.0, next_elapsed - elapsed)

        for spec in CONDITION_SPECS:
            name = spec["name"]
            st = stats[name]
            value = _float(row, spec["column"])
            if value is None:
                st["missing_count"] += 1
                continue

            passed = _check_condition(spec, row, thresholds)
            if passed:
                st["pass_count"] += 1
                t = _float(row, "elapsed_s")
                if t is not None:
                    if st["first_pass_time"] is None:
                        st["first_pass_time"] = t
                    st["last_pass_time"] = t
            else:
                st["fail_count"] += 1
                st["failing_seconds"] += dt

            dist = _condition_distance(spec, value, thresholds)
            if dist > st["worst_distance"]:
                st["worst_distance"] = dist
                st["worst_value"] = value
            if dist < st["closest_distance"]:
                st["closest_distance"] = dist
                st["closest_value"] = value

            st["latest_value"] = value

    # Finalize stats: compute pass_rate, clean up internal fields
    for spec in CONDITION_SPECS:
        name = spec["name"]
        st = stats[name]
        total = st["pass_count"] + st["fail_count"]
        st["pass_rate"] = st["pass_count"] / total if total > 0 else 0.0
        # Remove internal accumulators
        del st["pass_count"]
        del st["fail_count"]
        del st["missing_count"]
        del st["worst_distance"]
        del st["closest_distance"]

    return {
        "analyzed_state": analyzed_state,
        "analyzed_frame_range": [start_idx, end_idx],
        "analyzed_frame_count": frame_count,
        "conditions": stats,
    }


def _identify_primary_blocker(
    condition_layer: Dict[str, Any],
    state_layer: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Decide the primary failure reason from condition statistics and timing."""
    conditions = condition_layer["conditions"]

    # Rule 1 — never reached ALIGN_BEHIND
    if not state_layer["did_reach_align_behind"]:
        last = state_layer["last_state"] or "unknown"
        dur = state_layer["state_durations_s"].get(last, 0.0)
        return {
            "condition": "chase_too_slow_or_bad_heading",
            "evidence": (
                f"Robot never entered ALIGN_BEHIND state. "
                f"Stuck in {last} for {dur:.2f}s. "
                f"Ball min distance: {state_layer.get('ball_dist_min', 'N/A')}m."
            ),
            "pass_rate": None,
        }

    # Rule 2 — ALIGN_FACE_GOAL ran too briefly AND yaw never passed
    align_face_dur = state_layer.get("align_mode_durations_s", {}).get(
        "ALIGN_FACE_GOAL", 0.0
    )
    yaw_stats = conditions.get("yaw_error_small", {})
    if align_face_dur < 2.0 and yaw_stats.get("pass_rate", 1.0) == 0.0:
        return {
            "condition": "align_face_goal_too_late",
            "evidence": (
                f"ALIGN_FACE_GOAL only ran {align_face_dur:.2f}s; "
                f"final yaw_err={yaw_stats.get('latest_value', 'N/A'):.1f}deg "
                f"(threshold {thresholds['push_facing_error_deg']:.1f}deg). "
                f"Ball entered 0.5m at "
                f"{state_layer.get('time_to_enter_0_5m', 'N/A'):.2f}s "
                f"but facing correction arrived too late."
            ),
            "pass_rate": 0.0,
        }

    # Rule 3 — identify zero-pass conditions
    zero_pass = {
        name: st
        for name, st in conditions.items()
        if st["pass_rate"] == 0.0
    }

    if len(zero_pass) >= 2:
        # Multiple zero-pass: pick the one furthest from threshold (normalized)
        def _severity(name: str, st: Dict[str, Any]) -> float:
            spec = next(s for s in CONDITION_SPECS if s["name"] == name)
            keys = spec["threshold_keys"]
            if spec["kind"] == "abs_lt":
                t = thresholds[keys[0]]
                worst = abs(st["worst_value"]) if st["worst_value"] is not None else t
                return max(0.0, worst - t) / max(t, 0.001)
            elif spec["kind"] == "gt":
                t = thresholds[keys[0]]
                worst = st["worst_value"] if st["worst_value"] is not None else 0.0
                return max(0.0, t - worst) / max(t, 0.001)
            elif spec["kind"] == "range":
                lo, hi = thresholds[keys[0]], thresholds[keys[1]]
                worst = st["worst_value"] if st["worst_value"] is not None else lo
                return max(0.0, lo - worst, worst - hi) / max(hi - lo, 0.001)
            return 0.0

        worst_name = max(zero_pass, key=lambda n: _severity(n, zero_pass[n]))
        worst_st = zero_pass[worst_name]
        spec = next(s for s in CONDITION_SPECS if s["name"] == worst_name)
        keys = spec["threshold_keys"]
        thresh_desc = (
            f"[{thresholds[keys[0]]}, {thresholds[keys[1]]}]"
            if spec["kind"] == "range"
            else f"{thresholds[keys[0]]}"
        )
        return {
            "condition": worst_name,
            "evidence": (
                f"Zero pass rate. Multiple conditions never satisfied "
                f"({len(zero_pass)} total). "
                f"Worst offender: {spec['description']} "
                f"(worst_value={worst_st.get('worst_value', 'N/A'):.3f}, "
                f"threshold={thresh_desc})."
            ),
            "pass_rate": 0.0,
        }

    if len(zero_pass) == 1:
        name, st = list(zero_pass.items())[0]
        spec = next(s for s in CONDITION_SPECS if s["name"] == name)
        return {
            "condition": name,
            "evidence": (
                f"Condition '{spec['description']}' was never satisfied "
                f"(pass_rate=0%). "
                f"Worst value: {st.get('worst_value', 'N/A'):.3f}, "
                f"latest value: {st.get('latest_value', 'N/A'):.3f}."
            ),
            "pass_rate": 0.0,
        }

    # Rule 4 — all conditions passed at some point; find the last to pass
    last_satisfied = min(
        conditions.items(),
        key=lambda item: (
            item[1]["last_pass_time"]
            if item[1]["last_pass_time"] is not None
            else float("inf")
        ),
    )
    name, st = last_satisfied
    spec = next(s for s in CONDITION_SPECS if s["name"] == name)
    return {
        "condition": name,
        "evidence": (
            f"All conditions satisfied at some point. "
            f"'{spec['description']}' was the last to pass "
            f"(last_pass_time={st['last_pass_time']:.2f}s, "
            f"pass_rate={st['pass_rate']*100:.1f}%)."
        ),
        "pass_rate": st["pass_rate"],
    }


# ---------------------------------------------------------------------------
# Layer 3 — Execution Diagnosis
# ---------------------------------------------------------------------------


def _analyze_execution_layer(
    rows: List[Dict[str, str]],
    state_layer: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze PUSH_FORWARD execution quality (only when it was entered)."""
    if not state_layer["did_reach_push_forward"]:
        return {"entered_push_forward": False}

    push_indices = [
        i
        for i, r in enumerate(rows)
        if r.get("fsm_state", "") == "PUSH_FORWARD"
    ]
    if not push_indices:
        return {"entered_push_forward": False}

    # Split into contiguous segments
    segments: List[Tuple[int, int]] = []
    seg_start = push_indices[0]
    for i in range(1, len(push_indices)):
        if push_indices[i] != push_indices[i - 1] + 1:
            segments.append((seg_start, push_indices[i - 1]))
            seg_start = push_indices[i]
    segments.append((seg_start, push_indices[-1]))

    # Collect metrics during PUSH
    cmd_vy_nonzero = 0
    ball_positions: List[Tuple[float, float, float]] = []
    for seg_start, seg_end in segments:
        for i in range(seg_start, seg_end + 1):
            vy = _float(rows[i], "cmd_vy")
            if vy is not None and abs(vy) > 0.01:
                cmd_vy_nonzero += 1
            t = _float(rows[i], "elapsed_s")
            bx = _float(rows[i], "ball_x")
            by = _float(rows[i], "ball_y")
            if None not in (t, bx, by):
                ball_positions.append((t, bx, by))

    # Ball progress toward goal during PUSH
    goal_progress = None
    lateral_drift = None
    if len(ball_positions) >= 2:
        field_length = _float(rows[0], "field_length") or 9.0
        team = str(rows[0].get("team", "red")).strip().lower()
        goal_sign = -1.0 if team == "blue" else 1.0
        goal_x = goal_sign * field_length / 2.0

        dx = ball_positions[-1][1] - ball_positions[0][1]
        dy = ball_positions[-1][2] - ball_positions[0][2]
        goal_dx = goal_x - ball_positions[0][1]
        goal_dy = -ball_positions[0][2]
        goal_norm = math.hypot(goal_dx, goal_dy)
        if goal_norm > 1e-9:
            ux = goal_dx / goal_norm
            uy = goal_dy / goal_norm
            goal_progress = dx * ux + dy * uy
            lateral_drift = abs(dx * (-uy) + dy * ux)

    return {
        "entered_push_forward": True,
        "push_forward_segments": len(segments),
        "total_push_forward_frames": len(push_indices),
        "push_cmd_vy_nonzero_count": cmd_vy_nonzero,
        "ball_goal_progress_during_push": goal_progress,
        "ball_lateral_drift_during_push": lateral_drift,
        "has_goal_progress": goal_progress is not None and goal_progress > 0.01,
        "has_excess_lateral_drift": (
            lateral_drift is not None and lateral_drift > 0.15
        ),
    }


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


def _build_suggestions(
    primary_blocker: Dict[str, Any],
    state_layer: Dict[str, Any],
    push_execution: Dict[str, Any],
) -> List[str]:
    """Generate actionable Chinese suggestions from the diagnosis."""
    suggestions: List[str] = []
    cond = primary_blocker.get("condition", "")

    if cond == "chase_too_slow_or_bad_heading":
        suggestions.append(
            "提高 chase_vx_max 加快接近速度，或降低 chase_to_align_dist "
            "让机器人更早开始球后对齐"
        )
        suggestions.append(
            "检查 CHASE 阶段的 heading 是否正确指向球-门线方向"
        )
    elif cond == "align_face_goal_too_late":
        suggestions.append(
            "降低 chase_to_align_dist 让 ALIGN_BEHIND（含 ALIGN_FACE_GOAL）"
            "更早触发，给面向修正留足够时间"
        )
        suggestions.append(
            "提高 align_w_max 加快 ALIGN_FACE_GOAL 的转向速度"
        )
        suggestions.append(
            "放宽 push_facing_error_deg 阈值以增加容错"
        )
    elif cond == "yaw_error_small":
        suggestions.append(
            "提高 align_w_max 加快 ALIGN_FACE_GOAL 的转向速度"
        )
        suggestions.append(
            "放宽 push_facing_error_deg 阈值（当前 %.1f°）" %
            _DEFAULT_THRESHOLDS["push_facing_error_deg"]
        )
        suggestions.append(
            "检查 CHASE 阶段的 heading 是否过早偏离球-门线"
        )
    elif cond == "ball_local_x_in_window":
        suggestions.append(
            "扩大 push_ball_x_min / push_ball_x_max 范围"
        )
        suggestions.append(
            "检查 behind_offset 是否导致机器人目标位置离球过远或过近"
        )
    elif cond == "ball_local_y_centered":
        suggestions.append(
            "提高 push_ball_y_max 放宽球横向偏移容差"
        )
        suggestions.append(
            "检查 ALIGN_BEHIND 阶段的横向对齐逻辑（align_vy_max）"
        )
    elif cond == "lateral_err_small":
        suggestions.append(
            "提高 push_lateral_tol 放宽横向对齐容差"
        )
        suggestions.append(
            "提高 align_vy_max 加快横向纠正速度"
        )
    elif cond == "behind_depth_sufficient":
        suggestions.append(
            "降低 push_min_behind_depth 阈值"
        )
        suggestions.append(
            "检查机器人是否离球过近（ball_distance 应保持 0.2–0.4m）"
        )

    # Add push execution suggestions
    if push_execution.get("entered_push_forward"):
        if push_execution.get("ball_goal_progress_during_push", 0) is not None and not push_execution.get("has_goal_progress", True):
            suggestions.append(
                "PUSH 期间球未向球门推进：提高 push_vx 或检查推球方向"
            )
        if push_execution.get("has_excess_lateral_drift", False):
            suggestions.append(
                "PUSH 期间球横向漂移过大：提高 push_w_max 加强方向纠正"
            )

    if not suggestions:
        suggestions.append("轨迹正常，无需特定改进。")
    return suggestions


# ---------------------------------------------------------------------------
# TXT Report (Chinese)
# ---------------------------------------------------------------------------


def _format_txt_report(diagnosis: Dict[str, Any]) -> str:
    """Render diagnosis as a fixed-format Chinese text report."""
    lines: List[str] = []
    meta = diagnosis["meta"]
    state = diagnosis["state_summary"]

    lines.append("=" * 60)
    lines.append("  轨迹失败诊断报告")
    lines.append("=" * 60)
    lines.append("")
    lines.append("CSV路径: %s" % meta["csv_path"])
    lines.append("运行模式: %s  帧数: %d  时长: %.2fs" % (
        meta["run_mode"], meta["frame_count"], meta["duration_s"],
    ))
    lines.append("")

    # ---- Conclusion ----
    lines.append("结论:")
    if state["did_reach_push_forward"]:
        lines.append("  - 已进入 PUSH_FORWARD 状态。")
        pe = diagnosis.get("push_execution", {})
        if not pe.get("has_goal_progress", True):
            lines.append("  - 但 PUSH 期间球未向球门方向有效推进。")
        if pe.get("has_excess_lateral_drift", False):
            lines.append("  - PUSH 期间球横向漂移过大。")
    else:
        lines.append("  - 没有进入 PUSH_FORWARD。")
        pb = diagnosis.get("primary_blocker", {})
        cond = pb.get("condition", "unknown")
        if cond == "align_face_goal_too_late":
            face_dur = state.get("align_mode_durations_s", {}).get(
                "ALIGN_FACE_GOAL", 0.0
            )
            lines.append(
                "  - 主要卡在 align_face_goal_too_late："
                "ALIGN_FACE_GOAL 仅运行 %.2fs，"
                "来不及纠正偏航角误差。" % face_dur
            )
        elif cond == "chase_too_slow_or_bad_heading":
            lines.append("  - 主要卡在 chase_too_slow_or_bad_heading："
                         "CHASE 阶段未能在合理时间内接近球。")
        else:
            spec = next(
                (s for s in CONDITION_SPECS if s["name"] == cond), None
            )
            desc = spec["description"] if spec else cond
            yaw_st = (
                diagnosis.get("condition_analysis", {})
                .get("conditions", {})
                .get(cond, {})
            )
            pass_pct = yaw_st.get("pass_rate", 0.0) * 100
            lines.append(
                "  - 主要卡在 %s（通过率 %.1f%%）。" % (desc, pass_pct)
            )
    lines.append("")

    # ---- Key Evidence ----
    lines.append("关键证据:")
    lines.append("  - ball_dist_min = %.3fm" % (
        state.get("ball_dist_min") or 0.0
    ))
    t05 = state.get("time_to_enter_0_5m")
    if t05 is not None:
        lines.append("  - time_to_enter_0_5m = %.2fs" % t05)
    else:
        lines.append("  - 球从未进入 0.5m 范围")

    for sname, sdur in state.get("state_durations_s", {}).items():
        lines.append("  - %s = %.2fs" % (sname, sdur))

    for mname, mdur in state.get("align_mode_durations_s", {}).items():
        if mname in ("ALIGN_FACE_GOAL",) or mdur > 0.5:
            lines.append("  - align_mode %s = %.2fs" % (mname, mdur))

    # Add key condition values from the last frame
    conds = (
        diagnosis.get("condition_analysis", {}).get("conditions", {})
        if "condition_analysis" in diagnosis
        else {}
    )
    if conds:
        lines.append("")
        lines.append("  最后一帧条件值:")
        for name in sorted(conds.keys()):
            st = conds[name]
            lv = st.get("latest_value")
            if lv is not None:
                spec = next(
                    (s for s in CONDITION_SPECS if s["name"] == name), None
                )
                desc = spec["description"] if spec else name
                lines.append("  - %s = %.3f" % (desc, lv))

    lines.append("")

    # ---- Suggestions ----
    lines.append("建议:")
    for i, sug in enumerate(diagnosis.get("suggestions", []), 1):
        lines.append("  %d. %s" % (i, sug))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def diagnose_trajectory(
    csv_path: str,
    output_dir: Optional[str] = None,
    use_external_config: bool = True,
) -> Dict[str, Any]:
    """Run full trajectory failure diagnosis.

    Parameters
    ----------
    csv_path:
        Path to ``trajectory.csv``.
    output_dir:
        Directory for ``diagnosis.json`` / ``diagnosis.txt``.
        Defaults to the CSV's parent directory.
    use_external_config:
        When True, load thresholds from ``config.yaml`` (``fsm_mvp`` section).
        When False, use hardcoded MiniPushFSMConfig defaults only.

    Returns
    -------
    dict
        The complete diagnosis (also written to disk).
    """
    csv_path = Path(csv_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        diagnosis = {
            "meta": {
                "csv_path": str(csv_path),
                "frame_count": 0,
                "duration_s": 0.0,
                "run_mode": "unknown",
                "config_source": (
                    "config.yaml (fsm_mvp)"
                    if use_external_config
                    else "hardcoded_defaults"
                ),
            },
            "error": "Empty CSV (no data rows).",
        }
        _write_outputs(diagnosis, output_dir)
        return diagnosis

    thresholds, config_source = _load_thresholds(use_external_config)
    run_mode = rows[0].get("run_mode", "") or "unknown"
    start_t = _float(rows[0], "elapsed_s") or 0.0
    end_t = _float(rows[-1], "elapsed_s") or start_t
    duration = max(0.0, end_t - start_t)

    # Run layers
    state_layer = _analyze_state_layer(rows)

    diagnosis: Dict[str, Any] = {
        "meta": {
            "csv_path": str(csv_path),
            "frame_count": len(rows),
            "duration_s": duration,
            "run_mode": run_mode,
            "config_source": config_source,
            "diagnosis_applicable": run_mode in ("fsm_mvp", "push_to_goal"),
        },
        "state_summary": state_layer,
    }

    if not state_layer["did_reach_push_forward"]:
        condition_layer = _analyze_condition_layer(rows, state_layer, thresholds)
        diagnosis["condition_analysis"] = condition_layer
        diagnosis["primary_blocker"] = _identify_primary_blocker(
            condition_layer, state_layer, thresholds
        )
        diagnosis["push_execution"] = {"entered_push_forward": False}
    else:
        push_execution = _analyze_execution_layer(rows, state_layer)
        diagnosis["push_execution"] = push_execution
        if not push_execution.get("has_goal_progress", True):
            diagnosis["primary_blocker"] = {
                "condition": "push_no_goal_progress",
                "evidence": (
                    "PUSH_FORWARD was entered but ball made no progress "
                    "toward goal (goal_progress="
                    f"{push_execution.get('ball_goal_progress_during_push', 'N/A')}m)."
                ),
                "pass_rate": None,
            }

    diagnosis["suggestions"] = _build_suggestions(
        diagnosis.get("primary_blocker", {}),
        state_layer,
        diagnosis.get("push_execution", {}),
    )

    _write_outputs(diagnosis, output_dir)
    return diagnosis


def _write_outputs(diagnosis: Dict[str, Any], output_dir: Path) -> None:
    """Write diagnosis.json and diagnosis.txt to *output_dir*."""
    json_path = output_dir / "diagnosis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)
        f.write("\n")

    txt_path = output_dir / "diagnosis.txt"
    if "error" in diagnosis:
        txt_content = "诊断失败: %s\n" % diagnosis["error"]
    else:
        txt_content = _format_txt_report(diagnosis)
    txt_path.write_text(txt_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose why a MiniPushFSM trajectory failed to push the ball."
    )
    parser.add_argument(
        "csv_path",
        help="Path to trajectory.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: CSV parent directory)",
    )
    parser.add_argument(
        "--skip-external-config",
        action="store_true",
        help="Use only hardcoded defaults, skip config.yaml loading",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print diagnosis JSON to stdout instead of writing files",
    )
    args = parser.parse_args()

    if args.json_only:
        # Quick inspection mode — print JSON to stdout, don't write files
        csv_path = Path(args.csv_path).expanduser().resolve()
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        thresholds, _ = _load_thresholds(not args.skip_external_config)
        state_layer = _analyze_state_layer(rows)
        if not state_layer["did_reach_push_forward"]:
            condition_layer = _analyze_condition_layer(rows, state_layer, thresholds)
            blocker = _identify_primary_blocker(condition_layer, state_layer, thresholds)
        else:
            condition_layer = None
            blocker = None
        output = {
            "state_summary": {
                k: v
                for k, v in state_layer.items()
                if k in (
                    "states_entered",
                    "state_durations_s",
                    "align_mode_durations_s",
                    "did_reach_push_forward",
                    "last_state",
                    "time_to_enter_0_5m",
                    "ball_dist_min",
                )
            },
            "primary_blocker": blocker,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0

    diagnosis = diagnose_trajectory(
        args.csv_path,
        output_dir=args.output_dir,
        use_external_config=not args.skip_external_config,
    )
    print(
        "Diagnosis written to %s" % (
            Path(args.output_dir or Path(args.csv_path).parent)
            / "diagnosis.json"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
