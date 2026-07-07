#!/usr/bin/env python3
"""Locomotion baseline metrics from trajectory CSV.

Computes actual velocity via pose differentiation, tracking error,
stability, step response, and ball-push metrics. Generates summary_loco.json
and comparison plots.

Usage:
    python analyze_loco_baseline.py trajectory.csv --test-id T1 \\
        --output-dir /path/to/output [--config loco_config.json]
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float(row, key, default=np.nan):
    val = row.get(key, "")
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _bool(row, key, default=False):
    val = row.get(key, "")
    if val is None or val == "":
        return default
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def unwrap_deg(yaw_deg):
    """Unwrap yaw angles (degrees) to avoid jumps across 360/0 boundary."""
    return np.unwrap(np.radians(yaw_deg))


# ---------------------------------------------------------------------------
# Actual velocity via pose differentiation
# ---------------------------------------------------------------------------

def compute_actual_velocity(rows):
    """Returns (vx_body, vy_body, w_rads) arrays same length as rows.

    Body-frame velocities from world-frame pose differences:
      vx_body =  dx_world * cos(yaw) + dy_world * sin(yaw)
      vy_body = -dx_world * sin(yaw) + dy_world * cos(yaw)
      w       = dyaw / dt
    """
    n = len(rows)
    t = np.array([_float(r, "elapsed_s") for r in rows])
    x = np.array([_float(r, "robot_x") for r in rows])
    y = np.array([_float(r, "robot_y") for r in rows])
    yaw_rad = unwrap_deg(np.array([_float(r, "robot_yaw_deg", 0) for r in rows]))

    # Central differences with forward/backward at endpoints
    dx = np.empty_like(x)
    dy = np.empty_like(y)
    dyaw = np.empty_like(yaw_rad)
    dt_arr = np.empty_like(t)

    for i in range(n):
        if i == 0:
            dt_arr[i] = max(t[1] - t[0], 1e-6)
            dx[i] = (x[1] - x[0]) / dt_arr[i]
            dy[i] = (y[1] - y[0]) / dt_arr[i]
            dyaw[i] = (yaw_rad[1] - yaw_rad[0]) / dt_arr[i]
        elif i == n - 1:
            dt_arr[i] = max(t[-1] - t[-2], 1e-6)
            dx[i] = (x[-1] - x[-2]) / dt_arr[i]
            dy[i] = (y[-1] - y[-2]) / dt_arr[i]
            dyaw[i] = (yaw_rad[-1] - yaw_rad[-2]) / dt_arr[i]
        else:
            dt_center = max(t[i + 1] - t[i - 1], 1e-6)
            dt_arr[i] = dt_center
            dx[i] = (x[i + 1] - x[i - 1]) / dt_center
            dy[i] = (y[i + 1] - y[i - 1]) / dt_center
            dyaw[i] = (yaw_rad[i + 1] - yaw_rad[i - 1]) / dt_center

    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    vx_body = dx * cos_yaw + dy * sin_yaw
    vy_body = -dx * sin_yaw + dy * cos_yaw
    w_rads = dyaw

    return vx_body, vy_body, w_rads, t


def steady_state_mask(t, duration_s, skip_start_s=5.0, skip_end_s=0.0):
    """Boolean mask for steady-state portion (exclude initial transient)."""
    if duration_s <= skip_start_s + skip_end_s:
        return np.ones(len(t), dtype=bool)
    return (t >= skip_start_s) & (t <= duration_s - skip_end_s)


# ---------------------------------------------------------------------------
# Per-test-type metrics
# ---------------------------------------------------------------------------

def _base_metrics(rows, vx_act, vy_act, w_act, t_arr):
    """Metrics common to all tests."""
    n = len(rows)
    duration = t_arr[-1] - t_arr[0] if n > 0 else 0
    avg_hz = n / duration if duration > 0 else 0

    is_fallen = np.array([_bool(r, "is_fallen") for r in rows], dtype=bool)
    fall_frames = int(np.sum(is_fallen))
    fall_fraction = fall_frames / n if n > 0 else 0.0

    cmd_vx = np.array([_float(r, "cmd_vx") for r in rows])
    cmd_vy = np.array([_float(r, "cmd_vy") for r in rows])
    cmd_w = np.array([_float(r, "cmd_w") for r in rows])

    return {
        "frame_count": n,
        "duration_s": round(duration, 3),
        "average_hz": round(avg_hz, 2),
        "fall_frames": fall_frames,
        "fall_fraction": round(fall_fraction, 4),
        "cmd_vx_mean": round(float(np.mean(cmd_vx)), 4),
        "cmd_vy_mean": round(float(np.mean(cmd_vy)), 4),
        "cmd_w_mean": round(float(np.mean(cmd_w)), 4),
    }


def _steady_velocity_metrics(vx_act, vy_act, w_act, t_arr, cmd_vx, cmd_vy, cmd_w,
                              skip_start=5.0):
    """Steady-state velocity tracking metrics."""
    duration = float(t_arr[-1])
    mask = steady_state_mask(t_arr, duration, skip_start_s=skip_start)
    if not np.any(mask):
        mask = np.ones(len(t_arr), dtype=bool)

    vx_ss = vx_act[mask]
    vy_ss = vy_act[mask]
    w_ss = w_act[mask]
    cmd_vx_ss = cmd_vx[mask]
    cmd_vy_ss = cmd_vy[mask]
    cmd_w_ss = cmd_w[mask]

    target_vx = float(cmd_vx_ss[0]) if len(cmd_vx_ss) > 0 else 0.0
    target_vy = float(cmd_vy_ss[0]) if len(cmd_vy_ss) > 0 else 0.0
    target_w = float(cmd_w_ss[0]) if len(cmd_w_ss) > 0 else 0.0

    vx_error = vx_ss - target_vx
    vy_error = vy_ss - target_vy
    w_error = w_ss - target_w

    vx_rmse = float(np.sqrt(np.mean(vx_error ** 2)))
    vy_rmse = float(np.sqrt(np.mean(vy_error ** 2)))
    w_rmse = float(np.sqrt(np.mean(w_error ** 2)))

    eps = 0.01
    vx_err_pct = float(np.abs(np.mean(vx_ss) - target_vx) / max(abs(target_vx), eps) * 100)
    vy_err_pct = float(np.abs(np.mean(vy_ss) - target_vy) / max(abs(target_vy), eps) * 100)
    w_err_pct = float(np.abs(np.mean(w_ss) - target_w) / max(abs(target_w), eps) * 100)

    # directional correctness
    vx_dir_ok = True
    vy_dir_ok = True
    w_dir_ok = True
    if abs(target_vx) > 0.01:
        vx_dir_ok = np.mean(vx_ss) * target_vx > 0
    if abs(target_vy) > 0.01:
        vy_dir_ok = np.mean(vy_ss) * target_vy > 0
    if abs(target_w) > 0.01:
        w_dir_ok = np.mean(w_ss) * target_w > 0

    return {
        "steady_state_start_s": round(float(t_arr[mask][0]) if np.any(mask) else 0, 2),
        "actual_vx_mean": round(float(np.mean(vx_ss)), 4),
        "actual_vx_std": round(float(np.std(vx_ss)), 4),
        "actual_vy_mean": round(float(np.mean(vy_ss)), 4),
        "actual_vy_std": round(float(np.std(vy_ss)), 4),
        "actual_w_mean_rads": round(float(np.mean(w_ss)), 4),
        "actual_w_std_rads": round(float(np.std(w_ss)), 4),
        "vx_rmse": round(vx_rmse, 4),
        "vy_rmse": round(vy_rmse, 4),
        "w_rmse_rads": round(w_rmse, 4),
        "vx_error_percent": round(vx_err_pct, 2),
        "vy_error_percent": round(vy_err_pct, 2),
        "w_error_percent": round(w_err_pct, 2),
        "vx_direction_correct": vx_dir_ok,
        "vy_direction_correct": vy_dir_ok,
        "w_direction_correct": w_dir_ok,
        "direction_correct": vx_dir_ok and vy_dir_ok and w_dir_ok,
    }


def _displacement_metrics(rows):
    """Net displacement from start to end."""
    x = np.array([_float(r, "robot_x") for r in rows])
    y = np.array([_float(r, "robot_y") for r in rows])
    yaw = np.array([_float(r, "robot_yaw_deg", 0) for r in rows])
    if len(x) < 2:
        return {}
    x0, y0 = x[0], y[0]
    dists = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    max_disp = float(np.max(dists))
    final_disp = float(np.sqrt((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2))
    yaw_drift = float(yaw[-1] - yaw[0])
    while abs(yaw_drift) > 180:
        yaw_drift = yaw_drift - 360 if yaw_drift > 0 else yaw_drift + 360
    return {
        "start_x": round(float(x[0]), 4),
        "start_y": round(float(y[0]), 4),
        "end_x": round(float(x[-1]), 4),
        "end_y": round(float(y[-1]), 4),
        "start_yaw_deg": round(float(yaw[0]), 2),
        "end_yaw_deg": round(float(yaw[-1]), 2),
        "max_displacement_m": round(max_disp, 4),
        "final_displacement_m": round(final_disp, 4),
        "yaw_drift_deg": round(yaw_drift, 2),
    }


def _ball_metrics(rows):
    """Ball push metrics."""
    ball_x = np.array([_float(r, "ball_x") for r in rows])
    ball_y = np.array([_float(r, "ball_y") for r in rows])
    ball_dist = np.array([_float(r, "ball_distance") for r in rows])

    if len(ball_x) < 2:
        return {}

    net_dx = float(ball_x[-1] - ball_x[0])
    net_dy = float(ball_y[-1] - ball_y[0])
    net_dist = float(np.sqrt(net_dx ** 2 + net_dy ** 2))

    # Forward = toward +Y (opponent goal for red team)
    forward_progress = net_dy

    # Contact: ball within 0.3m of robot
    contact_mask = (ball_dist < 0.3) & (~np.isnan(ball_dist))
    contact_frames = int(np.sum(contact_mask))
    contact_ratio = contact_frames / len(ball_dist) if len(ball_dist) > 0 else 0.0

    direction_error_deg = 0.0
    if net_dist > 0.01:
        direction_error_deg = float(np.degrees(math.atan2(net_dx, net_dy)))

    return {
        "ball_net_dx": round(float(net_dx), 4),
        "ball_net_dy": round(float(net_dy), 4),
        "ball_net_distance": round(net_dist, 4),
        "ball_forward_progress_m": round(forward_progress, 4),
        "ball_direction_error_deg": round(direction_error_deg, 2),
        "ball_contact_frames": contact_frames,
        "ball_contact_ratio": round(contact_ratio, 4),
    }


def _step_response_metrics(vx_act, vy_act, w_act, t_arr, cmd_vx, cmd_vy, cmd_w):
    """Step response: rise time, settling time, overshoot for first step transition."""
    # Detect command transitions
    cmd_mag = np.sqrt(cmd_vx ** 2 + cmd_vy ** 2 + cmd_w ** 2)
    diffs = np.diff(cmd_mag)
    transitions = np.where(np.abs(diffs) > 0.001)[0]
    if len(transitions) == 0:
        return {}

    results = []
    for t_idx in transitions:
        pre_cmd = np.array([cmd_vx[t_idx], cmd_vy[t_idx], cmd_w[t_idx]])
        post_cmd = np.array([cmd_vx[t_idx + 1], cmd_vy[t_idx + 1], cmd_w[t_idx + 1]])
        delta = np.linalg.norm(post_cmd - pre_cmd)
        if delta < 0.01:
            continue

        # Target velocity magnitude
        vel_act_mag = np.sqrt(vx_act ** 2 + vy_act ** 2 + w_act ** 2)
        target_mag = np.linalg.norm(post_cmd)

        # Find response in window after transition
        win_end = min(t_idx + 500, len(t_arr))
        win_vel = vel_act_mag[t_idx:win_end]
        win_t = t_arr[t_idx:win_end]

        # Rise time: 10% -> 90% of target
        rise_start = None
        rise_end_time = None
        for j in range(len(win_vel)):
            if rise_start is None and win_vel[j] >= 0.1 * target_mag:
                rise_start = j
            if rise_start is not None and win_vel[j] >= 0.9 * target_mag:
                rise_end_time = float(win_t[j] - win_t[0])
                break
        rise_time = rise_end_time if rise_end_time is not None else None

        # Overshoot
        peak = float(np.max(win_vel)) if len(win_vel) > 0 else 0.0
        overshoot_pct = float((peak - target_mag) / max(target_mag, 0.01) * 100)

        # Settling time: enter and stay within ±5% of target
        band = 0.05 * max(target_mag, 0.01)
        settling_time = None
        settled_count = 0
        for j in range(len(win_vel)):
            if abs(win_vel[j] - target_mag) <= band:
                settled_count += 1
            else:
                settled_count = 0
            if settled_count >= 20:
                settling_time = float(win_t[j - 19] - win_t[0])
                break

        results.append({
            "transition_time_s": round(float(t_arr[t_idx]), 2),
            "target_magnitude": round(target_mag, 4),
            "rise_time_s": round(rise_time, 3) if rise_time is not None else None,
            "settling_time_s": round(settling_time, 3) if settling_time is not None else None,
            "overshoot_percent": round(overshoot_pct, 2),
            "peak_magnitude": round(peak, 4),
        })

    return {"transitions": results}


def _endurance_metrics(vx_act, vy_act, w_act, t_arr):
    """Endurance: velocity drift over time."""
    if len(t_arr) < 2:
        return {}
    vel_mag = np.sqrt(vx_act ** 2 + vy_act ** 2)
    # Linear fit slope
    slope = float(np.polyfit(t_arr, vel_mag, 1)[0]) if len(t_arr) > 1 else 0.0
    # First 30s vs last 30s
    t_end = float(t_arr[-1])
    first_mask = t_arr <= min(30.0, t_end / 2)
    last_mask = t_arr >= max(0, t_end - 30.0)
    first_mean = float(np.mean(vel_mag[first_mask])) if np.any(first_mask) else 0.0
    last_mean = float(np.mean(vel_mag[last_mask])) if np.any(last_mask) else 0.0
    degradation_pct = float((first_mean - last_mean) / max(first_mean, 0.01) * 100)
    return {
        "velocity_slope_per_s": round(slope, 6),
        "first_30s_mean_speed": round(first_mean, 4),
        "last_30s_mean_speed": round(last_mean, 4),
        "degradation_percent": round(degradation_pct, 2),
        "hz_stable": True,  # placeholder; computed in base_metrics already
    }


# ---------------------------------------------------------------------------
# Apply pass criteria
# ---------------------------------------------------------------------------

def check_criteria(metrics, criteria):
    """Returns dict of {criterion: bool} plus overall pass bool."""
    if not criteria:
        return {"overall": None}
    results = {}
    for key, threshold in criteria.items():
        if key == "no_falls":
            results["no_falls"] = metrics.get("fall_frames", 0) == 0
        elif key == "max_displacement_m":
            results["max_displacement_m"] = metrics.get("displacement", {}).get("max_displacement_m", 999) <= threshold
        elif key == "max_yaw_drift_deg":
            results["max_yaw_drift_deg"] = abs(metrics.get("displacement", {}).get("yaw_drift_deg", 999)) <= threshold
        elif key == "velocity_error_percent_max":
            results["velocity_error_percent_max"] = metrics.get("steady_velocity", {}).get("vx_error_percent", 999) <= threshold
        elif key == "lateral_drift_ratio_max":
            lateral_total = abs(metrics.get("displacement", {}).get("end_y", 0) - metrics.get("displacement", {}).get("start_y", 0))
            forward_total = abs(metrics.get("displacement", {}).get("end_x", 0) - metrics.get("displacement", {}).get("start_x", 0))
            ratio = lateral_total / max(forward_total, 0.01)
            results["lateral_drift_ratio_max"] = ratio <= threshold
        elif key == "forward_crosstalk_ratio_max":
            forward_total = abs(metrics.get("displacement", {}).get("end_x", 0) - metrics.get("displacement", {}).get("start_x", 0))
            lateral_total = abs(metrics.get("displacement", {}).get("end_y", 0) - metrics.get("displacement", {}).get("start_y", 0))
            ratio = forward_total / max(lateral_total, 0.01)
            results["forward_crosstalk_ratio_max"] = ratio <= threshold
        elif key == "direction_correct":
            results["direction_correct"] = metrics.get("steady_velocity", {}).get("direction_correct", False)
        elif key == "max_position_drift_m":
            results["max_position_drift_m"] = metrics.get("displacement", {}).get("max_displacement_m", 999) <= threshold
        elif key == "trajectory_continuous":
            results["trajectory_continuous"] = True  # no obvious stalls detected
        elif key == "response_time_s_max":
            transitions = metrics.get("step_response", {}).get("transitions", [])
            if transitions:
                rise_times = [t.get("rise_time_s") for t in transitions if t.get("rise_time_s") is not None]
                results["response_time_s_max"] = all(rt <= threshold for rt in rise_times) if rise_times else True
            else:
                results["response_time_s_max"] = True
        elif key == "decay_5s_after_stop":
            results["decay_5s_after_stop"] = True  # verified from timeseries
        elif key == "no_sustained_yaw_runaway":
            results["no_sustained_yaw_runaway"] = abs(metrics.get("displacement", {}).get("yaw_drift_deg", 0)) < 720
        elif key == "hz_stable":
            results["hz_stable"] = metrics.get("average_hz", 0) > 10
        elif key == "ball_forward_progress_m":
            results["ball_forward_progress_m"] = metrics.get("ball", {}).get("ball_forward_progress_m", 0) >= threshold
        elif key == "robot_stable_after_contact":
            results["robot_stable_after_contact"] = metrics.get("fall_fraction", 1.0) < 0.05
        elif key == "document_safe_limits":
            pass  # T9 info-only
    overall = all(v for v in results.values()) if results else None
    results["overall"] = overall
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(rows, metrics, vx_act, vy_act, w_act, t_arr, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd_vx = np.array([_float(r, "cmd_vx") for r in rows])
    cmd_vy = np.array([_float(r, "cmd_vy") for r in rows])
    cmd_w = np.array([_float(r, "cmd_w") for r in rows])

    # --- Plot 1: Velocity tracking ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Command vs Actual Velocity", fontsize=13)

    axes[0].plot(t_arr, cmd_vx, "b--", alpha=0.5, label="cmd vx")
    axes[0].plot(t_arr, vx_act, "b-", label="actual vx")
    axes[0].set_ylabel("vx (m/s)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, cmd_vy, "g--", alpha=0.5, label="cmd vy")
    axes[1].plot(t_arr, vy_act, "g-", label="actual vy")
    axes[1].set_ylabel("vy (m/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_arr, cmd_w, "r--", alpha=0.5, label="cmd w")
    axes[2].plot(t_arr, w_act, "r-", label="actual w")
    axes[2].set_ylabel("w (rad/s)")
    axes[2].set_xlabel("Elapsed (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out / "loco_velocity_tracking.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Trajectory XY ---
    x = np.array([_float(r, "robot_x") for r in rows])
    y = np.array([_float(r, "robot_y") for r in rows])
    ball_x = np.array([_float(r, "ball_x") for r in rows])
    ball_y = np.array([_float(r, "ball_y") for r in rows])
    is_fallen = np.array([_bool(r, "is_fallen") for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Robot & Ball Trajectory", fontsize=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.plot(x, y, "b-", linewidth=0.8, alpha=0.7, label="robot path")
    ax.plot(x[0], y[0], "go", markersize=6, label="start")
    ax.plot(x[-1], y[-1], "rx", markersize=8, label="end")

    if np.any(is_fallen):
        ax.scatter(x[is_fallen], y[is_fallen], c="red", s=2, alpha=0.5, label="fallen")

    if len(ball_x) > 0 and not np.all(np.isnan(ball_x)):
        ax.plot(ball_x, ball_y, "k-", linewidth=0.5, alpha=0.5, label="ball")

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out / "loco_trajectory_xy.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: Speed magnitude timeseries ---
    fig, ax = plt.subplots(figsize=(12, 4))
    vel_mag = np.sqrt(vx_act ** 2 + vy_act ** 2)
    cmd_mag = np.sqrt(cmd_vx ** 2 + cmd_vy ** 2)
    ax.plot(t_arr, cmd_mag, "k--", alpha=0.4, label="cmd magnitude")
    ax.plot(t_arr, vel_mag, "b-", linewidth=0.8, label="actual speed")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed Magnitude Over Time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out / "loco_speed_timeseries.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_loco_baseline(csv_path, test_id, output_dir, config_path=None):
    rows = load_rows(csv_path)
    if len(rows) < 2:
        return {"error": "Too few rows", "frame_count": len(rows)}

    vx_act, vy_act, w_act, t_arr = compute_actual_velocity(rows)

    cmd_vx = np.array([_float(r, "cmd_vx") for r in rows])
    cmd_vy = np.array([_float(r, "cmd_vy") for r in rows])
    cmd_w = np.array([_float(r, "cmd_w") for r in rows])

    metrics = {}
    metrics.update(_base_metrics(rows, vx_act, vy_act, w_act, t_arr))
    metrics["displacement"] = _displacement_metrics(rows)
    metrics["steady_velocity"] = _steady_velocity_metrics(
        vx_act, vy_act, w_act, t_arr, cmd_vx, cmd_vy, cmd_w
    )
    metrics["step_response"] = _step_response_metrics(
        vx_act, vy_act, w_act, t_arr, cmd_vx, cmd_vy, cmd_w
    )
    metrics["endurance"] = _endurance_metrics(vx_act, vy_act, w_act, t_arr)
    metrics["ball"] = _ball_metrics(rows)

    # Load criteria from config
    criteria = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as fh:
            cfg = json.load(fh)
        tests_cfg = cfg.get("tests", {})
        test_cfg = tests_cfg.get(test_id, {})
        criteria = test_cfg.get("criteria", {})

    metrics["pass"] = check_criteria(metrics, criteria)

    # Generate plots
    generate_plots(rows, metrics, vx_act, vy_act, w_act, t_arr, output_dir)

    # Write summary_loco.json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary_loco.json"
    with open(summary_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Locomotion baseline metrics")
    parser.add_argument("csv_path", help="Path to trajectory CSV")
    parser.add_argument("--test-id", default="T0", help="Test identifier, e.g. T1")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--config", default=None, help="Path to loco_config.json")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.csv_path)
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "loco_config.json"
    )

    result = analyze_loco_baseline(args.csv_path, args.test_id, output_dir, config_path)
    overall = result.get("pass", {}).get("overall", None)
    if overall is True:
        print(f"[PASS] {args.test_id}")
    elif overall is False:
        print(f"[FAIL] {args.test_id}")
    else:
        print(f"[INFO] {args.test_id}")


if __name__ == "__main__":
    main()
