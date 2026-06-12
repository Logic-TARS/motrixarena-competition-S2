#!/usr/bin/env python3
"""K1 motion calibration: measure actual robot movement under fixed velocity commands.

Usage:
    python tools/calibrate_k1_motion.py

Requires: requests (pip install requests)
Simulation and decider conda envs must exist (motrixsim0508, k1).
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_RUNNER = PROJECT_ROOT / "simulation" / "motrixsim" / "sim2sim_runner.py"
DECIDER = PROJECT_ROOT / "decider" / "decider.py"
POLICY = PROJECT_ROOT / "MotrixLab" / "exported" / "k1_top_run_1600_torchscript.pt"
WEBVIEW_PORT = 5811
SIM_PORT = 5555
DURATION = 5.0  # seconds per test
POLL_HZ = 2
REPEATS = 3

TEST_COMMANDS = [
    ([0.4, 0.0, 0.0], "vx+ forward"),
    ([-0.4, 0.0, 0.0], "vx- backward"),
    ([0.0, 0.4, 0.0], "vy+ left strafe"),
    ([0.0, -0.4, 0.0], "vy- right strafe"),
    ([0.0, 0.0, 0.4], "w+ turn left (CCW)"),
    ([0.0, 0.0, -0.4], "w- turn right (CW)"),
    ([0.4, 0.0, 0.1], "vx+ w+ forward+left"),
    ([0.4, 0.0, -0.1], "vx+ w- forward+right"),
]


def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _fetch_states() -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{WEBVIEW_PORT}/api/states", timeout=3
        ) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _post_command(event: str) -> bool:
    data = json.dumps({"event": event}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{WEBVIEW_PORT}/api/command",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception:
        return False


def wait_sim_ready(timeout: float = 120) -> bool:
    print("Waiting for simulation to be ready...", end=" ", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        states = _fetch_states()
        if states and "robot_rp0" in states and states["robot_rp0"].get("active"):
            print("ready.")
            return True
        time.sleep(2)
    print("TIMEOUT.")
    return False


def compute_metrics(samples: list[dict], cmd: list[float]) -> dict:
    if len(samples) < 2:
        return {"error": "not enough samples"}

    x0, y0, z0, yaw0 = samples[0]["x"], samples[0]["y"], samples[0]["z"], samples[0]["yaw"]
    xn, yn, zn, yawn = samples[-1]["x"], samples[-1]["y"], samples[-1]["z"], samples[-1]["yaw"]

    dx = xn - x0
    dy = yn - y0
    move_dir = math.atan2(dy, dx)
    yaw_delta = _angle_wrap(yawn - yaw0)
    dir_minus_yaw = _angle_wrap(move_dir - yawn)
    duration = samples[-1]["t"] - samples[0]["t"]
    speed = math.hypot(dx, dy) / max(duration, 0.01)
    z_vals = [s["z"] for s in samples]
    min_z = min(z_vals)
    fell = min_z < 0.45

    yaw_sign_ok = None
    if abs(cmd[2]) > 0.001:
        yaw_sign_ok = (yaw_delta > 0 and cmd[2] > 0) or (yaw_delta < 0 and cmd[2] < 0)

    return {
        "dx": dx,
        "dy": dy,
        "move_dir_deg": math.degrees(move_dir),
        "yaw_delta_deg": math.degrees(yaw_delta),
        "dir_minus_yaw_deg": math.degrees(dir_minus_yaw),
        "speed": speed,
        "min_z": min_z,
        "fell": fell,
        "yaw_sign_ok": yaw_sign_ok,
        "n_samples": len(samples),
        "duration": duration,
    }


def run_one_command(cmd: list[float], label: str) -> list[dict]:
    """Run one fixed-command test once, returning sampled trajectory."""
    samples = []
    cmd_str = f"{cmd[0]},{cmd[1]},{cmd[2]}"

    # Reset sim first
    _post_command("reset_env")
    time.sleep(0.5)

    # Start decider
    decider_args = [
        "conda", "run", "-n", "k1",
        "python", str(DECIDER),
        "--simulation", "--ip", "127.0.0.1", "--port", str(SIM_PORT),
        "--color", "red", "--id", "0",
        "--sim-fixed-cmd", cmd_str,
        "--sim-hz", "10",
    ]
    proc = subprocess.Popen(
        decider_args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    # Poll states
    interval = 1.0 / POLL_HZ
    deadline = time.time() + DURATION + 0.5  # small grace period
    start_time = None
    try:
        while time.time() < deadline:
            states = _fetch_states()
            if states and "robot_rp0" in states:
                r = states["robot_rp0"]
                t = time.time()
                if start_time is None:
                    start_time = t
                samples.append({
                    "t": t,
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "z": float(r["z"]),
                    "yaw": float(r["yaw"]),
                })
            time.sleep(interval)
    finally:
        # Kill decider
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    return samples


def main():
    import argparse
    ap = argparse.ArgumentParser(description="K1 motion calibration")
    ap.add_argument("--policy-flavor", default="motrixlab",
                    choices=["motrixlab", "legged_gym"],
                    help="K1 policy flavor (default: motrixlab)")
    ap.add_argument("--policy", default=None,
                    help="Override policy .pt file path")
    ns = ap.parse_args()

    # Check policy exists
    if ns.policy:
        policy_arg = ns.policy
    elif ns.policy_flavor == "motrixlab" and POLICY.exists():
        policy_arg = str(POLICY)
    else:
        policy_arg = None

    # Build sim args
    sim_args = [
        "conda", "run", "-n", "motrixsim0508",
        "python", str(SIM_RUNNER),
        "--team-size", "1",
        "--real-time",
        "--webview-port", str(WEBVIEW_PORT),
        "--k1-policy-flavor", ns.policy_flavor,
    ]
    if policy_arg:
        sim_args.extend(["--policy", policy_arg])

    print("=" * 60)
    print("K1 Motion Calibration Experiment")
    print("=" * 60)
    print(f"Policy: {policy_arg}, flavor: {ns.policy_flavor}")

    # Start sim
    print("Starting simulation...")
    sim_proc = subprocess.Popen(
        sim_args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    try:
        if not wait_sim_ready():
            print("ERROR: simulation failed to start", file=sys.stderr)
            return 1

        all_results = []
        for cmd, label in TEST_COMMANDS:
            print(f"\n--- {label} cmd={cmd} ({REPEATS} runs) ---")
            runs = []
            for r in range(1, REPEATS + 1):
                print(f"  run {r}/{REPEATS}...", end=" ", flush=True)
                samples = run_one_command(cmd, label)
                if len(samples) < 2:
                    print("FAILED (too few samples)")
                    runs.append({"error": "too few samples"})
                    continue
                m = compute_metrics(samples, cmd)
                runs.append(m)
                status = "FELL" if m["fell"] else "OK"
                print(
                    f"dx={m['dx']:+.3f} dy={m['dy']:+.3f} "
                    f"move_dir={m['move_dir_deg']:+.1f}° "
                    f"yaw_d={m['yaw_delta_deg']:+.1f}° "
                    f"dir-yaw={m['dir_minus_yaw_deg']:+.1f}° "
                    f"spd={m['speed']:.3f} z_min={m['min_z']:.3f} "
                    f"yaw_ok={m['yaw_sign_ok']} [{status}]"
                )
            all_results.append({"cmd": cmd, "label": label, "runs": runs})

        # Print summary table
        print("\n" + "=" * 60)
        print("RESULTS TABLE")
        print("=" * 60)
        print(
            f"{'cmd':>18} | {'dx(m)':>8} | {'dy(m)':>8} | {'move(°)':>8} | "
            f"{'yaw_d(°)':>8} | {'dir-yaw(°)':>8} | {'spd(m/s)':>8} | "
            f"{'z_min':>6} | fell | yaw_ok"
        )
        print("-" * 100)
        for entry in all_results:
            cmd, label = entry["cmd"], entry["label"]
            ok_runs = [r for r in entry["runs"] if "error" not in r]
            if not ok_runs:
                print(f"{label:>18} | {'-- no valid runs --':>60}")
                continue
            n = len(ok_runs)
            dx_m = sum(r["dx"] for r in ok_runs) / n
            dy_m = sum(r["dy"] for r in ok_runs) / n
            md_m = sum(r["move_dir_deg"] for r in ok_runs) / n
            yd_m = sum(r["yaw_delta_deg"] for r in ok_runs) / n
            dym_m = sum(r["dir_minus_yaw_deg"] for r in ok_runs) / n
            sp_m = sum(r["speed"] for r in ok_runs) / n
            z_m = sum(r["min_z"] for r in ok_runs) / n
            fell_any = any(r["fell"] for r in ok_runs)
            yaw_ok = ok_runs[0]["yaw_sign_ok"] if ok_runs else None
            cmd_str = f"[{cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f}]"
            print(
                f"{cmd_str:>18} | {dx_m:>+8.3f} | {dy_m:>+8.3f} | {md_m:>+8.1f} | "
                f"{yd_m:>+8.1f} | {dym_m:>+8.1f} | {sp_m:>8.3f} | "
                f"{z_m:>6.3f} | {str(fell_any):>4} | {str(yaw_ok):>6}"
            )

        # Calibration conclusions
        print("\n" + "=" * 60)
        print("CALIBRATION CONCLUSIONS")
        print("=" * 60)

        # Find turning results
        turn_left = None
        turn_right = None
        fwd = None
        left_strafe = None
        right_strafe = None
        for entry in all_results:
            cmd = entry["cmd"]
            ok_runs = [r for r in entry["runs"] if "error" not in r]
            if not ok_runs:
                continue
            if cmd == [0.0, 0.0, 0.4]:
                turn_left = ok_runs
            elif cmd == [0.0, 0.0, -0.4]:
                turn_right = ok_runs
            elif cmd == [0.4, 0.0, 0.0]:
                fwd = ok_runs
            elif cmd == [0.0, 0.4, 0.0]:
                left_strafe = ok_runs
            elif cmd == [0.0, -0.4, 0.0]:
                right_strafe = ok_runs

        if turn_left and turn_right:
            tl_yd = sum(r["yaw_delta_deg"] for r in turn_left) / len(turn_left)
            tr_yd = sum(r["yaw_delta_deg"] for r in turn_right) / len(turn_right)
            print(f"  w=+0.4 → yaw_delta={tl_yd:+.1f}°  (expect >0 for CCW)")
            print(f"  w=-0.4 → yaw_delta={tr_yd:+.1f}°  (expect <0 for CW)")
            if tl_yd > 0 and tr_yd < 0:
                print("  => Yaw sign is CONSISTENT: cmd_w>0 = CCW, cmd_w<0 = CW")
                print("  => Decider should use: cmd_w = -k * ball_angle")
            elif tl_yd < 0 and tr_yd > 0:
                print("  => Yaw sign is INVERTED: cmd_w>0 = CW")
                print("  => Decider should use: cmd_w = +k * ball_angle")
            else:
                print("  => Yaw response is UNCLEAR — check raw data")

        if fwd:
            fwd_dym = sum(r["dir_minus_yaw_deg"] for r in fwd) / len(fwd)
            fwd_spd = sum(r["speed"] for r in fwd) / len(fwd)
            print(f"  vx=+0.4 → dir_minus_yaw={fwd_dym:+.1f}° speed={fwd_spd:.3f} m/s")
            if abs(fwd_dym) < 15:
                print("  => Forward direction OK (moves along body x-axis)")
            elif abs(fwd_dym) > 75:
                print("  => WARNING: forward command may be swapped or base orientation is offset")
            else:
                print(f"  => Forward direction OFFSET by ~{fwd_dym:+.0f}°")

        if left_strafe and right_strafe:
            ls_spd = sum(r["speed"] for r in left_strafe) / len(left_strafe)
            rs_spd = sum(r["speed"] for r in right_strafe) / len(right_strafe)
            ls_fell = any(r["fell"] for r in left_strafe)
            rs_fell = any(r["fell"] for r in right_strafe)
            print(f"  vy=+0.4 → speed={ls_spd:.3f} m/s fell={ls_fell}")
            print(f"  vy=-0.4 → speed={rs_spd:.3f} m/s fell={rs_fell}")
            if ls_fell or rs_fell:
                print("  => WARNING: lateral commands cause falls — disable vy in chase controller")
            elif ls_spd < 0.03 and rs_spd < 0.03:
                print("  => Lateral movement is negligible — keep vy=0 in chase controller")
            else:
                print("  => Lateral movement works — small vy adjustments may be usable")

        # Chase speed recommendation
        all_fwd_speeds = []
        for entry in all_results:
            if entry["cmd"][0] > 0 and entry["cmd"][1] == 0 and entry["cmd"][2] == 0:
                for r in entry["runs"]:
                    if "speed" in r:
                        all_fwd_speeds.append(r["speed"])
        if all_fwd_speeds:
            ratio = sum(all_fwd_speeds) / len(all_fwd_speeds) / 0.4
            print(f"  Forward speed ratio: ~{ratio:.3f} m/s per unit cmd_x")
            print(f"  Recommended chase cmd_x range: 0.15–0.45 (→ ~{0.15*ratio:.2f}–{0.45*ratio:.2f} m/s)")

    finally:
        print("\nStopping simulation...")
        try:
            os.killpg(os.getpgid(sim_proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            sim_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sim_proc.kill()
            sim_proc.wait()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
