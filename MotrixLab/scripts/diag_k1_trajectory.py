# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""K1 locomotion trajectory diagnostic tool.

Runs a trained K1 walk policy and records the base position, orientation,
velocity, commands, and foot contact state at every control step. Outputs a
CSV file and a terminal summary to help diagnose asymmetric gait (e.g.,
persistent leftward drift instead of straight forward walking).
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

from _source_path import ensure_source_path

ensure_source_path()

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry as env_registry
from motrix_envs.math import quaternion
from motrix_rl import registry as rl_registry
from motrix_rl import utils as rl_utils
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap

logger = logging.getLogger(__name__)


def discover_latest_policy(env_name: str) -> tuple[str, Path, Path]:
    """Discover the RL framework and best policy from the most recent training run.

    Returns:
        Tuple of (framework name, path to run directory, path to best policy)
    """
    base_dir = Path(f"runs/{env_name}")
    if not base_dir.exists():
        raise FileNotFoundError(f"No training results found for '{env_name}' in {base_dir}")

    frameworks = []
    for framework in ["rslrl"]:
        framework_dir = base_dir / framework
        if framework_dir.exists() and framework_dir.is_dir():
            training_runs = [d for d in framework_dir.iterdir() if d.is_dir()]
            if training_runs:
                latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
                frameworks.append((framework, latest_run.stat().st_mtime, latest_run))

    if not frameworks:
        raise FileNotFoundError(f"No RSLRL training runs found for '{env_name}' in {base_dir}")

    _, _, latest_run_dir = max(frameworks, key=lambda x: x[1])
    model_files = list(latest_run_dir.glob("model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {latest_run_dir}")

    def extract_iteration(filename):
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0

    best_policy = max(model_files, key=lambda f: (f.stat().st_mtime, extract_iteration(f)))
    logger.info(f"Auto-discovered policy: {best_policy}")
    return "rslrl", latest_run_dir, best_policy


def make_env(env_name: str, num_envs: int, sim_backend: str | None = None):
    """Create an NpEnv and disable domain randomization + observation noise."""
    env = env_registry.make(env_name, sim_backend=sim_backend, num_envs=num_envs)
    # Deterministic evaluation
    env.cfg.noise.add_noise = False
    env.cfg.domain_rand.push_robots = False
    return env


def get_left_right_dof_indices(actuator_names: list[str]):
    """Return (left_indices, right_indices) into the DOF position array."""
    left_indices = []
    right_indices = []
    for i, name in enumerate(actuator_names):
        if "Left_" in name:
            left_indices.append(i)
        elif "Right_" in name:
            right_indices.append(i)
    return np.array(left_indices, dtype=np.int64), np.array(right_indices, dtype=np.int64)


class TrajectoryRecorder:
    """Collects per-step trajectory data and writes a CSV + terminal summary."""

    def __init__(self, output_path: Path, fixed_cmd: np.ndarray | None = None):
        self.output_path = output_path
        self.fixed_cmd = fixed_cmd  # shape (3,) or None for resample mode
        self.rows: list[dict] = []

    def record(
        self,
        step: int,
        sim_time: float,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        local_linvel: np.ndarray,
        gyro: np.ndarray,
        commands: np.ndarray,
        left_contact: float,
        right_contact: float,
        dof_pos: np.ndarray,
        left_dof_indices: np.ndarray,
        right_dof_indices: np.ndarray,
    ):
        roll, pitch, yaw = quaternion.get_euler_xyz(base_quat.reshape(1, 4))
        left_dof_mean = float(np.mean(dof_pos[left_dof_indices])) if len(left_dof_indices) > 0 else 0.0
        right_dof_mean = float(np.mean(dof_pos[right_dof_indices])) if len(right_dof_indices) > 0 else 0.0

        self.rows.append(
            {
                "step": step,
                "sim_time": sim_time,
                "pos_x": float(base_pos[0]),
                "pos_y": float(base_pos[1]),
                "pos_z": float(base_pos[2]),
                "roll": float(roll[0]),
                "pitch": float(pitch[0]),
                "yaw": float(yaw[0]),
                "vx_local": float(local_linvel[0]),
                "vy_local": float(local_linvel[1]),
                "vz_local": float(local_linvel[2]),
                "wx": float(gyro[0]),
                "wy": float(gyro[1]),
                "wz": float(gyro[2]),
                "cmd_vx": float(commands[0]),
                "cmd_vy": float(commands[1]),
                "cmd_yaw": float(commands[2]),
                "left_contact": float(left_contact),
                "right_contact": float(right_contact),
                "left_dof_mean": left_dof_mean,
                "right_dof_mean": right_dof_mean,
            }
        )

    def write_csv(self):
        fieldnames = [
            "step", "sim_time",
            "pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw",
            "vx_local", "vy_local", "vz_local", "wx", "wy", "wz",
            "cmd_vx", "cmd_vy", "cmd_yaw",
            "left_contact", "right_contact",
            "left_dof_mean", "right_dof_mean",
        ]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        logger.info(f"Trajectory CSV written to {self.output_path}")

    def print_summary(self, policy_label: str, ctrl_dt: float):
        if not self.rows:
            logger.warning("No trajectory data to summarize.")
            return

        first = self.rows[0]
        last = self.rows[-1]
        duration = last["sim_time"] - first["sim_time"]
        dx = last["pos_x"] - first["pos_x"]
        dy = last["pos_y"] - first["pos_y"]
        dz = last["pos_z"] - first["pos_z"]
        displacement = np.sqrt(dx**2 + dy**2)
        direction_deg = np.degrees(np.arctan2(dy, dx))
        lateral_drift_rate = dy / duration if duration > 0 else 0.0
        yaw_drift = last["yaw"] - first["yaw"]
        yaw_drift_rate = np.degrees(yaw_drift) / duration if duration > 0 else 0.0

        n = len(self.rows)
        contacts = np.array([[r["left_contact"], r["right_contact"]] for r in self.rows])
        avg_left_contact = np.mean(contacts[:, 0]) * 100
        avg_right_contact = np.mean(contacts[:, 1]) * 100
        contact_asymmetry = abs(avg_left_contact - avg_right_contact)

        left_dof = np.array([r["left_dof_mean"] for r in self.rows])
        right_dof = np.array([r["right_dof_mean"] for r in self.rows])
        avg_dof_diff = np.mean(np.abs(left_dof - right_dof))

        cmd_label = ""
        if self.fixed_cmd is not None:
            cmd_label = f"\n  Fixed command: vx={self.fixed_cmd[0]:.2f}, vy={self.fixed_cmd[1]:.2f}, wyaw={self.fixed_cmd[2]:.2f}"
        else:
            cmd_label = "\n  Command mode: resample (random)"

        direction_label = f"{abs(direction_deg):.1f} deg {'LEFT' if direction_deg > 0 else 'RIGHT'}"
        if abs(direction_deg) < 0.5:
            direction_label = f"{direction_deg:.1f} deg (straight)"

        print(
            f"""
{'='*60}
  Trajectory Diagnostic Summary
{'='*60}
  Policy: {policy_label}
  Duration: {duration:.2f}s ({n} steps), Ctrl dt: {ctrl_dt}s{cmd_label}

  Start pos:  ({first['pos_x']:7.3f}, {first['pos_y']:7.3f}, {first['pos_z']:7.3f})
  End pos:    ({last['pos_x']:7.3f}, {last['pos_y']:7.3f}, {last['pos_z']:7.3f})
  Net displacement: {displacement:.3f} m
  Direction: {direction_label}
  Avg lateral drift: {lateral_drift_rate:.4f} m/s
  Avg yaw drift:     {yaw_drift_rate:.2f} deg/s

  Avg left contact:  {avg_left_contact:.1f}%
  Avg right contact: {avg_right_contact:.1f}%
  Contact asymmetry: {contact_asymmetry:.2f}%  {'<-- ASYMMETRIC' if contact_asymmetry > 5.0 else '<-- symmetric'}
  Avg |left - right| DOF mean: {avg_dof_diff:.4f}
{'='*60}
"""
        )


def run_trajectory(
    env,
    vec_env: RslrlNpEnvWrap,
    policy,
    num_steps: int,
    output_path: Path,
    fixed_cmd: np.ndarray | None = None,
    policy_label: str = "",
) -> TrajectoryRecorder:
    """Run the policy and record trajectory data."""
    ctrl_dt = env.cfg.ctrl_dt
    left_dof_idx, right_dof_idx = get_left_right_dof_indices(env.model.actuator_names)

    recorder = TrajectoryRecorder(output_path, fixed_cmd)

    obs, _ = vec_env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    for step in range(num_steps):
        state = vec_env._state
        if state is None:
            raise RuntimeError("Environment state is not initialized")

        # Override commands for fixed-cmd mode
        if fixed_cmd is not None:
            state.info["commands"][:] = fixed_cmd.astype(np.float32)

        # Read trajectory data from the environment (single env: index 0)
        pose = env._body.get_pose(state.data)
        base_pos = pose[0, :3]
        base_quat = pose[0, 3:7]
        local_linvel = env.get_local_linvel(state.data)[0]
        gyro = env.get_gyro(state.data)[0]
        commands = state.info["commands"][0]
        left_contact = float(state.info.get("left_contact", np.zeros(1))[0])
        right_contact = float(state.info.get("right_contact", np.zeros(1))[0])
        dof_pos = env.get_dof_pos(state.data)[0]

        sim_time = step * ctrl_dt
        recorder.record(
            step, sim_time,
            base_pos, base_quat,
            local_linvel, gyro,
            commands,
            left_contact, right_contact,
            dof_pos, left_dof_idx, right_dof_idx,
        )

        # Run inference
        with torch.no_grad():
            actions = policy(obs)

        # Step environment
        obs, rewards, dones, extras = vec_env.step(actions)
        if hasattr(policy, "reset"):
            policy.reset(dones.bool())

    recorder.write_csv()
    recorder.print_summary(policy_label, ctrl_dt)
    return recorder


def plot_trajectory(recorders: list[tuple[str, TrajectoryRecorder]], plot_path: Path):
    """Generate a trajectory plot (XY scatter + time series)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is not installed. Install it with: pip install matplotlib")
        return

    n = len(recorders)
    fig, axes = plt.subplots(2, max(n, 1), figsize=(5 * max(n, 1), 8), squeeze=False)

    for idx, (label, recorder) in enumerate(recorders):
        if not recorder.rows:
            continue
        rows = recorder.rows
        t = np.array([r["sim_time"] for r in rows])
        x = np.array([r["pos_x"] for r in rows])
        y = np.array([r["pos_y"] for r in rows])
        yaw = np.array([r["yaw"] for r in rows])

        # XY trajectory
        ax_xy = axes[0, idx]
        ax_xy.plot(x, y, "b-", linewidth=0.8, alpha=0.7)
        ax_xy.scatter(x[0], y[0], c="green", s=60, zorder=5, label="Start")
        ax_xy.scatter(x[-1], y[-1], c="red", s=60, zorder=5, label="End")
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_title(f"XY Trajectory - {label}")
        ax_xy.legend(fontsize=8)
        ax_xy.grid(True, alpha=0.3)
        ax_xy.axis("equal")

        # Time series: yaw and lateral position
        ax_ts = axes[1, idx]
        ax_ts.plot(t, y, "b-", linewidth=0.8, label="Y position (m)")
        ax_ts.plot(t, np.degrees(yaw), "r-", linewidth=0.8, label="Yaw (deg)")
        ax_ts.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel("Y (m) / Yaw (deg)")
        ax_ts.set_title(f"Lateral & Yaw Drift - {label}")
        ax_ts.legend(fontsize=8)
        ax_ts.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, max(n, 1)):
        for row in range(2):
            axes[row, idx].set_visible(False)

    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    logger.info(f"Trajectory plot saved to {plot_path}")


def parse_fixed_cmd(s: str) -> np.ndarray | None:
    """Parse a comma-separated fixed command string like '0.5,0,0'."""
    if s is None:
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--fixed-cmd requires 3 comma-separated values: vx,vy,wyaw")
    return np.array([float(x) for x in parts], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="K1 locomotion trajectory diagnostic tool. "
        "Records base position, orientation, and velocity at every step."
    )
    parser.add_argument("--env", default="k1-flat-terrain-walk", help="Environment name.")
    parser.add_argument("--policy", default=None, help="Path to a specific policy checkpoint (.pt).")
    parser.add_argument(
        "--policies",
        default=None,
        help="Comma-separated list of policy paths for multi-checkpoint comparison.",
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of control steps (default: 500 = 10s).")
    parser.add_argument(
        "--fixed-cmd",
        type=parse_fixed_cmd,
        default=None,
        help="Fixed velocity command as 'vx,vy,wyaw' (e.g. '0.5,0,0'). "
        "When omitted, the environment's random command resampling is used.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (default: 1).")
    parser.add_argument("--plot", action="store_true", help="Generate a trajectory plot (requires matplotlib).")
    parser.add_argument("--output-dir", default=None, help="Override output directory for CSV/plot files.")
    parser.add_argument(
        "--no-render", action="store_true", default=True, help="Disable rendering (rendering is off by default)."
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering during trajectory collection.")
    args = parser.parse_args()

    env_name = args.env
    sim_backend = None  # auto-detect

    # Resolve policies to evaluate
    policies_to_run: list[tuple[str, Path]] = []  # (label, path)

    discovered_run_dir: Path | None = None

    if args.policies:
        for p in args.policies.split(","):
            p = p.strip()
            policy_path = Path(p)
            if not policy_path.is_absolute():
                policy_path = Path.cwd() / policy_path
            if not policy_path.exists():
                logger.error(f"Policy not found: {policy_path}")
                sys.exit(1)
            policies_to_run.append((policy_path.name, policy_path))
    elif args.policy:
        policy_path = Path(args.policy)
        if not policy_path.is_absolute():
            policy_path = Path.cwd() / policy_path
        if not policy_path.exists():
            logger.error(f"Policy not found: {policy_path}")
            sys.exit(1)
        policies_to_run.append((policy_path.name, policy_path))
    else:
        # Auto-discover
        _, discovered_run_dir, policy_path = discover_latest_policy(env_name)
        policies_to_run.append((f"latest ({policy_path.name})", policy_path))

    # Create environment (shared across policies when comparing)
    logger.info(f"Creating environment: {env_name} (num_envs={args.num_envs})")
    env = make_env(env_name, args.num_envs, sim_backend)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build RSLRL config
    rlcfg = rl_registry.default_rl_cfg(env_name, "rslrl", backend="torch")
    rslrl_cfg = rlcfg.runner.to_dict()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.policy or args.policies:
        # Use the first policy's parent directory
        output_dir = policies_to_run[0][1].parent
    elif discovered_run_dir is not None:
        output_dir = discovered_run_dir
    else:
        _, discovered_run_dir, _ = discover_latest_policy(env_name)
        output_dir = discovered_run_dir

    # Run each policy
    recorders: list[tuple[str, TrajectoryRecorder]] = []

    for idx, (label, policy_path) in enumerate(policies_to_run):
        logger.info(f"Evaluating policy [{idx+1}/{len(policies_to_run)}]: {label}")

        # Create fresh wrapper and runner for each policy (runner state may be stale)
        vec_env = RslrlNpEnvWrap(env, device)
        runner = OnPolicyRunner(vec_env, rslrl_cfg, log_dir=None, device=device)
        runner.load(str(policy_path), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})
        policy = runner.get_inference_policy(device=device)

        safe_label = label.replace("/", "_").replace(" ", "_")
        csv_path = output_dir / f"trajectory_{safe_label}.csv"

        recorder = run_trajectory(
            env,
            vec_env,
            policy,
            args.steps,
            csv_path,
            fixed_cmd=args.fixed_cmd,
            policy_label=label,
        )
        recorders.append((label, recorder))

        # Optional rendering (keeps viewer alive briefly between policies)
        if args.render:
            vec_env.render()

    # Generate plot if requested
    if args.plot:
        plot_path = output_dir / "trajectory_comparison.png"
        plot_trajectory(recorders, plot_path)

    logger.info("Diagnostic complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
