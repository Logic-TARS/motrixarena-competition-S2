"""Deterministic evaluation of K1 point-navigate policy across multiple episodes.

Usage:
    cd MotrixLab
    conda run -n sim_soccer_rl env PYTHONPATH=./motrix_envs/src:./motrix_rl/src \
        python scripts/eval_point_nav.py --checkpoints runs/k1-point-navigate/rslrl/<run>/model_299.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "motrix_envs" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "motrix_rl" / "src"))

import numpy as np
import torch
import torch.nn as nn
from motrix_envs import registry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=None,
                        help="Path to a single policy checkpoint")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="If set, run sweep over these checkpoint paths instead of --policy")
    return parser.parse_args()


def load_policy(checkpoint_path, device):
    """Load an RSLRL checkpoint and return a callable policy function."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_state = ckpt["actor_state_dict"]

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("norm_mean", actor_state["obs_normalizer._mean"].clone())
            self.register_buffer("norm_var", actor_state["obs_normalizer._var"].clone())
            self.register_buffer("norm_std", actor_state["obs_normalizer._std"].clone())
            self.register_buffer("norm_count", actor_state["obs_normalizer.count"].clone())

            layers = []
            in_dim = actor_state["mlp.0.weight"].shape[1]
            for i in range(0, 100, 2):
                w_key = f"mlp.{i}.weight"
                b_key = f"mlp.{i}.bias"
                if w_key not in actor_state:
                    break
                out_dim = actor_state[w_key].shape[0]
                linear = nn.Linear(in_dim, out_dim)
                linear.weight.data.copy_(actor_state[w_key])
                linear.bias.data.copy_(actor_state[b_key])
                layers.append(linear)
                if f"mlp.{i + 2}.weight" in actor_state:
                    layers.append(nn.ELU(alpha=1.0))
                in_dim = out_dim
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            mean = (x - self.norm_mean) / (self.norm_std + 1e-8)
            return self.mlp(mean)

    policy = Policy().to(device)
    policy.eval()
    return policy


def evaluate(env, policy, device, num_envs, max_steps):
    state = env.init_state()
    obs = torch.from_numpy(state.obs).to(device)

    reached = np.zeros(num_envs, dtype=bool)
    fallen = np.zeros(num_envs, dtype=bool)
    start_dist = state.info["target_dist"].copy()
    current_start_dist = start_dist.copy()
    step_count = 0
    done_count = 0
    completed_progress = []
    reward_sum = np.zeros(num_envs, dtype=np.float64)
    forward_vel_sum = np.zeros(num_envs, dtype=np.float64)
    progress_vel_sum = np.zeros(num_envs, dtype=np.float64)
    command_vx_sum = np.zeros(num_envs, dtype=np.float64)
    command_wz_abs_sum = np.zeros(num_envs, dtype=np.float64)

    for step in range(max_steps):
        with torch.no_grad():
            actions = policy(obs)
        actions_np = actions.cpu().numpy()

        state = env.step(actions_np)
        commands = state.info["commands"]
        progress_vel = (state.info["prev_target_dist"] - state.info["target_dist"]) / env.cfg.ctrl_dt
        progress_vel = np.clip(progress_vel, -0.5, 0.8)

        arrived = state.info.get(
            "arrived_done",
            state.info["target_dist"] <= env.cfg.point_config.arrival_radius,
        )
        fall_done = state.info.get("fall_done", state.terminated & ~arrived)
        done = state.done.astype(bool)

        reached |= arrived
        fallen |= fall_done
        done_count += int(np.sum(done))
        if np.any(done):
            arrived_done = done & arrived
            fall_done_mask = done & ~arrived
            if np.any(arrived_done):
                completed_progress.extend(
                    (current_start_dist[arrived_done] - env.cfg.point_config.arrival_radius).tolist()
                )
            if np.any(fall_done_mask):
                completed_progress.extend(np.zeros(int(np.sum(fall_done_mask)), dtype=np.float32).tolist())
            current_start_dist[done] = state.info["target_dist"][done]

        reward_sum += state.reward
        forward_vel_sum += env.get_local_linvel(state.data)[:, 0]
        progress_vel_sum += progress_vel
        command_vx_sum += commands[:, 0]
        command_wz_abs_sum += np.abs(commands[:, 2])

        step_count += 1
        if np.all(reached | fallen):
            break

        obs = torch.from_numpy(state.obs).to(device)

    ongoing_progress = current_start_dist - state.info["target_dist"]
    if completed_progress:
        progress_values = np.concatenate(
            [np.asarray(completed_progress, dtype=np.float32), ongoing_progress.astype(np.float32)]
        )
    else:
        progress_values = ongoing_progress

    reached_count = int(np.sum(reached))
    fall_count = int(np.sum(fallen))
    mean_target_dist = float(np.mean(state.info["target_dist"]))
    mean_progress = float(np.mean(progress_values))
    mean_reward = float(np.mean(reward_sum / max(step_count, 1)))
    mean_forward_vel = float(np.mean(forward_vel_sum / max(step_count, 1)))
    mean_progress_vel = float(np.mean(progress_vel_sum / max(step_count, 1)))
    mean_command_vx = float(np.mean(command_vx_sum / max(step_count, 1)))
    mean_abs_command_wz = float(np.mean(command_wz_abs_sum / max(step_count, 1)))

    return {
        "reached": reached_count,
        "fallen": fall_count,
        "done_count": done_count,
        "total": num_envs,
        "mean_target_dist": mean_target_dist,
        "mean_progress": mean_progress,
        "mean_start_dist": float(np.mean(start_dist)),
        "mean_forward_vel": mean_forward_vel,
        "mean_progress_vel": mean_progress_vel,
        "mean_command_vx": mean_command_vx,
        "mean_abs_command_wz": mean_abs_command_wz,
        "mean_reward": mean_reward,
        "steps": step_count,
    }


def main():
    args = parse_args()

    if not args.policy and not args.checkpoints:
        print("Error: --policy or --checkpoints is required")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = registry.make("k1-point-navigate", sim_backend="np", num_envs=args.num_envs)

    checkpoints = args.checkpoints if args.checkpoints else [args.policy]

    for ckpt in checkpoints:
        print(f"\n--- {ckpt} ---")
        policy = load_policy(ckpt, device)
        result = evaluate(env, policy, device, args.num_envs, args.max_steps)
        print(f"  Reached: {result['reached']}/{result['total']}")
        print(f"  Fallen: {result['fallen']}/{result['total']}  done events: {result['done_count']}")
        print(f"  Mean final dist: {result['mean_target_dist']:.3f} m  "
              f"progress: {result['mean_progress']:.3f} m  "
              f"start: {result['mean_start_dist']:.3f} m")
        print(f"  Mean fwd vel: {result['mean_forward_vel']:.4f} m/s  "
              f"reward: {result['mean_reward']:.4f}")
        print(f"  Mean progress vel: {result['mean_progress_vel']:.4f} m/s  "
              f"cmd vx: {result['mean_command_vx']:.4f} m/s  "
              f"|cmd wz|: {result['mean_abs_command_wz']:.4f} rad/s")
        print(f"  Steps: {result['steps']}/{args.max_steps}")


if __name__ == "__main__":
    main()
