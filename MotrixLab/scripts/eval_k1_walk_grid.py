#!/usr/bin/env python3
"""Evaluate K1 locomotion checkpoints over the deployment command envelope."""

import argparse
import json
from pathlib import Path

from _source_path import ensure_source_path

ensure_source_path()

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry as env_registry
from motrix_envs.locomotion.k1.command_envelope import apply_forward_yaw_envelope
from motrix_rl import registry as rl_registry
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap


def load_policy(env, checkpoint: Path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vec = RslrlNpEnvWrap(env, device)
    cfg = rl_registry.default_rl_cfg("k1-flat-terrain-walk", "rslrl", backend="torch")
    runner = OnPolicyRunner(vec, cfg.runner.to_dict(), log_dir=None, device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})
    return vec, runner.get_inference_policy(device=device)


def command_grid() -> list[list[float]]:
    commands = []
    for vx in (0.0, 0.2, 0.4, 0.65, 0.8):
        for yaw in (-1.5, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5):
            cmd = apply_forward_yaw_envelope(
                np.array([[vx, 0.0, yaw]], dtype=np.float32),
                max_forward_speed=0.8,
            )[0]
            commands.append(cmd.tolist())
    for vy in (-0.2, -0.1, 0.1, 0.2):
        commands.append([0.25, vy, 0.0])
    return commands


def run_case(env, vec, policy, command: list[float], seconds: float) -> dict:
    obs, _ = vec.reset()
    fell = False
    lin_errors = []
    yaw_errors = []
    steps = int(seconds / env.cfg.ctrl_dt)
    for _ in range(steps):
        vec._state.info["commands"][:] = np.asarray(command, dtype=np.float32)
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = vec.step(actions)
        state = vec._state
        velocity = env.get_local_linvel(state.data)[0]
        gyro = env.get_gyro(state.data)[0]
        lin_errors.append(float(np.linalg.norm(velocity[:2] - command[:2])))
        yaw_errors.append(float(abs(gyro[2] - command[2])))
        if bool(dones[0].item()):
            fell = True
            break
    return {
        "command": command,
        "seconds": seconds,
        "fell": fell,
        "linear_rmse": float(np.sqrt(np.mean(np.square(lin_errors)))),
        "yaw_rmse": float(np.sqrt(np.mean(np.square(yaw_errors)))),
    }


def evaluate(checkpoint: Path, seconds: float, seed: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = env_registry.make("k1-flat-terrain-walk", sim_backend="np", num_envs=1)
    env.cfg.noise.add_noise = False
    env.cfg.domain_rand.push_robots = False
    vec, policy = load_policy(env, checkpoint)
    cases = [run_case(env, vec, policy, command, seconds) for command in command_grid()]
    accident_command = apply_forward_yaw_envelope(
        np.array([[0.65, 0.0, 1.465]], dtype=np.float32),
        max_forward_speed=0.8,
    )[0].tolist()
    accident = run_case(env, vec, policy, accident_command, 60.0)
    fall_rate = sum(case["fell"] for case in cases) / len(cases)
    failures = []
    if fall_rate > 0.01:
        failures.append("grid_fall_rate_above_0.01")
    if accident["fell"]:
        failures.append("accident_replay_fell_before_60s")
    return {
        "policy": str(checkpoint.resolve()),
        "seconds_per_case": seconds,
        "seed": seed,
        "cases": cases,
        "fall_rate": fall_rate,
        "accident_replay": accident,
        "eligible": not failures,
        "promotion_failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("walk_grid_eval.json"))
    args = parser.parse_args()
    result = evaluate(args.policy, args.seconds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "fall_rate": result["fall_rate"],
                "accident_fell": result["accident_replay"]["fell"],
                "eligible": result["eligible"],
                "cases": len(result["cases"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
