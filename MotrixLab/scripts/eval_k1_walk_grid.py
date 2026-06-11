#!/usr/bin/env python3
"""Evaluate one or more K1 checkpoints over a locomotion command grid."""

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


def _parse_policy(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label:
            raise argparse.ArgumentTypeError("Policy label must not be empty")
    else:
        raw_path = value
        label = Path(value).stem
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Policy not found: {path}")
    return label, path


def load_policy(env, checkpoint: Path, env_name: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vec = RslrlNpEnvWrap(env, device)
    try:
        scripted = torch.jit.load(str(checkpoint), map_location=device).eval()
    except (RuntimeError, ValueError):
        scripted = None
    if scripted is not None:
        return vec, lambda obs: scripted(obs["policy"]), "torchscript"

    cfg = rl_registry.default_rl_cfg(env_name, "rslrl", backend="torch")
    runner = OnPolicyRunner(vec, cfg.runner.to_dict(), log_dir=None, device=device)
    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
        },
    )
    return vec, runner.get_inference_policy(device=device), "rslrl_checkpoint"


def command_grid(apply_envelope: bool, max_forward_speed: float) -> list[dict]:
    specs: list[tuple[str, list[float], bool]] = []
    for vx in (0.0, 0.2, 0.4, 0.65, 0.8, 0.9, 1.0):
        for yaw in (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0):
            specs.append((f"grid_vx{vx:g}_w{yaw:g}", [vx, 0.0, yaw], False))
    for vy in (-0.2, -0.1, 0.1, 0.2):
        specs.append((f"lateral_vy{vy:g}", [0.25, vy, 0.0], False))
    for vx, yaw in ((0.8, -1.0), (0.8, 1.0), (1.0, -1.0), (1.0, 1.0)):
        specs.append((f"diagnostic_vx{vx:g}_w{yaw:g}", [vx, 0.0, yaw], True))

    cases = []
    for name, raw, diagnostic in specs:
        command = np.asarray([raw], dtype=np.float32)
        if apply_envelope:
            command = apply_forward_yaw_envelope(
                command,
                max_forward_speed=max_forward_speed,
            )
        cases.append(
            {
                "name": name,
                "raw_command": raw,
                "command": command[0].tolist(),
                "diagnostic": diagnostic,
            }
        )
    return cases


def run_case(env, vec, policy, case: dict, seconds: float) -> dict:
    command = case["command"]
    obs, _ = vec.reset()
    fell = False
    forward_velocities = []
    lateral_velocities = []
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
        forward_velocities.append(float(velocity[0]))
        lateral_velocities.append(float(velocity[1]))
        lin_errors.append(float(np.linalg.norm(velocity[:2] - command[:2])))
        yaw_errors.append(float(abs(gyro[2] - command[2])))
        if bool(dones[0].item()):
            fell = True
            break
    elapsed = len(forward_velocities) * env.cfg.ctrl_dt
    return {
        **case,
        "seconds": elapsed,
        "fell": fell,
        "mean_forward_velocity": float(np.mean(forward_velocities)),
        "mean_abs_lateral_velocity": float(np.mean(np.abs(lateral_velocities))),
        "lateral_drift": float(np.sum(lateral_velocities) * env.cfg.ctrl_dt),
        "linear_rmse": float(np.sqrt(np.mean(np.square(lin_errors)))),
        "yaw_rmse": float(np.sqrt(np.mean(np.square(yaw_errors)))),
    }


def evaluate(
    checkpoint: Path,
    label: str,
    seconds: float,
    seed: int,
    env_name: str,
    apply_envelope: bool,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = env_registry.make(env_name, sim_backend="np", num_envs=1)
    env.cfg.noise.add_noise = False
    env.cfg.domain_rand.push_robots = False
    vec, policy, policy_format = load_policy(env, checkpoint, env_name)
    max_forward_speed = float(env.cfg.commands.lin_vel_x[1])
    cases = [
        run_case(env, vec, policy, case, seconds)
        for case in command_grid(apply_envelope, max_forward_speed)
    ]
    promotion_cases = [case for case in cases if not case["diagnostic"]]
    fall_rate = sum(case["fell"] for case in promotion_cases) / len(promotion_cases)
    failures = []
    if fall_rate > 0.01:
        failures.append("grid_fall_rate_above_0.01")
    return {
        "label": label,
        "policy": str(checkpoint.resolve()),
        "policy_format": policy_format,
        "env": env_name,
        "apply_envelope": apply_envelope,
        "seconds_per_case": seconds,
        "seed": seed,
        "cases": cases,
        "fall_rate": fall_rate,
        "eligible": not failures,
        "promotion_failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        action="append",
        type=_parse_policy,
        required=True,
        metavar="[LABEL=]PATH",
    )
    parser.add_argument("--env", default="k1-flat-terrain-walk")
    parser.add_argument(
        "--apply-envelope",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("walk_grid_eval.json"))
    args = parser.parse_args()
    apply_envelope = (
        args.env != "k1-flat-terrain-walk-speed"
        if args.apply_envelope is None
        else args.apply_envelope
    )
    evaluations = [
        evaluate(path, label, args.seconds, args.seed, args.env, apply_envelope)
        for label, path in args.policy
    ]
    failures = [
        f"{item['label']}:{failure}"
        for item in evaluations
        for failure in item["promotion_failures"]
    ]
    result = {
        "env": args.env,
        "apply_envelope": apply_envelope,
        "evaluations": evaluations,
        "eligible": not failures,
        "promotion_failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "env": args.env,
                "apply_envelope": apply_envelope,
                "eligible": result["eligible"],
                "policies": [
                    {
                        "label": item["label"],
                        "fall_rate": item["fall_rate"],
                        "cases": len(item["cases"]),
                    }
                    for item in evaluations
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
