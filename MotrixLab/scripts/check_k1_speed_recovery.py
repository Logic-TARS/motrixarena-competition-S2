#!/usr/bin/env python3
"""Validate the speed-recovery environment and warm-start checkpoint."""

import argparse
from pathlib import Path

from _source_path import ensure_source_path

ensure_source_path()

import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry as env_registry
from motrix_rl import registry as rl_registry
from motrix_rl.rslrl.torch.train.ppo import Trainer
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap


def check(checkpoint: Path, env_name: str) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cpu")
    env = env_registry.make(env_name, sim_backend="np", num_envs=1)
    vec_env = RslrlNpEnvWrap(env, device)
    rlcfg = rl_registry.default_rl_cfg(env_name, "rslrl", backend="torch")
    runner = OnPolicyRunner(
        vec_env,
        rlcfg.runner.to_dict(),
        log_dir=None,
        device=device,
    )
    trainer = Trainer(env_name, resume_policy=str(checkpoint))
    trainer._load_resume_policy(runner, str(device))

    if vec_env.num_obs != 47 or vec_env.num_actions != 12:
        raise RuntimeError(
            f"Expected 47->12 policy interface, got "
            f"{vec_env.num_obs}->{vec_env.num_actions}"
        )
    print(
        f"Preflight OK: env={env_name} checkpoint={checkpoint} "
        f"interface={vec_env.num_obs}->{vec_env.num_actions}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--env", default="k1-flat-terrain-walk-speed")
    args = parser.parse_args()
    check(args.checkpoint, args.env)


if __name__ == "__main__":
    main()
