#!/usr/bin/env python3
"""Evaluate a K1 get-up checkpoint over randomized fallen poses."""

import argparse
import json
from pathlib import Path

from _source_path import ensure_source_path

ensure_source_path()

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry as env_registry
from motrix_rl import registry as rl_registry
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap


POSE_NAMES = ["supine", "prone", "left", "right"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("getup_eval.json"))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = env_registry.make("k1-getup", sim_backend="np", num_envs=args.episodes)
    env._reset_count = env.cfg.reset_config.curriculum_resets_per_stage * 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vec = RslrlNpEnvWrap(env, device)
    cfg = rl_registry.default_rl_cfg("k1-getup", "rslrl", backend="torch")
    runner = OnPolicyRunner(vec, cfg.runner.to_dict(), log_dir=None, device=device)
    runner.load(str(args.policy), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})
    policy = runner.get_inference_policy(device=device)

    initial_pose = vec._state.info["fall_pose"].copy()
    completed = np.zeros((args.episodes,), dtype=bool)
    success = np.zeros((args.episodes,), dtype=bool)
    completion_time = np.full((args.episodes,), np.nan, dtype=np.float32)
    obs = vec.get_observations()
    for step in range(env.cfg.max_episode_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = vec.step(actions)
        done = dones.cpu().numpy().astype(bool) & ~completed
        if np.any(done):
            success[done] = vec._state.terminated[done]
            completion_time[done] = (step + 1) * env.cfg.ctrl_dt
            completed[done] = True
        if np.all(completed):
            break

    by_pose = {}
    for pose_id, name in enumerate(POSE_NAMES):
        mask = initial_pose == pose_id
        times = completion_time[mask & success]
        by_pose[name] = {
            "attempts": int(mask.sum()),
            "successes": int(success[mask].sum()),
            "success_rate": float(success[mask].mean()) if mask.any() else None,
            "median_seconds": float(np.nanmedian(times)) if times.size else None,
            "p95_seconds": float(np.nanpercentile(times, 95)) if times.size else None,
        }
    successful_times = completion_time[success]
    overall_rate = float(success.mean())
    class_rates = [
        values["success_rate"]
        for values in by_pose.values()
        if values["success_rate"] is not None
    ]
    p95_seconds = (
        float(np.nanpercentile(successful_times, 95))
        if successful_times.size
        else None
    )
    max_seconds = (
        float(np.nanmax(successful_times))
        if successful_times.size
        else None
    )
    failures = []
    if overall_rate < 0.95:
        failures.append("overall_success_rate_below_0.95")
    if len(class_rates) != len(POSE_NAMES) or min(class_rates, default=0.0) < 0.90:
        failures.append("pose_success_rate_below_0.90")
    if p95_seconds is None or p95_seconds >= 12.0:
        failures.append("p95_getup_time_not_below_12s")
    if max_seconds is None or max_seconds >= 20.0:
        failures.append("successful_case_reached_20s")
    result = {
        "policy": str(args.policy.resolve()),
        "attempts": args.episodes,
        "seed": args.seed,
        "successes": int(success.sum()),
        "success_rate": overall_rate,
        "median_seconds": float(np.nanmedian(successful_times)) if successful_times.size else None,
        "p95_seconds": p95_seconds,
        "max_success_seconds": max_seconds,
        "by_pose": by_pose,
        "eligible": not failures,
        "promotion_failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
