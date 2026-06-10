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

import argparse

from _source_path import ensure_source_path

ensure_source_path()

import numpy as np  # noqa: E402

from motrix_envs import registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the MotrixLab K1 locomotion environment.")
    parser.add_argument("--env", default="k1-flat-terrain-walk", help="Environment name.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of vectorized environments.")
    parser.add_argument("--steps", type=int, default=8, help="Number of random-control steps to execute.")
    parser.add_argument("--zero-action", action="store_true", help="Use zero actions instead of sampled actions.")
    return parser.parse_args()


def assert_finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        bad = np.argwhere(~np.isfinite(value))
        raise AssertionError(f"{name} contains non-finite values at {bad[:8].tolist()}")


def main() -> None:
    args = parse_args()
    env = registry.make(args.env, sim_backend="np", num_envs=args.num_envs)
    state = env.init_state()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    expected_by_env = {
        "k1-flat-terrain-walk": (47, 12),
        "k1-getup": (78, 22),
    }
    expected_obs, expected_act = expected_by_env.get(args.env, (obs_dim, act_dim))
    if obs_dim != expected_obs or act_dim != expected_act:
        raise AssertionError(f"Expected {expected_obs}->{expected_act}, got {obs_dim}->{act_dim}")

    if state.obs.shape != (args.num_envs, obs_dim):
        raise AssertionError(f"Unexpected reset obs shape {state.obs.shape}")
    assert_finite("reset obs", state.obs)
    if args.env == "k1-getup":
        pose_one_hot = state.obs[:, 9:12]
        if not np.all(np.isin(pose_one_hot, [0.0, 1.0])):
            raise AssertionError("Get-up pose class must be one-hot encoded")
        if not np.allclose(np.sum(pose_one_hot, axis=1), 1.0):
            raise AssertionError("Get-up pose class must contain exactly one active category")

    if "gait_phase" in state.info:
        progress_key = "gait_phase"
    elif "episode_length" in state.info:
        progress_key = "episode_length"
    else:
        progress_key = "motion_frame"
    first_progress = state.info[progress_key].copy()
    for step_idx in range(args.steps):
        if args.zero_action:
            actions = np.zeros((args.num_envs, act_dim), dtype=np.float32)
        else:
            actions = np.stack([env.action_space.sample() for _ in range(args.num_envs)]).astype(np.float32)
        state = env.step(actions)
        if state.obs.shape != (args.num_envs, obs_dim):
            raise AssertionError(f"Unexpected step obs shape {state.obs.shape} at step {step_idx}")
        if state.reward.shape != (args.num_envs,):
            raise AssertionError(f"Unexpected reward shape {state.reward.shape} at step {step_idx}")
        assert_finite("step obs", state.obs)
        assert_finite("reward", state.reward)

    if args.steps > 0 and not np.any(state.info[progress_key] != first_progress):
        raise AssertionError(f"{progress_key} did not advance")

    print(
        f"OK {args.env}: obs={state.obs.shape}, action=({act_dim},), "
        f"reward_mean={float(np.mean(state.reward)):.6f}, {progress_key}={state.info[progress_key][:4].tolist()}"
    )


if __name__ == "__main__":
    main()
