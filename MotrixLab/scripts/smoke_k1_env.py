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

import numpy as np

from motrix_envs import registry


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
    if obs_dim != 52 or act_dim != 12:
        raise AssertionError(f"K1 runtime expects 52->12, got {obs_dim}->{act_dim}")

    if state.obs.shape != (args.num_envs, obs_dim):
        raise AssertionError(f"Unexpected reset obs shape {state.obs.shape}")
    assert_finite("reset obs", state.obs)

    first_gait = state.info["gait_phase"].copy()
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

    if args.steps > 0 and not np.any(state.info["gait_phase"] != first_gait):
        raise AssertionError("gait_phase did not advance")

    print(
        f"OK {args.env}: obs={state.obs.shape}, action=({act_dim},), "
        f"reward_mean={float(np.mean(state.reward)):.6f}, gait_phase={state.info['gait_phase'][:4].tolist()}"
    )


if __name__ == "__main__":
    main()
