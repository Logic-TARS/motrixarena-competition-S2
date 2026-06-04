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

from motrix_envs import registry as env_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose K1 foot contact geometry and zero-action contact signals.")
    parser.add_argument("--env", default="k1-flat-terrain-walk", help="MotrixLab env name.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized envs to inspect.")
    parser.add_argument("--num-steps", type=int, default=300, help="Number of zero-action control steps.")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Steps to ignore before contact duty statistics.")
    parser.add_argument("--no-fail-on-zero-contact", action="store_true", help="Only report zero contact instead of failing.")
    return parser.parse_args()


def _fmt_array(values) -> str:
    return np.array2string(np.asarray(values), precision=3, suppress_small=True, separator=",")


def _print_geom_table(env) -> None:
    model = env._model
    print(f"num_geoms={len(model.geom_names)}")
    for idx, name in enumerate(model.geom_names):
        geom = model.get_geom(idx)
        geom_name = getattr(geom, "name", None)
        print(f"geom[{idx}] geom_names={name!r} geom.name={geom_name!r}")


def _print_contact_config(env) -> None:
    diag = getattr(env, "contact_geom_diagnostics", {})
    print("contact_geom_diagnostics:")
    for key in (
        "name_matched_ground_geoms",
        "name_matched_left_foot_geoms",
        "name_matched_right_foot_geoms",
        "name_matched_collision_geoms",
        "ground_geoms",
        "left_foot_geoms",
        "right_foot_geoms",
        "collision_geoms",
    ):
        print(f"  {key}={diag.get(key)}")
    print(f"left_foot_pairs={env.left_foot_pairs.tolist()}")
    print(f"right_foot_pairs={env.right_foot_pairs.tolist()}")
    print(f"foot_contact_pairs={env.foot_contact_pairs.tolist()}")
    print(f"collision_contact_pairs={env.collision_contact_pairs.tolist()}")
    print(f"action_scale={_fmt_array(getattr(env, 'action_scale', None))}")
    print(f"torque_limits={_fmt_array(getattr(env, 'torque_limits', None))}")


def main() -> None:
    args = parse_args()
    env = env_registry.make(args.env, sim_backend="np", num_envs=args.num_envs)
    _print_geom_table(env)
    _print_contact_config(env)

    state = env.init_state()
    print(f"reset_left_contact={_fmt_array(state.info.get('left_contact'))}")
    print(f"reset_right_contact={_fmt_array(state.info.get('right_contact'))}")
    print(f"reset_feet_pos={_fmt_array(state.info.get('feet_pos'))}")

    actions = np.zeros((args.num_envs, env.action_space.shape[0]), dtype=np.float32)
    left_samples = []
    right_samples = []
    done_count = 0
    too_low_count = 0
    too_tilted_count = 0
    torque_sat_samples = []

    torque_limit = getattr(env, "torque_limits", None)
    if torque_limit is None:
        torque_limit = getattr(env.cfg.control_config, "torque_limit", None)
    warmup_steps = min(max(args.warmup_steps, 0), args.num_steps)
    for step in range(args.num_steps):
        state = env.step(actions)
        if step >= warmup_steps:
            left_samples.append(state.info.get("left_contact", np.zeros((args.num_envs,), dtype=np.float32)).copy())
            right_samples.append(state.info.get("right_contact", np.zeros((args.num_envs,), dtype=np.float32)).copy())
            if torque_limit is not None and np.all(np.asarray(torque_limit) > 0):
                torque_sat_samples.append(np.mean(np.abs(state.data.actuator_ctrls) >= 0.95 * torque_limit, axis=1))
        done_count += int(np.sum(state.done))
        too_low_count += int(np.sum(state.info.get("termination_too_low", np.zeros((args.num_envs,), dtype=np.float32))))
        too_tilted_count += int(
            np.sum(state.info.get("termination_too_tilted", np.zeros((args.num_envs,), dtype=np.float32)))
        )

    left_contact = np.stack(left_samples) if left_samples else np.zeros((0, args.num_envs), dtype=np.float32)
    right_contact = np.stack(right_samples) if right_samples else np.zeros((0, args.num_envs), dtype=np.float32)
    left_duty = np.mean(left_contact, axis=0) if left_contact.size else np.zeros((args.num_envs,), dtype=np.float32)
    right_duty = np.mean(right_contact, axis=0) if right_contact.size else np.zeros((args.num_envs,), dtype=np.float32)
    torque_sat = (
        np.mean(np.stack(torque_sat_samples), axis=0)
        if torque_sat_samples
        else np.zeros((args.num_envs,), dtype=np.float32)
    )

    print(f"steps={args.num_steps} warmup_steps={warmup_steps}")
    print(f"left_contact_duty={_fmt_array(left_duty)}")
    print(f"right_contact_duty={_fmt_array(right_duty)}")
    print(f"done_count={done_count}")
    print(f"termination_too_low_count={too_low_count}")
    print(f"termination_too_tilted_count={too_tilted_count}")
    print(f"torque_saturation_ratio={_fmt_array(torque_sat)}")
    print(f"final_left_contact={_fmt_array(state.info.get('left_contact'))}")
    print(f"final_right_contact={_fmt_array(state.info.get('right_contact'))}")
    print(f"final_feet_pos={_fmt_array(state.info.get('feet_pos'))}")

    zero_contact = bool(np.any(left_duty <= 0.0) or np.any(right_duty <= 0.0))
    if zero_contact and not args.no_fail_on_zero_contact:
        raise SystemExit("K1 contact diagnostic failed: left or right contact duty is zero.")


if __name__ == "__main__":
    main()
