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

"""Motrix native K1 BeyondMimic-style motion tracking environment."""

from pathlib import Path

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.k1.cfg import (
    K1_AMP_DEFAULT_JOINT_ANGLES,
    K1_AMP_JOINT_ORDER,
    K1_BEYOND_MIMIC_NUM_ACT,
    K1_BEYOND_MIMIC_NUM_OBS,
    K1BeyondMimicMjDance002EnvCfg,
)
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState


class K1BeyondMimicMotion:
    """Loads booster_train-style reference data from the current lightweight NPZ."""

    def __init__(self, path: str | Path, joint_order: list[str], ctrl_dt: float):
        self.path = Path(path).expanduser()
        self.joint_order = list(joint_order)
        self.ctrl_dt = float(ctrl_dt)
        (
            self.root_pos,
            self.root_quat_xyzw,
            self.root_lin_vel,
            self.root_ang_vel,
            self.joint_pos,
            self.joint_vel,
            self.fps,
        ) = self._load()
        self.num_frames = int(self.joint_pos.shape[0])

    def _load(self):
        if self.path.suffix.lower() != ".npz":
            raise ValueError(f"K1 BeyondMimic motion must be an NPZ file: {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            if "joint_names" in data:
                joint_names = [str(name) for name in data["joint_names"].tolist()]
                if joint_names != self.joint_order:
                    raise ValueError(
                        "K1 BeyondMimic motion joint_names do not match K1_AMP_JOINT_ORDER. "
                        f"motion={joint_names!r} cfg={self.joint_order!r}"
                    )
            root_pos = np.asarray(data["root_pos"], dtype=np.float32)
            root_quat_xyzw = self._normalize_quat(np.asarray(data["root_quat_xyzw"], dtype=np.float32))
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            fps = float(np.asarray(data["fps"]).reshape(()))
            if "root_lin_vel" in data:
                root_lin_vel = np.asarray(data["root_lin_vel"], dtype=np.float32)
            else:
                root_lin_vel = np.gradient(root_pos, 1.0 / fps, axis=0).astype(np.float32)
            if "root_ang_vel" in data:
                root_ang_vel = np.asarray(data["root_ang_vel"], dtype=np.float32)
            else:
                root_ang_vel = self._quat_finite_difference(root_quat_xyzw, 1.0 / fps)
        self._validate_arrays(root_pos, root_quat_xyzw, root_lin_vel, root_ang_vel, joint_pos, joint_vel)
        return root_pos, root_quat_xyzw, root_lin_vel, root_ang_vel, joint_pos, joint_vel, fps

    def _validate_arrays(
        self,
        root_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
        root_lin_vel: np.ndarray,
        root_ang_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> None:
        expected = {
            "root_pos": (3,),
            "root_quat_xyzw": (4,),
            "root_lin_vel": (3,),
            "root_ang_vel": (3,),
            "joint_pos": (len(self.joint_order),),
            "joint_vel": (len(self.joint_order),),
        }
        arrays = {
            "root_pos": root_pos,
            "root_quat_xyzw": root_quat_xyzw,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }
        frame_counts = set()
        for name, arr in arrays.items():
            if arr.ndim != 2 or arr.shape[1:] != expected[name]:
                raise ValueError(f"K1 BeyondMimic {name} must have shape (N, {expected[name][0]}); got {arr.shape}")
            frame_counts.add(arr.shape[0])
        if len(frame_counts) != 1:
            raise ValueError("K1 BeyondMimic motion arrays must have matching frame counts.")
        if joint_pos.shape[0] < 2:
            raise ValueError("K1 BeyondMimic motion needs at least two frames.")

    def sample(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        idx = np.mod(frame.astype(np.int64), self.num_frames)
        return {
            "root_pos": self.root_pos[idx],
            "root_quat_xyzw": self.root_quat_xyzw[idx],
            "root_lin_vel": self.root_lin_vel[idx],
            "root_ang_vel": self.root_ang_vel[idx],
            "joint_pos": self.joint_pos[idx],
            "joint_vel": self.joint_vel[idx],
        }

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        return (quat / np.clip(norm, 1.0e-8, None)).astype(np.float32)

    @staticmethod
    def _quat_finite_difference(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
        prev_q = np.roll(quat_xyzw, 1, axis=0)
        next_q = np.roll(quat_xyzw, -1, axis=0)
        prev_q[0] = quat_xyzw[0]
        next_q[-1] = quat_xyzw[-1]
        q_rel = quaternion.mul(next_q, quaternion.conjugate(prev_q))
        q_rel = K1BeyondMimicMotion._normalize_quat(q_rel)
        angle = 2.0 * np.arctan2(np.linalg.norm(q_rel[:, :3], axis=1), np.clip(q_rel[:, 3], -1.0, 1.0))
        axis = q_rel[:, :3] / np.clip(np.linalg.norm(q_rel[:, :3], axis=1, keepdims=True), 1.0e-8, None)
        return (axis * angle[:, None] / (2.0 * dt)).astype(np.float32)


@registry.env("k1-beyond-mimic-mj-dance-002", sim_backend="np")
@registry.env("k1-mj-dance-002", sim_backend="np")
class K1BeyondMimicTask(NpEnv):
    def __init__(self, cfg: K1BeyondMimicMjDance002EnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._init_action_space()
        self._init_obs_space()
        self._body = self._model.get_body(cfg.asset.body_name)
        self._num_action = self._action_space.shape[0]
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._joint_dof_pos_indices = self._get_actuated_dof_pos_indices()
        self._joint_dof_vel_indices = np.arange(self._num_dof_vel - self._num_action, self._num_dof_vel, dtype=np.int64)
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)
        self._init_buffer()

    def _init_obs_space(self):
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (K1_BEYOND_MIMIC_NUM_OBS,), dtype=np.float32)

    def _init_action_space(self):
        self._action_space = gym.spaces.Box(-1.0, 1.0, (K1_BEYOND_MIMIC_NUM_ACT,), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def _get_actuated_dof_pos_indices(self) -> np.ndarray:
        indices = np.asarray(self._body.get_dof_pos_indices(), dtype=np.int64).reshape(-1)
        if indices.size < self._num_action:
            return np.arange(7, 7 + self._num_action, dtype=np.int64)
        return indices[-self._num_action:]

    def _init_buffer(self):
        cfg = self.cfg
        if list(self._model.actuator_names) != K1_AMP_JOINT_ORDER:
            raise ValueError(
                "K1 BeyondMimic actuator order must match K1_AMP_JOINT_ORDER. "
                f"model={list(self._model.actuator_names)!r} cfg={K1_AMP_JOINT_ORDER!r}"
            )
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.kps = np.zeros(self._num_action, dtype=np.float32)
        self.kds = np.zeros(self._num_action, dtype=np.float32)
        self.action_scale = np.zeros(self._num_action, dtype=np.float32)
        self.torque_limits = np.zeros(self._num_action, dtype=np.float32)
        for i, name in enumerate(self._model.actuator_names):
            self.default_angles[i] = self._resolve_actuator_value(K1_AMP_DEFAULT_JOINT_ANGLES, name, "default_angle")
            self.kps[i] = self._resolve_actuator_value(cfg.control_config.stiffness, name, "stiffness")
            self.kds[i] = self._resolve_actuator_value(cfg.control_config.damping, name, "damping")
            self.action_scale[i] = self._resolve_actuator_value(cfg.control_config.action_scale, name, "action_scale")
            self.torque_limits[i] = self._resolve_actuator_value(cfg.control_config.torque_limit, name, "torque_limit")

        joint_limits = np.asarray(self._model.joint_limits, dtype=np.float32)
        if joint_limits.shape == (2, self._num_action):
            self.dof_pos_lower = joint_limits[0].copy()
            self.dof_pos_upper = joint_limits[1].copy()
        else:
            self.dof_pos_lower = np.full(self._num_action, -1.5, dtype=np.float32)
            self.dof_pos_upper = np.full(self._num_action, 1.5, dtype=np.float32)
        self._soft_dof_pos_lower, self._soft_dof_pos_upper = self._make_soft_dof_pos_limits()

        ground_geoms = self._validate_geom_indices("ground", cfg.asset.ground_geom_indices)
        collision_geoms = self._validate_geom_indices("collision", cfg.asset.collision_geom_indices, allow_empty=True)
        self.collision_contact_pairs = np.array(
            [[c, g] for c in collision_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))
        self.collision_check_num = (
            self.collision_contact_pairs.shape[0] if self.collision_contact_pairs.size > 0 else 0
        )

        self._motion_ref = K1BeyondMimicMotion(
            path=cfg.motion_reference.file,
            joint_order=K1_AMP_JOINT_ORDER,
            ctrl_dt=cfg.ctrl_dt,
        )

    def _validate_geom_indices(self, label: str, indices: list[int], allow_empty: bool = False) -> list[int]:
        if not indices:
            if allow_empty:
                return []
            raise ValueError(f"K1 {label} geom indices are empty; explicit contact cfg is required.")
        num_geoms = len(self._model.geom_names)
        resolved = []
        for idx in indices:
            if idx < 0 or idx >= num_geoms:
                raise ValueError(f"K1 {label} geom index {idx} is out of range [0, {num_geoms}).")
            resolved.append(int(idx))
        return resolved

    def _resolve_actuator_value(self, values, actuator_name: str, label: str) -> float:
        if isinstance(values, dict):
            if actuator_name not in values:
                raise KeyError(f"K1 BeyondMimic {label} is missing actuator '{actuator_name}'.")
            return float(values[actuator_name])
        return float(values)

    def _make_soft_dof_pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        limit_center = 0.5 * (self.dof_pos_lower + self.dof_pos_upper)
        limit_range = self.dof_pos_upper - self.dof_pos_lower
        soft_limit = self.cfg.reward_config.soft_dof_pos_limit
        return (
            (limit_center - 0.5 * limit_range * soft_limit).astype(np.float32),
            (limit_center + 0.5 * limit_range * soft_limit).astype(np.float32),
        )

    def _compute_torques(self, actions, data):
        target_pos = actions * self.action_scale + self.default_angles
        torques = self.kps * (target_pos - self.get_dof_pos(data)) - self.kds * self.get_dof_vel(data)
        return np.clip(torques, -self.torque_limits, self.torque_limits)

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        base_lin_world = data.dof_vel[:, :3]
        pose = self._body.get_pose(data)
        return quaternion.rotate_inverse(pose[:, 3:7], base_lin_world.astype(np.float32))

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        try:
            return self._model.get_sensor_value(self.cfg.sensor.gyro, data)
        except BaseException:
            pass
        base_ang_world = data.dof_vel[:, 3:6]
        pose = self._body.get_pose(data)
        return quaternion.rotate_inverse(pose[:, 3:7], base_ang_world.astype(np.float32))

    def apply_action(self, actions, state):
        if actions.ndim == 1:
            actions = np.tile(actions, (self._num_envs, 1))
        actions = np.clip(actions, self._action_space.low, self._action_space.high).astype(np.float32)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["episode_length"] = state.info.get(
            "episode_length", np.zeros((self._num_envs,), dtype=np.int32)
        ) + 1
        state.info["motion_frame"] = (state.info["motion_frame"] + 1) % self._motion_ref.num_frames
        self._maybe_push_robots(state)
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state

    def _maybe_push_robots(self, state: NpEnvState) -> None:
        cfg = self.cfg.domain_rand
        if not cfg.push_robots:
            return
        push_steps = max(int(cfg.push_interval_s / self.cfg.ctrl_dt), 1)
        push_mask = (state.info["episode_length"] % push_steps) == 0
        if not np.any(push_mask):
            return
        pushed = np.random.uniform(-cfg.max_push_vel_xy, cfg.max_push_vel_xy, size=(int(np.sum(push_mask)), 2)).astype(
            np.float32
        )
        dof_vel = state.data.dof_vel.copy()
        dof_vel[push_mask, :2] = pushed
        state.data.set_dof_vel(dof_vel)

    def _motion_sample(self, info: dict) -> dict[str, np.ndarray]:
        return self._motion_ref.sample(info["motion_frame"])

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        motion = self._motion_sample(info)
        pose = self._body.get_pose(data)
        q_rel = quaternion.mul(quaternion.inverse(pose[:, 3:7]), motion["root_quat_xyzw"])
        motion_anchor_ori_b = _quat_to_matrix_6d(q_rel)
        return np.concatenate(
            [
                motion["joint_pos"],
                motion["joint_vel"],
                motion_anchor_ori_b,
                self.get_gyro(data),
                self.get_dof_pos(data),
                self.get_dof_vel(data),
                info["current_actions"],
            ],
            axis=1,
        ).astype(np.float32)

    def get_privileged_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        motion = self._motion_sample(info)
        pose = self._body.get_pose(data)
        q_rel = quaternion.mul(quaternion.inverse(pose[:, 3:7]), motion["root_quat_xyzw"])
        rel_pos = quaternion.rotate_inverse(pose[:, 3:7], motion["root_pos"] - pose[:, :3])
        return np.concatenate(
            [
                motion["joint_pos"],
                motion["joint_vel"],
                rel_pos,
                _quat_to_matrix_6d(q_rel),
                self.get_local_linvel(data),
                self.get_gyro(data),
                self.get_dof_pos(data),
                self.get_dof_vel(data),
                info["current_actions"],
            ],
            axis=1,
        ).astype(np.float32)

    def update_observation(self, state: NpEnvState):
        if self.collision_check_num > 0:
            cquery = self._model.get_contact_query(state.data)
            colliding = cquery.is_colliding(self.collision_contact_pairs)
            state.info["undesired_contacts"] = colliding.reshape((self._num_envs, self.collision_check_num))
        return state.replace(obs=self._get_obs(state.data, state.info))

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        pose = self._body.get_pose(state.data)
        base_quat = pose[:, 3:7]
        gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        too_low = pose[:, 2] < self.cfg.reward_config.min_base_height
        too_tilted = np.linalg.norm(gravity[:, :2], axis=1) > self.cfg.reward_config.max_tilt_xy
        motion = self._motion_sample(state.info)
        anchor_pos_bad = (
            np.abs(pose[:, 2] - motion["root_pos"][:, 2]) > self.cfg.reward_config.anchor_max_height_error
        )
        anchor_ori_bad = (
            quaternion.rotation_distance(motion["root_quat_xyzw"], base_quat)
            > self.cfg.reward_config.anchor_max_ori_error
        )
        state.info["termination_too_low"] = too_low.astype(np.float32)
        state.info["termination_too_tilted"] = too_tilted.astype(np.float32)
        state.info["termination_anchor_pos"] = anchor_pos_bad.astype(np.float32)
        state.info["termination_anchor_ori"] = anchor_ori_bad.astype(np.float32)
        return state.replace(terminated=too_low | too_tilted | anchor_pos_bad | anchor_ori_bad)

    def update_state(self, state):
        state = self.update_observation(state)
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        data.reset(self._model)
        if self.cfg.motion_reference.random_start:
            motion_frame = np.random.randint(0, self._motion_ref.num_frames, size=(num_reset,), dtype=np.int32)
        else:
            motion_frame = np.zeros((num_reset,), dtype=np.int32)
        motion = self._motion_ref.sample(motion_frame)

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))
        dof_pos[:, :3] = motion["root_pos"]
        dof_pos[:, 3:7] = motion["root_quat_xyzw"]
        dof_pos[:, self._joint_dof_pos_indices] = motion["joint_pos"]
        dof_vel[:, :3] = motion["root_lin_vel"] * self.cfg.motion_reference.reset_root_velocity_scale
        dof_vel[:, 3:6] = motion["root_ang_vel"] * self.cfg.motion_reference.reset_root_velocity_scale
        joint_vel = np.clip(
            motion["joint_vel"],
            -self.cfg.motion_reference.max_reset_joint_vel,
            self.cfg.motion_reference.max_reset_joint_vel,
        )
        dof_vel[:, self._joint_dof_vel_indices] = joint_vel * self.cfg.motion_reference.reset_joint_velocity_scale
        self._apply_reset_perturbations(dof_pos, dof_vel, motion_frame)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "episode_length": np.zeros((num_reset,), dtype=np.int32),
            "motion_frame": motion_frame,
            "undesired_contacts": np.zeros((num_reset, self.collision_check_num), dtype=bool),
        }
        return self._get_obs(data, info), info

    def _apply_reset_perturbations(self, dof_pos: np.ndarray, dof_vel: np.ndarray, motion_frame: np.ndarray) -> None:
        del motion_frame
        cfg = self.cfg.motion_reference
        n = dof_pos.shape[0]
        dof_pos[:, 0:2] += np.random.uniform(-cfg.root_pos_perturb, cfg.root_pos_perturb, size=(n, 2)).astype(
            np.float32
        )
        dof_pos[:, 2] += np.random.uniform(-cfg.root_z_perturb, cfg.root_z_perturb, size=(n,)).astype(np.float32)
        roll = np.random.uniform(-cfg.root_rpy_perturb, cfg.root_rpy_perturb, size=(n,)).astype(np.float32)
        pitch = np.random.uniform(-cfg.root_rpy_perturb, cfg.root_rpy_perturb, size=(n,)).astype(np.float32)
        yaw = np.random.uniform(-cfg.root_yaw_perturb, cfg.root_yaw_perturb, size=(n,)).astype(np.float32)
        dof_pos[:, 3:7] = quaternion.mul(quaternion.from_euler(roll, pitch, yaw), dof_pos[:, 3:7])
        dof_pos[:, self._joint_dof_pos_indices] += np.random.uniform(
            -cfg.joint_pos_perturb,
            cfg.joint_pos_perturb,
            size=(n, self._num_action),
        ).astype(np.float32)
        dof_pos[:, self._joint_dof_pos_indices] = np.clip(
            dof_pos[:, self._joint_dof_pos_indices], self._soft_dof_pos_lower, self._soft_dof_pos_upper
        )
        dof_vel[:, :3] += np.random.uniform(-cfg.root_lin_vel_perturb, cfg.root_lin_vel_perturb, size=(n, 3)).astype(
            np.float32
        )
        dof_vel[:, 3:6] += np.random.uniform(-cfg.root_ang_vel_perturb, cfg.root_ang_vel_perturb, size=(n, 3)).astype(
            np.float32
        )

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        reward_dict = self._get_reward(state.data, state.info)
        scales = self.cfg.reward_config.scales
        rewards = {k: v * scales[k] * self.cfg.ctrl_dt for k, v in reward_dict.items() if k in scales}
        rwd = sum(rewards.values())
        if self.cfg.reward_config.only_positive_rewards:
            rwd = np.clip(rwd, 0.0, None)
        else:
            rwd = np.clip(rwd, -1000.0, 10000.0)
        if state.terminated is not None:
            rwd = np.where(state.terminated, 0.0, rwd)
        if "termination" in scales:
            done = state.terminated if state.terminated is not None else np.zeros(state.data.shape[0], dtype=bool)
            rwd += done.astype(np.float32) * scales["termination"] * self.cfg.ctrl_dt
        return state.replace(reward=rwd)

    def _get_reward(self, data: mtx.SceneData, info: dict) -> dict[str, np.ndarray]:
        motion = self._motion_sample(info)
        result = {
            "motion_global_anchor_pos": self._reward_motion_global_anchor_pos(data, motion),
            "motion_global_anchor_ori": self._reward_motion_global_anchor_ori(data, motion),
            "motion_joint_pos": self._reward_motion_joint_pos(data, motion),
            "motion_joint_vel": self._reward_motion_joint_vel(data, motion),
            "action_rate_l2": np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1),
            "joint_limit": self._reward_joint_limit(data),
            "base_height": self._reward_base_height(data),
            "orientation": self._reward_orientation(data),
            "alive": self._upright_gate(data),
        }
        if self.collision_check_num > 0:
            result["undesired_contacts"] = np.sum(info["undesired_contacts"], axis=1).astype(np.float32)
        return result

    def _exp_reward(self, error: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(-error / max(float(sigma) ** 2, 1.0e-8)).astype(np.float32)

    def _reward_motion_global_anchor_pos(self, data, motion: dict[str, np.ndarray]):
        pose = self._body.get_pose(data)
        error = np.sum(np.square(motion["root_pos"] - pose[:, :3]), axis=1)
        return self._exp_reward(error, self.cfg.reward_config.motion_anchor_pos_sigma)

    def _reward_motion_global_anchor_ori(self, data, motion: dict[str, np.ndarray]):
        pose = self._body.get_pose(data)
        error = np.square(quaternion.rotation_distance(motion["root_quat_xyzw"], pose[:, 3:7]))
        return self._exp_reward(error, self.cfg.reward_config.motion_anchor_ori_sigma)

    def _reward_motion_joint_pos(self, data, motion: dict[str, np.ndarray]):
        error = np.mean(np.square(motion["joint_pos"] - self.get_dof_pos(data)), axis=1)
        return self._exp_reward(error, self.cfg.reward_config.motion_joint_pos_sigma)

    def _reward_motion_joint_vel(self, data, motion: dict[str, np.ndarray]):
        error = np.mean(np.square(motion["joint_vel"] - self.get_dof_vel(data)), axis=1)
        return self._exp_reward(error, self.cfg.reward_config.motion_joint_vel_sigma)

    def _reward_joint_limit(self, data):
        dof_pos = self.get_dof_pos(data)
        lower_violation = np.clip(self._soft_dof_pos_lower - dof_pos, 0.0, None)
        upper_violation = np.clip(dof_pos - self._soft_dof_pos_upper, 0.0, None)
        return np.sum(lower_violation + upper_violation, axis=1)

    def _reward_base_height(self, data):
        pose = self._body.get_pose(data)
        return np.square(pose[:, 2] - self.cfg.reward_config.target_base_height)

    def _reward_orientation(self, data):
        pose = self._body.get_pose(data)
        gravity = quaternion.rotate_inverse(pose[:, 3:7], self.gravity_vec)
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _upright_gate(self, data):
        pose = self._body.get_pose(data)
        gravity = quaternion.rotate_inverse(pose[:, 3:7], self.gravity_vec)
        tilt_xy = np.linalg.norm(gravity[:, :2], axis=1)
        height_span = max(self.cfg.reward_config.target_base_height - self.cfg.reward_config.min_base_height, 1.0e-5)
        height_gate = np.clip((pose[:, 2] - self.cfg.reward_config.min_base_height) / height_span, 0.0, 1.0)
        tilt_gate = np.clip(
            (self.cfg.reward_config.max_tilt_xy - tilt_xy) / self.cfg.reward_config.max_tilt_xy,
            0.0,
            1.0,
        )
        return (height_gate * tilt_gate).astype(np.float32)


def _quat_to_matrix_6d(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    col0 = np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)], axis=1)
    col1 = np.stack([2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)], axis=1)
    return np.concatenate([col0, col1], axis=1).astype(np.float32)
