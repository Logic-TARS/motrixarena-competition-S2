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

"""K1 AMP (Adversarial Motion Prior) walk environment — Scheme A: 375→22."""

import logging
from pathlib import Path

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.k1.cfg import (
    K1_AMP_DEFAULT_JOINT_ANGLES,
    K1_AMP_FRAME_STACK,
    K1_AMP_JOINT_ORDER,
    K1_AMP_NUM_SINGLE_OBS,
    K1_AMP_UPPER_BODY_JOINTS,
    K1AmpWalkEnvCfg,
)
from motrix_envs.locomotion.k1.motion_reference import K1AmpMotionReference
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

logger = logging.getLogger(__name__)


@registry.env("k1-amp-walk", sim_backend="np")
@registry.env("k1-amp-stand", sim_backend="np")
@registry.env("k1-amp-walk-small", sim_backend="np")
@registry.env("k1-amp-walk-lift", sim_backend="np")
class K1AmpWalkTask(NpEnv):
    def __init__(self, cfg: K1AmpWalkEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._init_action_space()
        self._init_obs_space()
        self._body = self._model.get_body(cfg.asset.body_name)
        self._num_action = self._action_space.shape[0]
        self._num_observation = self._observation_space.shape[0]
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._joint_dof_pos_indices = self._get_actuated_dof_pos_indices()
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_buffer()

    def _init_obs_space(self):
        num_obs = K1_AMP_NUM_SINGLE_OBS * K1_AMP_FRAME_STACK
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        self._action_space = gym.spaces.Box(-1.0, 1.0, (self._model.num_actuators,), dtype=np.float32)

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
        cfg = self._cfg

        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel_x, cfg.normalization.lin_vel_y, cfg.normalization.ang_vel],
            dtype=np.float32,
        )
        self.noise_scale_vec = self._get_noise_scale_vec()

        actuator_names = list(self._model.actuator_names)
        if actuator_names != K1_AMP_JOINT_ORDER:
            raise ValueError(
                "K1 AMP actuator order must match K1_AMP_JOINT_ORDER. "
                f"model={actuator_names!r} cfg={K1_AMP_JOINT_ORDER!r}"
            )
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.kps = np.zeros(self._num_action, dtype=np.float32)
        self.kds = np.zeros(self._num_action, dtype=np.float32)
        self.action_scale = np.zeros(self._num_action, dtype=np.float32)
        self.torque_limits = np.zeros(self._num_action, dtype=np.float32)
        for i, name in enumerate(actuator_names):
            self.default_angles[i] = self._resolve_actuator_value(
                K1_AMP_DEFAULT_JOINT_ANGLES, name, "default_angle"
            )
            self.kps[i] = self._resolve_actuator_value(cfg.control_config.stiffness, name, "stiffness")
            self.kds[i] = self._resolve_actuator_value(cfg.control_config.damping, name, "damping")
            self.action_scale[i] = self._resolve_actuator_value(
                cfg.control_config.action_scale, name, "action_scale"
            )
            self.torque_limits[i] = self._resolve_actuator_value(
                cfg.control_config.torque_limit, name, "torque_limit"
            )

        self._upper_body_action_indices = np.asarray(
            [i for i, name in enumerate(actuator_names) if name in K1_AMP_UPPER_BODY_JOINTS],
            dtype=np.int64,
        )
        self._leg_action_indices = np.asarray(
            [i for i, name in enumerate(actuator_names) if name not in K1_AMP_UPPER_BODY_JOINTS],
            dtype=np.int64,
        )
        self._joint_dof_vel_indices = np.arange(
            self._num_dof_vel - self._num_action,
            self._num_dof_vel,
            dtype=np.int64,
        )
        self._motion_ref = self._load_motion_reference()

        self._init_dof_pos[:3] = np.asarray(cfg.init_state.pos, dtype=np.float32)
        self._init_dof_pos[self._joint_dof_pos_indices] = self.default_angles

        joint_limits = np.asarray(self._model.joint_limits, dtype=np.float32)
        if joint_limits.shape == (2, self._num_action):
            self.dof_pos_lower = joint_limits[0].copy()
            self.dof_pos_upper = joint_limits[1].copy()
        else:
            self.dof_pos_lower = np.full(self._num_action, -1.5, dtype=np.float32)
            self.dof_pos_upper = np.full(self._num_action, 1.5, dtype=np.float32)
        self._soft_dof_pos_lower, self._soft_dof_pos_upper = self._make_soft_dof_pos_limits()

        # Foot contact pairs using explicit geom indices
        ground_geoms = self._validate_geom_indices("ground", cfg.asset.ground_geom_indices)
        left_foot_geoms = self._validate_geom_indices("left foot", cfg.asset.left_foot_geom_indices)
        right_foot_geoms = self._validate_geom_indices("right foot", cfg.asset.right_foot_geom_indices)
        collision_geoms = self._validate_geom_indices("collision", cfg.asset.collision_geom_indices, allow_empty=True)

        self._left_foot_geoms = [self._model.get_geom(idx) for idx in left_foot_geoms]
        self._right_foot_geoms = [self._model.get_geom(idx) for idx in right_foot_geoms]

        foot_geoms = left_foot_geoms + right_foot_geoms
        self.foot_contact_pairs = np.array(
            [[f, g] for f in foot_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))
        self.foot_check_num = self.foot_contact_pairs.shape[0] if self.foot_contact_pairs.size > 0 else 0

        self.left_foot_pairs = np.array(
            [[f, g] for f in left_foot_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))
        self.right_foot_pairs = np.array(
            [[f, g] for f in right_foot_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))

        self.collision_contact_pairs = np.array(
            [[c, g] for c in collision_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))
        self.collision_check_num = (
            self.collision_contact_pairs.shape[0] if self.collision_contact_pairs.size > 0 else 0
        )

        self._hip_indices = self._find_hip_indices()

    def _load_motion_reference(self) -> K1AmpMotionReference | None:
        motion_cfg = self.cfg.motion_reference
        if not motion_cfg.enabled or not motion_cfg.file:
            return None
        path = Path(motion_cfg.file).expanduser()
        if not path.exists():
            logger.warning("K1 AMP motion reference disabled; file not found: %s", path)
            return None
        return K1AmpMotionReference(
            path=path,
            joint_order=K1_AMP_JOINT_ORDER,
            input_fps=motion_cfg.input_fps,
            ctrl_dt=self.cfg.ctrl_dt,
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
                raise KeyError(f"K1 AMP {label} is missing actuator '{actuator_name}'.")
            return float(values[actuator_name])
        return float(values)

    def _find_hip_indices(self) -> np.ndarray:
        hip_names = ("Hip_Roll", "Hip_Yaw")
        indices = []
        for i, name in enumerate(self._model.actuator_names):
            if any(hip_name in name for hip_name in hip_names):
                indices.append(i)
        return np.array(indices, dtype=np.int64) if indices else np.array([], dtype=np.int64)

    def _make_soft_dof_pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        limit_center = 0.5 * (self.dof_pos_lower + self.dof_pos_upper)
        limit_range = self.dof_pos_upper - self.dof_pos_lower
        soft_limit = self.cfg.reward_config.soft_dof_pos_limit
        lower = limit_center - 0.5 * limit_range * soft_limit
        upper = limit_center + 0.5 * limit_range * soft_limit
        return lower.astype(np.float32), upper.astype(np.float32)

    def _get_noise_scale_vec(self) -> np.ndarray:
        single = K1_AMP_NUM_SINGLE_OBS
        noise_vec = np.zeros((single,), dtype=np.float32)
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.cfg.normalization.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[9:31] = noise_scales.dof_pos * noise_level * self.cfg.normalization.dof_pos
        noise_vec[31:53] = noise_scales.dof_vel * noise_level * self.cfg.normalization.dof_vel
        return noise_vec

    def _mask_actions(self, actions: np.ndarray) -> np.ndarray:
        masked = actions.copy()
        if self._upper_body_action_indices.size > 0:
            masked[:, self._upper_body_action_indices] = 0.0
        return masked

    def _compute_torques(self, actions, data):
        target_pos = actions * self.action_scale + self.default_angles
        torques = self.kps * (target_pos - self.get_dof_pos(data)) - self.kds * self.get_dof_vel(data)
        if self._upper_body_action_indices.size > 0:
            torques[:, self._upper_body_action_indices] = 0.0
        return np.clip(torques, -self.torque_limits, self.torque_limits)

    def _lock_upper_body(self, data: mtx.SceneData) -> None:
        if self._upper_body_action_indices.size == 0:
            return
        dof_pos = data.dof_pos.copy()
        dof_vel = data.dof_vel.copy()
        upper_pos_indices = self._joint_dof_pos_indices[self._upper_body_action_indices]
        upper_vel_indices = self._joint_dof_vel_indices[self._upper_body_action_indices]
        dof_pos[:, upper_pos_indices] = self.default_angles[self._upper_body_action_indices]
        dof_vel[:, upper_vel_indices] = 0.0
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        base_lin_world = data.dof_vel[:, :3]
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        return quaternion.rotate_inverse(base_quat, base_lin_world.astype(np.float32))

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        try:
            return self._model.get_sensor_value(self.cfg.sensor.gyro, data)
        except BaseException:
            pass
        base_ang_world = data.dof_vel[:, 3:6]
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        return quaternion.rotate_inverse(base_quat, base_ang_world.astype(np.float32))

    def get_privileged_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        linvel = self.get_local_linvel(data) * 2.0
        actor_obs = self._get_obs(data, info, add_noise=False)
        return np.concatenate([linvel, actor_obs], axis=1).astype(np.float32)

    def apply_action(self, actions, state):
        if actions.ndim == 1:
            actions = np.tile(actions, (self._num_envs, 1))
        actions = np.clip(actions, self._action_space.low, self._action_space.high).astype(np.float32)
        actions = self._mask_actions(actions)
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.info["episode_length"] = state.info.get(
            "episode_length", np.zeros((self._num_envs,), dtype=np.int32)
        ) + 1
        state.info["gait_phase"] = np.fmod(
            state.info["episode_length"] * self.cfg.ctrl_dt / self.cfg.commands.phase_period,
            1.0,
        ).astype(np.float32)
        self._maybe_resample_commands(state.info)
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state

    def _maybe_resample_commands(self, info: dict) -> None:
        resampling_steps = max(int(self.cfg.commands.resampling_time / self.cfg.ctrl_dt), 1)
        resample_mask = (info["episode_length"] % resampling_steps) == 0
        if np.any(resample_mask):
            info["commands"][resample_mask] = self.resample_commands(int(np.sum(resample_mask)))

    def _maybe_push_robots(self, state: NpEnvState) -> None:
        cfg = self.cfg.domain_rand
        if not cfg.push_robots:
            return
        push_steps = max(int(cfg.push_interval_s / self.cfg.ctrl_dt), 1)
        push_mask = (state.info["episode_length"] % push_steps) == 0
        if not np.any(push_mask):
            return
        pushed = np.random.uniform(
            -cfg.max_push_vel_xy,
            cfg.max_push_vel_xy,
            size=(int(np.sum(push_mask)), 2),
        ).astype(np.float32)
        dof_vel = state.data.dof_vel.copy()
        dof_vel[push_mask, :2] = pushed
        state.data.set_dof_vel(dof_vel)

    def _get_obs(self, data: mtx.SceneData, info: dict, add_noise: bool = True) -> np.ndarray:
        gyro = self.get_gyro(data)
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        local_gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)

        diff = self.get_dof_pos(data) - self.default_angles
        dof_vel = self.get_dof_vel(data)

        single_dim = K1_AMP_NUM_SINGLE_OBS
        batch = data.shape[0]
        single = np.zeros((batch, single_dim), dtype=np.float32)

        # [0:3]  = scaled velocity command  [vx*0.5, vy*0.4, wz*0.25]
        single[:, 0:3] = info["commands"] * self.commands_scale
        # [3:6]  = gravity vector in body frame
        single[:, 3:6] = local_gravity
        # [6:9]  = gyro * ang_vel_norm
        single[:, 6:9] = gyro * self.cfg.normalization.ang_vel
        # [9:31]  = joint position error * dof_pos_norm
        single[:, 9:31] = diff * self.cfg.normalization.dof_pos
        # [31:53] = joint velocity * dof_vel_norm
        single[:, 31:53] = dof_vel * self.cfg.normalization.dof_vel
        # [53:75] = last action (already in [-1, 1])
        single[:, 53:75] = info["current_actions"]

        if add_noise and self.cfg.noise.add_noise:
            single += (2.0 * np.random.rand(*single.shape).astype(np.float32) - 1.0) * self.noise_scale_vec

        # Frame stack: shift history buffer, append new frame
        hist = info["obs_history"]  # shape: (num_envs, FRAME_STACK, SINGLE_OBS)
        hist[:, :-1, :] = hist[:, 1:, :]
        hist[:, -1, :] = single
        return hist.reshape(batch, -1)

    def update_observation(self, state: NpEnvState):
        if self.foot_check_num > 0:
            cquery = self._model.get_contact_query(state.data)
            foot_contact = cquery.is_colliding(self.foot_contact_pairs)
            state.info["contacts"] = foot_contact.reshape((self._num_envs, self.foot_check_num))
            state.info["feet_air_time"] = self._update_feet_air_time(state.info)
            if self.left_foot_pairs.size > 0:
                left_contact = cquery.is_colliding(self.left_foot_pairs)
                state.info["left_contact"] = left_contact.any(axis=1).astype(np.float32)
            if self.right_foot_pairs.size > 0:
                right_contact = cquery.is_colliding(self.right_foot_pairs)
                state.info["right_contact"] = right_contact.any(axis=1).astype(np.float32)
        state.info["feet_pos"] = self._get_foot_positions(state.data)
        state.info["feet_vel"] = self._get_foot_velocities(state.data)
        state = state.replace(obs=self._get_obs(state.data, state.info))
        return state

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        pose = self._body.get_pose(state.data)
        base_quat = pose[:, 3:7]
        gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        too_low = pose[:, 2] < self.cfg.reward_config.min_base_height
        too_tilted = np.linalg.norm(gravity[:, :2], axis=1) > self.cfg.reward_config.max_tilt_xy
        state.info["termination_too_low"] = too_low.astype(np.float32)
        state.info["termination_too_tilted"] = too_tilted.astype(np.float32)
        return state.replace(terminated=too_low | too_tilted)

    def resample_commands(self, num_envs: int):
        commands = np.zeros((num_envs, 3), dtype=np.float32)
        commands[:, 0] = np.random.uniform(*self.cfg.commands.lin_vel_x, size=num_envs)
        commands[:, 1] = np.random.uniform(*self.cfg.commands.lin_vel_y, size=num_envs)
        commands[:, 2] = np.random.uniform(*self.cfg.commands.ang_vel_yaw, size=num_envs)
        moving = np.linalg.norm(commands[:, :2], axis=1) > self.cfg.commands.command_deadzone
        commands[:, :2] *= moving[:, None]
        return commands.astype(np.float32)

    def update_state(self, state):
        self._maybe_push_robots(state)
        self._lock_upper_body(state.data)
        state = self.update_observation(state)
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

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
            rwd += self._reward_termination(done) * scales["termination"] * self.cfg.ctrl_dt
        return state.replace(reward=rwd)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        data.reset(self._model)

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))

        if self._leg_action_indices.size > 0:
            perturb = np.zeros((num_reset, self._num_action), dtype=np.float32)
            perturb[:, self._leg_action_indices] = 0.01 * np.random.randn(
                num_reset,
                self._leg_action_indices.size,
            )
            dof_pos[:, self._joint_dof_pos_indices] += perturb

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        single_dim = K1_AMP_NUM_SINGLE_OBS
        obs_history = np.zeros((num_reset, K1_AMP_FRAME_STACK, single_dim), dtype=np.float32)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": self.resample_commands(num_reset),
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "gait_phase": np.zeros((num_reset,), dtype=np.float32),
            "episode_length": np.zeros((num_reset,), dtype=np.int32),
            "feet_air_time": np.zeros((num_reset, self.foot_check_num), dtype=np.float32),
            "contacts": np.zeros((num_reset, self.foot_check_num), dtype=bool),
            "left_contact": np.zeros((num_reset,), dtype=np.float32),
            "right_contact": np.zeros((num_reset,), dtype=np.float32),
            "feet_pos": self._get_foot_positions(data),
            "feet_vel": self._get_foot_velocities(data),
            "obs_history": obs_history,
        }
        return self._get_obs(data, info), info

    def _get_reward(self, data: mtx.SceneData, info: dict) -> dict[str, np.ndarray]:
        commands = info["commands"]
        motion_sample = self._motion_ref.sample(info["episode_length"]) if self._motion_ref is not None else None
        result = {
            "alive": self._reward_alive(data),
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, commands),
            "command_forward_vel": self._reward_command_forward_vel(data, commands),
            "overspeed": self._reward_overspeed(data, commands),
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, commands),
            "stand_still": self._reward_stand_still(data),
            "lin_vel_z": self._reward_lin_vel_z(data),
            "ang_vel_xy": self._reward_ang_vel_xy(data),
            "orientation": self._reward_orientation(data),
            "base_height": self._reward_base_height(data),
            "torques": self._reward_torques(data),
            "dof_vel": self._reward_dof_vel(data),
            "dof_acc": self._reward_dof_acc(data, info),
            "action_rate": self._reward_action_rate(info),
            "dof_pos_limits": self._reward_dof_pos_limits(data),
            "hip_pos": self._reward_hip_pos(data),
            "joint_regularization": self._reward_joint_regularization(data),
            "upper_body_regularization": self._reward_upper_body_regularization(data),
            "upper_body_velocity": self._reward_upper_body_velocity(data),
            "contact_no_vel": self._reward_contact_no_vel(info),
            "feet_swing_height": self._reward_feet_swing_height(info),
            "contact": self._reward_contact(info),
        }
        if motion_sample is not None:
            result["motion_leg_joint_pos"] = self._reward_motion_leg_joint_pos(data, motion_sample)
            result["motion_leg_joint_vel"] = self._reward_motion_leg_joint_vel(data, motion_sample)
            result["motion_base_height"] = self._reward_motion_base_height(data, motion_sample)
        if self.foot_check_num > 0:
            result["feet_air_time"] = self._reward_feet_air_time(commands, info)
        if self.collision_check_num > 0:
            result["collision"] = self._reward_collision(data)
        return result

    # --- Reward functions ---
    def _reward_lin_vel_z(self, data):
        return np.square(self.get_local_linvel(data)[:, 2])

    def _reward_ang_vel_xy(self, data):
        return np.sum(np.square(self.get_gyro(data)[:, :2]), axis=1)

    def _reward_orientation(self, data):
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _reward_base_height(self, data):
        pose = self._body.get_pose(data)
        height_error = pose[:, 2] - self.cfg.reward_config.target_base_height
        return np.square(height_error)

    def _reward_torques(self, data: mtx.SceneData):
        return np.sum(np.square(data.actuator_ctrls), axis=1)

    def _reward_dof_vel(self, data):
        return np.sum(np.square(self.get_dof_vel(data)), axis=1)

    def _reward_dof_acc(self, data, info: dict):
        return np.sum(
            np.square((info["last_dof_vel"] - self.get_dof_vel(data)) / self.cfg.ctrl_dt),
            axis=1,
        )

    def _reward_action_rate(self, info: dict):
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_termination(self, done):
        return done.astype(np.float32)

    def _reward_alive(self, data):
        return self._upright_gate(data)

    def _upright_gate(self, data):
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        tilt_xy = np.linalg.norm(gravity[:, :2], axis=1)
        height_span = max(
            self.cfg.reward_config.target_base_height - self.cfg.reward_config.min_base_height,
            1.0e-5,
        )
        height_gate = np.clip((pose[:, 2] - self.cfg.reward_config.min_base_height) / height_span, 0.0, 1.0)
        tilt_gate = np.clip(
            (self.cfg.reward_config.max_tilt_xy - tilt_xy) / self.cfg.reward_config.max_tilt_xy,
            0.0,
            1.0,
        )
        return (height_gate * tilt_gate).astype(np.float32)

    def _reward_command_forward_vel(self, data, commands: np.ndarray):
        local_vel = self.get_local_linvel(data)
        target = np.clip(commands[:, 0], 0.0, None)
        margin = self.cfg.reward_config.command_forward_vel_margin
        progress = np.clip(local_vel[:, 0] - target + margin, 0.0, 2.0 * margin) / max(
            2.0 * margin,
            1.0e-5,
        )
        moving = target > 1.0e-4
        return (progress * moving * self._upright_gate(data)).astype(np.float32)

    def _reward_overspeed(self, data, commands: np.ndarray):
        local_vel = self.get_local_linvel(data)
        target = np.clip(commands[:, 0], 0.0, None)
        margin = self.cfg.reward_config.command_forward_vel_margin
        excess = np.clip(local_vel[:, 0] - target - margin, 0.0, None)
        return (np.square(excess) * self._upright_gate(data)).astype(np.float32)

    def _reward_stand_still(self, data):
        local_vel = self.get_local_linvel(data)
        return np.sum(np.square(local_vel[:, :2]), axis=1)

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_dof_pos_limits(self, data):
        dof_pos = self.get_dof_pos(data)
        lower_violation = np.clip(self._soft_dof_pos_lower - dof_pos, 0.0, None)
        upper_violation = np.clip(dof_pos - self._soft_dof_pos_upper, 0.0, None)
        return np.sum(lower_violation + upper_violation, axis=1)

    def _reward_hip_pos(self, data):
        if self._hip_indices.size == 0:
            return np.zeros((data.shape[0],), dtype=np.float32)
        return np.sum(np.square(self.get_dof_pos(data)[:, self._hip_indices]), axis=1)

    def _reward_joint_regularization(self, data):
        return np.sum(np.square(self.get_dof_pos(data) - self.default_angles), axis=1)

    def _reward_upper_body_regularization(self, data):
        if self._upper_body_action_indices.size == 0:
            return np.zeros((data.shape[0],), dtype=np.float32)
        upper_diff = self.get_dof_pos(data)[:, self._upper_body_action_indices] - self.default_angles[
            self._upper_body_action_indices
        ]
        return np.sum(np.square(upper_diff), axis=1)

    def _reward_upper_body_velocity(self, data):
        if self._upper_body_action_indices.size == 0:
            return np.zeros((data.shape[0],), dtype=np.float32)
        upper_vel = self.get_dof_vel(data)[:, self._upper_body_action_indices]
        return np.sum(np.square(upper_vel), axis=1)

    def _exp_motion_reward(self, error: np.ndarray, sigma: float) -> np.ndarray:
        error_l2 = np.mean(np.square(error), axis=1)
        return np.exp(-error_l2 / max(float(sigma) ** 2, 1.0e-8)).astype(np.float32)

    def _reward_motion_leg_joint_pos(self, data, motion_sample: dict[str, np.ndarray]):
        if self._leg_action_indices.size == 0:
            return np.zeros((data.shape[0],), dtype=np.float32)
        error = self.get_dof_pos(data)[:, self._leg_action_indices] - motion_sample["joint_pos"][
            :, self._leg_action_indices
        ]
        return self._exp_motion_reward(error, self.cfg.reward_config.motion_joint_pos_sigma)

    def _reward_motion_leg_joint_vel(self, data, motion_sample: dict[str, np.ndarray]):
        if self._leg_action_indices.size == 0:
            return np.zeros((data.shape[0],), dtype=np.float32)
        error = self.get_dof_vel(data)[:, self._leg_action_indices] - motion_sample["joint_vel"][
            :, self._leg_action_indices
        ]
        return self._exp_motion_reward(error, self.cfg.reward_config.motion_joint_vel_sigma)

    def _reward_motion_base_height(self, data, motion_sample: dict[str, np.ndarray]):
        pose = self._body.get_pose(data)
        height_error = pose[:, 2:3] - motion_sample["root_pos"][:, 2:3]
        return self._exp_motion_reward(height_error, self.cfg.reward_config.motion_base_height_sigma)

    def _foot_contact_matrix(self, info: dict) -> np.ndarray:
        left = info.get("left_contact", np.zeros((self._num_envs,), dtype=np.float32)).astype(bool)
        right = info.get("right_contact", np.zeros((self._num_envs,), dtype=np.float32)).astype(bool)
        return np.stack([left, right], axis=1)

    def _reward_contact(self, info: dict):
        if not self.cfg.reward_config.trust_contact_rewards:
            return np.zeros((self._num_envs,), dtype=np.float32)
        contacts = self._foot_contact_matrix(info)
        phase = info["gait_phase"]
        stance = np.stack([phase < 0.55, np.fmod(phase + 0.5, 1.0) < 0.55], axis=1)
        return np.sum(~np.logical_xor(contacts, stance), axis=1).astype(np.float32)

    def _reward_feet_swing_height(self, info: dict):
        if not self.cfg.reward_config.trust_contact_rewards:
            return np.zeros((self._num_envs,), dtype=np.float32)
        contacts = self._foot_contact_matrix(info)
        feet_pos = info.get("feet_pos")
        if feet_pos is None:
            return np.zeros((contacts.shape[0],), dtype=np.float32)
        pos_error = np.square(feet_pos[:, :, 2] - self.cfg.reward_config.swing_height) * ~contacts
        return np.sum(pos_error, axis=1).astype(np.float32)

    def _reward_contact_no_vel(self, info: dict):
        if not self.cfg.reward_config.trust_contact_rewards:
            return np.zeros((self._num_envs,), dtype=np.float32)
        contacts = self._foot_contact_matrix(info)
        feet_vel = info.get("feet_vel")
        if feet_vel is None:
            return np.zeros((contacts.shape[0],), dtype=np.float32)
        contact_feet_vel = feet_vel * contacts[:, :, None]
        return np.sum(np.square(contact_feet_vel[:, :, :3]), axis=(1, 2)).astype(np.float32)

    def _update_feet_air_time(self, info: dict):
        feet_air_time = info["feet_air_time"]
        feet_air_time += self.cfg.ctrl_dt
        feet_air_time *= ~info["contacts"]
        return feet_air_time

    def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
        feet_air_time = info["feet_air_time"]
        first_contact = (feet_air_time > 0.0) * info["contacts"]
        rew_air_time = np.sum((feet_air_time - self.cfg.reward_config.target_feet_air_time) * first_contact, axis=1)
        rew_air_time *= np.linalg.norm(commands[:, :2], axis=1) > self.cfg.commands.command_deadzone
        return rew_air_time

    def _reward_collision(self, data: mtx.SceneData):
        cquery = self._model.get_contact_query(data)
        colliding = cquery.is_colliding(self.collision_contact_pairs)
        return np.sum(colliding.reshape((self._num_envs, self.collision_check_num)), axis=1)

    def _average_geom_vector(self, geoms, data: mtx.SceneData, getter: str) -> np.ndarray:
        if not geoms:
            return np.zeros((data.shape[0], 3), dtype=np.float32)
        values = []
        for geom in geoms:
            method = getattr(geom, getter, None)
            if method is None:
                return np.zeros((data.shape[0], 3), dtype=np.float32)
            value = method(data)
            if getter == "get_pose":
                value = value[:, :3]
            values.append(value.astype(np.float32))
        return np.mean(np.stack(values, axis=1), axis=1).astype(np.float32)

    def _get_foot_positions(self, data: mtx.SceneData) -> np.ndarray:
        left_pos = self._average_geom_vector(self._left_foot_geoms, data, "get_pose")
        right_pos = self._average_geom_vector(self._right_foot_geoms, data, "get_pose")
        return np.stack([left_pos, right_pos], axis=1).astype(np.float32)

    def _get_foot_velocities(self, data: mtx.SceneData) -> np.ndarray:
        left_vel = self._average_geom_vector(self._left_foot_geoms, data, "get_linear_velocity")
        right_vel = self._average_geom_vector(self._right_foot_geoms, data, "get_linear_velocity")
        return np.stack([left_vel, right_vel], axis=1).astype(np.float32)
