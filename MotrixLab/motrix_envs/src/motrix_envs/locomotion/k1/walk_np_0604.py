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

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.k1.cfg import K1WalkNpEnvCfg
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState


@registry.env("k1-flat-terrain-walk", sim_backend="np")
class K1WalkTask(NpEnv):
    def __init__(self, cfg: K1WalkNpEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._init_action_space()
        self._init_obs_space()
        self._body = self._model.get_body(self.cfg.asset.body_name)
        self._num_action = self._action_space.shape[0]
        self._num_observation = self._observation_space.shape[0]
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._joint_dof_pos_indices = self._get_actuated_dof_pos_indices()
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_buffer()

    def _init_obs_space(self):
        num_gravity = 3
        num_local_linvel = 3
        num_gyro = 3
        num_actions = self._model.num_actuators
        num_dof_vel = num_actions
        num_joint_angle = num_actions
        num_command = 3
        num_gait_phase = 2
        num_foot_contact = 2
        num_obs = (
            num_gravity
            + num_local_linvel
            + num_gyro
            + num_command
            + num_gait_phase
            + num_joint_angle
            + num_dof_vel
            + num_actions
            + num_foot_contact
        )
        assert num_obs == 52
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
        from motrix_envs.locomotion.k1.cfg import K1BallNavigateEnvCfg, K1PointNavigateEnvCfg  # noqa: F811

        assert isinstance(cfg, (K1WalkNpEnvCfg, K1BallNavigateEnvCfg, K1PointNavigateEnvCfg))

        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32,
        )

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.kps = np.zeros(self._num_action, dtype=np.float32)
        self.kds = np.zeros(self._num_action, dtype=np.float32)
        for i, name in enumerate(self._model.actuator_names):
            self.default_angles[i] = cfg.init_state.default_joint_angles.get(
                name,
                cfg.init_state.default_joint_angles["default"],
            )
            self.kps[i] = cfg.control_config.stiffness[name]
            self.kds[i] = cfg.control_config.damping[name]

        self._init_dof_pos[:3] = np.asarray(cfg.init_state.pos, dtype=np.float32)
        self._init_dof_pos[self._joint_dof_pos_indices] = self.default_angles

        joint_limits = np.asarray(self._model.joint_limits, dtype=np.float32)
        if joint_limits.shape == (2, self._num_action):
            self.dof_pos_lower = joint_limits[0].copy()
            self.dof_pos_upper = joint_limits[1].copy()
        else:
            self.dof_pos_lower = np.full(self._num_action, -1.5, dtype=np.float32)
            self.dof_pos_upper = np.full(self._num_action, 1.5, dtype=np.float32)
        self.dof_pos_limit_margin = 0.9
        self._hip_indices = self._find_hip_indices()

        # Build foot contact pairs (foot geoms <-> ground), separated by left/right.
        ground_geoms = []
        left_foot_geoms = []
        right_foot_geoms = []
        collision_geoms = []
        for geom_name in self._model.geom_names:
            if geom_name is None:
                continue
            if cfg.asset.ground_name in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))
            elif "Left" in geom_name and cfg.asset.foot_name in geom_name:
                left_foot_geoms.append(self._model.get_geom_index(geom_name))
            elif "Right" in geom_name and cfg.asset.foot_name in geom_name:
                right_foot_geoms.append(self._model.get_geom_index(geom_name))
            elif any(part in geom_name for part in cfg.asset.penalize_contacts_on):
                collision_geoms.append(self._model.get_geom_index(geom_name))

        if not ground_geoms:
            ground_geoms = [0]
        if not left_foot_geoms:
            left_foot_geoms = list(cfg.asset.left_foot_geom_indices)
        if not right_foot_geoms:
            right_foot_geoms = list(cfg.asset.right_foot_geom_indices)
        if not collision_geoms:
            collision_geoms = list(cfg.asset.collision_geom_indices)

        self._left_foot_geoms = [self._model.get_geom(idx) for idx in left_foot_geoms]
        self._right_foot_geoms = [self._model.get_geom(idx) for idx in right_foot_geoms]

        # Foot contact pairs for feet_air_time reward (all feet)
        foot_geoms = left_foot_geoms + right_foot_geoms
        self.foot_contact_pairs = np.array(
            [[f, g] for f in foot_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))
        self.foot_check_num = self.foot_contact_pairs.shape[0] if self.foot_contact_pairs.size > 0 else 0

        # Separate left/right contact pairs for observation
        self.left_foot_pairs = np.array(
            [[f, g] for f in left_foot_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))
        self.right_foot_pairs = np.array(
            [[f, g] for f in right_foot_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))

        self.collision_contact_pairs = np.array(
            [[c, g] for c in collision_geoms for g in ground_geoms], dtype=np.uint32
        ).reshape((-1, 2))
        self.collision_check_num = self.collision_contact_pairs.shape[0] if self.collision_contact_pairs.size > 0 else 0

    def _find_hip_indices(self) -> np.ndarray:
        hip_names = ("Hip_Roll", "Hip_Yaw")
        indices = []
        for i, name in enumerate(self._model.actuator_names):
            if any(hip_name in name for hip_name in hip_names):
                indices.append(i)
        return np.array(indices, dtype=np.int64) if indices else np.array([], dtype=np.int64)

    def apply_action(self, actions, state):
        if actions.ndim == 1:
            actions = np.tile(actions, (self._num_envs, 1))
        actions = np.clip(actions, self._action_space.low, self._action_space.high).astype(np.float32)
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.info["episode_length"] = state.info.get("episode_length", 0) + 1
        state.info["gait_phase"] = np.fmod(
            state.info["episode_length"] * self.cfg.ctrl_dt / 0.8, 1.0
        ).astype(np.float32)
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state

    def _compute_torques(self, actions, data):
        target_pos = actions * self.cfg.control_config.action_scale + self.default_angles
        torques = self.kps * (target_pos - self.get_dof_pos(data)) - self.kds * self.get_dof_vel(data)
        return np.clip(torques, -self.cfg.control_config.torque_limit, self.cfg.control_config.torque_limit)

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self.cfg.sensor.local_linvel, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self.cfg.sensor.gyro, data)

    def update_state(self, state):
        state = self.update_observation(state)
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        gyro = self.get_gyro(data)
        linvel = self.get_local_linvel(data)
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        local_gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        diff = self.get_dof_pos(data) - self.default_angles
        gait = info["gait_phase"]

        obs = np.zeros((data.shape[0], self._num_observation), dtype=np.float32)
        obs[:, 0:3] = local_gravity
        obs[:, 3:6] = linvel * self.cfg.normalization.lin_vel
        obs[:, 6:9] = gyro * self.cfg.normalization.ang_vel
        obs[:, 9:12] = info["commands"] * self.commands_scale
        obs[:, 12] = np.cos(2 * np.pi * gait)
        obs[:, 13] = np.sin(2 * np.pi * gait)
        obs[:, 14:26] = diff * self.cfg.normalization.dof_pos
        obs[:, 26:38] = self.get_dof_vel(data) * self.cfg.normalization.dof_vel
        obs[:, 38:50] = info["current_actions"]
        obs[:, 50] = info.get("left_contact", np.zeros(data.shape[0], dtype=np.float32))
        obs[:, 51] = info.get("right_contact", np.zeros(data.shape[0], dtype=np.float32))
        return obs

    def update_observation(self, state: NpEnvState):
        if self.foot_check_num > 0:
            cquery = self._model.get_contact_query(state.data)
            foot_contact = cquery.is_colliding(self.foot_contact_pairs)
            state.info["contacts"] = foot_contact.reshape((self._num_envs, self.foot_check_num))
            state.info["feet_air_time"] = self._update_feet_air_time(state.info)
            # Left/right foot contact for observation
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
        return state.replace(terminated=too_low | too_tilted)

    def resample_commands(self, num_envs: int):
        commands = np.random.uniform(
            low=self.cfg.commands.vel_limit[0],
            high=self.cfg.commands.vel_limit[1],
            size=(num_envs, 3),
        )
        return commands.astype(np.float32)

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        reward_dict = self._get_reward(state.data, state.info)
        scales = self.cfg.reward_config.scales
        rewards = {k: v * scales[k] for k, v in reward_dict.items() if k in scales}
        rwd = sum(rewards.values())
        rwd = np.clip(rwd, -1000.0, 10000.0)
        if state.terminated is not None:
            rwd = np.where(state.terminated, 0.0, rwd)
        if "termination" in scales:
            done = state.terminated if state.terminated is not None else np.zeros(state.data.shape[0], dtype=bool)
            rwd += self._reward_termination(done) * scales["termination"]
        return state.replace(reward=rwd)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        data.reset(self._model)

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))

        dof_pos[:, self._joint_dof_pos_indices] += 0.01 * np.random.randn(num_reset, self._num_action)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

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
        }
        return self._get_obs(data, info), info

    def _get_reward(self, data: mtx.SceneData, info: dict) -> dict[str, np.ndarray]:
        commands = info["commands"]
        result = {
            "alive": self._reward_alive(data),
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, commands),
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, commands),
            "command_forward_vel": self._reward_command_forward_vel(data, commands),
            "overspeed": self._reward_overspeed(data, commands),
            "straight_motion": self._reward_straight_motion(data, commands),
            "lin_vel_z": self._reward_lin_vel_z(data),
            "ang_vel_xy": self._reward_ang_vel_xy(data),
            "orientation": self._reward_orientation(data),
            "base_height": self._reward_base_height(data),
            "torques": self._reward_torques(data),
            "dof_vel": self._reward_dof_vel(data),
            "dof_acc": self._reward_dof_acc(data, info),
            "action_rate": self._reward_action_rate(info),
            "joint_regularization": self._reward_joint_regularization(data),
        }
        if self.foot_check_num > 0:
            result["feet_air_time"] = self._reward_feet_air_time(commands, info)
        if self.collision_check_num > 0:
            result["collision"] = self._reward_collision(data)
        return result

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
        return np.ones((data.shape[0],), dtype=np.float32)

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]), axis=1)
        return (
            np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)
            * self._reward_forward_posture_gate(data)
            * self._reward_forward_straight_gate(data, commands)
        )

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_command_forward_vel(self, data, commands: np.ndarray):
        forward_vel = self.get_local_linvel(data)[:, 0]
        command_vel = np.maximum(commands[:, 0], 1.0e-5)
        reward = np.clip(forward_vel, 0.0, command_vel) / command_vel
        return reward * self._reward_forward_posture_gate(data) * self._reward_forward_straight_gate(data, commands)

    def _reward_overspeed(self, data, commands: np.ndarray):
        forward_vel = self.get_local_linvel(data)[:, 0]
        max_forward_vel = commands[:, 0] + self.cfg.reward_config.forward_vel_margin
        return np.square(np.maximum(forward_vel - max_forward_vel, 0.0))

    def _reward_straight_motion(self, data, commands: np.ndarray):
        cfg = self.cfg.reward_config
        local_vel = self.get_local_linvel(data)
        yaw_error = self.get_gyro(data)[:, 2] - commands[:, 2]
        return cfg.straight_motion_yaw_weight * np.square(yaw_error) + cfg.straight_motion_lateral_weight * np.square(
            local_vel[:, 1]
        )

    def _reward_forward_posture_gate(self, data):
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        tilt_xy = np.linalg.norm(gravity[:, :2], axis=1)

        cfg = self.cfg.reward_config
        height_span = max(cfg.forward_reward_full_height - cfg.forward_reward_min_height, 1.0e-5)
        height_gate = np.clip((pose[:, 2] - cfg.forward_reward_min_height) / height_span, 0.0, 1.0)

        tilt_span = max(cfg.forward_reward_max_tilt_xy - cfg.forward_reward_full_tilt_xy, 1.0e-5)
        tilt_gate = np.clip((cfg.forward_reward_max_tilt_xy - tilt_xy) / tilt_span, 0.0, 1.0)
        gate = height_gate * tilt_gate
        return (cfg.forward_reward_min_gate + (1.0 - cfg.forward_reward_min_gate) * gate).astype(np.float32)

    def _reward_forward_straight_gate(self, data, commands: np.ndarray):
        cfg = self.cfg.reward_config
        local_vel = self.get_local_linvel(data)
        yaw_error = np.abs(self.get_gyro(data)[:, 2] - commands[:, 2])
        lateral_vel = np.abs(local_vel[:, 1])

        yaw_span = max(cfg.forward_reward_max_yaw_rate - cfg.forward_reward_full_yaw_rate, 1.0e-5)
        yaw_gate = np.clip((cfg.forward_reward_max_yaw_rate - yaw_error) / yaw_span, 0.0, 1.0)

        lateral_span = max(cfg.forward_reward_max_lateral_vel - cfg.forward_reward_full_lateral_vel, 1.0e-5)
        lateral_gate = np.clip((cfg.forward_reward_max_lateral_vel - lateral_vel) / lateral_span, 0.0, 1.0)
        return (cfg.forward_reward_min_gate + (1.0 - cfg.forward_reward_min_gate) * yaw_gate * lateral_gate).astype(
            np.float32
        )

    def _reward_stand_still(self, data, commands: np.ndarray):
        command_speed = np.linalg.norm(commands[:, :2], axis=1)
        actual_speed = np.linalg.norm(self.get_local_linvel(data)[:, :2], axis=1)
        return ((command_speed > 0.2) & (actual_speed < 0.08)).astype(np.float32)

    def _reward_joint_regularization(self, data):
        return np.sum(np.square(self.get_dof_pos(data) - self.default_angles), axis=1)

    def _update_feet_air_time(self, info: dict):
        feet_air_time = info["feet_air_time"]
        feet_air_time += self.cfg.ctrl_dt
        feet_air_time *= ~info["contacts"]
        return feet_air_time

    def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
        feet_air_time = info["feet_air_time"]
        first_contact = (feet_air_time > 0.0) * info["contacts"]
        rew_air_time = np.sum((feet_air_time - 0.5) * first_contact, axis=1)
        rew_air_time *= np.linalg.norm(commands[:, :2], axis=1) > 0.1
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
            value = getattr(geom, getter)(data)
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
