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

    def _init_buffer(self):
        cfg = self._cfg
        assert isinstance(cfg, K1WalkNpEnvCfg)

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
        self._init_dof_pos[-self._num_action :] = self.default_angles

        # Build foot contact pairs (foot geoms <-> ground), separated by left/right.
        # The current K1 MJCF leaves most geom names empty in MotrixSim, so cfg index
        # fallbacks keep feet_air_time/collision active for this asset.
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

    def apply_action(self, actions, state):
        actions = np.clip(actions, self._action_space.low, self._action_space.high).astype(np.float32)
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.info["gait_phase"] = np.fmod(
            state.info["gait_phase"] + self.cfg.ctrl_dt * self.cfg.reward_config.gait_frequency,
            1.0,
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
        gait_mask = float(self.cfg.reward_config.gait_frequency > 1.0e-8)

        obs = np.zeros((data.shape[0], self._num_observation), dtype=np.float32)
        obs[:, 0:3] = local_gravity
        obs[:, 3:6] = linvel * self.cfg.normalization.lin_vel
        obs[:, 6:9] = gyro * self.cfg.normalization.ang_vel
        obs[:, 9:12] = info["commands"] * self.commands_scale
        obs[:, 12] = np.cos(2 * np.pi * gait) * gait_mask
        obs[:, 13] = np.sin(2 * np.pi * gait) * gait_mask
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
        rewards = {k: v * self.cfg.reward_config.scales[k] for k, v in reward_dict.items()}
        rwd = sum(rewards.values())
        rwd = np.clip(rwd, 0.0, 10000.0)
        if "termination" in self.cfg.reward_config.scales:
            rwd += self._reward_termination(state.terminated) * self.cfg.reward_config.scales["termination"]
        return state.replace(reward=rwd)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        data.reset(self._model)

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))

        dof_pos[:, -self._num_action :] += 0.01 * np.random.randn(num_reset, self._num_action)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": self.resample_commands(num_reset),
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "gait_phase": np.zeros((num_reset,), dtype=np.float32),
            "feet_air_time": np.zeros((num_reset, self.foot_check_num), dtype=np.float32),
            "contacts": np.zeros((num_reset, self.foot_check_num), dtype=bool),
            "left_contact": np.zeros((num_reset,), dtype=np.float32),
            "right_contact": np.zeros((num_reset,), dtype=np.float32),
        }
        return self._get_obs(data, info), info

    def _get_reward(self, data: mtx.SceneData, info: dict) -> dict[str, np.ndarray]:
        commands = info["commands"]
        result = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, commands),
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, commands),
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
        return done

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

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
