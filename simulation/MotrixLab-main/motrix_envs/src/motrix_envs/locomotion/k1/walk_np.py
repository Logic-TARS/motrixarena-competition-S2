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
        num_dof_vel = self._model.num_dof_vel
        num_joint_angle = self._model.num_dof_pos - 7
        num_gravity = 3
        num_actions = self._model.num_actuators
        num_command = 3
        num_obs = num_dof_vel + num_joint_angle + num_gravity + num_actions + num_command
        assert num_obs == 48
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

    def apply_action(self, actions, state):
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
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
        linear_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        local_gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        diff = self.get_dof_pos(data) - self.default_angles

        obs = np.hstack(
            [
                linear_vel * self.cfg.normalization.lin_vel,
                gyro * self.cfg.normalization.ang_vel,
                local_gravity,
                diff * self.cfg.normalization.dof_pos,
                self.get_dof_vel(data) * self.cfg.normalization.dof_vel,
                info["current_actions"],
                info["commands"] * self.commands_scale,
            ]
        )
        return obs

    def update_observation(self, state: NpEnvState):
        return state.replace(obs=self._get_obs(state.data, state.info))

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
        rwd = np.clip(sum(rewards.values()), 0.0, 10000.0)
        rwd = np.where(state.terminated, np.array(0.0), rwd)
        return state.replace(reward=rwd)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        data.reset(self._model)
        data.set_dof_vel(self._init_dof_vel)
        data.set_dof_pos(self._init_dof_pos, self._model)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": self.resample_commands(num_reset),
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
        }
        return self._get_obs(data, info), info

    def _get_reward(self, data: mtx.SceneData, info: dict) -> dict[str, np.ndarray]:
        commands = info["commands"]
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, commands),
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, commands),
            "lin_vel_z": np.square(self.get_local_linvel(data)[:, 2]),
            "ang_vel_xy": np.sum(np.square(self.get_gyro(data)[:, :2]), axis=1),
            "orientation": self._reward_orientation(data),
            "torques": np.sum(np.square(data.actuator_ctrls), axis=1),
            "dof_acc": np.sum(
                np.square((info["last_dof_vel"] - self.get_dof_vel(data)) / self.cfg.ctrl_dt),
                axis=1,
            ),
            "action_rate": np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1),
            "stand_still": self._reward_stand_still(data, commands),
            "joint_regularization": np.sum(np.square(self.get_dof_pos(data) - self.default_angles), axis=1),
        }

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_orientation(self, data):
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _reward_stand_still(self, data, commands: np.ndarray):
        return np.sum(np.abs(self.get_dof_pos(data) - self.default_angles), axis=1) * (
            np.linalg.norm(commands, axis=1) < 0.1
        )
