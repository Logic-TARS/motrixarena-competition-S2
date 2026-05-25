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
from motrix_envs.locomotion.k1.cfg import K1BallNavigateEnvCfg
from motrix_envs.locomotion.k1.walk_np import K1WalkTask
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnvState


@registry.env("k1-ball-navigate", sim_backend="np")
class K1BallNavigateTask(K1WalkTask):
    def __init__(self, cfg: K1BallNavigateEnvCfg, num_envs=1):
        self._ball_body = None
        self._ball_geom_idx = -1
        super().__init__(cfg, num_envs)

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

    def _init_buffer(self):
        super()._init_buffer()
        cfg = self._cfg
        assert isinstance(cfg, K1BallNavigateEnvCfg)

        self._ball_body = self._model.get_body(cfg.ball_config.body_name)
        self._ball_geom_idx = self._model.get_geom_index(cfg.ball_config.geom_name)

        # Ball DOF velocity indices for tracking ball speed
        self._ball_dof_vel_indices = self._ball_body.get_dof_vel_indices()
        self._ball_dof_pos_indices = self._ball_body.get_dof_pos_indices()
        self._init_dof_pos[self._ball_dof_pos_indices[3:7]] = [1.0, 0.0, 0.0, 0.0]

        # Build ball-foot contact pairs
        foot_geoms = []
        for geom_name in self._model.geom_names:
            if geom_name is not None and cfg.asset.foot_name in geom_name:
                foot_geoms.append(self._model.get_geom_index(geom_name))
        if not foot_geoms:
            foot_geoms = list(cfg.asset.left_foot_geom_indices) + list(cfg.asset.right_foot_geom_indices)

        self.ball_foot_pairs = np.array(
            [[self._ball_geom_idx, f] for f in foot_geoms], dtype=np.uint32
        ).reshape((-1, 2))
        self.ball_foot_check_num = self.ball_foot_pairs.shape[0] if self.ball_foot_pairs.size > 0 else 0

    def _spawn_ball(self, num_reset: int):
        cfg = self._cfg
        assert isinstance(cfg, K1BallNavigateEnvCfg)
        bc = cfg.ball_config

        dist = np.random.uniform(bc.spawn_dist_min, bc.spawn_dist_max, size=num_reset)
        angle = np.random.uniform(-bc.spawn_angle_max, bc.spawn_angle_max, size=num_reset)
        ball_x = np.cos(angle) * dist
        ball_y = np.sin(angle) * dist
        ball_z = np.full(num_reset, bc.radius)
        return np.stack([ball_x, ball_y, ball_z], axis=1)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        data.reset(self._model)

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))
        dof_pos[:, self._joint_dof_pos_indices] += 0.01 * np.random.randn(num_reset, self._num_action)

        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)

        # Spawn ball at random position in front of robot (override ball free joint DOFs)
        ball_positions = self._spawn_ball(num_reset)
        ball_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_reset, 1))
        ball_dof_pos = np.concatenate([ball_positions, ball_quat], axis=1)
        dof_pos[:, self._ball_dof_pos_indices] = ball_dof_pos
        data.set_dof_pos(dof_pos, self._model)

        self._model.forward_kinematic(data)

        ball_rel = self._get_ball_relative_pos(data)
        ball_dist = np.linalg.norm(ball_rel[:, :2], axis=1).astype(np.float32)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": self._commands_to_ball(ball_rel),
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "gait_phase": np.zeros((num_reset,), dtype=np.float32),
            "feet_air_time": np.zeros((num_reset, self.foot_check_num), dtype=np.float32),
            "contacts": np.zeros((num_reset, self.foot_check_num), dtype=bool),
            "left_contact": np.zeros((num_reset,), dtype=np.float32),
            "right_contact": np.zeros((num_reset,), dtype=np.float32),
            "prev_ball_vel": np.zeros((num_reset, 3), dtype=np.float32),
            "ball_pos": ball_positions.copy(),
            "prev_ball_dist": ball_dist.copy(),
            "ball_dist": ball_dist.copy(),
        }
        return self._get_obs(data, info), info

    def _commands_to_ball(self, ball_rel: np.ndarray) -> np.ndarray:
        bc = self.cfg.ball_config
        heading = np.arctan2(ball_rel[:, 1], ball_rel[:, 0])
        dist = np.linalg.norm(ball_rel[:, :2], axis=1)
        arrived = dist <= bc.arrival_radius

        commands = np.zeros((ball_rel.shape[0], 3), dtype=np.float32)
        forward_gate = np.clip(np.cos(heading), 0.0, 1.0)
        commands[:, 0] = np.clip(
            bc.command_forward_gain * dist * forward_gate,
            0.0,
            bc.command_max_forward_vel,
        )
        commands[:, 2] = np.clip(bc.command_turn_gain * heading, -bc.command_max_yaw_rate, bc.command_max_yaw_rate)
        commands[arrived, :] = 0.0
        return commands

    def _get_ball_relative_pos(self, data: mtx.SceneData) -> np.ndarray:
        ball_dof_pos = data.dof_pos[:, self._ball_dof_pos_indices]
        ball_world_pos = ball_dof_pos[:, :3]
        base_pose = self._body.get_pose(data)
        base_pos = base_pose[:, :3]
        base_quat = base_pose[:, 3:7]
        relative = ball_world_pos - base_pos
        return quaternion.rotate_inverse(base_quat, relative)

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
        ball_dof_pos = state.data.dof_pos[:, self._ball_dof_pos_indices]
        ball_dof_vel = state.data.dof_vel[:, self._ball_dof_vel_indices]
        ball_rel = self._get_ball_relative_pos(state.data)
        ball_dist = np.linalg.norm(ball_rel[:, :2], axis=1).astype(np.float32)
        state.info["prev_ball_dist"] = state.info.get("ball_dist", ball_dist).copy()
        state.info["ball_dist"] = ball_dist.copy()
        state.info["commands"] = self._commands_to_ball(ball_rel)
        state = super().update_observation(state)
        state.info["ball_pos"] = ball_dof_pos[:, :3]
        state.info["ball_vel"] = ball_dof_vel[:, :3]
        return state

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        reward_dict = self._get_reward(state.data, state.info)
        scales = self.cfg.reward_config.scales
        rewards = {k: v * scales.get(k, 0.0) for k, v in reward_dict.items()}
        nav_pressure_keys = {"approach_ball", "low_speed_penalty"}
        base_rwd = sum(v for k, v in rewards.items() if k not in nav_pressure_keys)
        nav_pressure = sum(rewards[k] for k in nav_pressure_keys if k in rewards)
        rwd = np.clip(base_rwd, 0.0, 10000.0) + nav_pressure
        rwd = np.clip(rwd, -0.5, 10000.0)
        fall_done = state.info.get("fall_done", state.terminated)
        rwd = np.where(fall_done, 0.0, rwd)
        if "termination" in scales:
            rwd += self._reward_termination(fall_done) * scales["termination"]
        return state.replace(reward=rwd)

    def _get_reward(self, data: mtx.SceneData, info: dict) -> dict[str, np.ndarray]:
        result = super()._get_reward(data, info)
        result["approach_ball"] = self._reward_approach_ball(data, info)
        result["low_speed_penalty"] = self._reward_low_speed_penalty(data, info)
        result["ball_velocity"] = self._reward_ball_velocity(info)
        result["orientation_to_ball"] = self._reward_orientation_to_ball(data, info)
        result["arrival_bonus"] = self._reward_arrival_bonus(info)
        return result

    def _reward_approach_ball(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        progress_vel = (info["prev_ball_dist"] - info["ball_dist"]) / self.cfg.ctrl_dt
        return np.clip(progress_vel, -0.5, 0.8).astype(np.float32)

    def _reward_low_speed_penalty(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        ball_rel = self._get_ball_relative_pos(data)
        heading = np.arctan2(ball_rel[:, 1], ball_rel[:, 0])
        ball_in_front = np.cos(heading) > 0.5
        forward_vel = self.get_local_linvel(data)[:, 0]
        far_from_ball = info["ball_dist"] > self.cfg.ball_config.arrival_radius
        deficit = np.clip((0.12 - forward_vel) / 0.12, 0.0, 1.0)
        return (deficit * far_from_ball * ball_in_front).astype(np.float32)

    def _reward_ball_velocity(self, info: dict) -> np.ndarray:
        ball_vel = info.get("ball_vel", np.zeros((self._num_envs, 3), dtype=np.float32))
        return np.linalg.norm(ball_vel[:, :2], axis=1)

    def _reward_orientation_to_ball(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        ball_rel = self._get_ball_relative_pos(data)
        dist = np.linalg.norm(ball_rel[:, :2], axis=1)
        heading_error = np.arctan2(ball_rel[:, 1], ball_rel[:, 0])
        orientation_score = np.cos(heading_error)
        return np.where(dist < 0.3, orientation_score, orientation_score * 0.3)

    def _reward_arrival_bonus(self, info: dict) -> np.ndarray:
        radius = self.cfg.ball_config.arrival_radius
        just_arrived = (info["prev_ball_dist"] > radius) & (info["ball_dist"] <= radius)
        return just_arrived.astype(np.float32)

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        state = super().update_terminated(state)
        arrived = state.info["ball_dist"] <= self.cfg.ball_config.arrival_radius
        fall_done = state.terminated.copy()
        state.info["fall_done"] = fall_done
        state.info["arrived_done"] = arrived & ~fall_done
        return state.replace(terminated=fall_done | arrived)
