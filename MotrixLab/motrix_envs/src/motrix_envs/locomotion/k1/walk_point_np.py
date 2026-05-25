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

import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.k1.cfg import K1PointNavigateEnvCfg
from motrix_envs.locomotion.k1.walk_np import K1WalkTask
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnvState


@registry.env("k1-point-navigate", sim_backend="np")
class K1PointNavigateTask(K1WalkTask):
    """Point-to-point navigation using the walk policy's 52-dim observation space.

    Target guidance is encoded entirely through velocity commands (channels 9:12),
    which the pre-trained walk policy already knows how to track. No extra target
    observation channels — the walk warm-start transfers without dimension mismatch.
    """

    def __init__(self, cfg: K1PointNavigateEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        _, info = super().reset(data)
        target_pos = self._sample_target_positions(data)
        target_rel = self._target_relative_pos(data, target_pos)
        target_dist = np.linalg.norm(target_rel[:, :2], axis=1).astype(np.float32)
        info["target_pos"] = target_pos
        info["prev_target_dist"] = target_dist.copy()
        info["target_dist"] = target_dist.copy()
        info["reached_target"] = target_dist <= self.cfg.point_config.arrival_radius
        info["commands"] = self._commands_to_target(target_rel)
        return self._get_obs(data, info), info

    def update_observation(self, state: NpEnvState):
        target_pos = state.info["target_pos"]
        target_rel = self._target_relative_pos(state.data, target_pos)
        target_dist = np.linalg.norm(target_rel[:, :2], axis=1).astype(np.float32)
        state.info["prev_target_dist"] = state.info.get("target_dist", target_dist).copy()
        state.info["target_dist"] = target_dist.copy()
        state.info["reached_target"] = state.info.get("reached_target", False) | (
            target_dist <= self.cfg.point_config.arrival_radius
        )
        state.info["commands"] = self._commands_to_target(target_rel)
        return super().update_observation(state)

    def _sample_target_positions(self, data: mtx.SceneData) -> np.ndarray:
        cfg = self.cfg.point_config
        num_envs = data.shape[0]
        dist = np.random.uniform(cfg.spawn_dist_min, cfg.spawn_dist_max, size=num_envs).astype(np.float32)
        angle = np.random.uniform(-cfg.spawn_angle_max, cfg.spawn_angle_max, size=num_envs).astype(np.float32)
        local_target = np.zeros((num_envs, 3), dtype=np.float32)
        local_target[:, 0] = np.cos(angle) * dist
        local_target[:, 1] = np.sin(angle) * dist

        base_pose = self._body.get_pose(data)
        target_world = base_pose[:, :3] + quaternion.rotate_vector(base_pose[:, 3:7], local_target)
        target_world[:, 2] = 0.0
        return target_world.astype(np.float32)

    def _target_relative_pos(self, data: mtx.SceneData, target_pos: np.ndarray | None = None) -> np.ndarray:
        if target_pos is None:
            target_pos = np.zeros((data.shape[0], 3), dtype=np.float32)
        base_pose = self._body.get_pose(data)
        relative = target_pos - base_pose[:, :3]
        return quaternion.rotate_inverse(base_pose[:, 3:7], relative).astype(np.float32)

    def _commands_to_target(self, target_rel: np.ndarray) -> np.ndarray:
        """Generate target-relative velocity commands for the 52-dim walk policy."""
        cfg = self.cfg.point_config
        heading = np.arctan2(target_rel[:, 1], target_rel[:, 0])
        dist = np.linalg.norm(target_rel[:, :2], axis=1)
        arrived = dist <= cfg.arrival_radius

        commands = np.zeros((target_rel.shape[0], 3), dtype=np.float32)
        forward_gate = np.clip(np.cos(heading), 0.0, 1.0)
        commands[:, 0] = np.clip(
            cfg.command_forward_gain * dist * forward_gate,
            0.0,
            cfg.command_max_forward_vel,
        )
        commands[:, 2] = np.clip(
            cfg.command_turn_gain * heading,
            -cfg.command_max_yaw_rate,
            cfg.command_max_yaw_rate,
        )
        commands[arrived, :] = 0.0
        return commands

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        reward_dict = self._get_reward(state.data, state.info)
        scales = self.cfg.reward_config.scales
        rewards = {k: v * scales.get(k, 0.0) for k, v in reward_dict.items()}
        nav_pressure_keys = {"progress_to_target", "low_speed_penalty"}
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
        result["progress_to_target"] = self._reward_progress_to_target(info)
        result["heading_to_target"] = self._reward_heading_to_target(data, info)
        result["low_speed_penalty"] = self._reward_low_speed_penalty(data, info)
        result["arrival"] = self._reward_arrival(info)
        result["arrival_bonus"] = self._reward_arrival_bonus(info)
        result["stop_at_target"] = self._reward_stop_at_target(data, info)
        return result

    def _reward_progress_to_target(self, info: dict) -> np.ndarray:
        progress_vel = (info["prev_target_dist"] - info["target_dist"]) / self.cfg.ctrl_dt
        return np.clip(progress_vel, -0.5, 0.8).astype(np.float32)

    def _reward_heading_to_target(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        target_rel = self._target_relative_pos(data, info["target_pos"])
        heading_error = np.arctan2(target_rel[:, 1], target_rel[:, 0])
        heading_score = np.clip((np.cos(heading_error) + 1.0) * 0.5, 0.0, 1.0)
        far_from_target = info["target_dist"] > self.cfg.point_config.arrival_radius
        return (heading_score * far_from_target).astype(np.float32)

    def _reward_low_speed_penalty(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """Smooth low-speed deficit when the target is ahead and still far away."""
        target_rel = self._target_relative_pos(data, info["target_pos"])
        heading = np.arctan2(target_rel[:, 1], target_rel[:, 0])
        target_in_front = np.cos(heading) > 0.5
        forward_vel = self.get_local_linvel(data)[:, 0]
        far_from_target = info["target_dist"] > self.cfg.point_config.arrival_radius
        deficit = np.clip((0.12 - forward_vel) / 0.12, 0.0, 1.0)
        return (deficit * far_from_target * target_in_front).astype(np.float32)

    def _reward_arrival_bonus(self, info: dict) -> np.ndarray:
        """One-time bonus when the robot first crosses the arrival threshold."""
        radius = self.cfg.point_config.arrival_radius
        just_arrived = (info["prev_target_dist"] > radius) & (info["target_dist"] <= radius)
        return just_arrived.astype(np.float32)

    def _reward_arrival(self, info: dict) -> np.ndarray:
        return (info["target_dist"] <= self.cfg.point_config.arrival_radius).astype(np.float32)

    def _reward_stop_at_target(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        local_speed = np.linalg.norm(self.get_local_linvel(data)[:, :2], axis=1)
        stopped = local_speed <= self.cfg.point_config.stop_speed_threshold
        arrived = info["target_dist"] <= self.cfg.point_config.arrival_radius
        return (arrived & stopped).astype(np.float32)

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        state = super().update_terminated(state)
        arrived = state.info["target_dist"] <= self.cfg.point_config.arrival_radius
        fall_done = state.terminated.copy()
        state.info["fall_done"] = fall_done
        state.info["arrived_done"] = arrived & ~fall_done
        return state.replace(terminated=fall_done | arrived)
