"""Full-body K1 autonomous get-up environment."""

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.k1.getup_cfg import (
    K1_FULL_BODY_JOINTS,
    K1GetupNpEnvCfg,
)
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState


POSE_SUPINE = 0
POSE_PRONE = 1
POSE_LEFT = 2
POSE_RIGHT = 3


@registry.env("k1-getup", sim_backend="np")
class K1GetupTask(NpEnv):
    """Recover from fallen poses without modifying the base after reset."""

    def __init__(self, cfg: K1GetupNpEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        if self._model.num_actuators != 22:
            raise ValueError(f"k1-getup requires 22 actuators, got {self._model.num_actuators}")
        self._mjcf_joint_names = list(self._model.actuator_names)
        if set(self._mjcf_joint_names) != set(K1_FULL_BODY_JOINTS):
            raise ValueError("K1 full-body joint names do not match the runtime policy")
        self._actuator_indices = np.array(
            [self._mjcf_joint_names.index(name) for name in K1_FULL_BODY_JOINTS],
            dtype=np.int64,
        )
        self._action_space = gym.spaces.Box(-1.0, 1.0, (22,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (78,), dtype=np.float32)
        self._body = self._model.get_body("Trunk")
        self._joint_pos_indices, self._joint_vel_indices = self._actuated_indices()
        self._init_pos = self._model.compute_init_dof_pos()
        self._init_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        self._default_angles = self._make_pose(
            {
                "Left_Shoulder_Roll": -1.3,
                "Right_Shoulder_Roll": 1.3,
            }
        )
        self._key_poses = self._build_key_poses()
        self._action_scale = np.array([self._joint_action_scale(name) for name in K1_FULL_BODY_JOINTS], dtype=np.float32)
        self._kps = np.array([self._joint_kp(name) for name in K1_FULL_BODY_JOINTS], dtype=np.float32)
        self._kds = np.array([self._joint_kd(name) for name in K1_FULL_BODY_JOINTS], dtype=np.float32)
        self._torque_limits = np.array([self._joint_effort(name) for name in K1_FULL_BODY_JOINTS], dtype=np.float32)
        self._reset_count = 0
        self.curriculum_level = 0
        self.total_attempts = 0
        self.total_successes = 0
        self.success_by_pose = np.zeros((4,), dtype=np.int64)
        self.attempts_by_pose = np.zeros((4,), dtype=np.int64)
        self.gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _actuated_indices(self) -> tuple[np.ndarray, np.ndarray]:
        pos_mjcf = np.asarray(self._body.get_dof_pos_indices(), dtype=np.int64).reshape(-1)[-22:]
        vel_mjcf = np.asarray(self._body.get_dof_vel_indices(), dtype=np.int64).reshape(-1)[-22:]
        return pos_mjcf[self._actuator_indices], vel_mjcf[self._actuator_indices]

    @staticmethod
    def _joint_action_scale(name: str) -> float:
        if "Head" in name:
            return 0.375
        if "Shoulder" in name or "Elbow" in name:
            return 0.875
        if "Hip_Pitch" in name:
            return 0.09375
        if "Hip_Roll" in name:
            return 0.109375
        if "Hip_Yaw" in name:
            return 0.0625
        if "Knee_Pitch" in name:
            return 0.125
        return 1.0 / 6.0

    @staticmethod
    def _joint_kp(name: str) -> float:
        if "Head" in name or "Shoulder" in name or "Elbow" in name:
            return 4.0
        if "Ankle" in name:
            return 30.0
        return 80.0

    @staticmethod
    def _joint_kd(name: str) -> float:
        if "Head" in name or "Shoulder" in name or "Elbow" in name:
            return 1.0
        return 2.0

    @staticmethod
    def _joint_effort(name: str) -> float:
        if "Head" in name:
            return 6.0
        if "Shoulder" in name or "Elbow" in name:
            return 14.0
        if "Hip_Pitch" in name:
            return 30.0
        if "Hip_Roll" in name:
            return 35.0
        if "Hip_Yaw" in name or "Ankle" in name:
            return 20.0
        return 40.0

    def _make_pose(self, values: dict[str, float]) -> np.ndarray:
        return np.array([values.get(name, 0.0) for name in K1_FULL_BODY_JOINTS], dtype=np.float32)

    def _build_key_poses(self) -> np.ndarray:
        tuck = self._make_pose(
            {
                "ALeft_Shoulder_Pitch": -1.2,
                "ARight_Shoulder_Pitch": -1.2,
                "Left_Shoulder_Roll": -0.6,
                "Right_Shoulder_Roll": 0.6,
                "Left_Hip_Pitch": -1.0,
                "Right_Hip_Pitch": -1.0,
                "Left_Knee_Pitch": 1.8,
                "Right_Knee_Pitch": 1.8,
                "Left_Ankle_Pitch": -0.5,
                "Right_Ankle_Pitch": -0.5,
            }
        )
        support = self._make_pose(
            {
                "ALeft_Shoulder_Pitch": -1.5,
                "ARight_Shoulder_Pitch": -1.5,
                "Left_Shoulder_Roll": -0.9,
                "Right_Shoulder_Roll": 0.9,
                "Left_Elbow_Pitch": -1.0,
                "Right_Elbow_Pitch": -1.0,
                "Left_Hip_Pitch": -0.8,
                "Right_Hip_Pitch": -0.8,
                "Left_Knee_Pitch": 1.5,
                "Right_Knee_Pitch": 1.5,
                "Left_Ankle_Pitch": -0.45,
                "Right_Ankle_Pitch": -0.45,
            }
        )
        crouch = self._make_pose(
            {
                "Left_Shoulder_Roll": -1.1,
                "Right_Shoulder_Roll": 1.1,
                "Left_Hip_Pitch": -0.55,
                "Right_Hip_Pitch": -0.55,
                "Left_Knee_Pitch": 1.0,
                "Right_Knee_Pitch": 1.0,
                "Left_Ankle_Pitch": -0.45,
                "Right_Ankle_Pitch": -0.45,
            }
        )
        return np.stack([tuck, support, crouch, self._default_angles], axis=0)

    def get_joint_pos(self, data: mtx.SceneData) -> np.ndarray:
        return data.dof_pos[:, self._joint_pos_indices]

    def get_joint_vel(self, data: mtx.SceneData) -> np.ndarray:
        return data.dof_vel[:, self._joint_vel_indices]

    def _base_features(self, data: mtx.SceneData):
        pose = self._body.get_pose(data)
        quat = pose[:, 3:7]
        root_vel = data.dof_vel[:, :6]
        local_lin = quaternion.rotate_inverse(quat, root_vel[:, :3])
        local_ang = quaternion.rotate_inverse(quat, root_vel[:, 3:6])
        local_gravity = quaternion.rotate_inverse(quat, self.gravity)
        return pose, local_lin, local_ang, local_gravity

    def _observation(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        _, local_lin, local_ang, local_gravity = self._base_features(data)
        pose_class = np.minimum(info["fall_pose"], POSE_LEFT)
        pose_one_hot = np.eye(3, dtype=np.float32)[pose_class]
        obs = np.concatenate(
            [
                local_lin,
                local_ang * 0.2,
                local_gravity,
                pose_one_hot,
                self.get_joint_pos(data) - self._default_angles,
                self.get_joint_vel(data) * 0.05,
                info["current_actions"],
            ],
            axis=1,
        )
        return np.clip(np.nan_to_num(obs), -100.0, 100.0).astype(np.float32)

    def get_privileged_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        return self._observation(data, info)

    def apply_action(self, actions, state: NpEnvState) -> NpEnvState:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = np.tile(actions, (self._num_envs, 1))
        actions = np.clip(actions, -1.0, 1.0)
        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = actions
        state.info["episode_length"] += 1
        target = self._default_angles + actions * self._action_scale
        kp = self._kps[None, :] * state.info["motor_strength"][:, None]
        kd = self._kds[None, :] * state.info["damping_scale"][:, None]
        torque = kp * (target - self.get_joint_pos(state.data)) - kd * self.get_joint_vel(state.data)
        state.data.actuator_ctrls[:, self._actuator_indices] = np.clip(
            torque, -self._torque_limits, self._torque_limits
        )
        return state

    def _maybe_push(self, state: NpEnvState) -> None:
        if self.curriculum_level < 4:
            return
        interval = max(int(self.cfg.domain_rand.push_interval_s / self.cfg.ctrl_dt), 1)
        mask = (state.info["episode_length"] % interval) == 0
        if not np.any(mask):
            return
        vel = state.data.dof_vel.copy()
        vel[mask, :2] += np.random.uniform(
            -self.cfg.domain_rand.max_push_velocity,
            self.cfg.domain_rand.max_push_velocity,
            size=(int(mask.sum()), 2),
        )
        state.data.set_dof_vel(vel)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._maybe_push(state)
        pose, _, local_ang, gravity = self._base_features(state.data)
        tilt = np.linalg.norm(gravity[:, :2], axis=1)
        stable = (
            (pose[:, 2] >= self.cfg.reward_config.success_height)
            & (tilt <= self.cfg.reward_config.success_tilt)
            & (np.linalg.norm(local_ang, axis=1) <= self.cfg.reward_config.success_ang_vel)
        )
        state.info["stable_steps"] = np.where(stable, state.info["stable_steps"] + 1, 0)
        hold_steps = max(int(self.cfg.reward_config.success_hold_seconds / self.cfg.ctrl_dt), 1)
        success = state.info["stable_steps"] >= hold_steps
        newly_successful = success & ~state.info["success_recorded"]
        if np.any(newly_successful):
            poses = state.info["fall_pose"][newly_successful]
            self.total_successes += int(newly_successful.sum())
            for pose_id in range(4):
                self.success_by_pose[pose_id] += int(np.sum(poses == pose_id))
            state.info["success_recorded"][newly_successful] = True

        stage = np.minimum(
            (state.info["episode_length"] * 4) // max(self.cfg.max_episode_steps, 1),
            3,
        )
        target = self._key_poses[stage]
        joint_error = np.mean(np.square(self.get_joint_pos(state.data) - target), axis=1)
        cfg = self.cfg.reward_config
        reward = (
            cfg.upright * np.square(np.clip(-gravity[:, 2], 0.0, 1.0))
            + cfg.height * np.clip(pose[:, 2] / cfg.target_height, 0.0, 1.0)
            + cfg.target_pose * np.exp(-2.0 * joint_error)
            + cfg.stability * np.exp(-np.sum(np.square(local_ang), axis=1))
            + cfg.action_rate
            * np.sum(np.square(state.info["current_actions"] - state.info["last_actions"]), axis=1)
            + cfg.torque * np.sum(np.square(state.data.actuator_ctrls), axis=1)
        ) * self.cfg.ctrl_dt
        reward += newly_successful.astype(np.float32) * cfg.success_bonus
        state.info["success"] = success.astype(np.float32)
        return state.replace(
            obs=self._observation(state.data, state.info),
            reward=reward.astype(np.float32),
            terminated=success,
        )

    def _sample_fall_pose(self, num_reset: int) -> np.ndarray:
        if self.curriculum_level == 0:
            return np.full((num_reset,), POSE_SUPINE, dtype=np.int32)
        if self.curriculum_level == 1:
            return np.random.choice([POSE_SUPINE, POSE_PRONE], size=num_reset).astype(np.int32)
        if self.curriculum_level == 2:
            return np.random.choice([POSE_LEFT, POSE_RIGHT], size=num_reset).astype(np.int32)
        return np.random.randint(0, 4, size=num_reset, dtype=np.int32)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]
        data.reset(self._model)
        self._reset_count += num_reset
        self.curriculum_level = min(
            self._reset_count // max(self.cfg.reset_config.curriculum_resets_per_stage, 1),
            4,
        )
        fall_pose = self._sample_fall_pose(num_reset)
        self.total_attempts += num_reset
        for pose_id in range(4):
            self.attempts_by_pose[pose_id] += int(np.sum(fall_pose == pose_id))

        qpos = np.tile(self._init_pos, (num_reset, 1))
        qvel = np.tile(self._init_vel, (num_reset, 1))
        qpos[:, :2] = 0.0
        qpos[:, 2] = self.cfg.reset_config.base_height
        roll = np.zeros((num_reset,), dtype=np.float32)
        pitch = np.zeros((num_reset,), dtype=np.float32)
        roll[fall_pose == POSE_LEFT] = np.pi / 2
        roll[fall_pose == POSE_RIGHT] = -np.pi / 2
        pitch[fall_pose == POSE_SUPINE] = -np.pi / 2
        pitch[fall_pose == POSE_PRONE] = np.pi / 2
        if self.curriculum_level >= 3:
            roll += np.random.uniform(-0.25, 0.25, size=num_reset)
            pitch += np.random.uniform(-0.25, 0.25, size=num_reset)
        yaw = np.random.uniform(-np.pi, np.pi, size=num_reset)
        qpos[:, 3:7] = quaternion.from_euler(roll, pitch, yaw)
        qpos[:, self._joint_pos_indices] = self._default_angles
        qpos[:, self._joint_pos_indices] += np.random.uniform(
            -self.cfg.reset_config.joint_noise,
            self.cfg.reset_config.joint_noise,
            size=(num_reset, 22),
        )
        qvel[:, :3] = np.random.uniform(
            -self.cfg.reset_config.linear_velocity,
            self.cfg.reset_config.linear_velocity,
            size=(num_reset, 3),
        )
        qvel[:, 3:6] = np.random.uniform(
            -self.cfg.reset_config.angular_velocity,
            self.cfg.reset_config.angular_velocity,
            size=(num_reset, 3),
        )
        data.set_dof_vel(qvel)
        data.set_dof_pos(qpos, self._model)
        self._model.forward_kinematic(data)
        if self.curriculum_level >= 4:
            motor_strength = np.random.uniform(
                *self.cfg.domain_rand.motor_strength_range,
                size=num_reset,
            ).astype(np.float32)
            damping_scale = np.random.uniform(
                *self.cfg.domain_rand.damping_scale_range,
                size=num_reset,
            ).astype(np.float32)
        else:
            motor_strength = np.ones((num_reset,), dtype=np.float32)
            damping_scale = np.ones((num_reset,), dtype=np.float32)
        info = {
            "current_actions": np.zeros((num_reset, 22), dtype=np.float32),
            "last_actions": np.zeros((num_reset, 22), dtype=np.float32),
            "episode_length": np.zeros((num_reset,), dtype=np.int32),
            "stable_steps": np.zeros((num_reset,), dtype=np.int32),
            "fall_pose": fall_pose,
            "success": np.zeros((num_reset,), dtype=np.float32),
            "success_recorded": np.zeros((num_reset,), dtype=bool),
            "curriculum_level": np.full((num_reset,), self.curriculum_level, dtype=np.int32),
            "motor_strength": motor_strength,
            "damping_scale": damping_scale,
        }
        return self._observation(data, info), info
