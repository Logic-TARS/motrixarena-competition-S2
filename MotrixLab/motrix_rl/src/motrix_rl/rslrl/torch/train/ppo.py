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

"""PPO Trainer for RSLRL integration."""

import logging

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry as env_registry
from motrix_rl import registry as rl_registry
from motrix_rl import utils
from motrix_rl.rslrl.cfg import RslrlCfg
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap
from motrix_rl.skrl import get_log_dir

logger = logging.getLogger(__name__)


class Trainer:
    """RSLRL PPO Trainer.

    This class wraps RSLRL's OnPolicyRunner to provide a training interface
    consistent with the SKRL trainer implementation.
    """

    _env_name: str
    _sim_backend: str
    _rlcfg: RslrlCfg
    _enable_render: bool
    _resume_policy: str | None
    _resume_noise_std: float | None

    def __init__(
        self,
        env_name: str,
        sim_backend: str = None,
        enable_render: bool = False,
        cfg_override: dict = None,
        resume_policy: str | None = None,
        resume_noise_std: float | None = None,
    ) -> None:
        """Initialize the RSLRL PPO trainer.

        Args:
            env_name: Name of the environment to train
            sim_backend: Simulation backend to use (e.g., "mujoco", "npcm")
            enable_render: Whether to enable rendering during training
            cfg_override: Optional configuration overrides
        """
        rlcfg = rl_registry.default_rl_cfg(env_name, "rslrl", backend="torch")
        if cfg_override is not None:
            rlcfg = utils.cfg_override(rlcfg, cfg_override)
        self._rlcfg = rlcfg
        self._env_name = env_name
        self._sim_backend = sim_backend
        self._enable_render = enable_render
        self._resume_policy = resume_policy
        self._resume_noise_std = resume_noise_std

    def train(self) -> None:
        """Start training the agent.

        Creates the environment, wraps it for RSLRL, and runs the training loop.
        """
        rlcfg = self._rlcfg

        # Create environment
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.num_envs)

        # Set random seed
        if rlcfg.runner.seed is not None:
            torch.manual_seed(rlcfg.runner.seed)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Wrap environment for RSLRL
        vec_env = RslrlNpEnvWrap(env, device)

        # Create RSLRL config - use to_dict() method
        rslrl_cfg = self._create_rslrl_config()

        # Create RSLRL runner
        runner = OnPolicyRunner(
            vec_env, rslrl_cfg, log_dir=get_log_dir(self._env_name, rllib="rslrl", agent_name="PPO"), device=device
        )
        if self._resume_policy:
            logger.info(f"Resuming training from {self._resume_policy}")
            self._load_resume_policy(runner, str(device))
            if self._resume_noise_std is not None:
                if not hasattr(runner.alg.actor, "std"):
                    raise AttributeError("Loaded RSLRL actor does not expose a trainable std parameter")
                runner.alg.actor.std.data.fill_(self._resume_noise_std)
                logger.info(f"Reset actor exploration noise std to {self._resume_noise_std}")

        # Start training
        logger.info(f"Starting training for {self._env_name}")
        logger.info(f"Number of environments: {rlcfg.num_envs}")

        # Get max_iterations from config
        total_iterations = rslrl_cfg["max_iterations"]
        logger.info(f"Number of learning iterations: {total_iterations}")

        runner.learn(num_learning_iterations=total_iterations)

        logger.info("Training completed")

    def play(
        self,
        policy_path: str,
        log_state: bool = False,
        log_state_every: int = 30,
        log_state_envs: int = 4,
    ) -> None:
        """Evaluate a trained policy.

        Args:
            policy_path: Path to the saved policy file
            log_state: Whether to print robot state while playing
            log_state_every: Print robot state every N control steps
            log_state_envs: Number of vectorized environments to print
        """
        import time

        rlcfg = self._rlcfg

        # Create environment with play_num_envs
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.play_num_envs)

        # Set random seed
        if rlcfg.runner.seed is not None:
            torch.manual_seed(rlcfg.runner.seed)

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Wrap environment for RSLRL
        vec_env = RslrlNpEnvWrap(env, device)

        # Create RSLRL config (minimal for evaluation)
        rslrl_cfg = self._create_rslrl_config()

        # Create RSLRL runner with log_dir=None to disable logging (no git diff storage in play mode)
        runner = OnPolicyRunner(vec_env, rslrl_cfg, log_dir=None, device=device)

        # Load policy
        logger.info(f"Loading policy from {policy_path}")
        runner.load(policy_path)

        # Run evaluation loop
        logger.info("Starting evaluation loop...")
        logger.info("Press Ctrl+C to stop")
        obs, _ = vec_env.reset()
        policy = runner.get_inference_policy(device=device)
        if hasattr(policy, "reset"):
            policy.reset()
        fps = 60
        step = 0
        log_state_every = max(log_state_every, 1)
        log_state_envs = max(log_state_envs, 1)

        try:
            while True:
                t = time.time()

                # Get actions from policy
                with torch.no_grad():
                    actions = policy(obs)

                # Step environment
                obs, rewards, dones, infos = vec_env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones.bool())
                step += 1

                if log_state and step % log_state_every == 0:
                    logger.info(self._format_env_state(vec_env, rewards, dones, step, log_state_envs))

                # Render the environment
                vec_env.render()

                delta_time = time.time() - t
                if delta_time < 1.0 / fps:
                    time.sleep(1.0 / fps - delta_time)

        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")

    def _format_env_state(
        self,
        vec_env: RslrlNpEnvWrap,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        step: int,
        max_envs: int,
    ) -> str:
        from motrix_envs.math import quaternion

        env = vec_env._env
        state = vec_env._state
        if state is None:
            return f"[state step={step}] env state is not initialized"

        pose = env._body.get_pose(state.data)
        quat = pose[:, 3:7]
        roll, pitch, yaw = quaternion.get_euler_xyz(quat)
        rpy = np.stack([roll, pitch, yaw], axis=1)
        local_linvel = env.get_local_linvel(state.data)
        gyro = env.get_gyro(state.data)
        dof_pos = env.get_dof_pos(state.data)
        dof_vel = env.get_dof_vel(state.data)
        commands = state.info.get("commands")
        left_contact = state.info.get("left_contact")
        right_contact = state.info.get("right_contact")
        termination_too_low = state.info.get("termination_too_low")
        termination_too_tilted = state.info.get("termination_too_tilted")
        feet_pos = state.info.get("feet_pos")
        feet_vel = state.info.get("feet_vel")
        torque_limit = getattr(env, "torque_limits", None)
        if torque_limit is None:
            torque_limit = getattr(env.cfg.control_config, "torque_limit", None)
        torque_saturation = None
        if torque_limit is not None and np.all(np.asarray(torque_limit) > 0):
            torque_saturation = np.mean(np.abs(state.data.actuator_ctrls) >= 0.95 * torque_limit, axis=1)

        rewards_np = rewards.detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy().astype(bool)
        count = min(max_envs, env.num_envs)
        lines = [f"[state step={step} showing={count}/{env.num_envs}]"]

        for idx in range(count):
            parts = [
                f"env[{idx}]",
                f"pos={self._fmt_vec(pose[idx, :3])}",
                f"rpy={self._fmt_vec(rpy[idx])}",
                f"quat={self._fmt_vec(quat[idx])}",
                f"linvel_local={self._fmt_vec(local_linvel[idx])}",
                f"gyro={self._fmt_vec(gyro[idx])}",
                f"reward={rewards_np[idx]:.4f}",
                f"done={bool(dones_np[idx])}",
            ]
            if commands is not None:
                parts.append(f"cmd={self._fmt_vec(commands[idx])}")
            if left_contact is not None and right_contact is not None:
                parts.append(f"contact=({int(left_contact[idx])},{int(right_contact[idx])})")
            if termination_too_low is not None or termination_too_tilted is not None:
                too_low = int(termination_too_low[idx]) if termination_too_low is not None else 0
                too_tilted = int(termination_too_tilted[idx]) if termination_too_tilted is not None else 0
                parts.append(f"term=(low:{too_low},tilt:{too_tilted})")
            if torque_saturation is not None:
                parts.append(f"torque_sat={float(torque_saturation[idx]):.3f}")
            if feet_pos is not None:
                parts.append(f"feet_pos={self._fmt_vec(feet_pos[idx].reshape(-1))}")
            if feet_vel is not None:
                parts.append(f"feet_vel={self._fmt_vec(feet_vel[idx].reshape(-1))}")
            parts.append(f"dof_pos_mean={float(np.mean(dof_pos[idx])):.4f}")
            parts.append(f"dof_vel_rms={float(np.sqrt(np.mean(np.square(dof_vel[idx])))):.4f}")
            lines.append(" ".join(parts))

        return "\n".join(lines)

    @staticmethod
    def _fmt_vec(values: np.ndarray, precision: int = 3) -> str:
        return np.array2string(np.asarray(values), precision=precision, suppress_small=True, separator=",")

    def _create_rslrl_config(self) -> dict:
        return self._rlcfg.runner.to_dict()

    def _load_resume_policy(self, runner: OnPolicyRunner, map_location: str) -> None:
        loaded_dict = torch.load(self._resume_policy, weights_only=False, map_location=map_location)
        loaded_dict["actor_state_dict"] = self._expand_state_dict(
            loaded_dict["actor_state_dict"], runner.alg.actor.state_dict()
        )
        loaded_dict["critic_state_dict"] = self._expand_state_dict(
            loaded_dict["critic_state_dict"], runner.alg.critic.state_dict()
        )
        runner.alg.load(
            loaded_dict,
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            strict=False,
        )
        logger.info("Loaded actor/critic warm start; skipped optimizer and checkpoint iteration.")

    @staticmethod
    def _expand_state_dict(source_state: dict, target_state: dict) -> dict:
        expanded = {key: value.clone() for key, value in target_state.items()}
        for key, source_value in source_state.items():
            if key not in expanded:
                continue
            target_value = expanded[key]
            if source_value.shape == target_value.shape:
                expanded[key] = source_value
                continue
            if source_value.ndim == target_value.ndim == 1:
                copy_len = min(source_value.shape[0], target_value.shape[0])
                target_value[:copy_len] = source_value[:copy_len]
            elif source_value.ndim == target_value.ndim == 2:
                rows = min(source_value.shape[0], target_value.shape[0])
                cols = min(source_value.shape[1], target_value.shape[1])
                target_value[:rows, :cols] = source_value[:rows, :cols]
            expanded[key] = target_value
        return expanded
