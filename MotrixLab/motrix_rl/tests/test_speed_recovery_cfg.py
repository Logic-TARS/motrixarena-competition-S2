"""Tests for the k1-flat-terrain-walk-speed training variant."""

import importlib.util
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest

from motrix_envs import registry
from motrix_envs.locomotion.k1.cfg import (
    Commands,
    K1WalkNpEnvCfg,
    K1WalkSpeedEnvCfg,
)
from motrix_envs.locomotion.k1.walk_np import K1WalkTask


def _make_env(cfg=None, num_envs=4):
    return K1WalkTask(cfg or K1WalkSpeedEnvCfg(), num_envs=num_envs)


def _load_eval_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_k1_walk_grid.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("eval_k1_walk_grid", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSpeedEnvironment:
    def test_registry_constructs_speed_environment(self):
        env = registry.make(
            "k1-flat-terrain-walk-speed",
            sim_backend="np",
            num_envs=1,
        )
        assert isinstance(env, K1WalkTask)
        assert isinstance(env.cfg, K1WalkSpeedEnvCfg)

    def test_original_environment_still_constructs(self):
        env = registry.make(
            "k1-flat-terrain-walk",
            sim_backend="np",
            num_envs=1,
        )
        assert isinstance(env.cfg, K1WalkNpEnvCfg)
        assert env.cfg.commands.apply_forward_yaw_envelope is True


class TestSpeedConfiguration:
    def setup_method(self):
        self.cfg = K1WalkSpeedEnvCfg()
        self.cmd = self.cfg.commands

    def test_sampling_breakdown(self):
        assert self.cmd.stand_probability == pytest.approx(0.10)
        assert self.cmd.straight_probability == pytest.approx(0.55)
        assert self.cmd.turn_probability == pytest.approx(0.10)
        assert self.cmd.mixed_turn_probability == pytest.approx(0.05)
        assert self.cmd.direction_change_probability == 0.0
        assert self.cmd.sprint_turn_probability == 0.0
        configured = sum(
            (
                self.cmd.stand_probability,
                self.cmd.straight_probability,
                self.cmd.turn_probability,
                self.cmd.mixed_turn_probability,
                self.cmd.direction_change_probability,
                self.cmd.sprint_turn_probability,
            )
        )
        assert 1.0 - configured == pytest.approx(0.20)

    def test_command_ranges_and_hold(self):
        assert self.cmd.lin_vel_x == [0.0, 1.0]
        assert self.cmd.lin_vel_y == [-0.2, 0.2]
        assert self.cmd.ang_vel_yaw == [-1.0, 1.0]
        assert self.cmd.straight_vx_range == [0.65, 1.0]
        assert self.cmd.resampling_time == 10.0
        assert self.cmd.lin_vel_x[0] == 0.0

    def test_speed_rewards(self):
        scales = self.cfg.reward_config.scales
        assert scales["tracking_lin_vel"] == 2.0
        assert scales["tracking_ang_vel"] == 0.5
        assert self.cfg.reward_config.tracking_sigma == 0.25
        assert scales["command_forward_vel"] == 0.5
        assert scales["straight_motion"] == -1.0
        assert scales["overspeed"] == -0.3
        for key in (
            "turn_stability",
            "turn_survival",
            "direction_change_tracking",
            "sprint_stability",
        ):
            assert scales[key] == 0.0
        assert scales["termination"] == -10.0
        assert scales["base_height"] == -10.0
        assert scales["collision"] == -1.0

    def test_turn_curricula_and_envelope_are_disabled(self):
        assert self.cmd.apply_forward_yaw_envelope is False
        assert all(value == 0 for value in self.cmd.turn_sustain_curriculum)
        assert self.cmd.direction_change_period_steps == 0


class TestRealSampling:
    def test_each_mode_changes_real_commands(self):
        cfg = K1WalkSpeedEnvCfg()
        env = _make_env(cfg, num_envs=7)
        mode_values = np.array([0.05, 0.20, 0.70, 0.77, 0.82, 0.90, 0.99])
        uniform_values = [
            np.full(7, 0.5),
            np.zeros(7),
            np.full(7, 0.2),
            np.full(1, 0.8),
            np.full(1, 0.3),
            np.full(1, 0.9),
        ]
        with (
            mock.patch("numpy.random.rand", return_value=mode_values),
            mock.patch("numpy.random.uniform", side_effect=uniform_values),
            mock.patch("numpy.random.choice", return_value=np.ones(1)),
        ):
            commands = env.resample_commands(7)

        assert np.allclose(commands[0], 0.0)
        assert commands[1, 0] == pytest.approx(0.8)
        assert commands[1, 1] == 0.0
        assert commands[1, 2] == 0.0
        assert np.allclose(commands[2, :2], 0.0)
        assert commands[3, 0] == pytest.approx(0.3)
        assert commands[3, 1] == 0.0
        assert commands[4, 0] == pytest.approx(0.5)

    def test_speed_sampling_distribution(self):
        np.random.seed(123)
        env = _make_env(num_envs=100_000)
        commands = env.resample_commands(100_000)
        modes = env._last_command_modes

        assert np.all(commands[:, 0] >= 0.0)
        assert np.mean(modes["stand"]) == pytest.approx(0.10, abs=0.01)
        assert np.mean(modes["straight"]) == pytest.approx(0.55, abs=0.01)
        assert np.all(commands[modes["straight"], 0] >= 0.65)
        high_speed_straight = (commands[:, 0] >= 0.9) & (
            np.abs(commands[:, 2]) < 0.01
        )
        assert np.mean(high_speed_straight) >= 0.15

    def test_envelope_is_skipped_for_speed_environment(self):
        cfg = K1WalkSpeedEnvCfg()
        cfg.commands.stand_probability = 0.0
        cfg.commands.straight_probability = 0.0
        cfg.commands.turn_probability = 0.0
        cfg.commands.mixed_turn_probability = 0.0
        env = _make_env(cfg, num_envs=1)
        with (
            mock.patch("numpy.random.uniform", side_effect=[
                np.array([1.0]),
                np.array([0.0]),
                np.array([1.0]),
            ]),
            mock.patch("numpy.random.rand", return_value=np.array([0.99])),
        ):
            command = env.resample_commands(1)[0]
        assert command[0] == pytest.approx(1.0)
        assert command[2] == pytest.approx(1.0)

    def test_base_environment_applies_envelope(self):
        cfg = K1WalkNpEnvCfg()
        cfg.commands = Commands(
            lin_vel_x=[0.0, 1.0],
            lin_vel_y=[-0.35, 0.35],
            yaw_curriculum=[1.5],
            stand_probability=0.0,
            straight_probability=0.0,
            turn_probability=0.0,
            mixed_turn_probability=0.0,
            direction_change_probability=0.0,
            sprint_turn_probability=0.0,
            apply_forward_yaw_envelope=True,
        )
        env = _make_env(cfg, num_envs=1)
        with (
            mock.patch("numpy.random.uniform", side_effect=[
                np.array([1.0]),
                np.array([0.0]),
                np.array([1.0]),
            ]),
            mock.patch("numpy.random.rand", return_value=np.array([0.99])),
        ):
            command = env.resample_commands(1)[0]
        assert command[0] < 1.0
        assert command[2] == pytest.approx(1.0)


class TestCommandHold:
    def test_no_resample_before_ten_seconds_and_resamples_at_boundary(self):
        env = _make_env(num_envs=1)
        info = {
            "episode_length": np.array([499], dtype=np.int32),
            "commands": np.array([[0.8, 0.0, 0.0]], dtype=np.float32),
            "turn_sustain_counter": np.zeros(1, dtype=np.int32),
            "direction_change_counter": np.zeros(1, dtype=np.int32),
        }
        with mock.patch.object(env, "resample_commands") as resample:
            env._maybe_resample_commands(info)
            resample.assert_not_called()

            info["episode_length"][:] = 500
            resample.return_value = np.array([[0.9, 0.0, 0.0]], dtype=np.float32)
            env._maybe_resample_commands(info)
            resample.assert_called_once_with(1)
            assert info["commands"][0, 0] == pytest.approx(0.9)


class TestEvaluationCommands:
    def test_speed_evaluation_defaults_to_raw_commands(self):
        module = _load_eval_module()
        cases = module.command_grid(False, 1.0)
        raw = next(case for case in cases if case["raw_command"] == [1.0, 0.0, 1.0])
        assert raw["command"] == pytest.approx([1.0, 0.0, 1.0])
        assert raw["diagnostic"] is False

    def test_enveloped_evaluation_reduces_high_yaw_forward_speed(self):
        module = _load_eval_module()
        cases = module.command_grid(True, 1.0)
        case = next(item for item in cases if item["raw_command"] == [1.0, 0.0, 1.0])
        assert case["command"][0] < 1.0

    def test_grid_contains_full_speed_straight_cases(self):
        module = _load_eval_module()
        cases = module.command_grid(False, 1.0)
        commands = [case["command"] for case in cases]
        assert any(command == pytest.approx([0.9, 0.0, 0.0]) for command in commands)
        assert any(command == pytest.approx([1.0, 0.0, 0.0]) for command in commands)
