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

from dataclasses import dataclass, field
from pathlib import Path

from motrix_envs import registry
from motrix_envs.base import EnvCfg


REPO_ROOT = Path(__file__).resolve().parents[6]
K1_ASSET_DIR = REPO_ROOT / "legged_gym" / "resources" / "robots" / "K1"


@dataclass
class ControlConfig:
    stiffness: dict[str, float] = field(
        default_factory=lambda: {
            "Left_Hip_Pitch": 100.0,
            "Right_Hip_Pitch": 100.0,
            "Left_Hip_Roll": 100.0,
            "Right_Hip_Roll": 100.0,
            "Left_Hip_Yaw": 100.0,
            "Right_Hip_Yaw": 100.0,
            "Left_Knee_Pitch": 150.0,
            "Right_Knee_Pitch": 150.0,
            "Left_Ankle_Pitch": 40.0,
            "Right_Ankle_Pitch": 40.0,
            "Left_Ankle_Roll": 40.0,
            "Right_Ankle_Roll": 40.0,
        }
    )
    damping: dict[str, float] = field(
        default_factory=lambda: {
            "Left_Hip_Pitch": 2.0,
            "Right_Hip_Pitch": 2.0,
            "Left_Hip_Roll": 2.0,
            "Right_Hip_Roll": 2.0,
            "Left_Hip_Yaw": 2.0,
            "Right_Hip_Yaw": 2.0,
            "Left_Knee_Pitch": 4.0,
            "Right_Knee_Pitch": 4.0,
            "Left_Ankle_Pitch": 2.0,
            "Right_Ankle_Pitch": 2.0,
            "Left_Ankle_Roll": 2.0,
            "Right_Ankle_Roll": 2.0,
        }
    )
    action_scale: dict[str, float] = field(
        default_factory=lambda: {
            "Left_Hip_Pitch": 0.1700,
            "Right_Hip_Pitch": 0.1700,
            "Left_Hip_Roll": 0.1900,
            "Right_Hip_Roll": 0.1900,
            "Left_Hip_Yaw": 0.09575,
            "Right_Hip_Yaw": 0.09575,
            "Left_Knee_Pitch": 0.1867,
            "Right_Knee_Pitch": 0.1867,
            "Left_Ankle_Pitch": 0.2394,
            "Right_Ankle_Pitch": 0.2394,
            "Left_Ankle_Roll": 0.2394,
            "Right_Ankle_Roll": 0.2394,
        }
    )
    torque_limit: dict[str, float] = field(
        default_factory=lambda: {
            "Left_Hip_Pitch": 68.0,
            "Right_Hip_Pitch": 68.0,
            "Left_Hip_Roll": 76.0,
            "Right_Hip_Roll": 76.0,
            "Left_Hip_Yaw": 38.3,
            "Right_Hip_Yaw": 38.3,
            "Left_Knee_Pitch": 112.0,
            "Right_Knee_Pitch": 112.0,
            "Left_Ankle_Pitch": 38.3,
            "Right_Ankle_Pitch": 38.3,
            "Left_Ankle_Roll": 38.3,
            "Right_Ankle_Roll": 38.3,
        }
    )


@dataclass
class InitState:
    pos: list = field(default_factory=lambda: [0.0, 0.0, 0.57])
    joint_pos_noise: float = 0.04
    base_roll_pitch_noise: float = 0.08
    base_lin_vel_noise: float = 0.12
    base_ang_vel_noise: float = 0.20
    default_joint_angles: dict = field(default_factory=lambda: {
        "Left_Hip_Pitch": -0.2,
        "Right_Hip_Pitch": -0.2,
        "Left_Knee_Pitch": 0.4,
        "Right_Knee_Pitch": 0.4,
        "Left_Ankle_Pitch": -0.25,
        "Right_Ankle_Pitch": -0.25,
        "default": 0.0,
    })


@dataclass
class Commands:
    lin_vel_x: list = field(default_factory=lambda: [0.0, 1.0])
    lin_vel_y: list = field(default_factory=lambda: [-0.35, 0.35])
    ang_vel_yaw: list = field(default_factory=lambda: [-1.5, 1.5])
    yaw_curriculum: list = field(default_factory=lambda: [0.3, 0.6, 1.0, 1.5])
    curriculum_min_episodes: int = 2048
    curriculum_success_threshold: float = 0.82
    curriculum_ema_alpha: float = 0.05
    resampling_time: float = 3.0
    command_deadzone: float = 0.2
    phase_period: float = 0.8
    stand_probability: float = 0.10
    straight_probability: float = 0.18
    turn_probability: float = 0.10
    mixed_turn_probability: float = 0.10
    direction_change_probability: float = 0.10
    sprint_turn_probability: float = 0.10
    straight_vx_range: list | None = None
    yaw_full_speed: float = 0.25
    yaw_zero_speed: float = 1.5
    # Sustained-turn curriculum (steps at 50 Hz; 0 = off).
    # Each entry matches a yaw_curriculum level.  The policy must hold
    # high-yaw / low-vx commands for this many consecutive steps before
    # a resample is allowed.
    turn_sustain_curriculum: list = field(
        default_factory=lambda: [0, 100, 300, 600]  # 0 s, 2 s, 6 s, 12 s
    )
    # Minimum vx for mixed-turn commands so the policy always has
    # non-zero forward velocity during sustained turns (mirrors the
    # decider's continuous blending).
    mixed_turn_vx_range: list = field(default_factory=lambda: [0.05, 0.40])
    # Direction-change (agility) mode: vx > 0 with yaw sign flipping
    # every direction_change_period_steps.  Trains rapid re-orientation
    # that soccer (orbit, escape, interception) requires.
    direction_change_period_steps: int = 75  # 1.5 s @ 50 Hz
    direction_change_vx_range: list = field(default_factory=lambda: [0.05, 0.35])
    direction_change_yaw_range: list = field(default_factory=lambda: [0.6, 1.2])
    # Sprint-turn mode: high forward speed + large yaw simultaneously.
    # The hardest locomotion pattern — requires the policy to maintain
    # stability while sprinting through a sharp turn.
    sprint_turn_vx_range: list = field(default_factory=lambda: [0.50, 0.80])
    sprint_turn_yaw_range: list = field(default_factory=lambda: [0.8, 1.2])
    apply_forward_yaw_envelope: bool = True


@dataclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05


@dataclass
class NoiseScales:
    dof_pos: float = 0.01
    dof_vel: float = 1.5
    ang_vel: float = 0.2
    gravity: float = 0.05


@dataclass
class Noise:
    add_noise: bool = True
    noise_level: float = 1.2
    noise_scales: NoiseScales = field(default_factory=NoiseScales)


@dataclass
class DomainRand:
    push_robots: bool = True
    push_interval_s: float = 5.0
    max_push_vel_xy: float = 0.7


@dataclass
class Asset:
    body_name: str = "Trunk"
    foot_name: str = "Foot"
    ground_name: str = "ground"
    penalize_contacts_on: list = field(default_factory=lambda: ["Trunk", "Shank"])
    ground_geom_indices: list[int] = field(default_factory=lambda: [0])
    left_foot_geom_indices: list[int] = field(default_factory=lambda: [33])
    right_foot_geom_indices: list[int] = field(default_factory=lambda: [44])
    collision_geom_indices: list[int] = field(default_factory=lambda: [2, 3, 29, 30, 40, 41])


@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -10.0,
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 2.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.5,
            "turn_stability": -0.8,
            "turn_survival": 0.15,
            "direction_change_tracking": 1.5,
            "sprint_stability": -1.0,
            "base_height": -10.0,
            "torques": -1.0e-5,
            "dof_vel": 0.0,  # Zeroed — was -1e-3, penalises joint velocity
            "dof_acc": -2.5e-7,
            "feet_air_time": 0.0,
            "collision": -1.0,
            "action_rate": -0.01,
            "dof_pos_limits": -5.0,
            "alive": 0.05,
            "hip_pos": -1.0,
            "contact_no_vel": -0.2,
            "feet_swing_height": -20.0,
            "contact": 0.18,
            "straight_motion": -1.5,
            "command_forward_vel": 0.3,
            "overspeed": -0.3,
        }
    )
    only_positive_rewards: bool = False
    trust_contact_rewards: bool = True
    tracking_sigma: float = 0.15
    min_base_height: float = 0.45
    swing_height: float = 0.08
    target_base_height: float = 0.54
    max_tilt_xy: float = 0.55
    soft_dof_pos_limit: float = 0.9
    forward_vel_margin: float = 0.15
    forward_reward_min_height: float = 0.45
    forward_reward_full_height: float = 0.56
    forward_reward_full_tilt_xy: float = 0.40
    forward_reward_max_tilt_xy: float = 0.55
    forward_reward_min_gate: float = 0.35
    forward_reward_full_yaw_rate: float = 0.15
    forward_reward_max_yaw_rate: float = 0.8
    forward_reward_full_lateral_vel: float = 0.05
    forward_reward_max_lateral_vel: float = 0.35
    straight_motion_yaw_weight: float = 2.0
    straight_motion_lateral_weight: float = 2.0
    gait_frequency: float = 1.5
    turn_stability_tilt_threshold: float = 0.25
    turn_stability_min_yaw: float = 0.5
    direction_change_flip_window: int = 25  # 0.5 s after flip for boosted tracking reward
    sprint_stability_min_vx: float = 0.5
    sprint_stability_min_yaw: float = 0.8


@registry.envcfg("k1-flat-terrain-walk")
@dataclass
class K1WalkNpEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0
    model_file: str = str(K1_ASSET_DIR / "k1_train_scene.xml")
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    noise: Noise = field(default_factory=Noise)
    domain_rand: DomainRand = field(default_factory=DomainRand)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02


def _speed_reward_scales() -> dict[str, float]:
    return {
        "termination": -10.0,
        "tracking_lin_vel": 2.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -1.5,
        "turn_stability": 0.0,
        "turn_survival": 0.0,
        "direction_change_tracking": 0.0,
        "sprint_stability": 0.0,
        "base_height": -10.0,
        "torques": -1.0e-5,
        "dof_vel": 0.0,
        "dof_acc": -2.5e-7,
        "feet_air_time": 0.0,
        "collision": -1.0,
        "action_rate": -0.01,
        "dof_pos_limits": -5.0,
        "alive": 0.05,
        "hip_pos": -1.0,
        "contact_no_vel": -0.2,
        "feet_swing_height": -20.0,
        "contact": 0.18,
        "straight_motion": -1.0,
        "command_forward_vel": 0.5,
        "overspeed": -0.3,
    }


@registry.envcfg("k1-flat-terrain-walk-speed")
@dataclass
class K1WalkSpeedEnvCfg(K1WalkNpEnvCfg):
    commands: Commands = field(
        default_factory=lambda: Commands(
            lin_vel_x=[0.0, 1.0],
            lin_vel_y=[-0.2, 0.2],
            ang_vel_yaw=[-1.0, 1.0],
            yaw_curriculum=[0.3, 0.6, 1.0],
            resampling_time=10.0,
            stand_probability=0.10,
            straight_probability=0.55,
            turn_probability=0.10,
            mixed_turn_probability=0.05,
            direction_change_probability=0.0,
            sprint_turn_probability=0.0,
            straight_vx_range=[0.65, 1.0],
            apply_forward_yaw_envelope=False,
            turn_sustain_curriculum=[0, 0, 0],
            direction_change_period_steps=0,
        )
    )
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(scales=_speed_reward_scales(), tracking_sigma=0.25)
    )
