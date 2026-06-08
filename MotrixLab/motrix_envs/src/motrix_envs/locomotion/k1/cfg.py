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
    pos = [0.0, 0.0, 0.57]
    default_joint_angles = {
        "Left_Hip_Pitch": -0.2,
        "Right_Hip_Pitch": -0.2,
        "Left_Knee_Pitch": 0.4,
        "Right_Knee_Pitch": 0.4,
        "Left_Ankle_Pitch": -0.25,
        "Right_Ankle_Pitch": -0.25,
        "default": 0.0,
    }


@dataclass
class Commands:
    vel_limit = [
        [0.35, 0.0, 0.0],
        [0.55, 0.0, 0.0],
        [0.8, 0.0, 0.0],
    ]
    lin_vel_x = [-1.0, 1.0]
    lin_vel_y = [-0.2, 0.2]
    ang_vel_yaw = [-1.0, 1.0]
    resampling_time: float = 10.0
    command_deadzone: float = 0.2
    phase_period: float = 0.8


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
    noise_level: float = 1.0
    noise_scales: NoiseScales = field(default_factory=NoiseScales)


@dataclass
class DomainRand:
    push_robots: bool = True
    push_interval_s: float = 8.0
    max_push_vel_xy: float = 0.5


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
            "termination": -0.0,
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "base_height": -10.0,
            "torques": -1.0e-5,
            "dof_vel": -1.0e-3,
            "dof_acc": -2.5e-7,
            "feet_air_time": 0.0,
            "collision": 0.0,
            "action_rate": -0.01,
            "dof_pos_limits": -5.0,
            "alive": 0.15,
            "hip_pos": -1.0,
            "contact_no_vel": -0.2,
            "feet_swing_height": -20.0,
            "contact": 0.18,
            "straight_motion": 0.8,
            "command_forward_vel": 0.3,
            "overspeed": -0.3,
        }
    )
    only_positive_rewards: bool = True
    trust_contact_rewards: bool = True
    tracking_sigma: float = 0.25
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
    straight_motion_yaw_weight: float = 1.0
    straight_motion_lateral_weight: float = 2.0
    gait_frequency: float = 1.5


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
