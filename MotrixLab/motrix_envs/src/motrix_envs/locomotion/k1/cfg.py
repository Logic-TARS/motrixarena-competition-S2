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

from motrix_envs import registry
from motrix_envs.base import EnvCfg


@dataclass
class ControlConfig:
    stiffness: dict[str, float] = field(
        default_factory=lambda: {
            "Left_Hip_Pitch": 200.0,
            "Right_Hip_Pitch": 200.0,
            "Left_Hip_Roll": 200.0,
            "Right_Hip_Roll": 200.0,
            "Left_Hip_Yaw": 200.0,
            "Right_Hip_Yaw": 200.0,
            "Left_Knee_Pitch": 200.0,
            "Right_Knee_Pitch": 200.0,
            "Left_Ankle_Pitch": 50.0,
            "Right_Ankle_Pitch": 50.0,
            "Left_Ankle_Roll": 50.0,
            "Right_Ankle_Roll": 50.0,
        }
    )
    damping: dict[str, float] = field(
        default_factory=lambda: {
            "Left_Hip_Pitch": 5.0,
            "Right_Hip_Pitch": 5.0,
            "Left_Hip_Roll": 5.0,
            "Right_Hip_Roll": 5.0,
            "Left_Hip_Yaw": 5.0,
            "Right_Hip_Yaw": 5.0,
            "Left_Knee_Pitch": 5.0,
            "Right_Knee_Pitch": 5.0,
            "Left_Ankle_Pitch": 1.0,
            "Right_Ankle_Pitch": 1.0,
            "Left_Ankle_Roll": 1.0,
            "Right_Ankle_Roll": 1.0,
        }
    )
    action_scale: float = 0.35
    torque_limit: float = 40.0


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.72]
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
        [0.25, -0.05, -0.10],
        [0.65, 0.05, 0.10],
    ]


@dataclass
class Normalization:
    lin_vel = 1.0
    ang_vel = 1.0
    dof_pos = 1.0
    dof_vel = 0.1


@dataclass
class Asset:
    body_name: str = "Trunk"
    foot_name: str = "Foot"
    ground_name: str = "ground"
    penalize_contacts_on: list = field(default_factory=lambda: ["Trunk", "Shank"])
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
            "termination": -0.5,
            "tracking_lin_vel": 3.0,
            "tracking_ang_vel": 0.25,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "base_height": -4.0,
            "torques": -0.0002,
            "dof_vel": -0.003,
            "dof_acc": -5.0e-6,
            "action_rate": -0.02,
            "joint_regularization": -0.2,
            "feet_air_time": 0.1,
            "collision": -2.0,
        }
    )
    tracking_sigma: float = 0.25
    min_base_height: float = 0.35
    max_foot_height: float = 0.15
    target_base_height: float = 0.55
    max_tilt_xy: float = 0.8
    gait_frequency: float = 1.5


@registry.envcfg("k1-flat-terrain-walk")
@dataclass
class K1WalkNpEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0
    model_file: str = "/opt/sim_soccer2/legged_gym/resources/robots/K1/k1_train_scene.xml"
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
