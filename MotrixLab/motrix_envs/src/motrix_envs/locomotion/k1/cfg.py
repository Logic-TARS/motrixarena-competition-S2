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
        [0.35, 0.0, 0.0],
        [0.55, 0.0, 0.0],
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
            "termination": -100.0,
            "alive": 0.05,
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.25,
            "command_forward_vel": 0.3,
            "overspeed": -8.0,
            "straight_motion": -2.0,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -3.0,
            "base_height": -2.0,
            "torques": -0.0002,
            "dof_vel": -0.003,
            "dof_acc": -2.5e-6,
            "action_rate": -0.02,
            "joint_regularization": -0.2,
            "feet_air_time": 1.0,
            "collision": -1.0,
        }
    )
    tracking_sigma: float = 0.08
    min_base_height: float = 0.45
    max_foot_height: float = 0.15
    target_base_height: float = 0.68
    max_tilt_xy: float = 0.55
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


@dataclass
class BallConfig:
    radius: float = 0.11
    mass: float = 0.1
    arrival_radius: float = 0.35
    spawn_dist_min: float = 1.0
    spawn_dist_max: float = 3.0
    spawn_angle_max: float = 0.6
    kick_speed_threshold: float = 0.5
    command_forward_gain: float = 0.45
    command_turn_gain: float = 1.2
    command_max_forward_vel: float = 0.55
    command_max_yaw_rate: float = 0.8
    close_control_radius: float = 0.8
    close_command_max_forward_vel: float = 0.22
    close_command_turn_gain: float = 0.8
    kick_alignment_cos: float = 0.9
    kick_target_dir: list[float] = field(default_factory=lambda: [1.0, 0.0])
    kick_success_distance: float = 0.6
    kick_push_forward_vel: float = 0.35
    ball_progress_vel_clip: float = 1.2
    goal_side_mode: str = "random"
    goal_x_abs: float = 4.0
    goal_width: float = 1.6
    shot_arc_radius_min: float = 2.0
    shot_arc_radius_max: float = 3.0
    shot_arc_angle_max: float = 0.7853981633974483
    robot_ball_backoff_min: float = 0.8
    robot_ball_backoff_max: float = 1.2
    robot_lateral_jitter: float = 0.15
    geom_name: str = "ball_geom"
    body_name: str = "ball"


@dataclass
class BallRewardConfig(RewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -100.0,
            "alive": 0.0,
            "tracking_lin_vel": 0.8,
            "tracking_ang_vel": 0.35,
            "command_forward_vel": 0.2,
            "overspeed": -8.0,
            "straight_motion": -2.0,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -3.0,
            "base_height": -2.0,
            "torques": -0.0002,
            "dof_vel": -0.003,
            "dof_acc": -2.5e-6,
            "action_rate": -0.02,
            "joint_regularization": -0.2,
            "feet_air_time": 1.0,
            "collision": -1.0,
            "approach_ball": 0.4,
            "low_speed_penalty": -0.25,
            "ball_forward_progress": 6.0,
            "effective_kick": 5.0,
            "face_ball": 0.5,
            "near_ball": 0.0,
            "stuck_near_ball": -1.0,
            "gait_contact_phase": 0.25,
            "single_foot_contact": 0.15,
            "double_support_penalty": -0.10,
            "arrival_bonus": 0.0,
        }
    )


@registry.envcfg("k1-ball-navigate")
@dataclass
class K1BallNavigateEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0
    model_file: str = "/opt/sim_soccer2/legged_gym/resources/robots/K1/k1_ball_scene.xml"
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: BallRewardConfig = field(default_factory=BallRewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    ball_config: BallConfig = field(default_factory=BallConfig)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02


@dataclass
class PointNavigateConfig:
    spawn_dist_min: float = 1.0
    spawn_dist_max: float = 3.0
    spawn_angle_max: float = 0.8
    arrival_radius: float = 0.25
    stop_speed_threshold: float = 0.15
    command_forward_gain: float = 0.45
    command_turn_gain: float = 1.2
    command_max_forward_vel: float = 0.55
    command_max_yaw_rate: float = 0.8


@dataclass
class PointNavigateRewardConfig(RewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -100.0,
            "alive": 0.0,
            "tracking_lin_vel": 0.8,
            "tracking_ang_vel": 0.35,
            "command_forward_vel": 0.2,
            "overspeed": -8.0,
            "straight_motion": -2.0,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -3.0,
            "base_height": -2.0,
            "torques": -0.0002,
            "dof_vel": -0.003,
            "dof_acc": -2.5e-6,
            "action_rate": -0.02,
            "joint_regularization": -0.2,
            "feet_air_time": 1.0,
            "collision": -1.0,
            "progress_to_target": 1.0,
            "heading_to_target": 0.0,
            "low_speed_penalty": -0.25,
            "arrival": 0.0,
            "arrival_bonus": 80.0,
            "stop_at_target": 0.0,
        }
    )


@registry.envcfg("k1-point-navigate")
@dataclass
class K1PointNavigateEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0
    model_file: str = "/opt/sim_soccer2/legged_gym/resources/robots/K1/k1_train_scene.xml"
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: PointNavigateRewardConfig = field(default_factory=PointNavigateRewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    point_config: PointNavigateConfig = field(default_factory=PointNavigateConfig)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
