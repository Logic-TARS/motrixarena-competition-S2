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
    lin_vel_y = [-0.5, 0.5]
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
    model_file: str = str(K1_ASSET_DIR / "k1_ball_scene.xml")
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: BallRewardConfig = field(default_factory=BallRewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    noise: Noise = field(default_factory=Noise)
    domain_rand: DomainRand = field(default_factory=DomainRand)
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
    model_file: str = str(K1_ASSET_DIR / "k1_train_scene.xml")
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: PointNavigateRewardConfig = field(default_factory=PointNavigateRewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    noise: Noise = field(default_factory=Noise)
    domain_rand: DomainRand = field(default_factory=DomainRand)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    point_config: PointNavigateConfig = field(default_factory=PointNavigateConfig)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02


# --- K1 AMP (Scheme A) ---

K1_AMP_JOINT_ORDER = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

K1_AMP_DEFAULT_JOINT_ANGLES = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.3,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.3,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.5,
    "Left_Hip_Pitch": -0.15,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.3,
    "Left_Ankle_Pitch": -0.15,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.15,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.3,
    "Right_Ankle_Pitch": -0.15,
    "Right_Ankle_Roll": 0.0,
}

K1_AMP_NUM_SINGLE_OBS = 75
K1_AMP_FRAME_STACK = 5
K1_AMP_NUM_ACT = 22
K1_AMP_ACTION_SCALE = 0.25
K1_AMP_UPPER_BODY_JOINTS = set(K1_AMP_JOINT_ORDER[:10])
K1_AMP_LEG_JOINTS = K1_AMP_JOINT_ORDER[10:]
K1_AMP_DEFAULT_MOTION_FILE = str(
    Path(__file__).resolve().parents[5] / "motions" / "K1" / "k1_mj2_seg1_50fps.npz"
)
K1_BEYOND_MIMIC_NUM_OBS = 119
K1_BEYOND_MIMIC_NUM_ACT = 22


@dataclass
class AmpCommands:
    lin_vel_x = [0.0, 0.0]
    lin_vel_y = [0.0, 0.0]
    ang_vel_yaw = [0.0, 0.0]
    max_lin_vel_x: float = 0.5
    max_lin_vel_y: float = 0.4
    max_ang_vel_yaw: float = 1.0
    resampling_time: float = 10.0
    command_deadzone: float = 0.0
    phase_period: float = 0.8


@dataclass
class AmpNormalization:
    lin_vel_x: float = 0.5
    lin_vel_y: float = 0.4
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05


@dataclass
class AmpControlConfig:
    stiffness: dict[str, float] = field(
        default_factory=lambda: {
            "AAHead_yaw": 30.0,
            "Head_pitch": 30.0,
            "ALeft_Shoulder_Pitch": 80.0,
            "Left_Shoulder_Roll": 80.0,
            "Left_Elbow_Pitch": 60.0,
            "Left_Elbow_Yaw": 60.0,
            "ARight_Shoulder_Pitch": 80.0,
            "Right_Shoulder_Roll": 80.0,
            "Right_Elbow_Pitch": 60.0,
            "Right_Elbow_Yaw": 60.0,
            "Left_Hip_Pitch": 80.0,
            "Left_Hip_Roll": 80.0,
            "Left_Hip_Yaw": 80.0,
            "Left_Knee_Pitch": 80.0,
            "Left_Ankle_Pitch": 30.0,
            "Left_Ankle_Roll": 30.0,
            "Right_Hip_Pitch": 80.0,
            "Right_Hip_Roll": 80.0,
            "Right_Hip_Yaw": 80.0,
            "Right_Knee_Pitch": 80.0,
            "Right_Ankle_Pitch": 30.0,
            "Right_Ankle_Roll": 30.0,
        }
    )
    damping: dict[str, float] = field(
        default_factory=lambda: {
            "AAHead_yaw": 5.0,
            "Head_pitch": 5.0,
            "ALeft_Shoulder_Pitch": 6.0,
            "Left_Shoulder_Roll": 6.0,
            "Left_Elbow_Pitch": 5.0,
            "Left_Elbow_Yaw": 5.0,
            "ARight_Shoulder_Pitch": 6.0,
            "Right_Shoulder_Roll": 6.0,
            "Right_Elbow_Pitch": 5.0,
            "Right_Elbow_Yaw": 5.0,
            "Left_Hip_Pitch": 2.0,
            "Left_Hip_Roll": 2.0,
            "Left_Hip_Yaw": 2.0,
            "Left_Knee_Pitch": 2.0,
            "Left_Ankle_Pitch": 2.0,
            "Left_Ankle_Roll": 2.0,
            "Right_Hip_Pitch": 2.0,
            "Right_Hip_Roll": 2.0,
            "Right_Hip_Yaw": 2.0,
            "Right_Knee_Pitch": 2.0,
            "Right_Ankle_Pitch": 2.0,
            "Right_Ankle_Roll": 2.0,
        }
    )
    action_scale: dict[str, float] = field(
        default_factory=lambda: {
            name: 0.0 if name in K1_AMP_UPPER_BODY_JOINTS else 0.25
            for name in K1_AMP_JOINT_ORDER
        }
    )
    torque_limit: dict[str, float] = field(
        default_factory=lambda: {
            "AAHead_yaw": 20.0,
            "Head_pitch": 20.0,
            "ALeft_Shoulder_Pitch": 60.0,
            "Left_Shoulder_Roll": 60.0,
            "Left_Elbow_Pitch": 40.0,
            "Left_Elbow_Yaw": 40.0,
            "ARight_Shoulder_Pitch": 60.0,
            "Right_Shoulder_Roll": 60.0,
            "Right_Elbow_Pitch": 40.0,
            "Right_Elbow_Yaw": 40.0,
            "Left_Hip_Pitch": 30.0,
            "Left_Hip_Roll": 35.0,
            "Left_Hip_Yaw": 20.0,
            "Left_Knee_Pitch": 40.0,
            "Left_Ankle_Pitch": 20.0,
            "Left_Ankle_Roll": 20.0,
            "Right_Hip_Pitch": 30.0,
            "Right_Hip_Roll": 35.0,
            "Right_Hip_Yaw": 20.0,
            "Right_Knee_Pitch": 40.0,
            "Right_Ankle_Pitch": 20.0,
            "Right_Ankle_Roll": 20.0,
        }
    )


@dataclass
class AmpRewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -500.0,
            "tracking_lin_vel": 0.0,
            "tracking_ang_vel": 0.0,
            "stand_still": -5.0,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.05,
            "orientation": -10.0,
            "base_height": -1.0,
            "torques": -1.0e-5,
            "dof_vel": -2.0e-4,
            "dof_acc": -2.5e-8,
            "feet_air_time": 0.0,
            "collision": -1.0,
            "action_rate": -0.02,
            "dof_pos_limits": -5.0,
            "alive": 10.0,
            "hip_pos": -0.2,
            "joint_regularization": -0.02,
            "upper_body_regularization": -1.0,
            "upper_body_velocity": -0.002,
            "motion_leg_joint_pos": 0.0,
            "motion_leg_joint_vel": 0.0,
            "motion_base_height": 0.0,
            "contact_no_vel": 0.0,
            "feet_swing_height": 0.0,
            "contact": 0.0,
        }
    )
    only_positive_rewards: bool = False
    trust_contact_rewards: bool = True
    tracking_sigma: float = 0.25
    min_base_height: float = 0.42
    swing_height: float = 0.08
    target_base_height: float = 0.54
    max_tilt_xy: float = 0.55
    soft_dof_pos_limit: float = 0.9
    gait_frequency: float = 1.5
    motion_joint_pos_sigma: float = 0.35
    motion_joint_vel_sigma: float = 3.0
    motion_base_height_sigma: float = 0.12
    command_forward_vel_margin: float = 0.03
    target_feet_air_time: float = 0.50


@dataclass
class AmpMotionReference:
    enabled: bool = True
    file: str = K1_AMP_DEFAULT_MOTION_FILE
    input_fps: float = 50.0


@dataclass
class AmpDomainRand(DomainRand):
    push_robots: bool = False


@dataclass
class AmpSensor:
    local_linvel = "local_linvel"
    gyro = "angular-velocity"


@registry.envcfg("k1-amp-walk")
@dataclass
class K1AmpWalkEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0
    model_file: str = str(K1_ASSET_DIR / "K1_22dof.xml")
    control_config: AmpControlConfig = field(default_factory=AmpControlConfig)
    reward_config: AmpRewardConfig = field(default_factory=AmpRewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: AmpCommands = field(default_factory=AmpCommands)
    normalization: AmpNormalization = field(default_factory=AmpNormalization)
    noise: Noise = field(default_factory=Noise)
    domain_rand: AmpDomainRand = field(default_factory=AmpDomainRand)
    motion_reference: AmpMotionReference = field(default_factory=AmpMotionReference)
    asset: Asset = field(default_factory=Asset)
    sensor: AmpSensor = field(default_factory=AmpSensor)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02


@registry.envcfg("k1-amp-stand")
@dataclass
class K1AmpStandEnvCfg(K1AmpWalkEnvCfg):
    pass


@dataclass
class AmpWalkSmallCommands(AmpCommands):
    lin_vel_x = [0.02, 0.08]


@dataclass
class AmpWalkSmallRewardConfig(AmpRewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -500.0,
            "tracking_lin_vel": 0.4,
            "command_forward_vel": 8.0,
            "overspeed": -6.0,
            "tracking_ang_vel": 0.0,
            "stand_still": -0.2,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.05,
            "orientation": -10.0,
            "base_height": -1.0,
            "torques": -1.0e-5,
            "dof_vel": -2.0e-4,
            "dof_acc": -2.5e-8,
            "feet_air_time": 0.0,
            "collision": -1.0,
            "action_rate": -0.02,
            "dof_pos_limits": -5.0,
            "alive": 10.0,
            "hip_pos": -0.2,
            "joint_regularization": -0.02,
            "upper_body_regularization": -1.0,
            "upper_body_velocity": -0.002,
            "motion_leg_joint_pos": 0.0,
            "motion_leg_joint_vel": 0.0,
            "motion_base_height": 0.0,
            "contact_no_vel": 0.0,
            "feet_swing_height": 0.0,
            "contact": 0.0,
        }
    )


@registry.envcfg("k1-amp-walk-small")
@dataclass
class K1AmpWalkSmallEnvCfg(K1AmpWalkEnvCfg):
    commands: AmpWalkSmallCommands = field(default_factory=AmpWalkSmallCommands)
    reward_config: AmpWalkSmallRewardConfig = field(default_factory=AmpWalkSmallRewardConfig)


@dataclass
class AmpWalkLiftRewardConfig(AmpWalkSmallRewardConfig):
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -500.0,
            "tracking_lin_vel": 0.4,
            "command_forward_vel": 8.0,
            "overspeed": -6.0,
            "tracking_ang_vel": 0.0,
            "stand_still": -0.2,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.05,
            "orientation": -10.0,
            "base_height": -1.0,
            "torques": -1.0e-5,
            "dof_vel": -2.0e-4,
            "dof_acc": -2.5e-8,
            "feet_air_time": 0.0,
            "collision": -1.0,
            "action_rate": -0.02,
            "dof_pos_limits": -5.0,
            "alive": 10.0,
            "hip_pos": -0.2,
            "joint_regularization": -0.02,
            "upper_body_regularization": -1.0,
            "upper_body_velocity": -0.002,
            "motion_leg_joint_pos": 0.03,
            "motion_leg_joint_vel": 0.005,
            "motion_base_height": 0.0,
            "contact_no_vel": -0.2,
            "feet_swing_height": -20.0,
            "contact": 0.18,
        }
    )
    target_feet_air_time: float = 0.12  # unused when feet_air_time scale is 0


@registry.envcfg("k1-amp-walk-lift")
@dataclass
class K1AmpWalkLiftEnvCfg(K1AmpWalkEnvCfg):
    commands: AmpWalkSmallCommands = field(default_factory=AmpWalkSmallCommands)
    reward_config: AmpWalkLiftRewardConfig = field(default_factory=AmpWalkLiftRewardConfig)
    motion_reference: AmpMotionReference = field(default_factory=lambda: AmpMotionReference(enabled=True))


# --- K1 BeyondMimic / booster_train native port ---


@dataclass
class BeyondMimicMotionReference:
    file: str = K1_AMP_DEFAULT_MOTION_FILE
    input_fps: float = 50.0
    random_start: bool = True
    reset_root_velocity_scale: float = 0.0
    reset_joint_velocity_scale: float = 0.0
    max_reset_joint_vel: float = 5.0
    root_pos_perturb: float = 0.02
    root_z_perturb: float = 0.01
    root_rpy_perturb: float = 0.05
    root_yaw_perturb: float = 0.10
    root_lin_vel_perturb: float = 0.0
    root_ang_vel_perturb: float = 0.0
    joint_pos_perturb: float = 0.02


@dataclass
class BeyondMimicRewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "motion_global_anchor_pos": 0.5,
            "motion_global_anchor_ori": 0.5,
            "motion_joint_pos": 1.0,
            "motion_joint_vel": 0.1,
            "action_rate_l2": -0.1,
            "joint_limit": -10.0,
            "undesired_contacts": -0.1,
            "base_height": -0.1,
            "orientation": -0.1,
            "alive": 0.1,
            "termination": -0.0,
        }
    )
    only_positive_rewards: bool = False
    motion_anchor_pos_sigma: float = 0.3
    motion_anchor_ori_sigma: float = 0.4
    motion_joint_pos_sigma: float = 0.3
    motion_joint_vel_sigma: float = 1.0
    target_base_height: float = 0.54
    min_base_height: float = 0.42
    max_tilt_xy: float = 0.80
    soft_dof_pos_limit: float = 0.9
    anchor_max_height_error: float = 1.0
    anchor_max_ori_error: float = 1.5


@dataclass
class BeyondMimicDomainRand(DomainRand):
    push_robots: bool = True
    push_interval_s: float = 3.0
    max_push_vel_xy: float = 0.5


@registry.envcfg("k1-beyond-mimic-mj-dance-002")
@dataclass
class K1BeyondMimicMjDance002EnvCfg(EnvCfg):
    max_episode_seconds: float = 10.0
    model_file: str = str(K1_ASSET_DIR / "K1_22dof.xml")
    control_config: AmpControlConfig = field(default_factory=AmpControlConfig)
    reward_config: BeyondMimicRewardConfig = field(default_factory=BeyondMimicRewardConfig)
    init_state: InitState = field(default_factory=InitState)
    normalization: AmpNormalization = field(default_factory=AmpNormalization)
    noise: Noise = field(default_factory=Noise)
    domain_rand: BeyondMimicDomainRand = field(default_factory=BeyondMimicDomainRand)
    motion_reference: BeyondMimicMotionReference = field(default_factory=BeyondMimicMotionReference)
    asset: Asset = field(default_factory=Asset)
    sensor: AmpSensor = field(default_factory=AmpSensor)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02


@registry.envcfg("k1-mj-dance-002")
@dataclass
class K1MjDance002EnvCfg(K1BeyondMimicMjDance002EnvCfg):
    pass
