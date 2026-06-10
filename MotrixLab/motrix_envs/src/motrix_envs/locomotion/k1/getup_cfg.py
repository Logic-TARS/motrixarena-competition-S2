"""Configuration for the K1 full-body get-up task."""

from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg
from motrix_envs.locomotion.k1.cfg import K1_ASSET_DIR


K1_FULL_BODY_JOINTS = [
    "AAHead_yaw",
    "ALeft_Shoulder_Pitch",
    "ARight_Shoulder_Pitch",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Head_pitch",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
]


@dataclass
class GetupControlConfig:
    action_scale: float = 1.0


@dataclass
class GetupResetConfig:
    base_height: float = 0.30
    joint_noise: float = 0.12
    linear_velocity: float = 0.15
    angular_velocity: float = 0.35
    curriculum_resets_per_stage: int = 8192


@dataclass
class GetupRewardConfig:
    stand_progress: float = 8.0
    height: float = 2.0
    upright: float = 2.0
    target_pose: float = 0.75
    stability: float = 0.5
    action_rate: float = -0.02
    torque: float = -2.0e-5
    time: float = -0.35
    success_bonus: float = 20.0
    target_height: float = 0.54
    success_height: float = 0.50
    success_tilt: float = 0.25
    success_ang_vel: float = 0.50
    success_hold_seconds: float = 0.50


@dataclass
class GetupDomainRand:
    push_interval_s: float = 2.0
    max_push_velocity: float = 0.25
    motor_strength_range: tuple[float, float] = (0.85, 1.15)
    damping_scale_range: tuple[float, float] = (0.80, 1.20)


@registry.envcfg("k1-getup")
@dataclass
class K1GetupNpEnvCfg(EnvCfg):
    model_file: str = str(K1_ASSET_DIR / "K1_22dof.xml")
    max_episode_seconds: float = 20.0
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    control: GetupControlConfig = field(default_factory=GetupControlConfig)
    reset_config: GetupResetConfig = field(default_factory=GetupResetConfig)
    reward_config: GetupRewardConfig = field(default_factory=GetupRewardConfig)
    domain_rand: GetupDomainRand = field(default_factory=GetupDomainRand)
