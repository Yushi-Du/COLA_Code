"""Shared schemas for the released COLA environments."""

from dataclasses import MISSING
import math

from isaaclab.utils import configclass


@configclass
class TerminationObservationCfg:
    robot_contact_force_threshold: float = MISSING
    feet_contact_force_threshold: float = MISSING
    object_min_height: float | None = MISSING


@configclass
class LocomotionPoseTrajectoryCfg:
    num_waypoints: int = MISSING
    lateral_randomization_scale: float = MISSING
    vertical_randomization_scale: float = MISSING
    left_center_xyz: tuple[float, float, float] = MISSING
    right_center_xyz: tuple[float, float, float] = MISSING
    left_center_quat: tuple[float, float, float, float] = MISSING
    right_center_quat: tuple[float, float, float, float] = MISSING


@configclass
class CollaborationGraspCfg:
    left_palm_quat: tuple[float, float, float, float] = MISSING
    right_palm_quat: tuple[float, float, float, float] = MISSING
    left_palm_xyz: tuple[float, float, float] = MISSING
    right_palm_xyz: tuple[float, float, float] = MISSING


@configclass
class CollaborationObservationCfg:
    mask_planar_velocity_command: bool = MISSING
    masked_height_command: float | None = MISSING


@configclass
class SupportMotionCfg:
    no_object_probability: float = MISSING
    bar_height_command_offset: float = MISSING
    bar_height_command_range: tuple[float, float] = MISSING
    no_object_height_command: float = MISSING
    planar_velocity_noise: float = MISSING
    vertical_velocity_noise: float = MISSING
    angular_velocity_noise: float = MISSING
    angular_velocity_noise_interval_steps: int = MISSING
    angular_velocity_noise_activation_probability: float = MISSING


@configclass
class SupportResetCfg:
    positive_y_probability: float = MISSING
    root_x_jitter_range: tuple[float, float] = MISSING
    root_y_jitter_range: tuple[float, float] = MISSING


@configclass
class LocomotionExperimentCfg:
    termination_observation: TerminationObservationCfg = MISSING
    pose_trajectory: LocomotionPoseTrajectoryCfg = MISSING


@configclass
class CollaborationExperimentCfg:
    termination_observation: TerminationObservationCfg = MISSING
    grasp: CollaborationGraspCfg = MISSING
    observations: CollaborationObservationCfg = MISSING
    support_motion: SupportMotionCfg = MISSING
    support_reset: SupportResetCfg = MISSING


@configclass
class SupportHeightControlCfg:
    ankle_height_offset: float = MISSING
    ankle_height_correction_threshold: float = MISSING
    corrected_height_range: tuple[float, float] = MISSING
    vertical_joint_height_offset: float = MISSING
    vertical_joint_velocity_gain: float = MISSING


@configclass
class BarControllerCfg:
    """Physical bar and world-frame controller parameters in SI units."""

    control_frequency_hz: float = 400.0
    bar_size: tuple[float, float, float] = (0.01, 1.60, 0.01)
    bar_mass: float = 1.0
    initial_center_position: tuple[float, float, float] = (0.298, 0.0, 0.924)
    human_endpoint_offset: tuple[float, float, float] = (0.0, -0.80, 0.0)
    robot_endpoint_offset: tuple[float, float, float] = (0.0, 0.80, 0.0)
    no_object_park_offset: tuple[float, float, float] = (1.15, 0.0, 1.40)

    height_kp: float = 800.0
    height_kd: float = 35.0
    height_force_limit: float = 300.0
    height_target_rate_limit: float = 0.20

    velocity_kp: float = 30.0
    velocity_ki: float = 60.0
    velocity_kd: float = 0.10
    horizontal_force_limit: float = 100.0
    integral_force_limit: float = 15.0
    derivative_cutoff_hz: float = 20.0
    velocity_target_slew_limit: float = 2.0
    velocity_error_deadband: float = 0.005

    yaw_kp: float = 20.0
    yaw_kd: float = 2.0
    yaw_torque_limit: float = 5.0
    yaw_target_rate_limit: float = math.radians(45.0)
    default_target_yaw_w: float = -0.5 * math.pi
    target_yaw_offset_range: tuple[float, float] = (
        math.radians(-115.0),
        math.radians(-65.0),
    )
    gain_randomization_enabled: bool = False
    gain_randomization_scale_range: tuple[float, float] = (0.8, 1.2)
    gravity_magnitude: float = 9.81


__all__ = [
    "BarControllerCfg",
    "CollaborationExperimentCfg",
    "CollaborationGraspCfg",
    "CollaborationObservationCfg",
    "LocomotionExperimentCfg",
    "LocomotionPoseTrajectoryCfg",
    "SupportHeightControlCfg",
    "SupportMotionCfg",
    "SupportResetCfg",
    "TerminationObservationCfg",
]
