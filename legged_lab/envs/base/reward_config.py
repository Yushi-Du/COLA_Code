"""Reward profiles used by the released collaboration tasks."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.base import collaboration_mdp
from legged_lab.envs.base.locomotion_config import (
    LocomotionRewardCfg as LocomotionRewardBaseCfg,
)


@configclass
class CollaborationRobotRewardCfg:
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names="(?!.*(ankle|hand|wrist).*).*"
            ),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "threshold": 0.4,
            "moving_command_threshold": 0.1,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
            "contact_force_threshold": 1.0,
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": 0.2,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*ankle_roll.*"]
            ),
            "horizontal_vertical_force_ratio": 5.0,
        },
    )
    feet_height = RewTerm(
        func=mdp.penalty_feet_height,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*ankle_roll.*"]
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": 0.02,
            "target_height": 0.13,
            "contact_force_threshold": 1.0,
            "moving_command_threshold": 0.1,
        },
    )
    knees_too_near = RewTerm(
        func=mdp.knees_too_near_humanoid,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*knee_link.*"]),
            "threshold": 0.18,
        },
    )
    feet_orientation = RewTerm(
        func=mdp.feet_orientation,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": 0.2,
        },
    )
    feet_too_far = RewTerm(
        func=mdp.feet_too_far_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": 0.36,
        },
    )
    knees_too_far = RewTerm(
        func=mdp.knees_too_far_humanoid,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*knee_link.*"]),
            "threshold": 0.34,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    ankle_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch.*", ".*_ankle_roll.*"]
            )
        },
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.5,
        params={"soft_ratio": 0.9},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw.*",
                    ".*_hip_roll.*",
                    ".*_shoulder_pitch.*",
                    ".*_elbow.*",
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*waist.*", ".*_shoulder_yaw.*", ".*_wrist.*"],
            )
        },
    )
    waist_roll_zero_position_penalty = RewTerm(
        func=mdp.joint_position_l1_to_target,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_roll_joint"]
            ),
            "target_position": 0.0,
        },
    )
    waist_roll_position_jitter_penalty = RewTerm(
        func=collaboration_mdp.waist_roll_position_jitter_penalty,
        weight=-0.10,
        params={
            "history_length": 10,
            "position_scale": 0.02,
            "maximum_penalty": 4.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_roll_joint"]
            ),
        },
    )
    left_shoulder_roll_default_position_penalty = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_roll_joint"]
            )
        },
    )
    right_shoulder_roll_default_position_penalty = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["right_shoulder_roll_joint"]
            )
        },
    )
    feet_vertical_load_imbalance_penalty = RewTerm(
        func=mdp.feet_vertical_load_imbalance_penalty,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "foot_balance_contact_sensor",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "contact_force_threshold": 25.0,
            "command_threshold": 0.1,
            "dead_zone": 0.1,
        },
    )


@configclass
class FixedBarRewardCfg(CollaborationRobotRewardCfg):
    target_body_z_vel_penalty = RewTerm(
        func=collaboration_mdp.endpoint_z_velocity_penalty,
        weight=-0.005,
        params={"threshold": 0.05},
    )
    target_height_difference_penalty = RewTerm(
        func=collaboration_mdp.bar_endpoint_height_difference_penalty,
        weight=-20.0,
    )
    human_effort_reward = RewTerm(
        func=collaboration_mdp.human_effort_reward,
        weight=1.0,
        params={"force_scale": 100.0, "epsilon": 1.0e-8},
    )
    bar_center_height_tracking_reward = RewTerm(
        func=collaboration_mdp.bar_center_height_tracking_reward,
        weight=1.0,
        params={"std": 0.10},
    )
    bar_center_horizontal_velocity_tracking_reward = RewTerm(
        func=collaboration_mdp.bar_center_horizontal_velocity_tracking_reward,
        weight=1.0,
        params={"std": 0.50},
    )
    target_bar_vector_alignment_reward = RewTerm(
        func=collaboration_mdp.target_bar_vector_alignment_reward,
        weight=6.0,
        params={"epsilon": 1.0e-8},
    )
    loose_upper_body_ee_left_quat_reward = RewTerm(
        func=collaboration_mdp.virtual_palm_quaternion_reward,
        weight=0.50,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="left_wrist_yaw_link"
            ),
            "std": 0.25,
            "deadband": 0.06,
            "command_start": 0,
            "wrist_to_palm_position": (0.0415, 0.003, 0.0),
        },
    )
    loose_upper_body_ee_right_quat_reward = RewTerm(
        func=collaboration_mdp.virtual_palm_quaternion_reward,
        weight=0.50,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="right_wrist_yaw_link"
            ),
            "std": 0.25,
            "deadband": 0.06,
            "command_start": 4,
            "wrist_to_palm_position": (0.0415, -0.003, 0.0),
        },
    )
    loose_upper_body_ee_left_xyz_quat_reward = RewTerm(
        func=collaboration_mdp.virtual_palm_position_reward,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="left_wrist_yaw_link"
            ),
            "std": 0.25,
            "deadband": 0.06,
            "command_start": 8,
            "wrist_to_palm_position": (0.0415, 0.003, 0.0),
        },
    )
    loose_upper_body_ee_right_xyz_quat_reward = RewTerm(
        func=collaboration_mdp.virtual_palm_position_reward,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names="right_wrist_yaw_link"
            ),
            "std": 0.25,
            "deadband": 0.06,
            "command_start": 11,
            "wrist_to_palm_position": (0.0415, -0.003, 0.0),
        },
    )
    bar_translational_jitter_penalty = RewTerm(
        func=collaboration_mdp.bar_translational_jitter_penalty,
        weight=-0.08,
        params={
            "history_length": 10,
            "horizontal_velocity_scale": 0.10,
            "vertical_velocity_scale": 0.05,
            "settled_tolerance": 1.0e-4,
            "maximum_penalty": 4.0,
        },
    )
    bar_vector_rate_jitter_penalty = RewTerm(
        func=collaboration_mdp.bar_vector_rate_jitter_penalty,
        weight=-0.05,
        params={
            "history_length": 10,
            "vector_rate_scale": 0.35,
            "settled_tolerance": 1.0e-4,
            "maximum_penalty": 4.0,
        },
    )
    height_controller_force_jitter_penalty = RewTerm(
        func=collaboration_mdp.height_controller_force_jitter_penalty,
        weight=-0.40,
        params={
            "history_length": 40,
            "force_scale": 10.0,
            "settled_tolerance": 1.0e-4,
            "maximum_penalty": 4.0,
        },
    )
    horizontal_controller_force_jitter_penalty = RewTerm(
        func=collaboration_mdp.horizontal_controller_force_jitter_penalty,
        weight=-0.10,
        params={
            "history_length": 40,
            "force_scale": 5.0,
            "settled_tolerance": 1.0e-4,
            "maximum_penalty": 4.0,
        },
    )


@configclass
class NoObjectRewardCfg(CollaborationRobotRewardCfg):
    loose_upper_body_ee_left_quat_reward = RewTerm(
        func=mdp.loose_upper_body_ee_left_quat_reward,
        weight=0.50,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*left_hand_palm_link"]
            ),
            "std": 0.25,
            "deadband": 0.06,
        },
    )
    loose_upper_body_ee_right_quat_reward = RewTerm(
        func=mdp.loose_upper_body_ee_right_quat_reward,
        weight=0.50,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*right_hand_palm_link"]
            ),
            "std": 0.25,
            "deadband": 0.06,
        },
    )
    loose_upper_body_ee_left_xyz_quat_reward = RewTerm(
        func=mdp.loose_upper_body_ee_left_xyz_quat_reward,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*left_hand_palm_link"]
            ),
            "std": 0.25,
            "deadband": 0.06,
        },
    )
    loose_upper_body_ee_right_xyz_quat_reward = RewTerm(
        func=mdp.loose_upper_body_ee_right_xyz_quat_reward,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*right_hand_palm_link"]
            ),
            "std": 0.25,
            "deadband": 0.06,
        },
    )


@configclass
class Phase1RewardCfg(LocomotionRewardBaseCfg):
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.8, params={"std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=3.0, params={"std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names="(?!.*(ankle|hand|wrist).*).*"
            ),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.18,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 0.4,
            "moving_command_threshold": 0.1,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
            "contact_force_threshold": 1.0,
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": 0.2,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*ankle_roll.*"]
            ),
            "horizontal_vertical_force_ratio": 5.0,
        },
    )
    knees_too_near = RewTerm(
        func=mdp.knees_too_near_humanoid,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*knee_link.*"]),
            "threshold": 0.18,
        },
    )
    feet_orientation = RewTerm(
        func=mdp.feet_orientation,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": 0.2,
        },
    )
    feet_too_far = RewTerm(
        func=mdp.feet_too_far_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": 0.42,
        },
    )
    knees_too_far = RewTerm(
        func=mdp.knees_too_far_humanoid,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*knee_link.*"]),
            "threshold": 0.4,
        },
    )
    ankle_action_rate_l2 = RewTerm(
        func=mdp.ankle_action_rate_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch.*", ".*_ankle_roll.*"]
            )
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    ankle_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch.*", ".*_ankle_roll.*"]
            )
        },
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-0.5, params={"soft_ratio": 0.9}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw.*",
                    ".*_hip_roll.*",
                    ".*_shoulder_pitch.*",
                    ".*_elbow.*",
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*waist_roll.*",
                    ".*waist_pitch.*",
                    ".*_shoulder_yaw.*",
                    ".*_wrist.*",
                ],
            )
        },
    )
    joint_deviation_waist_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist_yaw.*"])},
    )
    upper_body_ee_left_quat_reward = RewTerm(
        func=mdp.upper_body_ee_left_quat_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*left_hand_palm_link"]),
            "std": 0.25,
        },
    )
    upper_body_ee_right_quat_reward = RewTerm(
        func=mdp.upper_body_ee_right_quat_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*right_hand_palm_link"]),
            "std": 0.25,
        },
    )
    upper_body_ee_left_xyz_quat_reward = RewTerm(
        func=mdp.upper_body_ee_left_xyz_quat_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*left_hand_palm_link"]),
            "std": 0.25,
        },
    )
    upper_body_ee_right_xyz_quat_reward = RewTerm(
        func=mdp.upper_body_ee_right_xyz_quat_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*right_hand_palm_link"]),
            "std": 0.25,
        },
    )
    loose_robot_height = RewTerm(
        func=mdp.loose_robot_height,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "deadband": 0.02},
    )
    loose_robot_height_reward = RewTerm(
        func=mdp.loose_robot_height_reward,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.25,
            "deadband": 0.02,
        },
    )
    zero_velocity_command_stillness = RewTerm(
        func=mdp.zero_velocity_command_stillness_reward,
        weight=0.2,
        params={
            "std": 0.5,
            "command_threshold": 1.0e-3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class TeacherFixedBarRewardCfg(
    FixedBarRewardCfg
):
    track_lin_vel_xy_yaw_frame_exp_world = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_world,
        weight=2.0,
        params={"std": 0.5},
    )
    track_ang_vel_z_world_exp_world = RewTerm(
        func=mdp.track_ang_vel_z_world_exp_world,
        weight=0.5,
        params={"std": 0.5},
    )
    no_object_stand_anchor = RewTerm(
        func=mdp.no_object_stand_anchor,
        weight=-5.0,
        params={"std": 0.5},
    )
    zero_velocity_command_stillness = RewTerm(
        func=mdp.zero_velocity_command_target_aligned_stillness_reward,
        weight=0.2,
        params={
            "std": 0.5,
            "command_threshold": 1.0e-3,
            "maximum_vector_angle": math.radians(15.0),
            "epsilon": 1.0e-8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class StudentFixedBarRewardCfg(
    FixedBarRewardCfg
):
    track_lin_vel_xy_yaw_frame_exp_world = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_world,
        weight=2.0,
        params={"std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"std": 0.5},
    )
    zero_velocity_command_stillness = RewTerm(
        func=mdp.zero_velocity_command_target_aligned_stillness_reward,
        weight=0.2,
        params={
            "std": 0.5,
            "command_threshold": 1.0e-3,
            "maximum_vector_angle": math.radians(15.0),
            "epsilon": 1.0e-8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class TeacherNoObjectRewardCfg(
    NoObjectRewardCfg
):
    track_lin_vel_xy_yaw_frame_exp_world = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_world,
        weight=2.0,
        params={"std": 0.5},
    )
    track_ang_vel_z_world_exp_world = RewTerm(
        func=mdp.track_ang_vel_z_world_exp_world,
        weight=0.5,
        params={"std": 0.5},
    )
    zero_velocity_command_stillness = RewTerm(
        func=mdp.zero_velocity_command_stillness_reward,
        weight=0.2,
        params={
            "std": 0.5,
            "command_threshold": 1.0e-3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class StudentNoObjectRewardCfg(
    NoObjectRewardCfg
):
    track_lin_vel_xy_yaw_frame_exp_world = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_world,
        weight=2.0,
        params={"std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"std": 0.5},
    )
    zero_velocity_command_stillness = RewTerm(
        func=mdp.zero_velocity_command_stillness_reward,
        weight=0.2,
        params={
            "std": 0.5,
            "command_threshold": 1.0e-3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

__all__ = [
    "CollaborationRobotRewardCfg",
    "FixedBarRewardCfg",
    "NoObjectRewardCfg",
    "Phase1RewardCfg",
    "StudentFixedBarRewardCfg",
    "StudentNoObjectRewardCfg",
    "TeacherFixedBarRewardCfg",
    "TeacherNoObjectRewardCfg",
]
