# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 articulation used by the COLA training pipeline."""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import G1_FIXED_HAND_USD_PATH


G1_HAND_FIXED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_FIXED_HAND_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_wrist_roll_joint": np.pi / 2,
            "left_wrist_roll_joint": -np.pi / 2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.80,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 50.0,
                ".*waist_pitch_joint": 50.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 37.0,
                ".*waist_pitch_joint": 37.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 40.17923847137318,
                ".*_hip_roll_joint": 99.09842777666113,
                ".*_hip_pitch_joint": 40.17923847137318,
                ".*_knee_joint": 99.09842777666113,
                ".*waist_yaw_joint": 40.17923847137318,
                ".*waist_roll_joint": 28.50124619574858,
                ".*waist_pitch_joint": 28.50124619574858,
            },
            damping={
                ".*_hip_yaw_joint": 2.5578897650279457,
                ".*_hip_roll_joint": 6.3088018534966395,
                ".*_hip_pitch_joint": 2.5578897650279457,
                ".*_knee_joint": 6.3088018534966395,
                ".*waist_yaw_joint": 2.5578897650279457,
                ".*waist_roll_joint": 1.814445686584846,
                ".*waist_pitch_joint": 1.814445686584846,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            stiffness=28.50124619574858,
            damping=1.814445686584846,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
            },
            stiffness=14.25062309787429,
            damping=0.907222843292423,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
            },
            stiffness=14.25062309787429,
            damping=0.907222843292423,
            armature=0.01,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": 22.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
            },
            stiffness={
                ".*_wrist_roll_joint": 14.25062309787429,
                ".*_wrist_pitch_joint": 16.77832748089279,
                ".*_wrist_yaw_joint": 16.77832748089279,
            },
            damping={
                ".*_wrist_roll_joint": 0.907222843292423,
                ".*_wrist_pitch_joint": 1.06814150219,
                ".*_wrist_yaw_joint": 1.06814150219,
            },
            armature=0.01,
        ),
    },
)

__all__ = ["G1_HAND_FIXED_CFG"]
