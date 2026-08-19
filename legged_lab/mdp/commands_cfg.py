# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local configuration objects for the COLA command generators."""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .commands import (
    UniformEEPoseCommandWorldFollowingQuatVel,
    UniformEEPoseCommandWorldQuat_Straight,
    UniformVelocityHeightCommand,
)


@configclass
class UniformVelocityHeightCommandCfg(CommandTermCfg):
    """Planar velocity plus robot/support height command configuration."""

    class_type: type = UniformVelocityHeightCommand
    asset_name: str = MISSING
    heading_command: bool = False
    heading_control_stiffness: float = 1.0
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING
        height: tuple[float, float] = MISSING
        link_height: tuple[float, float] = (0.0, 0.0)
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING


@configclass
class UniformEEPoseCommandWorldCfg(CommandTermCfg):
    """Fixed collaborative grasp-pose command configuration."""

    class_type: type = UniformEEPoseCommandWorldFollowingQuatVel
    asset_name: str = MISSING

    @configclass
    class Ranges:
        left_r: tuple[float, float] = MISSING
        left_p: tuple[float, float] = MISSING
        left_yaw: tuple[float, float] = MISSING
        right_r: tuple[float, float] = MISSING
        right_p: tuple[float, float] = MISSING
        right_yaw: tuple[float, float] = MISSING
        left_x: tuple[float, float] = MISSING
        left_y: tuple[float, float] = MISSING
        left_z: tuple[float, float] = MISSING
        right_x: tuple[float, float] = MISSING
        right_y: tuple[float, float] = MISSING
        right_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING


@configclass
class UniformEEPoseCommandQuatCfg(CommandTermCfg):
    """Random locomotion end-effector trajectory configuration."""

    class_type: type = UniformEEPoseCommandWorldQuat_Straight
    asset_name: str = MISSING

    @configclass
    class Ranges:
        orientation_cone_rad: float = MISSING
        cube_size: float = MISSING

    ranges: Ranges = MISSING
