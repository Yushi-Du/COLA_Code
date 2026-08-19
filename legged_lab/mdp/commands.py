# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local command generators for COLA collaborative carrying."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from legged_lab.utils.bar_controller import target_vector_from_yaw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import (
        UniformEEPoseCommandQuatCfg,
        UniformEEPoseCommandWorldCfg,
        UniformVelocityHeightCommandCfg,
    )


logger = logging.getLogger(__name__)


class UniformVelocityHeightCommand(CommandTerm):
    """Sample base-frame planar velocity and base-height commands."""

    def __init__(self, cfg: UniformVelocityHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_command_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self._validate_heading_cfg()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.vel_command_b, self.height_command_b), dim=1)

    def _validate_heading_cfg(self) -> None:
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError("heading_command=True requires a heading range")
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            logger.warning("A heading range is configured while heading_command is disabled.")

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.linalg.vector_norm(
                self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        samples = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = samples.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = samples.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = samples.uniform_(*self.cfg.ranges.ang_vel_z)
        self.height_command_b[env_ids, 0] = samples.uniform_(*self.cfg.ranges.height)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = samples.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = (
                samples.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        self.is_standing_env[env_ids] = (
            samples.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

    def _update_command(self) -> None:
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clamp(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        self.vel_command_b[self.is_standing_env] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError


class UniformVelocityWorldHeightVelCommand(UniformVelocityHeightCommand):
    """Sample world-frame planar velocity, base height, and partner-link height."""

    def __init__(self, cfg: UniformVelocityHeightCommandCfg, env: ManagerBasedEnv):
        CommandTerm.__init__(self, cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_command_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.link_height_command_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self._validate_heading_cfg()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.vel_command_w, self.height_command_b), dim=1)

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.linalg.vector_norm(
                self.vel_command_w[:, :2] - self.robot.data.root_lin_vel_w[:, :2], dim=-1
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_w[:, 2] - self.robot.data.root_ang_vel_w[:, 2])
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        samples = torch.empty(len(env_ids), device=self.device)
        self.vel_command_w[env_ids, 0] = samples.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_w[env_ids, 1] = samples.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_w[env_ids, 2] = samples.uniform_(*self.cfg.ranges.ang_vel_z)
        self.height_command_b[env_ids, 0] = samples.uniform_(*self.cfg.ranges.height)
        self.link_height_command_b[env_ids, 0] = samples.uniform_(
            *self.cfg.ranges.link_height
        )
        if self.cfg.heading_command:
            self.heading_target[env_ids] = samples.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = (
                samples.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        self.is_standing_env[env_ids] = (
            samples.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

    def _update_command(self) -> None:
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_w[env_ids, 2] = torch.clamp(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        self.vel_command_w[self.is_standing_env] = 0.0


class UniformVelocityWorldHeightVelVisCommand(UniformVelocityWorldHeightVelCommand):
    """Compatibility name for the teacher's visualization-enabled generator."""


class UniformVelocityWorldHeightVelTargetVectorCommand(
    UniformVelocityWorldHeightVelCommand
):
    """Add a world-frame target-vector to the centered-bar command.

    The configured yaw interval is relative to the robot heading only while it
    is sampled. The stored yaw and the derived unit vector are world-frame
    quantities for the entire command interval.
    """

    def __init__(self, cfg: UniformVelocityHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.target_vector_yaw_w = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def target_vector_w(self) -> torch.Tensor:
        return target_vector_from_yaw(self.target_vector_yaw_w)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        offsets = torch.empty(len(ids), device=self.device).uniform_(
            *self.env.cfg.bar_controller.target_yaw_offset_range
        )
        self.target_vector_yaw_w[ids] = math_utils.wrap_to_pi(
            self.robot.data.heading_w[ids] + offsets
        )


class UniformEEPoseCommandWorldFollowingQuatVel(CommandTerm):
    """Emit the fixed grasp pose used while following the partner support."""

    def __init__(self, cfg: UniformEEPoseCommandWorldCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        self.left_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.right_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.left_xyz = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_xyz = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            (self.left_quat, self.right_quat, self.left_xyz, self.right_xyz), dim=1
        )

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        count = len(env_ids)
        grasp_cfg = self.env.cfg.experiment.grasp
        self.left_quat[env_ids] = torch.as_tensor(
            grasp_cfg.left_palm_quat, device=self.device
        ).expand(count, -1)
        self.right_quat[env_ids] = torch.as_tensor(
            grasp_cfg.right_palm_quat, device=self.device
        ).expand(count, -1)
        self.left_xyz[env_ids] = torch.as_tensor(
            grasp_cfg.left_palm_xyz, device=self.device
        ).expand(count, -1)
        self.right_xyz[env_ids] = torch.as_tensor(
            grasp_cfg.right_palm_xyz, device=self.device
        ).expand(count, -1)

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError


class UniformEEPoseCommandWorldQuat_Straight(CommandTerm):
    """Command generator that generates a velocity command in SE(2) from uniform distribution in the world frame.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the **world frame**.
    """
    cfg: UniformEEPoseCommandQuatCfg

    def __init__(self, cfg: UniformEEPoseCommandQuatCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        self.left_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.right_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.left_xyz = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_xyz = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose_trajectory_cfg = env.cfg.experiment.pose_trajectory
        self.num_waypoints = self.pose_trajectory_cfg.num_waypoints
        self.resample_count = 0

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = 'UniformEEPoseCommandWorld:\n'
        msg += f'\tCommand dimension: {tuple(self.command.shape[1:])}\n'
        msg += f'\tResampling time range: {self.cfg.resampling_time_range}\n'
        return msg
    '\n    Properties\n    '

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the world frame. Shape is (num_envs, 3)."""
        return torch.cat([self.left_quat, self.right_quat, self.left_xyz, self.right_xyz], dim=1)
    '\n    Implementation specific functions.\n    '

    def _update_metrics(self):
        pass

    def sample_quaternion_in_cone(self, base_quat, max_angle_rad):
        batch_size = base_quat.shape[0]
        device = base_quat.device
        axis = torch.randn(batch_size, 3, device=device)
        axis = axis / torch.norm(axis, dim=-1, keepdim=True)
        u = torch.rand(batch_size, device=device)
        angle = max_angle_rad * torch.pow(u, 1 / 3)
        half_angle = angle / 2
        w = torch.cos(half_angle)
        xyz = axis * torch.sin(half_angle).unsqueeze(-1)
        delta_quat = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
        result_quat = math_utils.quat_mul(delta_quat, base_quat)
        return result_quat

    def interpolate_quaternion_trajectory(self, q_start, q_end, num_points=8):
        """Interpolate batched quaternion trajectories on the GPU.

        Args:
            q_start: Starting quaternions with shape ``(N, 4)``.
            q_end: Ending quaternions with shape ``(N, 4)``.
            num_points: Number of interpolation samples.

        Returns:
            Interpolated quaternions with shape ``(num_points, N, 4)``.
        """
        batch_size = q_start.shape[0]
        device = q_start.device
        q_start = q_start / torch.norm(q_start, dim=-1, keepdim=True)
        q_end = q_end / torch.norm(q_end, dim=-1, keepdim=True)
        dot = torch.sum(q_start * q_end, dim=-1)
        q_end = torch.where(dot.unsqueeze(-1) < 0, -q_end, q_end)
        dot = torch.abs(dot)
        dot = torch.clamp(dot, 0.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        t_values = torch.linspace(0, 1, num_points, device=device)
        t_grid = t_values.unsqueeze(0).expand(batch_size, -1)
        theta_grid = theta.unsqueeze(-1).expand(-1, num_points)
        sin_theta_grid = sin_theta.unsqueeze(-1).expand(-1, num_points)
        weight1 = torch.sin((1 - t_grid) * theta_grid) / sin_theta_grid
        weight2 = torch.sin(t_grid * theta_grid) / sin_theta_grid
        linear_mask = sin_theta_grid < 1e-06
        weight1 = torch.where(linear_mask, 1 - t_grid, weight1)
        weight2 = torch.where(linear_mask, t_grid, weight2)
        q_start_expanded = q_start.unsqueeze(1).expand(-1, num_points, -1)
        q_end_expanded = q_end.unsqueeze(1).expand(-1, num_points, -1)
        trajectory = weight1.unsqueeze(-1) * q_start_expanded + weight2.unsqueeze(-1) * q_end_expanded
        trajectory = trajectory / torch.norm(trajectory, dim=-1, keepdim=True)
        return trajectory.transpose(0, 1)

    def interpolate_position_trajectory(self, pos_start, pos_end, num_points=8):
        """Linearly interpolate batched position trajectories.

        Args:
            pos_start: Starting positions with shape ``(N, 3)``.
            pos_end: Ending positions with shape ``(N, 3)``.
            num_points: Number of samples, including both endpoints.

        Returns:
            Interpolated positions with shape ``(num_points, N, 3)``.
        """
        batch_size = pos_start.shape[0]
        device = pos_start.device
        t_values = torch.linspace(0, 1, num_points, device=device)
        t_grid = t_values.unsqueeze(0).unsqueeze(-1).expand(batch_size, num_points, 3)
        pos_start_expanded = pos_start.unsqueeze(1).expand(-1, num_points, -1)
        pos_end_expanded = pos_end.unsqueeze(1).expand(-1, num_points, -1)
        trajectory = (1 - t_grid) * pos_start_expanded + t_grid * pos_end_expanded
        return trajectory.transpose(0, 1)

    def sample_positions_fully_vectorized(self, center_pos, cube_size):
        """Sample two batched positions around each center.

        Args:
            center_pos: Center positions with shape ``(N, 3)``.
            cube_size: Maximum offset magnitude along each axis.

        Returns:
            Two sampled position tensors, each with shape ``(N, 3)``.
        """
        batch_size = center_pos.shape[0]
        device = center_pos.device
        dtype = center_pos.dtype
        random_offset1 = torch.rand(batch_size, 3, device=device, dtype=dtype)
        random_offset2 = torch.rand(batch_size, 3, device=device, dtype=dtype)
        random_offset1 = (random_offset1 * 2 - 1) * cube_size
        random_offset2 = (random_offset2 * 2 - 1) * cube_size
        random_offset1[:, 1] *= self.pose_trajectory_cfg.lateral_randomization_scale
        random_offset2[:, 1] *= self.pose_trajectory_cfg.lateral_randomization_scale
        random_offset1[:, 2] *= self.pose_trajectory_cfg.vertical_randomization_scale
        random_offset2[:, 2] *= self.pose_trajectory_cfg.vertical_randomization_scale
        pos1 = center_pos + random_offset1
        pos2 = center_pos + random_offset2
        return (pos1, pos2)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Official IsaacLab 2.3.2/Isaac Sim 5.1 increments command_counter
        # after this callback.  A freshly reset command therefore has counter
        # zero here; sampling at one would leave the initial target zero/stale.
        first_resample_mask = self.command_counter[env_ids] == 0
        if not hasattr(self, 'ee_traj_start'):
            self.ee_traj_start = torch.zeros(self.num_envs, 14, device=self.device)
            self.ee_traj_end = torch.zeros(self.num_envs, 14, device=self.device)
            self.ee_traj_waypoints = torch.zeros(self.num_waypoints, self.num_envs, 14, device=self.device)
            self.ee_traj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        first_env_ids = env_ids[first_resample_mask]
        n_first = len(first_env_ids)
        if n_first > 0:
            start = torch.zeros(n_first, 14, device=self.device)
            end = torch.zeros(n_first, 14, device=self.device)
            pose_cfg = self.pose_trajectory_cfg
            center_left_xyz = torch.as_tensor(pose_cfg.left_center_xyz, device=self.device).expand(n_first, -1)
            center_right_xyz = torch.as_tensor(pose_cfg.right_center_xyz, device=self.device).expand(n_first, -1)
            center_left_quat = torch.as_tensor(pose_cfg.left_center_quat, device=self.device).expand(n_first, -1)
            center_right_quat = torch.as_tensor(pose_cfg.right_center_quat, device=self.device).expand(n_first, -1)
            ee_pos_traj_left_start, ee_pos_traj_left_end = self.sample_positions_fully_vectorized(center_left_xyz, cube_size=self.cfg.ranges.cube_size)
            ee_pos_traj_right_start, ee_pos_traj_right_end = self.sample_positions_fully_vectorized(center_right_xyz, cube_size=self.cfg.ranges.cube_size)
            ee_quat_traj_left_start = self.sample_quaternion_in_cone(center_left_quat, max_angle_rad=self.cfg.ranges.orientation_cone_rad)
            ee_quat_traj_left_end = self.sample_quaternion_in_cone(center_left_quat, max_angle_rad=self.cfg.ranges.orientation_cone_rad)
            ee_quat_traj_right_start = self.sample_quaternion_in_cone(center_right_quat, max_angle_rad=self.cfg.ranges.orientation_cone_rad)
            ee_quat_traj_right_end = self.sample_quaternion_in_cone(center_right_quat, max_angle_rad=self.cfg.ranges.orientation_cone_rad)
            start = torch.cat([ee_quat_traj_left_start, ee_quat_traj_right_start, ee_pos_traj_left_start, ee_pos_traj_right_start], dim=1)
            end = torch.cat([ee_quat_traj_left_end, ee_quat_traj_right_end, ee_pos_traj_left_end, ee_pos_traj_right_end], dim=1)
            self.ee_traj_start[first_env_ids] = start
            self.ee_traj_end[first_env_ids] = end
            interpolated_quat_traj_left = self.interpolate_quaternion_trajectory(start[:, 0:4], end[:, 0:4], num_points=self.num_waypoints)
            interpolated_quat_traj_right = self.interpolate_quaternion_trajectory(start[:, 4:8], end[:, 4:8], num_points=self.num_waypoints)
            interpolated_pos_traj_left = self.interpolate_position_trajectory(start[:, 8:11], end[:, 8:11], num_points=self.num_waypoints)
            interpolated_pos_traj_right = self.interpolate_position_trajectory(start[:, 11:14], end[:, 11:14], num_points=self.num_waypoints)
            self.ee_traj_waypoints[:, first_env_ids, :] = torch.cat([interpolated_quat_traj_left, interpolated_quat_traj_right, interpolated_pos_traj_left, interpolated_pos_traj_right], dim=-1)
            self.ee_traj_idx[first_env_ids] = 0
            self.left_quat[first_env_ids] = self.ee_traj_waypoints[0][first_env_ids][:, 0:4]
            self.right_quat[first_env_ids] = self.ee_traj_waypoints[0][first_env_ids][:, 4:8]
            self.left_xyz[first_env_ids] = self.ee_traj_waypoints[0][first_env_ids][:, 8:11]
            self.right_xyz[first_env_ids] = self.ee_traj_waypoints[0][first_env_ids][:, 11:14]
        not_first_env_ids = env_ids[~first_resample_mask]
        if len(not_first_env_ids) > 0:
            idx = self.ee_traj_idx[not_first_env_ids]
            self.ee_traj_idx[not_first_env_ids] = (self.ee_traj_idx[not_first_env_ids] + 1) % self.num_waypoints
            idx = self.ee_traj_idx[not_first_env_ids]
            self.left_quat[not_first_env_ids] = torch.stack([self.ee_traj_waypoints[idx[j]][eid][0:4] for j, eid in enumerate(not_first_env_ids)])
            self.right_quat[not_first_env_ids] = torch.stack([self.ee_traj_waypoints[idx[j]][eid][4:8] for j, eid in enumerate(not_first_env_ids)])
            self.left_xyz[not_first_env_ids] = torch.stack([self.ee_traj_waypoints[idx[j]][eid][8:11] for j, eid in enumerate(not_first_env_ids)])
            self.right_xyz[not_first_env_ids] = torch.stack([self.ee_traj_waypoints[idx[j]][eid][11:14] for j, eid in enumerate(not_first_env_ids)])

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError
