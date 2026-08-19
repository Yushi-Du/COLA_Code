"""Reset, tracking, grasp, effort, and jitter terms for collaboration tasks."""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
import isaaclab.utils.math as math_utils

from legged_lab.utils.bar_controller import target_vector_from_yaw, wrap_to_pi
from legged_lab.utils.bar_geometry import (
    positive_cosine_alignment_reward,
    positive_gaussian_tracking_reward,
    positive_mean_force_reward,
)
from legged_lab.utils.jitter import linear_detrended_variance

def _virtual_palm_pose(
    env,
    asset_cfg: SceneEntityCfg,
    wrist_to_palm_position: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[asset_cfg.name]
    wrist_position = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
    wrist_orientation = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    offset = torch.tensor(
        wrist_to_palm_position,
        device=asset.device,
        dtype=wrist_position.dtype,
    ).repeat(env.num_envs, 1)
    palm_position = wrist_position + math_utils.quat_apply(wrist_orientation, offset)
    return palm_position, wrist_orientation


def virtual_palm_quaternion_reward(
    env,
    std: float,
    deadband: float,
    command_start: int,
    wrist_to_palm_position: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Match the original palm-orientation reward using a virtual palm frame."""

    _, palm_orientation = _virtual_palm_pose(
        env, asset_cfg, wrist_to_palm_position
    )
    target_orientation_b = env.pose_command_generator.command[
        :, command_start : command_start + 4
    ]
    target_orientation_w = math_utils.quat_mul(
        env.robot.data.root_quat_w, target_orientation_b
    )
    error = math_utils.quat_error_magnitude(palm_orientation, target_orientation_w)
    error = torch.clamp(error - deadband, min=0.0)
    return torch.exp(-error / std**2)


def virtual_palm_position_reward(
    env,
    std: float,
    deadband: float,
    command_start: int,
    wrist_to_palm_position: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Match the original palm-position reward using a virtual palm frame."""

    palm_position, _ = _virtual_palm_pose(
        env, asset_cfg, wrist_to_palm_position
    )
    target_position_b = env.pose_command_generator.command[
        :, command_start : command_start + 3
    ]
    target_position_w = env.robot.data.root_pos_w + math_utils.quat_apply(
        env.robot.data.root_quat_w, target_position_b
    )
    error = torch.linalg.vector_norm(target_position_w - palm_position, dim=1)
    error = torch.clamp(error - deadband, min=0.0)
    return torch.exp(-error / std**2)


__all__ = ["virtual_palm_position_reward", "virtual_palm_quaternion_reward"]


def reset_robot_and_bar_uniform(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_range: tuple[float, float] | None = None,
    joint_velocity_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset G1 and preserve the configured robot-to-bar relative pose.

    The two possible Y-side placements mirror the configured negative-Y pose;
    the robot and bar then receive the same sampled root transform.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    bar: RigidObject = env.scene["carried_bar"]
    robot_states = robot.data.default_root_state[env_ids].clone()
    bar_states = bar.data.default_root_state[env_ids].clone()
    reset_cfg = env.cfg.experiment.support_reset
    controller_cfg = env.cfg.bar_controller

    signs = torch.where(
        torch.rand(len(env_ids), device=robot.device)
        < reset_cfg.positive_y_probability,
        1.0,
        -1.0,
    )

    initial_center = torch.tensor(
        controller_cfg.initial_center_position, device=robot.device
    ).repeat(len(env_ids), 1)
    initial_center[:, 1] = torch.abs(initial_center[:, 1]) * signs
    initial_center[:, 0] += math_utils.sample_uniform(
        *reset_cfg.root_x_jitter_range, (len(env_ids),), robot.device
    )
    initial_center[:, 1] += math_utils.sample_uniform(
        *reset_cfg.root_y_jitter_range, (len(env_ids),), robot.device
    )

    pose_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=robot.device,
    )
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), robot.device
    )
    robot_positions = (
        robot_states[:, :3] + env.scene.env_origins[env_ids] + pose_samples[:, :3]
    )
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    robot_orientations = math_utils.quat_mul(robot_states[:, 3:7], orientation_delta)

    relative_center = initial_center - robot_states[:, :3]
    bar_positions = robot_positions + math_utils.quat_apply(
        robot_orientations, relative_center
    )
    yaw = torch.where(signs > 0.0, torch.full_like(signs, math.pi), torch.zeros_like(signs))
    zeros = torch.zeros_like(yaw)
    side_orientation = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    bar_orientations = math_utils.quat_mul(robot_orientations, side_orientation)

    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=robot.device,
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), robot.device
    )
    robot_velocities = robot_states[:, 7:13] + velocity_samples
    bar_velocities = bar_states[:, 7:13] + velocity_samples

    robot.write_root_pose_to_sim(
        torch.cat((robot_positions, robot_orientations), dim=-1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(robot_velocities, env_ids=env_ids)
    bar.write_root_pose_to_sim(
        torch.cat((bar_positions, bar_orientations), dim=-1), env_ids=env_ids
    )
    bar.write_root_velocity_to_sim(bar_velocities, env_ids=env_ids)


def _active_bar_mask(env) -> torch.Tensor:
    if not hasattr(env, "no_object_mask"):
        return torch.ones(env.num_envs, device=env.device)
    return (~env.no_object_mask).float()


def endpoint_z_velocity_penalty(env, threshold: float = 0.15) -> torch.Tensor:
    """Match the original top-marker vertical-velocity penalty at the endpoint."""

    endpoint_velocity = env.get_human_endpoint_state()[1]
    return torch.clamp(threshold - endpoint_velocity[:, 2], min=0.0) * _active_bar_mask(env)


def bar_endpoint_height_difference_penalty(env) -> torch.Tensor:
    """Penalize tilt using the vertical difference between both bar ends."""

    human_position = env.get_human_endpoint_state()[0]
    robot_position = env.get_robot_endpoint_position()
    return torch.abs(robot_position[:, 2] - human_position[:, 2]) * _active_bar_mask(env)


def human_effort_reward(
    env,
    force_scale: float,
    epsilon: float,
) -> torch.Tensor:
    """Reward low episode-mean force applied by the human-end controllers.

    Force is integrated inside the 400 Hz controller loop. This term is
    evaluated by the reward manager at 50 Hz and stays in ``(0, 1]`` for
    active-bar environments, so an agent cannot avoid a stream of negative
    effort costs by ending an episode.
    """

    reward = positive_mean_force_reward(
        env.human_effort_force_time_integral,
        env.human_effort_elapsed_time,
        force_scale=force_scale,
        epsilon=epsilon,
    )
    return reward * _active_bar_mask(env)


def _all_controller_references_settled(
    env, settled_tolerance: float
) -> torch.Tensor:
    """Return environments in which no controller command is still slewing."""

    controller = env.bar_controller
    height_settled = torch.abs(
        env.controller_requested_height_w - controller.height_reference
    ) <= settled_tolerance
    velocity_settled = torch.all(
        torch.abs(
            env.controller_requested_velocity_xy - controller.velocity_reference_xy
        )
        <= settled_tolerance,
        dim=1,
    )
    yaw_settled = torch.abs(
        wrap_to_pi(env.controller_requested_yaw_w - controller.yaw_reference_w)
    ) <= settled_tolerance
    return height_settled & velocity_settled & yaw_settled


def bar_center_height_tracking_reward(env, std: float) -> torch.Tensor:
    """Track the requested world height at the bar center of mass."""

    center_height = env.get_controller_point_state()[0][:, 2]
    error = env.controller_requested_height_w - center_height
    return positive_gaussian_tracking_reward(error, std=std) * _active_bar_mask(env)


def bar_center_horizontal_velocity_tracking_reward(env, std: float) -> torch.Tensor:
    """Track the requested world-frame XY velocity at the bar center."""

    center_velocity_xy = env.get_controller_point_state()[1][:, :2]
    error = env.controller_requested_velocity_xy - center_velocity_xy
    return positive_gaussian_tracking_reward(
        error, std=std, vector_dim=1
    ) * _active_bar_mask(env)


def target_bar_vector_alignment_reward(env, epsilon: float) -> torch.Tensor:
    """Align the world-frame bar-vector with its commanded target-vector."""

    reward = positive_cosine_alignment_reward(
        env.get_bar_vector_w(),
        env.get_target_vector_w(),
        epsilon=epsilon,
    )
    return reward * _active_bar_mask(env)


class _SettledRollingVariancePenalty(ManagerTermBase):
    """Base class for command-aware rolling jitter penalties."""

    signal_width: int

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        history_length = int(cfg.params["history_length"])
        if history_length < 2:
            raise ValueError("history_length must be at least two")
        self.history_length = history_length
        self.history = torch.zeros(
            env.num_envs,
            history_length,
            self.signal_width,
            device=env.device,
        )
        self.valid_samples = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.cursor = 0

    def reset(self, env_ids=None):
        if env_ids is None:
            self.history.zero_()
            self.valid_samples.zero_()
            self.cursor = 0
            return
        self.history[env_ids] = 0.0
        self.valid_samples[env_ids] = 0

    def _rolling_variance(
        self,
        signal: torch.Tensor,
        settled: torch.Tensor,
        maximum_penalty: float,
    ) -> torch.Tensor:
        unsettled = ~settled
        self.history[unsettled] = 0.0
        self.valid_samples[unsettled] = 0
        self.history[:, self.cursor] = torch.where(
            settled.unsqueeze(1), signal, torch.zeros_like(signal)
        )
        self.valid_samples.copy_(
            torch.where(
                settled,
                torch.clamp(self.valid_samples + 1, max=self.history_length),
                torch.zeros_like(self.valid_samples),
            )
        )
        self.cursor = (self.cursor + 1) % self.history_length

        variance = torch.var(self.history, dim=1, unbiased=False).sum(dim=1)
        ready = self.valid_samples >= self.history_length
        return (
            torch.clamp(variance, max=maximum_penalty)
            * ready
            * _active_bar_mask(self._env)
        )


class bar_translational_jitter_penalty(_SettledRollingVariancePenalty):
    """Penalize settled-command variation in normalized bar velocity error."""

    signal_width = 3

    def __call__(
        self,
        env,
        history_length: int,
        horizontal_velocity_scale: float,
        vertical_velocity_scale: float,
        settled_tolerance: float,
        maximum_penalty: float,
    ) -> torch.Tensor:
        del history_length
        controller = env.bar_controller
        center_velocity = env.get_controller_point_state()[1]
        reference_velocity = torch.cat(
            (
                controller.velocity_reference_xy,
                controller.height_reference_velocity.unsqueeze(1),
            ),
            dim=1,
        )
        scales = center_velocity.new_tensor(
            [
                horizontal_velocity_scale,
                horizontal_velocity_scale,
                vertical_velocity_scale,
            ]
        )
        signal = (center_velocity - reference_velocity) / scales
        settled = _all_controller_references_settled(env, settled_tolerance)
        return self._rolling_variance(signal, settled, maximum_penalty)


class bar_vector_rate_jitter_penalty(_SettledRollingVariancePenalty):
    """Penalize settled-target variation of the world-frame bar-vector rate."""

    signal_width = 3

    def __call__(
        self,
        env,
        history_length: int,
        vector_rate_scale: float,
        settled_tolerance: float,
        maximum_penalty: float,
    ) -> torch.Tensor:
        del history_length
        controller = env.bar_controller
        bar_vector = env.get_bar_vector_w()
        unit_bar_vector = bar_vector / torch.clamp(
            torch.linalg.vector_norm(bar_vector, dim=1, keepdim=True), min=1.0e-8
        )
        angular_velocity = env.carried_bar.data.root_com_ang_vel_w
        measured_vector_rate = torch.linalg.cross(
            angular_velocity, unit_bar_vector, dim=1
        )

        reference_vector = target_vector_from_yaw(controller.yaw_reference_w)
        world_z = torch.zeros_like(reference_vector)
        world_z[:, 2] = 1.0
        reference_vector_rate = (
            controller.yaw_reference_rate_w.unsqueeze(1)
            * torch.linalg.cross(world_z, reference_vector, dim=1)
        )
        signal = (measured_vector_rate - reference_vector_rate) / vector_rate_scale
        settled = _all_controller_references_settled(env, settled_tolerance)
        return self._rolling_variance(signal, settled, maximum_penalty)


class waist_roll_position_jitter_penalty(ManagerTermBase):
    """Penalize high-frequency waist-roll motion, not offset or linear drift."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        history_length = int(cfg.params["history_length"])
        position_scale = float(cfg.params["position_scale"])
        if history_length < 3:
            raise ValueError("history_length must be at least three")
        if position_scale <= 0.0:
            raise ValueError("position_scale must be positive")
        self.history_length = history_length
        self.history = torch.zeros(
            env.num_envs, history_length, 1, device=env.device
        )
        self.valid_samples = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.cursor = 0

    def reset(self, env_ids=None):
        if env_ids is None:
            self.history.zero_()
            self.valid_samples.zero_()
            self.cursor = 0
            return
        self.history[env_ids] = 0.0
        self.valid_samples[env_ids] = 0

    def __call__(
        self,
        env,
        history_length: int,
        position_scale: float,
        maximum_penalty: float,
        asset_cfg,
    ) -> torch.Tensor:
        del history_length
        joint_position = env.scene[asset_cfg.name].data.joint_pos[
            :, asset_cfg.joint_ids
        ]
        if joint_position.shape[1] != 1:
            raise ValueError("waist-roll jitter reward requires exactly one joint")
        self.history[:, self.cursor] = joint_position
        self.valid_samples.copy_(
            torch.clamp(self.valid_samples + 1, max=self.history_length)
        )
        self.cursor = (self.cursor + 1) % self.history_length
        chronological = torch.roll(self.history, shifts=-self.cursor, dims=1)
        variance = linear_detrended_variance(
            chronological, signal_scale=position_scale
        )
        ready = self.valid_samples >= self.history_length
        return torch.clamp(variance, max=maximum_penalty) * ready


class _ControllerForceDetrendedVariancePenalty(ManagerTermBase):
    """Penalize 400 Hz force oscillation after removing a linear trend."""

    component_slice: slice

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        history_length = int(cfg.params["history_length"])
        settled_tolerance = float(cfg.params["settled_tolerance"])
        force_scale = float(cfg.params["force_scale"])
        if force_scale <= 0.0:
            raise ValueError("force_scale must be positive")
        env.configure_controller_force_jitter_tracking(
            history_length, settled_tolerance
        )

    def __call__(
        self,
        env,
        history_length: int,
        force_scale: float,
        settled_tolerance: float,
        maximum_penalty: float,
    ) -> torch.Tensor:
        del history_length, settled_tolerance
        history, valid_samples = env.get_controller_force_jitter_history()
        signal = history[:, :, self.component_slice]
        detrended_variance = linear_detrended_variance(
            signal, signal_scale=force_scale
        )
        ready = valid_samples >= signal.shape[1]
        return (
            torch.clamp(detrended_variance, max=maximum_penalty)
            * ready
            * _active_bar_mask(env)
        )


class height_controller_force_jitter_penalty(
    _ControllerForceDetrendedVariancePenalty
):
    """Penalize high-frequency variation of the applied vertical force."""

    component_slice = slice(2, 3)


class horizontal_controller_force_jitter_penalty(
    _ControllerForceDetrendedVariancePenalty
):
    """Penalize high-frequency variation of the applied world-XY force."""

    component_slice = slice(0, 2)


__all__ = [
    "bar_translational_jitter_penalty",
    "bar_vector_rate_jitter_penalty",
    "waist_roll_position_jitter_penalty",
    "height_controller_force_jitter_penalty",
    "horizontal_controller_force_jitter_penalty",
    "bar_center_height_tracking_reward",
    "bar_center_horizontal_velocity_tracking_reward",
    "target_bar_vector_alignment_reward",
]


__all__ = [
    "bar_center_height_tracking_reward",
    "bar_center_horizontal_velocity_tracking_reward",
    "bar_endpoint_height_difference_penalty",
    "bar_translational_jitter_penalty",
    "bar_vector_rate_jitter_penalty",
    "endpoint_z_velocity_penalty",
    "height_controller_force_jitter_penalty",
    "horizontal_controller_force_jitter_penalty",
    "human_effort_reward",
    "reset_robot_and_bar_uniform",
    "target_bar_vector_alignment_reward",
    "virtual_palm_position_reward",
    "virtual_palm_quaternion_reward",
    "waist_roll_position_jitter_penalty",
]
