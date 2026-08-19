"""Batched 400 Hz controllers for the carried object."""

from __future__ import annotations

import math
from typing import Protocol

import torch


class BarControllerParameters(Protocol):
    """Parameters consumed by :class:`BarController`."""

    height_kp: float
    height_kd: float
    height_force_limit: float
    height_target_rate_limit: float
    velocity_kp: float
    velocity_ki: float
    velocity_kd: float
    horizontal_force_limit: float
    integral_force_limit: float
    derivative_cutoff_hz: float
    velocity_target_slew_limit: float
    velocity_error_deadband: float
    gravity_magnitude: float
    yaw_kp: float
    yaw_kd: float
    yaw_torque_limit: float
    yaw_target_rate_limit: float


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi, pi)`` without changing dtype or device."""

    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def target_vector_from_yaw(yaw_w: torch.Tensor) -> torch.Tensor:
    """Return horizontal world-frame unit vectors for batched yaw angles."""

    target = torch.zeros(*yaw_w.shape, 3, device=yaw_w.device, dtype=yaw_w.dtype)
    target[..., 0] = torch.cos(yaw_w)
    target[..., 1] = torch.sin(yaw_w)
    return target


def bar_vector_yaw_and_rate(
    bar_vector_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame azimuth and exact azimuth rate of a rigid vector.

    This intentionally does not use ``angular_velocity_w[:, 2]``.  For a
    tilted vector, its XY-projection azimuth follows ``v_dot = omega x v``.
    """

    vector_rate_w = torch.linalg.cross(angular_velocity_w, bar_vector_w, dim=-1)
    horizontal_norm_squared = bar_vector_w[:, 0].square() + bar_vector_w[:, 1].square()
    safe_norm_squared = torch.clamp(horizontal_norm_squared, min=1.0e-12)
    yaw_w = torch.atan2(bar_vector_w[:, 1], bar_vector_w[:, 0])
    yaw_rate_w = (
        bar_vector_w[:, 0] * vector_rate_w[:, 1]
        - bar_vector_w[:, 1] * vector_rate_w[:, 0]
    ) / safe_norm_squared
    return yaw_w, yaw_rate_w


def projected_world_yaw_axis(bar_vector_w: torch.Tensor) -> torch.Tensor:
    """Project world Z onto the plane normal to the bar's long axis."""

    unit_vector_w = bar_vector_w / torch.clamp(
        torch.linalg.vector_norm(bar_vector_w, dim=1, keepdim=True), min=1.0e-12
    )
    world_z = torch.zeros_like(unit_vector_w)
    world_z[:, 2] = 1.0
    return world_z - torch.sum(world_z * unit_vector_w, dim=1, keepdim=True) * unit_vector_w


class BarController:
    """Height PD, horizontal-velocity PID, and bar-vector-yaw PD.

    All inputs and outputs use the simulation world frame.  The force is
    intended for the bar center of mass.  The yaw torque is projected to have
    no component along the thin bar's long axis.
    """

    _GAIN_NAMES = (
        "height_kp",
        "height_kd",
        "velocity_kp",
        "velocity_ki",
        "velocity_kd",
        "yaw_kp",
        "yaw_kd",
    )

    def __init__(
        self,
        cfg: BarControllerParameters,
        *,
        num_envs: int,
        device: str | torch.device,
        dt: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("controller dt must be positive")
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = float(dt)
        self.dtype = dtype

        self.height_reference = self._zeros()
        self.height_reference_velocity = self._zeros()
        self.velocity_reference_xy = self._zeros(2)
        self.velocity_integral_xy = self._zeros(2)
        self.filtered_acceleration_xy = self._zeros(2)
        self.previous_center_velocity_xy = self._zeros(2)
        self.yaw_reference_w = self._zeros()
        self.yaw_reference_rate_w = self._zeros()

        self.force_w = self._zeros(3)
        self.torque_w = self._zeros(3)
        self.height_p_force = self._zeros()
        self.height_d_force = self._zeros()
        self.height_gravity_force = self._zeros()
        self.velocity_p_force_xy = self._zeros(2)
        self.velocity_i_force_xy = self._zeros(2)
        self.velocity_d_force_xy = self._zeros(2)
        self.current_bar_yaw_w = self._zeros()
        self.current_bar_yaw_rate_w = self._zeros()
        self.yaw_error = self._zeros()
        self.yaw_p_torque = self._zeros()
        self.yaw_d_torque = self._zeros()
        self.yaw_control_torque = self._zeros()
        self.yaw_axis_w = self._zeros(3)

        for name in self._GAIN_NAMES:
            gain = self._zeros()
            gain.fill_(float(getattr(cfg, name)))
            setattr(self, name, gain)

    def _zeros(self, width: int | None = None) -> torch.Tensor:
        shape = (self.num_envs,) if width is None else (self.num_envs, width)
        return torch.zeros(shape, device=self.device, dtype=self.dtype)

    @staticmethod
    def _clamp_vector_norm(values: torch.Tensor, limit: float) -> torch.Tensor:
        magnitude = torch.linalg.vector_norm(values, dim=1, keepdim=True)
        scale = torch.clamp(limit / torch.clamp(magnitude, min=1.0e-12), max=1.0)
        return values * scale

    def reset_gains(
        self,
        env_ids: torch.Tensor,
        *,
        randomize: bool,
        scale_range: tuple[float, float],
    ) -> None:
        """Set selected environments to nominal or independently randomized gains."""

        lower, upper = scale_range
        if lower <= 0.0 or upper < lower:
            raise ValueError(
                "controller gain scale range must satisfy 0 < lower <= upper, "
                f"received {scale_range}"
            )
        scales = torch.ones(
            (len(env_ids), len(self._GAIN_NAMES)),
            device=self.device,
            dtype=self.dtype,
        )
        if randomize:
            scales.uniform_(lower, upper)
        for index, name in enumerate(self._GAIN_NAMES):
            getattr(self, name)[env_ids] = (
                float(getattr(self.cfg, name)) * scales[:, index]
            )

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        center_position_w: torch.Tensor,
        center_velocity_w: torch.Tensor,
        bar_vector_w: torch.Tensor,
        angular_velocity_w: torch.Tensor,
    ) -> None:
        """Clear controller state and start references from measured state."""

        current_yaw, _ = bar_vector_yaw_and_rate(bar_vector_w, angular_velocity_w)
        self.height_reference[env_ids] = center_position_w[env_ids, 2]
        self.height_reference_velocity[env_ids] = 0.0
        self.velocity_reference_xy[env_ids] = center_velocity_w[env_ids, :2]
        self.velocity_integral_xy[env_ids] = 0.0
        self.filtered_acceleration_xy[env_ids] = 0.0
        self.previous_center_velocity_xy[env_ids] = center_velocity_w[env_ids, :2]
        self.yaw_reference_w[env_ids] = current_yaw[env_ids]
        self.yaw_reference_rate_w[env_ids] = 0.0
        self.force_w[env_ids] = 0.0
        self.torque_w[env_ids] = 0.0

    def compute(
        self,
        *,
        requested_height_w: torch.Tensor,
        requested_velocity_xy_w: torch.Tensor,
        requested_yaw_w: torch.Tensor,
        center_position_w: torch.Tensor,
        center_velocity_w: torch.Tensor,
        bar_vector_w: torch.Tensor,
        angular_velocity_w: torch.Tensor,
        bar_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance references and return the world-frame force and torque."""

        cfg = self.cfg

        maximum_height_delta = cfg.height_target_rate_limit * self.dt
        height_delta = torch.clamp(
            requested_height_w - self.height_reference,
            -maximum_height_delta,
            maximum_height_delta,
        )
        self.height_reference.add_(height_delta)
        self.height_reference_velocity.copy_(height_delta / self.dt)

        maximum_velocity_delta = cfg.velocity_target_slew_limit * self.dt
        self.velocity_reference_xy.add_(
            torch.clamp(
                requested_velocity_xy_w - self.velocity_reference_xy,
                -maximum_velocity_delta,
                maximum_velocity_delta,
            )
        )

        requested_yaw_delta = wrap_to_pi(requested_yaw_w - self.yaw_reference_w)
        maximum_yaw_delta = cfg.yaw_target_rate_limit * self.dt
        applied_yaw_delta = torch.clamp(
            requested_yaw_delta, -maximum_yaw_delta, maximum_yaw_delta
        )
        self.yaw_reference_w.copy_(wrap_to_pi(self.yaw_reference_w + applied_yaw_delta))
        self.yaw_reference_rate_w.copy_(applied_yaw_delta / self.dt)

        self.height_gravity_force.copy_(bar_mass * cfg.gravity_magnitude)
        height_error = self.height_reference - center_position_w[:, 2]
        self.height_p_force.copy_(self.height_kp * height_error)
        self.height_d_force.copy_(
            self.height_kd
            * (self.height_reference_velocity - center_velocity_w[:, 2])
        )
        height_force = (
            self.height_p_force + self.height_d_force + self.height_gravity_force
        ).clamp(-cfg.height_force_limit, cfg.height_force_limit)

        raw_acceleration_xy = (
            center_velocity_w[:, :2] - self.previous_center_velocity_xy
        ) / self.dt
        alpha = math.exp(-2.0 * math.pi * cfg.derivative_cutoff_hz * self.dt)
        self.filtered_acceleration_xy.mul_(alpha).add_(
            raw_acceleration_xy, alpha=1.0 - alpha
        )
        self.previous_center_velocity_xy.copy_(center_velocity_w[:, :2])

        velocity_error_xy = self.velocity_reference_xy - center_velocity_w[:, :2]
        integration_error_xy = velocity_error_xy.clone()
        integration_error_xy[
            torch.abs(integration_error_xy) < cfg.velocity_error_deadband
        ] = 0.0
        self.velocity_p_force_xy.copy_(
            self.velocity_kp.unsqueeze(1) * velocity_error_xy
        )
        self.velocity_d_force_xy.copy_(
            -self.velocity_kd.unsqueeze(1) * self.filtered_acceleration_xy
        )
        current_i_force_xy = self.velocity_ki.unsqueeze(1) * self.velocity_integral_xy
        unsaturated_force_xy = (
            self.velocity_p_force_xy + current_i_force_xy + self.velocity_d_force_xy
        )

        unsaturated_norm = torch.linalg.vector_norm(unsaturated_force_xy, dim=1)
        saturated = unsaturated_norm >= cfg.horizontal_force_limit
        force_direction = unsaturated_force_xy / torch.clamp(
            unsaturated_norm.unsqueeze(1), min=1.0e-12
        )
        outward_error = torch.sum(integration_error_xy * force_direction, dim=1)
        remove_outward = saturated & (outward_error > 0.0)
        integration_error_xy[remove_outward] -= (
            outward_error[remove_outward].unsqueeze(1) * force_direction[remove_outward]
        )

        if cfg.velocity_ki > 0.0:
            self.velocity_integral_xy.add_(integration_error_xy * self.dt)
            self.velocity_i_force_xy.copy_(
                self._clamp_vector_norm(
                    self.velocity_ki.unsqueeze(1) * self.velocity_integral_xy,
                    cfg.integral_force_limit,
                )
            )
            self.velocity_integral_xy.copy_(
                self.velocity_i_force_xy / self.velocity_ki.unsqueeze(1)
            )
        else:
            self.velocity_integral_xy.zero_()
            self.velocity_i_force_xy.zero_()
        horizontal_force_xy = self._clamp_vector_norm(
            self.velocity_p_force_xy
            + self.velocity_i_force_xy
            + self.velocity_d_force_xy,
            cfg.horizontal_force_limit,
        )

        current_yaw_w, current_yaw_rate_w = bar_vector_yaw_and_rate(
            bar_vector_w, angular_velocity_w
        )
        self.current_bar_yaw_w.copy_(current_yaw_w)
        self.current_bar_yaw_rate_w.copy_(current_yaw_rate_w)
        self.yaw_error.copy_(wrap_to_pi(self.yaw_reference_w - current_yaw_w))
        self.yaw_p_torque.copy_(self.yaw_kp * self.yaw_error)
        self.yaw_d_torque.copy_(
            self.yaw_kd * (self.yaw_reference_rate_w - current_yaw_rate_w)
        )
        self.yaw_control_torque.copy_(
            (self.yaw_p_torque + self.yaw_d_torque).clamp(
                -cfg.yaw_torque_limit, cfg.yaw_torque_limit
            )
        )
        self.yaw_axis_w.copy_(projected_world_yaw_axis(bar_vector_w))

        self.force_w[:, :2] = horizontal_force_xy
        self.force_w[:, 2] = height_force
        self.torque_w.copy_(self.yaw_control_torque.unsqueeze(1) * self.yaw_axis_w)
        return self.force_w, self.torque_w


__all__ = [
    "BarController",
    "bar_vector_yaw_and_rate",
    "projected_world_yaw_axis",
    "target_vector_from_yaw",
    "wrap_to_pi",
]
