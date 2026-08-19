"""Pure tensor geometry used by the fixed-bar collaboration tasks."""

from __future__ import annotations

import torch


def positive_gaussian_tracking_reward(
    error: torch.Tensor,
    *,
    std: float,
    vector_dim: int | None = None,
) -> torch.Tensor:
    """Return ``exp(-||error||^2/std^2)`` for scalar or vector errors."""

    if std <= 0.0:
        raise ValueError(f"std must be positive, received {std}.")
    squared_error = error.square()
    if vector_dim is not None:
        squared_error = torch.sum(squared_error, dim=vector_dim)
    return torch.exp(-squared_error / std**2)


def controller_force_effort(controller_force_w: torch.Tensor) -> torch.Tensor:
    """Return total horizontal-PID plus vertical-PD force effort in newtons."""

    if controller_force_w.shape[-1] != 3:
        raise ValueError(
            "Controller force must end in XYZ components, received shape "
            f"{tuple(controller_force_w.shape)}."
        )
    horizontal_effort = torch.linalg.vector_norm(controller_force_w[..., :2], dim=-1)
    vertical_effort = torch.abs(controller_force_w[..., 2])
    return horizontal_effort + vertical_effort


def positive_mean_force_reward(
    force_time_integral: torch.Tensor,
    elapsed_time: torch.Tensor,
    *,
    force_scale: float,
    epsilon: float,
) -> torch.Tensor:
    """Map episode-mean controller force to a positive decreasing reward."""

    if force_scale <= 0.0:
        raise ValueError(f"Force scale must be positive, received {force_scale}.")
    if epsilon <= 0.0:
        raise ValueError(f"Epsilon must be positive, received {epsilon}.")
    mean_force = force_time_integral / torch.clamp(elapsed_time, min=epsilon)
    return torch.exp(-mean_force / force_scale)


def reset_human_effort_statistics(
    force_time_integral: torch.Tensor,
    elapsed_time: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Reset selected environments without changing other running means."""

    if force_time_integral.shape != elapsed_time.shape:
        raise ValueError(
            "Force integral and elapsed time must have identical shapes, received "
            f"{tuple(force_time_integral.shape)} and {tuple(elapsed_time.shape)}."
        )
    force_time_integral[env_ids] = 0.0
    elapsed_time[env_ids] = 0.0


def environment_relative_root_state(
    root_state_w: torch.Tensor,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    """Return root state with position expressed relative to each environment.

    Isaac Lab orders ``root_state_w`` as world-frame position, quaternion,
    linear velocity, and angular velocity.  Only the position is translated;
    orientation and velocities remain in the world frame.  A clone prevents
    observation construction from modifying Isaac Lab's state buffer.
    """

    if root_state_w.shape[-1] != 13:
        raise ValueError(
            f"Expected a 13-D root state, received shape {tuple(root_state_w.shape)}."
        )
    if env_origins.shape != root_state_w.shape[:-1] + (3,):
        raise ValueError(
            "Environment origins must match the root-state batch dimensions: "
            f"root state {tuple(root_state_w.shape)}, origins {tuple(env_origins.shape)}."
        )

    relative_state = root_state_w.clone()
    relative_state[..., :3] -= env_origins
    return relative_state


def centered_bar_teacher_observation(
    object_state: torch.Tensor,
    object_mass: torch.Tensor,
    requested_height: torch.Tensor,
    requested_velocity_xy_w: torch.Tensor,
    target_vector_w: torch.Tensor,
) -> torch.Tensor:
    """Append the seven centered-bar privileged inputs to a 13-D bar state."""

    batch_shape = object_state.shape[:-1]
    expected_shapes = {
        "object_state": (13,),
        "object_mass": (1,),
        "requested_height": (1,),
        "requested_velocity_xy_w": (2,),
        "target_vector_w": (3,),
    }
    values = {
        "object_state": object_state,
        "object_mass": object_mass,
        "requested_height": requested_height,
        "requested_velocity_xy_w": requested_velocity_xy_w,
        "target_vector_w": target_vector_w,
    }
    for name, value in values.items():
        expected = batch_shape + expected_shapes[name]
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}, received {tuple(value.shape)}."
            )
    return torch.cat(tuple(values.values()), dim=-1)


def bar_vector_from_endpoints(
    robot_endpoint_w: torch.Tensor,
    human_endpoint_w: torch.Tensor,
) -> torch.Tensor:
    """Return the world-frame vector from the robot end to the human end."""

    return human_endpoint_w - robot_endpoint_w


def cosine_deviation(
    current_vector: torch.Tensor,
    reference_vector: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return ``1 - cosine_similarity`` for corresponding vector batches.

    The result is zero for aligned vectors, one for perpendicular vectors, and
    two for opposite vectors. Vector magnitude does not affect the result.
    """

    current_norm = torch.linalg.vector_norm(current_vector, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference_vector, dim=-1)
    denominator = torch.clamp(current_norm * reference_norm, min=epsilon)
    similarity = torch.sum(current_vector * reference_vector, dim=-1) / denominator
    return 1.0 - torch.clamp(similarity, min=-1.0, max=1.0)


def positive_cosine_alignment_reward(
    current_vector: torch.Tensor,
    reference_vector: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return a non-negative reward for alignment with the reference vector.

    This is the positive affine counterpart of :func:`cosine_deviation`:
    aligned, perpendicular, and opposite vectors score two, one, and zero.
    It therefore preserves the original cosine objective while avoiding a
    negative per-step reward.
    """

    return 2.0 - cosine_deviation(
        current_vector,
        reference_vector,
        epsilon=epsilon,
    )
