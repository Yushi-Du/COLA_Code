"""Pure-tensor helper for the phase-1 stillness objective."""

from __future__ import annotations

import math

import torch


def zero_velocity_command_stillness(
    root_linear_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    joint_velocity: torch.Tensor,
    current_action: torch.Tensor,
    previous_action: torch.Tensor,
    command: torch.Tensor,
    *,
    std: float,
    command_threshold: float,
) -> torch.Tensor:
    motion = torch.sum(torch.square(root_linear_velocity), dim=1)
    motion += torch.sum(torch.square(root_angular_velocity), dim=1)
    motion += torch.mean(torch.square(joint_velocity), dim=1)
    motion += torch.mean(torch.square(current_action - previous_action), dim=1)
    zero_velocity = (
        torch.linalg.vector_norm(command[:, :3], dim=1) <= command_threshold
    )
    return torch.exp(-motion / std**2) * zero_velocity.float()


def vector_angle_within_threshold(
    current_vector: torch.Tensor,
    target_vector: torch.Tensor,
    *,
    maximum_angle: float,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return a full-3D cosine mask for vector errors within an angle."""

    if not 0.0 <= maximum_angle <= math.pi:
        raise ValueError("maximum_angle must be in [0, pi]")
    denominator = torch.clamp(
        torch.linalg.vector_norm(current_vector, dim=-1)
        * torch.linalg.vector_norm(target_vector, dim=-1),
        min=epsilon,
    )
    cosine_similarity = torch.sum(
        current_vector * target_vector, dim=-1
    ) / denominator
    return cosine_similarity >= math.cos(maximum_angle)


def gate_stillness_by_vector_alignment(
    stillness_reward: torch.Tensor,
    current_vector: torch.Tensor,
    target_vector: torch.Tensor,
    *,
    maximum_angle: float,
    no_object_mask: torch.Tensor | None = None,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Gate active-object stillness while preserving no-object standing."""

    aligned = vector_angle_within_threshold(
        current_vector,
        target_vector,
        maximum_angle=maximum_angle,
        epsilon=epsilon,
    )
    if no_object_mask is not None:
        aligned = aligned | no_object_mask
    return stillness_reward * aligned.float()


__all__ = [
    "gate_stillness_by_vector_alignment",
    "vector_angle_within_threshold",
    "zero_velocity_command_stillness",
]
