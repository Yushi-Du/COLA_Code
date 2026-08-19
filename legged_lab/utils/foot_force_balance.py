"""Pure tensor helpers for balancing the vertical loads on two feet."""

from __future__ import annotations

import torch


def normalized_two_foot_load_imbalance(
    vertical_force_history: torch.Tensor,
    velocity_commands: torch.Tensor,
    *,
    contact_force_threshold: float,
    command_threshold: float,
    dead_zone: float,
) -> torch.Tensor:
    """Return a normalized two-foot load imbalance in ``[0, 1]``.

    The force input is expected to contain the world-frame vertical contact
    force for exactly two feet over a short physics-rate history, with shape
    ``(num_envs, history, 2)``. The penalty is active only during commanded
    standing and while both feet carry a meaningful load.
    """

    if vertical_force_history.ndim != 3 or vertical_force_history.shape[-1] != 2:
        raise ValueError(
            "vertical_force_history must have shape (num_envs, history, 2); "
            f"received {tuple(vertical_force_history.shape)}."
        )
    if (
        velocity_commands.ndim != 2
        or velocity_commands.shape[0] != vertical_force_history.shape[0]
    ):
        raise ValueError(
            "velocity_commands must have shape (num_envs, command_dims) and "
            "match the force batch size."
        )
    if velocity_commands.shape[1] < 3:
        raise ValueError(
            "velocity_commands must contain x, y, and yaw velocity commands."
        )
    if not 0.0 <= dead_zone < 1.0:
        raise ValueError(f"dead_zone must be in [0, 1); received {dead_zone}.")

    mean_vertical_force = vertical_force_history.clamp_min(0.0).mean(dim=1)
    total_force = mean_vertical_force.sum(dim=1)
    relative_difference = (
        torch.abs(mean_vertical_force[:, 0] - mean_vertical_force[:, 1])
        / total_force.clamp_min(torch.finfo(vertical_force_history.dtype).eps)
    )
    imbalance = ((relative_difference - dead_zone) / (1.0 - dead_zone)).clamp(
        min=0.0, max=1.0
    )
    imbalance = imbalance.square()

    both_feet_loaded = torch.all(
        mean_vertical_force > contact_force_threshold, dim=1
    )
    commanded_standing = torch.linalg.vector_norm(
        velocity_commands[:, :3], dim=1
    ) < command_threshold
    return imbalance * (both_feet_loaded & commanded_standing).to(imbalance.dtype)


__all__ = ["normalized_two_foot_load_imbalance"]
