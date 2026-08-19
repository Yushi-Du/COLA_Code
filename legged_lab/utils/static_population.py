"""Pure helpers for deterministic distributed static-topology populations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


FIXED_BAR_TOPOLOGY_ID = 0
NO_OBJECT_TOPOLOGY_ID = 1
RIGHT_FIXED_BAR_TOPOLOGY_ID = 2
NO_OBJECT_PRIVILEGED_DIM = 20


@dataclass(frozen=True)
class StaticPopulationAssignment:
    """Topology assigned to one distributed rank."""

    topology_id: int
    topology_name: str
    is_no_object: bool
    is_right_fixed: bool = False


def assign_static_population(
    global_rank: int,
    world_size: int,
    no_object_rank_count: int,
) -> StaticPopulationAssignment:
    """Assign the highest ranks to the static no-object population."""

    if world_size < 2:
        raise ValueError("static mixed-topology training requires at least two ranks")
    if not 0 < no_object_rank_count < world_size:
        raise ValueError(
            "no_object_rank_count must be between one and world_size - 1"
        )
    if not 0 <= global_rank < world_size:
        raise ValueError("global_rank must be in [0, world_size)")

    is_no_object = global_rank >= world_size - no_object_rank_count
    if is_no_object:
        return StaticPopulationAssignment(
            topology_id=NO_OBJECT_TOPOLOGY_ID,
            topology_name="no_object_fixed_hand",
            is_no_object=True,
        )
    return StaticPopulationAssignment(
        topology_id=FIXED_BAR_TOPOLOGY_ID,
        topology_name="left_fixed_bar",
        is_no_object=False,
    )


def assign_three_static_populations(
    global_rank: int,
    world_size: int,
    *,
    no_object_rank_count: int,
    right_fixed_rank_count: int,
) -> StaticPopulationAssignment:
    """Assign low ranks left-fixed, middle ranks right-fixed, and high ranks no-object."""

    if world_size < 3:
        raise ValueError("three static populations require at least three ranks")
    if not 0 <= global_rank < world_size:
        raise ValueError("global_rank must be in [0, world_size)")
    left_fixed_rank_count = (
        world_size - no_object_rank_count - right_fixed_rank_count
    )
    if min(
        left_fixed_rank_count,
        right_fixed_rank_count,
        no_object_rank_count,
    ) <= 0:
        raise ValueError("all three static populations require at least one rank")

    if global_rank < left_fixed_rank_count:
        return StaticPopulationAssignment(
            topology_id=FIXED_BAR_TOPOLOGY_ID,
            topology_name="left_fixed_bar",
            is_no_object=False,
        )
    if global_rank < left_fixed_rank_count + right_fixed_rank_count:
        return StaticPopulationAssignment(
            topology_id=RIGHT_FIXED_BAR_TOPOLOGY_ID,
            topology_name="right_fixed_bar",
            is_no_object=False,
            is_right_fixed=True,
        )
    return StaticPopulationAssignment(
        topology_id=NO_OBJECT_TOPOLOGY_ID,
        topology_name="no_object_fixed_hand",
        is_no_object=True,
    )


def static_no_object_privileged_tail(
    num_envs: int,
    device: str | torch.device,
    *,
    height_command: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the 20-D teacher tail for a scene that contains no object.

    Layout matches the fixed-middle teacher exactly: 13-D object root state,
    mass, height command, two horizontal velocity commands, and a three-D
    world-frame target vector.  A mass of zero is outside the real bar's
    randomized support and therefore identifies the static no-object topology.
    """

    if height_command.shape != (num_envs, 1):
        raise ValueError(
            "height_command must have shape "
            f"({num_envs}, 1), got {tuple(height_command.shape)}"
        )
    target_vector = torch.zeros(
        (num_envs, 3), device=device, dtype=dtype
    )
    target_vector[:, 1] = -1.0
    tail = torch.cat(
        (
            torch.zeros((num_envs, 13), device=device, dtype=dtype),
            torch.zeros((num_envs, 1), device=device, dtype=dtype),
            height_command.to(device=device, dtype=dtype),
            torch.zeros((num_envs, 2), device=device, dtype=dtype),
            target_vector,
        ),
        dim=1,
    )
    if tail.shape[1] != NO_OBJECT_PRIVILEGED_DIM:
        raise RuntimeError("static no-object privileged tail has wrong width")
    return tail


__all__ = [
    "FIXED_BAR_TOPOLOGY_ID",
    "NO_OBJECT_PRIVILEGED_DIM",
    "NO_OBJECT_TOPOLOGY_ID",
    "RIGHT_FIXED_BAR_TOPOLOGY_ID",
    "StaticPopulationAssignment",
    "assign_static_population",
    "assign_three_static_populations",
    "static_no_object_privileged_tail",
]
