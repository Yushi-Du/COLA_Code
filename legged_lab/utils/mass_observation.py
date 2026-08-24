"""Per-episode object-mass observations for COLA distillation tasks."""

from __future__ import annotations

import torch


def validate_uniform_range(
    value_range: tuple[float, float], *, name: str
) -> tuple[float, float]:
    """Return an ordered finite uniform range or raise a descriptive error."""

    low, high = (float(value_range[0]), float(value_range[1]))
    if not torch.isfinite(torch.tensor((low, high))).all():
        raise ValueError(f"{name} must contain finite values")
    if low > high:
        raise ValueError(f"{name} must be ordered low <= high, got {value_range}")
    return low, high


def sample_uniform_like(
    reference: torch.Tensor,
    value_range: tuple[float, float],
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a tensor-shaped uniform value on the reference device and dtype."""

    low, high = validate_uniform_range(value_range, name="uniform range")
    if low == high:
        return torch.full_like(reference, low)
    return torch.empty_like(reference).uniform_(low, high, generator=generator)


def noisy_mass_observation(
    true_mass_kg: torch.Tensor,
    bias_range_kg: tuple[float, float],
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one additive uniform bias and return unclipped mass observations."""

    bias_kg = sample_uniform_like(
        true_mass_kg,
        bias_range_kg,
        generator=generator,
    )
    return true_mass_kg + bias_kg, bias_kg


def no_object_mass_observation(
    reference: torch.Tensor,
    true_mass_range_kg: tuple[float, float],
    bias_range_kg: tuple[float, float],
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the active-bar mass-plus-bias distribution without creating a bar."""

    pseudo_true_mass_kg = sample_uniform_like(
        reference,
        true_mass_range_kg,
        generator=generator,
    )
    observation_kg, bias_kg = noisy_mass_observation(
        pseudo_true_mass_kg,
        bias_range_kg,
        generator=generator,
    )
    return observation_kg, pseudo_true_mass_kg, bias_kg


__all__ = [
    "no_object_mass_observation",
    "noisy_mass_observation",
    "sample_uniform_like",
    "validate_uniform_range",
]
