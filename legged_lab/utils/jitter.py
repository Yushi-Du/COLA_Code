"""Small tensor utilities shared by high-frequency jitter rewards."""

from __future__ import annotations

import torch


def linear_detrended_variance(
    history: torch.Tensor,
    signal_scale: float,
) -> torch.Tensor:
    """Return per-environment residual variance after a best linear fit."""

    if signal_scale <= 0.0:
        raise ValueError("signal_scale must be positive")
    time = torch.arange(
        history.shape[1], device=history.device, dtype=history.dtype
    )
    centered_time = time - torch.mean(time)
    normalized = history / signal_scale
    centered = normalized - torch.mean(normalized, dim=1, keepdim=True)
    slope = torch.sum(
        centered * centered_time.view(1, -1, 1), dim=1, keepdim=True
    ) / torch.sum(centered_time.square())
    residual = centered - slope * centered_time.view(1, -1, 1)
    return torch.mean(residual.square(), dim=1).sum(dim=1)


__all__ = ["linear_detrended_variance"]
