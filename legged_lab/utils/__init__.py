"""Utility package with lazy simulator-dependent imports.

Keeping ``task_registry`` lazy lets command-line arguments be defined before
Isaac Sim's ``SimulationApp`` starts, which is required by Isaac Sim 5.1.
"""

from __future__ import annotations

from typing import Any

__all__ = ["task_registry"]


def __getattr__(name: str) -> Any:
    if name == "task_registry":
        from .task_registry import task_registry

        return task_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
