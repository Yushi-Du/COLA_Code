"""Isaac Sim entry-point lifecycle helpers."""

from __future__ import annotations

from collections.abc import Callable
import os
import sys
import traceback
from typing import Any


def run_with_simulation_app(simulation_app: Any, entrypoint: Callable[[], None]) -> None:
    """Run an entry point and terminate Kit without hiding Python failures.

    Isaac Sim 5.1 cleanup can obscure an active exception and can stall when
    another Kit process owns the shared key-value database. Each COLA entry
    point is a standalone process, so fast cleanup is sufficient here.
    """

    try:
        entrypoint()
    except BaseException:
        # Isaac Sim's skip-cleanup path performs an immediate successful
        # process exit, so calling it here hides both the traceback and the
        # non-zero status.  Emit the complete failure first, then terminate
        # this standalone worker process with an unambiguous error code.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close(skip_cleanup=True)
