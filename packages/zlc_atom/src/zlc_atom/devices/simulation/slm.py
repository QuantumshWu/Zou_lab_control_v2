"""Phase-only virtual SLM backed by the installation's simulation world."""

from __future__ import annotations

import numpy as np


class VirtualSLM:
    """A device leaf with no phase state separate from ``SimulationWorld``."""

    def __init__(self, world: object, *, identity: str) -> None:
        apply = getattr(world, "apply_slm_phase", None)
        if not callable(apply):
            raise TypeError("virtual SLM requires a phase-capable SimulationWorld")
        value = str(identity).strip()
        if not value:
            raise ValueError("virtual SLM identity must be non-empty")
        self._world = world
        self._identity = value

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self._world.slm_shape_yx

    def apply_phase(self, radians: object) -> np.ndarray:
        return self._world.apply_slm_phase(radians)

    @property
    def last_commanded_phase(self) -> np.ndarray:
        return self._world.commanded_phase

    def close(self) -> None:
        # Closing an editor or session is not an optical blank command.  The
        # last explicit phase remains the simulated hardware's last command.
        return None


__all__ = ["VirtualSLM"]
