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
        self._command_revision = 0
        self._outcome = "known-new"
        self._stage = "simulation-state"

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self._world.slm_shape_yx

    def apply_phase(self, radians: object) -> np.ndarray:
        self._command_revision += 1
        try:
            commanded = self._world.apply_slm_phase(radians)
        except BaseException:
            self._outcome = "known-old"
            self._stage = "validation"
            raise
        self._outcome = "known-new"
        self._stage = "simulation-applied"
        return commanded

    @property
    def last_commanded_phase(self) -> np.ndarray:
        return self._world.commanded_phase

    @property
    def command_revision(self) -> int:
        return self._command_revision

    @property
    def mapping_revision(self) -> int:
        return 0

    @property
    def last_command_receipt(self) -> dict[str, object]:
        return {
            "transport": "virtual",
            "identity": self.identity,
            "profile": "simulation",
            "model": "SimulationWorld",
            "serial": self.identity,
            "wavelength_nm": None,
            "flip_x": False,
            "flip_y": False,
            "correction_path": "",
            "correction_enabled": False,
            "mapping_revision": 0,
            "settle_seconds": 0.0,
            "phase_curve_source": "simulation",
            "outcome": self._outcome,
            "command_revision": self._command_revision,
            "stage": self._stage,
            "readback": "simulation-state",
        }

    def close(self) -> None:
        # Closing an editor or session is not an optical blank command.  The
        # last explicit phase remains the simulated hardware's last command.
        return None


__all__ = ["VirtualSLM"]
