"""Phase-only virtual SLM backed by the installation's simulation world."""

from __future__ import annotations

from threading import Event
import time
from typing import Callable

import numpy as np

from ...slm.device import phase_from_codes, phase_sequence_codes


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
        self._sequence = None
        self._sequence_cancel = Event()
        self._sequence_receipt = None

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self._world.slm_shape_yx

    def apply_phase(self, radians: object) -> np.ndarray:
        self._sequence = None
        self._sequence_receipt = None
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
        receipt = {
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
        if self._sequence_receipt is not None:
            receipt["sequence"] = self._sequence_receipt
        return receipt

    def prepare_phase_sequence(self, codes: object, frame_interval_seconds: object) -> dict[str, object]:
        started = time.perf_counter()
        frames, intervals = phase_sequence_codes(codes, self.shape_yx, frame_interval_seconds)
        if np.asarray(codes).flags.writeable:
            frames = np.frombuffer(frames.tobytes(), dtype=np.uint8).reshape(frames.shape)
        prepared = {"frame_count": len(frames), "frame_intervals_seconds": intervals.tolist(),
                    "prepare_ms": (time.perf_counter() - started) * 1000,
                    "upload_roundtrip_ms": 0.0, "mapping_revision": 0}
        self._sequence = (frames, intervals, self._command_revision, prepared)
        self._sequence_cancel.clear()
        return dict(prepared)

    def cancel_phase_sequence(self) -> None:
        self._sequence_cancel.set()

    def release_phase_sequence(self) -> None:
        self._sequence = None

    def play_phase_sequence(self, stop_requested: Callable[[], bool] | None = None) -> dict[str, object]:
        sequence, self._sequence = self._sequence, None
        if sequence is None:
            raise RuntimeError("SLM phase sequence has not been prepared")
        codes, intervals, revision, prepared = sequence
        if revision != self._command_revision:
            raise RuntimeError("stale prepared SLM sequence")
        started = time.perf_counter()
        dispatch, acknowledgments = [], []
        result = {**prepared, "played_frames": 0, "cancelled": False,
                  "acknowledgment": "simulation-state", "physical_vblank_observed": False,
                  "final_settle_ms": 0.0, "final_settle_completed": False}
        self._command_revision += 1
        failed = False
        try:
            for frame, interval in zip(codes, intervals):
                if self._sequence_cancel.is_set() or (stop_requested is not None and stop_requested()):
                    self._sequence_cancel.set()
                    break
                frame_started = time.perf_counter()
                dispatch.append((frame_started - started) * 1000)
                canonical = phase_from_codes(frame, self.shape_yx)
                self._world.apply_slm_phase(canonical)
                acknowledgments.append((time.perf_counter() - started) * 1000)
                self._outcome = "known-new"
                result["played_frames"] += 1
                deadline = frame_started + float(interval)
                while time.perf_counter() < deadline and not self._sequence_cancel.is_set():
                    if stop_requested is not None and stop_requested():
                        self._sequence_cancel.set()
                        break
                    self._sequence_cancel.wait(min(0.01, max(0.0, deadline - time.perf_counter())))
        except BaseException:
            failed = True
            self._outcome = "known-old"
            raise
        finally:
            result.update(cancelled=self._sequence_cancel.is_set(), dispatch_ms=dispatch,
                          acknowledged_ms=acknowledgments, actual_frame_intervals_ms=np.diff(dispatch).tolist(),
                          play_ms=(time.perf_counter() - started) * 1000)
            result["final_settle_completed"] = result["played_frames"] == len(codes) and not result["cancelled"]
            self._stage = "sequence-failed" if failed else "sequence-cancelled" if result["cancelled"] else "sequence-complete"
            self._sequence_receipt = result
        return {**result, "receipt": self.last_command_receipt}

    def close(self) -> None:
        # Closing an editor or session is not an optical blank command.  The
        # last explicit phase remains the simulated hardware's last command.
        self.cancel_phase_sequence()
        self.release_phase_sequence()


__all__ = ["VirtualSLM"]
