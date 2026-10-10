"""Phase-only virtual SLM backed by the installation's simulation world."""

from __future__ import annotations

from threading import Event
from queue import Empty, Full, Queue
import time
from typing import Callable

import numpy as np

from ...slm.device import _same_phase_codes, phase_from_codes, phase_sequence_codes


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

    def prepare_phase_sequence(self, codes: object, frame_interval_seconds: object, *, frame_count: int | None = None) -> dict[str, object]:
        started = time.perf_counter()
        frames, intervals = phase_sequence_codes(codes, self.shape_yx, frame_interval_seconds, frame_count=frame_count)
        if frames is not None and np.asarray(codes).flags.writeable:
            frames = np.frombuffer(frames.tobytes(), dtype=np.uint8).reshape(frames.shape)
        prepared = {"frame_count": len(intervals), "frame_intervals_seconds": intervals.tolist(),
                    "prepare_ms": (time.perf_counter() - started) * 1000,
                    "upload_roundtrip_ms": 0.0, "mapping_revision": 0,
                    "streaming": frames is None, "queue_capacity": 2 if frames is None else 0}
        self._sequence = {"codes": frames, "intervals": intervals, "command_revision": self._command_revision,
                          "prepared": prepared, "queue": Queue(maxsize=2) if frames is None else None,
                          "submitted": 0, "playing": False, "last_submitted": None}
        self._sequence_cancel.clear()
        return dict(prepared)

    def submit_phase_frame(self, index: int, codes: object) -> None:
        sequence = self._sequence
        if sequence is None or sequence["queue"] is None:
            raise RuntimeError("Streaming SLM phase sequence has not been prepared")
        if type(index) is not int or index != sequence["submitted"] or index >= len(sequence["intervals"]):
            raise ValueError("SLM streaming frame index is not the next declared frame")
        previous = sequence["last_submitted"]
        if codes is None:
            if previous is None:
                raise ValueError("SLM repeated frame has no preceding admitted frame")
            frame = previous
        else:
            frame = np.asarray(codes)
            if frame.dtype != np.uint8 or frame.shape != self.shape_yx:
                raise ValueError("SLM streaming frame must be uint8 matching the full device shape")
            if frame.flags.writeable:
                frame = np.frombuffer(frame.tobytes(), np.uint8).reshape(self.shape_yx)
            if previous is not None and _same_phase_codes(frame, previous):
                frame = previous
        while not self._sequence_cancel.is_set() and self._sequence is sequence:
            try:
                sequence["queue"].put(frame, timeout=0.01)
                sequence["last_submitted"] = frame
                sequence["submitted"] += 1
                return
            except Full:
                pass
        raise RuntimeError("SLM phase sequence cancelled before accepting its frame")

    def cancel_phase_sequence(self) -> None:
        self._sequence_cancel.set()

    def release_phase_sequence(self) -> None:
        self._sequence = None

    def play_phase_sequence(self, stop_requested: Callable[[], bool] | None = None) -> dict[str, object]:
        sequence = self._sequence
        if sequence is None:
            raise RuntimeError("SLM phase sequence has not been prepared")
        if sequence["playing"]:
            raise RuntimeError("SLM phase sequence is already playing")
        sequence["playing"] = True
        codes, intervals, prepared = sequence["codes"], sequence["intervals"], sequence["prepared"]
        if sequence["command_revision"] != self._command_revision:
            self.release_phase_sequence()
            raise RuntimeError("stale prepared SLM sequence")
        started = time.perf_counter()
        dispatch, acknowledgments, queue_wait = [], [], []
        step_started, confirmations, frame_actions = [], [], []
        result = {**prepared, "played_frames": 0, "cancelled": False,
                  "acknowledgment": "simulation-state", "physical_vblank_observed": False,
                  "authored_timing_completed": False}
        self._command_revision += 1
        failed = False
        previous_frame, canonical = None, None
        confirmed_phase = None
        repeated_frames = 0
        try:
            for index, interval in enumerate(intervals):
                if self._sequence_cancel.is_set() or (stop_requested is not None and stop_requested()):
                    self._sequence_cancel.set()
                    break
                if sequence["queue"] is not None:
                    queued_started = time.perf_counter()
                    while not self._sequence_cancel.is_set():
                        if stop_requested is not None and stop_requested():
                            self._sequence_cancel.set()
                            break
                        try:
                            frame = sequence["queue"].get(timeout=0.01)
                            break
                        except Empty:
                            pass
                    queue_wait.append((time.perf_counter() - queued_started) * 1000)
                    if self._sequence_cancel.is_set():
                        break
                else:
                    frame = codes[index]
                frame_started = time.perf_counter()
                step_started.append((frame_started - started) * 1000)
                repeated = previous_frame is not None and _same_phase_codes(frame, previous_frame)
                if repeated:
                    repeated_frames += 1
                else:
                    canonical = phase_from_codes(frame, self.shape_yx)
                held = (repeated and self._outcome == "known-new"
                        and self._world.commanded_phase is confirmed_phase)
                if not held:
                    dispatch.append((frame_started - started) * 1000)
                    confirmed_phase = self._world.apply_slm_phase(canonical)
                    acknowledgments.append((time.perf_counter() - started) * 1000)
                previous_frame = frame
                confirmations.append((time.perf_counter() - started) * 1000)
                frame_actions.append("held" if held else "presented")
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
            result.update(queue_wait_ms=queue_wait, first_dispatch_ms=dispatch[0] if dispatch else None,
                          first_acknowledged_ms=acknowledgments[0] if acknowledgments else None,
                          repeated_frames=repeated_frames, mapped_frame_count=result["played_frames"] - repeated_frames,
                          step_started_ms=step_started, confirmed_ms=confirmations, frame_actions=frame_actions,
                          actual_step_intervals_ms=np.diff(step_started).tolist(),
                          held_frames=frame_actions.count("held"), newly_presented_frames=len(acknowledgments))
            result["authored_timing_completed"] = (not failed and result["played_frames"] == len(intervals)
                                                    and not result["cancelled"])
            self._stage = "sequence-failed" if failed else "sequence-cancelled" if result["cancelled"] else "sequence-complete"
            self._sequence_receipt = result
            self.release_phase_sequence()
        return {**result, "receipt": self.last_command_receipt}

    def close(self) -> None:
        # Closing an editor or session is not an optical blank command.  The
        # last explicit phase remains the simulated hardware's last command.
        self.cancel_phase_sequence()
        self.release_phase_sequence()


__all__ = ["VirtualSLM"]
