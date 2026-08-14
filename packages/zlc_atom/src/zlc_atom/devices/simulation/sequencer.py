"""Virtual sequencer backed by the same zlc_pulse device implementation."""

from __future__ import annotations

import threading
import time

from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_pulse import load_streamer_config, pulse_target_from_xdc
from zlc_pulse.device import AppliedState, DoneReport, PulseStreamer, SafeReadback
from zlc_pulse.schedule import run_duration_seconds
from zlc_pulse.transport import MemoryRegisterTransport


#: The board line wired to the camera trigger, named as ``board.xdc`` names it.
CAMERA_TRIGGER_CHANNEL = "emCCD"


class VirtualPulseStreamer(PulseStreamer):
    """The real pulse device on a memory transport, plus one world callback.

    Open/load/scan/fire/wait/safe/applied/snapshot are inherited unchanged from
    :class:`zlc_pulse.device.PulseStreamer`.  The virtual apparatus adds only
    the physical consequence that a fired camera-trigger window advances its
    simulation world.
    """

    def __init__(
        self,
        *,
        world: object | None = None,
        camera_trigger_channel: str = CAMERA_TRIGGER_CHANNEL,
    ) -> None:
        config = load_streamer_config()
        if config["source"] is None:
            raise RuntimeError(
                "no streamer config was found, so the virtual board geometry is unknown"
            )
        geometry = config["params"]
        clock_hz = float(config["clock_hz"])
        channel = str(camera_trigger_channel).strip()
        if not channel:
            raise ValueError("camera_trigger_channel must be non-empty")
        self.world = world
        self.camera_trigger_channel = channel
        self._world_thread: threading.Thread | None = None
        self._logical_deadline: float | None = None
        super().__init__(
            MemoryRegisterTransport(
                geom=geometry,
                auto_done=True,
                record_history=False,
            ),
            geometry,
            float(clock_hz),
            target=pulse_target_from_xdc(config_path=config["source"]),
        )

    def fire(self, *, forever: bool = False) -> None:
        super().fire(forever=forever)
        applied = self.applied()
        if applied is None:  # PulseStreamer.fire() requires a loaded program.
            raise RuntimeError("virtual sequencer fired without an applied program")
        points = self._world_points(applied)
        duration = sum(
            run_duration_seconds(
                applied.program,
                None if point is None else point,
            )
            for point in points
        )
        with self._lock:
            self._logical_deadline = (
                None
                if forever
                else time.monotonic() + duration
            )
        self._fire_world(applied, points)
        if forever:
            self._world_thread = threading.Thread(
                target=self._repeat_world,
                args=(applied, points, duration),
                name="zlc-virtual-world",
                daemon=True,
            )
            self._world_thread.start()

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        with self._lock:
            deadline = self._logical_deadline
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
            if timeout is not None and remaining > max(0.0, float(timeout)):
                # A timeout is "answer within this long", not "answer now":
                # returning immediately turns every polling caller into a spin
                # at full speed for the length of the table.  The slice the
                # caller asked for is the slice it waits.
                time.sleep(max(0.0, float(timeout)))
                return None
            if remaining > 0.0:
                time.sleep(remaining)
            timeout = (
                None
                if timeout is None
                else max(0.0, float(timeout) - remaining)
            )
        report = super().wait_done(timeout)
        if report is not None:
            with self._lock:
                self._logical_deadline = None
        return report

    def safe(self) -> SafeReadback:
        try:
            return super().safe()
        finally:
            with self._lock:
                self._logical_deadline = None
            self._join_world()
            callback = getattr(self.world, "safe", None)
            if callable(callback):
                callback()

    def _repeat_world(
        self,
        applied: AppliedState,
        points: tuple[tuple[int, ...] | None, ...],
        duration: float,
    ) -> None:
        cadence = max(0.001, duration)
        while not self._stop.wait(cadence):
            state = self.snapshot()
            if not state["firing"] or not state["forever"]:
                return
            self._fire_world(applied, points)

    def _join_world(self) -> None:
        worker, self._world_thread = self._world_thread, None
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)

    def _world_points(
        self,
        applied: AppliedState,
    ) -> tuple[tuple[int, ...] | None, ...]:
        if applied.slot_values:
            rows = (applied.slot_values,)
        elif applied.scan_rows:
            rows = applied.scan_rows
        else:
            return (None,)
        with self._lock:
            sweeps = self._scan_sweeps
        return rows * sweeps

    def _fire_world(
        self,
        applied: AppliedState,
        points: tuple[tuple[int, ...] | None, ...],
    ) -> None:
        callback = getattr(self.world, "fire", None)
        if not callable(callback):
            return
        for point in points:
            callback(
                applied.program,
                table=point,
                camera_channel=self.camera_trigger_channel,
            )


class VirtualSequencer(SequencerDevice):
    def __init__(
        self,
        *,
        world: object,
        camera_trigger_channel: str = CAMERA_TRIGGER_CHANNEL,
    ) -> None:
        super().__init__(
            VirtualPulseStreamer(
                world=world,
                camera_trigger_channel=camera_trigger_channel,
            )
        )
        self.world = world

    @property
    def camera_trigger_channel(self) -> str:
        return self.streamer.camera_trigger_channel

    @camera_trigger_channel.setter
    def camera_trigger_channel(self, value: str) -> None:
        channel = str(value).strip()
        if not channel:
            raise ValueError("camera_trigger_channel must be non-empty")
        self.streamer.camera_trigger_channel = channel


__all__ = [
    "CAMERA_TRIGGER_CHANNEL",
    "VirtualPulseStreamer",
    "VirtualSequencer",
]
