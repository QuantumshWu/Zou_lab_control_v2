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
        self._world_error: BaseException | None = None
        self._logical_time_seconds = 0.0
        self._logical_wall_at = time.monotonic()
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

    def fire(self, *, cycles: int | None = 1) -> None:
        worker = self._world_thread
        if worker is not None and worker.is_alive():
            raise RuntimeError("the previous virtual pulse is still playing")
        super().fire(cycles=cycles)
        applied = self.applied()
        if applied is None:  # PulseStreamer.fire() requires a loaded program.
            raise RuntimeError("virtual sequencer fired without an applied program")
        points = self._world_points(applied)
        with self._lock:
            self._world_error = None
        worker = threading.Thread(
            target=self._play_world,
            args=(applied, points, cycles is None),
            name="zlc-virtual-world",
            daemon=True,
        )
        self._world_thread = worker
        try:
            worker.start()
        except BaseException:
            self._world_thread = None
            super().safe()
            raise

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        started = time.monotonic()
        worker = self._world_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(None if timeout is None else max(0.0, float(timeout)))
            if worker.is_alive():
                return None
            self._world_thread = None
        with self._lock:
            error, self._world_error = self._world_error, None
        if error is not None:
            raise RuntimeError("virtual world playback failed") from error
        if timeout is not None:
            timeout = max(0.0, float(timeout) - (time.monotonic() - started))
        report = super().wait_done(timeout)
        return report

    def safe(self) -> SafeReadback:
        try:
            return super().safe()
        finally:
            try:
                self._join_world()
            finally:
                callback = getattr(self.world, "safe", None)
                if callable(callback):
                    callback()

    def _play_world(
        self,
        applied: AppliedState,
        points: tuple[tuple[int, ...] | None, ...],
        forever: bool,
    ) -> None:
        callback = getattr(self.world, "fire", None)
        try:
            while True:
                for point in points:
                    if self._stop.is_set():
                        return
                    cycle_start = time.monotonic()
                    with self._lock:
                        logical_cycle_start = self._logical_time_seconds + max(
                            0.0, cycle_start - self._logical_wall_at
                        )
                    if callable(callback):
                        callback(
                            applied.program,
                            table=point,
                            camera_channel=self.camera_trigger_channel,
                            cycle_start_seconds=logical_cycle_start,
                        )
                    duration = run_duration_seconds(
                        applied.program,
                        None if point is None else point,
                    )
                    cycle_end = cycle_start + duration
                    with self._lock:
                        self._logical_time_seconds = logical_cycle_start + duration
                        self._logical_wall_at = cycle_end
                    if self._stop.wait(max(0.0, cycle_end - time.monotonic())):
                        return
                if not forever:
                    return
        except BaseException as error:
            with self._lock:
                self._world_error = error

    def _join_world(self) -> None:
        worker = self._world_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
            if worker.is_alive():
                raise RuntimeError("virtual world playback did not stop")
        self._world_thread = None
        with self._lock:
            error, self._world_error = self._world_error, None
        if error is not None:
            raise RuntimeError("virtual world playback failed") from error

    def _world_points(
        self,
        applied: AppliedState,
    ) -> tuple[tuple[int, ...] | None, ...]:
        rows: tuple[tuple[int, ...] | None, ...] = applied.rows or (None,)
        if applied.cycles is None:
            return rows
        return tuple(rows[index % len(rows)] for index in range(applied.cycles))

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
