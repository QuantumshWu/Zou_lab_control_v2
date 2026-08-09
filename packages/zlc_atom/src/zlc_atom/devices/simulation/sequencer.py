"""Virtual sequencer backed by the same zlc_pulse device implementation."""

from __future__ import annotations

import math
import threading
import time

from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_pulse import load_streamer_config, pulse_target_from_xdc
from zlc_pulse.compile import CompiledProgram
from zlc_pulse.device import DoneReport, PulseStreamer, SafeReadback
from zlc_pulse.transport import MemoryRegisterTransport


#: The board line wired to the camera trigger, named as ``board.xdc`` names it.
CAMERA_TRIGGER_CHANNEL = "emCCD"
TRAP_CHANNEL = "trap"


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
            MemoryRegisterTransport(geom=geometry, auto_done=True),
            geometry,
            float(clock_hz),
            target=pulse_target_from_xdc(config_path=config["source"]),
        )

    def fire(self, *, forever: bool = False) -> None:
        super().fire(forever=forever)
        applied = self.applied()
        if applied is None:  # PulseStreamer.fire() requires a loaded program.
            raise RuntimeError("virtual sequencer fired without an applied program")
        with self._lock:
            self._logical_deadline = (
                None
                if forever
                else time.monotonic() + float(applied.program.duration_seconds)
            )
        self._fire_world(applied.program)
        if forever:
            self._world_thread = threading.Thread(
                target=self._repeat_world,
                args=(applied.program,),
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

    def _repeat_world(self, program: CompiledProgram) -> None:
        cadence = max(0.001, float(program.duration_seconds))
        while not self._stop.wait(cadence):
            state = self.snapshot()
            if not state["firing"] or not state["forever"]:
                return
            self._fire_world(program)

    def _join_world(self) -> None:
        worker, self._world_thread = self._world_thread, None
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)

    def _fire_world(self, program: CompiledProgram) -> None:
        callback = getattr(self.world, "fire", None)
        if not callable(callback):
            return
        windows = int(program.camera_window_count(self.camera_trigger_channel))
        if windows <= 0:
            raise ValueError("loaded program camera window count must be positive")
        try:
            exposures = tuple(
                float(value)
                for value in program.camera_window_exposures(self.camera_trigger_channel)
            )
        except ValueError:
            exposures = ()
        if exposures:
            if any(not math.isfinite(value) or value <= 0 for value in exposures):
                raise ValueError("loaded program camera window exposures must be positive")
            callback(
                windows,
                frame_exposures=exposures,
                trap_off_seconds=_trap_off_before_camera(
                    program,
                    camera_channel=self.camera_trigger_channel,
                ),
            )
        else:
            callback(
                windows,
                trap_off_seconds=_trap_off_before_camera(
                    program,
                    camera_channel=self.camera_trigger_channel,
                ),
            )


def _trap_off_before_camera(
    program: CompiledProgram,
    *,
    camera_channel: str,
) -> float:
    """Return the trap-low time before the first camera window."""

    from zlc_pulse.schedule import trigger_windows

    camera_windows = trigger_windows(program, camera_channel)
    if not camera_windows:
        return 0.0
    first_camera_tick = int(camera_windows[0][0])
    if first_camera_tick <= 0:
        return 0.0
    try:
        trap_windows = trigger_windows(program, TRAP_CHANNEL)
    except ValueError:
        return 0.0
    trap_on_ticks = sum(
        max(0, min(end, first_camera_tick) - max(start, 0))
        for start, end in trap_windows
    )
    return max(0.0, (first_camera_tick - trap_on_ticks) / float(program.clock_hz))


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
    "TRAP_CHANNEL",
    "VirtualPulseStreamer",
    "VirtualSequencer",
]
