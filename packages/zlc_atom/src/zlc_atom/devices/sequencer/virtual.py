"""Sequencer device adapters over the one zlc_pulse device implementation."""

from __future__ import annotations

from collections.abc import Sequence
import math
import threading
from typing import TypeAlias

from zlc_pulse import load_streamer_config, pulse_target_from_xdc
from zlc_pulse.compile import CompiledProgram
from zlc_pulse.device import (
    AppliedState,
    BoardDescription,
    DoneReport,
    PulseStreamer,
    SafeReadback,
)
from zlc_pulse.model import PulseSequence
from zlc_pulse.remote import RemotePulseStreamer
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
        self._fire_world(applied.program)
        if forever:
            self._world_thread = threading.Thread(
                target=self._repeat_world,
                args=(applied.program,),
                name="zlc-virtual-world",
                daemon=True,
            )
            self._world_thread.start()

    def safe(self) -> SafeReadback:
        try:
            return super().safe()
        finally:
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
            callback(windows, frame_exposures=exposures)
        else:
            callback(windows)


Streamer: TypeAlias = PulseStreamer | RemotePulseStreamer


class SequencerDevice:
    """Installed sequencer capability forwarding the true device surface."""

    def __init__(self, streamer: Streamer) -> None:
        if not isinstance(streamer, (PulseStreamer, RemotePulseStreamer)):
            raise TypeError("streamer must be a zlc_pulse device")
        self.streamer = streamer

    def open(self) -> None:
        self.streamer.open()

    def close(self) -> None:
        self.streamer.close()

    def describe(self) -> BoardDescription:
        return self.streamer.describe()

    def load(
        self,
        prog: CompiledProgram,
        *,
        source: PulseSequence | None = None,
    ) -> None:
        self.streamer.load(prog, source=source)

    def write_slots(self, values: Sequence[int]) -> None:
        self.streamer.write_slots(values)

    def write_scan_table(
        self,
        rows: Sequence[Sequence[int]],
        *,
        sweeps: int = 1,
    ) -> None:
        self.streamer.write_scan_table(rows, sweeps=sweeps)

    def fire(self, *, forever: bool = False) -> None:
        self.streamer.fire(forever=forever)

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        return self.streamer.wait_done(timeout)

    def cursor(self) -> int | None:
        return self.streamer.cursor()

    def safe(self) -> SafeReadback:
        return self.streamer.safe()

    def snapshot(self) -> dict[str, object]:
        return self.streamer.snapshot()

    def applied(self) -> AppliedState | None:
        return self.streamer.applied()


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


__all__ = ["SequencerDevice", "VirtualPulseStreamer", "VirtualSequencer"]
