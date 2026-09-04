"""Virtual sequencer backed by the same zlc_pulse device implementation."""

from __future__ import annotations

import threading
import time

from zlc_atom.devices.sequencer.device import SequencerDevice
from zlc_pulse import load_streamer_config, pulse_target_from_xdc
from zlc_pulse.device import AppliedState, DoneReport, PulseStreamer, SafeReadback
from zlc_pulse.schedule import run_duration_seconds
from zlc_pulse.transport import MemoryRegisterTransport
from zlc_pulse.wire import STATUS_DONE, STATUS_ERROR, STATUS_RUNNING


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
        super().__init__(
            MemoryRegisterTransport(
                geom=geometry,
                # The physical-world worker below advances the same STATUS and
                # CURSOR registers as the RTL.  Instant completion would make
                # the device snapshot claim DONE while cameras are still being
                # triggered.
                auto_done=False,
                record_history=False,
            ),
            geometry,
            float(clock_hz),
            target=pulse_target_from_xdc(config_path=config["source"]),
        )

    def fire(self, *, run_repeats: int, scan_repeats: int = 1) -> None:
        worker = self._world_thread
        if worker is not None and worker.is_alive():
            raise RuntimeError("the previous virtual pulse is still playing")
        super().fire(
            run_repeats=run_repeats,
            scan_repeats=scan_repeats,
        )
        applied = self.applied()
        if applied is None:  # PulseStreamer.fire() requires a loaded program.
            raise RuntimeError("virtual sequencer fired without an applied program")
        with self._lock:
            self._world_error = None
        worker = threading.Thread(
            target=self._play_world,
            args=(applied,),
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
        if timeout is not None:
            timeout = max(0.0, float(timeout) - (time.monotonic() - started))
        report = super().wait_done(timeout)
        if report is None:
            return None
        with self._lock:
            error, self._world_error = self._world_error, None
        if error is not None:
            raise RuntimeError("virtual world playback failed") from error
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
    ) -> None:
        callback = getattr(self.world, "fire", None)
        rows: tuple[tuple[int, ...] | None, ...] = applied.rows or (None,)
        run_repeats = applied.run_repeats
        scan_repeats = applied.scan_repeats
        cursor = 0
        try:
            if run_repeats == 0:
                # An infinite Run repeat never completes its first row visit,
                # so CURSOR truthfully remains zero and no later row is seen.
                while self._play_world_run(applied, rows[0], callback):
                    pass
                return

            sweep = 0
            while scan_repeats == 0 or sweep < scan_repeats:
                for row_index, point in enumerate(rows):
                    for _run in range(run_repeats):
                        if not self._play_world_run(applied, point, callback):
                            return
                    if self._stop.is_set():
                        return
                    final = (
                        scan_repeats != 0
                        and sweep + 1 == scan_repeats
                        and row_index + 1 == len(rows)
                    )
                    if final:
                        self._memory_transport().publish_execution_readback(
                            status=STATUS_DONE,
                            cursor=cursor,
                        )
                        return
                    cursor = (cursor + 1) & 0xFFFFFFFF
                    if not self._memory_transport().publish_execution_readback(
                        status=STATUS_RUNNING,
                        cursor=cursor,
                    ):
                        return
                sweep += 1
        except BaseException as error:
            with self._lock:
                self._world_error = error
            self._memory_transport().publish_execution_readback(
                status=STATUS_ERROR,
                cursor=cursor,
            )

    def _play_world_run(
        self,
        applied: AppliedState,
        point: tuple[int, ...] | None,
        callback: object,
    ) -> bool:
        """Play one whole Pulse and wait through its physical wall duration."""

        if self._stop.is_set():
            return False
        cycle_start = time.monotonic()
        if callable(callback):
            callback(
                applied.program,
                table=point,
                camera_channel=self.camera_trigger_channel,
            )
        duration = run_duration_seconds(
            applied.program,
            None if point is None else point,
        )
        cycle_end = cycle_start + duration
        return not self._stop.wait(max(0.0, cycle_end - time.monotonic()))

    def _memory_transport(self) -> MemoryRegisterTransport:
        transport = self.transport
        if not isinstance(transport, MemoryRegisterTransport):
            raise RuntimeError("virtual sequencer lost its memory transport")
        return transport

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
