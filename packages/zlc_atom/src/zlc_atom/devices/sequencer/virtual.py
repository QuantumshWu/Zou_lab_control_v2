"""Real and virtual sequencer implementations sharing one streamer surface."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Sequence

import numpy as np

from .protocol import DoneReport, PulseStreamer, RegisterTransport, SafeReadback


class DictRegisterTransport:
    def __init__(self) -> None:
        self.registers: dict[int, int] = {}

    def read(self, address: int) -> int:
        return int(self.registers.get(int(address), 0))

    def write(self, address: int, value: int) -> None:
        self.registers[int(address)] = int(value)


#: The board line wired to the camera trigger, named as ``board.xdc`` names it.
#: A pulse that triggers a different line declares so in its metadata and the
#: orchestrating task passes that declaration down; this is only the default for
#: a device built without one.  It is NOT free to be an invented name: the
#: virtual streamer counts camera windows by finding this channel in the loaded
#: program, so a name the board does not have fails at load instead of silently
#: producing the wrong frame count.
CAMERA_TRIGGER_CHANNEL = "emCCD"


@dataclass(frozen=True)
class _VirtualAppliedState:
    """The editor-facing echo of what this virtual streamer currently holds."""

    program: object
    source: object | None
    slot_values: tuple[int, ...] = ()
    scan_rows: tuple[tuple[int, ...], ...] = ()
    forever: bool = False
    loaded_at: float = 0.0


class VirtualPulseStreamer:
    """Register-backed PulseStreamer fake with programmable terminal report."""

    def __init__(
        self,
        *,
        transport: RegisterTransport | None = None,
        world: object | None = None,
        done_report: DoneReport | None = None,
        camera_trigger_channel: str = CAMERA_TRIGGER_CHANNEL,
    ) -> None:
        self.transport = DictRegisterTransport() if transport is None else transport
        self.world = world
        self.camera_trigger_channel = str(camera_trigger_channel).strip()
        if not self.camera_trigger_channel:
            raise ValueError("camera_trigger_channel must be non-empty")
        self._done_report = done_report
        self._opened = False
        self._loaded: object | None = None
        self._applied: _VirtualAppliedState | None = None
        self._slots: tuple[int, ...] = ()
        self._scan_table: np.ndarray | None = None
        self._cursor: int | None = None
        self._clk_enable = False
        self._status = 0
        self._lock = threading.RLock()
        self._firing = False

    def open(self) -> None:
        with self._lock:
            self._opened = True
            self._status = 0

    def close(self) -> None:
        with self._lock:
            self._clk_enable = False
            self._firing = False
            self._opened = False

    def _require_idle(self) -> None:
        """Refuse anything that would change a shot already in progress.

        The rules here come from the hardware, not from convenience.  A board
        detects its commands on a rising edge and never clears the command
        register itself, so a second command issued during a shot is DROPPED --
        silently, with a normal-looking report.  A twin that accepts it lets a
        scan loop look correct in simulation and take exactly one shot at the
        bench, which is what happened.
        """

        if self._firing:
            raise RuntimeError("the streamer is already firing")

    def load(self, prog: object, *, source: object | None = None) -> None:
        with self._lock:
            if not self._opened:
                raise RuntimeError("PulseStreamer is closed")
            self._require_idle()
            self._loaded = prog
            self._applied = _VirtualAppliedState(
                prog,
                source,
                loaded_at=time.monotonic(),
            )
            self._slots = ()
            self._scan_table = None
            self._cursor = 0

    def write_slots(self, values: Sequence[int]) -> None:
        with self._lock:
            if not self._opened:
                raise RuntimeError("PulseStreamer is closed")
            self._require_idle()
            self._slots = tuple(int(value) for value in values)
            if self._applied is not None:
                self._applied = replace(
                    self._applied,
                    slot_values=self._slots,
                    scan_rows=(),
                )

    def write_scan_table(self, rows: np.ndarray, *, sweeps: int = 1) -> None:
        with self._lock:
            if not self._opened:
                raise RuntimeError("PulseStreamer is closed")
            self._require_idle()
            count = int(sweeps)
            if count <= 0:
                raise ValueError("scan sweeps must be positive")
            table = np.asarray(rows)
            if table.ndim != 2 or table.shape[0] <= 0:
                raise ValueError("scan rows must be a non-empty two-dimensional table")
            self._scan_table = np.tile(table, (count, 1))
            if self._applied is not None:
                self._applied = replace(
                    self._applied,
                    slot_values=(),
                    scan_rows=tuple(
                        tuple(int(value) for value in row)
                        for row in self._scan_table
                    ),
                )

    def fire(self, *, forever: bool = False) -> None:
        with self._lock:
            if not self._opened:
                raise RuntimeError("PulseStreamer is closed")
            if self._loaded is None:
                raise RuntimeError("no compiled program is loaded")
            self._require_idle()
            self._firing = True
            self._clk_enable = True
            self._status = 1 if forever else 0
            if self._applied is not None:
                self._applied = replace(self._applied, forever=bool(forever))
            self._cursor = 0 if self._cursor is None else self._cursor + 1
        callback = getattr(self.world, "fire", None)
        if callback is not None:
            windows = self._camera_window_count()
            exposures = self._camera_window_exposures()
            if exposures is None:
                callback(windows)
            else:
                callback(windows, frame_exposures=exposures)
        if not forever:
            with self._lock:
                self._clk_enable = False
                self._status = 0

    def _camera_window_count(self) -> int:
        """How many camera windows the loaded program opens.

        Asked of the program.  This used to re-derive it -- prefix, body, tail,
        repeat count, rising edges, fifty lines of it -- which is a second copy
        of the rule the board plays in hardware, in the one place whose whole
        job is to behave exactly like that board.
        """

        program = self._loaded
        if program is None:
            return 1
        ask = getattr(program, "camera_window_count", None)
        if not callable(ask):
            raise TypeError(
                "a loaded program must be able to say how many camera windows "
                "it opens; this one cannot"
            )
        count = int(ask(self.camera_trigger_channel))
        if count <= 0:
            raise ValueError("loaded program camera window count must be positive")
        return count

    def _camera_window_exposures(self) -> tuple[float, ...] | None:
        """How long each camera window is open, from the program itself."""

        program = self._loaded
        if program is None:
            return None
        ask = getattr(program, "camera_window_exposures", None)
        if not callable(ask):
            return None
        try:
            values = tuple(float(value) for value in ask(self.camera_trigger_channel))
        except ValueError:
            # A forever program has no finite window list, which is not a fault.
            return None
        if not values or any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("loaded program camera window exposures must be positive")
        return values

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        """Report the shot that just ran, once.

        None means no shot is running -- it must not be confused with a report,
        or a caller believes a shot completed when none was started.  The report
        is consumed, so the same shot is never reported twice.
        """

        del timeout
        with self._lock:
            if not self._firing:
                return None
            self._firing = False
            if self._done_report is not None:
                return self._done_report
            return DoneReport(self._status, self._cursor, False, 0.0)

    def cursor(self) -> int | None:
        with self._lock:
            return self._cursor

    def safe(self) -> SafeReadback:
        with self._lock:
            self._clk_enable = False
            self._firing = False
            self._status = 0
            if self._applied is not None:
                self._applied = replace(self._applied, forever=False)
            return SafeReadback((0, 0), (0,), True)

    def applied(self) -> _VirtualAppliedState | None:
        with self._lock:
            return self._applied

    def snapshot(self) -> dict[str, object]:
        """The same questions the real board answers, in the same words.

        A twin that reports its state under different keys is a twin nothing
        written for the board can read: the caller asks whether it is firing,
        gets nothing back, and concludes an idle board -- which is the one
        answer that was never true.
        """

        with self._lock:
            return {
                "opened": self._opened,
                "loaded": self._loaded is not None,
                "firing": self._firing,
                "forever": bool(self._applied is not None and self._applied.forever),
                "reloaded_before_fire": False,
                "cursor": self._cursor,
                "scan_count": 0 if self._scan_table is None else int(self._scan_table.shape[0]),
                "scan_next_chunk": 0,
                "underflow": False,
                "status": self._status,
                "applied_digest": str(getattr(self._loaded, "digest", "")),
                # Its own, beyond the board's: how many slot values are held.
                "slot_count": len(self._slots),
            }


class SequencerDevice:
    """Thin device wrapper: bind/require capability then forward streamer calls."""

    def __init__(self, streamer: PulseStreamer) -> None:
        if not isinstance(streamer, PulseStreamer):
            raise TypeError("streamer must implement the PulseStreamer contract")
        self.streamer = streamer

    def open(self) -> None:
        self.streamer.open()

    def close(self) -> None:
        self.streamer.close()

    def load(self, prog: object, *, source: object | None = None) -> None:
        self.streamer.load(prog, source=source)

    def write_slots(self, values: Sequence[int]) -> None:
        self.streamer.write_slots(values)

    def write_scan_table(self, rows: np.ndarray, *, sweeps: int = 1) -> None:
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
        return dict(self.streamer.snapshot())

    def applied(self) -> object | None:
        applied = getattr(self.streamer, "applied", None)
        return applied() if callable(applied) else applied


class VirtualSequencer(SequencerDevice):
    def __init__(self, *, world: object, camera_trigger_channel: str = CAMERA_TRIGGER_CHANNEL) -> None:
        super().__init__(VirtualPulseStreamer(world=world, camera_trigger_channel=camera_trigger_channel))
        self.world = world


__all__ = ["DictRegisterTransport", "SequencerDevice", "VirtualPulseStreamer", "VirtualSequencer"]
