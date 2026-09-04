"""Physical pulse-streamer session with a deliberately small state surface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence
import math
from numbers import Integral
import threading
import time

from collections.abc import Mapping

from .compile import CompiledProgram, evaluate_affine_tick, slot_operand_width
from .model import MAXIMUM_REPEAT_COUNT, PORT_DAC, PulseSequence, PulseTarget
from .schedule import trigger_edge_ticks
from .transport.base import DEFAULT_OBSERVER_INTERVAL, RegisterTransport
from .wire import (
    CMD_FIRE,
    CMD_LOAD,
    CMD_SAFE,
    CTRL_WORDS,
    CtrlWords,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_LINK_ERROR,
    STATUS_LOADED,
    STATUS_RUNNING,
    STATUS_UNDERFLOW,
    StreamerParams,
    build_fingerprint,
    pack_program,
    pack_scan_rows,
    region_bases,
)


# Loader and SAFE handshakes share the same five-second action budget.
LOAD_TIMEOUT = 5.0
SAFE_TIMEOUT = 5.0
SAFE_RETRY_AFTER = 0.05
SAFE_POLL_INTERVAL = 0.001
_MIN_SEAM_SPAN_TICKS = 3


def _repeat_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= MAXIMUM_REPEAT_COUNT:
        raise ValueError(f"{name} must be in the hardware range [0, 2^32-1]")
    return result


@dataclass(frozen=True)
class DoneReport:
    status: int
    cursor: int | None  # cumulative row-visit ordinal; table row = cursor % len(rows)
    underflow: bool
    elapsed_seconds: float
    status_reads: tuple[int, int] = ()
    cursor_reads: tuple[int, int] = ()
    observer_error: str = ""
    #: How many status/cursor polls failed during this shot, and how many
    #: frames the line had to send again for it.  A shot that finished clean
    #: over a line that was quietly dropping bytes must say so -- a run that
    #: dies four hours in is diagnosed from the shots before it, not from the
    #: one that died.
    poll_failures: int = 0
    resent_frames: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", int(self.status))
        object.__setattr__(self, "cursor", None if self.cursor is None else int(self.cursor))
        object.__setattr__(self, "underflow", bool(self.underflow))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "status_reads", tuple(int(value) for value in self.status_reads))
        object.__setattr__(self, "cursor_reads", tuple(int(value) for value in self.cursor_reads))
        object.__setattr__(self, "observer_error", str(self.observer_error))
        object.__setattr__(self, "poll_failures", int(self.poll_failures))
        object.__setattr__(self, "resent_frames", int(self.resent_frames))

    @property
    def fault(self) -> str:
        """Why this shot was not a good shot, or "" if it was.

        On the report, so that what counts as a bad shot is decided where the
        report is defined.  Every caller used to throw the whole report away --
        a shot that reported ERROR, that underran the scan bank, or that never
        finished at all was indistinguishable from a clean one, and the data it
        did not take was published as though it had.
        """

        from .wire import STATUS_ERROR, STATUS_UNDERFLOW

        reasons = []
        if self.observer_error:
            reasons.append(f"pulse observer failed: {self.observer_error}")
            if self.poll_failures or self.resent_frames:
                reasons.append(
                    f"the line failed {self.poll_failures} poll(s) and resent "
                    f"{self.resent_frames} frame(s) during this shot"
                )
        elif self.status & STATUS_ERROR:
            reasons.append("the board reported an error")
        if self.status & STATUS_UNDERFLOW or self.underflow:
            reasons.append("the scan bank underran")
        return "; ".join(reasons)

    @property
    def link_error(self) -> bool:
        return bool(self.status & STATUS_LINK_ERROR)

    @property
    def status_first(self) -> int | None:
        return self.status_reads[0] if self.status_reads else None

    @property
    def status_second(self) -> int | None:
        return self.status_reads[1] if len(self.status_reads) > 1 else None

    @property
    def cursor_first(self) -> int | None:
        return self.cursor_reads[0] if self.cursor_reads else None

    @property
    def cursor_second(self) -> int | None:
        return self.cursor_reads[1] if len(self.cursor_reads) > 1 else None


@dataclass(frozen=True)
class SafeReadback:
    status_reads: tuple[int, int]
    clock_enable_words: tuple[int, ...]
    stable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "status_reads", tuple(int(value) for value in self.status_reads))
        object.__setattr__(self, "clock_enable_words", tuple(int(value) for value in self.clock_enable_words))
        object.__setattr__(self, "stable", bool(self.stable))

    @property
    def status(self) -> int:
        return self.status_reads[-1]


@dataclass(frozen=True)
class AppliedState:
    """Immutable echo of the executable application owned by the device."""

    program: CompiledProgram
    source: PulseSequence | None
    rows: tuple[tuple[int, ...], ...]
    run_repeats: int
    scan_repeats: int
    loaded_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.program, CompiledProgram):
            raise TypeError("applied program must be CompiledProgram")
        if self.source is not None and not isinstance(self.source, PulseSequence):
            raise TypeError("applied source must be PulseSequence or None")
        rows = tuple(tuple(row) for row in self.rows)
        if any(len(row) != self.program.slot_count for row in rows):
            raise ValueError("applied row width differs from the compiled program")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for row in rows
            for value in row
        ):
            raise TypeError("applied row values must be integers")
        if self.program.slot_count and not rows:
            raise ValueError("a slotted program requires applied rows")
        if not self.program.slot_count and rows:
            raise ValueError("an unslotted program has no value rows")
        run_repeats = _repeat_count(self.run_repeats, "run_repeats")
        scan_repeats = _repeat_count(self.scan_repeats, "scan_repeats")
        if not rows and scan_repeats != 1:
            raise ValueError("scan_repeats must be 1 when no scan table is loaded")
        loaded_at = float(self.loaded_at)
        if not math.isfinite(loaded_at) or loaded_at < 0:
            raise ValueError("applied loaded_at must be a finite non-negative timestamp")
        object.__setattr__(self, "rows", tuple(tuple(int(value) for value in row) for row in rows))
        object.__setattr__(self, "run_repeats", run_repeats)
        object.__setattr__(self, "scan_repeats", scan_repeats)
        object.__setattr__(self, "loaded_at", loaded_at)


@dataclass(frozen=True)
class BoardDescription:
    """What a board is, in its own words.

    A client must never supply a hardware fact.  It had no choice: the protocol
    could open, load and fire a board but not ask what the board WAS, so an
    editor wanting ports, pins or a clock had to read the local XDC and config
    and hope they were the ones the board was built from.  That is the exact
    mistake the layout handshake exists to catch, made one layer up.

    Every field here is already proven at the moment it can be asked for.
    ``pulse_target_from_xdc`` refuses an XDC whose lane, bus and width counts
    disagree with the streamer config, and ``open()`` refuses a board whose
    LAYOUT_ID word disagrees with that same config -- so a description handed
    out by an open streamer is the board's, not a guess about it.
    """

    target: PulseTarget
    geometry: StreamerParams
    clock_hz: float

    def __post_init__(self) -> None:
        if not isinstance(self.target, PulseTarget):
            raise TypeError("board target must be PulseTarget")
        if not isinstance(self.geometry, StreamerParams):
            raise TypeError("board geometry must be StreamerParams")
        clock_hz = float(self.clock_hz)
        if not math.isfinite(clock_hz) or clock_hz <= 0:
            raise ValueError("board clock_hz must be finite and positive")
        object.__setattr__(self, "clock_hz", clock_hz)

    @property
    def time_step_ns(self) -> float:
        return 1e9 / self.clock_hz

    @property
    def layout_fingerprint(self) -> int:
        return int(build_fingerprint(self.geometry))


class PulseStreamer:
    """Host control of the frozen streamer.

    After FIRE, the observer is the sole caller that reads status/cursor or
    refills scan banks.  Public cursor reads use its cached sample.
    """

    def __init__(
        self,
        transport: RegisterTransport,
        geom: StreamerParams,
        clock_hz: float,
        *,
        target: PulseTarget,
    ) -> None:
        if not isinstance(geom, StreamerParams):
            raise TypeError("geom must be StreamerParams")
        if not isinstance(target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        if isinstance(clock_hz, bool) or not isinstance(clock_hz, (int, float)):
            raise TypeError("clock_hz must be numeric")
        if not math.isfinite(float(clock_hz)) or clock_hz <= 0:
            raise ValueError("clock_hz must be positive and finite")
        buses = tuple(port for port in target.ports if port.kind == PORT_DAC)
        widths = {port.width for port in buses}
        mismatches = []
        if len(target.raw_lanes) != geom.channel_count:
            mismatches.append(
                f"target lanes={len(target.raw_lanes)} but geometry has {geom.channel_count}"
            )
        if len(buses) != geom.bus_count:
            mismatches.append(
                f"target DAC buses={len(buses)} but geometry has {geom.bus_count}"
            )
        if widths != {geom.bus_width}:
            mismatches.append(
                f"target DAC widths={sorted(widths)} but geometry has {geom.bus_width}"
            )
        if mismatches:
            raise ValueError("target/geometry mismatch: " + "; ".join(mismatches))
        self.transport = transport
        observer_interval = getattr(transport, "observer_interval", DEFAULT_OBSERVER_INTERVAL)
        if isinstance(observer_interval, bool) or not math.isfinite(float(observer_interval)) or observer_interval <= 0:
            raise ValueError("transport observer_interval must be finite and positive")
        self._observer_interval = float(observer_interval)
        self.geom = geom
        self.clock_hz = float(clock_hz)
        self._target = target
        self._lock = threading.RLock()
        self._opened = False
        self._program: CompiledProgram | None = None
        self._applied: AppliedState | None = None
        # Digested once, when the program is applied.  ``snapshot`` is polled
        # every few milliseconds while a shot runs, and hashing a whole program
        # at that rate would make asking what the board holds cost more than
        # running it.
        self._applied_digest = ""
        self._loaded = False
        self._hardware_loaded = self._last_fire_reloaded = False
        self._firing = False
        self._run_repeats = 1
        self._scan_repeats = 1
        self._scan_rows: tuple[tuple[int, ...], ...] = ()
        self._scan_next_chunk = 2
        self._scan_ready = 0
        self._scan_armed = False
        self._scan_count = 0
        self._scan_last_cursor = 0
        self._scan_cursor_total = 0
        self._cursor_value: int | None = None
        self._underflow = False
        self._terminal_status_reads: tuple[int, int] = ()
        self._terminal_cursor_reads: tuple[int, int] = ()
        self._observer_error = ""
        self._poll_failures = 0
        self._resends_at_fire = 0
        self._worker: threading.Thread | None = None
        self._fire_gate: threading.Event | None = None
        self._stop = threading.Event()
        self._done = threading.Event()
        self._terminal_status = 0
        self._fire_started = 0.0
        self._safe_status_word: int | None = None
        self._safe_clock_enable_words: tuple[int, ...] | None = None

    def open(self) -> None:
        with self._lock:
            if self._opened:
                return
            start = getattr(self.transport, "start", None)
            if callable(start):
                start()
            try:
                self._check_register_layout_locked()
            except BaseException:
                close = getattr(self.transport, "close", None)
                if callable(close): close()
                raise
            self._opened = True
    def check_register_layout(self) -> None:
        with self._lock:
            self._require_open()
            self._check_register_layout_locked()
    def transport_self_test(self, *, count: int = 16) -> None:
        with self._lock:
            self._require_open()
            self._check_register_layout_locked()
            length = max(2, min(int(count), CTRL_WORDS - self.geom.ctrl_scratch_base))
            base = self.geom.ctrl_scratch_base
            pattern = tuple((base + i, (0xC0DE0000 + i) & 0xFFFFFFFF) for i in range(length))
            try:
                self._write(pattern)
                actual = tuple(self._read(base + i) for i in range(length))
            finally:
                self._write(tuple((base + i, 0) for i in range(length)))
            if actual != tuple(value for _address, value in pattern):
                raise RuntimeError(f"{getattr(self.transport, 'transport_id', 'register')} register self-test readback mismatch")
    def close(self) -> None:
        with self._lock:
            if not self._opened:
                return
        self._stop_worker()
        with self._lock:
            self.safe()
            close = getattr(self.transport, "close", None)
            if callable(close):
                close()
            self._opened = False
            self._loaded = self._hardware_loaded = self._last_fire_reloaded = False
            self._program = None
            self._applied = None
            self._applied_digest = ""
            self._safe_status_word = None
            self._safe_clock_enable_words = None
    def load(
        self,
        prog: CompiledProgram,
        *,
        source: PulseSequence | None = None,
        rows: Sequence[Sequence[int]] = (),
    ) -> None:
        if not isinstance(prog, CompiledProgram):
            raise TypeError("prog must be CompiledProgram")
        if source is not None and not isinstance(source, PulseSequence):
            raise TypeError("source must be PulseSequence or None")
        if (
            source is not None
            and source.target.abi_fingerprint != prog.target_abi_fingerprint
        ):
            raise ValueError("source target ABI differs from the compiled program")
        normalized = tuple(tuple(row) for row in rows)
        if len(normalized) > MAXIMUM_REPEAT_COUNT:
            raise ValueError("scan table row count does not fit the 32-bit SCAN_COUNT")
        if any(len(row) != prog.slot_count for row in normalized):
            raise ValueError("application row width differs from the compiled program")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for row in normalized
            for value in row
        ):
            raise TypeError("application row values must be integers")
        normalized = tuple(tuple(int(value) for value in row) for row in normalized)
        if prog.slot_count and not normalized:
            raise ValueError("a slotted program requires a non-empty value table")
        if not prog.slot_count and normalized:
            raise ValueError("an unslotted program does not accept value rows")
        with self._lock:
            self._require_open()
            self._require_idle()
            self._stop.clear()
            self._validate_application(prog, normalized)
            words = pack_program(prog, self.geom)
            if not self._safe_readback_current_locked():
                self._drive_physical_safe(deadline=time.monotonic() + SAFE_TIMEOUT)
            self._clear_safe_readback_locked()
            self._loaded = self._hardware_loaded = False; self._program = None; self._applied = None
            self._applied_digest = ""
            self._scan_rows = normalized
            self._scan_count = len(normalized)
            self._run_repeats = 1
            self._scan_repeats = 1
            self._scan_armed = False
            self._write(
                tuple(sorted(words.items()))
                + ((CtrlWords.BANK_READY, 0b11),),
                stop=self._stop,
            )
            self._strobe(CMD_LOAD, repeatable=True, stop=self._stop)
            self._await_loaded(stop=self._stop)
            self._hardware_loaded = True
            self._program = prog
            self._loaded = True
            self._scan_next_chunk = 2
            self._scan_ready = self._initial_ready(self._scan_count)
            self._scan_armed = False
            self._scan_last_cursor = 0
            self._scan_cursor_total = 0
            self._cursor_value = 0
            self._underflow = False
            self._terminal_status_reads = ()
            self._terminal_cursor_reads = ()
            self._observer_error = ""
            self._applied = AppliedState(
                program=prog,
                source=source,
                rows=normalized,
                run_repeats=1,
                scan_repeats=1,
                loaded_at=time.time(),
            )
            self._applied_digest = prog.digest

    def fire(self, *, run_repeats: int, scan_repeats: int = 1) -> None:
        run_repeats = _repeat_count(run_repeats, "run_repeats")
        scan_repeats = _repeat_count(scan_repeats, "scan_repeats")
        with self._lock:
            self._require_open()
            self._require_loaded()
            self._require_idle()
            self._stop.clear()
            assert self._program is not None
            if not self._scan_rows and scan_repeats != 1:
                raise ValueError("scan_repeats must be 1 when no scan table is loaded")
            if (
                self._scan_rows
                and run_repeats != 0
                and scan_repeats != 0
                and len(self._scan_rows) * scan_repeats > (1 << 32)
            ):
                raise ValueError(
                    "finite scan row visits exceed the 32-bit CURSOR range"
                )
            self._validate_delay_capacity(
                self._program,
                self._scan_rows,
                run_repeats,
                scan_repeats,
            )
            # The single registered affine cache is prepared two clocks before
            # every frame seam.  A one-shot may be only one tick long, but every
            # point which is followed by another point must reach that schedule
            # tick after starting at tick 1.  Refuse an impossible seamless run
            # before touching the mailbox instead of letting RTL underflow or
            # consume the previous point's cache.
            table = self._scan_rows or ((),)
            if run_repeats == 0:
                seam_rows = table[:1]
            elif run_repeats > 1 or scan_repeats != 1:
                seam_rows = table
            else:
                seam_rows = table[:-1]
            for row in seam_rows:
                self._validate_slot_row(
                    self._program,
                    row,
                    require_outer_seam=True,
                )
            self._run_repeats = run_repeats
            self._scan_repeats = scan_repeats
            self._scan_armed = False
            assert self._applied is not None
            self._applied = replace(
                self._applied,
                run_repeats=run_repeats,
                scan_repeats=scan_repeats,
            )
            # DONE/SAFE clear the RTL's LOADED gate; replay only its resident mini-loader.
            self._last_fire_reloaded = not self._hardware_loaded
            if self._last_fire_reloaded:
                self._write(self._scan_bank_arming(), stop=self._stop)
                self._strobe(CMD_LOAD, repeatable=True, stop=self._stop)
                self._await_loaded(stop=self._stop)
                self._hardware_loaded = True
            self._firing = True; self._hardware_loaded = False
            self._done.clear()
            self._underflow = False
            self._cursor_value = 0
            self._scan_last_cursor = 0
            self._scan_cursor_total = 0
            self._terminal_status_reads = ()
            self._terminal_cursor_reads = ()
            self._observer_error = ""
            self._poll_failures = 0
            self._resends_at_fire = int(getattr(self.transport, "resends", 0) or 0)
            self._terminal_status = STATUS_RUNNING
            self._fire_started = time.monotonic()
            self._clear_safe_readback_locked()
            self._fire_gate = threading.Event()
            self._worker = threading.Thread(target=self._observe, name="zlc-pulse-observer", daemon=True)
            self._worker.start()
            try:
                self._write(
                    (
                        (CtrlWords.SCAN_COUNT, self._scan_count),
                        (CtrlWords.SCAN_ENABLE, int(bool(self._scan_rows))),
                        (CtrlWords.RUN_REPEAT_COUNT, run_repeats),
                        (CtrlWords.SCAN_REPEAT_COUNT, scan_repeats),
                    )
                    + self._scan_bank_arming(),
                    stop=self._stop,
                )
                self._strobe(
                    CMD_FIRE,
                    took_effect=self._fire_took_effect,
                    stop=self._stop,
                )
            except BaseException:
                self._stop_worker()
                raise
            self._fire_gate.set()

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        with self._lock:
            self._require_open()
            if not self._firing:
                return None
        if not self._done.wait(timeout):
            return None
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
            if worker.is_alive():
                raise RuntimeError("pulse observer did not exit after terminal readback")
        with self._lock:
            status_reads = self._terminal_status_reads
            cursor_reads = self._terminal_cursor_reads
            poll_failures = self._poll_failures
            resent_frames = (
                int(getattr(self.transport, "resends", 0) or 0) - self._resends_at_fire
            )
        if len(status_reads) != 2 or len(cursor_reads) != 2:
            raise RuntimeError("observer completed without terminal readback")
        report = DoneReport(
            status=status_reads[-1],
            cursor=cursor_reads[-1],
            underflow=bool(status_reads[-1] & STATUS_UNDERFLOW) or self._underflow,
            elapsed_seconds=max(0.0, time.monotonic() - self._fire_started),
            status_reads=status_reads,
            cursor_reads=cursor_reads,
            observer_error=self._observer_error,
            poll_failures=poll_failures,
            resent_frames=resent_frames,
        )
        with self._lock:
            self._firing = False
            self._worker = None
            self._fire_gate = None
            self._terminal_status = report.status
        return report
    def cursor(self) -> int | None:
        with self._lock:
            if self._firing:
                return self._cursor_value
        return self._read(CtrlWords.CURSOR) if self._opened else None
    def safe(self) -> SafeReadback:
        self._stop_worker()
        with self._lock:
            self._require_open()
            if self._safe_readback_current_locked():
                status_reads = (0, 0)
                clock_words = self._safe_clock_enable_words or ()
            else:
                status_reads, clock_words = self._drive_physical_safe(
                    deadline=time.monotonic() + SAFE_TIMEOUT
                )
                self._record_safe_readback_locked(status_reads[-1], clock_words)
            self._firing = self._hardware_loaded = self._scan_armed = False
            self._done.clear()
            self._terminal_status = status_reads[-1]
            return SafeReadback(status_reads, clock_words, True)
    def describe(self) -> BoardDescription:
        """The board this streamer drives, as its handshake proved it to be.

        Open first: before the layout word has been read back there is nothing
        to be confident about, and a description that might be of a different
        bitstream is worse than none.
        """

        with self._lock:
            self._require_open()
            params = self.geom
            return BoardDescription(
                target=self._target,
                geometry=params,
                clock_hz=float(self.clock_hz),
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "opened": self._opened,
                "loaded": self._loaded,
                "firing": self._firing,
                "run_repeats": self._run_repeats,
                "scan_repeats": self._scan_repeats,
                "reloaded_before_fire": self._last_fire_reloaded,
                "cursor": self._cursor_value,
                "scan_count": self._scan_count,
                "scan_next_chunk": self._scan_next_chunk,
                "underflow": self._underflow,
                "status": self._terminal_status,
                # What it holds, not just that it holds something: a client
                # comparing this with its own program learns whether the board
                # is playing what that client is showing, without either side
                # remembering an answer that goes stale the moment anyone else
                # loads.
                "applied_digest": self._applied_digest,
            }
    def applied(self) -> AppliedState | None:
        with self._lock:
            return self._applied
    def _observe(self) -> None:
        """Poll STATUS/CURSOR until the board reports a terminal state.

        A poll that fails is a WARNING, not a verdict.  STATUS and CURSOR are
        idempotent reads and DONE is a level the board holds until SAFE, so
        nothing is lost by asking again: the answer the next poll gets is the
        same answer.  One failed poll used to end the observation with a
        fabricated ERROR, and the shot it condemned -- the board playing all
        200 Pulse runs, the camera collecting all 200 frames, SAFE acknowledged
        afterwards -- was thrown away over a single dropped byte.

        The transport's own transaction deadline is the grace: a poll fails
        only after the line has been asked again for that long without one
        good answer (``UartRegisterTransport._until_answered``).  The
        observation ends in error only when the poll that FOLLOWS such a
        failure fails too -- the same read-twice rule the terminal readback
        applies -- because two whole deadlines without an answer is a line
        that is down, not a byte that was lost.  One good poll in between
        restores the observation completely; the failure stays counted on
        the report.

        Two deadlines is therefore the longest the observation stays quiet
        on a dead line (about ten seconds at the default) -- twice what one
        verdict cost, by design, and no second knob.  A streamed scan does
        not wait that long to find out: its bank refill (``_refill``) is
        outside this grace, and a refill write that fails ends the
        observation at once, where a board starved of its next chunk would
        report UNDERFLOW on its own anyway.
        """

        try:
            gate = self._fire_gate
            while gate is not None and not gate.wait(self._observer_interval):
                if self._stop.is_set():
                    return
            consecutive_failures = 0
            while not self._stop.is_set():
                try:
                    status = self._read(CtrlWords.STATUS, stop=self._stop)
                    cursor = self._read(CtrlWords.CURSOR, stop=self._stop)
                except Exception as error:
                    if self._stop.is_set():
                        return
                    consecutive_failures += 1
                    with self._lock:
                        self._poll_failures += 1
                    if consecutive_failures >= 2:
                        self._record_observer_failure(error)
                        self._done.set()
                        return
                    continue
                consecutive_failures = 0
                with self._lock:
                    self._cursor_value = cursor
                    self._underflow = self._underflow or bool(status & STATUS_UNDERFLOW)
                if status & STATUS_UNDERFLOW:
                    self._finish_observation(status, cursor)
                    self._done.set()
                    return
                if status & (STATUS_DONE | STATUS_ERROR):
                    self._finish_observation(status, cursor)
                    self._done.set()
                    return
                if self._scan_rows:
                    self._refill(cursor)
                if self._stop.wait(self._observer_interval):
                    return
        except Exception as error:
            if not self._stop.is_set():
                self._record_observer_failure(error)
                self._done.set()
    def _record_observer_failure(self, error: BaseException) -> None:
        with self._lock:
            cursor = self._cursor_value or 0
            self._terminal_status_reads = (STATUS_ERROR, STATUS_ERROR)
            self._terminal_cursor_reads = (cursor, cursor)
            self._terminal_status = STATUS_ERROR
            self._observer_error = f"{type(error).__name__}: {error}"
    def _finish_observation(self, first_status: int, first_cursor: int) -> None:
        try:
            second_status = self._read(CtrlWords.STATUS, stop=self._stop)
            second_cursor = self._read(CtrlWords.CURSOR, stop=self._stop)
        except Exception:
            second_status = first_status
            second_cursor = first_cursor
            with self._lock:
                self._poll_failures += 1
        with self._lock:
            self._terminal_status_reads = (first_status, second_status)
            self._terminal_cursor_reads = (first_cursor, second_cursor)
            self._terminal_status = second_status
    def _enter_safe(self, *, deadline: float) -> tuple[int, int]:
        self._write(((CtrlWords.STATUS, STATUS_ERROR),), deadline=deadline)
        self._strobe(CMD_SAFE, deadline=deadline, repeatable=True)
        retry_at = time.monotonic() + min(
            SAFE_RETRY_AFTER,
            max(SAFE_POLL_INTERVAL, (deadline - time.monotonic()) / 2),
        )
        retried = False
        stable_zero = 0
        while time.monotonic() < deadline:
            status = self._read(CtrlWords.STATUS, deadline=deadline)
            stable_zero = stable_zero + 1 if status == 0 else 0
            if stable_zero >= 2:
                return (0, 0)
            if not retried and time.monotonic() >= retry_at:
                self._strobe(CMD_SAFE, deadline=deadline, repeatable=True)
                retried = True
            remaining = deadline - time.monotonic()
            # Once zero is observed, take the adjacent confirming read
            # immediately; a non-zero acknowledgement keeps the 1 ms cadence.
            if remaining > 0 and status != 0:
                time.sleep(min(SAFE_POLL_INTERVAL, remaining))
        raise TimeoutError("pulse streamer did not acknowledge SAFE with stable STATUS=0")
    def _drive_physical_safe(self, *, deadline: float) -> tuple[tuple[int, int], tuple[int, ...]]:
        self._enter_safe(deadline=deadline)
        self._write(
            tuple((CtrlWords.CLK_ENABLE + i, 0) for i in range(self.geom.clk_enable_words)),
            deadline=deadline,
        )
        clock_words = tuple(
            self._read(CtrlWords.CLK_ENABLE + i, deadline=deadline)
            for i in range(self.geom.clk_enable_words)
        )
        if any(clock_words):
            raise RuntimeError("pulse SAFE could not verify that every live clock mux is disabled")
        status_reads = self._enter_safe(deadline=deadline)
        return status_reads, clock_words
    @staticmethod
    def _command(code: int) -> tuple[tuple[int, int], ...]:
        """Strobe one command by returning COMMAND to zero first."""

        return ((CtrlWords.COMMAND, 0), (CtrlWords.COMMAND, int(code)))

    def _validate_application(
        self,
        program: CompiledProgram,
        rows: tuple[tuple[int, ...], ...],
    ) -> None:
        if program.target_abi_fingerprint != self._target.abi_fingerprint:
            raise ValueError("compiled target ABI does not match the connected sequencer")
        if float(program.clock_hz) != self.clock_hz:
            raise ValueError("compiled clock does not match the connected sequencer")
        expected_geometry = build_fingerprint(self.geom)
        if program.geometry_fingerprint != expected_geometry:
            raise ValueError("compiled geometry does not match the connected sequencer")
        for row in rows or ((),):
            self._validate_slot_row(program, row)
        self._validate_delay_capacity(program, rows, 1, 1)

    def _validate_slot_row(
        self,
        program: CompiledProgram,
        row: Sequence[int],
        *,
        require_outer_seam: bool = False,
    ) -> None:
        # A value the multiplier cannot hold is refused, not wrapped.  The host
        # and the board now agree about what a wrapped value plays.  Duration
        # slots are signed deltas around a full-width base, so this limits the
        # scan span rather than the absolute period.
        width = slot_operand_width()
        limit = 1 << (width - 1)
        for index, value in enumerate(row):
            if not -limit <= int(value) < limit:
                raise ValueError(
                    f"scan slot {index} value {int(value)} does not fit the board's "
                    f"{width}-bit signed multiplier operand "
                    f"([{-limit}, {limit - 1}])"
                )
        effective = tuple(
            evaluate_affine_tick(base, coeffs, row, program.scan_coeff_frac_bits)
            for base, coeffs in zip(program.ticks, program.tick_slot_coeffs)
        )
        tick_limit = 1 << self.geom.tick_width
        if (
            effective[0] != 0
            or any(value < 0 or value >= tick_limit for value in effective)
            or any(right <= left for left, right in zip(effective, effective[1:]))
        ):
            raise ValueError(
                "slot row makes compiled edge ticks collide or leave the "
                "unsigned hardware tick range"
            )
        loop_start = evaluate_affine_tick(
            program.ticks[program.loop_start_index],
            program.tick_slot_coeffs[program.loop_start_index],
            row,
            program.scan_coeff_frac_bits,
        )
        loop_end = evaluate_affine_tick(
            program.loop_end_tick,
            program.loop_end_slot_coeffs,
            row,
            program.scan_coeff_frac_bits,
        )
        if loop_end <= loop_start or loop_end > effective[-1]:
            raise ValueError("slot row makes compiled loop metadata invalid")
        if program.loop_count == 2 and loop_end < _MIN_SEAM_SPAN_TICKS:
            raise ValueError(
                "PulseBracket boundary must occur at or after "
                f"hardware tick {_MIN_SEAM_SPAN_TICKS}"
            )
        if program.loop_count > 2 and loop_end - loop_start < _MIN_SEAM_SPAN_TICKS:
            raise ValueError(
                "PulseBracket span must be at least "
                f"{_MIN_SEAM_SPAN_TICKS} hardware ticks"
            )
        outer_origin = loop_start if program.loop_count > 1 else 0
        if require_outer_seam and effective[-1] - outer_origin < _MIN_SEAM_SPAN_TICKS:
            raise ValueError(
                "each Pulse run before another run must leave at least "
                f"{_MIN_SEAM_SPAN_TICKS} hardware ticks after its final restart"
            )

    def _validate_delay_capacity(
        self,
        program: CompiledProgram,
        rows: tuple[tuple[int, ...], ...],
        run_repeats: int,
        scan_repeats: int,
    ) -> None:
        """Reject an application whose delayed events overflow frozen FIFOs.

        A queue entry remains live through the edge on which it is emitted;
        the RTL computes push eligibility from the pre-edge count.  Capacity
        therefore uses a closed ``delay`` window, including events exactly one
        delay apart.
        """

        ttl = tuple(
            (index, int(delay))
            for index, delay in enumerate(program.channel_delays)
            if int(delay) >= 2
        )
        bus_delays = {
            int(item.bus_index): int(item.delay_ticks)
            for item in program.bus_delays
            if int(item.delay_ticks) > 0
        }
        if not ttl and not bus_delays:
            return

        table = rows or ((),)
        # After ``depth`` identical whole-Pulse executions the FIFO has either
        # overflowed or reached its periodic state.  Preserve the real nesting
        # order while bounding each row's Run repeats.  The final complete
        # sweep plus at most ``depth`` preceding executions validates every row,
        # the sweep seam, and finite terminal SAFE without materializing a
        # 32-bit repeat count.
        depth = max(self.geom.evt_fifo_depth, self.geom.bus_evt_fifo_depth)
        if run_repeats == 0:
            execution_rows = (table[0],) * (depth + 1)
            finite_completion = False
        else:
            bounded_run_repeats = min(run_repeats, depth + 1)
            one_sweep = tuple(
                row
                for row in table
                for _ in range(bounded_run_repeats)
            )
            finite_completion = scan_repeats != 0
            preceding_executions = (
                depth
                if scan_repeats == 0
                else min(depth, (scan_repeats - 1) * len(one_sweep))
            )
            if preceding_executions:
                copies = (
                    preceding_executions + len(one_sweep) - 1
                ) // len(one_sweep)
                warmup = (one_sweep * copies)[-preceding_executions:]
            else:
                warmup = ()
            # Model the last complete sweep after the exact periodic suffix
            # which can still own FIFO entries.  This includes an intermediate
            # sweep seam, ends on the real terminal row, and remains bounded by
            # one sweep plus ``depth`` whole-Pulse executions.
            execution_rows = warmup + one_sweep
        # The same argument bounds an internal PulseBracket: depth+1 bodies are
        # enough to prove overflow or periodic boundedness.
        checked_program = (
            program
            if program.loop_count <= depth + 1
            else replace(program, loop_count=depth + 1)
        )

        physical_to_logical = {
            physical: logical
            for logical, physical in program.logical_digital_outputs
        }
        for bit, _delay in ttl:
            if bit >= len(program.channels):
                raise ValueError(f"channel delay index {bit} is outside the program")
        # ONE WALK of the run, not one per delayed channel.  A single
        # negative delay makes every driven lane a delayed channel, and this
        # runs inside fire() before the board is strobed -- so the walk that
        # is identical for all of them ran nine times while the operator
        # waited on Run.
        asked = tuple(
            physical_to_logical.get(program.channels[bit], program.channels[bit])
            for bit, _delay in ttl
        )
        # Compiled digital masks always end low, so the finite terminal SAFE
        # creates no additional TTL transition: the final falling edge is
        # already part of this schedule.  DAC state may end away from its safe
        # code, which is why its explicit terminal descriptor is added below.
        edges = trigger_edge_ticks(
            checked_program,
            asked,
            execution_rows,
            run_repeats=1,
            scan_repeats=1,
        )
        for (bit, delay), logical in zip(ttl, asked):
            self._check_delay_window(
                edges[logical],
                delay,
                self.geom.evt_fifo_depth,
                f"channel {program.channels[bit]!r}",
            )

        if bus_delays:
            by_bus: dict[int, list[int]] = {bus: [] for bus in bus_delays}
            run_offset = 0
            for point in execution_rows:
                effective = tuple(
                    evaluate_affine_tick(
                        base,
                        coefficients,
                        point,
                        checked_program.scan_coeff_frac_bits,
                    )
                    for base, coefficients in zip(
                        checked_program.ticks,
                        checked_program.tick_slot_coeffs,
                    )
                )
                loop_start = effective[checked_program.loop_start_index]
                loop_end = evaluate_affine_tick(
                    checked_program.loop_end_tick,
                    checked_program.loop_end_slot_coeffs,
                    point,
                    checked_program.scan_coeff_frac_bits,
                )
                span = loop_end - loop_start
                final = effective[-1]
                total = final + (checked_program.loop_count - 1) * span
                for segment in checked_program.bus_segments:
                    bus = int(segment.bus_index)
                    if bus not in by_bus:
                        continue
                    start = evaluate_affine_tick(
                        segment.start_tick,
                        segment.start_tick_coeffs,
                        point,
                        checked_program.scan_coeff_frac_bits,
                    )
                    if start < loop_start:
                        by_bus[bus].append(run_offset + start)
                    elif start < loop_end:
                        by_bus[bus].extend(
                            run_offset + start + iteration * span
                            for iteration in range(checked_program.loop_count)
                        )
                    else:
                        by_bus[bus].append(
                            run_offset
                            + start
                            + (checked_program.loop_count - 1) * span
                        )
                run_offset += total
            # Finite completion captures one final SAFE descriptor per bus.
            if finite_completion:
                for events in by_bus.values():
                    events.append(run_offset)
            for bus, events in by_bus.items():
                self._check_delay_window(
                    sorted(events),
                    bus_delays[bus],
                    self.geom.bus_evt_fifo_depth,
                    f"DAC bus {bus}",
                )

    @staticmethod
    def _check_delay_window(
        events: Sequence[int],
        delay: int,
        capacity: int,
        label: str,
    ) -> None:
        first = 0
        for last, tick in enumerate(events):
            while first < last and events[first] < tick - delay:
                first += 1
            required = last - first + 1
            if required > capacity:
                raise ValueError(
                    f"{label} needs {required} delayed events in flight but the "
                    f"connected geometry holds {capacity}"
                )

    def _await_loaded(self, *, stop: threading.Event | None = None) -> None:
        """The RTL gates FIRE on STATUS_LOADED, so an incomplete load would
        turn every later fire into a silent no-op.  Report it here instead."""

        deadline = time.monotonic() + LOAD_TIMEOUT
        while True:
            status = self._read(CtrlWords.STATUS, stop=stop)
            if status & STATUS_ERROR:
                raise RuntimeError(f"pulse streamer reported STATUS_ERROR during load (0x{status:08X})")
            if status & STATUS_LOADED:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"pulse streamer did not report LOADED within {LOAD_TIMEOUT}s "
                    f"(STATUS=0x{status:08X})"
                )
            time.sleep(self._observer_interval)

    def _scan_bank_arming(self) -> tuple[tuple[int, int], ...]:
        """Rows that put the scan banks back at chunks 0/1, ready for point 0.

        Single source for arming the application rows for a fire; empty when
        the banks already hold chunks 0/1.
        """

        if self._scan_count == 0 or self._scan_armed:
            return ()
        ready = self._initial_ready(self._scan_count)
        rows: list[tuple[int, int]] = []
        for chunk in (0, 1):
            if chunk * self.geom.bank_size < self._scan_count:
                rows.extend(sorted(pack_scan_rows(
                    self._scan_rows, self.geom, chunk & 1, chunk
                ).items()))
        rows.append((CtrlWords.BANK0_CHUNK, 0))
        if self.geom.bank_size < self._scan_count:
            rows.append((CtrlWords.BANK1_CHUNK, 1))
        rows.append((CtrlWords.BANK_READY, ready))
        self._scan_next_chunk = 2
        self._scan_ready = ready
        self._scan_armed = True
        self._scan_last_cursor = 0
        self._scan_cursor_total = 0
        return tuple(rows)

    def _refill(self, cursor: int) -> None:
        """Keep the far bank one chunk ahead of where the engine is playing.

        CURSOR is a cumulative row-visit ordinal, not a sampled table index, so
        division by the unique row count identifies the sweep without requiring
        the observer to witness a wrap.  When the engine enters chunk c+1 it
        frees c's bank and the host loads the next monotonic chunk into it.
        """

        if self._scan_count <= 2 * self.geom.bank_size:
            return
        if cursor < 0 or cursor > MAXIMUM_REPEAT_COUNT:
            raise RuntimeError(
                f"board cursor {cursor} is outside its unsigned 32-bit range"
            )
        self._scan_cursor_total += (
            cursor - self._scan_last_cursor
        ) & MAXIMUM_REPEAT_COUNT
        self._scan_last_cursor = cursor
        chunks_per_sweep = (
            self._scan_count + self.geom.bank_size - 1
        ) // self.geom.bank_size
        sweep_index, table_row = divmod(self._scan_cursor_total, self._scan_count)
        current_stream_chunk = (
            sweep_index * chunks_per_sweep
            + table_row // self.geom.bank_size
        )
        while self._scan_next_chunk <= current_stream_chunk + 1:
            if (
                self._scan_repeats != 0
                and self._scan_next_chunk >= chunks_per_sweep * self._scan_repeats
            ):
                return
            stream_chunk = self._scan_next_chunk
            table_chunk = stream_chunk % chunks_per_sweep
            bank = stream_chunk & 1
            bit = 1 << bank
            unarmed = self._scan_ready & ~bit
            words = pack_scan_rows(
                self._scan_rows, self.geom, bank, table_chunk
            )
            chunk_reg = CtrlWords.BANK0_CHUNK if bank == 0 else CtrlWords.BANK1_CHUNK
            self._write((
                (CtrlWords.BANK_READY, unarmed),
                *tuple(sorted(words.items())),
                (chunk_reg, table_chunk),
                (CtrlWords.BANK_READY, self._scan_ready | bit),
            ), stop=self._stop)
            self._scan_next_chunk += 1
            self._scan_armed = False

    def _stop_worker(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker is None:
            return
        if self._fire_gate is not None:
            self._fire_gate.set()
        if worker is not threading.current_thread():
            worker.join(timeout=2.0)
        if worker.is_alive():
            raise RuntimeError("pulse observer did not stop")
        with self._lock:
            if self._worker is worker:
                self._worker = None
                self._fire_gate = None

    def _read(
        self,
        address: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> int:
        options = {} if stop is None else {"stop": stop}
        if deadline is not None:
            options["deadline"] = deadline
        value = self.transport.read_word(address, **options)
        return int(value) & 0xFFFFFFFF

    def _write(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> None:
        """Write register words; a frame the link loses is sent again.

        Data only.  A COMMAND strobe goes through _strobe, which does NOT
        resend: a command that was executed and whose acknowledgement was lost
        would be executed twice, and "fire twice" is not a recoverable
        arithmetic error.
        """

        normalized = tuple((int(address), int(value) & 0xFFFFFFFF) for address, value in rows)
        assert not any(address == CtrlWords.COMMAND for address, _ in normalized), (
            "a command strobe must go through _strobe, which never resends"
        )
        options = {} if stop is None else {"stop": stop}
        if deadline is not None:
            options["deadline"] = deadline
        self.transport.write_words(normalized, **options)

    def _strobe(
        self,
        code: int,
        *,
        deadline: float | None = None,
        repeatable: bool = False,
        took_effect: "Callable[[], bool] | None" = None,
        stop: threading.Event | None = None,
    ) -> None:
        """Fire one command, and never a second time BLINDLY.

        Sent after the data it acts on has been acknowledged, so the board is
        never asked to act on a program that is still arriving.

        Three kinds of command, three policies, because "may this be sent
        again?" has three honest answers:

        * ``repeatable=True`` -- SAFE and LOAD.  Their effect is idempotent
          (safing a safe board is safe, reloading the resident image is the
          same image), so a lost acknowledgement is handled like any lost data
          frame: the line resends it.

        * ``took_effect`` given -- FIRE.  Running twice is two shots, so a
          lost acknowledgement may not be resolved by guessing.  It does not
          have to be: the board KNOWS whether it fired -- accepting FIRE
          consumes the LOADED gate and raises RUNNING -- so the ambiguity is
          resolved by reading the status.  Executed: done, nobody mourns the
          acknowledgement.  Provably not executed: strobing again is exactly
          as safe as the first attempt was.  On a line that loses one byte in
          a hundred, this is the difference between an experiment that runs
          and one that dies every sixth On Pulse.

        * Neither -- an unanswered strobe stays fatal, because guessing is
          the one thing this path must never do.

        How long the acknowledgement is waited for is the line's business,
        not this method's: a non-resending strobe is one attempt, and the
        line budgets an attempt by the bytes in flight
        (``UartRegisterTransport._attempt_budget``).  This method once handed
        the line a 0.3 s window of its own for the verified case -- the same
        claim owned twice, and the second owner dead, because the line's
        budget was already the shorter of the two.
        """

        rows = self._command(code)
        # Verification-by-status is sound only on a line that LOSES things: a
        # UART frame either executes within microseconds or is gone forever,
        # so after a timeout nothing is still in flight.  A Vivado TCL that
        # timed out may still execute later -- verify would read "idle", the
        # strobe would go again, and the late original would make it two
        # shots.  The transport declares which world it lives in.
        verified = took_effect is not None and bool(
            getattr(self.transport, "lossy_line", False)
        )
        attempts = 3 if verified else 1
        for attempt in range(attempts):
            options: dict = {} if deadline is None else {"deadline": deadline}
            if stop is not None:
                options["stop"] = stop
            try:
                self.transport.write_words(rows, resend=repeatable, **options)
                return
            except TimeoutError:
                if not verified:
                    raise
                if took_effect():
                    return
                if attempt + 1 == attempts:
                    raise

    def _fire_took_effect(self) -> bool:
        """Did the board hear CMD_FIRE?  Its status register is the witness.

        Accepting FIRE consumes the RTL's LOADED gate and raises RUNNING; a
        short program may already be DONE by the time anyone looks.  A board
        still advertising LOADED with none of that happened did not hear the
        command.
        """

        status = self._read(CtrlWords.STATUS, stop=self._stop)
        heard = bool(status & (STATUS_RUNNING | STATUS_DONE))
        return heard or not bool(status & STATUS_LOADED)

    def _initial_ready(self, count: int) -> int:
        return (1 if count > 0 else 0) | (2 if count > self.geom.bank_size else 0)

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("PulseStreamer is not open")

    def _require_loaded(self) -> None:
        if not self._loaded or self._program is None:
            raise RuntimeError("no compiled program is loaded")

    def _require_idle(self) -> None:
        if self._firing:
            raise RuntimeError("the streamer is already firing")

    def _check_register_layout_locked(self) -> None:
        layout = self._read(CtrlWords.LAYOUT_ID)
        expected = build_fingerprint(self.geom)
        if layout != expected:
            raise RuntimeError(
                f"geometry/layout mismatch: device=0x{layout:08X}, host=0x{expected:08X}"
            )

    def _safe_readback_current_locked(self) -> bool:
        return (
            self._safe_status_word == 0
            and self._safe_clock_enable_words is not None
            and len(self._safe_clock_enable_words) == self.geom.clk_enable_words
            and not any(self._safe_clock_enable_words)
        )

    def _clear_safe_readback_locked(self) -> None:
        self._safe_status_word = None
        self._safe_clock_enable_words = None

    def _record_safe_readback_locked(
        self,
        status_word: int,
        clock_enable_words: tuple[int, ...],
    ) -> None:
        if status_word != 0 or len(clock_enable_words) != self.geom.clk_enable_words or any(clock_enable_words):
            raise ValueError("physical SAFE readback facts are not safe")
        self._safe_status_word = int(status_word)
        self._safe_clock_enable_words = tuple(int(value) for value in clock_enable_words)


__all__ = [
    "AppliedState",
    "BoardDescription",
    "DoneReport",
    "PulseStreamer",
    "SafeReadback",
]
