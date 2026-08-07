"""Physical pulse-streamer session with a deliberately small state surface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence
import math
import threading
import time

from collections.abc import Mapping

from .compile import CompiledProgram, evaluate_affine_tick, slot_operand_width
from .model import PORT_DAC, PulseSequence, PulseTarget
from .transport.base import DEFAULT_OBSERVER_INTERVAL, RegisterTransport
from .wire import (
    CMD_FIRE,
    CMD_LOAD,
    CMD_SAFE,
    CTRL_WORDS,
    CtrlWords,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_LOADED,
    STATUS_RUNNING,
    STATUS_UNDERFLOW,
    StreamerParams,
    build_fingerprint,
    pack_program,
    pack_scan_rows,
    region_bases,
)


# v1 transport/session.py used the same 5 s budget for the loader handshake.
LOAD_TIMEOUT = 5.0
SAFE_TIMEOUT = 5.0
SAFE_RETRY_AFTER = 0.05
#: How long a VERIFIED strobe waits for its acknowledgement before asking the
#: status register instead.  Short on purpose: the read is the arbiter, the
#: acknowledgement is a courtesy, and waiting the full action timeout for a
#: courtesy stalled every lost FIRE ack for five seconds.
STROBE_VERIFY_AFTER = 0.3
SAFE_POLL_INTERVAL = 0.001


@dataclass(frozen=True)
class DoneReport:
    status: int
    cursor: int | None
    underflow: bool
    tail_elapsed: float
    status_reads: tuple[int, int] = ()
    cursor_reads: tuple[int, int] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", int(self.status))
        object.__setattr__(self, "cursor", None if self.cursor is None else int(self.cursor))
        object.__setattr__(self, "underflow", bool(self.underflow))
        object.__setattr__(self, "tail_elapsed", float(self.tail_elapsed))
        object.__setattr__(self, "status_reads", tuple(int(value) for value in self.status_reads))
        object.__setattr__(self, "cursor_reads", tuple(int(value) for value in self.cursor_reads))


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
        if self.status & STATUS_ERROR:
            reasons.append("the board reported an error")
        if self.status & STATUS_UNDERFLOW or self.underflow:
            reasons.append("the scan bank underran")
        return "; ".join(reasons)

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
    """Immutable echo of the last program/table application owned by the device."""

    program: CompiledProgram
    source: PulseSequence | None
    slot_values: tuple[int, ...]
    scan_rows: tuple[tuple[int, ...], ...]
    forever: bool
    loaded_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.program, CompiledProgram):
            raise TypeError("applied program must be CompiledProgram")
        if self.source is not None and not isinstance(self.source, PulseSequence):
            raise TypeError("applied source must be PulseSequence or None")
        slot_values = tuple(int(value) for value in self.slot_values)
        scan_rows = tuple(tuple(int(value) for value in row) for row in self.scan_rows)
        if slot_values and scan_rows:
            raise ValueError("applied state cannot contain both slot_values and scan_rows")
        if len(slot_values) not in (0, self.program.slot_count):
            raise ValueError("applied slot row width differs from the compiled program")
        if any(len(row) != self.program.slot_count for row in scan_rows):
            raise ValueError("applied scan row width differs from the compiled program")
        loaded_at = float(self.loaded_at)
        if not math.isfinite(loaded_at) or loaded_at < 0:
            raise ValueError("applied loaded_at must be a finite non-negative timestamp")
        object.__setattr__(self, "slot_values", slot_values)
        object.__setattr__(self, "scan_rows", scan_rows)
        object.__setattr__(self, "forever", bool(self.forever))
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
    clock_hz: float
    time_step_ns: float
    layout_fingerprint: int
    channel_count: int
    bus_count: int
    bus_width: int


class PulseStreamer:
    """Host control of the frozen streamer.

    After FIRE, the observer is the sole caller that reads status/cursor or
    refills scan banks.  Public cursor reads use its cached sample.
    """

    def __init__(self, transport: RegisterTransport, geom: StreamerParams, clock_hz: float) -> None:
        if not isinstance(geom, StreamerParams):
            raise TypeError("geom must be StreamerParams")
        if clock_hz <= 0:
            raise ValueError("clock_hz must be positive")
        self.transport = transport
        observer_interval = getattr(transport, "observer_interval", DEFAULT_OBSERVER_INTERVAL)
        if isinstance(observer_interval, bool) or not math.isfinite(float(observer_interval)) or observer_interval <= 0:
            raise ValueError("transport observer_interval must be finite and positive")
        self._observer_interval = float(observer_interval)
        self.geom = geom
        self.clock_hz = float(clock_hz)
        self._lock = threading.RLock()
        self._opened = False
        # The board's pin-aware target, read from the XDC beside this config the
        # first time anyone asks what the board is.
        self._target: PulseTarget | None = None
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
        self._forever = False
        self._scan_rows: tuple[tuple[int, ...], ...] = ()
        self._scan_next_chunk = 2
        self._scan_ready = 0
        self._scan_armed = False
        self._scan_count = 0
        #: How many times the table is played.  One table, many sweeps.
        self._scan_sweeps = 1
        self._cursor_value: int | None = None
        self._underflow = False
        self._terminal_status_reads: tuple[int, int] = ()
        self._terminal_cursor_reads: tuple[int, int] = ()
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
    def load(self, prog: CompiledProgram, *, source: PulseSequence | None = None) -> None:
        if not isinstance(prog, CompiledProgram):
            raise TypeError("prog must be CompiledProgram")
        if source is not None and not isinstance(source, PulseSequence):
            raise TypeError("source must be PulseSequence or None")
        with self._lock:
            self._require_open()
            self._require_idle()
            if not self._safe_readback_current_locked():
                self._drive_physical_safe(deadline=time.monotonic() + SAFE_TIMEOUT)
            self._clear_safe_readback_locked()
            self._loaded = self._hardware_loaded = False; self._program = None; self._applied = None
            self._applied_digest = ""
            self._scan_rows = (); self._scan_count = 0; self._scan_sweeps = 1; self._scan_armed = False
            words = pack_program(prog, self.geom)
            self._write(
                tuple(sorted(words.items()))
                + ((CtrlWords.BANK_READY, 0b11),)
            )
            self._strobe(CMD_LOAD, repeatable=True)
            self._await_loaded(); self._hardware_loaded = True; self._program = prog
            self._loaded = True
            self._scan_rows = tuple(prog.scan_points)
            self._scan_count = len(self._scan_rows)
            self._scan_next_chunk = 2
            self._scan_ready = self._initial_ready(self._scan_count)
            self._scan_armed = False
            self._cursor_value = 0
            self._underflow = False
            self._terminal_status_reads = ()
            self._terminal_cursor_reads = ()
            self._applied = AppliedState(
                program=prog,
                source=source,
                slot_values=(),
                scan_rows=tuple(self._scan_rows),
                forever=False,
                loaded_at=time.time(),
            )
            self._applied_digest = prog.digest
    def write_slots(self, values: Sequence[int]) -> None:
        with self._lock:
            self._require_open()
            self._require_loaded()
            self._require_idle()
            assert self._program is not None
            values = tuple(int(value) for value in values)
            if len(values) != self._program.slot_count:
                raise ValueError("slot row width differs from the compiled program")
            self._validate_slot_row(values)
            self._scan_rows = (values,)
            self._scan_count = 1; self._scan_armed = False
            self._write(
                ((CtrlWords.SCAN_COUNT, 1), (CtrlWords.SCAN_ENABLE, 1))
                + self._scan_bank_arming()
            )
            assert self._applied is not None
            self._applied = replace(self._applied, slot_values=values, scan_rows=())
    def write_scan_table(self, rows: Sequence[Sequence[int]], *, sweeps: int = 1) -> None:
        """Give the board one scan table, played ``sweeps`` times.

        The table crosses the wire ONCE.  There is no sweep register in the
        RTL -- scan_count is a point count, repeat_forever is a bit, and
        loop_count is the pulse timeline's own loop inside each point -- so N
        sweeps is N*len(rows) points, and which row a point takes is decided
        when a bank is refilled.  Duplicating the rows to say "again" would
        send the same numbers N times and hold N copies of them here.
        """

        with self._lock:
            self._require_open()
            self._require_loaded()
            self._require_idle()
            assert self._program is not None
            normalized = tuple(tuple(int(value) for value in row) for row in rows)
            if not normalized:
                raise ValueError("scan table must contain at least one row")
            if any(len(row) != self._program.slot_count for row in normalized):
                raise ValueError("scan table width differs from the compiled program")
            for row in normalized:
                self._validate_slot_row(row)
            self._scan_rows = normalized
            self._scan_sweeps = max(1, int(sweeps))
            self._scan_count = len(normalized) * self._scan_sweeps
            self._scan_armed = False
            self._write(
                (
                    (CtrlWords.SCAN_COUNT, self._scan_count),
                    (CtrlWords.SCAN_ENABLE, 1),
                )
                + self._scan_bank_arming()
            )
            assert self._applied is not None
            self._applied = replace(self._applied, slot_values=(), scan_rows=normalized)
    def fire(self, *, forever: bool = False) -> None:
        with self._lock:
            self._require_open()
            self._require_loaded()
            self._require_idle()
            self._forever = bool(forever)
            assert self._applied is not None
            self._applied = replace(self._applied, forever=self._forever)
            # DONE/SAFE clear the RTL's LOADED gate; replay only its resident mini-loader.
            self._last_fire_reloaded = not self._hardware_loaded
            if self._last_fire_reloaded:
                self._write(self._scan_bank_arming())
                self._strobe(CMD_LOAD, repeatable=True)
                self._await_loaded(); self._hardware_loaded = True
            self._firing = True; self._hardware_loaded = False
            self._done.clear()
            self._stop.clear()
            self._underflow = False
            self._cursor_value = 0
            self._terminal_status_reads = ()
            self._terminal_cursor_reads = ()
            self._terminal_status = STATUS_RUNNING
            self._fire_started = time.monotonic()
            self._clear_safe_readback_locked()
            self._fire_gate = threading.Event()
            self._worker = threading.Thread(target=self._observe, name="zlc-pulse-observer", daemon=True)
            self._worker.start()
            try:
                self._write(
                    ((CtrlWords.REPEAT_FOREVER, int(self._forever)),)
                    + self._scan_bank_arming()
                )
                self._strobe(CMD_FIRE, took_effect=self._fire_took_effect)
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
        with self._lock:
            status_reads = self._terminal_status_reads
            cursor_reads = self._terminal_cursor_reads
        if len(status_reads) != 2 or len(cursor_reads) != 2:
            raise RuntimeError("observer completed without terminal readback")
        report = DoneReport(
            status=status_reads[-1],
            cursor=cursor_reads[-1],
            underflow=bool(status_reads[-1] & STATUS_UNDERFLOW) or self._underflow,
            tail_elapsed=max(0.0, time.monotonic() - self._fire_started),
            status_reads=status_reads,
            cursor_reads=cursor_reads,
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
            self._firing = self._hardware_loaded = False
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
            target = self._board_target()
            params = self.geom
            return BoardDescription(
                target=target,
                clock_hz=float(self.clock_hz),
                time_step_ns=1e9 / float(self.clock_hz),
                layout_fingerprint=int(build_fingerprint(params)),
                channel_count=int(params.channel_count),
                bus_count=int(params.bus_count),
                bus_width=int(params.bus_width),
            )

    def _board_target(self) -> PulseTarget:
        """The pin-aware target for THIS board, read once.

        The XDC is not proven by anything -- the LAYOUT_ID handshake proves the
        geometry this streamer was constructed with, and nothing else.  So the
        target read off disk is checked against that geometry before it is
        handed out: a description used to mix the two, taking its counts from
        the proven geometry and its ports and pins from whatever XDC happened
        to be the default, which is two different boards in one answer.
        """

        if self._target is None:
            from .manifest import pulse_target_from_xdc

            target = pulse_target_from_xdc()
            buses = tuple(port for port in target.ports if port.kind == PORT_DAC)
            widths = sorted({port.width for port in buses})
            mismatches = []
            if len(target.raw_lanes) != int(self.geom.channel_count):
                mismatches.append(
                    f"lanes={len(target.raw_lanes)} but this board has "
                    f"{int(self.geom.channel_count)}"
                )
            if len(buses) != int(self.geom.bus_count):
                mismatches.append(
                    f"DAC buses={len(buses)} but this board has {int(self.geom.bus_count)}"
                )
            if widths and widths != [int(self.geom.bus_width)]:
                mismatches.append(
                    f"DAC widths={widths} but this board has {int(self.geom.bus_width)}"
                )
            if mismatches:
                raise ValueError(
                    "the XDC on this machine describes a different board than the "
                    "one this streamer is driving: " + "; ".join(mismatches)
                )
            self._target = target
        return self._target

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "opened": self._opened,
                "loaded": self._loaded,
                "firing": self._firing,
                "forever": self._forever,
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
        try:
            gate = self._fire_gate
            while gate is not None and not gate.wait(self._observer_interval):
                if self._stop.is_set():
                    return
            while not self._stop.is_set():
                try:
                    status = self._read(CtrlWords.STATUS)
                    cursor = self._read(CtrlWords.CURSOR)
                except Exception:
                    self._record_observer_failure()
                    self._done.set()
                    return
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
                if self._scan_rows and not self._forever:
                    self._refill(cursor)
                time.sleep(self._observer_interval)
        except Exception:
            if not self._stop.is_set():
                self._record_observer_failure()
                self._done.set()
    def _record_observer_failure(self) -> None:
        with self._lock:
            cursor = self._cursor_value or 0
            self._terminal_status_reads = (STATUS_ERROR, STATUS_ERROR)
            self._terminal_cursor_reads = (cursor, cursor)
            self._terminal_status = STATUS_ERROR
    def _finish_observation(self, first_status: int, first_cursor: int) -> None:
        try:
            second_status = self._read(CtrlWords.STATUS)
            second_cursor = self._read(CtrlWords.CURSOR)
        except Exception:
            second_status = first_status
            second_cursor = first_cursor
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
            # immediately; a non-zero acknowledgement keeps v1's 1 ms cadence.
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

    def _validate_slot_row(self, row: Sequence[int]) -> None:
        assert self._program is not None
        # A value the multiplier cannot hold is refused, not wrapped.  The host
        # and the board now agree about what a wrapped value plays, which means
        # they agree on something nobody asked for: a duration slot is in ticks,
        # so a period approaching a second at 50 MHz is already past the
        # operand width and would come back as a few microseconds.
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
            evaluate_affine_tick(base, coeffs, row, self._program.scan_coeff_frac_bits)
            for base, coeffs in zip(self._program.ticks, self._program.tick_slot_coeffs)
        )
        if effective[0] != 0 or any(right <= left for left, right in zip(effective, effective[1:])):
            raise ValueError("slot row makes compiled edge ticks collide or move before time zero")
        loop_start = evaluate_affine_tick(
            self._program.ticks[self._program.loop_start_index],
            self._program.tick_slot_coeffs[self._program.loop_start_index],
            row,
            self._program.scan_coeff_frac_bits,
        )
        loop_end = evaluate_affine_tick(
            self._program.loop_end_tick,
            self._program.loop_end_slot_coeffs,
            row,
            self._program.scan_coeff_frac_bits,
        )
        if loop_end <= loop_start or loop_end > effective[-1]:
            raise ValueError("slot row makes compiled loop metadata invalid")

    def _await_loaded(self) -> None:
        """The RTL gates FIRE on STATUS_LOADED, so an incomplete load would
        turn every later fire into a silent no-op.  Report it here instead."""

        deadline = time.monotonic() + LOAD_TIMEOUT
        while True:
            status = self._read(CtrlWords.STATUS)
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

        Single source for arming (write_slots / write_scan_table / a replaying
        fire); empty when the banks already hold chunks 0/1.
        """

        if self._scan_count == 0 or self._scan_armed:
            return ()
        ready = self._initial_ready(self._scan_count)
        rows: list[tuple[int, int]] = []
        for chunk in (0, 1):
            if chunk * self.geom.bank_size < self._scan_count:
                rows.extend(sorted(pack_scan_rows(
                    self._scan_rows, self.geom, chunk & 1, chunk, self._scan_sweeps
                ).items()))
        rows.append((CtrlWords.BANK0_CHUNK, 0))
        if self.geom.bank_size < self._scan_count:
            rows.append((CtrlWords.BANK1_CHUNK, 1))
        rows.append((CtrlWords.BANK_READY, ready))
        self._scan_next_chunk = 2
        self._scan_ready = ready
        self._scan_armed = True
        return tuple(rows)

    def _refill(self, cursor: int) -> None:
        """Keep the far bank one chunk ahead of where the engine is playing.

        The contract the RTL is built to, and which the cycle model implements,
        is one-ahead: when the engine crosses from chunk c into c+1 it frees the
        bank holding c and the host refills it with c+2.  This waited for the
        cursor to REACH the chunk it was about to write, which is one whole bank
        late -- so any scan longer than the two pre-armed chunks stalled at
        every chunk boundary while the host caught up.
        """

        while (self._scan_next_chunk - 1) * self.geom.bank_size <= cursor:
            first = self._scan_next_chunk * self.geom.bank_size
            if first >= self._scan_count:
                return
            chunk = self._scan_next_chunk
            bank = chunk & 1
            bit = 1 << bank
            unarmed = self._scan_ready & ~bit
            words = pack_scan_rows(self._scan_rows, self.geom, bank, chunk, self._scan_sweeps)
            chunk_reg = CtrlWords.BANK0_CHUNK if bank == 0 else CtrlWords.BANK1_CHUNK
            self._write((
                (CtrlWords.BANK_READY, unarmed),
                *tuple(sorted(words.items())),
                (chunk_reg, chunk),
                (CtrlWords.BANK_READY, self._scan_ready | bit),
            ))
            self._scan_next_chunk += 1
            self._scan_armed = False

    def _stop_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._stop.set()
        if self._fire_gate is not None:
            self._fire_gate.set()
        if worker is not threading.current_thread():
            worker.join(timeout=2.0)
        with self._lock:
            self._worker = None
            self._fire_gate = None
            self._firing = False

    def _read(self, address: int, *, deadline: float | None = None) -> int:
        if deadline is None:
            value = self.transport.read_word(address)
        else:
            value = self.transport.read_word(address, deadline=deadline)
        return int(value) & 0xFFFFFFFF

    def _write(self, rows: Sequence[tuple[int, int]], *, deadline: float | None = None) -> None:
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
        if deadline is None:
            self.transport.write_words(normalized)
        else:
            self.transport.write_words(normalized, deadline=deadline)

    def _strobe(
        self,
        code: int,
        *,
        deadline: float | None = None,
        repeatable: bool = False,
        took_effect: "Callable[[], bool] | None" = None,
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
        """

        rows = self._command(code)
        attempts = 3 if took_effect is not None else 1
        for attempt in range(attempts):
            options: dict = {} if deadline is None else {"deadline": deadline}
            if took_effect is not None and deadline is None:
                # The acknowledgement gets a short window, because it is not
                # the authority -- the status read below is.
                options = {"deadline": time.monotonic() + STROBE_VERIFY_AFTER}
            try:
                self.transport.write_words(rows, resend=repeatable, **options)
                return
            except TimeoutError:
                if took_effect is None:
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

        status = self._read(CtrlWords.STATUS)
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


__all__ = ["AppliedState", "DoneReport", "PulseStreamer", "SafeReadback"]
