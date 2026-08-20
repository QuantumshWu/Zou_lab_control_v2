"""Persistent Vivado/JTAG word transport for the frozen pulse streamer."""

from __future__ import annotations

import os
import math
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from .base import JTAG_AXI_OBSERVER_INTERVAL, TransportAborted


AXI_BURST_BOUNDARY_BYTES = 4096
_MAX_WORD_ADDRESS = 0xFFFFFFFF // 4
VIVADO_SEARCH_ROOTS = tuple(
    Path(f"{drive}:/{vendor}/Vivado")
    for drive in ("C", "D")
    for vendor in ("Xilinx", "AMD")
)


def _default_vivado() -> str:
    configured = os.environ.get("ZLC_PS_VIVADO_BIN")
    if configured:
        return configured
    candidates: list[Path] = []
    for root in VIVADO_SEARCH_ROOTS:
        if root.is_dir():
            candidates.extend(path for path in root.glob("*/bin/vivado.bat") if path.is_file())
    if candidates:
        candidates.sort(key=lambda path: str(path.parent.parent), reverse=True)
        return str(candidates[0])
    return shutil.which("vivado.bat") or shutil.which("vivado") or "vivado"


def _default_probes() -> str | None:
    for name in ("ZLC_PS_VIVADO_LTX", "ZLC_PS_LTX"):
        value = os.environ.get(name)
        if value:
            return value
    root = os.environ.get("ZLC_PS_PROJECT_DIR")
    if not root:
        return None
    candidate = Path(root) / "ps.runs" / "impl_1" / "zlc_pulse_streamer_top.ltx"
    return str(candidate) if candidate.exists() else None


def _byte_address(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_WORD_ADDRESS
    ):
        raise ValueError("AXI word address is outside the 32-bit byte-address range")
    return value * 4


def _word_value(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("AXI word value is outside the unsigned 32-bit range")
    return value


class VivadoAxiRegisterTransport:
    """Ordered 32-bit register/BRAM access through one persistent Vivado Tcl owner."""

    transport_id = "vivado-axi"
    #: Nothing on this transport is ever lost -- it may be slow, and a
    #: timed-out action may still complete later, which is exactly why the
    #: strobe verify-and-retry machinery must NOT run here.
    lossy_line = False
    observer_interval = JTAG_AXI_OBSERVER_INTERVAL

    def __init__(
        self,
        *,
        state_dir: str | Path,
        vivado: str | None = None,
        probes: str | None = None,
        hw_server_url: str | None = None,
        startup_timeout: float = 180.0,
        action_timeout: float = 120.0,
        write_batch: int = 200,
        burst_max: int = 256,
        tcl_executor: Callable[[Sequence[str], str, float | None], str] | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.vivado = vivado or _default_vivado()
        self.probes = probes if probes is not None else _default_probes()
        self.hw_server_url = (
            hw_server_url
            or os.environ.get("ZLC_PS_HW_SERVER_URL")
            or os.environ.get("ZLC_HW_SERVER_URL")
            or ""
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in (startup_timeout, action_timeout)
        ):
            raise ValueError("AXI transport timeouts must be finite and positive")
        self.startup_timeout = float(startup_timeout)
        self.action_timeout = float(action_timeout)
        if isinstance(write_batch, bool) or not isinstance(write_batch, int) or write_batch <= 0:
            raise ValueError("write_batch must be a positive integer")
        if isinstance(burst_max, bool) or not isinstance(burst_max, int) or not 1 <= burst_max <= 256:
            raise ValueError("burst_max must be in the hardware range [1, 256]")
        self.write_batch = write_batch
        self.burst_max = burst_max
        self._external_executor = tcl_executor
        self._io_lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._counter = 0
        self._closed = True
        self._log_path = self.state_dir / "vivado_axi_transport.log"

    def start(self) -> None:
        with self._io_lock:
            if not self._closed:
                return
            self._closed = False
            if self._external_executor is not None:
                return
            try:
                process = subprocess.Popen(
                    [self.vivado, "-mode", "tcl", "-nolog", "-nojournal"],
                    cwd=self.state_dir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as error:
                self._closed = True
                self._record_diagnostic(
                    "start_failure",
                    f"Vivado executable was not found: {self.vivado!r}",
                )
                raise RuntimeError("persistent Vivado transport could not start") from error
            self._process = process
            generation_queue: queue.Queue[str | None] = queue.Queue()
            self._queue = generation_queue
            self._reader = threading.Thread(
                target=self._read_stdout,
                args=(process, generation_queue),
                name="zlc-vivado-axi-reader",
                daemon=True,
            )
            self._reader.start()
            try:
                self._run_tcl(
                    self._init_tcl(),
                    action="transport_start",
                    deadline=time.monotonic() + self.startup_timeout,
                )
            except BaseException:
                self._closed = True
                self._stop_process(graceful=False)
                raise

    def close(self) -> None:
        self._closed = True
        with self._io_lock:
            self._stop_process(graceful=True)

    def write_words(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
        #: Whether a frame the link never answered may be sent again.  False
        #: for a command strobe: one that WAS executed and whose acknowledgement
        #: was lost would be executed twice.  A transport with no frames to lose
        #: takes the argument and ignores it, because the caller's meaning is
        #: the same either way and a caller should not have to ask which
        #: transport it has.
        resend: bool = True,
    ) -> None:
        absolute_deadline = self._effective_deadline(deadline)
        pending = tuple(
            (_byte_address(word_offset), _word_value(value))
            for word_offset, value in rows
        )
        lines: list[str] = []
        bursts = 0
        for base, values in self._burst_runs(pending):
            lines.extend(self._write_burst_tcl(base, values))
            bursts += 1
            if bursts >= self.write_batch:
                self._run_tcl(
                    lines,
                    action="axi_write",
                    deadline=absolute_deadline,
                    stop=stop,
                )
                lines = []
                bursts = 0
        if lines:
            self._run_tcl(
                lines,
                action="axi_write",
                deadline=absolute_deadline,
                stop=stop,
            )

    def read_word(
        self,
        word_offset: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> int:
        absolute_deadline = self._effective_deadline(deadline)
        marker = "ZLCDATA"
        output = self._run_tcl(
            self._read_txn_tcl(_byte_address(word_offset), marker),
            action="axi_read",
            deadline=absolute_deadline,
            stop=stop,
        )
        return self._parse_read(output, marker)

    def _record_diagnostic(self, name: str, text: str) -> None:
        try:
            (self.state_dir / f"{name}.log").write_text(
                text,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            pass

    def _effective_deadline(self, value: float | None) -> float:
        deadline = (
            time.monotonic() + self.action_timeout
            if value is None
            else float(value)
        )
        if not math.isfinite(deadline):
            raise ValueError("AXI transaction deadline must be finite")
        if deadline <= time.monotonic():
            raise TimeoutError("AXI transaction deadline expired before issue")
        return deadline

    def _init_tcl(self) -> list[str]:
        lines = [
            "if {[llength [info commands load_features]]} { catch {load_features labtools} }",
            "if {[llength [info commands open_hw_manager]]} { open_hw_manager } elseif {[llength [info commands open_hw]]} { open_hw }",
        ]
        if self.hw_server_url:
            lines.append(f"connect_hw_server -url {self.hw_server_url}")
        else:
            lines.append("if {[catch {connect_hw_server}]} { connect_hw_server }")
        lines.extend(
            [
                "catch {refresh_hw_server}",
                "set zlc_targets [get_hw_targets]",
                'if {[llength $zlc_targets] != 1} { error "Expected exactly one Vivado hardware target" }',
                "current_hw_target $zlc_targets",
                "if {[catch {open_hw_target}]} { catch {close_hw_target}; after 2000; open_hw_target }",
                "set zlc_devices [get_hw_devices]",
                'if {[llength $zlc_devices] != 1} { error "Expected exactly one Vivado hardware device" }',
                "current_hw_device $zlc_devices",
            ]
        )
        if self.probes:
            lines.append(
                f"set_property PROBES.FILE {{{self.probes}}} [current_hw_device]"
            )
            lines.append(
                f"set_property FULL_PROBES.FILE {{{self.probes}}} [current_hw_device]"
            )
        # Frozen deployment: this owner never selects or programs a bitstream.
        lines.extend(
            [
                "refresh_hw_device [current_hw_device]",
                "set zlc_axi [get_hw_axis]",
                'if {[llength $zlc_axi] != 1} { error "Expected exactly one JTAG-to-AXI core" }',
            ]
        )
        return lines

    def _burst_runs(
        self,
        pending: Sequence[tuple[int, int]],
    ) -> list[tuple[int, list[int]]]:
        """Coalesce only insertion-order contiguous writes; never reorder commands."""

        runs: list[tuple[int, list[int]]] = []
        index = 0
        while index < len(pending):
            base, value = pending[index]
            values = [value]
            cursor = index + 1
            while (
                cursor < len(pending)
                and len(values) < self.burst_max
                and pending[cursor][0] == base + 4 * len(values)
                and (base + 4 * len(values)) // AXI_BURST_BOUNDARY_BYTES
                == base // AXI_BURST_BOUNDARY_BYTES
            ):
                values.append(pending[cursor][1])
                cursor += 1
            runs.append((base, values))
            index = cursor
        return runs

    @staticmethod
    def _write_burst_tcl(byte_address: int, values: Sequence[int]) -> list[str]:
        address = f"{byte_address:08X}"
        data = "".join(
            f"{value:08X}" for value in reversed(values)
        )
        return [
            f"create_hw_axi_txn zlc_w [get_hw_axis] -address {address} -data {data} -len {len(values)} -type write -burst INCR -force",
            "run_hw_axi zlc_w",
            "delete_hw_axi_txn zlc_w",
        ]

    @staticmethod
    def _read_txn_tcl(byte_address: int, marker: str) -> list[str]:
        address = f"{byte_address:08X}"
        return [
            f"create_hw_axi_txn zlc_r [get_hw_axis] -address {address} -len 1 -type read -force",
            "run_hw_axi zlc_r",
            f'puts "{marker} [get_property DATA [get_hw_axi_txns zlc_r]]"',
            "delete_hw_axi_txn zlc_r",
        ]

    @staticmethod
    def _parse_read(output: str, marker: str) -> int:
        token = ""
        for line in output.splitlines():
            if marker in line:
                fields = line.split(marker, 1)[1].strip().split()
                token = fields[-1] if fields else ""
        if not token:
            raise RuntimeError("hw_axi read returned no DATA")
        if re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{1,8}", token) is None:
            raise RuntimeError(f"hw_axi read returned non-binary DATA {token!r}")
        return int(token, 16) & 0xFFFFFFFF

    def _run_tcl(
        self,
        lines: Sequence[str],
        *,
        action: str,
        deadline: float,
        stop: threading.Event | None = None,
    ) -> str:
        remaining = self._remaining(deadline, action)
        if not self._io_lock.acquire(timeout=remaining):
            raise TimeoutError(f"{action} timed out waiting for the AXI I/O owner")
        try:
            if self._closed:
                raise RuntimeError(
                    "Vivado AXI transport is closed; call start() before another transaction"
                )
            if stop is not None and stop.is_set():
                raise TransportAborted(f"{action} aborted before issue")
            if self._external_executor is not None:
                try:
                    return self._external_executor(
                        list(lines),
                        action,
                        self._remaining(deadline, action),
                    )
                except (TimeoutError, TransportAborted):
                    self._closed = True
                    raise
            return self._execute(
                lines,
                action=action,
                deadline=deadline,
                stop=stop,
            )
        finally:
            self._io_lock.release()

    def _execute(
        self,
        lines: Sequence[str],
        *,
        action: str,
        deadline: float,
        stop: threading.Event | None,
    ) -> str:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError(
                "persistent Vivado transport is not running; call start() explicitly"
            )
        self._remaining(deadline, action)
        self._counter += 1
        marker = f"ZLC_AXI_{self._counter:06d}"
        try:
            process.stdin.write(self._wrap_tcl(lines, marker))
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise RuntimeError(f"Vivado stopped before {action}") from error
        output = self._read_until_marker(marker, deadline=deadline, stop=stop)
        self._record_diagnostic(action, output)
        if f"{marker}_ERROR" in output:
            detail = next(
                (
                    line.strip()
                    for line in output.splitlines()
                    if line.strip().startswith("ERROR:")
                ),
                "Vivado Tcl action returned an error",
            )
            raise RuntimeError(f"persistent Vivado {action} failed: {detail}")
        return output

    def _stop_process(self, *, graceful: bool) -> None:
        process, self._process = self._process, None
        reader, self._reader = self._reader, None
        if process is not None:
            if graceful:
                try:
                    if process.stdin is not None:
                        process.stdin.write(
                            "catch {close_hw_target}\n"
                            "catch {disconnect_hw_server}\n"
                            "exit\n"
                        )
                        process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            else:
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                process.wait(timeout=5 if graceful else 3)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait(timeout=3)
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=5)
            if reader.is_alive():
                raise RuntimeError("Vivado AXI reader did not stop")
        self._queue = queue.Queue()

    @staticmethod
    def _wrap_tcl(lines: Sequence[str], marker: str) -> str:
        body = "\n".join(lines)
        return (
            f'puts "{marker}_BEGIN"\n'
            # A failed run_hw_axi skips the transaction helper's normal delete.
            # Clear only our private fixed names before and after every action so
            # one Xicom fault cannot poison the later physical-SAFE attempt.
            "catch {delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_w]}\n"
            "catch {delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_r]}\n"
            "set zlc_axi_code [catch {\n"
            f"{body}\n"
            "} zlc_axi_result zlc_axi_options]\n"
            "catch {delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_w]}\n"
            "catch {delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_r]}\n"
            "if {$zlc_axi_code} {\n"
            f'    puts "{marker}_ERROR $zlc_axi_result"\n'
            "    if {[dict exists $zlc_axi_options -errorinfo]} { puts [dict get $zlc_axi_options -errorinfo] }\n"
            "} else {\n"
            f'    puts "{marker}_OK"\n'
            "}\n"
            f'puts "{marker}_END"\n'
            "flush stdout\n"
        )

    def _read_until_marker(
        self,
        marker: str,
        *,
        deadline: float,
        stop: threading.Event | None,
    ) -> str:
        lines: list[str] = []
        while True:
            if self._closed:
                self._stop_process(graceful=False)
                raise TransportAborted(f"transport closed waiting for {marker}")
            if stop is not None and stop.is_set():
                self._closed = True
                self._stop_process(graceful=False)
                raise TransportAborted(f"read aborted waiting for {marker}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._closed = True
                self._stop_process(graceful=False)
                raise TimeoutError(f"Vivado action timed out waiting for {marker}")
            slice_seconds = min(0.2, remaining)
            try:
                item = self._queue.get(timeout=slice_seconds)
            except queue.Empty:
                continue
            if item is None:
                raise RuntimeError("persistent Vivado process exited unexpectedly")
            lines.append(item)
            if f"{marker}_END" in item:
                return "".join(lines)

    @staticmethod
    def _remaining(deadline: float, action: str) -> float:
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{action} exceeded its absolute deadline")
        return remaining

    def _read_stdout(
        self,
        process: subprocess.Popen,
        generation_queue: queue.Queue[str | None],
    ) -> None:
        stdout = process.stdout
        if stdout is None:
            generation_queue.put(None)
            return
        with self._log_path.open("a", encoding="utf-8", errors="replace") as log:
            for line in stdout:
                log.write(line)
                log.flush()
                generation_queue.put(line)
        generation_queue.put(None)


__all__ = ["AXI_BURST_BOUNDARY_BYTES", "VivadoAxiRegisterTransport"]
