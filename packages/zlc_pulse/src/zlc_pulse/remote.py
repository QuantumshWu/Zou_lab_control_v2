"""Small single-connection RPC for a separated FPGA control machine.

The server owns one real :class:`PulseStreamer`; the wire layer only carries
plain JSON trees and never grows a second device state machine.  Every request
is one length-prefixed frame and every response is one frame.  In particular,
``wait_done`` is a non-blocking server probe; the client implements the public
wait by polling short ``snapshot``/``cursor``/``wait_done`` calls so ``safe``
can use the same connection at any time.
"""

from __future__ import annotations

from .endpoint import DEFAULT_BIND_HOST, DEFAULT_HOST, DEFAULT_PORT

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
import argparse
import json
import math
import socket
import socketserver
import struct
import sys
import threading
import time
from typing import Any

from .compile import CompiledProgram, TargetBusDelay as _TargetBusDelay, TargetBusSegment as _TargetBusSegment
from .device import (
    AppliedState,
    BoardDescription,
    DoneReport,
    PulseStreamer,
    SafeReadback,
)
from .manifest import DEFAULT_XDC_PATH, pulse_target_from_xdc
from .model import (
    AnalogStep,
    OutputDelay,
    PulseFieldRef as _PulseFieldRef,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseSlot,
    PulseTarget,
    RepeatRegion,
)
from .wire import StreamerParams, load_streamer_config


MAX_FRAME_BYTES = 8 * 1024 * 1024
_FRAME_HEADER = struct.Struct("!I")
UART_PROBE_TIMEOUT = 0.5
DEFAULT_CLIENT_IDLE_TIMEOUT = 300.0
BUSY_REQUEST_TIMEOUT = 1.0
BACKEND_CHOICES = ("auto", "uart", "jtag-axi", "memory")

REMOTE_METHODS = (
    "open",
    "describe",
    "close",
    "load",
    "write_slots",
    "write_scan_table",
    "fire",
    "wait_done",
    "cursor",
    "safe",
    "snapshot",
    "applied",
)

_TREE_TYPES = {
    cls.__name__: cls
    for cls in (
        AnalogStep,
        AppliedState,
        BoardDescription,
        CompiledProgram,
        DoneReport,
        OutputDelay,
        _PulseFieldRef,
        PulsePeriod,
        PulsePortSpec,
        PulseSequence,
        PulseSlot,
        PulseTarget,
        RepeatRegion,
        SafeReadback,
        _TargetBusDelay,
        _TargetBusSegment,
    )
}


def _log_fields(**values: object) -> str:
    """Render compact, human-readable key/value fields for server events."""

    rendered: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        text = str(value)
        if any(character.isspace() for character in text) or '"' in text:
            text = json.dumps(text, ensure_ascii=False)
        rendered.append(f"{key}={text}")
    return " ".join(rendered)


def _server_log(event: str, *, client: str | None = None, detail: str = "") -> None:
    """Print one timestamped lifecycle event without dumping payload contents."""

    fields = _log_fields(client=client)
    if detail:
        fields = f"{fields} {detail}" if fields else detail
    suffix = f" {fields}" if fields else ""
    now = time.time()
    local = time.localtime(now)
    timestamp = (
        time.strftime("%Y-%m-%dT%H:%M:%S", local)
        + f".{int(now * 1000) % 1000:03d}"
        + time.strftime("%z", local)
    )
    print(f"[{timestamp}] ZLC {event}{suffix}", flush=True)


def _program_summary(program: object, *, source: object = None) -> str:
    """Describe a loaded program without printing its full edge or DAC data."""

    if not isinstance(program, CompiledProgram):
        return _log_fields(program_type=type(program).__name__)
    return _log_fields(
        edges=len(program.ticks),
        lanes=len(program.channels),
        dac_segments=len(program.bus_segments),
        slots=program.slot_count,
        duration_us=f"{program.duration_seconds * 1e6:.3f}",
        repeat_forever=program.repeat_forever,
        source="provided" if source is not None else "none",
    )


def _compact_tuple(values: object) -> str:
    """Keep readback tuples readable without turning them into quoted fields."""

    return "(" + ",".join(str(value) for value in values) + ")"


def _client_addresses(bind_host: str) -> tuple[str, ...]:
    """Return addresses a client may use for the server's bind host."""

    host = str(bind_host).strip()
    if host and host not in {"0.0.0.0", "::"}:
        return ("127.0.0.1",) if host.lower() == "localhost" else (host,)
    return _local_ipv4_addresses()


def _local_ipv4_addresses() -> tuple[str, ...]:
    """Discover non-loopback IPv4 addresses without requiring a network request."""

    addresses: list[str] = []
    packed_addresses: set[bytes] = set()

    def add(value: object) -> None:
        try:
            address = str(value).strip()
            packed = socket.inet_aton(address)
        except (OSError, ValueError):
            return
        if address == "0.0.0.0" or address.startswith("127.") or packed in packed_addresses:
            return
        packed_addresses.add(packed)
        addresses.append(address)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            add(sock.getsockname()[0])
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            add(info[4][0])
    except OSError:
        pass

    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            add(address)
    except OSError:
        pass

    return tuple(addresses)


def _print_client_endpoints(bind_host: str, port: int) -> None:
    """Print copyable client endpoints for both local and separated-machine use."""

    host = str(bind_host).strip() or "0.0.0.0"
    normalized = host.lower()
    wildcard = normalized in {"0.0.0.0", "::"}
    if wildcard:
        same_computer_host = "127.0.0.1"
        other_computer_hosts = _client_addresses(host)
    elif normalized in {"localhost", "127.0.0.1", "::1"}:
        same_computer_host = "127.0.0.1"
        other_computer_hosts = ()
    else:
        same_computer_host = host
        other_computer_hosts = (host,)

    port = int(port)
    _server_log(
        "CLIENT ENDPOINTS",
        detail=_log_fields(
            listen_bind=f"{host}:{port}",
            same_computer=f"{same_computer_host}:{port}",
        ),
    )
    _server_log(
        "SERVER ADDRESS",
        detail=_log_fields(scope="same_computer", address=f"{same_computer_host}:{port}"),
    )
    if wildcard:
        _server_log(
            "CLIENT HOST NOTE",
            detail="0.0.0.0 is listen-only; use 127.0.0.1 locally or a listed LAN address remotely",
        )
    _server_log(
        "CLIENT CONNECT EXAMPLE",
        detail=(
            f'same_computer=RemotePulseStreamer("{same_computer_host}", {port}, '
            "request_timeout=30.0, poll_interval=0.01)"
        ),
    )
    if other_computer_hosts:
        for address in other_computer_hosts:
            _server_log("SERVER ADDRESS", detail=_log_fields(scope="other_computer", address=f"{address}:{port}"))
            _server_log("CLIENT ENDPOINT", detail=_log_fields(other_computer=f"{address}:{port}"))
        _server_log(
            "CLIENT CONNECT EXAMPLE",
            detail=(
                f'other_computer=RemotePulseStreamer("{other_computer_hosts[0]}", {port}, '
                "request_timeout=30.0, poll_interval=0.01)"
            ),
        )
    else:
        if wildcard:
            _server_log(
                "SERVER ADDRESS",
                detail="scope=other_computer address=NOT_DISCOVERED (run ipconfig and use this machine's LAN IPv4)",
            )
        _server_log("CLIENT ENDPOINT", detail=_log_fields(other_computer="NOT_AVAILABLE"))


class RemoteError(RuntimeError):
    """An exception raised by the server while handling one request."""

    def __init__(self, remote_type: str, message: str) -> None:
        super().__init__(f"{remote_type}: {message}")
        self.remote_type = remote_type
        self.message = message


@dataclass(frozen=True)
class BackendResolution:
    """The one startup decision that selects the physical transport."""

    requested: str
    backend: str
    uart_port: str | None
    reason: str
    attempts: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()


class BackendResolutionError(RuntimeError):
    """An explicitly requested backend could not be established."""

    def __init__(self, message: str, *, attempts: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


def _list_uart_ports() -> tuple[object, ...]:
    """Enumerate serial device names lazily so importing zlc_pulse stays UART-optional."""
    from serial.tools import list_ports
    return tuple(list_ports.comports())


def _uart_candidates(
    uart_port: str | None,
    port_provider: Callable[[], Iterable[object]] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return candidate ports plus an enumeration failure, if one occurred."""
    selected = str(uart_port).strip() if uart_port is not None else ""
    if selected:
        return (selected,), ()
    provider = port_provider or _list_uart_ports
    try:
        raw_ports = provider()
    except ModuleNotFoundError as exc:
        name = str(getattr(exc, "name", "") or "")
        if name.startswith("serial") or "serial" in str(exc).lower():
            return (), (f'pyserial missing; install with: "{sys.executable}" -m pip install pyserial',)
        return (), (f"port enumeration failed: {type(exc).__name__}",)
    except Exception as exc:
        return (), (f"port enumeration failed: {type(exc).__name__}: {exc}",)
    if isinstance(raw_ports, str):
        raw_ports = (raw_ports,)
    raw_ports = sorted(raw_ports, key=lambda item: getattr(item, "vid", None) is None or getattr(item, "pid", None) is None)
    ports: list[str] = []
    for descriptor in raw_ports:
        port = str(getattr(descriptor, "device", descriptor)).strip()
        if port and port not in ports:
            ports.append(port)
    return tuple(ports), (() if ports else ("no UART ports detected",))


def _probe_uart_port(
    port: str,
    timeout: float,
    *,
    params: StreamerParams,
    clock_hz: float,
    state_dir: str,
    baud: int,
) -> None:
    """Probe one UART by reusing ``PulseStreamer.open``'s word63 handshake."""
    from .transport import UartRegisterTransport
    transport = UartRegisterTransport(
        state_dir=state_dir,
        port=port,
        baud=baud,
        action_timeout=timeout,
    )
    streamer = PulseStreamer(transport, params, clock_hz)
    try:
        streamer.open()
    finally:
        # Close the probe transport directly; do not issue a second hardware transaction.
        transport.close()


def _probe_failure_reason(exc: BaseException) -> str:
    """Collapse probe exceptions into the operator-facing failure categories."""

    message = " ".join(str(exc).split())
    lower = message.lower()
    pyserial_hint = f'"{sys.executable}" -m pip install pyserial'
    if isinstance(exc, ModuleNotFoundError) and (
        str(getattr(exc, "name", "") or "").startswith("serial") or "serial" in lower
    ):
        return f"pyserial missing; install with: {pyserial_hint}"
    if "geometry/layout mismatch" in lower or "fingerprint" in lower:
        return "fingerprint mismatch"
    if type(exc).__name__ == "FrameError" or "crc" in lower:
        return "CRC error"
    if isinstance(exc, TimeoutError) or "timeout" in lower or "timed out" in lower:
        return "timeout"
    if type(exc).__name__ in {"SerialException", "FileNotFoundError", "PermissionError"}:
        return "open failed"
    if "could not open port" in lower or "access is denied" in lower:
        return "open failed"
    return f"{type(exc).__name__}: {message or 'probe failed'}"


def resolve_backend(
    requested: str = "auto",
    *,
    uart_port: str | None = None,
    uart_baud: int = 3_000_000,
    state_dir: str = "fpga/build/state",
    params: StreamerParams,
    clock_hz: float,
    port_provider: Callable[[], Iterable[object]] | None = None,
    probe: Callable[[str, float], None] | None = None,
    on_candidates: Callable[[tuple[str, ...], tuple[str, ...]], None] | None = None,
) -> BackendResolution:
    """Resolve one physical backend, probing UART only when requested by policy."""

    choice = str(requested).strip().lower()
    if choice not in BACKEND_CHOICES:
        raise ValueError(f"unsupported backend: {requested}")
    if choice == "memory":
        return BackendResolution(choice, "memory", None, "explicit memory backend")
    if choice == "jtag-axi":
        return BackendResolution(choice, "jtag-axi", None, "explicit jtag-axi backend; UART probe skipped")

    ports, setup_attempts = _uart_candidates(uart_port, port_provider)
    if on_candidates is not None:
        on_candidates(ports, setup_attempts)
    attempts = list(setup_attempts)
    probe_fn = probe
    if probe_fn is None:
        probe_fn = lambda port, timeout: _probe_uart_port(
            port,
            timeout,
            params=params,
            clock_hz=clock_hz,
            state_dir=state_dir,
            baud=int(uart_baud),
        )
    for port in ports:
        try:
            probe_fn(port, UART_PROBE_TIMEOUT)
        except Exception as exc:
            attempts.append(f"{port}: {_probe_failure_reason(exc)}")
            continue
        attempts.append(f"{port}: word63 fingerprint matched")
        if choice == "uart":
            reason = f"explicit uart selected {port}@{int(uart_baud) / 1_000_000:g}M after word63 fingerprint match"
        else:
            reason = f"auto selected UART {port}@{int(uart_baud) / 1_000_000:g}M after word63 fingerprint match"
        return BackendResolution(choice, "uart", port, reason, tuple(attempts), ports)

    failure = "; ".join(attempts) if attempts else "no UART ports detected"
    if choice == "uart":
        raise BackendResolutionError(
            f"explicit UART backend failed; no matching device: {failure}",
            attempts=attempts,
        )
    return BackendResolution(
        choice,
        "jtag-axi",
        None,
        f"auto fallback to jtag-axi after UART probe: {failure}",
        tuple(attempts),
        ports,
    )


def encode_tree(value: Any) -> Any:
    """Encode public model dataclasses as an ordinary JSON tree."""

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("remote trees cannot contain non-finite floats")
        return value
    if isinstance(value, (tuple, list)):
        return [encode_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): encode_tree(item) for key, item in value.items()}
    if is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            **{
                field.name: encode_tree(getattr(value, field.name))
                for field in fields(value)
                if not field.name.startswith("_")
            },
            **{
                name: encode_tree(getattr(value, name))
                for name in _TREE_KEPT.get(type(value).__name__, ())
            },
        }
    raise TypeError(f"unsupported remote tree value: {type(value).__name__}")


#: Public state a model keeps behind a private field.
#:
#: The generic encoder skips underscored fields because they are derived or
#: internal.  A target's package pins are neither: they are half of what a
#: target IS, and dropping them handed a remote client a target whose pin map
#: was empty while the description beside it still carried one -- two copies of
#: the pin map, and the wire silently emptied the one nested deeper.
_TREE_KEPT = {"PulseTarget": ("package_pins",)}


def decode_tree(value: Any) -> Any:
    """Decode a JSON tree into the small set of public model dataclasses."""

    if isinstance(value, list):
        return [decode_tree(item) for item in value]
    if not isinstance(value, dict):
        return value
    type_name = value.get("__type__")
    if type_name is not None:
        if not isinstance(type_name, str) or type_name not in _TREE_TYPES:
            raise ValueError(f"unknown remote tree type: {type_name!r}")
        cls = _TREE_TYPES[type_name]
        kwargs = {
            key: decode_tree(item)
            for key, item in value.items()
            if key != "__type__"
        }
        return cls(**kwargs)
    return {str(key): decode_tree(item) for key, item in value.items()}


def _recv_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            if not chunks:
                return None
            raise ConnectionError("remote connection closed mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_frame(connection: socket.socket, value: Any) -> None:
    payload = json.dumps(encode_tree(value), separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("remote frame exceeds the maximum size")
    connection.sendall(_FRAME_HEADER.pack(len(payload)) + payload)


def _recv_frame(connection: socket.socket) -> Any | None:
    header = _recv_exact(connection, _FRAME_HEADER.size)
    if header is None:
        return None
    (size,) = _FRAME_HEADER.unpack(header)
    if size > MAX_FRAME_BYTES:
        raise ValueError("remote frame exceeds the maximum size")
    payload = _recv_exact(connection, size)
    if payload is None:
        raise ConnectionError("remote connection closed before frame payload")
    return decode_tree(json.loads(payload.decode("utf-8")))


class _RemoteHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, PulseRemoteServer)
        client = f"{self.client_address[0]}:{self.client_address[1]}"
        # Keep the human-facing "client CONNECTED" wording recognizable in the server log.
        if not server.claim_client(client):
            _server_log("CLIENT CONNECTED", client=client, detail="status=BUSY")
            self._reject_busy(server, client)
            return
        _server_log("CLIENT CONNECTED", client=client, detail="status=OWNER")
        disconnect_reason = "client disconnect"
        try:
            self.request.settimeout(server.client_idle_timeout)
            while True:
                try:
                    request = _recv_frame(self.request)
                except socket.timeout:
                    if server.client_idle_expired(client):
                        disconnect_reason = "idle timeout; SAFE released"
                        _server_log("CLIENT IDLE TIMEOUT", client=client, detail=_log_fields(timeout_s=server.client_idle_timeout))
                        return
                    continue
                if request is None:
                    return
                server.touch_client(client)
                request_id = request.get("id") if isinstance(request, dict) else None
                method = "<invalid>"
                try:
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    method = request.get("method")
                    params = request.get("params", {})
                    if not isinstance(method, str):
                        raise ValueError("request method must be text")
                    if not isinstance(params, Mapping):
                        raise ValueError("request params must be an object")
                    result = server.dispatch(method, params, client=client)
                    response = {"id": request_id, "ok": True, "result": result}
                except Exception as exc:
                    _server_log(
                        "RPC ERROR",
                        client=client,
                        detail=_log_fields(
                            method=method,
                            error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}",
                        ),
                    )
                    response = {
                        "id": request_id,
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                _send_frame(self.request, response)
        except ConnectionError as exc:
            # Disconnect/RST ends this client session; it is not a server defect.
            disconnect_reason = f"client connection dropped: {type(exc).__name__}"
        finally:
            outputs_safe = server.client_disconnected(client=client, reason=disconnect_reason)
            _server_log(
                "CLIENT DISCONNECTED",
                client=client,
                detail=_log_fields(outputs="SAFE" if outputs_safe else "NOT_VERIFIED", reason=disconnect_reason),
            )

    def _reject_busy(self, server: "PulseRemoteServer", client: str) -> None:
        """Reply to the first request so the client sees a useful busy error."""

        try:
            self.request.settimeout(BUSY_REQUEST_TIMEOUT)
            request = _recv_frame(self.request)
            request_id = request.get("id") if isinstance(request, dict) else None
            owner, held = server.owner_status()
            message = (
                f"server is busy; current owner={owner or 'unknown'}, "
                f"held_for={held:.1f}s; wait for that client to disconnect and retry"
            )
            _send_frame(
                self.request,
                {
                    "id": request_id,
                    "ok": False,
                    "error": {"type": "RemoteBusyError", "message": message},
                },
            )
            _server_log("CLIENT BUSY REJECTED", client=client, detail=_log_fields(owner=owner, held_s=f"{held:.1f}"))
        except (OSError, ConnectionError, socket.timeout, ValueError) as exc:
            _server_log(
                "CLIENT BUSY REJECTED",
                client=client,
                detail=_log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"),
            )


class PulseRemoteServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """A serialized, one-client-at-a-time RPC façade over one PulseStreamer."""

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False
    request_queue_size = 8

    def __init__(
        self,
        address: tuple[str, int],
        streamer: PulseStreamer,
        *,
        client_idle_timeout: float = DEFAULT_CLIENT_IDLE_TIMEOUT,
    ) -> None:
        if not isinstance(streamer, PulseStreamer):
            raise TypeError("streamer must be a PulseStreamer")
        if isinstance(client_idle_timeout, bool) or not math.isfinite(float(client_idle_timeout)) or client_idle_timeout <= 0:
            raise ValueError("client_idle_timeout must be finite and positive")
        self.streamer = streamer
        self.client_idle_timeout = float(client_idle_timeout)
        self._client_lock = threading.RLock()
        self._owner_client: str | None = None
        self._owner_started = 0.0
        self._owner_last_activity = 0.0
        super().__init__(address, _RemoteHandler)

    def handle_error(self, request, client_address) -> None:
        """Prefix real handler defects with timestamp/client, then keep the traceback."""
        exc = sys.exc_info()[1]
        client = f"{client_address[0]}:{client_address[1]}" if client_address else None
        detail = _log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}")
        _server_log("HANDLER ERROR", client=client, detail=detail)
        super().handle_error(request, client_address)

    def claim_client(self, client: str) -> bool:
        with self._client_lock:
            if self._owner_client is not None:
                return False
            now = time.monotonic()
            self._owner_client = client
            self._owner_started = now
            self._owner_last_activity = now
            return True

    def touch_client(self, client: str) -> None:
        with self._client_lock:
            if self._owner_client == client:
                self._owner_last_activity = time.monotonic()

    def client_idle_expired(self, client: str) -> bool:
        with self._client_lock:
            return (
                self._owner_client == client
                and time.monotonic() - self._owner_last_activity >= self.client_idle_timeout
            )

    def owner_status(self) -> tuple[str | None, float]:
        with self._client_lock:
            if self._owner_client is None:
                return None, 0.0
            return self._owner_client, max(0.0, time.monotonic() - self._owner_started)

    def _release_client(self, client: str | None, *, force: bool = False) -> None:
        with self._client_lock:
            if force or client is None or self._owner_client == client:
                self._owner_client = None
                self._owner_started = 0.0
                self._owner_last_activity = 0.0

    def dispatch(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        client: str = "server",
    ) -> Any:
        if method not in REMOTE_METHODS:
            raise ValueError(f"unknown remote method: {method}")
        if method == "open":
            self.streamer.open()
            _server_log("OPEN", client=client, detail=_log_fields(device_session="ready"))
            return None
        if method == "describe":
            result = self.streamer.describe()
            _server_log(
                "DESCRIBE",
                client=client,
                detail=_log_fields(
                    ports=len(result.target.ports),
                    lanes=len(result.target.raw_lanes),
                    pins=len(result.target.package_pins),
                    clock_hz=result.clock_hz,
                    layout=f"0x{result.layout_fingerprint:08X}",
                ),
            )
            return result
        if method == "close":
            before = self.streamer.snapshot()
            self.streamer.close()
            _server_log(
                "CLOSE",
                client=client,
                detail=_log_fields(
                    firing=before.get("firing"),
                    device_session="closed",
                ),
            )
            return None
        if method == "load":
            program = params.get("program")
            source = params.get("source")
            self.streamer.load(program, source=source)
            _server_log("LOAD", client=client, detail=_program_summary(program, source=source))
            return None
        if method == "write_slots":
            values = tuple(int(value) for value in params.get("values", ()))
            self.streamer.write_slots(values)
            _server_log("WRITE SLOTS", client=client, detail=_log_fields(slots=len(values)))
            return None
        if method == "write_scan_table":
            rows = tuple(tuple(int(value) for value in row) for row in params.get("rows", ()))
            self.streamer.write_scan_table(rows)
            _server_log("WRITE SCAN TABLE", client=client, detail=_log_fields(rows=len(rows)))
            return None
        if method == "fire":
            forever = bool(params.get("forever", False))
            self.streamer.fire(forever=forever)
            applied = self.streamer.applied()
            program = applied.program if applied is not None else None
            fire_fields = _log_fields(
                mode="FOREVER" if forever else "ONCE",
                reloaded_before_fire=self.streamer.snapshot().get("reloaded_before_fire"),
            )
            _server_log(
                "FIRE",
                client=client,
                detail=f"{fire_fields} "
                f"{_program_summary(program, source=applied.source if applied is not None else None)}",
            )
            return None
        if method == "wait_done":
            # Never honor a network timeout here: the client owns the short poll loop.
            result = self.streamer.wait_done(0.0)
            if result is not None:
                _server_log(
                    "DONE",
                    client=client,
                    detail=_log_fields(
                        status=f"0x{result.status:02X}",
                        cursor=result.cursor,
                        underflow=result.underflow,
                        tail_ms=f"{result.tail_elapsed * 1e3:.3f}",
                    ),
                )
            else:
                _server_log("WAIT DONE", client=client, detail="state=PENDING")
            return result
        if method == "cursor":
            result = self.streamer.cursor()
            _server_log("CURSOR", client=client, detail=_log_fields(value=result))
            return result
        if method == "safe":
            result = self.streamer.safe()
            _server_log(
                "STOP/SAFE",
                client=client,
                detail=_log_fields(
                    stable=result.stable,
                    status_reads=_compact_tuple(result.status_reads),
                    clock_enable_words=_compact_tuple(result.clock_enable_words),
                ),
            )
            return result
        if method == "snapshot":
            result = self.streamer.snapshot()
            _server_log("SNAPSHOT", client=client, detail=_log_fields(**{key: result.get(key) for key in ("opened", "loaded", "firing", "forever", "cursor")}))
            return result
        result = self.streamer.applied()
        _server_log("APPLIED", client=client, detail=_log_fields(present=result is not None, slots=len(result.slot_values) if result is not None else None, scan_rows=len(result.scan_rows) if result is not None else None, forever=result.forever if result is not None else None))
        return result

    def client_disconnected(self, *, client: str | None = None, reason: str = "client disconnect") -> bool:
        """Release physical outputs without clearing the server-process echo; outputs are SAFE on success."""

        owned = client in {None, "server"} or self.owner_status()[0] == client
        if not owned:
            return True
        try:
            if not self.streamer.snapshot().get("opened"):
                self._release_client(client, force=client == "server")
                return True
            _server_log("AUTO-SAFE", client=client, detail=_log_fields(reason=reason))
            result = self.streamer.safe()
            _server_log(
                "AUTO-SAFE DONE",
                client=client,
                detail=_log_fields(
                    stable=result.stable,
                    status_reads=_compact_tuple(result.status_reads),
                    clock_enable_words=_compact_tuple(result.clock_enable_words),
                ),
            )
            self._release_client(client, force=client == "server")
            return result.stable
        except Exception as exc:
            _server_log(
                "AUTO-SAFE FAILED",
                client=client,
                detail=_log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"),
            )
            self._release_client(client, force=client == "server")
            return False


class RemotePulseStreamer:
    """The local ``PulseStreamer`` method surface backed by one TCP connection."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        request_timeout: float = 5.0,
        poll_interval: float = 0.01,
    ) -> None:
        if not host:
            raise ValueError("remote host is required")
        if int(port) <= 0 or int(port) > 65535:
            raise ValueError("remote port is outside the TCP range")
        if request_timeout <= 0 or not math.isfinite(float(request_timeout)):
            raise ValueError("request_timeout must be finite and positive")
        if poll_interval <= 0 or not math.isfinite(float(poll_interval)):
            raise ValueError("poll_interval must be finite and positive")
        self.host = str(host)
        self.port = int(port)
        self.request_timeout = float(request_timeout)
        self.poll_interval = float(poll_interval)
        self._socket: socket.socket | None = None
        self._request_id = 0
        self._io_lock = threading.RLock()

    def open(self) -> None:
        with self._io_lock:
            try:
                self._connect_locked()
                self._call_locked("open", {})
            except Exception:
                self._disconnect_locked()
                raise

    def describe(self):
        """Ask the board what it is, rather than assuming from local files.

        This is the call that lets an editor show the ports, pins and clock of
        the board it is actually attached to.  Without it a client had to read
        its own XDC and config and hope they matched -- the same mistake the
        layout handshake exists to catch, one layer up.
        """

        return self._call("describe", {})

    def close(self) -> None:
        with self._io_lock:
            if self._socket is None:
                return
            try:
                self._call_locked("close", {})
            finally:
                self._disconnect_locked()

    def disconnect(self) -> None:
        """Drop the TCP connection without invoking the device ``close`` method."""

        with self._io_lock:
            self._disconnect_locked()

    def load(self, prog: CompiledProgram, *, source: PulseSequence | None = None) -> None:
        self._call("load", {"program": prog, "source": source})

    def write_slots(self, values) -> None:
        self._call("write_slots", {"values": tuple(int(value) for value in values)})

    def write_scan_table(self, rows) -> None:
        self._call(
            "write_scan_table",
            {"rows": tuple(tuple(int(value) for value in row) for row in rows)},
        )

    def fire(self, *, forever: bool = False) -> None:
        self._call("fire", {"forever": bool(forever)})

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        """Poll the board until the shot reports done, or the deadline passes.

        One request per poll.  It used to send three and throw two away, so a
        wait ran at 300 round trips a second instead of 100 -- and every one of
        them made the server print a log line, on the machine whose job is to
        keep a scan bank ahead of the engine.
        """

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            result = self._call("wait_done", {"timeout": 0.0})
            if result is not None:
                return result
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                time.sleep(min(self.poll_interval, remaining))
            else:
                time.sleep(self.poll_interval)

    def cursor(self) -> int | None:
        return self._call("cursor", {})

    def safe(self) -> SafeReadback:
        return self._call("safe", {})

    def snapshot(self) -> dict[str, object]:
        return self._call("snapshot", {})

    def applied(self) -> AppliedState | None:
        return self._call("applied", {})

    def _call(self, method: str, params: Mapping[str, Any]) -> Any:
        with self._io_lock:
            return self._call_locked(method, params)

    def _connect_locked(self) -> None:
        if self._socket is not None:
            return
        self._socket = socket.create_connection((self.host, self.port), timeout=self.request_timeout)
        self._socket.settimeout(self.request_timeout)

    def _disconnect_locked(self) -> None:
        connection, self._socket = self._socket, None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def _call_locked(self, method: str, params: Mapping[str, Any]) -> Any:
        if self._socket is None:
            raise RuntimeError("remote PulseStreamer is not open")
        self._request_id += 1
        request = {
            "id": self._request_id,
            "method": method,
            "params": encode_tree(dict(params)),
        }
        try:
            _send_frame(self._socket, request)
            response = _recv_frame(self._socket)
        except (OSError, ConnectionError, TimeoutError):
            self._disconnect_locked()
            raise
        if not isinstance(response, Mapping):
            raise ConnectionError("remote response is not an object")
        if response.get("id") != self._request_id:
            raise ConnectionError("remote response id differs from the request")
        if not response.get("ok"):
            error = response.get("error")
            if isinstance(error, Mapping):
                raise RemoteError(str(error.get("type", "RemoteError")), str(error.get("message", "")))
            raise RemoteError("RemoteError", "remote request failed")
        return decode_tree(response.get("result"))

    def __enter__(self) -> "RemotePulseStreamer":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            with self._io_lock:
                self._disconnect_locked()
        except Exception:
            pass


def serve(
    streamer: PulseStreamer,
    host: str = DEFAULT_BIND_HOST,
    port: int = DEFAULT_PORT,
    *,
    client_idle_timeout: float = DEFAULT_CLIENT_IDLE_TIMEOUT,
) -> None:
    """Serve one supplied device until interrupted."""

    with PulseRemoteServer((host, int(port)), streamer, client_idle_timeout=client_idle_timeout) as server:
        listen_host = host or "0.0.0.0"
        actual_port = int(server.server_address[1])
        if listen_host == "0.0.0.0":
            listen_host = "0.0.0.0 (all interfaces)"
        _server_log("RPC LISTENING", detail=_log_fields(endpoint=f"{listen_host}:{actual_port}"))
        _server_log("READY", detail=_log_fields(hardware="connected", waiting_for_client=True))
        _print_client_endpoints(host, actual_port)
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            _server_log("SERVER STOPPING", detail=_log_fields(reason="keyboard interrupt"))
            pass
        finally:
            server.client_disconnected(client="server", reason="server shutdown")


def connect(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, **kwargs: Any) -> RemotePulseStreamer:
    """Construct a remote streamer for the server endpoint shown at startup."""

    return RemotePulseStreamer(host, port, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the server CLI parser so its default backend is directly testable."""

    parser = argparse.ArgumentParser(description="Serve one zlc_pulse PulseStreamer over a thin TCP RPC.")
    parser.add_argument("--host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="auto",
        help="transport policy: probe UART first, or explicitly select a backend",
    )
    parser.add_argument("--state-dir", default="fpga/build/state")
    parser.add_argument("--uart-port", default=None, help="probe only this UART port")
    parser.add_argument("--uart-baud", type=int, default=3_000_000)
    parser.add_argument(
        "--client-idle-timeout",
        type=float,
        default=DEFAULT_CLIENT_IDLE_TIMEOUT,
        help="seconds without an RPC request before the server AUTO-SAFEs and releases the client",
    )
    parser.add_argument("--check-config", action="store_true")
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_streamer_config()
    target = pulse_target_from_xdc(config_path=config["source"])
    if args.check_config:
        print(f"python={sys.executable}")
        print(f"backend={args.backend}")
        print(f"uart_port={args.uart_port or 'auto-discover'}")
        print(f"listen_bind={args.host}:{args.port}")
        print(f"client_idle_timeout_s={args.client_idle_timeout:g}")
        normalized_host = str(args.host).strip().lower()
        same_host = (
            "127.0.0.1"
            if normalized_host in {"", "0.0.0.0", "::", "localhost", "127.0.0.1", "::1"}
            else str(args.host).strip()
        )
        print(f"same_computer_client={same_host}:{args.port}")
        print(f"config={config['source']}")
        print(f"clock_hz={config['clock_hz']}")
        print(f"geometry={config['params']}")
        print(f"xdc={DEFAULT_XDC_PATH}")
        print(f"target_ports={len(target.ports)}")
        print("remote RPC: length-prefixed JSON, one client, non-blocking wait_done polls")
        return 0
    _server_log(
        "SERVER STARTING",
        detail=_log_fields(
            requested_backend=args.backend,
            endpoint=f"{args.host}:{args.port}",
            python=sys.executable,
        ),
    )

    def report_candidates(candidates: tuple[str, ...], setup_attempts: tuple[str, ...]) -> None:
        _server_log(
            "UART PROBE",
            detail=_log_fields(
                ports=",".join(candidates) if candidates else "none",
                setup_reason="; ".join(setup_attempts) or None,
            ),
        )

    try:
        resolution = resolve_backend(
            args.backend,
            uart_port=args.uart_port,
            uart_baud=args.uart_baud,
            state_dir=args.state_dir,
            params=config["params"],
            clock_hz=config["clock_hz"],
            on_candidates=report_candidates,
        )
    except BackendResolutionError as exc:
        _server_log(
            "BACKEND FAILED",
            detail=_log_fields(
                requested=args.backend,
                error=str(exc),
                attempts="; ".join(exc.attempts),
            ),
        )
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2

    _server_log(
        "BACKEND RESOLVED",
        detail=_log_fields(
            requested=resolution.requested,
            selected=resolution.backend,
            uart_port=resolution.uart_port,
            reason=resolution.reason,
            attempts="; ".join(resolution.attempts) or None,
        ),
    )
    if resolution.backend == "jtag-axi":
        _server_log(
            "BACKEND NOTE",
            detail="resident vivado.exe ~1-2 GB; prefer UART on memory-constrained hosts",
        )

    streamer: PulseStreamer | None = None
    try:
        from .transport import (
            MemoryRegisterTransport,
            UartRegisterTransport,
            VivadoAxiRegisterTransport,
        )

        if resolution.backend == "memory":
            transport = MemoryRegisterTransport(geom=config["params"])
        elif resolution.backend == "uart":
            if not resolution.uart_port:  # pragma: no cover - resolver guarantees this.
                raise RuntimeError("UART resolution did not return a port")
            transport = UartRegisterTransport(
                state_dir=args.state_dir,
                port=resolution.uart_port,
                baud=args.uart_baud,
            )
        else:
            transport = VivadoAxiRegisterTransport(state_dir=args.state_dir)
        streamer = PulseStreamer(transport, config["params"], config["clock_hz"])
        backend_label = {
            "jtag-axi": "JTAG-to-AXI",
            "uart": "UART",
            "memory": "memory mock",
        }[resolution.backend]
        _server_log(
            "CONFIG",
            detail=_log_fields(
                backend=f"{backend_label} ({resolution.backend})",
                geometry=config["source"],
                channels=config["params"].channel_count,
                dac_buses=config["params"].bus_count,
                target_ports=len(target.ports),
                clock_hz=f"{config['clock_hz']:.0f}",
            ),
        )
        _server_log("HARDWARE CONNECTING", detail=_log_fields(action="open_deployed_streamer"))
        streamer.open()
        initial_safe = streamer.safe()
        if not initial_safe.stable:
            raise RuntimeError(
                "initial SAFE readback was not stable: "
                f"status_reads={_compact_tuple(initial_safe.status_reads)}"
            )
        _server_log(
            "HARDWARE CONNECTED",
            # Keep the human-facing "hardware CONNECTED" wording aligned with run_server.bat.
            detail=_log_fields(
                geometry_handshake=True,
                safe_readback=initial_safe.stable,
                status_reads=_compact_tuple(initial_safe.status_reads),
                clock_enable_words=_compact_tuple(initial_safe.clock_enable_words),
            ),
        )
        serve(streamer, args.host, args.port, client_idle_timeout=args.client_idle_timeout)
    except KeyboardInterrupt:
        _server_log("SERVER STOPPING", detail=_log_fields(reason="keyboard interrupt"))
        return 0
    except Exception as exc:
        _server_log(
            "SERVER FAILED",
            detail=_log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"),
        )
        print("ZLC server did not enter LISTENING state; no client can connect.", file=sys.stderr, flush=True)
        return 3
    finally:
        if streamer is not None:
            streamer.close()
            _server_log("SERVER CLOSED", detail=_log_fields(device_session="closed"))
    return 0


__all__ = [
    "BACKEND_CHOICES",
    "BackendResolution",
    "BackendResolutionError",
    "DEFAULT_CLIENT_IDLE_TIMEOUT",
    "MAX_FRAME_BYTES",
    "REMOTE_METHODS",
    "PulseRemoteServer",
    "RemoteError",
    "RemotePulseStreamer",
    "UART_PROBE_TIMEOUT",
    "build_arg_parser",
    "connect",
    "decode_tree",
    "encode_tree",
    "resolve_backend",
    "serve",
]


if __name__ == "__main__":
    raise SystemExit(_main())
