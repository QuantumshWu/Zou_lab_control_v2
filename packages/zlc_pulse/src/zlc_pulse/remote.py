"""Thin JSON RPC between one PulseStreamer server and its current client."""

from __future__ import annotations

from .endpoint import (
    DEFAULT_BIND_HOST,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT,
)

from collections.abc import Iterable, Mapping
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
from typing import Any, Callable

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
    PulseApiParameter,
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
#: What a client is told when its connection ends under it.  The board goes
#: to whoever connected last, so this is the ordinary way an editor finds out
#: that a newer one took over -- and it must not read like a protocol defect.
_CONNECTION_ENDED = (
    "the connection to the pulse server ended; if another editor has connected "
    "since, that one now holds the board"
)
BACKEND_CHOICES = ("auto", "uart", "jtag-axi", "memory")

REMOTE_METHODS = (
    "open",
    "describe",
    "close",
    "load",
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
        PulseApiParameter,
        _PulseFieldRef,
        PulsePeriod,
        PulsePortSpec,
        PulseSequence,
        PulseSlot,
        PulseTarget,
        RepeatRegion,
        SafeReadback,
        StreamerParams,
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


#: The last line each client got for each POLLED event.  A poll is a question,
#: not an event: wait_done is asked every 10 ms by a client that owns its own
#: poll loop, so one five-second shot printed four hundred identical lines and
#: buried the run around them.  A log narrates what HAPPENED.
_LAST_POLL: dict[tuple[str, str], str] = {}
_POLL_LOCK = threading.Lock()


def _server_log_change(event: str, *, client: str = "", detail: str = "") -> None:
    """Log a polled answer only when it differs from the last one.

    Which is also what makes the interesting ones visible: a cursor that stays
    at 3 says nothing, and a cursor that becomes 4 is the scan advancing.
    """

    key = (str(client), str(event))
    with _POLL_LOCK:
        if _LAST_POLL.get(key) == detail:
            return
        _LAST_POLL[key] = detail
    _server_log(event, client=client, detail=detail)


def _forget_polls(client: str) -> None:
    """Drop one client's poll memory when it goes away."""

    with _POLL_LOCK:
        for key in [key for key in _LAST_POLL if key[0] == str(client)]:
            del _LAST_POLL[key]


def _drop_connection(connection: socket.socket) -> None:
    """End a connection nobody wants any more, from this side.

    Shutting down our own half unblocks whatever handler is waiting on it, and
    it works on a peer that is no longer there at all -- the kernel does not
    have to reach anybody to stop listening.
    """

    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


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
        # Said here because this is the machine that can act on it, and said
        # with the interpreter's own path because a Windows firewall rule is
        # per-program: a rule made for a previous Python does not cover this
        # one, which is why the same LAN that worked before starts timing out
        # after the server moves interpreters.
        _server_log(
            "FIREWALL NOTE",
            detail=_log_fields(
                allow_inbound_program=sys.executable,
                port=port,
                note="another machine timing out is almost always blocked HERE, on the private network",
            ),
        )
    _server_log(
        "CLIENT CONNECT EXAMPLE",
        detail=(
            f'same_computer=RemotePulseStreamer("{same_computer_host}", {port}, '
            f"request_timeout={DEFAULT_REQUEST_TIMEOUT:g}, poll_interval=0.01)"
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
                f"request_timeout={DEFAULT_REQUEST_TIMEOUT:g}, poll_interval=0.01)"
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


class BackendResolutionError(RuntimeError):
    """An explicitly requested backend could not be established."""

    def __init__(self, message: str, *, attempts: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


def _list_uart_ports() -> tuple[object, ...]:
    """Enumerate serial descriptors lazily so importing zlc_pulse stays optional."""

    from serial.tools import list_ports

    return tuple(list_ports.comports())


def _uart_candidates(
    uart_port: str | None,
    port_provider: Callable[[], Iterable[object]] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return explicit port or every enumerated COM, with USB descriptors first."""

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
    raw_ports = sorted(
        raw_ports,
        key=lambda item: (
            getattr(item, "vid", None) is None
            or getattr(item, "pid", None) is None
        ),
    )
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
    target: PulseTarget,
    params: StreamerParams,
    clock_hz: float,
    baud: int,
) -> None:
    """Probe one UART by reusing ``PulseStreamer.open``'s word63 handshake."""
    from .transport import UartRegisterTransport
    transport = UartRegisterTransport(
        port=port,
        baud=baud,
        action_timeout=timeout,
    )
    streamer = PulseStreamer(transport, params, clock_hz, target=target)
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
    target: PulseTarget,
    params: StreamerParams,
    clock_hz: float,
    port_provider: Callable[[], Iterable[object]] | None = None,
) -> BackendResolution:
    """Probe enumerated UARTs by word 63, then fall back to JTAG."""

    choice = str(requested).strip().lower()
    if choice not in BACKEND_CHOICES:
        raise ValueError(f"unsupported backend: {requested}")
    if choice == "memory":
        return BackendResolution(choice, "memory", None, "explicit memory backend")
    if choice == "jtag-axi":
        return BackendResolution(choice, "jtag-axi", None, "explicit jtag-axi backend; UART probe skipped")

    if uart_port is not None and not isinstance(uart_port, str):
        raise TypeError("uart_port must be text or None")
    ports, setup_attempts = _uart_candidates(uart_port, port_provider)
    attempts = list(setup_attempts)
    for port in ports:
        try:
            _probe_uart_port(
                port,
                UART_PROBE_TIMEOUT,
                target=target,
                params=params,
                clock_hz=clock_hz,
                baud=int(uart_baud),
            )
        except Exception as exc:
            attempts.append(f"{port}: {_probe_failure_reason(exc)}")
            continue
        attempts.append(f"{port}: word63 fingerprint matched")
        mode = "explicit uart" if choice == "uart" else "auto"
        return BackendResolution(
            choice,
            "uart",
            port,
            f"{mode} selected UART {port}@{int(uart_baud) / 1_000_000:g}M after word63 fingerprint match",
            tuple(attempts),
        )

    failure = "; ".join(attempts) if attempts else "no board UART detected"
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
        if any(not isinstance(key, str) for key in value):
            raise TypeError("remote tree mapping keys must be strings")
        return {key: encode_tree(item) for key, item in value.items()}
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

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("remote trees cannot contain non-finite floats")
    if isinstance(value, list):
        return [decode_tree(item) for item in value]
    if not isinstance(value, dict):
        return value
    if any(not isinstance(key, str) for key in value):
        raise TypeError("remote tree mapping keys must be strings")
    if "__type__" in value:
        type_name = value["__type__"]
        if not isinstance(type_name, str) or type_name not in _TREE_TYPES:
            raise ValueError(f"unknown remote tree type: {type_name!r}")
        cls = _TREE_TYPES[type_name]
        expected = {
            field.name
            for field in fields(cls)
            if not field.name.startswith("_")
        }
        expected.update(_TREE_KEPT.get(type_name, ()))
        actual = set(value).difference(("__type__",))
        unknown = sorted(actual.difference(expected))
        missing = sorted(expected.difference(actual))
        if unknown:
            raise ValueError(
                f"unknown {type_name} remote field(s): {', '.join(unknown)}"
            )
        if missing:
            raise ValueError(
                f"missing {type_name} remote field(s): {', '.join(missing)}"
            )
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

    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in remote JSON")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value!r} in remote JSON")

    # THE TREE, NOT THE OBJECTS IT DESCRIBES.  Decoding domain values
    # here made a payload the server could not interpret indistinguishable
    # from a broken socket: decode_tree raises ValueError, only OSError is
    # caught around the read loop, so the error escaped the handler
    # entirely.  socketserver then tore the connection down, the finally
    # clause reported "client disconnect", and a running board was
    # AUTO-SAFEd -- while the operator was told another editor had taken
    # it.  Version skew between a rig-machine server and a newer editor is
    # all it takes.
    #
    # Reading a frame and understanding what is in it are two jobs.  This
    # one ends at the JSON.
    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )


class _RemoteHandler(socketserver.BaseRequestHandler):
    def setup(self) -> None:
        super().setup()
        server = self.server
        assert isinstance(server, PulseRemoteServer)
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        with server._client_lock:
            server._connections.add(self.request)

    def finish(self) -> None:
        server = self.server
        assert isinstance(server, PulseRemoteServer)
        with server._client_lock:
            server._connections.discard(self.request)
        super().finish()

    def handle(self) -> None:
        server = self.server
        assert isinstance(server, PulseRemoteServer)
        client = f"{self.client_address[0]}:{self.client_address[1]}"
        disconnect_reason = "client disconnect"
        claimed = False
        try:
            while True:
                request = _recv_frame(self.request)
                if request is None:
                    return
                request_id = request.get("id") if isinstance(request, dict) else None
                method = "<invalid>"
                try:
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    unknown = tuple(
                        key for key in request if key not in {"id", "method", "params"}
                    )
                    missing = tuple(
                        key for key in ("id", "method", "params") if key not in request
                    )
                    if unknown:
                        raise ValueError(
                            f"unknown request field(s): {', '.join(unknown)}"
                        )
                    if missing:
                        raise ValueError(
                            f"missing request field(s): {', '.join(missing)}"
                        )
                    request_id = request["id"]
                    if (
                        isinstance(request_id, bool)
                        or not isinstance(request_id, int)
                        or request_id <= 0
                    ):
                        raise TypeError("request id must be a positive integer")
                    method = request["method"]
                    params = request["params"]
                    if not isinstance(method, str):
                        raise ValueError("request method must be text")
                    if method not in REMOTE_METHODS:
                        raise ValueError(f"unknown remote method: {method}")
                    if not isinstance(params, Mapping):
                        raise ValueError("request params must be an object")
                    # Interpreted HERE, inside the try that turns a bad
                    # request into an error response, so a payload this
                    # server cannot read costs the client its answer and
                    # nothing else.
                    params = decode_tree(params)
                    if not claimed:
                        server.claim_client(client, self.request)
                        claimed = True
                        _server_log(
                            "CLIENT CONNECTED", client=client, detail="status=OWNER"
                        )
                    result = server.dispatch(
                        method,
                        params,
                        client=client,
                        connection=self.request,
                    )
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
                if not claimed:
                    return
        except OSError as exc:
            # Closed, reset, or dropped by a newer client taking the board.
            # Each of those ends this session; none is a server defect.
            disconnect_reason = f"client connection dropped: {type(exc).__name__}"
        finally:
            outputs_safe = (
                server.client_disconnected(
                    client=client,
                    connection=self.request,
                    reason=disconnect_reason,
                )
                if claimed
                else None
            )
            _forget_polls(client)
            status = (
                "SAFE"
                if outputs_safe is True
                else "NOT_VERIFIED"
                if outputs_safe is False
                else "NO_ACTION"
            )
            _server_log(
                "CLIENT DISCONNECTED",
                client=client,
                detail=_log_fields(outputs=status, reason=disconnect_reason),
            )


class PulseRemoteServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """One board owner; a newcomer takes over only after stable physical SAFE."""

    allow_reuse_address = True
    daemon_threads = False
    block_on_close = True
    request_queue_size = 8

    def __init__(
        self,
        address: tuple[str, int],
        streamer: PulseStreamer,
    ) -> None:
        if not isinstance(streamer, PulseStreamer):
            raise TypeError("streamer must be a PulseStreamer")
        self.streamer = streamer
        self._client_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._owner_epoch = 0
        self._owner_client: str | None = None
        self._owner_connection: socket.socket | None = None
        self._owner_started = 0.0
        self._fault: str | None = None
        self._connections: set[socket.socket] = set()
        super().__init__(address, _RemoteHandler)

    def server_close(self) -> None:
        self.client_disconnected(client="server", reason="server shutdown")
        with self._client_lock:
            connections = tuple(self._connections)
        for connection in connections:
            _drop_connection(connection)
        super().server_close()

    def handle_error(self, request, client_address) -> None:
        """Prefix real handler defects with timestamp/client, then keep the traceback."""
        exc = sys.exc_info()[1]
        client = f"{client_address[0]}:{client_address[1]}" if client_address else None
        detail = _log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}")
        _server_log("HANDLER ERROR", client=client, detail=detail)
        super().handle_error(request, client_address)

    def claim_client(self, client: str, connection: socket.socket) -> None:
        """Transfer ownership only after the previous physical state is SAFE."""

        with self._client_lock:
            previous, connection_to_drop = self._owner_client, self._owner_connection
            if previous is None and self._fault is None:
                self._owner_epoch += 1
                self._owner_client = client
                self._owner_connection = connection
                self._owner_started = time.monotonic()
                return
            self._owner_client = None
            self._owner_connection = None
            self._owner_started = 0.0
            self._owner_epoch += 1
            transition_epoch = self._owner_epoch
            self._fault = "takeover SAFE has not completed"
        if previous is not None:
            _server_log(
                "CLIENT REPLACED",
                client=previous,
                detail=_log_fields(by=client),
            )
        if connection_to_drop is not None:
            _drop_connection(connection_to_drop)

        # This first SAFE is the only operation allowed beside the active
        # command lane: its stop event interrupts a pending transport action.
        try:
            result = self.streamer.safe()
            if not result.stable:
                raise RuntimeError("SAFE readback was not stable")
        except Exception:
            pass

        failure_message: str | None = None
        with self._command_lock:
            safe_failure: Exception | None = None
            try:
                result = self.streamer.safe()
                if not result.stable:
                    raise RuntimeError("SAFE readback was not stable")
            except Exception as exc:
                try:
                    opened = bool(self.streamer.snapshot().get("opened"))
                except Exception:
                    opened = True
                if opened:
                    safe_failure = exc
            with self._client_lock:
                if self._owner_epoch != transition_epoch:
                    raise RuntimeError("client takeover was superseded")
                if safe_failure is None:
                    self._fault = None
                    self._owner_client = client
                    self._owner_connection = connection
                    self._owner_started = time.monotonic()
                else:
                    failure_message = (
                        "takeover SAFE failed: "
                        f"{type(safe_failure).__name__}: "
                        f"{str(safe_failure).replace(chr(10), ' ')}"
                    )
                    self._fault = failure_message
        if failure_message is not None:
            _server_log(
                "TAKEOVER FAILED",
                client=client,
                detail=_log_fields(error=failure_message),
            )
            raise RuntimeError(failure_message) from safe_failure

    def owner_status(self) -> tuple[str | None, float]:
        with self._client_lock:
            if self._owner_client is None:
                return None, 0.0
            return self._owner_client, max(0.0, time.monotonic() - self._owner_started)

    def _link_health(self) -> str:
        """Frames that had to be sent again, when there have been any.

        A line that loses the occasional frame now recovers instead of failing
        a load, which is right -- and silent, which is not: a cable or an
        adapter that is degrading should be visible before it stops working.
        Nothing is printed while the count is zero.
        """

        transport = getattr(self.streamer, "transport", None)
        resends = int(getattr(transport, "resends", 0) or 0)
        return f" resent_frames={resends}" if resends else ""

    def dispatch(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        client: str,
        connection: socket.socket,
    ) -> Any:
        with self._client_lock:
            if (
                self._fault is not None
                or self._owner_client != client
                or self._owner_connection is not connection
            ):
                raise RuntimeError("this connection no longer owns the pulse server")
            owner_epoch = self._owner_epoch

        with self._command_lock:
            with self._client_lock:
                if (
                    self._fault is not None
                    or self._owner_epoch != owner_epoch
                    or self._owner_client != client
                    or self._owner_connection is not connection
                ):
                    raise RuntimeError("this connection no longer owns the pulse server")
            try:
                no_params = {
                    "open",
                    "describe",
                    "close",
                    "wait_done",
                    "cursor",
                    "safe",
                    "snapshot",
                    "applied",
                }
                expected = (
                    {"program", "source", "rows"}
                    if method == "load"
                    else {"cycles"}
                    if method == "fire"
                    else set()
                )
                if method not in REMOTE_METHODS:
                    raise ValueError(f"unknown remote method: {method}")
                if method in no_params and params:
                    raise ValueError(f"{method} does not accept parameters")
                if method not in no_params and set(params) != expected:
                    raise ValueError(
                        f"{method} parameters must be exactly {', '.join(sorted(expected))}"
                    )
                if method == "open":
                    self.streamer.open()
                    _server_log("OPEN", client=client, detail=_log_fields(device_session="ready"))
                    result = None
                elif method == "describe":
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
                elif method == "close":
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
                    result = None
                elif method == "load":
                    program = params["program"]
                    source = params["source"]
                    rows = params["rows"]
                    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, list):
                        raise TypeError("load rows must be a JSON array")
                    self.streamer.load(program, source=source, rows=rows)
                    _server_log(
                        "LOAD",
                        client=client,
                        detail=(
                            f"{_program_summary(program, source=source)} "
                            f"rows={len(rows)}{self._link_health()}"
                        ),
                    )
                    result = None
                elif method == "fire":
                    cycles = params["cycles"]
                    if cycles is not None and (
                        isinstance(cycles, bool) or not isinstance(cycles, int)
                    ):
                        raise TypeError("fire cycles must be an integer or null")
                    self.streamer.fire(cycles=cycles)
                    applied = self.streamer.applied()
                    program = applied.program if applied is not None else None
                    fire_fields = _log_fields(
                        cycles="FOREVER" if cycles is None else cycles,
                        reloaded_before_fire=self.streamer.snapshot().get("reloaded_before_fire"),
                    )
                    _server_log(
                        "FIRE",
                        client=client,
                        detail=f"{fire_fields} "
                        f"{_program_summary(program, source=applied.source if applied is not None else None)}",
                    )
                    result = None
                elif method == "wait_done":
                    # Never honor a network timeout here: the client owns the short poll loop.
                    result = self.streamer.wait_done(0.0)
                    if result is not None:
                        _server_log(
                            "ERROR" if result.fault else "DONE",
                            client=client,
                            detail=_log_fields(
                                status=f"0x{result.status:02X}",
                                cursor=result.cursor,
                                underflow=result.underflow,
                                link_error=result.link_error,
                                elapsed_ms=f"{result.elapsed_seconds * 1e3:.3f}",
                                status_reads=result.status_reads,
                                cursor_reads=result.cursor_reads,
                                observer_error=result.observer_error or None,
                            ),
                        )
                    else:
                        _server_log_change("WAIT DONE", client=client, detail="state=PENDING")
                elif method == "cursor":
                    result = self.streamer.cursor()
                    _server_log_change("CURSOR", client=client, detail=_log_fields(value=result))
                elif method == "safe":
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
                elif method == "snapshot":
                    result = self.streamer.snapshot()
                    _server_log_change("SNAPSHOT", client=client, detail=_log_fields(**{key: result.get(key) for key in ("opened", "loaded", "firing", "cycles", "cursor")}))
                else:
                    result = self.streamer.applied()
                    _server_log("APPLIED", client=client, detail=_log_fields(present=result is not None, rows=len(result.rows) if result is not None else None, cycles=result.cycles if result is not None else None))
            except BaseException as exc:
                with self._client_lock:
                    stale = (
                        self._fault is not None
                        or self._owner_epoch != owner_epoch
                        or self._owner_client != client
                        or self._owner_connection is not connection
                    )
                if stale:
                    raise RuntimeError(
                        "this connection no longer owns the pulse server"
                    ) from exc
                raise
            with self._client_lock:
                if (
                    self._fault is not None
                    or self._owner_epoch != owner_epoch
                    or self._owner_client != client
                    or self._owner_connection is not connection
                ):
                    raise RuntimeError("this connection no longer owns the pulse server")
            return result

    def client_disconnected(
        self,
        *,
        client: str | None = None,
        connection: socket.socket | None = None,
        reason: str = "client disconnect",
    ) -> bool | None:
        """Release physical outputs without clearing the server-process echo; outputs are SAFE on success."""

        with self._client_lock:
            force = client in {None, "server"}
            if not force and (
                self._owner_client != client
                or self._owner_connection is not connection
            ):
                return None
            connection_to_drop = self._owner_connection
            self._owner_client = None
            self._owner_connection = None
            self._owner_started = 0.0
            self._owner_epoch += 1
            transition_epoch = self._owner_epoch
            self._fault = "disconnect SAFE has not completed"
        if connection_to_drop is not None:
            _drop_connection(connection_to_drop)
        _server_log("AUTO-SAFE", client=client, detail=_log_fields(reason=reason))

        # Cancel first; the command lane remains owned by the old handler until
        # it observes the revocation and retires.
        try:
            result = self.streamer.safe()
            if not result.stable:
                raise RuntimeError("SAFE readback was not stable")
        except Exception:
            pass

        failure_message: str | None = None
        with self._command_lock:
            result = None
            safe_failure: Exception | None = None
            try:
                result = self.streamer.safe()
                if not result.stable:
                    raise RuntimeError("SAFE readback was not stable")
            except Exception as exc:
                try:
                    opened = bool(self.streamer.snapshot().get("opened"))
                except Exception:
                    opened = True
                if opened:
                    safe_failure = exc
            with self._client_lock:
                if self._owner_epoch != transition_epoch:
                    return None
                if safe_failure is None:
                    self._fault = None
                else:
                    failure_message = (
                        "disconnect SAFE failed: "
                        f"{type(safe_failure).__name__}: "
                        f"{str(safe_failure).replace(chr(10), ' ')}"
                    )
                    self._fault = failure_message
        if failure_message is not None:
            _server_log(
                "AUTO-SAFE FAILED",
                client=client,
                detail=_log_fields(error=failure_message),
            )
            return False
        if result is not None:
            _server_log(
                "AUTO-SAFE DONE",
                client=client,
                detail=_log_fields(
                    stable=result.stable,
                    status_reads=_compact_tuple(result.status_reads),
                    clock_enable_words=_compact_tuple(result.clock_enable_words),
                ),
            )
        return True


class RemotePulseStreamer:
    """The local ``PulseStreamer`` method surface backed by one TCP connection."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        poll_interval: float = 0.01,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("remote host is required")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port <= 65535:
            raise ValueError("remote port is outside the TCP range")
        if request_timeout <= 0 or not math.isfinite(float(request_timeout)):
            raise ValueError("request_timeout must be finite and positive")
        if connect_timeout <= 0 or not math.isfinite(float(connect_timeout)):
            raise ValueError("connect_timeout must be finite and positive")
        if poll_interval <= 0 or not math.isfinite(float(poll_interval)):
            raise ValueError("poll_interval must be finite and positive")
        self.host = host
        self.port = port
        self.request_timeout = float(request_timeout)
        self.connect_timeout = float(connect_timeout)
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

    def load(
        self,
        prog: CompiledProgram,
        *,
        source: PulseSequence | None = None,
        rows=(),
    ) -> None:
        self._call(
            "load",
            {
                "program": prog,
                "source": source,
                "rows": tuple(tuple(row) for row in rows),
            },
        )

    def fire(self, *, cycles: int | None = 1) -> None:
        self._call("fire", {"cycles": cycles})

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        """Poll the board until the shot reports done, or the deadline passes.

        One request per poll.  It used to send three and throw two away, so a
        wait ran at 300 round trips a second instead of 100 -- and every one of
        them made the server print a log line, on the machine whose job is to
        keep a scan bank ahead of the engine.
        """

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            result = self._call("wait_done", {})
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
        try:
            self._socket = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout
            )
        except OSError as exc:
            # Reaching the server and being answered by it are different
            # failures and an operator acts on them differently, but both
            # used to arrive as a bare "timed out" -- a description of the
            # clock rather than of what is wrong.
            if isinstance(exc, ConnectionRefusedError):
                reason = "the machine is up but nothing is listening on that port"
            else:
                reason = (
                    "the connection itself was never answered, which is what a firewall "
                    "does to an inbound port -- on the server machine, allow its python.exe "
                    "on the private network"
                )
            raise ConnectionError(
                f"could not reach a pulse server at {self.host}:{self.port} "
                f"within {self.connect_timeout:g}s: {reason}"
            ) from exc
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
            "params": dict(params),
        }
        try:
            _send_frame(self._socket, request)
            response = _recv_frame(self._socket)
        except TimeoutError as exc:
            self._disconnect_locked()
            raise ConnectionError(
                f"the pulse server did not answer within {self.request_timeout:g}s"
            ) from exc
        except OSError as exc:
            self._disconnect_locked()
            raise ConnectionError(f"{_CONNECTION_ENDED} ({type(exc).__name__})") from exc
        if response is None:
            # End of stream: the same thing, arriving as silence rather than as
            # an error.  It used to be reported as "the response is not an
            # object", which describes the shape of nothing instead of saying
            # what happened.
            self._disconnect_locked()
            raise ConnectionError(_CONNECTION_ENDED)
        try:
            if not isinstance(response, Mapping):
                raise ConnectionError("remote response is not an object")
            if response.get("id") != self._request_id:
                raise ConnectionError("remote response id differs from the request")
            ok = response.get("ok")
            if not isinstance(ok, bool):
                raise ConnectionError("remote response ok field is not boolean")
            expected = {"id", "ok", "result"} if ok else {"id", "ok", "error"}
            if set(response) != expected:
                raise ConnectionError("remote response fields differ from the protocol")
            if ok:
                # The frame reader stops at the JSON, so the domain values
                # a result carries are interpreted here -- on the side
                # that asked for them, and inside the guard that turns a
                # malformed answer into a connection error.
                return decode_tree(response["result"])
            error = response["error"]
            if (
                not isinstance(error, Mapping)
                or set(error) != {"type", "message"}
                or not isinstance(error["type"], str)
                or not isinstance(error["message"], str)
            ):
                raise ConnectionError("remote error response differs from the protocol")
        except ConnectionError:
            self._disconnect_locked()
            raise
        raise RemoteError(error["type"], error["message"])

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
) -> None:
    """Serve one supplied device until interrupted."""

    with PulseRemoteServer((host, int(port)), streamer) as server:
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
        help="transport policy; auto probes UART word 63, then falls back to JTAG",
    )
    parser.add_argument("--state-dir", default="fpga/build/state")
    parser.add_argument("--uart-port", default=None, help="the one configured Pulse UART port")
    parser.add_argument("--uart-baud", type=int, default=3_000_000)
    parser.add_argument("--check-config", action="store_true")
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        config = load_streamer_config()
    except Exception as exc:
        print(f"ERROR: invalid streamer_config.json: {exc}", file=sys.stderr, flush=True)
        return 2
    if config["source"] is None or config["warnings"]:
        detail = "; ".join(config["warnings"]) or "canonical config source is missing"
        print(
            f"ERROR: deployment requires a valid streamer_config.json without fallback: {detail}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    target = pulse_target_from_xdc(config_path=config["source"])
    if args.check_config:
        print(f"python={sys.executable}")
        print(f"backend={args.backend}")
        print(f"uart_port={args.uart_port or 'auto-discover'}")
        print(f"listen_bind={args.host}:{args.port}")
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

    try:
        resolution = resolve_backend(
            args.backend,
            uart_port=args.uart_port,
            uart_baud=args.uart_baud,
            target=target,
            params=config["params"],
            clock_hz=config["clock_hz"],
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
            transport = MemoryRegisterTransport(
                geom=config["params"],
                record_history=False,
            )
        elif resolution.backend == "uart":
            if not resolution.uart_port:  # pragma: no cover - resolver guarantees this.
                raise RuntimeError("UART resolution did not return a port")
            transport = UartRegisterTransport(
                port=resolution.uart_port,
                baud=args.uart_baud,
            )
        else:
            transport = VivadoAxiRegisterTransport(state_dir=args.state_dir)
        streamer = PulseStreamer(
            transport,
            config["params"],
            config["clock_hz"],
            target=target,
        )
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
        serve(streamer, args.host, args.port)
    except KeyboardInterrupt:
        _server_log("SERVER STOPPING", detail=_log_fields(reason="keyboard interrupt"))
        return 0
    except Exception as exc:
        _server_log(
            "SERVER FAILED",
            detail=_log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"),
        )
        print(
            f"ERROR before LISTENING: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
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
