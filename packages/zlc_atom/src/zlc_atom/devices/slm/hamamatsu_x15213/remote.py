"""The SLM over a socket: the packet protocol, the server and the client.

The X15213 is plugged into the machine that stands beside it, and the bench
that drives it is another machine.  So this device is installed as a client
of a small server that owns the USB adapter: the protocol below is the whole
of that arrangement -- framing, the request vocabulary, the server loop and
the adapter the bench holds.  The phase contract it carries is the family's
(``canonical_phase``); everything here is this device's.
"""

from __future__ import annotations

import json
import logging
import socket
import socketserver
import struct
from threading import Event, Lock, Thread
import time
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
from zlc_durable import strict_json_loads
from zlc_pulse.endpoint import (
    bind_exclusive,
    drop_connection,
    drop_peer_connections,
    is_loopback_host,
)

from ..device import SlmAdapter, _shape, _validated_state, canonical_phase, phase_from_codes, phase_sequence_codes


#: The server's narration channel: the machine that owns the SLM shows these
#: records in its bench window, where a dedicated console used to scroll.
_LOG = logging.getLogger(__name__)

_REMOTE_VERSION = 2
_REMOTE_HEADER = struct.Struct("!II")
_MAX_REMOTE_METADATA_BYTES = 1024 * 1024
_MAX_REMOTE_PHASE_BYTES = 16 * 1024 * 1024
_MAX_REMOTE_SEQUENCE_BYTES = 256 * 1024 * 1024
_SERVER_SOCKET_TIMEOUT = 10.0


def _remote_phase_bytes(shape_yx: object) -> int:
    shape = _shape(tuple(shape_yx))
    size = shape[0] * shape[1] * np.dtype("<f4").itemsize
    if size > _MAX_REMOTE_PHASE_BYTES:
        raise ValueError("SLM shape exceeds the remote phase payload bound")
    return size


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray(size)
    target = memoryview(result)
    received = 0
    while received < size:
        count = connection.recv_into(target[received:])
        if not count:
            raise ConnectionError("SLM remote connection closed mid-message")
        received += count
    return bytes(result)


def _send_packet(
    connection: socket.socket, metadata: Mapping[str, object], payload: bytes | memoryview = b""
) -> None:
    encoded = json.dumps(
        dict(metadata), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > _MAX_REMOTE_METADATA_BYTES or len(payload) > _MAX_REMOTE_SEQUENCE_BYTES:
        raise ValueError("SLM remote message exceeds the maximum size")
    connection.sendall(_REMOTE_HEADER.pack(len(encoded), len(payload)) + encoded)
    if payload:
        connection.sendall(payload)


def _recv_packet(connection: socket.socket) -> tuple[dict[str, object], bytes]:
    metadata_size, payload_size = _REMOTE_HEADER.unpack(
        _recv_exact(connection, _REMOTE_HEADER.size)
    )
    if metadata_size > _MAX_REMOTE_METADATA_BYTES or payload_size > _MAX_REMOTE_SEQUENCE_BYTES:
        raise ValueError("SLM remote message exceeds the maximum size")
    decoded = strict_json_loads(
        _recv_exact(connection, metadata_size).decode("utf-8"), "SLM remote metadata"
    )
    if not isinstance(decoded, dict):
        raise TypeError("SLM remote metadata must be an object")
    return decoded, _recv_exact(connection, payload_size)


def _open_slm_server(
    slm: SlmAdapter, host: str, port: int, *, peers: bool = True
) -> socketserver.TCPServer:
    """Return the one-command-at-a-time RPC owner for an existing SLM.

    ``peers`` says whether a client on another machine is admitted.  A
    server run for the bench (the CLI) admits peers from the start; a
    server a bench holds for its own head admits nobody but this machine
    until the device is published (``admit_peers(True)``), and drops every
    peer when it is withdrawn (``admit_peers(False)``).
    """

    if not isinstance(slm, SlmAdapter):
        raise TypeError("SLM server requires a canonical SlmAdapter")
    _remote_phase_bytes(slm.shape_yx)
    bind_host = str(host).strip()
    if not bind_host or bind_host != host or any(char.isspace() for char in bind_host):
        raise ValueError("SLM server host must be non-empty text without whitespace")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("SLM server port must be an integer from 0 through 65535")

    sequence_token = None
    sequence_owner = None

    def response(ok: bool, error: str | None, *, include_phase: bool, sequence=None):
        phase = slm.last_commanded_phase if include_phase else None
        payload = (
            np.asarray(phase, dtype="<f4").tobytes()
            if include_phase and phase is not None
            else b""
        )
        state = {
            "identity": slm.identity,
            "shape_yx": list(slm.shape_yx),
            "command_revision": slm.command_revision,
            "mapping_revision": slm.mapping_revision,
            "receipt": dict(slm.last_command_receipt),
            "phase_bytes": len(payload),
        }
        metadata = {"version": _REMOTE_VERSION, "ok": ok, "error": error, "state": state}
        if sequence is not None:
            metadata["sequence"] = sequence
        return metadata, payload

    def command(request, payload, connection):
        nonlocal sequence_token, sequence_owner
        fields = set(request)
        if (
            type(request.get("version")) is not int
            or request["version"] != _REMOTE_VERSION
        ):
            reply = response(False, "unsupported SLM remote protocol", include_phase=True)
        elif (
            request.get("method") == "describe"
            and fields == {"version", "method"}
            and not payload
        ):
            reply = response(True, None, include_phase=True)
        elif request.get("method") in {"play_sequence", "release_sequence"} and fields == {"version", "method", "sequence_token"} and not payload:
            if request["sequence_token"] != sequence_token or connection is not sequence_owner:
                return response(False, "SLM sequence belongs to another or expired preparation", include_phase=True)
            try:
                if request["method"] == "play_sequence":
                    slm.play_phase_sequence()
                else:
                    slm.release_phase_sequence()
            except Exception as error:
                reply = response(False, f"{type(error).__name__}: {error}", include_phase=True)
            else:
                reply = response(True, None,
                                 include_phase=False,
                                 sequence={"sequence_token": sequence_token} if request["method"] == "play_sequence" else None)
            with connections_lock:
                sequence_token, sequence_owner = None, None
        elif request.get("method") == "prepare_sequence" and fields == {
            "version", "method", "command_revision", "mapping_revision", "shape_yx", "frame_count", "frame_intervals_seconds"
        }:
            if (type(request["command_revision"]) is not int or type(request["mapping_revision"]) is not int
                or request["command_revision"] != slm.command_revision or request["mapping_revision"] != slm.mapping_revision):
                return response(False, "stale SLM command; refresh from the physical device before sending", include_phase=True)
            if (type(request["frame_count"]) is not int or request["frame_count"] <= 0
                or request["frame_count"] > _MAX_REMOTE_SEQUENCE_BYTES // (slm.shape_yx[0] * slm.shape_yx[1])
                or not isinstance(request["shape_yx"], list)
                or any(type(value) is not int for value in request["shape_yx"])
                or request["shape_yx"] != list(slm.shape_yx)
                or len(payload) != request["frame_count"] * slm.shape_yx[0] * slm.shape_yx[1]):
                return response(False, "invalid SLM phase sequence payload", include_phase=True)
            try:
                frames = np.frombuffer(payload, dtype=np.uint8).reshape(request["frame_count"], *slm.shape_yx)
                prepared = slm.prepare_phase_sequence(frames, request["frame_intervals_seconds"])
            except Exception as error:
                reply = response(False, f"{type(error).__name__}: {error}", include_phase=True)
            else:
                with connections_lock:
                    sequence_token, sequence_owner = uuid4().hex, connection
                reply = response(True, None, include_phase=False, sequence={**prepared, "sequence_token": sequence_token})
        elif request.get("method") not in {"apply", "apply_codes"} or fields != {
            "version", "method", "command_revision", "mapping_revision", "shape_yx"
        }:
            reply = response(False, "invalid SLM remote request", include_phase=True)
        elif (
            type(request["command_revision"]) is not int
            or type(request["mapping_revision"]) is not int
            or request["command_revision"] != slm.command_revision
            or request["mapping_revision"] != slm.mapping_revision
        ):
            reply = response(
                False,
                "stale SLM command; refresh from the physical device before sending",
                include_phase=True,
            )
        elif (
            not isinstance(request["shape_yx"], list)
            or any(type(value) is not int for value in request["shape_yx"])
            or request["shape_yx"] != list(slm.shape_yx)
            or len(payload) != _remote_phase_bytes(slm.shape_yx) // (
                4 if request["method"] == "apply_codes" else 1
            )
        ):
            reply = response(False, "invalid SLM phase payload", include_phase=True)
        else:
            try:
                with connections_lock:
                    sequence_token, sequence_owner = None, None
                phase = (phase_from_codes(np.frombuffer(payload, dtype=np.uint8).reshape(slm.shape_yx), slm.shape_yx)
                         if request["method"] == "apply_codes" else
                         np.frombuffer(payload, dtype="<f4").reshape(slm.shape_yx))
                slm.apply_phase(phase)
            except Exception as error:
                reply = response(
                    False, f"{type(error).__name__}: {error}", include_phase=True
                )
            else:
                reply = response(True, None, include_phase=False)
        return reply

    command_lock, connections_lock = Lock(), Lock()
    connections: set[socket.socket] = set()
    closing = False

    def handle(connection: socket.socket, address, server) -> None:
        nonlocal sequence_token, sequence_owner
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client = f"{address[0]}:{address[1]}" if address else "?"
        with connections_lock:
            if closing:
                return
            # Admission is asked again under the lock withdrawal takes: a
            # peer verified just before Remote went off is otherwise
            # registered after the withdrawal's sweep and kept for good.
            if not (server.peers or is_loopback_host(address[0] if address else "")):
                return
            connections.add(connection)
        try:
            while True:
                # Idle sessions do not expire. Once a frame starts, retain the
                # existing bounded receive timeout for incomplete messages.
                connection.settimeout(None)
                if not connection.recv(1, socket.MSG_PEEK):
                    return
                connection.settimeout(_SERVER_SOCKET_TIMEOUT)
                request, payload = _recv_packet(connection)
                if request.get("method") == "cancel_sequence":
                    # Cancellation does not wait behind the playback command.
                    # The prepared token identifies this one sequence only.
                    with connections_lock:
                        valid = (set(request) == {"version", "method", "sequence_token"}
                                 and type(request.get("version")) is int and request["version"] == _REMOTE_VERSION
                                 and not payload and sequence_token is not None
                                 and request.get("sequence_token") == sequence_token)
                        if valid:
                            slm.cancel_phase_sequence()
                    # Its reply is an acknowledgment of the stop request only;
                    # the command connection supplies the final device receipt.
                    reply = ({"version": _REMOTE_VERSION, "ok": valid,
                              "error": None if valid else "SLM sequence cancellation token is not current"}, b"")
                else:
                    with command_lock:
                        if closing:
                            return
                        reply = command(request, payload, connection)
                metadata = reply[0]
                _LOG.info(
                    "SLM %s client=%s ok=%s%s command_revision=%s",
                    str(request.get("method", "?")).upper(), client,
                    metadata["ok"],
                    "" if metadata["error"] is None else f" error={metadata['error']!r}",
                    metadata.get("state", {}).get("command_revision", slm.command_revision),
                )
                _send_packet(connection, *reply)
        except (OSError, ValueError, TypeError) as error:
            if not closing:
                _LOG.info("SLM CONNECTION FAILED client=%s error=%s: %s", client, type(error).__name__, error)
        finally:
            with command_lock:
                with connections_lock:
                    owns_preparation = sequence_owner is connection
                    if owns_preparation:
                        sequence_token, sequence_owner = None, None
                if owns_preparation:
                    slm.release_phase_sequence()
            with connections_lock:
                connections.discard(connection)

    class _Server(socketserver.ThreadingTCPServer):
        """The listener, with the say over WHO it serves."""

        peers = False

        def server_bind(self) -> None:
            bind_exclusive(self.socket, self.server_address)
            self.server_address = self.socket.getsockname()

        def verify_request(self, request, client_address) -> bool:
            """Admit this machine always; admit a peer only while on offer."""

            client_host = client_address[0] if client_address else ""
            if self.peers or is_loopback_host(client_host):
                return True
            _LOG.info(
                "SLM PEER REFUSED client=%s:%s reason=not published",
                client_address[0], client_address[1],
            )
            return False

        def admit_peers(self, admitted: bool) -> None:
            """Put the head on offer to other machines, or take it back.

            Taking it back drops every peer's connection now; this
            machine's own clients are not touched.
            """

            with connections_lock:
                self.peers = bool(admitted)
                active = tuple(connections)
            if admitted:
                _LOG.info(
                    "SLM PEERS ADMITTED endpoint=%s:%d",
                    bind_host, int(self.server_address[1]),
                )
                return
            _LOG.info("SLM PEERS REFUSED dropped=%d", drop_peer_connections(active))

    server = _Server((bind_host, port), handle)
    server.peers = bool(peers)
    original_close = server.server_close

    def close() -> None:
        nonlocal closing
        with connections_lock:
            closing = True
            active = tuple(connections)
        cancel = getattr(slm, "cancel_phase_sequence", None)
        if cancel is not None:
            cancel()
        for connection in active:
            drop_connection(connection)
        original_close()

    server.server_close = close
    _LOG.info(
        "SLM SERVER LISTENING endpoint=%s:%d device=%s peers=%s",
        bind_host, int(server.server_address[1]), slm.identity,
        "admitted" if server.peers else "refused until published",
    )
    return server


def _rpc_call(
    endpoint: tuple[str, int] | socket.socket, method: str, arguments: tuple[object, ...], timeout: float
) -> tuple[dict[str, object], bytes]:
    if method == "describe" and not arguments:
        metadata, payload = {"version": _REMOTE_VERSION, "method": method}, b""
    elif method in {"apply", "apply_codes"} and len(arguments) == 4:
        command_revision, mapping_revision, shape_yx, payload = arguments
        metadata = {
            "version": _REMOTE_VERSION,
            "method": method,
            "command_revision": command_revision,
            "mapping_revision": mapping_revision,
            "shape_yx": shape_yx,
        }
        payload = bytes(payload)
    elif method == "prepare_sequence" and len(arguments) == 6:
        command_revision, mapping_revision, shape_yx, frame_count, intervals, payload = arguments
        metadata = {"version": _REMOTE_VERSION, "method": method,
                    "command_revision": command_revision, "mapping_revision": mapping_revision,
                    "shape_yx": shape_yx, "frame_count": frame_count, "frame_intervals_seconds": intervals}
        payload = memoryview(payload).cast("B")
    elif method in {"play_sequence", "cancel_sequence", "release_sequence"} and len(arguments) == 1:
        metadata, payload = {"version": _REMOTE_VERSION, "method": method, "sequence_token": arguments[0]}, b""
    else:
        raise ValueError("invalid local SLM remote call")
    if isinstance(endpoint, socket.socket):
        _send_packet(endpoint, metadata, payload)
        return _recv_packet(endpoint)
    with socket.create_connection(endpoint, timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send_packet(connection, metadata, payload)
        return _recv_packet(connection)


class _RemoteSlmAdapter:
    """Cached SLM proxy that contacts its server only to describe or send."""

    def __init__(self, host: str, port: int, timeout_seconds: float) -> None:
        remote_host = str(host).strip()
        if not remote_host or remote_host != host or any(char.isspace() for char in remote_host):
            raise ValueError("remote SLM host must be non-empty text without whitespace")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("remote SLM port must be an integer from 1 through 65535")
        timeout = float(timeout_seconds)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("remote SLM timeout must be finite and positive")
        self._endpoint = (remote_host, port)
        self._timeout = timeout
        #: ``_lock`` serialises the connection for a whole round trip; the
        #: cached state has its own short lock, so a question about it --
        #: from the Qt thread, say -- never waits behind an apply on the wire.
        self._lock = Lock()
        self._state_lock = Lock()
        self._connection: socket.socket | None = None
        self._identity = ""
        self._shape_yx = (1, 1)
        self._command_revision = 0
        self._mapping_revision = 0
        self._phase: np.ndarray | None = None
        self._receipt: dict[str, object] = {}
        self._uncertain = False
        self._closed = False
        self._sequence_token: str | None = None
        self._sequence_intervals: list[float] = []
        self._sequence_codes: np.ndarray | None = None
        self._describe()

    def _accept_state(
        self,
        value: object,
        payload: bytes,
        *,
        commanded: np.ndarray | None = None,
    ) -> None:
        if not isinstance(value, dict) or set(value) != {
            "identity", "shape_yx", "command_revision", "mapping_revision",
            "receipt", "phase_bytes",
        }:
            raise ValueError("SLM remote state has an invalid field set")
        identity = value["identity"]
        shape = _shape(tuple(value["shape_yx"]))
        phase_bytes = _remote_phase_bytes(shape)
        command_revision = value["command_revision"]
        mapping_revision = value["mapping_revision"]
        receipt = value["receipt"]
        if type(value["phase_bytes"]) is not int or value["phase_bytes"] != len(payload):
            raise ValueError("SLM remote phase length differs from its metadata")
        if payload:
            if commanded is not None:
                raise ValueError("SLM remote apply returned a redundant phase")
            if len(payload) != phase_bytes:
                raise ValueError("SLM remote phase byte count differs from its shape")
            phase = np.frombuffer(payload, dtype="<f4").reshape(shape)
        else:
            phase = commanded
        identity, shape, phase, command_revision, mapping_revision, receipt = (
            _validated_state(
                identity, shape, phase, command_revision, mapping_revision, receipt,
                commanded_phase=commanded,
            )
        )
        if self._identity and (identity != self._identity or shape != self._shape_yx):
            raise RuntimeError("SLM remote endpoint changed physical identity or shape")
        with self._state_lock:
            self._identity = identity
            self._shape_yx = shape
            self._command_revision = command_revision
            self._mapping_revision = mapping_revision
            self._phase = phase
            self._receipt = receipt
        self._uncertain = False

    def _request(
        self,
        method: str,
        arguments: tuple[object, ...] = (),
        *,
        commanded: np.ndarray | None = None,
        sequence: dict[str, object] | None = None,
    ) -> str | None:
        try:
            if self._connection is None:
                self._connection = socket.create_connection(self._endpoint, timeout=self._timeout)
                self._connection.settimeout(self._timeout)
                self._connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            value, payload = _rpc_call(self._connection, method, arguments, self._timeout)
            if not isinstance(value, dict) or set(value) not in ({"version", "ok", "error", "state"}, {"version", "ok", "error", "state", "sequence"}):
                raise ValueError("SLM remote response has an invalid field set")
            if (
                type(value["version"]) is not int
                or value["version"] != _REMOTE_VERSION
                or type(value["ok"]) is not bool
            ):
                raise ValueError("SLM remote response has an invalid protocol version")
            if method == "play_sequence" and value["ok"]:
                state = value["state"]
                receipt = state.get("receipt") if isinstance(state, dict) else None
                playback = receipt.get("sequence") if isinstance(receipt, dict) else None
                token = value.get("sequence")
                if (self._sequence_codes is None or not isinstance(token, dict)
                    or set(token) != {"sequence_token"}
                    or token.get("sequence_token") != arguments[0]
                    or not isinstance(playback, dict)
                    or type(state.get("command_revision")) is not int
                    or state["command_revision"] != self._command_revision + 1
                    or type(state.get("mapping_revision")) is not int
                    or state["mapping_revision"] != self._mapping_revision
                    or receipt.get("mapping_revision") != self._mapping_revision
                    or playback.get("mapping_revision") != self._mapping_revision
                    or type(playback.get("played_frames")) is not int
                    or not 0 <= playback["played_frames"] <= len(self._sequence_codes)
                    or type(playback.get("frame_count")) is not int
                    or playback["frame_count"] != len(self._sequence_codes)
                    or type(playback.get("cancelled")) is not bool
                    or (not playback["cancelled"] and playback["played_frames"] != len(self._sequence_codes))):
                    raise ValueError("SLM playback returned an invalid confirmation receipt")
                if playback["played_frames"]:
                    if receipt.get("outcome") != "known-new":
                        raise ValueError("SLM playback did not confirm its last displayed frame")
                    commanded = phase_from_codes(
                        self._sequence_codes[playback["played_frames"] - 1], self._shape_yx
                    )
                else:
                    commanded = self.last_commanded_phase
            self._accept_state(
                value["state"], payload, commanded=commanded if value["ok"] else None
            )
            if sequence is not None and value["ok"]:
                if not isinstance(value.get("sequence"), dict):
                    raise ValueError("SLM remote preparation did not return sequence metadata")
                sequence.update(value["sequence"])
            if not value["ok"]:
                if not isinstance(value["error"], str) or not value["error"]:
                    raise ValueError("SLM remote error is missing its message")
                return value["error"]
            if value["error"] is not None:
                raise ValueError("successful SLM remote response contains an error")
            return None
        except BaseException:
            self._close_connection()
            raise

    def _close_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def _describe(self) -> None:
        error = self._request("describe")
        if error is not None:
            raise RuntimeError(error)

    def _mark_unknown(self) -> None:
        with self._state_lock:
            self._phase = None
            self._receipt = {
                **self._receipt,
                "outcome": "unknown",
                "stage": "remote-transport",
                "readback": "not-run",
            }
            self._receipt.pop("sequence", None)
        self._uncertain = True

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self._shape_yx

    @property
    def last_commanded_phase(self) -> np.ndarray | None:
        with self._state_lock:
            return self._phase

    @property
    def command_revision(self) -> int:
        with self._state_lock:
            return self._command_revision

    @property
    def mapping_revision(self) -> int:
        with self._state_lock:
            return self._mapping_revision

    @property
    def last_command_receipt(self) -> Mapping[str, object]:
        with self._state_lock:
            return dict(self._receipt)

    def apply_phase(self, radians: object) -> np.ndarray:
        canonical = canonical_phase(radians, self._shape_yx)
        return self._apply(canonical, "apply", canonical.tobytes())

    def apply_phase_codes(self, codes: object) -> np.ndarray:
        """Send one already-computed 8-bit phase frame through the same command lane."""
        source = np.asarray(codes)
        if source.shape != self._shape_yx or source.dtype != np.uint8:
            raise ValueError("SLM phase codes must be a uint8 matrix matching the full device shape")
        payload = source.tobytes()
        canonical = phase_from_codes(np.frombuffer(payload, np.uint8).reshape(self._shape_yx), self._shape_yx)
        return self._apply(canonical, "apply_codes", payload)

    def _apply(self, canonical: np.ndarray, method: str, payload: bytes) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("remote SLM is closed")
            if self._uncertain:
                self._describe()
            self._sequence_token = None
            self._sequence_codes = None
            self._sequence_intervals = []
            expected_command = self._command_revision
            expected_mapping = self._mapping_revision
            try:
                error = self._request(
                    method,
                    (
                        expected_command,
                        expected_mapping,
                        list(self._shape_yx),
                        payload,
                    ),
                    commanded=canonical,
                )
                if error is None and (
                    self._command_revision != expected_command + 1
                    or self._mapping_revision != expected_mapping
                    or self._receipt.get("outcome") != "known-new"
                ):
                    raise ValueError("SLM remote apply returned an invalid state transition")
            except BaseException:
                self._mark_unknown()
                raise
            if error is not None:
                raise RuntimeError(error)
            return canonical

    def prepare_phase_sequence(self, codes: object, frame_interval_seconds: object) -> dict[str, object]:
        started = time.perf_counter()
        frames, intervals = phase_sequence_codes(codes, self._shape_yx, frame_interval_seconds)
        if frames.nbytes > _MAX_REMOTE_SEQUENCE_BYTES:
            raise ValueError("SLM phase sequence exceeds the remote sequence payload bound")
        prepared = {}
        with self._lock:
            if self._closed:
                raise RuntimeError("remote SLM is closed")
            if self._uncertain:
                self._describe()
            self._sequence_token = None
            self._sequence_codes = None
            self._sequence_intervals = []
            borrowed = not np.asarray(codes).flags.writeable and frames.flags.c_contiguous
            payload = memoryview(frames).cast("B") if borrowed else frames.tobytes()
            error = self._request("prepare_sequence", (
                self._command_revision, self._mapping_revision, list(self._shape_yx),
                len(frames), intervals.tolist(), payload,
            ), sequence=prepared, commanded=self.last_commanded_phase)
            if error is not None:
                raise RuntimeError(error)
            token = prepared.pop("sequence_token", None)
            if not isinstance(token, str) or not token:
                raise ValueError("SLM sequence preparation token is missing")
            self._sequence_token = token
            self._sequence_intervals = intervals.tolist()
            # The solver hands out protected readonly storage; retain its
            # view. Writable callers use the already-serialized upload bytes,
            # so no second full movie copy is needed for receipt reconstruction.
            self._sequence_codes = (frames if borrowed else
                                    np.frombuffer(payload, dtype=np.uint8).reshape(frames.shape))
        return {**prepared, "upload_roundtrip_ms": (time.perf_counter() - started) * 1000}

    def cancel_phase_sequence(self) -> None:
        token = self._sequence_token
        if token is None:
            return
        value, payload = _rpc_call(self._endpoint, "cancel_sequence", (token,), self._timeout)
        if (not isinstance(value, dict) or set(value) != {"version", "ok", "error"}
            or value["version"] != _REMOTE_VERSION or payload):
            raise ValueError("SLM sequence cancellation returned an invalid acknowledgment")
        if not value["ok"]:
            raise RuntimeError(value["error"])

    def release_phase_sequence(self) -> None:
        with self._lock:
            token, self._sequence_token = self._sequence_token, None
            self._sequence_codes = None
            self._sequence_intervals = []
            if token is None:
                return
            error = self._request("release_sequence", (token,), commanded=self.last_commanded_phase)
            if error is not None:
                raise RuntimeError(error)

    def play_phase_sequence(self, stop_requested: Callable[[], bool] | None = None) -> dict[str, object]:
        finished = Event()
        cancellation_errors = []

        def watch_stop():
            while not finished.wait(0.01):
                if stop_requested():
                    try:
                        self.cancel_phase_sequence()
                    except Exception as error:
                        cancellation_errors.append(f"{type(error).__name__}: {error}")
                    return

        with self._lock:
            if self._closed:
                raise RuntimeError("remote SLM is closed")
            token = self._sequence_token
            if token is None:
                raise RuntimeError("SLM phase sequence has not been prepared")
            watcher = None
            if stop_requested is not None:
                if stop_requested():
                    self.cancel_phase_sequence()
                else:
                    watcher = Thread(target=watch_stop, name="slm-sequence-stop", daemon=True)
                    watcher.start()
            started = time.perf_counter()
            try:
                # Long authored playback does not shorten its command timeout.
                timeout = self._timeout + sum(self._sequence_intervals)
                self._connection.settimeout(timeout)
                try:
                    error = self._request("play_sequence", (token,))
                except BaseException:
                    self._mark_unknown()
                    raise
                if error is not None:
                    raise RuntimeError(error)
                result = self.last_command_receipt.get("sequence")
                if not isinstance(result, dict):
                    self._mark_unknown()
                    raise ValueError("SLM playback did not return a final sequence receipt")
                return {**result, "play_roundtrip_ms": (time.perf_counter() - started) * 1000,
                        "cancellation_errors": cancellation_errors, "receipt": self.last_command_receipt}
            finally:
                finished.set()
                if watcher is not None:
                    watcher.join(self._timeout)
                self._sequence_token = None
                self._sequence_intervals = []
                self._sequence_codes = None
                if self._connection is not None:
                    self._connection.settimeout(self._timeout)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.cancel_phase_sequence()
        finally:
            with self._lock:
                self._closed = True
                self._sequence_token = None
                self._sequence_intervals = []
                self._sequence_codes = None
                self._close_connection()


__all__ = ["_RemoteSlmAdapter", "_open_slm_server", "_remote_phase_bytes"]
