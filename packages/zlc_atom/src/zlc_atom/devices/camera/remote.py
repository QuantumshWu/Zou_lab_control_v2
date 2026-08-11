"""Small single-connection RPC for a camera on a separated machine.

The server owns one real ``CameraAdapter`` (virtual, pylon or dcam); the wire
layer only carries plain JSON trees and raw frame bytes and never grows a
second camera state machine.  The control plane is the pulse server's frame
protocol: one length-prefixed JSON frame per request, one per response, with
the method whitelist being exactly the ``CameraAdapter`` contract plus
``open``/``close``.

Frame data never rides inside JSON.  A ``read_frame_records`` response is the
JSON frame carrying each record's metadata (shape and dtype included), followed
on the SAME connection by one length-prefixed raw byte block per record, in
order.  The client rebuilds each image with ``np.frombuffer`` from the declared
shape and dtype, so the sensor's native integer dtype survives end to end --
no JSON number ever touches a pixel.

Bandwidth is a physical fact, not a tunable: a 2048x2048 uint8 frame is 4 MiB,
so 10 Hz full-frame streaming is 336 Mbps and needs gigabit wired Ethernet,
and 100 Hz is 3.36 Gbps and needs 10GbE or an ROI/binning working point that
shrinks the frame.  WiFi cannot sustain full-frame streaming at all.
"""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import struct
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from .contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from .endpoint import DEFAULT_BIND_HOST, DEFAULT_HOST, DEFAULT_PORT


MAX_FRAME_BYTES = 8 * 1024 * 1024
_FRAME_HEADER = struct.Struct("!I")
#: What a client is told when its connection ends under it.  The camera goes
#: to whoever connected last, so this is the ordinary way a client finds out
#: that a newer one took over -- and it must not read like a protocol defect.
_CONNECTION_ENDED = (
    "the connection to the camera server ended; if another client has "
    "connected since, that one now holds the camera"
)

#: The whole remote surface: the seven CameraAdapter members plus the
#: duck-typed open/close lifecycle the installation binding already probes.
REMOTE_METHODS = (
    "open",
    "close",
    "timeout",
    "configure_measurement",
    "capture_working_point",
    "arm",
    "read_frame_records",
    "finish_record_capture",
    "capture_state",
)



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
    print(f"[{timestamp}] ZLC CAMERA {event}{suffix}", flush=True)


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


# --------------------------------------------------------------------- wire

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
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
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
    return json.loads(payload.decode("utf-8"))


def _send_block(connection: socket.socket, block: bytes) -> None:
    """Send one raw image block; the JSON size cap does not apply to pixels."""

    connection.sendall(_FRAME_HEADER.pack(len(block)))
    connection.sendall(block)


def _recv_block(connection: socket.socket, expected_bytes: int) -> bytes:
    """Receive one raw image block whose size the metadata already declared.

    The declared shape and dtype are the only truth about this block, so a
    header that disagrees with them means the stream is out of step and the
    connection cannot be trusted further -- not a value to be clamped.
    """

    header = _recv_exact(connection, _FRAME_HEADER.size)
    if header is None:
        raise ConnectionError("remote connection closed before an image block")
    (size,) = _FRAME_HEADER.unpack(header)
    if size != int(expected_bytes):
        raise ConnectionError(
            f"image block carries {size} bytes but its metadata declared "
            f"{int(expected_bytes)}; the frame stream is out of step"
        )
    payload = _recv_exact(connection, size)
    if payload is None:
        raise ConnectionError("remote connection closed inside an image block")
    return payload


# ------------------------------------------------------------------ encoding

def _optional_seconds(value: object) -> float | None:
    return None if value is None else float(value)


def _encode_working_point(point: CameraWorkingPoint) -> dict:
    """Field name for field name; dtype as its little-endian ``.str`` form.

    ``acquisition_mode`` travels as the enum VALUE: the field is typed as text
    and the virtual camera already stores plain text, so the wire carries the
    one spelling both ends construct from.
    """

    return {
        "acquisition_mode": getattr(point.acquisition_mode, "value", point.acquisition_mode),
        "frame_shape_yx": list(point.frame_shape_yx),
        "sensor_shape_yx": list(point.sensor_shape_yx),
        "roi_origin_yx": list(point.roi_origin_yx),
        "roi_shape_yx": list(point.roi_shape_yx),
        "binning_yx": list(point.binning_yx),
        "dtype": np.dtype(point.dtype).str,
        "count_unit": str(point.count_unit),
        "exposure_seconds": float(point.exposure_seconds),
        "required_external_trigger_interval_seconds": _optional_seconds(
            point.required_external_trigger_interval_seconds
        ),
        "external_trigger_integration_start_offset_seconds": _optional_seconds(
            point.external_trigger_integration_start_offset_seconds
        ),
        "gain": float(point.gain),
        "readout_mode": str(point.readout_mode),
    }


def _decode_working_point(tree: Mapping[str, Any]) -> CameraWorkingPoint:
    return CameraWorkingPoint(
        acquisition_mode=str(tree["acquisition_mode"]),
        frame_shape_yx=tuple(tree["frame_shape_yx"]),
        sensor_shape_yx=tuple(tree["sensor_shape_yx"]),
        roi_origin_yx=tuple(tree["roi_origin_yx"]),
        roi_shape_yx=tuple(tree["roi_shape_yx"]),
        binning_yx=tuple(tree["binning_yx"]),
        dtype=np.dtype(str(tree["dtype"])),
        count_unit=str(tree["count_unit"]),
        exposure_seconds=float(tree["exposure_seconds"]),
        required_external_trigger_interval_seconds=_optional_seconds(
            tree["required_external_trigger_interval_seconds"]
        ),
        external_trigger_integration_start_offset_seconds=_optional_seconds(
            tree["external_trigger_integration_start_offset_seconds"]
        ),
        gain=float(tree["gain"]),
        readout_mode=str(tree["readout_mode"]),
    )


def _encode_terminal(record: CameraCaptureTerminalRecord) -> dict:
    return {
        "produced_count": int(record.produced_count),
        "source_stopped": bool(record.source_stopped),
        "no_more_frames": bool(record.no_more_frames),
        "joined": bool(record.joined),
    }


def _decode_terminal(tree: Mapping[str, Any]) -> CameraCaptureTerminalRecord:
    return CameraCaptureTerminalRecord(
        int(tree["produced_count"]),
        bool(tree["source_stopped"]),
        bool(tree["no_more_frames"]),
        bool(tree["joined"]),
    )


def _frame_meta(record: CameraFrameRecord) -> dict:
    """Everything about one frame except its pixels; those go as raw bytes."""

    image = np.asarray(record.image)
    return {
        "source_ordinal": record.source_ordinal,
        "produced_count": record.produced_count,
        "frame_stamp": record.frame_stamp,
        "camera_stamp": record.camera_stamp,
        "timestamp_seconds": record.timestamp_seconds,
        "timestamp_microseconds": record.timestamp_microseconds,
        "host_received_at_ns": record.host_received_at_ns,
        "driver_buffer_index": record.driver_buffer_index,
        "shape": list(image.shape),
        "dtype": image.dtype.str,
    }


def _decode_frame_record(meta: Mapping[str, Any], block: bytes) -> CameraFrameRecord:
    image = np.frombuffer(block, dtype=np.dtype(str(meta["dtype"]))).reshape(
        tuple(int(item) for item in meta["shape"])
    )
    return CameraFrameRecord(
        image,
        int(meta["source_ordinal"]),
        meta["produced_count"],
        meta["frame_stamp"],
        meta["camera_stamp"],
        meta["timestamp_seconds"],
        meta["timestamp_microseconds"],
        int(meta["host_received_at_ns"]),
        meta["driver_buffer_index"],
    )


class RemoteError(RuntimeError):
    """An exception raised by the server while handling one request."""

    def __init__(self, remote_type: str, message: str) -> None:
        super().__init__(f"{remote_type}: {message}")
        self.remote_type = remote_type
        self.message = message


# -------------------------------------------------------------------- server

class _CameraRemoteHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, CameraRemoteServer)
        client = f"{self.client_address[0]}:{self.client_address[1]}"
        server.claim_client(client, self.request)
        _server_log("CLIENT CONNECTED", client=client, detail="status=OWNER")
        disconnect_reason = "client disconnect"
        try:
            # A blocking read, with no deadline of any kind: an owner that has
            # gone away is displaced by the next one to connect, and until
            # somebody wants the camera nobody cares.  A monitor that polls
            # every few seconds and an operator thinking for minutes must be
            # indistinguishable from here.
            while True:
                request = _recv_frame(self.request)
                if request is None:
                    return
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
                except Exception as exc:
                    _server_log(
                        "RPC ERROR",
                        client=client,
                        detail=_log_fields(
                            method=method,
                            error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}",
                        ),
                    )
                    _send_frame(
                        self.request,
                        {
                            "id": request_id,
                            "ok": False,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        },
                    )
                    continue
                if method == "read_frame_records":
                    # Two-part response: metadata as the ordinary JSON frame,
                    # then one raw block per record on the same connection.
                    # The records are materialized before the JSON frame is
                    # sent so a failing frame never leaves a half-told story.
                    records = tuple(result)
                    _send_frame(
                        self.request,
                        {
                            "id": request_id,
                            "ok": True,
                            "result": [_frame_meta(record) for record in records],
                        },
                    )
                    for record in records:
                        _send_block(self.request, np.asarray(record.image).tobytes())
                else:
                    _send_frame(self.request, {"id": request_id, "ok": True, "result": result})
        except OSError as exc:
            # Closed, reset, or dropped by a newer client taking the camera.
            # Each of those ends this session; none is a server defect.
            disconnect_reason = f"client connection dropped: {type(exc).__name__}"
        finally:
            capture_idle = server.client_disconnected(client=client, reason=disconnect_reason)
            _server_log(
                "CLIENT DISCONNECTED",
                client=client,
                detail=_log_fields(
                    capture="IDLE" if capture_idle else "NOT_VERIFIED",
                    reason=disconnect_reason,
                ),
            )


class CameraRemoteServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """An RPC façade over one local CameraAdapter, held by whoever connected last.

    One camera cannot take two conversations at once, so one connection owns
    it, and which one is decided by arrival: a new client takes the camera and
    the previous connection is dropped.  Same policy as the pulse server, for
    the same reason -- it lets this server never ask whether a silent client is
    still alive, because nobody cares until somebody else wants the camera.
    """

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False
    request_queue_size = 8

    def __init__(self, address: tuple[str, int], adapter: object) -> None:
        # Structural, not isinstance: the contract is a Protocol and the local
        # adapter is injected by whoever composed the server.
        for name in ("configure_measurement", "arm", "read_frame_records"):
            if not callable(getattr(adapter, name, None)):
                raise TypeError("adapter must implement the CameraAdapter contract")
        self.adapter = adapter
        self._client_lock = threading.RLock()
        self._owner_client: str | None = None
        self._owner_connection: socket.socket | None = None
        super().__init__(address, _CameraRemoteHandler)

    def handle_error(self, request, client_address) -> None:
        """Prefix real handler defects with timestamp/client, then keep the traceback."""

        exc = sys.exc_info()[1]
        client = f"{client_address[0]}:{client_address[1]}" if client_address else None
        detail = _log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}")
        _server_log("HANDLER ERROR", client=client, detail=detail)
        super().handle_error(request, client_address)

    def claim_client(self, client: str, connection: socket.socket) -> None:
        """Give the camera to the newest connection and drop the previous one.

        Ending the old session first is what makes this safe: its capture is
        finished and it stops owning anything before the newcomer is told it
        may start, so the two never overlap on the adapter.
        """

        with self._client_lock:
            previous, connection_to_drop = self._owner_client, self._owner_connection
            if previous is not None:
                _server_log("CLIENT REPLACED", client=previous, detail=_log_fields(by=client))
                self.client_disconnected(client=previous, reason=f"replaced by {client}")
                if connection_to_drop is not None:
                    _drop_connection(connection_to_drop)
            self._owner_client = client
            self._owner_connection = connection

    def _release_client(self, client: str | None, *, force: bool = False) -> None:
        with self._client_lock:
            if force or client is None or self._owner_client == client:
                self._owner_client = None
                self._owner_connection = None

    def dispatch(self, method: str, params: Mapping[str, Any], *, client: str = "server") -> Any:
        if method not in REMOTE_METHODS:
            raise ValueError(f"unknown remote method: {method}")
        adapter = self.adapter
        if method == "open":
            # Duck-typed on both lifecycle ends, exactly as the installation
            # binding probes them: the virtual camera has no open at all.
            opener = getattr(adapter, "open", None)
            if callable(opener):
                opener()
            _server_log("OPEN", client=client, detail=_log_fields(adapter=type(adapter).__name__))
            return None
        if method == "close":
            closer = getattr(adapter, "close", None)
            if callable(closer):
                closer()
            _server_log("CLOSE", client=client)
            return None
        if method == "timeout":
            return float(adapter.timeout)
        if method == "configure_measurement":
            roi = params.get("roi_xywh")
            point = adapter.configure_measurement(
                exposure_seconds=float(params["exposure_seconds"]),
                roi_xywh=None if roi is None else tuple(int(value) for value in roi),
            )
            _server_log(
                "CONFIGURE",
                client=client,
                detail=_log_fields(
                    exposure_s=f"{point.exposure_seconds:g}",
                    roi_origin_yx=tuple(point.roi_origin_yx),
                    frame_shape_yx=tuple(point.frame_shape_yx),
                    dtype=np.dtype(point.dtype).str,
                ),
            )
            return _encode_working_point(point)
        if method == "capture_working_point":
            return _encode_working_point(adapter.capture_working_point())
        if method == "arm":
            frames = params.get("frames")
            groups = params.get("source_group_sizes")
            adapter.arm(
                None if frames is None else int(frames),
                source_group_sizes=None if groups is None else tuple(int(value) for value in groups),
                buffer_frame_count=int(params["buffer_frame_count"]),
                timeout=float(params["timeout"]),
            )
            _server_log(
                "ARM",
                client=client,
                detail=_log_fields(
                    frames="continuous" if frames is None else int(frames),
                    buffer=int(params["buffer_frame_count"]),
                ),
            )
            return None
        if method == "read_frame_records":
            # Returned as records; the handler owns the two-part wire encoding
            # because only it holds the connection the raw blocks ride on.
            return adapter.read_frame_records(
                int(params["n"]),
                timeout=float(params["timeout"]),
                exact=bool(params["exact"]),
            )
        if method == "finish_record_capture":
            terminal = adapter.finish_record_capture()
            _server_log(
                "FINISH",
                client=client,
                detail=_log_fields(
                    produced=terminal.produced_count,
                    no_more_frames=terminal.no_more_frames,
                    joined=terminal.joined,
                ),
            )
            return _encode_terminal(terminal)
        return bool(adapter.capture_state())

    def client_disconnected(self, *, client: str | None = None, reason: str = "client disconnect") -> bool:
        """Finish any capture the departing owner left armed; True means idle.

        Only the current owner's departure touches the adapter: a displaced
        handler unwinding after a takeover finds it no longer owns anything
        and must not finish the NEW owner's capture.
        """

        with self._client_lock:
            owned = client in {None, "server"} or self._owner_client == client
        if not owned:
            return True
        try:
            if self.adapter.capture_state():
                _server_log("AUTO-FINISH", client=client, detail=_log_fields(reason=reason))
                terminal = self.adapter.finish_record_capture()
                _server_log(
                    "AUTO-FINISH DONE",
                    client=client,
                    detail=_log_fields(produced=terminal.produced_count, joined=terminal.joined),
                )
            self._release_client(client, force=client == "server")
            return True
        except Exception as exc:
            _server_log(
                "AUTO-FINISH FAILED",
                client=client,
                detail=_log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"),
            )
            self._release_client(client, force=client == "server")
            return False


# -------------------------------------------------------------------- client

class RemoteCameraAdapter:
    """The local ``CameraAdapter`` surface backed by one TCP connection.

    ``request_timeout`` bounds every control-plane exchange.  ``arm`` and
    ``read_frame_records`` carry their own operation timeout, so for those the
    network deadline is the operation's own timeout plus this allowance --
    the server is expected to block for the operation, not to be slow.
    """

    def __init__(self, host: str, port: int, *, request_timeout: float = 10.0) -> None:
        if not host:
            raise ValueError("remote host is required")
        if int(port) <= 0 or int(port) > 65535:
            raise ValueError("remote port is outside the TCP range")
        if not (float(request_timeout) > 0) or not np.isfinite(float(request_timeout)):
            raise ValueError("request_timeout must be finite and positive")
        self.host = str(host)
        self.port = int(port)
        self.request_timeout = float(request_timeout)
        self._socket: socket.socket | None = None
        self._request_id = 0
        self._io_lock = threading.RLock()
        self._timeout: float | None = None

    def open(self) -> None:
        with self._io_lock:
            try:
                if self._socket is None:
                    self._socket = socket.create_connection(
                        (self.host, self.port), timeout=self.request_timeout
                    )
                    self._socket.settimeout(self.request_timeout)
                self._call_locked("open", {})
                # The adapter's timeout is a hardware-side fact; fetch it once
                # at the handshake instead of paying a round trip per read.
                self._timeout = float(self._call_locked("timeout", {}))
            except Exception:
                self._disconnect_locked()
                raise

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

    @property
    def timeout(self) -> float:
        if self._timeout is None:
            raise RuntimeError("remote camera is not open; its timeout lives on the server")
        return self._timeout

    def configure_measurement(
        self,
        *,
        exposure_seconds: float,
        roi_xywh: tuple[int, int, int, int] | None,
    ) -> CameraWorkingPoint:
        tree = self._call(
            "configure_measurement",
            {
                "exposure_seconds": float(exposure_seconds),
                "roi_xywh": None if roi_xywh is None else [int(value) for value in roi_xywh],
            },
        )
        return _decode_working_point(tree)

    def capture_working_point(self) -> CameraWorkingPoint:
        return _decode_working_point(self._call("capture_working_point", {}))

    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        self._call(
            "arm",
            {
                "frames": None if frames is None else int(frames),
                "source_group_sizes": (
                    None
                    if source_group_sizes is None
                    else [int(value) for value in source_group_sizes]
                ),
                "buffer_frame_count": int(buffer_frame_count),
                "timeout": float(timeout),
            },
            operation_seconds=float(timeout),
        )

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> tuple[CameraFrameRecord, ...]:
        with self._io_lock:
            metas = self._call_locked(
                "read_frame_records",
                {"n": int(n), "timeout": float(timeout), "exact": bool(exact)},
                operation_seconds=float(timeout),
            )
            # The raw blocks follow the JSON answer on the same connection, in
            # metadata order, still under the lock -- another thread's request
            # interleaved here would read pixels as a frame header.
            records: list[CameraFrameRecord] = []
            try:
                for meta in metas:
                    expected = int(np.prod([int(item) for item in meta["shape"]], dtype=np.int64))
                    expected *= np.dtype(str(meta["dtype"])).itemsize
                    block = _recv_block(self._socket, expected)
                    records.append(_decode_frame_record(meta, block))
            except (TimeoutError, OSError) as exc:
                self._disconnect_locked()
                raise ConnectionError(f"{_CONNECTION_ENDED} ({type(exc).__name__})") from exc
            return tuple(records)

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        return _decode_terminal(self._call("finish_record_capture", {}))

    def capture_state(self) -> bool:
        return bool(self._call("capture_state", {}))

    def _call(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        operation_seconds: float = 0.0,
    ) -> Any:
        with self._io_lock:
            return self._call_locked(method, params, operation_seconds=operation_seconds)

    def _disconnect_locked(self) -> None:
        self._timeout = None
        connection, self._socket = self._socket, None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def _call_locked(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        operation_seconds: float = 0.0,
    ) -> Any:
        if self._socket is None:
            raise RuntimeError("remote camera is not open")
        self._request_id += 1
        request = {"id": self._request_id, "method": method, "params": dict(params)}
        deadline = self.request_timeout + max(0.0, float(operation_seconds))
        try:
            self._socket.settimeout(deadline)
            try:
                _send_frame(self._socket, request)
                response = _recv_frame(self._socket)
            finally:
                # Restore the base deadline even on failure: _disconnect_locked
                # may run later, and a poisoned long timeout must not leak into
                # the next short control call on a surviving socket.
                if self._socket is not None:
                    self._socket.settimeout(self.request_timeout)
        except TimeoutError as exc:
            self._disconnect_locked()
            raise ConnectionError(
                f"the camera server did not answer within {deadline:g}s"
            ) from exc
        except OSError as exc:
            self._disconnect_locked()
            raise ConnectionError(f"{_CONNECTION_ENDED} ({type(exc).__name__})") from exc
        if response is None:
            self._disconnect_locked()
            raise ConnectionError(_CONNECTION_ENDED)
        if not isinstance(response, Mapping):
            raise ConnectionError("remote response is not an object")
        if response.get("id") != self._request_id:
            raise ConnectionError("remote response id differs from the request")
        if not response.get("ok"):
            error = response.get("error")
            if isinstance(error, Mapping):
                raise RemoteError(str(error.get("type", "RemoteError")), str(error.get("message", "")))
            raise RemoteError("RemoteError", "remote request failed")
        return response.get("result")


# ----------------------------------------------------------------------- CLI

def serve(adapter: object, host: str = DEFAULT_BIND_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve one supplied local adapter until interrupted."""

    with CameraRemoteServer((host, int(port)), adapter) as server:
        listen_host = host or "0.0.0.0"
        actual_port = int(server.server_address[1])
        _server_log("RPC LISTENING", detail=_log_fields(endpoint=f"{listen_host}:{actual_port}"))
        _server_log(
            "CLIENT CONNECT EXAMPLE",
            detail=(
                f'RemoteCameraAdapter("{"127.0.0.1" if listen_host in ("0.0.0.0", "::") else listen_host}", '
                f"{actual_port})"
            ),
        )
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            _server_log("SERVER STOPPING", detail=_log_fields(reason="keyboard interrupt"))
        finally:
            server.client_disconnected(client="server", reason="server shutdown")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the server CLI parser so its defaults are directly testable."""

    parser = argparse.ArgumentParser(
        description="Serve one local camera adapter over a thin TCP RPC."
    )
    parser.add_argument("--host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--adapter",
        choices=ADAPTER_CHOICES,
        default="virtual",
        help="which local camera this server owns; virtual is for loopback tests",
    )
    parser.add_argument("--exposure-seconds", type=float, default=0.02)
    parser.add_argument(
        "--free-running",
        action="store_true",
        help="virtual only: produce frames on a clock instead of on triggers",
    )
    parser.add_argument("--serial", default=None, help="pylon only: camera serial number")
    parser.add_argument("--trigger-source", default="Line1", help="pylon only: hardware trigger line")
    parser.add_argument("--device-index", type=int, default=0, help="dcam only: SDK device index")
    parser.add_argument("--readout-speed", type=int, default=1, help="dcam only: readout speed setting")
    return parser


def _virtual_adapter(args: argparse.Namespace) -> object:
    from ..simulation.camera import VirtualCamera, VirtualCameraConfig

    config = VirtualCameraConfig(exposure_seconds=float(args.exposure_seconds))
    shape = config.frame_shape_yx
    limits = np.iinfo(np.dtype(config.frame_dtype))
    ramp = np.arange(shape[0] * shape[1], dtype=np.uint32).reshape(shape)

    def synthetic(ordinal: int, exposure: float) -> np.ndarray:
        # Deterministic and ordinal-dependent, so a loopback client can
        # see frames advance without any simulation world behind them.
        del exposure
        return (ramp + int(ordinal)) % (int(limits.max) + 1)

    return VirtualCamera(
        config,
        frame_source=synthetic,
        free_running=bool(args.free_running),
    )


def _pylon_adapter(args: argparse.Namespace) -> object:
    if not args.serial:
        raise SystemExit("--adapter pylon requires --serial")
    from .pylon import PylonCameraAdapter, PylonCameraConfig

    return PylonCameraAdapter(
        PylonCameraConfig(
            serial=str(args.serial),
            exposure_seconds=float(args.exposure_seconds),
            trigger_source=str(args.trigger_source),
        )
    )


def _dcam_adapter(args: argparse.Namespace) -> object:
    from .dcam import DcamCameraAdapter, DcamCameraConfig

    return DcamCameraAdapter(
        DcamCameraConfig(
            exposure_seconds=float(args.exposure_seconds),
            readout_speed=int(args.readout_speed),
            device_index=int(args.device_index),
        )
    )


#: The adapter registry IS the CLI choice vocabulary: one declaration decides
#: what can be served, and each builder keeps its SDK import lazy.
_ADAPTER_BUILDERS = {
    "virtual": _virtual_adapter,
    "pylon": _pylon_adapter,
    "dcam": _dcam_adapter,
}

ADAPTER_CHOICES = tuple(_ADAPTER_BUILDERS)


def _build_adapter(args: argparse.Namespace) -> object:
    """Construct the one local adapter this server owns."""

    return _ADAPTER_BUILDERS[args.adapter](args)


def _main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _server_log(
        "SERVER STARTING",
        detail=_log_fields(
            adapter=args.adapter,
            endpoint=f"{args.host}:{args.port}",
            python=sys.executable,
        ),
    )
    adapter: object | None = None
    try:
        adapter = _build_adapter(args)
        opener = getattr(adapter, "open", None)
        if callable(opener):
            opener()
        _server_log("CAMERA READY", detail=_log_fields(adapter=type(adapter).__name__))
        serve(adapter, args.host, args.port)
    except KeyboardInterrupt:
        _server_log("SERVER STOPPING", detail=_log_fields(reason="keyboard interrupt"))
        return 0
    except Exception as exc:
        _server_log(
            "SERVER FAILED",
            detail=_log_fields(error=f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"),
        )
        print("ZLC camera server did not enter LISTENING state; no client can connect.", file=sys.stderr, flush=True)
        return 3
    finally:
        if adapter is not None:
            closer = getattr(adapter, "close", None)
            if callable(closer):
                closer()
                _server_log("SERVER CLOSED", detail=_log_fields(camera="closed"))
    return 0


__all__ = [
    "ADAPTER_CHOICES",
    "CameraRemoteServer",
    "MAX_FRAME_BYTES",
    "REMOTE_METHODS",
    "RemoteCameraAdapter",
    "RemoteError",
    "build_arg_parser",
    "serve",
]


if __name__ == "__main__":
    raise SystemExit(_main())
