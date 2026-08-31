"""Canonical phase contract and installation binding for SLM adapters."""

from __future__ import annotations

import json
import logging
import socket
import socketserver
import struct
from threading import Lock
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from zlc_atom.install.descriptors import InstalledLeaf


#: The server's narration channel: the machine that owns the SLM shows these
#: records in its bench window, where a dedicated console used to scroll.
_LOG = logging.getLogger(__name__)

_TWO_PI = 2.0 * np.pi
_MAX_WRAPPED_PHASE = np.nextafter(np.float32(_TWO_PI), np.float32(0.0))
_REMOTE_VERSION = 1
_REMOTE_HEADER = struct.Struct("!II")
_MAX_REMOTE_METADATA_BYTES = 1024 * 1024
_MAX_REMOTE_PHASE_BYTES = 16 * 1024 * 1024
_SERVER_SOCKET_TIMEOUT = 10.0


def _shape(shape_yx: object) -> tuple[int, int]:
    if (
        not isinstance(shape_yx, tuple)
        or len(shape_yx) != 2
        or any(type(value) is not int or value <= 0 for value in shape_yx)
    ):
        raise TypeError("SLM shape_yx must be a two-positive-integer tuple")
    return shape_yx


def _remote_phase_bytes(shape_yx: object) -> int:
    shape = _shape(tuple(shape_yx))
    size = shape[0] * shape[1] * np.dtype("<f4").itemsize
    if size > _MAX_REMOTE_PHASE_BYTES:
        raise ValueError("SLM shape exceeds the remote phase payload bound")
    return size


def canonical_phase(radians: object, shape_yx: tuple[int, int]) -> np.ndarray:
    """Return one immutable owned phase snapshot in wrapped float32 radians."""

    shape = _shape(shape_yx)
    source = np.asarray(radians)
    if source.shape != shape:
        raise ValueError(
            f"SLM phase shape {source.shape!r} differs from device shape {shape!r}"
        )
    if source.dtype.kind not in "iuf":
        raise TypeError("SLM phase must contain real numeric values")
    values = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("SLM phase must contain only finite values")
    wrapped = np.asarray(np.remainder(values, _TWO_PI), dtype=np.float32)
    # float32(2*pi) rounds above the mathematical upper bound.  Clamp that
    # single rounding case so the public interval stays strictly [0, 2*pi)
    # and canonicalizing an already-canonical snapshot is idempotent.
    wrapped = np.minimum(wrapped, _MAX_WRAPPED_PHASE)
    # An ndarray backed by immutable bytes cannot be made writable again by a
    # caller, unlike an owning array with only its WRITEABLE flag cleared.
    return np.frombuffer(
        np.ascontiguousarray(wrapped).tobytes(),
        dtype=np.float32,
    ).reshape(shape)


@runtime_checkable
class SlmAdapter(Protocol):
    """The complete device-independent surface of one phase-only SLM."""

    @property
    def identity(self) -> str: ...

    @property
    def shape_yx(self) -> tuple[int, int]: ...

    def apply_phase(self, radians: object) -> np.ndarray: ...

    @property
    def last_commanded_phase(self) -> np.ndarray | None: ...

    @property
    def command_revision(self) -> int: ...

    @property
    def mapping_revision(self) -> int: ...

    @property
    def last_command_receipt(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


def _validated_state(
    identity: object,
    shape_yx: object,
    phase: object,
    command_revision: object,
    mapping_revision: object,
    receipt: object,
) -> tuple[str, tuple[int, int], np.ndarray | None, int, int, dict[str, object]]:
    if (
        not isinstance(identity, str)
        or not identity.strip()
        or identity != identity.strip()
    ):
        raise ValueError("SLM identity must be non-empty text without surrounding space")
    shape = _shape(tuple(shape_yx))
    if type(command_revision) is not int or command_revision < 0:
        raise ValueError("SLM command_revision must be a non-negative integer")
    if type(mapping_revision) is not int or mapping_revision < 0:
        raise ValueError("SLM mapping_revision must be a non-negative integer")
    if not isinstance(receipt, Mapping):
        raise TypeError("SLM command receipt must be a mapping")
    frozen_receipt = dict(receipt)
    if frozen_receipt.get("outcome") not in {"known-old", "known-new", "unknown"}:
        raise ValueError("SLM command receipt has an invalid outcome")
    if frozen_receipt.get("identity") != identity:
        raise ValueError("SLM command receipt identity differs from device truth")
    if (
        type(frozen_receipt.get("command_revision")) is not int
        or frozen_receipt["command_revision"] != command_revision
    ):
        raise ValueError("SLM command receipt revision differs from device truth")
    if (
        type(frozen_receipt.get("mapping_revision")) is not int
        or frozen_receipt["mapping_revision"] != mapping_revision
    ):
        raise ValueError("SLM command receipt mapping differs from device truth")
    canonical = None if phase is None else canonical_phase(phase, shape)
    if phase is not None and (
        np.asarray(phase).flags.writeable or not np.array_equal(phase, canonical)
    ):
        raise ValueError("SLM last_commanded_phase must be an immutable canonical snapshot")
    if (canonical is None) != (frozen_receipt["outcome"] == "unknown"):
        raise ValueError("SLM phase knowledge differs from its command receipt")
    return identity, shape, canonical, command_revision, mapping_revision, frozen_receipt


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise ConnectionError("SLM remote connection closed mid-message")
        result.extend(chunk)
    return bytes(result)


def _send_packet(
    connection: socket.socket, metadata: Mapping[str, object], payload: bytes = b""
) -> None:
    encoded = json.dumps(
        dict(metadata), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > _MAX_REMOTE_METADATA_BYTES or len(payload) > _MAX_REMOTE_PHASE_BYTES:
        raise ValueError("SLM remote message exceeds the maximum size")
    connection.sendall(_REMOTE_HEADER.pack(len(encoded), len(payload)) + encoded + payload)


def _recv_packet(connection: socket.socket) -> tuple[dict[str, object], bytes]:
    metadata_size, payload_size = _REMOTE_HEADER.unpack(
        _recv_exact(connection, _REMOTE_HEADER.size)
    )
    if metadata_size > _MAX_REMOTE_METADATA_BYTES or payload_size > _MAX_REMOTE_PHASE_BYTES:
        raise ValueError("SLM remote message exceeds the maximum size")

    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate SLM remote field {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite SLM remote value {value!r}")

    decoded = json.loads(
        _recv_exact(connection, metadata_size).decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_constant,
    )
    if not isinstance(decoded, dict):
        raise TypeError("SLM remote metadata must be an object")
    return decoded, _recv_exact(connection, payload_size)


def _open_slm_server(slm: SlmAdapter, host: str, port: int) -> socketserver.TCPServer:
    """Return the one-command-at-a-time RPC owner for an existing SLM."""

    if not isinstance(slm, SlmAdapter):
        raise TypeError("SLM server requires a canonical SlmAdapter")
    _remote_phase_bytes(slm.shape_yx)
    bind_host = str(host).strip()
    if not bind_host or bind_host != host or any(char.isspace() for char in bind_host):
        raise ValueError("SLM server host must be non-empty text without whitespace")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("SLM server port must be an integer from 0 through 65535")

    def response(ok: bool, error: str | None, *, include_phase: bool):
        phase = slm.last_commanded_phase
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
        return {"version": _REMOTE_VERSION, "ok": ok, "error": error, "state": state}, payload

    def handle(connection: socket.socket, address, _server) -> None:
        connection.settimeout(_SERVER_SOCKET_TIMEOUT)
        client = f"{address[0]}:{address[1]}" if address else "?"
        try:
            request, payload = _recv_packet(connection)
        except Exception as error:
            _LOG.info(
                "SLM RECEIVE FAILED client=%s error=%s: %s",
                client, type(error).__name__, error,
            )
            return
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
        elif request.get("method") != "apply" or fields != {
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
            or len(payload) != _remote_phase_bytes(slm.shape_yx)
        ):
            reply = response(False, "invalid SLM phase payload", include_phase=True)
        else:
            try:
                slm.apply_phase(
                    np.frombuffer(payload, dtype="<f4").reshape(slm.shape_yx)
                )
            except Exception as error:
                reply = response(
                    False, f"{type(error).__name__}: {error}", include_phase=True
                )
            else:
                reply = response(True, None, include_phase=False)
        metadata = reply[0]
        _LOG.info(
            "SLM %s client=%s ok=%s%s command_revision=%s",
            str(request.get("method", "?")).upper(),
            client,
            metadata["ok"],
            "" if metadata["error"] is None else f" error={metadata['error']!r}",
            metadata["state"]["command_revision"],
        )
        try:
            _send_packet(connection, *reply)
        except OSError:
            pass

    server = socketserver.TCPServer((bind_host, port), handle)
    _LOG.info(
        "SLM SERVER LISTENING endpoint=%s:%d device=%s",
        bind_host, int(server.server_address[1]), slm.identity,
    )
    return server


def _rpc_call(
    endpoint: tuple[str, int], method: str, arguments: tuple[object, ...], timeout: float
) -> tuple[dict[str, object], bytes]:
    if method == "describe" and not arguments:
        metadata, payload = {"version": _REMOTE_VERSION, "method": method}, b""
    elif method == "apply" and len(arguments) == 4:
        command_revision, mapping_revision, shape_yx, payload = arguments
        metadata = {
            "version": _REMOTE_VERSION,
            "method": method,
            "command_revision": command_revision,
            "mapping_revision": mapping_revision,
            "shape_yx": shape_yx,
        }
        payload = bytes(payload)
    else:
        raise ValueError("invalid local SLM remote call")
    with socket.create_connection(endpoint, timeout=timeout) as connection:
        connection.settimeout(timeout)
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
        self._lock = Lock()
        self._identity = ""
        self._shape_yx = (1, 1)
        self._command_revision = 0
        self._mapping_revision = 0
        self._phase: np.ndarray | None = None
        self._receipt: dict[str, object] = {}
        self._uncertain = False
        self._closed = False
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
            phase = None if commanded is None else canonical_phase(commanded, shape)
        identity, shape, phase, command_revision, mapping_revision, receipt = (
            _validated_state(
                identity, shape, phase, command_revision, mapping_revision, receipt
            )
        )
        if self._identity and (identity != self._identity or shape != self._shape_yx):
            raise RuntimeError("SLM remote endpoint changed physical identity or shape")
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
    ) -> str | None:
        value, payload = _rpc_call(self._endpoint, method, arguments, self._timeout)
        if not isinstance(value, dict) or set(value) != {"version", "ok", "error", "state"}:
            raise ValueError("SLM remote response has an invalid field set")
        if (
            type(value["version"]) is not int
            or value["version"] != _REMOTE_VERSION
            or type(value["ok"]) is not bool
        ):
            raise ValueError("SLM remote response has an invalid protocol version")
        self._accept_state(
            value["state"], payload, commanded=commanded if value["ok"] else None
        )
        if not value["ok"]:
            if not isinstance(value["error"], str) or not value["error"]:
                raise ValueError("SLM remote error is missing its message")
            return value["error"]
        if value["error"] is not None:
            raise ValueError("successful SLM remote response contains an error")
        return None

    def _describe(self) -> None:
        error = self._request("describe")
        if error is not None:
            raise RuntimeError(error)

    def _mark_unknown(self) -> None:
        self._phase = None
        self._receipt = {
            **self._receipt,
            "outcome": "unknown",
            "stage": "remote-transport",
            "readback": "not-run",
        }
        self._uncertain = True

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self._shape_yx

    @property
    def last_commanded_phase(self) -> np.ndarray | None:
        with self._lock:
            return self._phase

    @property
    def command_revision(self) -> int:
        with self._lock:
            return self._command_revision

    @property
    def mapping_revision(self) -> int:
        with self._lock:
            return self._mapping_revision

    @property
    def last_command_receipt(self) -> Mapping[str, object]:
        with self._lock:
            return dict(self._receipt)

    def apply_phase(self, radians: object) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("remote SLM is closed")
            if self._uncertain:
                self._describe()
            canonical = canonical_phase(radians, self._shape_yx)
            expected_command = self._command_revision
            expected_mapping = self._mapping_revision
            try:
                error = self._request(
                    "apply",
                    (
                        expected_command,
                        expected_mapping,
                        list(self._shape_yx),
                        np.asarray(canonical, dtype="<f4").tobytes(),
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

    def close(self) -> None:
        with self._lock:
            self._closed = True


def bind_slm(
    context: object,
    key: str,
    slm: SlmAdapter,
    type_id: str,
) -> InstalledLeaf:
    """Bind one adapter through the installation's existing device broker."""

    from zlc_atom.execution import (
        DeviceIdentityEvidenceKind,
        PhysicalDeviceIdentity,
        ResourceKey,
        bind_verified_device,
    )
    from zlc_atom.install.descriptors import InstalledLeaf

    if not isinstance(slm, SlmAdapter):
        raise TypeError("slm must implement the canonical SlmAdapter contract")
    try:
        identity, _shape_yx, _phase, _command, _mapping, _receipt = _validated_state(
            slm.identity,
            slm.shape_yx,
            slm.last_commanded_phase,
            slm.command_revision,
            slm.mapping_revision,
            slm.last_command_receipt,
        )
        binding, proof = bind_verified_device(
            context.broker,
            key=ResourceKey.parse(f"device/{key}"),
            identity_probe=lambda: PhysicalDeviceIdentity(
                identity,
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            ),
            capability_probe=lambda: {"slm.phase": slm},
        )
    except BaseException:
        slm.close()
        raise
    return InstalledLeaf(
        key,
        type_id,
        slm,
        dict(proof.snapshot),
        binding=binding,
        closer=slm.close,
    )


__all__ = ["SlmAdapter", "bind_slm", "canonical_phase"]
