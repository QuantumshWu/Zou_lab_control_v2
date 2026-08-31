"""The remote device fabric: discovery, identity, and one tunable data plane.

Two machines, one bench.  The instruments live on the machine beside them
(PC2); the operator works on another (PC1).  Today every remote device is a
hand-rolled pair -- its own server started by its own .bat, its own client,
its own host/port typed into a form -- and adding a third device means
writing a fourth server.

The fabric deliberately does NOT replace those data planes.  The pulse
server's single-owner SAFE-gated protocol and the SLM's revision-checked
one exist for reasons the devices own, and a generic layer that tried to
absorb them would be a third protocol pretending to be a generalization.
What every remote device SHARES is only this:

* it must be FINDABLE -- one UDP broadcast from PC1 answers "who is
  publishing devices on this bench?", no addresses typed;
* it must be NAMEABLE -- each published device is an announce record
  (instance, role, type, and the authoring parameters a PC1 device manager
  needs to connect), so "scan hardware" on PC1 lists PC2's devices as
  one-click adds;
* and a device with no protocol of its own -- an RF synthesizer, anything
  that speaks the tunable quartet -- gets the fabric's ONE generic data
  plane: fields / tune / values / provenance over the same socket.

Wire format: length-prefixed JSON, one request per connection (the SLM
server's shape -- no request ids, no session state, nothing to resynchronize
after a dropped frame).  The UDP responder answers the broadcast with the
TCP port; everything else is TCP.
"""

from __future__ import annotations

import json
import logging
import socket
import socketserver
import struct
import threading
from typing import Any, Callable, Mapping

FABRIC_VERSION = 1
DEFAULT_FABRIC_PORT = 18859
PROBE_MESSAGE = b"zlc-device-fabric?"
_HEADER = struct.Struct("!I")
MAX_FRAME_BYTES = 1 << 20  # 1 MiB: announce records and scalar tunes, not data.
_REQUEST_TIMEOUT_SECONDS = 10.0


def local_lan_ip() -> str:
    """This machine's LAN address, the way the pulse and SLM servers learn it."""

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def _send_frame(connection: socket.socket, value: Any) -> None:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("fabric frame exceeds the maximum size")
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_frame(connection: socket.socket) -> Any:
    header = b""
    while len(header) < _HEADER.size:
        chunk = connection.recv(_HEADER.size - len(header))
        if not chunk:
            raise ConnectionError("fabric connection closed before the frame")
        header += chunk
    (size,) = _HEADER.unpack(header)
    if size > MAX_FRAME_BYTES:
        raise ValueError("fabric frame exceeds the maximum size")
    payload = b""
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("fabric connection closed mid-frame")
        payload += chunk
    return json.loads(payload.decode("utf-8"))


# --------------------------------------------------------------- announcing
class PublishedDevice:
    """One device this machine offers, and how a peer should reach it."""

    def __init__(
        self,
        *,
        instance_id: str,
        role: str,
        type_id: str,
        parameters: Mapping[str, Any],
        tunable: object | None = None,
    ) -> None:
        self.instance_id = str(instance_id)
        self.role = str(role)
        self.type_id = str(type_id)
        self.parameters = dict(parameters)
        #: The live device object, when it speaks the tunable quartet --
        #: that is what the generic data plane serves.  None for devices
        #: with their own server (pulse, SLM): their record alone is enough,
        #: because the parameters already say where that server is.
        self.tunable = tunable
        self.lock = threading.Lock()

    def record(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "role": self.role,
            "type_id": self.type_id,
            "parameters": dict(self.parameters),
            "tunable": self.tunable is not None,
        }


#: The fabric's narration channel, shown by the bench window that owns
#: the announcer so the serving machine can watch its published devices.
_LOG = logging.getLogger(__name__)


class DeviceAnnouncer:
    """PC2's half: answer the broadcast, list the published, serve the tunes."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = DEFAULT_FABRIC_PORT) -> None:
        self._published: dict[str, PublishedDevice] = {}
        self._registry_lock = threading.Lock()

        announcer = self

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:  # one request per connection
                try:
                    request = _recv_frame(self.request)
                    response = announcer._dispatch(request)
                except Exception as error:  # noqa: BLE001 -- answered, not fatal
                    response = {
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    }
                try:
                    _send_frame(self.request, response)
                except OSError:
                    pass

        class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = _Server((host, int(port)), _Handler)
        self.port = int(self._server.server_address[1])
        self._tcp_thread = threading.Thread(
            target=self._server.serve_forever,
            name="zlc-fabric-tcp",
            daemon=True,
        )
        self._tcp_thread.start()

        # The UDP responder is what makes "no addresses typed" true: PC1
        # broadcasts one datagram, every announcer on the subnet answers
        # with its TCP port.
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp.bind((host, self.port))
        self._udp_thread = threading.Thread(
            target=self._answer_probes,
            name="zlc-fabric-udp",
            daemon=True,
        )
        self._udp_thread.start()

    # ------------------------------------------------------------- publish
    def publish(self, device: PublishedDevice) -> None:
        with self._registry_lock:
            self._published[device.instance_id] = device
        _LOG.info(
            "FABRIC PUBLISH device=%s type=%s plane=%s",
            device.instance_id,
            device.type_id,
            "tunable" if device.tunable is not None else "own protocol",
        )

    def withdraw(self, instance_id: str) -> None:
        with self._registry_lock:
            known = self._published.pop(str(instance_id), None)
        if known is not None:
            _LOG.info("FABRIC WITHDRAW device=%s", instance_id)

    def published_ids(self) -> tuple[str, ...]:
        with self._registry_lock:
            return tuple(sorted(self._published))

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        try:
            self._udp.close()
        except OSError:
            pass

    # ------------------------------------------------------------ serving
    def _answer_probes(self) -> None:
        while True:
            try:
                message, sender = self._udp.recvfrom(256)
            except OSError:
                return
            if message != PROBE_MESSAGE:
                continue
            try:
                self._udp.sendto(
                    json.dumps(
                        {"fabric": FABRIC_VERSION, "port": self.port}
                    ).encode("utf-8"),
                    sender,
                )
            except OSError:
                continue

    def _device(self, request: Mapping[str, Any]) -> PublishedDevice:
        instance = str(request.get("instance", ""))
        with self._registry_lock:
            device = self._published.get(instance)
        if device is None:
            raise LookupError(f"no published device {instance!r}")
        return device

    def _dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("fabric request must be an object")
        method = str(request.get("method", ""))
        if method == "list":
            with self._registry_lock:
                records = [
                    self._published[key].record()
                    for key in sorted(self._published)
                ]
            return {"fabric": FABRIC_VERSION, "devices": records}
        device = self._device(request)
        if device.tunable is None:
            raise TypeError(
                f"{device.instance_id!r} is served by its own protocol; the "
                "fabric only lists it"
            )
        if method == "fields":
            with device.lock:
                fields = device.tunable.tunable_fields()
            return {
                "fields": [
                    {
                        "name": field.metadata.name,
                        "value_type": field.metadata.value_type,
                        "label": field.metadata.label,
                        "default": field.metadata.default,
                        "minimum": field.metadata.minimum,
                        "maximum": field.metadata.maximum,
                        "unit": field.metadata.unit,
                        "current": field.current,
                        "live_write": field.live_write,
                        "dependency_group": list(field.dependency_group),
                    }
                    for field in fields
                ]
            }
        if method == "tune":
            name = str(request.get("name", ""))
            value = request.get("value")
            try:
                with device.lock:
                    effective = device.tunable.tune(name, value)
            except Exception as error:
                _LOG.info(
                    "FABRIC TUNE REFUSED device=%s field=%s value=%r error=%s: %s",
                    device.instance_id, name, value, type(error).__name__, error,
                )
                raise
            _LOG.info(
                "FABRIC TUNE device=%s field=%s value=%r effective=%r",
                device.instance_id, name, value, effective,
            )
            return {"effective": effective}
        if method == "values":
            with device.lock:
                return {"values": dict(device.tunable.tunable_values())}
        if method == "provenance":
            with device.lock:
                return {
                    "provenance": dict(device.tunable.settings_provenance())
                }
        raise ValueError(f"unknown fabric method {method!r}")


# --------------------------------------------------------------- consuming
def _call(host: str, port: int, request: Mapping[str, Any]) -> dict[str, Any]:
    with socket.create_connection(
        (host, int(port)), timeout=_REQUEST_TIMEOUT_SECONDS
    ) as connection:
        connection.settimeout(_REQUEST_TIMEOUT_SECONDS)
        _send_frame(connection, dict(request))
        response = _recv_frame(connection)
    if not isinstance(response, Mapping):
        raise TypeError("fabric response must be an object")
    error = response.get("error")
    if error is not None:
        raise RuntimeError(
            f"{error.get('type', 'Error')}: {error.get('message', '')}"
        )
    return dict(response)


def discover_announcers(
    *,
    timeout_seconds: float = 1.0,
    port: int = DEFAULT_FABRIC_PORT,
    extra_hosts: tuple[str, ...] = (),
) -> tuple[tuple[str, int], ...]:
    """Every fabric on the subnet, by one broadcast -- plus any named peers.

    ``extra_hosts`` is for the bench whose machines sit on different
    subnets, where a broadcast cannot reach: name the peer once in
    configuration and it is probed directly, same protocol.
    """

    found: dict[tuple[str, int], None] = {}
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        probe.settimeout(0.2)
        targets = [("255.255.255.255", int(port))] + [
            (str(host), int(port)) for host in extra_hosts
        ]
        for target in targets:
            try:
                probe.sendto(PROBE_MESSAGE, target)
            except OSError:
                continue
        import time

        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            try:
                message, sender = probe.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                answer = json.loads(message.decode("utf-8"))
                found[(str(sender[0]), int(answer["port"]))] = None
            except (ValueError, KeyError, TypeError):
                continue
    finally:
        probe.close()
    return tuple(found)


def list_remote_devices(host: str, port: int) -> tuple[dict[str, Any], ...]:
    response = _call(host, port, {"method": "list"})
    devices = response.get("devices")
    if not isinstance(devices, list):
        raise TypeError("fabric list must contain devices")
    return tuple(dict(record) for record in devices)


class RemoteTunableDevice:
    """PC1's handle on a fabric-served knob: the tunable quartet, remotely.

    It speaks exactly what every local tunable device speaks, so the scan
    axis combo, the generic control panel and the device-axis executor use
    it without knowing it is remote.  Field metadata is fetched once at
    open -- device facts, stable for the session -- while values, tunes and
    provenance go to the wire every time, because the truth lives on the
    other machine.
    """

    def __init__(self, *, host: str, port: int, instance_id: str) -> None:
        from zlc_atom.authoring import AuthoringField, TunableField

        self._host = str(host)
        self._port = int(port)
        self._instance = str(instance_id)
        #: How this proxy's OWN log lines are tagged on the consuming bench;
        #: the serving machine tags the same actions with its instance id.
        self.identity = f"fabric:{self._instance}@{self._host}:{self._port}"
        described = _call(
            self._host,
            self._port,
            {"method": "fields", "instance": self._instance},
        )["fields"]
        fields = []
        for entry in described:
            fields.append(
                (
                    AuthoringField(
                        str(entry["name"]),
                        str(entry["value_type"]),
                        str(entry["label"]),
                        entry.get("default"),
                        minimum=entry.get("minimum"),
                        maximum=entry.get("maximum"),
                        unit=entry.get("unit"),
                    ),
                    bool(entry["live_write"]),
                    tuple(str(name) for name in entry["dependency_group"]),
                )
            )
        self._field_shapes = tuple(fields)
        self._tunable_field = TunableField

    def _call(self, method: str, **extra: Any) -> dict[str, Any]:
        return _call(
            self._host,
            self._port,
            {"method": method, "instance": self._instance, **extra},
        )

    def tunable_fields(self):
        current = self.tunable_values()
        return tuple(
            self._tunable_field(
                metadata=metadata,
                current=current.get(metadata.name, metadata.default),
                live_write=live_write,
                dependency_group=group,
            )
            for metadata, live_write, group in self._field_shapes
        )

    def tune(self, name: str, value: Any) -> Any:
        try:
            effective = self._call("tune", name=str(name), value=value)[
                "effective"
            ]
        except Exception as error:
            _LOG.info(
                "TUNE REFUSED field=%s value=%r error=%s: %s -- device=%s",
                name,
                value,
                type(error).__name__,
                str(error).replace(chr(10), " "),
                self.identity,
            )
            raise
        _LOG.info(
            "TUNE field=%s value=%r effective=%r device=%s",
            name,
            value,
            effective,
            self.identity,
        )
        return effective

    def tunable_values(self) -> dict[str, Any]:
        return dict(self._call("values")["values"])

    def settings_provenance(self) -> dict[str, Any]:
        return dict(self._call("provenance")["provenance"])

    def close(self) -> None:
        """Closing the handle closes nothing remote: PC2 owns its device."""


__all__ = [
    "DEFAULT_FABRIC_PORT",
    "DeviceAnnouncer",
    "FABRIC_VERSION",
    "PROBE_MESSAGE",
    "PublishedDevice",
    "RemoteTunableDevice",
    "discover_announcers",
    "list_remote_devices",
    "local_lan_ip",
]
