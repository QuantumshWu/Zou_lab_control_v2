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

Wire format: length-prefixed JSON, serialized requests on an owned connection.
Every request and every answer carries ``FABRIC_VERSION``, and either side
refuses a peer of another version by name.
A broken connection fails its current request without replaying a write; the
next request dials again.
The UDP responder answers the broadcast with its version and the TCP port, and
a scan skips an announcer of another version by name; everything else is TCP.
"""

from __future__ import annotations

#: What an installation calls a device another machine published.  The type
#: id is the FABRIC's word, not one device folder's: the workbench reads it
#: to tell a remote row from a local one, and it has to keep meaning that
#: whether or not this bench installs remote tunables at all.
FABRIC_TUNABLE_TYPE = "remote.tunable"

import json
import logging
import socket
import socketserver
import struct
import threading
import time
from typing import Any, Mapping

from zlc_durable import strict_json_loads
from zlc_pulse.endpoint import bind_exclusive

#: The wire vocabulary.  It moves with the vocabulary -- a new method, a
#: changed key -- and each side refuses a peer that speaks another, naming
#: both: PC1 and PC2 updated at different times otherwise met as an
#: "unknown fabric method" halfway through an editor, or a missing key.
FABRIC_VERSION = 2
DEFAULT_FABRIC_PORT = 18859
PROBE_MESSAGE = b"zlc-device-fabric?"
_HEADER = struct.Struct("!I")
MAX_FRAME_BYTES = 1 << 20  # 1 MiB: announce records and scalar tunes, not data.
_REQUEST_TIMEOUT_SECONDS = 10.0
#: A tune answers only once the device has settled, and some take far longer
#: than a question: an N100 rate change is six console writes 1.2 s apart, a
#: restart and a re-timing of its stream, and undoing one takes longer still.
_TUNE_TIMEOUT_SECONDS = 120.0
#: A named peer whose probe took longer than this was resolved only by a
#: fallback, after a resolver that never answered: the scan names it with
#: its seconds, which came out of the time to list the announcers, so its
#: address can be written instead.
_SLOW_PEER_SECONDS = 1.0


def _version_skew(peer: str, spoken: object) -> str:
    """Why ``peer``, speaking fabric version ``spoken``, is refused.

    Version 1 stamped only its broadcast reply and its list answer, so a v1
    peer's other words carry no version at all: that is a v1 peer, not a
    peer of version None.
    """

    said = (
        "is unversioned (fabric v1)"
        if spoken is None
        else f"speaks fabric version {spoken!r}"
    )
    return (
        f"{peer} {said} and this machine speaks {FABRIC_VERSION}; "
        "update the older side and restart it"
    )


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
    return strict_json_loads(payload.decode("utf-8"), "fabric frame")


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
        self._connections: set[socket.socket] = set()

        announcer = self

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                while True:
                    try:
                        request = _recv_frame(self.request)
                    except (OSError, ValueError):
                        return
                    try:
                        response = announcer._dispatch(request)
                    except Exception as error:  # noqa: BLE001 -- answered, not fatal
                        response = {
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        }
                    response = {"fabric": FABRIC_VERSION, **response}
                    try:
                        _send_frame(self.request, response)
                    except OSError:
                        return

        class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
            daemon_threads = False

            def server_bind(self):
                # A second announcer on this port fails here, where the
                # presenter reports it, instead of announcing into the void.
                bind_exclusive(self.socket, self.server_address)
                self.server_address = self.socket.getsockname()

            def get_request(self):
                connection, address = super().get_request()
                with announcer._registry_lock:
                    announcer._connections.add(connection)
                return connection, address

            def shutdown_request(self, request):
                try:
                    super().shutdown_request(request)
                finally:
                    with announcer._registry_lock:
                        announcer._connections.discard(request)

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
        try:
            bind_exclusive(self._udp, (host, self.port))
        except BaseException:
            # An announcer that cannot answer the broadcast is none, and its
            # listener must not outlive it: the next Remote-on binds anew.
            self._udp.close()
            self._server.shutdown()
            self._server.server_close()
            raise
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
        """Take one device off offer, and back from any peer using it.

        Returns only once a peer's request that found the device before it
        was withdrawn is finished with it -- a tune can take minutes -- so
        the machine it is handed back to never has a peer's write land
        after its own run began, or a close under it.
        """

        with self._registry_lock:
            known = self._published.pop(str(instance_id), None)
        if known is not None:
            with known.lock:
                pass
            _LOG.info("FABRIC WITHDRAW device=%s", instance_id)

    def close(self) -> None:
        self._server.shutdown()
        with self._registry_lock:
            connections = tuple(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self._server.server_close()
        try:
            self._udp.close()
        except OSError:
            pass

    # ------------------------------------------------------------ serving
    def _answer_probes(self) -> None:
        failing = False
        while True:
            try:
                message, sender = self._udp.recvfrom(256)
            except ConnectionResetError:
                # Windows hands an earlier answer's ICMP port-unreachable (a
                # scanner that had already closed) to the NEXT read as a
                # reset: news of one datagram, not of this socket.
                continue
            except OSError as error:
                if self._udp.fileno() == -1:
                    # Closed: the one way this responder ends.
                    return
                # Anything else fails a read, not the responder: Windows
                # refuses a datagram longer than the buffer instead of
                # truncating it, and ending here left the machine published
                # and unfindable until Remote was switched off and on.  An
                # error every read repeats (the network stack gone) is said
                # once and retried at a walk, not spun on at a full core.
                if not failing:
                    _LOG.warning(
                        "FABRIC PROBE READ FAILED error=%s: %s -- still answering",
                        type(error).__name__, error,
                    )
                failing = True
                time.sleep(0.1)
                continue
            failing = False
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
        spoken = request.get("fabric")
        if spoken != FABRIC_VERSION:
            raise ValueError(_version_skew("the peer", spoken))
        method = str(request.get("method", ""))
        if method == "list":
            with self._registry_lock:
                records = [
                    self._published[key].record()
                    for key in sorted(self._published)
                ]
            return {"devices": records}
        device = self._device(request)
        if device.tunable is None:
            raise TypeError(
                f"{device.instance_id!r} is served by its own protocol; the "
                "fabric only lists it"
            )
        with device.lock:
            # Asked again under the lock ``withdraw`` waits on: a request
            # that found the device before it was withdrawn, and reached the
            # lock after, must not touch a device handed back.
            with self._registry_lock:
                if self._published.get(device.instance_id) is not device:
                    raise LookupError(f"no published device {device.instance_id!r}")
            return self._serve(device, method, request)

    def _serve(
        self, device: PublishedDevice, method: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        """One request on a published tunable, under its lock."""

        if method in {"fields", "read_tunable_in_unit"}:
            if method == "fields":
                from zlc_atom.authoring import refresh_tunable_fields

                fields = (refresh_tunable_fields(device.tunable) if request.get("refresh", False)
                          else device.tunable.tunable_fields())
            else:
                from zlc_atom.authoring import read_tunable_in_unit

                fields = (read_tunable_in_unit(
                    device.tunable, str(request.get("name", "")),
                    str(request.get("unit", "")),
                ),)
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
                        "device_limits": (
                            None
                            if field.device_limits is None
                            else list(field.device_limits)
                        ),
                    }
                    for field in fields
                ]
            }
        if method == "convert_tunable_value":
            from zlc_atom.authoring import convert_tunable_value

            value = convert_tunable_value(
                device.tunable, str(request.get("name", "")), request.get("value"),
                str(request.get("source_unit", "")), str(request.get("target_unit", "")),
            )
            return {"value": value}
        if method in {"tune", "tune_in_unit"}:
            name = str(request.get("name", ""))
            value = request.get("value")
            try:
                if method == "tune":
                    effective = device.tunable.tune(name, value)
                else:
                    from zlc_atom.authoring import tune_in_unit

                    effective = tune_in_unit(
                        device.tunable, name, value, str(request.get("unit", ""))
                    )
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
            return {"values": dict(device.tunable.tunable_values())}
        if method == "provenance":
            return {"provenance": dict(device.tunable.settings_provenance())}
        raise ValueError(f"unknown fabric method {method!r}")


# --------------------------------------------------------------- consuming
def _call(connection: socket.socket, request: Mapping[str, Any]) -> dict[str, Any]:
    _send_frame(connection, {"fabric": FABRIC_VERSION, **request})
    response = _recv_frame(connection)
    if not isinstance(response, Mapping):
        raise TypeError("fabric response must be an object")
    spoken = response.get("fabric")
    if spoken != FABRIC_VERSION:
        # Before anything else it says is read: a peer of another version may
        # mean something else by the same words, its errors included.
        raise ConnectionError(_version_skew("the peer", spoken))
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
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    """Every fabric on the subnet, by one broadcast -- plus any named peers.

    ``extra_hosts`` is for the bench whose machines sit on different
    subnets, where a broadcast cannot reach: name the peer once in
    configuration and it is probed directly, same protocol.

    Returns the announcers to list, and why each other one that answered is
    not listed.  An announcer of another version is not: every answer it
    gave would be refused, and the first refusal ended the whole scan,
    every other announcer's devices with it.  It is named instead, for the
    scan to report beside what the others publish.  So is a named peer that
    could not be probed, or whose probe took more than
    ``_SLOW_PEER_SECONDS``, with the seconds it took: its name is resolved
    as it is probed, seconds for one no resolver answers or only a fallback
    does, and those seconds come out of the scan's time to list the
    announcers.
    """

    found: dict[tuple[str, int], None] = {}
    skewed: dict[tuple[str, int], object] = {}
    peer_notes: list[str] = []
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        probe.settimeout(0.2)
        try:
            probe.sendto(PROBE_MESSAGE, ("255.255.255.255", int(port)))
        except OSError:
            pass
        for host in extra_hosts:
            began = time.monotonic()
            try:
                probe.sendto(PROBE_MESSAGE, (str(host), int(port)))
            except (OSError, TypeError) as error:
                # A name the resolver cannot even encode (a copied
                # left-to-right mark, an empty label) is a TypeError here.
                peer_notes.append(
                    f"the named peer {host!r} was not probed: {error} "
                    f"({time.monotonic() - began:.1f}s)"
                )
                continue
            took = time.monotonic() - began
            if took > _SLOW_PEER_SECONDS:
                peer_notes.append(
                    f"the named peer {host!r} took {took:.1f}s to resolve; "
                    "write its address"
                )
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            try:
                # Any datagram fits: Windows refuses one longer than the
                # buffer instead of truncating it, and that ended the scan.
                message, sender = probe.recvfrom(65535)
            except (socket.timeout, ConnectionResetError):
                # A reset is Windows reporting a probe that reached a named
                # peer where nothing listens -- an earlier send, not this read.
                # Stopping there closed the port every live announcer was
                # still answering.
                continue
            except OSError:
                break
            try:
                answer = strict_json_loads(message.decode("utf-8"), "fabric announcement")
                announcer = (str(sender[0]), int(answer["port"]))
                spoken = answer.get("fabric")
            except (ValueError, KeyError, TypeError):
                continue
            if spoken == FABRIC_VERSION:
                found[announcer] = None
            else:
                skewed[announcer] = spoken
    finally:
        probe.close()
    return tuple(found), (*peer_notes, *(
        _version_skew(f"the announcer at {host}:{port}", spoken)
        for (host, port), spoken in skewed.items()
    ))


def list_remote_devices(host: str, port: int) -> tuple[dict[str, Any], ...]:
    with socket.create_connection((host, int(port)), timeout=_REQUEST_TIMEOUT_SECONDS) as connection:
        response = _call(connection, {"method": "list"})
    devices = response.get("devices")
    if not isinstance(devices, list):
        raise TypeError("fabric list must contain devices")
    return tuple(dict(record) for record in devices)


class RemoteTunableDevice:
    """PC1's handle on a fabric-served knob: the tunable quartet, remotely.

    It speaks exactly what every local tunable device speaks, so the scan
    axis combo, the generic control panel and the device-axis executor use
    it without knowing it is remote.  The connection is reused, but device
    fields are not remembered between calls:
    fields, values, tunes and provenance all go to the wire every time,
    because the truth lives on the other machine -- and a field's bounds
    are part of that truth.  An RF source moves its commandable window
    when a policy edge is tuned, so a Refresh that kept the bounds seen at
    open would keep offering Control and Scan a range the device no longer
    accepts, until the proxy was rebuilt.
    """

    def __init__(self, *, host: str, port: int, instance_id: str) -> None:
        self._host = str(host)
        self._port = int(port)
        self._instance = str(instance_id)
        self._io_lock = threading.RLock()
        self._connection: socket.socket | None = None
        self._closed = False
        #: How this proxy's OWN log lines are tagged on the consuming bench;
        #: the serving machine tags the same actions with its instance id.
        self.identity = f"fabric:{self._instance}@{self._host}:{self._port}"
        # Opening is asking: the record must exist, and be one the fabric's
        # generic plane serves -- a device with its own protocol refuses
        # here, by name, rather than at the first tune.
        try:
            self.tunable_fields()
        except BaseException:
            self.close()
            raise

    def _call(
        self, method: str, *, timeout: float = _REQUEST_TIMEOUT_SECONDS, **extra: Any
    ) -> dict[str, Any]:
        with self._io_lock:
            if self._closed:
                raise ConnectionError("remote tunable connection is closed")
            if self._connection is None:
                # The request that broke the last connection is never sent
                # again -- it may have been a write the device already made
                # -- but the device itself is still there: the next request
                # dials it anew.
                self._connection = socket.create_connection(
                    (self._host, self._port), timeout=_REQUEST_TIMEOUT_SECONDS
                )
            self._connection.settimeout(timeout)
            try:
                return _call(self._connection,
                             {"method": method, "instance": self._instance, **extra})
            except (OSError, ValueError, TypeError):
                self._drop_connection()
                raise

    def tunable_fields(self):
        return self._read_fields("fields")

    def refresh_tunable_fields(self):
        return self._read_fields("fields", refresh=True)

    def read_tunable_in_unit(self, name: str, unit: str = ""):
        return self._read_fields("read_tunable_in_unit", name=str(name), unit=str(unit))[0]

    def _read_fields(self, method: str, **arguments: Any):
        from zlc_atom.authoring import AuthoringField, TunableField

        return tuple(
            TunableField(
                metadata=AuthoringField(
                    str(entry["name"]),
                    str(entry["value_type"]),
                    str(entry["label"]),
                    entry.get("default"),
                    minimum=entry.get("minimum"),
                    maximum=entry.get("maximum"),
                    unit=entry.get("unit"),
                ),
                current=entry.get("current"),
                live_write=bool(entry["live_write"]),
                dependency_group=tuple(
                    str(name) for name in entry["dependency_group"]
                ),
                device_limits=(
                    None
                    if entry.get("device_limits") is None
                    else tuple(float(edge) for edge in entry["device_limits"])
                ),
            )
            for entry in self._call(method, **arguments)["fields"]
        )

    def tune_in_unit(self, name: str, value: Any, unit: str) -> Any:
        return self.tune(name, value, unit=unit)

    def convert_tunable_value(self, name: str, value: Any, source_unit: str, target_unit: str):
        converted = self._call("convert_tunable_value", name=str(name), value=value,
                               source_unit=source_unit, target_unit=target_unit)["value"]
        return tuple(float(item) for item in converted) if isinstance(converted, (tuple, list)) else float(converted)

    def tune(self, name: str, value: Any, *, unit: str | None = None) -> Any:
        try:
            effective = self._call(
                "tune" if unit is None else "tune_in_unit",
                timeout=_TUNE_TIMEOUT_SECONDS,
                name=str(name), value=value, **({} if unit is None else {"unit": unit}),
            )[
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

    def _drop_connection(self) -> None:
        with self._io_lock:
            connection, self._connection = self._connection, None
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()

    def close(self) -> None:
        """Closing the handle closes nothing remote: PC2 owns its device."""

        with self._io_lock:
            self._closed = True
            self._drop_connection()


__all__ = [
    "DEFAULT_FABRIC_PORT",
    "FABRIC_TUNABLE_TYPE",
    "DeviceAnnouncer",
    "FABRIC_VERSION",
    "PROBE_MESSAGE",
    "PublishedDevice",
    "RemoteTunableDevice",
    "discover_announcers",
    "list_remote_devices",
]
