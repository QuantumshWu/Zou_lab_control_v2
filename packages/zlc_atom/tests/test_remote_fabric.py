"""The bench fabric: publish on one machine, discover and drive from another.

Both halves run in this process over loopback -- which is exactly the
production code path, since the fabric is plain sockets with no machine
identity anywhere in it.
"""

from __future__ import annotations

import pytest

from zlc_atom.devices.remote.fabric import (
    DeviceAnnouncer,
    PublishedDevice,
    RemoteTunableDevice,
    discover_announcers,
    list_remote_devices,
)
from zlc_atom.devices.rf.contract import RfSource
from zlc_atom.devices.rf.vaunix_lms import VaunixLmsConfig
from zlc_atom.devices.simulation.rf import virtual_rf_source


@pytest.fixture
def announcer():
    fabric = DeviceAnnouncer(host="127.0.0.1", port=0)
    try:
        yield fabric
    finally:
        fabric.close()


def test_a_published_tunable_is_listed_and_driven_over_the_wire(announcer, monkeypatch) -> None:
    """The remote handle speaks the same quartet the local device does.

    It is the REAL Vaunix driver on the serving side, the generic proxy on
    the consuming side, and neither the scan axis machinery nor the control
    panel can tell the difference -- which is the entire point.
    """

    from zlc_atom.devices.remote import fabric as module

    connections = []
    original_connect = module.socket.create_connection
    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection
    monkeypatch.setattr(module.socket, "create_connection", connect)
    source = virtual_rf_source(
        VaunixLmsConfig(
            serial=1001,
            frequency_low_hz=500e6,
            frequency_high_hz=8e9,
        )
    )
    refreshes = []
    refresh_fields = source.refresh_tunable_fields
    def refresh():
        refreshes.append(True)
        return refresh_fields()
    monkeypatch.setattr(source, "refresh_tunable_fields", refresh)
    announcer.publish(
        PublishedDevice(
            instance_id="rf_main",
            role="detuning",
            type_id="rf.vaunix_lms",
            parameters={"serial": 1001},
            tunable=source,
        )
    )

    records = list_remote_devices("127.0.0.1", announcer.port)
    assert [record["instance_id"] for record in records] == ["rf_main"]
    assert records[0]["tunable"] is True

    remote = RemoteTunableDevice(
        host="127.0.0.1", port=announcer.port, instance_id="rf_main"
    )
    assert refreshes == [], "the initial metadata request uses the adapter's initialized facts"
    # The proxy satisfies the same capability contract as the local device.
    assert isinstance(remote, RfSource)

    fields = {field.metadata.name: field for field in remote.refresh_tunable_fields()}
    assert len(refreshes) == 1
    frequency = fields["frequency"].metadata
    assert (frequency.minimum, frequency.maximum) == (500e6, 8e9)
    assert frequency.unit == "Hz"

    assert remote.tune("frequency", 2.5e9) == 2.5e9
    assert source.tunable_values()["frequency"] == pytest.approx(2.5e9), (
        "the tune must have reached the machine that owns the instrument"
    )
    assert remote.tunable_values()["frequency"] == pytest.approx(2.5e9)
    assert remote.settings_provenance()["device_session_id"] == (
        source.settings_provenance()["device_session_id"]
    )

    # A refusal crosses the wire as a refusal, message intact.
    with pytest.raises(RuntimeError, match="10.*Hz grid"):
        remote.tune("frequency", 2_500_000_005.0)

    # Bounds are device truth, not open-time facts: the RF owner moves its
    # commandable window when a policy edge is tuned, and a Refresh (or the
    # scan port projection) must offer the window the device accepts NOW.
    from zlc_atom.nodes.scan.plan import scan_ports_for_devices

    assert remote.tune("frequency_low", 2e9) == 2e9
    refreshed = {field.metadata.name: field for field in remote.tunable_fields()}
    local = {field.metadata.name: field for field in source.tunable_fields()}
    assert (
        refreshed["frequency"].metadata.minimum,
        refreshed["frequency"].metadata.maximum,
    ) == (
        local["frequency"].metadata.minimum,
        local["frequency"].metadata.maximum,
    ) == (2e9, 8e9)
    assert refreshed["frequency"].current == pytest.approx(2.5e9)
    port = next(
        item
        for item in scan_ports_for_devices({"rf": remote})
        if item.port.endswith(":frequency")
    )
    assert (port.lo, port.hi) == (2e9, 8e9)
    with pytest.raises(RuntimeError, match="must lie in"):
        remote.tune("frequency", 1e9)
    assert len(connections) == 2, "one discovery connection and one persistent device connection"
    assert len(refreshes) == 1, "metadata requests do not trigger a device refresh"
    assert remote.tunable_values()["frequency"] == 2.5e9, "a refused command does not retire a healthy connection"

    # An acknowledged server-side write whose reply is lost must not be sent twice.
    original_receive = module._recv_frame
    connection = remote._connection
    def lost_reply(sock):
        response = original_receive(sock)
        if sock is connection:
            raise ConnectionError("reply lost after the device answered")
        return response
    with monkeypatch.context() as patch:
        patch.setattr(module, "_recv_frame", lost_reply)
        with pytest.raises(ConnectionError, match="reply lost"):
            remote.tune("power", -5.0)
    assert source.tunable_values()["power"] == -5.0
    assert len(connections) == 2, "the failed write was not retried on a new connection"
    assert remote.tunable_values()["power"] == -5.0
    assert len(connections) == 3, "the next request dials the device again"
    remote.close()
    with pytest.raises(ConnectionError, match="closed"):
        remote.tunable_values()

    peer = RemoteTunableDevice(host="127.0.0.1", port=announcer.port, instance_id="rf_main")
    announcer.close()
    with pytest.raises(ConnectionError):
        peer.tunable_values()
    peer.close()
    assert not announcer._connections, "server close released its accepted connections"
    source.close()


def test_unit_requests_cross_the_existing_fabric_dispatch(announcer) -> None:
    from types import SimpleNamespace
    from zlc_atom.authoring import AuthoringField, TunableField

    calls = []
    def read(name, unit=""):
        calls.append(("read", name, unit))
        return TunableField(AuthoringField(name, "float", "Power", unit=unit or "Vpp",
                            minimum=0.0, maximum=1000.0),
                            100.0 if unit == "mVpp" else .1, True, (name,))
    def tune(name, value, unit):
        calls.append(("tune", name, value, unit))
        return value + .125
    def convert(name, value, source_unit, target_unit):
        calls.append(("convert", name, value, source_unit, target_unit))
        return tuple(item * 2 for item in value) if isinstance(value, (list, tuple)) else value * 2
    source = SimpleNamespace(tunable_fields=lambda: (read("power"),),
        read_tunable_in_unit=read, tune_in_unit=tune, convert_tunable_value=convert)
    announcer.publish(PublishedDevice(instance_id="rf", role="rf", type_id="rf",
                                     parameters={}, tunable=source))
    remote = RemoteTunableDevice(host="127.0.0.1", port=announcer.port, instance_id="rf")
    assert remote.read_tunable_in_unit("power").metadata.unit == "Vpp"
    projected = remote.read_tunable_in_unit("power", "mVpp")
    assert projected.metadata.unit == "mVpp" and projected.current == 100.0
    assert remote.tune_in_unit("power", 135.0, "mVpp") == 135.125
    assert calls[-1] == ("tune", "power", 135.0, "mVpp")
    assert remote.convert_tunable_value("power", (1.0, 2.0), "Vpp", "dBm") == (2.0, 4.0)
    assert calls[-1] == ("convert", "power", [1.0, 2.0], "Vpp", "dBm")
    remote.close()


def test_an_endpoint_device_is_announced_for_its_own_protocol(announcer) -> None:
    """A pulse or SLM record carries its server's address, nothing more.

    The fabric lists it; the existing client remains the data plane, so
    asking the fabric to TUNE it is refused by name.
    """

    announcer.publish(
        PublishedDevice(
            instance_id="board",
            role="sequencer",
            type_id="sequencer.hardware",
            parameters={"host": "198.51.100.7", "port": 18861},
        )
    )
    (record,) = list_remote_devices("127.0.0.1", announcer.port)
    assert record["tunable"] is False
    assert record["parameters"] == {"host": "198.51.100.7", "port": 18861}

    with pytest.raises(RuntimeError, match="its own protocol"):
        RemoteTunableDevice(
            host="127.0.0.1", port=announcer.port, instance_id="board"
        )


def test_withdrawing_removes_the_record(announcer) -> None:
    source = virtual_rf_source(VaunixLmsConfig(serial=7))
    announcer.publish(
        PublishedDevice(
            instance_id="rf_7",
            role="probe",
            type_id="rf.vaunix_lms",
            parameters={"serial": 7},
            tunable=source,
        )
    )
    assert [
        record["instance_id"]
        for record in list_remote_devices("127.0.0.1", announcer.port)
    ] == ["rf_7"]
    announcer.withdraw("rf_7")
    assert list_remote_devices("127.0.0.1", announcer.port) == ()


def test_a_peer_speaking_another_fabric_version_is_refused_by_name(announcer, monkeypatch) -> None:
    """Two machines updated at different times meet as a named version skew,
    not as an unknown method or a missing key halfway through an editor.
    Each side says so: the server of a request, the client of an answer, a
    scan of a broadcast reply."""

    import socket

    from zlc_atom.devices.remote import fabric as module

    older = module.FABRIC_VERSION - 1
    with socket.create_connection(("127.0.0.1", announcer.port)) as connection:
        module._send_frame(connection, {"fabric": older, "method": "list"})
        answer = module._recv_frame(connection)
    assert answer["fabric"] == module.FABRIC_VERSION
    assert f"fabric version {older}" in answer["error"]["message"]
    # Version 1 stamped only its list answer and broadcast reply: a request
    # with no version at all is a v1 peer, not a peer of version None.
    with socket.create_connection(("127.0.0.1", announcer.port)) as connection:
        module._send_frame(connection, {"method": "list"})
        answer = module._recv_frame(connection)
    assert "unversioned (fabric v1)" in answer["error"]["message"]

    # The broadcast reply says its version too, and an announcer of another
    # version is named, not listed: listing it ended the whole scan.
    load = module.strict_json_loads
    monkeypatch.setattr(
        module,
        "strict_json_loads",
        lambda text, what: {**load(text, what), "fabric": older}
        if what == "fabric announcement" else load(text, what),
    )
    found, (skipped,) = discover_announcers(
        timeout_seconds=0.6, port=announcer.port, extra_hosts=("127.0.0.1",)
    )
    assert found == ()
    assert f"announcer at 127.0.0.1:{announcer.port} speaks fabric version {older}" in skipped

    receive = module._recv_frame
    monkeypatch.setattr(
        module, "_recv_frame", lambda connection: {**receive(connection), "fabric": older}
    )
    with pytest.raises(ConnectionError, match="update the older side"):
        list_remote_devices("127.0.0.1", announcer.port)


def test_named_peers_are_probed_where_a_broadcast_cannot_reach(
    announcer, monkeypatch
) -> None:
    """A cross-subnet bench names its peer once, not per device.

    A named peer where nothing listens is no reason to stop, and neither is
    a scanner that left before its answer came: Windows reports either as a
    connection reset on the socket's NEXT read, which ended the scan before
    the live answers were read -- and the responder, for good.  So did a
    stray datagram longer than the responder reads, which Windows refuses
    rather than truncates.  A name no resolver answers is no reason either,
    but it is named with the seconds it took, not dropped unseen; so is one
    a fallback resolver answered only after those seconds.
    """

    import socket

    from zlc_atom.devices.remote import fabric
    from zlc_atom.devices.remote.fabric import PROBE_MESSAGE

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as gone:
        gone.sendto(PROBE_MESSAGE, ("127.0.0.1", announcer.port))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as stray:
        stray.sendto(b"x" * 1000, ("127.0.0.1", announcer.port))
    found, skipped = discover_announcers(
        timeout_seconds=0.6,
        port=announcer.port,
        extra_hosts=("127.0.0.2", "no..such..peer", "127.0.0.1"),
    )
    assert ("127.0.0.1", announcer.port) in found
    assert len(skipped) == 1
    assert skipped[0].startswith("the named peer 'no..such..peer' was not probed: ")
    assert skipped[0].endswith("s)")
    # A name the resolver cannot encode is named the same way, not the end
    # of the scan.
    found, skipped = discover_announcers(
        timeout_seconds=0.6, port=announcer.port, extra_hosts=("pc2\u200e.lab", "127.0.0.1"),
    )
    assert ("127.0.0.1", announcer.port) in found
    assert len(skipped) == 1
    assert skipped[0].startswith(f"the named peer {'pc2\u200e.lab'!r} was not probed: ")
    # Every probe is slow once the bar is below zero: probed, and named.
    monkeypatch.setattr(fabric, "_SLOW_PEER_SECONDS", -1.0)
    found, slow = discover_announcers(
        timeout_seconds=0.6, port=announcer.port, extra_hosts=("127.0.0.1",)
    )
    assert ("127.0.0.1", announcer.port) in found
    assert len(slow) == 1
    assert slow[0].startswith("the named peer '127.0.0.1' took ")
    assert slow[0].endswith("s to resolve; write its address")


def test_an_announcer_that_cannot_be_listed_leaves_the_others_devices(
    announcer, monkeypatch
) -> None:
    """The scan offers what every other announcer publishes, and names the rest.

    One announcer of another version, or one gone between its broadcast
    reply and its list, used to end the whole scan -- or, skipped, to be
    named only in a log no view shows.  Nor do announcers that answered the
    broadcast and then never list add up: asked one after another, two of
    them cost the whole family its deadline, every live device with it.
    """

    import threading
    import time

    import zlc_atom.devices.remote.tunable.device_types as family

    announcer.publish(
        PublishedDevice(
            instance_id="board",
            role="pulse",
            type_id="sequencer.hardware",
            parameters={"host": "127.0.0.1", "port": 18861},
        )
    )
    skew = (
        "the announcer at 192.0.2.8:18859 speaks fabric version 1 and this "
        "machine speaks 2; update the older side and restart it"
    )
    monkeypatch.setattr(
        family,
        "discover_announcers",
        lambda **_peers: (
            (
                ("192.0.2.7", 18859),
                ("192.0.2.9", 18859),
                ("192.0.2.11", 18859),
                ("127.0.0.1", announcer.port),
            ),
            (skew,),
        ),
    )
    monkeypatch.setattr(family, "_LISTING_DEADLINE_SECONDS", 0.5)
    listed = family.list_remote_devices
    released = threading.Event()

    def list_unless_gone(host, port):
        if host == "192.0.2.7":
            raise ConnectionRefusedError("refused")
        if host in {"192.0.2.9", "192.0.2.11"}:
            # A firewall that lets the broadcast through and drops TCP.
            released.wait(10.0)
            raise TimeoutError("timed out")
        return listed(host, port)

    monkeypatch.setattr(family, "list_remote_devices", list_unless_gone)
    try:
        entries, notes = family._discover_fabric()
    finally:
        released.set()
    assert [(entry.instance_id, entry.type_id) for entry in entries] == [
        ("remote_board", "sequencer.hardware")
    ]
    assert notes == (
        skew,
        "the announcer at 192.0.2.7:18859 did not list: refused",
        "the announcer at 192.0.2.9:18859 did not list within 0.5s",
        "the announcer at 192.0.2.11:18859 did not list within 0.5s",
    )

    # Named peers no resolver answers can take the whole window: the live
    # announcer is then not the one blamed.
    unresolved = "the named peer 'pc3' was not probed: getaddrinfo failed (0.6s)"

    def probing_all_window(**_peers):
        time.sleep(0.6)
        return (("127.0.0.1", announcer.port),), (unresolved,)

    held = threading.Event()
    monkeypatch.setattr(family, "discover_announcers", probing_all_window)
    monkeypatch.setattr(
        family, "list_remote_devices", lambda host, port: held.wait(10.0) or listed(host, port)
    )
    try:
        entries, notes = family._discover_fabric()
    finally:
        held.set()
    assert entries == ()
    assert notes == (
        unresolved,
        f"the announcer at 127.0.0.1:{announcer.port} was not listed: probing "
        "the named peers took the whole 0.5s",
    )


def test_a_name_two_announcers_publish_is_offered_once_and_the_other_named(
    monkeypatch,
) -> None:
    """Two benches that each publish their default ``rf`` are two devices.

    Named by instance alone they were one card, and the second could not be
    added, with no line saying why.  The lower address's is offered (by
    number, not by spelling), whichever answered the broadcast first, so the
    card reaches the same bench on every scan; the other is named beside the
    scan result.
    """

    import zlc_atom.devices.remote.tunable.device_types as family

    monkeypatch.setattr(
        family,
        "discover_announcers",
        lambda **_peers: ((("192.0.2.10", 18859), ("192.0.2.9", 18859)), ()),
    )
    monkeypatch.setattr(
        family,
        "list_remote_devices",
        lambda _host, _port: (
            {"instance_id": "rf", "role": "rf", "type_id": "rf.vaunix_lms",
             "parameters": {"serial": 7}, "tunable": True},
        ),
    )
    entries, notes = family._discover_fabric()
    assert [(entry.instance_id, entry.parameters["host"]) for entry in entries] == [
        ("remote_rf", "192.0.2.9")
    ]
    assert notes == (
        "'rf' at 192.0.2.10:18859 is not offered: the announcer at "
        "192.0.2.9:18859 publishes one by that name, or is the same machine "
        "reached at another address",
    )


def test_a_published_knob_is_named_by_the_machine_it_reaches(monkeypatch) -> None:
    """Every spelling of one announcer names one knob, as it names one board.

    The leaf's identity was the host as written, so one published knob
    reached as localhost and as 127.0.0.1 was two devices to the broker, and
    a scan on one raced Control on the other with nothing refused.
    """

    from types import SimpleNamespace

    import zlc_atom.devices.remote.tunable.device_types as family
    from zlc_atom.execution import DeviceBroker
    from zlc_atom.install import InstallationFactoryContext

    monkeypatch.setattr(
        family, "RemoteTunableDevice", lambda **_dialled: SimpleNamespace(close=lambda: None)
    )
    context = InstallationFactoryContext(None, DeviceBroker())
    assert {
        family._remote_tunable_factory(
            context,
            f"knob_{index}",
            {"host": host, "port": 18859, "instance_id": "rf_main"},
        ).physical_identity.stable_device_identity
        for index, host in enumerate(("127.0.0.1", "localhost", " ::1 "))
    } == {"fabric:127.0.0.1:18859/rf_main"}
