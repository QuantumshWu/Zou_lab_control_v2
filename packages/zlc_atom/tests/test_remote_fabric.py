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


def test_a_published_tunable_is_listed_and_driven_over_the_wire(announcer) -> None:
    """The remote handle speaks the same quartet the local device does.

    It is the REAL Vaunix driver on the serving side, the generic proxy on
    the consuming side, and neither the scan axis machinery nor the control
    panel can tell the difference -- which is the entire point.
    """

    source = virtual_rf_source(
        VaunixLmsConfig(
            serial=1001,
            frequency_low_hz=500e6,
            frequency_high_hz=8e9,
        )
    )
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
    # The proxy satisfies the same capability contract as the local device.
    assert isinstance(remote, RfSource)

    fields = {field.metadata.name: field for field in remote.tunable_fields()}
    frequency = fields["frequency_hz"].metadata
    assert (frequency.minimum, frequency.maximum) == (500e6, 8e9)
    assert frequency.unit == "Hz"

    assert remote.tune("frequency_hz", 2.5e9) == 2.5e9
    assert source.tunable_values()["frequency_hz"] == pytest.approx(2.5e9), (
        "the tune must have reached the machine that owns the instrument"
    )
    assert remote.tunable_values()["frequency_hz"] == pytest.approx(2.5e9)
    assert remote.settings_provenance()["device_session_id"] == (
        source.settings_provenance()["device_session_id"]
    )

    # A refusal crosses the wire as a refusal, message intact.
    with pytest.raises(RuntimeError, match="10.*Hz grid"):
        remote.tune("frequency_hz", 2_500_000_005.0)

    # Bounds are device truth, not open-time facts: the RF owner moves its
    # commandable window when a policy edge is tuned, and a Refresh (or the
    # scan port projection) must offer the window the device accepts NOW.
    from zlc_atom.nodes.scan.plan import scan_ports_for_devices

    assert remote.tune("frequency_low_hz", 2e9) == 2e9
    refreshed = {field.metadata.name: field for field in remote.tunable_fields()}
    local = {field.metadata.name: field for field in source.tunable_fields()}
    assert (
        refreshed["frequency_hz"].metadata.minimum,
        refreshed["frequency_hz"].metadata.maximum,
    ) == (
        local["frequency_hz"].metadata.minimum,
        local["frequency_hz"].metadata.maximum,
    ) == (2e9, 8e9)
    assert refreshed["frequency_hz"].current == pytest.approx(2.5e9)
    port = next(
        item
        for item in scan_ports_for_devices({"rf": remote})
        if item.port.endswith(":frequency_hz")
    )
    assert (port.lo, port.hi) == (2e9, 8e9)
    with pytest.raises(RuntimeError, match="must lie in"):
        remote.tune("frequency_hz", 1e9)


def test_unit_requests_cross_the_existing_fabric_dispatch(monkeypatch) -> None:
    import threading
    from types import SimpleNamespace
    from zlc_atom.authoring import AuthoringField, TunableField
    from zlc_atom.devices.remote import fabric as module

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
    source = SimpleNamespace(tunable_fields=lambda: (read("power_dbm"),),
        read_tunable_in_unit=read, tune_in_unit=tune, convert_tunable_value=convert)
    announcer = object.__new__(DeviceAnnouncer)
    announcer._registry_lock = threading.Lock()
    announcer._published = {"rf": PublishedDevice(instance_id="rf", role="rf",
        type_id="rf", parameters={}, tunable=source)}
    monkeypatch.setattr(module, "_call", lambda _host, _port, request: announcer._dispatch(request))
    remote = RemoteTunableDevice(host="unused", port=0, instance_id="rf")
    assert remote.read_tunable_in_unit("power_dbm").metadata.unit == "Vpp"
    projected = remote.read_tunable_in_unit("power_dbm", "mVpp")
    assert projected.metadata.unit == "mVpp" and projected.current == 100.0
    assert remote.tune_in_unit("power_dbm", 135.0, "mVpp") == 135.125
    assert calls[-1] == ("tune", "power_dbm", 135.0, "mVpp")
    assert remote.convert_tunable_value("power_dbm", (1.0, 2.0), "Vpp", "dBm") == (2.0, 4.0)
    assert calls[-1] == ("convert", "power_dbm", (1.0, 2.0), "Vpp", "dBm")


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
    assert announcer.published_ids() == ("rf_7",)
    announcer.withdraw("rf_7")
    assert list_remote_devices("127.0.0.1", announcer.port) == ()


def test_named_peers_are_probed_where_a_broadcast_cannot_reach(announcer) -> None:
    """A cross-subnet bench names its peer once, not per device."""

    found = discover_announcers(
        timeout_seconds=0.6,
        port=announcer.port,
        extra_hosts=("127.0.0.1",),
    )
    assert ("127.0.0.1", announcer.port) in found
