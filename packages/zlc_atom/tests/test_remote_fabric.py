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
    assert remote.settings_provenance()["device_session_id"] == "vaunix-lms:1001"

    # A refusal crosses the wire as a refusal, message intact.
    with pytest.raises(RuntimeError, match="10.*Hz grid"):
        remote.tune("frequency_hz", 2_500_000_005.0)


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
