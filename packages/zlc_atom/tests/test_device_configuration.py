"""Every device can be described in writing, and reopened from that writing.

Two regressions from the split, with one shape: a device that could only be
built by handing the factory a live object.  A live object is not something a
saved configuration can hold, so an apparatus could never be written down and
reopened -- and the parameters an operator needs were not even askable.

* every camera shared the VIRTUAL camera's schema, so a real qCMOS could be
  picked from the catalogue and then given a grid of simulated atoms and a
  random seed, while device index, ROI, binning and readout speed were rejected
  outright;
* the real sequencer had NO parameters at all and demanded an injected
  connection, so its endpoint existed nowhere.

Both now declare what they actually have.  The pulse client is still not
imported here -- this package says WHERE a board is and the composition root
knows HOW to dial it -- which keeps the boundary that makes the domain
independent of the device packages.
"""

from __future__ import annotations

import pytest

from zlc_atom.devices.camera.device_types import DCAM_CAMERA_SCHEMA, VIRTUAL_CAMERA_SCHEMA
from zlc_atom.devices.sequencer.device_types import (
    HARDWARE_SEQUENCER_SCHEMA,
    VIRTUAL_SEQUENCER_SCHEMA,
)
from zlc_atom.install import create_installation
from zlc_atom.install.discovery import discover_device_types
from zlc_atom.install.templates import INSTALLATION_TEMPLATES


def _fields(schema) -> set[str]:
    return {field.name for field in schema.fields}


def test_a_real_camera_can_be_given_the_parameters_a_real_camera_has() -> None:
    assert _fields(DCAM_CAMERA_SCHEMA) >= {
        "device_index",
        "exposure_seconds",
        "readout_speed",
        "binning",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
    }
    # And it is NOT asked for a simulation's vocabulary.
    assert not _fields(DCAM_CAMERA_SCHEMA) & {"grid_shape_yx", "seed"}


def test_the_virtual_camera_keeps_its_own_vocabulary() -> None:
    assert _fields(VIRTUAL_CAMERA_SCHEMA) >= {"grid_shape_yx", "seed"}
    assert "device_index" not in _fields(VIRTUAL_CAMERA_SCHEMA)


def test_a_real_board_has_an_endpoint_and_a_virtual_one_does_not() -> None:
    assert _fields(HARDWARE_SEQUENCER_SCHEMA) == {"host", "port", "request_timeout"}
    assert _fields(VIRTUAL_SEQUENCER_SCHEMA) == set()


def test_every_discovered_type_declares_its_own_parameters() -> None:
    """No two device types may share a schema object.

    Sharing one is what produced the regression: the fields that suited one
    device became the only fields any device could have.
    """

    schemas = {}
    for descriptor in discover_device_types():
        assert id(descriptor.authoring_schema) not in schemas, (
            f"{descriptor.type_id} shares a schema with {schemas.get(id(descriptor.authoring_schema))}"
        )
        schemas[id(descriptor.authoring_schema)] = descriptor.type_id
    assert len(schemas) >= 3


def test_a_configuration_is_enough_to_ask_for_a_real_board() -> None:
    """No live object required -- the endpoint alone is a complete request.

    The dialler is refused here because none was supplied, and the report must
    name the endpoint it would have used: that proves the configuration reached
    the factory intact rather than being rejected for lacking an object.

    The failure is REPORTED per device rather than raised, which is what an
    operator needs -- a camera that opens and a board that does not should leave
    you with a working camera and a reason, not an empty apparatus.
    """

    installation = create_installation(
        (
            {"key": "sequencer", "type_id": "sequencer.hardware",
             "config": {"host": "10.0.0.7", "port": 20000}},
        ),
    )
    try:
        assert "sequencer" in installation.failures
        reason = str(installation.failures["sequencer"])
        assert "10.0.0.7:20000" in reason
        assert "connect_pulse" in reason
    finally:
        installation.close()


def test_the_composition_root_supplies_the_dialler(monkeypatch) -> None:
    """The boundary in action: this package never imports the pulse client."""

    dialled: list[tuple] = []

    class _Streamer:
        def open(self): ...
        def close(self): ...
        def load(self, prog, **kw): ...
        def write_slots(self, values): ...
        def write_scan_table(self, rows): ...
        def fire(self, *, forever: bool = False): ...
        def wait_done(self, timeout=None): return None
        def cursor(self): return 0
        def safe(self): return None
        def snapshot(self): return {"opened": True}

    def dial(host, port, **kwargs):
        dialled.append((host, port, kwargs))
        return _Streamer()

    installation = create_installation(
        ({"key": "sequencer", "type_id": "sequencer.hardware",
          "config": {"host": "10.0.0.7", "port": 20000}},),
        connect_pulse=dial,
    )
    try:
        assert installation.failures == {}
        assert dialled and dialled[0][0] == "10.0.0.7" and dialled[0][1] == 20000
    finally:
        installation.close()


def test_both_ends_of_the_spectrum_are_named_and_mixing_needs_no_mode() -> None:
    assert set(INSTALLATION_TEMPLATES) == {"virtual", "hardware"}
    # Mixing is a list, not a mode: devices are installed one by one.
    mixed = (
        {"key": "camera", "type_id": "camera.virtual"},
        {"key": "sequencer", "type_id": "sequencer.hardware", "config": {"host": "127.0.0.1"}},
    )
    assert len(mixed) == 2
