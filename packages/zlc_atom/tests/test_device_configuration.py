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

import numpy as np
import pytest

from zlc_atom.devices.camera.device_types import DCAM_CAMERA_SCHEMA, PYLON_CAMERA_SCHEMA
from zlc_atom.devices.sequencer.device_types import HARDWARE_SEQUENCER_SCHEMA
from zlc_atom.devices.simulation.device_types import (
    SIMULATION_WORLD_SCHEMA,
    VIRTUAL_CAMERA_SCHEMA,
    VIRTUAL_MOT_CAMERA_SCHEMA,
    VIRTUAL_SEQUENCER_SCHEMA,
    VIRTUAL_SLM_SCHEMA,
)
from zlc_atom.install import create_installation, discover_device_catalog


def _fields(schema) -> set[str]:
    return {field.name for field in schema.fields}


def test_real_camera_authoring_contains_only_operator_owned_settings() -> None:
    assert _fields(DCAM_CAMERA_SCHEMA) == {
        "device_index",
        "exposure_seconds",
        "readout_speed",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        # What one count is worth: a sensor fact the operator writes down
        # once, beside the ROI, because a measurement has to be able to offer
        # the unit before any camera is open.
        "offset_counts",
        "electrons_per_count",
    }
    assert _fields(PYLON_CAMERA_SCHEMA) == {
        "serial",
        "exposure_seconds",
        # Analog gain is the operator's, in every sense: they set it, a scan
        # sweeps it, and the bench window turns it while the camera runs.
        "gain_db",
        "trigger_source",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        "offset_counts",
        "electrons_per_count",
    }
    with pytest.raises(ValueError, match="required.*serial"):
        PYLON_CAMERA_SCHEMA.project_values({})
    for schema, forbidden in (
        (DCAM_CAMERA_SCHEMA, {"binning", "timeout_seconds"}),
        (PYLON_CAMERA_SCHEMA, {"pixel_format", "timeout_seconds"}),
    ):
        assert _fields(schema).isdisjoint(forbidden | {"grid_shape_yx", "seed"})
        for field in forbidden:
            with pytest.raises(ValueError, match="unknown authoring fields"):
                schema.project_values({field: 2})


def test_virtual_devices_and_shared_world_have_disjoint_strict_vocabularies() -> None:
    assert _fields(VIRTUAL_CAMERA_SCHEMA) == {"exposure_seconds"}
    assert _fields(SIMULATION_WORLD_SCHEMA) == {
        "image_shape_yx",
        "grid_shape_yx",
        "seed",
        "world_profile",
    }
    assert "device_index" not in _fields(VIRTUAL_CAMERA_SCHEMA)
    assert _fields(VIRTUAL_MOT_CAMERA_SCHEMA) == {
        "frame_shape_yx",
        "exposure_seconds",
    }
    with pytest.raises(ValueError, match="unknown authoring fields"):
        SIMULATION_WORLD_SCHEMA.project_values({"camera_seed": 3})
    with pytest.raises(TypeError, match="Seed must be an integer"):
        SIMULATION_WORLD_SCHEMA.project_values({"seed": 1.5})
    with pytest.raises(TypeError, match="item must be an integer"):
        SIMULATION_WORLD_SCHEMA.project_values({"image_shape_yx": [80, 1.5]})


def test_a_real_board_has_an_endpoint_and_a_virtual_one_does_not() -> None:
    assert _fields(HARDWARE_SEQUENCER_SCHEMA) == {"host", "port"}
    assert _fields(VIRTUAL_SEQUENCER_SCHEMA) == set()
    assert _fields(VIRTUAL_SLM_SCHEMA) == set()


def test_every_discovered_type_declares_its_own_parameters() -> None:
    """No two device types may share a schema object.

    Sharing one is what produced the regression: the fields that suited one
    device became the only fields any device could have.
    """

    schemas = {}
    for descriptor in discover_device_catalog().available:
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
        assert installation.world is None
        assert "sequencer" in installation.failures
        reason = str(installation.failures["sequencer"])
        assert "10.0.0.7:20000" in reason
        assert "connect_pulse" in reason
    finally:
        installation.close()


def test_the_composition_root_supplies_the_dialler(monkeypatch) -> None:
    """The saved endpoint reaches one real zlc_pulse device surface."""

    dialled: list[tuple] = []

    def dial(host, port, **kwargs):
        from zlc_pulse import (
            PulseStreamer,
            load_streamer_config,
            pulse_target_from_xdc,
        )
        from zlc_pulse.transport import MemoryRegisterTransport

        dialled.append((host, port, kwargs))
        config = load_streamer_config()
        geometry = config["params"]
        return PulseStreamer(
            MemoryRegisterTransport(geom=geometry, auto_done=True),
            geometry,
            config["clock_hz"],
            target=pulse_target_from_xdc(config_path=config["source"]),
        )

    installation = create_installation(
        ({"key": "sequencer", "type_id": "sequencer.hardware",
          "config": {"host": "10.0.0.7", "port": 20000}},),
        connect_pulse=dial,
    )
    try:
        assert installation.failures == {}
        assert dialled and dialled[0][0] == "10.0.0.7" and dialled[0][1] == 20000
        assert dialled[0][2] == {"request_timeout": 30.0}
    finally:
        installation.close()


def test_both_ends_of_the_spectrum_are_named_and_mixing_needs_no_mode() -> None:
    from zlc_atom.install import (
        discover_device_catalog,
        installation_config_from_template,
        installation_template_names,
    )

    catalog = discover_device_catalog()
    assert set(installation_template_names()) == {"virtual", "hardware"}
    virtual = installation_config_from_template(catalog, "virtual")
    assert [item.type_id for item in virtual.devices] == [
        "camera.virtual",
        "sequencer.virtual",
        "camera.virtual_mot",
        "slm.virtual",
    ]
    assert virtual.devices[1].parameters == {}
    assert virtual.devices[2].instance_id == "mot_camera"
    assert virtual.devices[3].instance_id == "slm"
    installation = create_installation("virtual")
    try:
        mot_camera = installation.device("mot_camera")
        point = mot_camera.working_point()
        assert point.sensor_shape_yx == (1200, 1920)
        assert point.acquisition_mode == "FREE_RUNNING"
        # The real MOT monitor is a Basler read out as Mono8.
        assert point.dtype.str == "|u1"
        mot_camera.set_roi((100, 50, 320, 240))
        mot_camera.set_exposure_seconds(0.05)
        mot_camera.arm(
            1,
            source_group_sizes=(1,),
            buffer_frame_count=1,
            timeout=1.0,
        )
        record = mot_camera.read_frame_records(1, timeout=2.0, exact=True)[0]
        assert installation.world._fire_count == 0
        assert record.image.shape == (240, 320)
        assert record.image.dtype.str == "|u1"
        assert float(np.std(record.image)) > 0.0
        mot_camera.finish_record_capture()
    finally:
        installation.close()
    # Mixing is a list, not a mode: devices are installed one by one.
    mixed = (
        {"key": "camera", "type_id": "camera.virtual"},
        {"key": "sequencer", "type_id": "sequencer.hardware", "config": {"host": "127.0.0.1"}},
    )
    assert len(mixed) == 2
