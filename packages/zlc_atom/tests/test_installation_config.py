"""An apparatus can be written down, reopened, and reopened correctly.

Nothing persisted an apparatus, so what was in the lab lived only in whatever
code happened to construct it.  Two properties matter more than round-tripping:

* the document is UNFORGIVING about its shape.  A configuration that silently
  drops a key it does not recognise is how an experiment runs all afternoon with
  a setting that was never applied.
* what comes back builds the same shared world and devices, verified by actually
  installing from a reopened file rather than comparing dictionaries.
"""

from __future__ import annotations

import json

import pytest

from zlc_atom.install import create_installation
from zlc_atom.install.configuration import (
    DEVICE_ENTRY_KEYS,
    DOCUMENT_FORMAT,
    DeviceInstanceConfig,
    InstallationConfig,
    load_installation_config,
    save_installation_config,
)


def _virtual_apparatus() -> InstallationConfig:
    return InstallationConfig(
        (
            DeviceInstanceConfig(
                "camera",
                "camera",
                "camera.virtual",
                {"exposure_seconds": 0.02},
            ),
            DeviceInstanceConfig("sequencer", "sequencer", "sequencer.virtual", {}),
        ),
        simulation={
            "image_shape_yx": (80, 110),
            "grid_shape_yx": (4, 6),
            "seed": 7,
            "world_profile": "",
        },
    )


def test_a_reopened_apparatus_installs_the_same_world_and_devices(tmp_path) -> None:
    path = save_installation_config(_virtual_apparatus(), tmp_path / "apparatus.json")
    reopened = load_installation_config(path)
    assert reopened == _virtual_apparatus()
    installation = create_installation(
        reopened.specs(), simulation=reopened.simulation
    )
    try:
        assert installation.failures == {}
        assert set(installation.devices) == {"camera", "sequencer"}
        assert installation.world.geometry.image_shape_yx == (80, 110)
        assert installation.world.geometry.grid_shape_yx == (4, 6)
        assert installation.world.config.seed == 7
    finally:
        installation.close()


def test_a_real_apparatus_writes_its_endpoint_and_its_sensor_settings(tmp_path) -> None:
    config = InstallationConfig(
        (
            DeviceInstanceConfig(
                "camera", "camera", "camera.dcam",
                {"device_index": 1, "exposure_seconds": 0.003, "readout_speed": 1,
                 "roi_x": 128, "roi_y": 64, "roi_width": 512, "roi_height": 256},
            ),
            DeviceInstanceConfig(
                "sequencer", "sequencer", "sequencer.hardware",
                {"host": "10.0.0.7", "port": 18861},
            ),
        )
    )
    path = save_installation_config(config, tmp_path / "bench.json")
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["format"] == DOCUMENT_FORMAT
    assert written["devices"][0]["parameters"]["roi_width"] == 512
    assert written["devices"][1]["parameters"]["host"] == "10.0.0.7"
    assert load_installation_config(path) == config


@pytest.mark.parametrize(
    "entry",
    [
        {"instance_id": "camera", "role": "camera", "type_id": "camera.virtual"},
        {"instance_id": "camera", "role": "camera", "type_id": "camera.virtual",
         "parameters": {}, "extra": 1},
    ],
    ids=["missing-key", "unknown-key"],
)
def test_a_device_entry_with_the_wrong_keys_is_refused(entry) -> None:
    """Silently ignoring an unknown key is how a setting goes unapplied."""

    with pytest.raises(ValueError, match="exactly"):
        DeviceInstanceConfig.from_dict(entry)
    assert DEVICE_ENTRY_KEYS == {"instance_id", "role", "type_id", "parameters"}


def test_two_devices_may_not_share_a_name_or_a_role() -> None:
    with pytest.raises(ValueError, match="duplicate device"):
        InstallationConfig(
            (
                DeviceInstanceConfig("camera", "camera", "camera.virtual", {}),
                DeviceInstanceConfig("camera", "viewer", "camera.pylon", {}),
            )
        )


def test_a_document_that_is_not_ours_is_refused(tmp_path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"format": "something.else", "devices": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown installation format"):
        load_installation_config(path)


def test_legacy_camera_owned_world_configuration_is_refused(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "format": DOCUMENT_FORMAT,
                "devices": [
                    {
                        "instance_id": "camera",
                        "role": "camera",
                        "type_id": "camera.virtual",
                        "parameters": {"grid_shape_yx": [4, 6], "seed": 7},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="root simulation"):
        load_installation_config(path)


def test_a_duplicate_key_or_a_nan_is_refused(tmp_path) -> None:
    """Both are ways a file can look valid and mean something else."""

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"format":"zlc.installation","simulation":{},'
        '"devices":[],"devices":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_installation_config(duplicate)

    nan = tmp_path / "nan.json"
    nan.write_text(
        '{"format":"zlc.installation","simulation":{},'
        '"devices":[{"instance_id":"c","role":"c",'
        '"type_id": "camera.virtual", "parameters": {"exposure_seconds": NaN}}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="NaN"):
        load_installation_config(nan)


def test_a_parameter_that_a_file_cannot_hold_is_refused() -> None:
    """A live object in a configuration is the regression this prevents."""

    with pytest.raises(TypeError, match="plain data"):
        DeviceInstanceConfig("camera", "camera", "camera.dcam", {"driver": object()})


def test_an_explicit_world_and_simulation_mapping_are_two_owners() -> None:
    with pytest.raises(
        ValueError,
        match="pass an explicit world or simulation config, not both",
    ):
        create_installation((), world=object(), simulation={})
