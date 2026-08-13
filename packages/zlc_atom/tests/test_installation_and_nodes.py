from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from zlc_data import OwnedSnapshot, READOUT_EVENT, REPEAT, SITE

from zlc_atom.install import CAPABILITY_TYPES, create_installation, discover_device_catalog
from zlc_atom.nodes import discover_logic_nodes
from zlc_atom.nodes.calibration import (
    CalibrationRequest,
    CalibrationTask,
    ReadoutModelKind,
)
from zlc_atom.nodes.occupancy import OccupancyProcessor

from tests.fakes import FakePlane, camera_cycle_snapshot
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _calibration_request(*, repeats: int = 30) -> CalibrationRequest:
    return CalibrationRequest(
        camera_key="camera",
        sequencer_key="sequencer",
        pulse_template="imaging_template.json",
        repeats=repeats,
        reference_exposure_seconds=0.02,
        readout_exposure_seconds=0.005,
        roi_xywh=None,
        default_model_kind=ReadoutModelKind.BOX,
        threshold_method="empirical",
        box_half_width=1,
        box_reducer="mean",
        psf_half_width=3,
        psf_padding=3,
        detection_spot_sigma=1.0,
        detection_min_distance=3,
        detection_sigma=6.0,
    )


def test_device_discovery_is_the_leaf_manifest() -> None:
    descriptors = discover_device_catalog().available
    assert tuple(item.type_id for item in descriptors) == (
        "camera.dcam",
        "camera.pylon",
        "camera.virtual",
        "camera.virtual_mot",
        "sequencer.hardware",
        "sequencer.virtual",
    )
    assert not any(item.type_id.startswith(("rf", "mot", "temperature")) for item in descriptors)


def test_sequencer_control_is_plugin_owned_and_lazily_discovered() -> None:
    script = r"""
import zou_lab_control_v2
import sys
import zlc_atom.install as tested_module
print(zou_lab_control_v2.ROOT)
print(tested_module.__file__)
assert "zlc_ui" not in sys.modules
assert "zlc_workbench" not in sys.modules
items = {item.type_id: item for item in tested_module.discover_device_catalog().available}
factory = items["sequencer.hardware"].control_factory
assert factory is not None
assert items["sequencer.virtual"].control_factory is factory
assert "zlc_ui" not in sys.modules
assert "zlc_workbench" not in sys.modules
import types
import zlc_workbench.apps.pulse_editor as pulse_editor
calls = []
pulse_editor.create_bound_window = lambda **kwargs: calls.append(kwargs) or "window"
named_device = object()
workspace = object()
device_use = object()
session = types.SimpleNamespace(
    installation=types.SimpleNamespace(
        device=lambda key: named_device if key == "named-sequencer" else None
    ),
    workspace=workspace,
    device_use=device_use,
)
assert factory(session, "named-sequencer", window_ratio=0.4) == "window"
assert calls == [{
    "workspace": workspace,
    "sequence": None,
    "sequencer": named_device,
    "device_use": device_use,
    "path": "",
    "window_ratio": 0.4,
}]
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_capability_tokens_have_machine_visible_types() -> None:
    assert set(CAPABILITY_TYPES) == {
        "camera.adapter",
        "camera.working_point",
        "sequencer.streamer",
    }
    assert all(isinstance(value, type) for value in CAPABILITY_TYPES.values())


def test_logic_discovery_is_derived_from_leaf_modules() -> None:
    descriptors = discover_logic_nodes()
    leaf_count = len(tuple((Path(__file__).parents[1] / "src" / "zlc_atom" / "nodes").rglob("logic_node.py")))
    assert len(descriptors) == leaf_count
    assert tuple(item.api_name for item in descriptors) == (
        "calibration",
        "camera_measurement",
        "occupancy",
        "seamless_scan",
        "stepped_scan",
        "temperature",
    )


def test_device_requirements_name_build_arguments_and_exclusive_access() -> None:
    from zlc_atom.nodes._framework import DeviceAccess

    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    camera = descriptors["camera_measurement"].device_requirements
    calibration = descriptors["calibration"].device_requirements

    assert [
        (value.capability_token, value.argument_name, value.access)
        for value in camera
    ] == [("camera.adapter", "camera", DeviceAccess.EXCLUSIVE)]
    assert [
        (value.capability_token, value.argument_name, value.access)
        for value in calibration
    ] == [
        ("camera.adapter", "camera", DeviceAccess.EXCLUSIVE),
        ("sequencer.streamer", "sequencer", DeviceAccess.EXCLUSIVE),
    ]
    assert all(not hasattr(value, "device_key") for value in (*camera, *calibration))


def test_logic_build_namespace_rejects_authored_resolved_or_reserved_collisions() -> None:
    from zlc_atom.authoring import AuthoringField, AuthoringSchema
    from zlc_atom.nodes import (
        DeviceAccess,
        DeviceRequirement,
        LogicNodeDescriptor,
        NodeKind,
    )

    with pytest.raises(ValueError, match="camera"):
        LogicNodeDescriptor(
            "bad-device-override",
            NodeKind.MEASUREMENT,
            AuthoringSchema((AuthoringField("camera", "text", "Camera"),)),
            device_requirements=(
                DeviceRequirement(
                    "camera.adapter",
                    "camera",
                    DeviceAccess.EXCLUSIVE,
                ),
            ),
        )
    with pytest.raises(ValueError, match="signal_plane"):
        LogicNodeDescriptor(
            "bad-runtime-override",
            NodeKind.MEASUREMENT,
            AuthoringSchema(
                (AuthoringField("signal_plane", "text", "Signal plane"),)
            ),
        )


def test_virtual_installation_runs_measurement_occupancy_and_same_shot_front(
    tmp_path: Path,
) -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        result = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=installation.device("sequencer"),
            request=_calibration_request(),
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            artifact_directory=tmp_path,
        ).run()
        assert plane.freeze().signals == {}
        second = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=installation.device("sequencer"),
            request=_calibration_request(),
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            artifact_directory=tmp_path,
        ).run()
        assert result.artifact_path.name == "calibration.json"
        assert second.artifact_path.name == "calibration-2.json"
        assert result.artifact_path != second.artifact_path
        artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
        assert set(artifact) == {
            "site_map",
            "models",
            "default_model_kind",
            "frame_contract",
            "report",
        }
        assert tuple(model["kind"] for model in artifact["models"]) == (
            "box",
            "psf",
            "uniform_psf",
        )
        assert set(artifact["report"]) == {"models", "run_record"}
        assert all(
            set(model_report)
            == {"site_n_test", "site_n_train_dark", "site_n_train_bright"}
            for model_report in artifact["report"]["models"].values()
        )
        assert result.calibration.n_sites == len(
            installation.world.geometry.site_centers_xy
        )
        assert result.calibration.frame_contract.image_shape == (96, 128)
        assert result.calibration.frame_contract.sensor_shape == (96, 128)
        assert result.calibration.frame_contract.roi_xywh == (0, 0, 128, 96)
        assert result.calibration.frame_contract.exposure_seconds == 0.005
        assert result.calibration.frame_contract.camera_id == "camera"
        json.dumps(result.run_record)
        occupancy_node = OccupancyProcessor(
            result.calibration,
        )
        # 30 cycles, each holding the one readout frame this task judged.
        occupancy = occupancy_node.process(
            camera_cycle_snapshot([(record,) for record in result.short]),
            generation="calibration-task",
            revision=1,
        )
        np.testing.assert_array_equal(occupancy.rate, np.mean(occupancy.occupied, axis=-1))
        assert occupancy.counts.shape == (30, 1, 35)
        assert occupancy.rate.shape == (30, 1)
        np.testing.assert_array_equal(occupancy.artifacts["counts"].block.values, occupancy.counts)
        assert len(result.capture.frames) == 90
        assert sum(len(group) for group in result.reference) == 60
        assert len(result.short) == 30
    finally:
        plane.close()
        installation.close()


def test_virtual_installation_auto_calibration_path_matches_usage_notebook(
    tmp_path: Path,
) -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        result = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=installation.device("sequencer"),
            request=_calibration_request(),
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            artifact_directory=tmp_path,
        ).run()
        frames = camera_cycle_snapshot([(record,) for record in result.short])
        occupancy = OccupancyProcessor(result.calibration).process(
            frames,
            generation="calibration-task",
            revision=1,
        )
        assert occupancy.counts.shape == (30, 1, 35)
        assert occupancy.rate.shape == (30, 1)
        counts_artifact = occupancy.artifacts["counts"]
        rate_artifact = occupancy.artifacts["rate"]
        assert isinstance(counts_artifact, OwnedSnapshot)
        assert counts_artifact.block.schema.repeat_axis.role is REPEAT
        # The point axis is the parent's, verbatim -- not rebuilt, not guessed
        # from the shape.  A one-frame cycle keeps its frame point axis.
        parent_column = frames.block.schema.point_table.columns[0]
        assert counts_artifact.block.schema.point_table.columns == (parent_column,)
        assert parent_column.role is READOUT_EVENT
        # Sites are CELL data: one image resampled onto the trap lattice.
        (site_axis,) = counts_artifact.block.schema.cell_schema.data_axes
        assert site_axis is result.calibration.site_map.site_axis
        assert site_axis.role is SITE
        assert site_axis.coordinates == tuple(range(1, 36))
        assert site_axis.coordinate_labels == result.calibration.site_map.site_ids
        assert counts_artifact.block.values.shape == (30, 1, 35)
        assert rate_artifact.block.schema.cell_schema.is_scalar
        assert rate_artifact.block.schema.point_table.columns == (parent_column,)
        assert rate_artifact.block.values.shape == (30, 1, 1)
    finally:
        plane.close()
        installation.close()


def test_every_discovered_node_can_actually_be_driven_by_its_host() -> None:
    """A descriptor that declares a node the runtime cannot run is a dead entry.

    NodeHost drives a finite node through ``execute(ctx)`` and a reactive one
    through ``evaluate``.  The calibration task had neither: it exposed only
    ``run()``, so a console that offered Add Logic -> calibration failed the
    instant it started with "finite node must provide execute(ctx)" -- the
    descriptor promised something nothing could keep.
    """

    import typing

    from zlc_atom.nodes._framework.descriptor import NodeKind

    undriveable = []
    checked = []
    for descriptor in discover_logic_nodes():
        # Resolved, not read raw: every module here has postponed annotations,
        # so __annotations__ holds the NAME of the class and an isinstance
        # check against it skips every node in silence.
        produced = typing.get_type_hints(descriptor.build).get("return")
        assert isinstance(produced, type), (
            f"{descriptor.api_name}'s build must annotate the node class it returns"
        )
        wanted = "evaluate" if descriptor.kind is NodeKind.PROCESSOR else "execute"
        checked.append(descriptor.api_name)
        if not callable(getattr(produced, wanted, None)):
            undriveable.append(f"{descriptor.api_name} has no {wanted}()")
    # A guard that inspected nothing would pass in silence, which is the exact
    # failure it exists to catch.
    assert set(checked) == {item.api_name for item in discover_logic_nodes()}, (
        f"only checked {checked}: a build with no return annotation was skipped"
    )
    assert not undriveable, undriveable


def test_a_device_family_that_cannot_import_is_reported_not_raised(monkeypatch) -> None:
    """A missing vendor runtime is the ordinary case on a lab machine.

    The DCAM SDK is not installed on the bench this was written on.  Discovery
    imported every family bare, so one missing driver meant the apparatus
    editor could not open AT ALL -- a traceback about a DLL in place of the
    window you open to fix your configuration.
    """

    from zlc_atom.install import discovery

    real = discovery._modules()
    broken = "zlc_atom.devices.notinstalled.device_types"
    walks: list[None] = []

    def modules():
        walks.append(None)
        return real + (broken,)

    monkeypatch.setattr(discovery, "_modules", modules)

    catalog = discovery.discover_device_catalog()
    assert [value.type_id for value in catalog.available], (
        "the families that DO import must still arrive"
    )

    assert [value.module for value in catalog.unavailable] == [broken]
    assert "ModuleNotFoundError" in catalog.unavailable[0].reason
    assert catalog.unavailable[0].family == "notinstalled"
    assert len(walks) == 1, "available and unavailable must be one atomic discovery"
