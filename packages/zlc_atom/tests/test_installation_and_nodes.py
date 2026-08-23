from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from zlc_data import OwnedSnapshot, READOUT_EVENT, REPEAT, SITE
from zlc_runtime import DatasetOutputDeclaration

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
        reference_before_slot=1,
        readout_slot=2,
        reference_after_slot=3,
        default_model_kind=ReadoutModelKind.BOX,
        threshold_method="empirical",
        box_half_width=1,
        box_reducer="mean",
        psf_half_width=3,
        psf_padding=3,
        detection_spot_sigma=1.0,
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
        "slm.hamamatsu_x15213",
        "slm.virtual",
    )
    assert not any(item.type_id.startswith(("rf", "mot", "temperature")) for item in descriptors)


def test_sequencer_control_is_plugin_owned_and_lazily_discovered(tmp_path: Path) -> None:
    script = r"""
import zou_lab_control
import sys
import zlc_atom.install as tested_module
print(zou_lab_control.ROOT)
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
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_capability_tokens_have_machine_visible_types() -> None:
    assert set(CAPABILITY_TYPES) == {
        "camera.adapter",
        "sequencer.streamer",
        "slm.phase",
    }
    assert all(isinstance(value, type) for value in CAPABILITY_TYPES.values())


def test_virtual_apparatus_installs_one_canonical_slm_phase_device() -> None:
    from zlc_atom.devices.slm import SlmAdapter

    installation = create_installation("virtual")
    try:
        slm = installation.device("slm")
        assert isinstance(slm, SlmAdapter)
        assert installation.capability("slm.phase", key="slm") is slm
        assert slm.identity == "virtual-slm"
        assert slm.shape_yx == (128, 128)
        assert slm.command_revision == 0
        assert slm.mapping_revision == 0
        assert slm.last_command_receipt["outcome"] == "known-new"

        before_revision = installation.world._slm_phase_revision
        authored = np.linspace(
            -np.pi,
            5.0 * np.pi,
            num=np.prod(slm.shape_yx),
            dtype=np.float32,
        ).reshape(slm.shape_yx)
        commanded = slm.apply_phase(authored)
        assert commanded.shape == slm.shape_yx
        assert not commanded.flags.writeable
        assert float(np.min(commanded)) >= 0.0
        assert float(np.max(commanded)) < 2.0 * np.pi
        assert installation.world._slm_phase_revision == before_revision + 1
        np.testing.assert_array_equal(slm.last_commanded_phase, commanded)
        assert slm.command_revision == 1
        assert slm.last_command_receipt == {
            "transport": "virtual",
            "identity": "virtual-slm",
            "profile": "simulation",
            "model": "SimulationWorld",
            "serial": "virtual-slm",
            "wavelength_nm": None,
            "flip_x": False,
            "flip_y": False,
            "correction_path": "",
            "correction_enabled": False,
            "mapping_revision": 0,
            "settle_seconds": 0.0,
            "phase_curve_source": "simulation",
            "outcome": "known-new",
            "command_revision": 1,
            "stage": "simulation-applied",
            "readback": "simulation-state",
        }
        with pytest.raises(ValueError, match="read-only"):
            commanded[0, 0] = 0.0
        slm.close()
        np.testing.assert_array_equal(slm.last_commanded_phase, commanded)
        assert installation.world._slm_phase_revision == before_revision + 1
    finally:
        installation.close()

    duplicate = create_installation(
        (
            {"key": "slm_a", "type_id": "slm.virtual", "config": {}},
            {"key": "slm_b", "type_id": "slm.virtual", "config": {}},
        )
    )
    try:
        assert set(duplicate.devices) == {"slm_a"}
        assert isinstance(duplicate.failures["slm_b"], RuntimeError)
        assert "already bound" in str(duplicate.failures["slm_b"])
    finally:
        duplicate.close()


def test_logic_discovery_is_derived_from_leaf_modules() -> None:
    descriptors = discover_logic_nodes()
    leaf_count = len(tuple((Path(__file__).parents[1] / "src" / "zlc_atom" / "nodes").rglob("logic_node.py")))
    assert len(descriptors) == leaf_count
    assert tuple(item.api_name for item in descriptors) == (
        "calibration",
        "camera_measurement",
        "occupancy",
        "seamless_scan",
        "slm_feedback",
        "stepped_scan",
        "temperature",
    )
    assert all(
        isinstance(output, DatasetOutputDeclaration)
        for descriptor in descriptors
        for output in descriptor.outputs
    )
    assert all(
        preview.producer
        or all(
            any(declaration is output for output in descriptor.outputs)
            for declaration in (preview.output, preview.overlay)
            if declaration is not None
        )
        for descriptor in descriptors
        for preview in descriptor.node_previews
    )
    assert {
        input_spec.delivery
        for descriptor in descriptors
        for input_spec in descriptor.input_specs
        if hasattr(input_spec, "delivery")
    } == {"exact"}


def test_task_preview_policy_and_typed_output_reference_are_explicit() -> None:
    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes._framework.descriptor import (
        LogicNodeDescriptor,
        NodeKind,
        NodePreviewSpec,
    )

    with pytest.raises(ValueError, match="explicitly declare node_previews"):
        LogicNodeDescriptor("silent", NodeKind.TASK, AuthoringSchema())

    explicit_none = LogicNodeDescriptor(
        "artifact-only",
        NodeKind.TASK,
        AuthoringSchema(),
        node_previews=(),
    )
    assert explicit_none.node_previews == ()

    output = DatasetOutputDeclaration("frames", "camera.frames")
    overlay = DatasetOutputDeclaration("occupied", "atom.occupied")
    own_preview = NodePreviewSpec(output, "facet_grid", overlay=overlay)
    descriptor = LogicNodeDescriptor(
        "camera-task",
        NodeKind.TASK,
        AuthoringSchema(),
        outputs=(output, overlay),
        node_previews=(own_preview,),
    )
    assert descriptor.outputs[0] is descriptor.node_previews[0].output
    assert descriptor.outputs[1] is descriptor.node_previews[0].overlay

    with pytest.raises(ValueError, match="primary output"):
        NodePreviewSpec(output, "image", overlay=output)
    with pytest.raises(TypeError, match="DatasetOutputDeclaration"):
        NodePreviewSpec(output, "image", overlay=object())

    copied_declaration = DatasetOutputDeclaration("frames", "camera.frames")
    with pytest.raises(ValueError, match="undeclared outputs"):
        LogicNodeDescriptor(
            "copied-output",
            NodeKind.TASK,
            AuthoringSchema(),
            outputs=(output,),
            node_previews=(NodePreviewSpec(copied_declaration, "image"),),
        )
    copied_overlay = DatasetOutputDeclaration("occupied", "atom.occupied")
    with pytest.raises(ValueError, match="undeclared outputs"):
        LogicNodeDescriptor(
            "copied-overlay",
            NodeKind.TASK,
            AuthoringSchema(),
            outputs=(output, overlay),
            node_previews=(
                NodePreviewSpec(output, "image", overlay=copied_overlay),
            ),
        )

    companion = DatasetOutputDeclaration("frames", "camera.frames")
    companion_overlay = DatasetOutputDeclaration("occupied", "atom.occupied")
    companion_preview = NodePreviewSpec(
        companion,
        "image",
        {"fate:frame": 1},
        producer="camera",
        overlay=companion_overlay,
    )
    companion_task = LogicNodeDescriptor(
        "feedback",
        NodeKind.TASK,
        AuthoringSchema(),
        node_previews=(companion_preview,),
    )
    assert companion_task.node_previews[0].producer == "camera"
    assert companion_task.node_previews[0].overlay is companion_overlay
    with pytest.raises(TypeError):
        companion_task.node_previews[0].semantic["fate:frame"] = 2


def test_device_requirements_name_build_arguments() -> None:
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    camera = descriptors["camera_measurement"].device_requirements
    calibration = descriptors["calibration"].device_requirements

    assert [
        (value.capability_token, value.argument_name)
        for value in camera
    ] == [("camera.adapter", "camera")]
    assert [
        (value.capability_token, value.argument_name)
        for value in calibration
    ] == [
        ("camera.adapter", "camera"),
        ("sequencer.streamer", "sequencer"),
    ]
    assert all(not hasattr(value, "device_key") for value in (*camera, *calibration))


def test_logic_build_namespace_rejects_authored_resolved_or_reserved_collisions() -> None:
    from zlc_atom.authoring import AuthoringField, AuthoringSchema
    from zlc_atom.nodes import (
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
                DeviceRequirement("camera.adapter", "camera"),
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
            signal_plane=FakePlane(),
        ).run(tmp_path)
        assert plane.freeze().signals == {}
        second = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=installation.device("sequencer"),
            request=_calibration_request(),
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            signal_plane=FakePlane(),
        ).run(tmp_path)
        assert result.artifact_path.name == "calibration.json"
        assert second.artifact_path.name == "calibration.json"
        assert result.artifact_path != second.artifact_path
        artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
        assert set(artifact) == {
            "format",
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
        all_sites = len(installation.world.geometry.site_centers_xy)
        observable_sites = int(
            np.count_nonzero(installation.world._site_loading_probabilities() > 0.0)
        )
        assert result.calibration.n_sites == observable_sites
        assert all_sites - observable_sites >= int(np.ceil(0.10 * all_sites))
        assert result.calibration.frame_contract.image_shape == (96, 128)
        assert result.calibration.frame_contract.sensor_shape == (96, 128)
        assert result.calibration.frame_contract.roi_xywh == (0, 0, 128, 96)
        # What the SENSOR integrated, which under an edge trigger is the
        # exposure the camera was armed at -- not the shorter window the
        # pulse gates the probe light with.  The contract records the
        # condition the thresholds were measured under; it does not
        # legislate the exposure of any later run.
        assert result.calibration.frame_contract.exposure_seconds == 0.02
        assert result.calibration.frame_contract.camera_id == "camera"
        # The runtime view is recursively immutable; the artifact codec owns
        # the independent plain JSON projection.
        json.dumps(result.calibration.to_dict()["report"]["run_record"])
        occupancy_node = OccupancyProcessor(
            result.calibration,
        )
        # 30 cycles, each holding the one readout frame this task judged.
        occupancy = occupancy_node.process(
            camera_cycle_snapshot([(record,) for record in result.capture.short]),
        )
        assert occupancy.counts.shape == (
            30,
            1,
            result.calibration.n_sites,
        )
        np.testing.assert_array_equal(occupancy.artifacts["counts"].block.values, occupancy.counts)
        assert len(result.capture.frames) == 90
        assert sum(len(group) for group in result.capture.reference) == 60
        assert len(result.capture.short) == 30
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
            signal_plane=FakePlane(),
        ).run(tmp_path)
        frames = camera_cycle_snapshot([(record,) for record in result.capture.short])
        occupancy = OccupancyProcessor(result.calibration).process(
            frames,
        )
        expected_shape = (30, 1, result.calibration.n_sites)
        assert occupancy.counts.shape == expected_shape
        counts_artifact = occupancy.artifacts["counts"]
        assert isinstance(counts_artifact, OwnedSnapshot)
        assert counts_artifact.block.schema.repeat_axis.role is REPEAT
        # The point axis is the parent's, verbatim -- not rebuilt, not guessed
        # from the shape.  A one-frame cycle keeps its frame point axis.
        parent_column = frames.block.schema.point_table.columns[0]
        assert counts_artifact.block.schema.point_table.columns == (parent_column,)
        assert parent_column.role is READOUT_EVENT
        # Sites are CELL data: one image resampled onto the trap lattice.
        (site_axis,) = counts_artifact.block.schema.cell_schema.data_axes
        # The occupancy publication carries the calibration's site axis, not one
        # of its own: same id, name, size and 1..n coordinates.  By value rather
        # than by identity, because a schema built from these exact axes is
        # shared between publications that describe the same thing.
        assert site_axis == result.calibration.site_map.site_axis
        assert site_axis.role is SITE
        assert site_axis.coordinates == tuple(
            range(1, result.calibration.n_sites + 1)
        )
        assert site_axis.coordinate_labels is None
        assert tuple(site_axis.coordinates) == tuple(
            range(1, len(result.calibration.site_map.site_ids) + 1)
        )
        assert counts_artifact.block.values.shape == expected_shape
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


def test_every_discovered_measurement_declares_live_data_and_a_preview() -> None:
    """A new Measurement cannot silently become a final-only black box.

    Runtime enforces the actual commit at terminal and the hosted vertical
    tests exercise publication before terminal.  Discovery owns the earlier,
    deterministic part of the contract: every product Measurement must name
    both the Dataset it publishes while running and what Start opens to watch.
    """

    from zlc_atom.nodes._framework.descriptor import NodeKind

    measurements = tuple(
        descriptor
        for descriptor in discover_logic_nodes()
        if descriptor.kind is NodeKind.MEASUREMENT
    )
    assert measurements, "the product discovery found no Measurements"
    for descriptor in measurements:
        assert descriptor.outputs, f"{descriptor.api_name} declares no live Dataset"
        assert descriptor.node_previews, (
            f"{descriptor.api_name} declares no preview for its live Dataset"
        )


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
