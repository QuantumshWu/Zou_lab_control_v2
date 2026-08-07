from __future__ import annotations

from pathlib import Path

import numpy as np
from zlc_data import OwnedSnapshot, REPEAT, SITE

from zlc_atom.install import CAPABILITY_TYPES, create_installation, discover_device_types
from zlc_atom.nodes import discover_logic_nodes
from zlc_atom.nodes.calibration import CalibrationTask
from zlc_atom.nodes.occupancy import OccupancyProcessor

from tests.fakes import FakePlane

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _short_stamps(result: object) -> dict[str, object]:
    """The run the short frames came from, which a derived signal inherits."""

    from zlc_atom.nodes.occupancy.processor import inherited_stamps

    publication = result.short.publication
    name = next(iter(publication.signals))
    return inherited_stamps(publication.signals[name].snapshot)


def test_device_discovery_is_the_leaf_manifest() -> None:
    descriptors = discover_device_types()
    assert tuple(item.type_id for item in descriptors) == (
        "camera.dcam",
        "camera.pylon",
        "camera.virtual",
        "sequencer.hardware",
        "sequencer.virtual",
    )
    assert not any(item.type_id.startswith(("rf", "mot", "temperature")) for item in descriptors)


def test_capability_tokens_have_machine_visible_types() -> None:
    assert set(CAPABILITY_TYPES) == {
        "camera.adapter",
        "camera.working_point",
        "sequencer.streamer",
        "simulation.world",
    }
    assert all(isinstance(value, type) for value in CAPABILITY_TYPES.values())


def test_logic_discovery_is_derived_from_three_leaf_modules() -> None:
    descriptors = discover_logic_nodes()
    leaf_count = len(tuple((Path(__file__).parents[1] / "src" / "zlc_atom" / "nodes").rglob("logic_node.py")))
    assert len(descriptors) == leaf_count == 3
    assert tuple(item.api_name for item in descriptors) == ("calibration", "camera_measurement", "occupancy")


def test_virtual_installation_runs_measurement_occupancy_and_same_shot_front() -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        result = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=installation.device("sequencer"),
            signal_plane=plane,
            pulse_search_paths=(REPO_ROOT / "pulses",),
            expected_centers_xy=installation.world.geometry.site_centers_xy,
        ).run()
        occupancy_node = OccupancyProcessor(
            result.calibration,
            signal_plane=plane,
        )
        occupancy = occupancy_node.process(
            result.short.frames, **_short_stamps(result)
        )
        np.testing.assert_array_equal(occupancy.rate, np.mean(occupancy.occupied, axis=-1))
        assert occupancy.counts.shape == (30, 6)
        assert occupancy.rate.shape == (30,)
        np.testing.assert_array_equal(occupancy.artifacts["counts"].block.values[:, :, 0], occupancy.counts)
        assert len(result.capture.frames) == 90
        assert len(result.reference.frames) == 60
        assert len(result.short.frames) == 30
        assert result.publication is not None
    finally:
        plane.close()
        installation.close()


def test_virtual_installation_auto_calibration_path_matches_usage_notebook() -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        result = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=installation.device("sequencer"),
            signal_plane=plane,
            pulse_search_paths=(REPO_ROOT / "pulses",),
            expected_centers_xy=installation.world.geometry.site_centers_xy,
        ).run()
        occupancy = OccupancyProcessor(result.calibration).process(
            result.short.frames, **_short_stamps(result)
        )
        assert occupancy.counts.shape == (30, 6)
        assert occupancy.rate.shape == (30,)
        counts_artifact = occupancy.artifacts["counts"]
        rate_artifact = occupancy.artifacts["rate"]
        assert isinstance(counts_artifact, OwnedSnapshot)
        assert counts_artifact.block.schema.repeat_axis.role is REPEAT
        assert counts_artifact.block.schema.point_table.column(counts_artifact.block.schema.point_table.columns[0].coordinate_id).role is SITE
        assert counts_artifact.block.values.shape == (30, 6, 1)
        assert rate_artifact.block.schema.cell_schema.is_scalar
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

    from zlc_atom.nodes._framework.descriptor import NodeKind, runtime_kind

    drivers = {
        runtime_kind(NodeKind.TASK): "execute",
        runtime_kind(NodeKind.MEASUREMENT): "execute",
        runtime_kind(NodeKind.PROCESSOR): "evaluate",
    }
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
        wanted = drivers[runtime_kind(descriptor.kind)]
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
    monkeypatch.setattr(discovery, "_modules", lambda: real + (broken,))

    types = discovery.discover_device_types()
    assert [value.type_id for value in types], "the families that DO import must still arrive"

    missing = discovery.unavailable_device_types()
    assert [value.module for value in missing] == [broken]
    assert "ModuleNotFoundError" in missing[0].reason
    assert missing[0].family == "notinstalled"
