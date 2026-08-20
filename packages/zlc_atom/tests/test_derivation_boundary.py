"""Which way you derive depends on whether the source is still moving.

Two packages appeared to disagree about "what a shot's output is": the domain
publishes a finished measurement with no coverage, while the runtime's
latest-only lane demands monitor coverage.  Reading that as a contract bug
leads to adding coverage to finished data, which would be a lie.

They do not disagree.  A live monitor Processor is latest-only: it keeps up with
a signal that is still moving and skips honestly when it cannot, which is
exactly what monitor coverage records.  A finished measurement has nothing left
to keep up with, so the same hosted Processor derives from it once.  Both paths
exist and this file states which is which so nobody "fixes" the boundary away.
"""

from __future__ import annotations

from dataclasses import replace
import sys
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.install import create_installation
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.camera_measurement.measurement import (
    CAMERA_FRAMES_OUTPUT,
    CameraMeasurementNode,
    CameraMeasurementRequest,
    frames_snapshot,
)
from zlc_atom.devices.camera.contract import CameraFrameRecord, CameraWorkingPoint
from zlc_atom.nodes.occupancy.logic_node import LOGIC_NODE as OCCUPANCY_LOGIC_NODE
from zlc_atom.nodes.occupancy.processor import OCCUPANCY_OUTPUTS, OccupancyProcessor
from zlc_plot import (
    IMAGE_POINT_OVERLAY_GEOMETRY_RECORD,
    image_point_overlay_geometry,
)
from zlc_data import AxisId, AxisSpec, DatasetSchema, SITE, owned_snapshot_from_arrays
from zlc_runtime import DatasetCoverage, LiveDatasetOutput, MonitorCoverage
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane

from tests.pulse_fixture import CALIBRATION_FRAMES_PER_CYCLE, build_calibration_pulse


@pytest.fixture
def bench():
    plane = SignalDataPlane()
    installation = create_installation("virtual")
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build_calibration_pulse(sequencer.describe())
        sequencer.load(program)
        yield plane, camera, sequencer, CALIBRATION_FRAMES_PER_CYCLE
    finally:
        installation.close()
        plane.close()


def _finished_shot(plane, camera, sequencer, windows):
    node = CameraMeasurementNode(
        camera=camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 1, windows),
        signal_plane=plane,
        producer="cm",
    )
    capture = node.prepare()
    sequencer.fire()
    sequencer.wait_done(1.0)
    return node, capture.collect()


def _single_site_calibration(node, source) -> TrapCalibration:
    height, width = source.values.shape[-2:]
    actual = node.actual_working_point
    assert actual is not None
    roi_y, roi_x = actual.roi_origin_yx
    roi_height, roi_width = actual.roi_shape_yx
    site_ids = ("site_0000",)
    return TrapCalibration(
        SiteMap(
            site_ids,
            np.asarray([[width // 2, height // 2]], dtype=float),
            [True],
            [1.0],
        ),
        (ReadoutModel(site_ids, [0.0], [-1.0], [1.0], [True], [1.0]),),
        ReadoutModelKind.BOX,
        FrameContract(
            (height, width),
            sensor_shape=actual.sensor_shape_yx,
            roi_xywh=(roi_x, roi_y, roi_width, roi_height),
            binning_yx=actual.binning_yx,
            exposure_seconds=actual.exposure_seconds,
            camera_id=node.camera_key,
            readout_mode=actual.readout_mode,
        ),
    )


def test_site_geometry_uses_sensor_axes_after_nonzero_roi_and_binning() -> None:
    point = CameraWorkingPoint(
        "EXTERNAL_TRIGGERED",
        (4, 3),
        (30, 40),
        (10, 20),
        (8, 9),
        (2, 3),
        np.dtype("<u2"),
        "count",
        0.01,
        0.02,
        0.0,
        1.0,
        "default",
    )
    frame = frames_snapshot(
        ((CameraFrameRecord(np.zeros((4, 3), dtype="<u2"), 0),),),
        producer="roi-camera",
        generation="geometry-run",
        revision=1,
        working_point=point,
    )
    geometry = image_point_overlay_geometry(
        frame,
        np.asarray(((1.5, 2.0),)),
        ("site-1",),
        status_axis=AxisSpec(AxisId("site"), "site", SITE, 1, (1,)),
        coordinates_are_indices=True,
    )
    assert geometry["coordinate_frame"] == "sensor_pixel_xy"
    assert geometry["coordinates_xy"] == [[24.5, 14.0]]


def test_a_finished_measurement_keeps_extent_while_runtime_owns_terminal(bench) -> None:
    plane, camera, sequencer, windows = bench
    node, result = _finished_shot(plane, camera, sequencer, windows)
    signal = node.signal_key("frames")
    value = result.publication.value(signal)
    assert isinstance(value.coverage, DatasetCoverage) and value.coverage.complete
    assert not plane.is_generation_live(signal)
    assert plane.current_dataset(signal) is result.snapshot


def test_occupancy_classifies_only_event_cells_and_runtime_owns_full_history(
    bench,
) -> None:
    plane, camera, sequencer, windows = bench
    node, result = _finished_shot(plane, camera, sequencer, windows)
    source = result.publication.value(node.signal_key("frames"))
    assert source is not None
    calibration = _single_site_calibration(node, source)
    processor = OccupancyProcessor(
        calibration,
        source_signal="event-camera/frames",
        producer="event-occupancy",
    )
    assert processor.dataset_output_declarations is OCCUPANCY_OUTPUTS
    assert OCCUPANCY_LOGIC_NODE.outputs is OCCUPANCY_OUTPUTS
    event_schema = source.snapshot.block.schema
    canonical_schema = DatasetSchema(
        replace(
            event_schema.repeat_axis,
            size=3,
            coordinates=(0, 1, 2),
        ),
        event_schema.point_table,
        event_schema.grid_topology,
        event_schema.cell_schema,
    )
    event_cells = event_schema.repeat_axis.size * event_schema.point_table.row_count
    coverage = DatasetCoverage(event_cells, 3 * event_cells)
    source_event = replace(
        source,
        coverage=coverage,
        canonical_schema=canonical_schema,
        cell_origin=(0, 0),
    )
    outputs = processor.evaluate(source_event)
    for name in ("counts", "occupied", "frame_judged"):
        output = outputs[name]
        assert output.snapshot.block.schema.repeat_axis.size == 1
        assert output.canonical_schema.repeat_axis.size == 3
        assert output.cell_origin == (0, 0)
        assert output.coverage == coverage
    assert outputs["frame_judged"].snapshot is source.snapshot

    validity = np.ones(source.snapshot.block.values.shape, dtype=np.bool_)
    validity[:, 0] = False
    invalid_snapshot = owned_snapshot_from_arrays(
        event_schema,
        source.snapshot.block.values,
        source.snapshot.block.revision,
        validity=validity,
        stream_generation=source.snapshot.ref.stream_generation,
    )
    invalid_source = replace(
        source,
        snapshot=invalid_snapshot,
        coverage=MonitorCoverage(event_cells, event_cells),
    )
    invalid = processor.evaluate(invalid_source)
    assert not np.any(invalid["occupied"].snapshot.block.values[:, 0])
    assert np.all(np.isnan(invalid["counts"].snapshot.block.values[:, 0]))
    assert not np.any(
        invalid["occupied"].snapshot.expanded_validity()[:, 0]
    )

    class EventCamera:
        instance_id = "event-camera"
        dataset_output_declarations = (CAMERA_FRAMES_OUTPUT,)

        @staticmethod
        def signal_key(name: str) -> str:
            return f"event-camera/{name}"

    event_plane = SignalDataPlane()
    camera_owner = EventCamera()
    event_host = None
    try:
        event_plane.begin_generation(camera_owner)
        event_plane.commit_live(
            camera_owner,
            {
                "frames": LiveDatasetOutput(
                    CAMERA_FRAMES_OUTPUT,
                    source.snapshot,
                    coverage,
                    source.run_record,
                    canonical_schema,
                    (0, 0),
                )
            },
        )
        wake = Event()
        event_host = NodeHost(
            processor,
            event_plane,
            wake.set,
            instance_id=processor.instance_id,
            kind="processor",
            dataset_output_declarations=OCCUPANCY_OUTPUTS,
            input_signal="event-camera/frames",
            input_delivery="exact",
        )
        event_host.start()
        counts_signal = event_host.signal_key("counts")
        deadline = time.monotonic() + 5.0
        current = None
        while current is None and time.monotonic() < deadline:
            event_host.poll()
            try:
                current = event_plane.current_dataset(counts_signal)
            except LookupError:
                wake.wait(0.01)
                wake.clear()
        assert current is not None
        assert current.block.values.shape == (3, windows, 1)
        current_validity = current.expanded_validity()
        assert np.all(current_validity[0])
        assert not np.any(current_validity[1:])
        latest = event_plane.latest_publication(counts_signal)
        assert latest is not None
        assert latest.value(counts_signal).snapshot.block.values.shape == (
            1,
            windows,
            1,
        )
        event_plane.seal_committed(camera_owner, cut_short=True)
        deadline = time.monotonic() + 5.0
        while event_host.running and time.monotonic() < deadline:
            event_host.poll()
            wake.wait(0.01)
            wake.clear()
        assert event_host.poll().phase == "done"
    finally:
        if event_host is not None:
            if event_host.running:
                event_host.cancel("cleanup")
                event_host.poll()
            if not event_host.running:
                event_host.shutdown()
        event_plane.close()


def test_hosting_a_processor_on_a_finished_signal_derives_once(bench, tmp_path: Path) -> None:
    plane, camera, sequencer, windows = bench
    node, result = _finished_shot(plane, camera, sequencer, windows)
    source_name = node.signal_key("frames")
    source = result.publication.value(source_name)
    assert source is not None
    actual = node.actual_working_point
    assert actual is not None
    calibration = _single_site_calibration(node, source)
    frame_contract = calibration.frame_contract
    site_ids = ("site_0000",)
    calibration_path = calibration.save(tmp_path / "calibration.json")
    processor = OCCUPANCY_LOGIC_NODE.instantiate(
        calibration=CALIBRATION_ARTIFACT_CODEC.resolve(calibration_path),
        source_signal=source_name,
    )
    assert isinstance(processor, OccupancyProcessor)
    assert processor.calibration_path == calibration_path.resolve()
    different_nonstructural_context = TrapCalibration(
        calibration.site_map,
        calibration.models,
        calibration.default_model_kind,
        FrameContract(
            frame_contract.image_shape,
            sensor_shape=frame_contract.sensor_shape,
            roi_xywh=frame_contract.roi_xywh,
            binning_yx=frame_contract.binning_yx,
            exposure_seconds=frame_contract.exposure_seconds * 2.0,
            camera_id="another-camera",
            readout_mode="another-readout-mode",
        ),
    )
    different_context_path = different_nonstructural_context.save(
        tmp_path / "different-context.json"
    )
    different_context_processor = OCCUPANCY_LOGIC_NODE.instantiate(
        calibration=CALIBRATION_ARTIFACT_CODEC.resolve(different_context_path),
        source_signal=source_name,
    )
    different_outputs = different_context_processor.evaluate(source)
    # (repeat, point, site): one cycle, `windows` frame POINTS inherited from
    # the camera publication, one site of CELL data per frame.
    assert different_outputs["counts"].snapshot.block.values.shape == (1, windows, 1)
    # A run may crop the sensor differently from the calibration.  Where a
    # trap IS, is a fact about the SENSOR, so a crop one pixel to the side
    # numbers the same places differently and the calibration is read against
    # it -- it is not a different apparatus.
    x, y, width, height = frame_contract.roi_xywh
    shifted = TrapCalibration(
        calibration.site_map,
        calibration.models,
        calibration.default_model_kind,
        FrameContract(
            frame_contract.image_shape,
            sensor_shape=None,
            roi_xywh=(x + 1, y, width, height),
            binning_yx=frame_contract.binning_yx,
        ),
    )
    shifted_path = shifted.save(tmp_path / "shifted.json")
    shifted_processor = OCCUPANCY_LOGIC_NODE.instantiate(
        calibration=CALIBRATION_ARTIFACT_CODEC.resolve(shifted_path),
        source_signal=source_name,
    )
    shifted_outputs = shifted_processor.evaluate(source)
    assert shifted_outputs["counts"].snapshot.block.values.shape == (1, windows, 1)
    assert shifted_processor.readout.site_map.centers_xy[0][0] == (
        calibration.site_map.centers_xy[0][0] + 1
    )
    assert shifted_processor.model is shifted_processor.readout.select_model(
        ReadoutModelKind.BOX
    )

    # The bench case: a run that takes a SMALLER crop than the calibration was
    # measured on.  The frame shape then differs too, and checking it before
    # the run record had been read compared it against the crop the
    # calibration was measured on -- "frame shape (h, w) differs from the crop
    # this readout is placed against (H, W)" -- with the translation that
    # answers it computed one line too late.
    height, width = frame_contract.image_shape
    wider = TrapCalibration(
        calibration.site_map,
        calibration.models,
        calibration.default_model_kind,
        FrameContract(
            (height + 4, width + 4),
            sensor_shape=None,
            roi_xywh=(x, y, width + 4, height + 4),
            binning_yx=frame_contract.binning_yx,
        ),
    )
    wider_path = wider.save(tmp_path / "wider.json")
    wider_processor = OCCUPANCY_LOGIC_NODE.instantiate(
        calibration=CALIBRATION_ARTIFACT_CODEC.resolve(wider_path),
        source_signal=source_name,
    )
    wider_outputs = wider_processor.evaluate(source)
    assert wider_outputs["counts"].snapshot.block.values.shape == (1, windows, 1)
    assert wider_processor.readout.frame_contract.image_shape == (height, width)
    # Same origin, so the sites do not move; what changes is how much of the
    # sensor the run is looking at, and the readout now knows it.
    assert wider_processor.readout.site_map.centers_xy[0][0] == (
        calibration.site_map.centers_xy[0][0]
    )

    # What IS refused is a crop that does not cover the sites: reading a box
    # that runs off the edge would return a number that looks like a
    # measurement.  Refused by name, so an operator knows which traps moved
    # out of the picture.
    uncovered = TrapCalibration(
        calibration.site_map,
        calibration.models,
        calibration.default_model_kind,
        FrameContract(
            frame_contract.image_shape,
            sensor_shape=None,
            roi_xywh=(x + width, y, width, height),
            binning_yx=frame_contract.binning_yx,
        ),
    )
    uncovered_path = uncovered.save(tmp_path / "uncovered.json")
    uncovered_processor = OCCUPANCY_LOGIC_NODE.instantiate(
        calibration=CALIBRATION_ARTIFACT_CODEC.resolve(uncovered_path),
        source_signal=source_name,
    )
    with pytest.raises(ValueError, match="does not cover"):
        uncovered_processor.evaluate(source)
    wake = Event()
    host = NodeHost(
        processor,
        plane,
        wake.set,
        instance_id=processor.instance_id,
        kind="processor",
        dataset_output_declarations=OCCUPANCY_OUTPUTS,
        input_signal=source_name,
        input_delivery="exact",
    )

    try:
        host.start()
        deadline = time.monotonic() + 5.0
        while not host.terminal and time.monotonic() < deadline:
            host.poll()
            if host.terminal:
                break
            wake.wait(max(0.0, deadline - time.monotonic()))
            wake.clear()
        observation = host.poll()
        assert observation.terminal and observation.phase == "done"

        publication = plane.freeze().publication("@logic/occupancy/occupied")
        assert publication is not None
        assert set(publication.signals) == {
            "@logic/occupancy/counts",
            "@logic/occupancy/occupied",
            "@logic/occupancy/frame_judged",
        }
        np.testing.assert_array_equal(
            publication.value("@logic/occupancy/frame_judged").values,
            source.values,
        )
        assert np.all(
            publication.value("@logic/occupancy/occupied").snapshot.expanded_validity()
        )

        from zlc_data import READOUT_EVENT, SITE

        # Every derived signal INHERITS the camera's point column, object for
        # object: the frames it judged are the points it reports over.
        (parent_column,) = source.schema.point_table.columns
        assert parent_column.role is READOUT_EVENT
        for name in ("counts", "occupied", "frame_judged"):
            value = publication.value(f"@logic/occupancy/{name}")
            assert value.schema.point_table.columns == (parent_column,), name
        # SITE is CELL data, carried by the calibration's one site axis.
        for name in ("counts", "occupied"):
            value = publication.value(f"@logic/occupancy/{name}")
            (site_axis,) = value.schema.cell_schema.data_axes
            assert site_axis == calibration.site_map.site_axis, name
            assert site_axis.role is SITE
            assert site_axis.coordinates == (1,)
            # The ids identify sites to other records; the axis is read by
            # a person, and reads 1..n.
            assert site_axis.coordinate_labels is None
            assert tuple(site_axis.coordinates) == tuple(range(1, len(site_ids) + 1))
            assert value.values.shape == (1, windows, 1), name
        from zlc_runtime import DatasetCoverage

        assert all(
            isinstance(value.coverage, DatasetCoverage)
            and value.coverage.complete
            for value in publication.signals.values()
        )
        expected_geometry = image_point_overlay_geometry(
            source.snapshot,
            calibration.site_map.centers_xy,
            calibration.site_map.site_ids,
            status_axis=calibration.site_map.site_axis,
            coordinates_are_indices=True,
        )
        expected_geometry["point_ids"] = tuple(expected_geometry["point_ids"])
        expected_geometry["coordinates_xy"] = tuple(
            tuple(center) for center in expected_geometry["coordinates_xy"]
        )
        expected_geometry["labels"] = tuple(expected_geometry["labels"])
        expected_geometry["status_coordinates"] = tuple(
            expected_geometry["status_coordinates"]
        )
        expected_record = {
            "node": "occupancy",
            IMAGE_POINT_OVERLAY_GEOMETRY_RECORD: expected_geometry,
            "parameters": {
                "frames_signal": source_name,
                "calibration_path": str(calibration_path.resolve()),
                "model_kind": "box",
            },
        }
        assert publication.run_record == expected_record
        assert all(
            value.run_record == expected_record
            for value in publication.signals.values()
        )
        assert "device_snapshots" not in publication.run_record
        assert source.run_record["device_snapshots"]["camera"][
            "exposure_seconds"
        ] == actual.exposure_seconds
        assert not plane.is_generation_live("@logic/occupancy/occupied")
        assert plane.direct_parent_publications(publication) == (result.publication,)
    finally:
        if host.running:
            host.cancel("cleanup")
            host.poll()
        host.shutdown()


def test_a_live_monitor_signal_does_carry_coverage(bench) -> None:
    """The other side of the boundary: a live signal uses latest-only."""

    plane, camera, sequencer, _windows = bench
    node = CameraMeasurementNode(
        camera=camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 0, 1),
        signal_plane=plane,
        producer="cm",
    )
    monitor = node.monitor()
    try:
        sequencer.fire()
        sequencer.wait_done(1.0)
        deadline = time.monotonic() + 5.0
        while monitor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        value = plane.freeze().value(node.signal_key("frames"))
        assert value is not None, "the monitor published nothing"
        assert value.coverage is not None, "a live signal reports its current window"
    finally:
        monitor.close()
