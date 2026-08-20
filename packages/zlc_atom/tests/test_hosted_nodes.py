"""A NodeHost can drive domain nodes without conflating role and data extent."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.install import create_installation
from zlc_atom.nodes._framework import SelectionMapping
from zlc_atom.nodes._framework.descriptor import NodeKind
from zlc_atom.nodes._framework.discovery import discover_logic_nodes
from zlc_atom.nodes.camera_measurement import logic_node as camera_logic_node
from zlc_atom.nodes.camera_measurement.measurement import (
    CAMERA_FRAMES_OUTPUT,
    CameraMeasurementNode,
    CameraMeasurementRequest,
    _finite_cycle_output,
    _strict_cycle_ordinals,
    _strict_terminal,
)
from zlc_atom.devices.camera import CameraCaptureTerminalRecord, CameraFrameRecord
from zlc_data import READOUT_EVENT, SPATIAL_X, SPATIAL_Y
from zlc_runtime import SelectionRange, SelectionState
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane

from tests.pulse_fixture import CALIBRATION_FRAMES_PER_CYCLE, build_calibration_pulse


def _camera_host(
    node: CameraMeasurementNode,
    plane: SignalDataPlane,
    wake: Event,
) -> NodeHost:
    return NodeHost(
        node,
        plane,
        wake.set,
        instance_id=node.instance_id,
        kind="measurement",
        dataset_output_declarations=(CAMERA_FRAMES_OUTPUT,),
    )


def test_descriptor_role_is_independent_of_camera_measurement_extent() -> None:
    descriptors = {descriptor.api_name: descriptor for descriptor in discover_logic_nodes()}
    camera = descriptors["camera_measurement"]

    assert camera.kind is NodeKind.MEASUREMENT
    assert descriptors["calibration"].kind is NodeKind.TASK
    assert descriptors["occupancy"].kind is NodeKind.PROCESSOR
    assert camera.authoring_schema.project_values({})["repeat"] == 0
    assert camera.authoring_schema.project_values({"repeat": 0})["repeat"] == 0
    assert camera.authoring_schema.project_values({"repeat": 3})["repeat"] == 3


def test_authoring_schema_rejects_lossy_integer_projection() -> None:
    schema = AuthoringSchema(
        (
            AuthoringField("count", "int", "Count", required=True),
            AuthoringField("origin", "pair", "Origin", required=True),
        )
    )

    assert schema.project_values({"count": 3, "origin": (4, 5)}) == {
        "count": 3,
        "origin": [4, 5],
    }
    assert schema.project_values({"count": "3", "origin": "4, 5"}) == {
        "count": 3,
        "origin": [4, 5],
    }
    with pytest.raises((TypeError, ValueError)):
        schema.project_values({"count": 1.5, "origin": (4, 5)})
    with pytest.raises((TypeError, ValueError)):
        schema.project_values({"count": 3, "origin": (4.5, 5)})


def test_camera_descriptor_builds_repeat_zero_and_finite_measurements() -> None:
    descriptor = next(
        value
        for value in discover_logic_nodes()
        if value.api_name == "camera_measurement"
    )
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        infinite = descriptor.instantiate(
            camera=installation.capability("camera.adapter"),
            camera_key="camera",
            signal_plane=plane,
            exposure_seconds=0.01,
            roi_x=2,
            roi_y=3,
            roi_width=20,
            roi_height=16,
            repeat=0,
        )
        finite = descriptor.instantiate(
            camera=installation.capability("camera.adapter"),
            camera_key="camera",
            signal_plane=plane,
            repeat=2,
        )
        assert infinite.repeat == 0
        assert infinite.request.camera_key == "camera"
        assert infinite.request.exposure_seconds == 0.01
        assert infinite.request.roi_xywh == (2, 3, 20, 16)
        assert finite.repeat == 2
        assert finite.request.roi_xywh is None
        with pytest.raises(ValueError, match="all four fields"):
            descriptor.instantiate(
                camera=installation.capability("camera.adapter"),
                camera_key="camera",
                signal_plane=plane,
                roi_x=2,
            )
    finally:
        installation.close()
        plane.close()


def test_camera_descriptor_maps_image_area_to_sensor_roi_draft() -> None:
    print(camera_logic_node.__file__)
    descriptor = camera_logic_node.LOGIC_NODE
    draft = descriptor.authoring_schema.project_values(
        {
            "roi_x": 100,
            "roi_y": 200,
            "roi_width": 90,
            "roi_height": 80,
        }
    )
    selection = SelectionState(
        plot_kind="image",
        selector_kind="area",
        ranges=(
            SelectionRange(
                "camera_measurement.frames.2.spatial-x", 1.2, 5.8
            ),
            SelectionRange("spatial-y", -3.2, 6.0),
        ),
    )

    patch = descriptor.selection_patch(
        selection,
        draft=draft,
        context={
            "frame_shape_yx": (10, 30),
            "sensor_shape_yx": (40, 100),
            "binning_yx": (2, 3),
            "roi_origin_yx": (7, 11),
        },
    )

    # A frame's axes carry the sensor pixels it covers, so the region IS the
    # ROI: x from 2 to just past 5 (one 3-wide sample), y from 0 (the region
    # reaches off the sensor) to just past 6.  Adding the current origin on top
    # of coordinates that already include it moved every region by the origin.
    assert patch == {
        "roi_x": 2,
        "roi_y": 0,
        "roi_width": 6,
        "roi_height": 8,
    }
    assert all(type(value) is int for value in patch.values())
    # And the same fields, read back from what the run is set to: what an
    # operator sees in the form once the region is taken away.
    assert descriptor.applied_selection_values(
        selection,
        context={
            "frame_shape_yx": (10, 30),
            "sensor_shape_yx": (40, 100),
            "binning_yx": (2, 3),
            "roi_origin_yx": (7, 11),
            "roi_shape_yx": (20, 90),
        },
    ) == {"roi_x": 11, "roi_y": 7, "roi_width": 90, "roi_height": 20}
    assert descriptor.authoring_schema.project_values({**draft, **patch}) == {
        **draft,
        **patch,
    }
    assert [
        (mapping.plot_kind, mapping.selector_kind)
        for mapping in descriptor.selection_mappings
    ] == [("image", "area")]
    assert isinstance(descriptor.selection_mappings[0], SelectionMapping)
    assert descriptor.selection_patch(
        SelectionState(
            plot_kind="image",
            selector_kind="x_range",
            ranges=(SelectionRange("spatial-x", 1.0, 3.0),),
        ),
        draft=draft,
        context={
            "frame_shape_yx": (10, 30),
            "sensor_shape_yx": (40, 100),
            "binning_yx": (2, 3),
            "roi_origin_yx": (7, 11),
        },
    ) is None


def test_a_node_host_runs_a_camera_measurement_to_completion() -> None:
    """The Start button's path: construct a host, start it, get a publication."""

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    host = None
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program = build_calibration_pulse(sequencer.describe())
        sequencer.load(program)
        windows = CALIBRATION_FRAMES_PER_CYCLE

        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=1,
                frames_per_cycle=windows,
            ),
            signal_plane=plane,
            producer="cm",
        )
        wake = Event()
        host = _camera_host(node, plane, wake)
        host.start()

        # The host arms the camera on its worker; the triggers are ours to supply.
        deadline = time.monotonic() + 10.0
        fired = False
        while time.monotonic() < deadline:
            if not fired and camera.is_armed if hasattr(camera, "is_armed") else not fired:
                sequencer.fire()
                sequencer.wait_done(1.0)
                fired = True
            host.poll()
            if host.observation.terminal:
                break
            time.sleep(0.01)

        observation = host.observation
        assert observation.phase == "done", f"host ended in {observation.phase}: {observation.error}"
        assert host.final_result is not None
        assert node.actual_working_point is not None
        assert node.actual_working_point.exposure_seconds == 0.02
        declarations = host.dataset_output_declarations
        assert tuple(value.name for value in declarations) == ("frames",)
        publication = plane.latest_publication(host.signal_key("frames"))
        assert publication is not None
        value = publication.value(host.signal_key("frames"))
        assert value is not None
        # One signal for the whole cycle: the frames ARE the point rows.
        schema = value.snapshot.block.schema
        assert tuple(
            axis.role for axis in schema.cell_schema.data_axes
        ) == (SPATIAL_Y, SPATIAL_X)
        frame_column = schema.point_table.columns[0]
        assert frame_column.name == "frame"
        assert frame_column.role == READOUT_EVENT
        assert frame_column.values == tuple(range(windows))
        frames = np.asarray(value.snapshot.block.values)
        assert frames.shape[:2] == (1, windows)
        record = publication.run_record
        assert value.run_record == record
        assert record["node"] == "cm"
        assert record["parameters"] == {
            "exposure_seconds": 0.02,
            "roi_xywh": None,
            "repeat": 1,
            "frames_per_cycle": windows,
            "photoelectrons": False,
        }
        assert record["named_devices"] == {"camera": "camera"}
        actual = record["device_snapshots"]["camera"]
        assert actual["exposure_seconds"] == 0.02
        assert tuple(actual["frame_shape_yx"]) == tuple(frames.shape[-2:])
        assert actual["dtype"] == frames.dtype.str
    finally:
        if host is not None:
            host.shutdown()
        installation.close()
        plane.close()


def test_a_node_host_runs_and_stops_repeat_zero_camera_measurement() -> None:
    plane = SignalDataPlane()
    installation = create_installation("virtual")
    host = None
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program = build_calibration_pulse(sequencer.describe())
        sequencer.load(program)

        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=0,
                frames_per_cycle=CALIBRATION_FRAMES_PER_CYCLE,
            ),
            signal_plane=plane,
            producer="cm-live",
        )
        wake = Event()
        host = _camera_host(node, plane, wake)
        host.start()

        deadline = time.monotonic() + 5.0
        while not camera.capture_state() and time.monotonic() < deadline:
            host.poll()
            time.sleep(0.005)
        assert camera.capture_state(), "hosted repeat-zero worker did not arm the camera"

        sequencer.fire()
        sequencer.wait_done(1.0)
        signal_key = host.signal_key("frames")
        live_value = None
        deadline = time.monotonic() + 5.0
        while live_value is None and time.monotonic() < deadline:
            host.poll()
            live_value = plane.freeze().value(signal_key)
            time.sleep(0.005)
        assert live_value is not None, "hosted repeat-zero worker published no latest frame"
        assert np.asarray(live_value.snapshot.block.values).size
        live_record = live_value.run_record
        assert live_record["node"] == "cm-live"
        assert live_record["parameters"]["repeat"] == 0
        assert live_record["named_devices"] == {"camera": "camera"}
        assert live_record["device_snapshots"]["camera"]["exposure_seconds"] == 0.02

        host.cancel("test completed")
        deadline = time.monotonic() + 5.0
        while not host.observation.terminal and time.monotonic() < deadline:
            host.poll()
            time.sleep(0.005)
        assert host.observation.terminal
        assert camera.capture_state() is False
        assert plane.latest_publication(signal_key) is None
    finally:
        if host is not None:
            if host.observation.running:
                host.cancel("test cleanup")
                deadline = time.monotonic() + 5.0
                while host.observation.running and time.monotonic() < deadline:
                    host.poll()
                    time.sleep(0.005)
            host.shutdown()
        installation.close()
        plane.close()


def test_the_hosted_and_direct_paths_share_one_acquisition() -> None:
    """Two implementations of a shot is how a virtual bench starts to lie.

    The hosted entry differs from the notebook entry only in who owns the
    generation and where the publication goes; the arming, reading and snapshot
    building are the same code.
    """

    source = (
        ROOT / "src" / "zlc_atom" / "nodes" / "camera_measurement" / "measurement.py"
    ).read_text(encoding="utf-8")
    execute = source[source.index("    def execute(self, context"):]
    execute = execute[: execute.index("\n    def ", 1)] if "\n    def " in execute[1:] else execute
    # execute() delegates; it must not grow its own read loop or publish path.
    assert "self.prepare(" in execute
    assert "self.monitor(" in execute
    assert "collect(" in execute
    assert "commit_live" in execute
    assert "attach_live_outputs" not in execute
    assert "publish_final" not in execute
    assert "read_frame_records" not in execute
    assert "snapshot_from_array" not in execute


def test_a_finite_run_shows_its_dataset_filling_and_stops_when_asked() -> None:
    """Two things an operator does with a repeat=N run: watch it, and stop it.

    WATCH: the dataset is every cycle the run will take, filling up -- so the
    live output states that whole geometry and counts the cells measured so
    far.  It used to publish nothing at all until the last cycle had been
    taken, which for a long run is a blank panel for as long as it runs.

    STOP: a cancel is only ever seen BETWEEN reads, and the read was the
    camera's whole timeout (2 s virtual, 10 s on the qCMOS), so Stop was
    refused for that long while the console said the node was still stopping.
    """

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    host = None
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program = build_calibration_pulse(sequencer.describe())
        sequencer.load(program)
        repeats = 4
        windows = CALIBRATION_FRAMES_PER_CYCLE
        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=repeats,
                frames_per_cycle=windows,
            ),
            signal_plane=plane,
            producer="cm-live",
        )
        wake = Event()
        host = _camera_host(node, plane, wake)
        host.start()
        signal = host.signal_key("frames")

        # One cycle event at a time; the Runtime-owned current Dataset keeps
        # the fixed authored geometry and invalid future cells.
        seen: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []
        deadline = time.monotonic() + 20.0
        fired = 0
        while time.monotonic() < deadline and not host.observation.terminal:
            if fired < repeats:
                sequencer.fire()
                sequencer.wait_done(2.0)
                fired += 1
            host.poll()
            value = plane.freeze().value(signal)
            if value is not None and getattr(value, "coverage", None) is not None:
                event = np.asarray(value.snapshot.block.values)
                current = plane.current_dataset(
                    signal,
                    plane.latest_publication(signal),
                )
                seen.append(
                    (
                        tuple(event.shape[:2]),
                        tuple(current.block.values.shape[:2]),
                        int(value.coverage.written_cells),
                        int(value.coverage.total_cells),
                    )
                )
            time.sleep(0.01)

        assert seen, "a finite run published nothing while it ran"
        assert {event for event, _current, _written, _total in seen} == {
            (1, windows)
        }
        assert {current for _event, current, _written, _total in seen} == {
            (repeats, windows)
        }
        assert {
            total for _event, _current, _written, total in seen
        } == {repeats * windows}
        written = [written for _event, _current, written, _total in seen]
        assert written == sorted(written)
        assert written[0] < repeats * windows, written
        assert host.observation.phase == "done", host.observation.error

        # The event schema never changes at terminal; explicit materialization
        # returns the one sealed full Dataset.
        publication = plane.latest_publication(signal)
        assert publication is not None
        final = publication.value(signal)
        assert final is not None
        assert np.asarray(final.snapshot.block.values).shape[:2] == (1, windows)
        assert plane.current_dataset(signal).block.values.shape[:2] == (
            repeats,
            windows,
        )
    finally:
        if host is not None:
            host.shutdown()
        installation.close()
        plane.close()


def test_a_finite_run_stops_within_a_read_slice_not_a_camera_timeout() -> None:
    """The cancel latency is the loop's slice, not the device's deadline."""

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    host = None
    try:
        camera = installation.capability("camera.adapter")
        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=32,
                frames_per_cycle=1,
            ),
            signal_plane=plane,
            producer="cm-stop",
        )
        wake = Event()
        host = _camera_host(node, plane, wake)
        host.start()
        # Wait until the run is inside its first read, waiting for a trigger
        # that will never come.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and node.actual_working_point is None:
            host.poll()
            time.sleep(0.005)
        assert node.actual_working_point is not None, "the camera never armed"

        started = time.monotonic()
        host.cancel("the operator pressed Stop")
        while time.monotonic() - started < 30.0:
            host.poll()
            if host.observation.terminal:
                break
            time.sleep(0.005)
        latency = time.monotonic() - started
        assert host.observation.terminal, "the run never stopped"
        # Generous against a loaded machine, and still far below the camera's
        # own timeout, which is what this used to wait out.
        assert latency < float(camera.timeout) / 2.0, latency
    finally:
        if host is not None:
            host.shutdown()
        installation.close()
        plane.close()


def test_repeat_100_builds_only_one_cycle_per_camera_commit() -> None:
    shape = (96, 128)
    node = SimpleNamespace(
        instance_id="copy-proof",
        generation="copy-proof-generation",
        repeat=100,
        frames_per_cycle=1,
        actual_working_point=None,
        run_record={},
    )
    payload = 0
    for index in range(100):
        cycle = (
            CameraFrameRecord(
                np.full(shape, index, dtype=np.uint16),
                index,
            ),
        )
        output = _finite_cycle_output(node, cycle, index)
        assert output.snapshot.block.values.shape == (1, 1, *shape)
        assert output.canonical_schema.physical_shape == (100, 1, *shape)
        assert output.cell_origin == (index, 0)
        payload += output.snapshot.block.values.nbytes
    assert payload == 100 * np.empty(shape, dtype=np.uint16).nbytes


def test_stop_partial_is_identical_whether_or_not_ui_freezes() -> None:
    def stopped(*, freeze: bool):
        plane = SignalDataPlane()
        installation = create_installation("virtual")
        host = None
        try:
            camera = installation.capability("camera.adapter")
            sequencer = installation.device("sequencer")
            program = build_calibration_pulse(sequencer.describe())
            sequencer.load(program)
            node = CameraMeasurementNode(
                camera=camera,
                request=CameraMeasurementRequest(
                    camera_key="camera",
                    exposure_seconds=0.02,
                    roi_xywh=None,
                    repeat=4,
                    frames_per_cycle=CALIBRATION_FRAMES_PER_CYCLE,
                ),
                signal_plane=plane,
                producer="cm-stop-proof",
            )
            wake = Event()
            host = _camera_host(node, plane, wake)
            host.start()
            deadline = time.monotonic() + 5.0
            while not camera.capture_state() and time.monotonic() < deadline:
                host.poll()
                time.sleep(0.005)
            sequencer.fire()
            sequencer.wait_done(1.0)
            signal = host.signal_key("frames")
            deadline = time.monotonic() + 5.0
            while plane.latest_publication(signal) is None and time.monotonic() < deadline:
                host.poll()
                time.sleep(0.005)
            assert plane.latest_publication(signal) is not None
            if freeze:
                plane.freeze()
            host.cancel("stop after one cycle")
            deadline = time.monotonic() + 5.0
            while not host.terminal and time.monotonic() < deadline:
                host.poll()
                time.sleep(0.005)
            assert host.observation.phase == "cancelled", host.observation
            snapshot = plane.current_dataset(signal)
            return (
                snapshot.block.schema,
                np.array(snapshot.block.values, copy=True),
                np.array(snapshot.expanded_validity(), copy=True),
            )
        finally:
            if host is not None:
                host.shutdown()
            installation.close()
            plane.close()

    without = stopped(freeze=False)
    with_freeze = stopped(freeze=True)
    assert without[0] == with_freeze[0]
    np.testing.assert_array_equal(without[1], with_freeze[1])
    np.testing.assert_array_equal(without[2], with_freeze[2])
    assert without[2][:1].all() and not without[2][1:].any()


def test_finite_cycle_rejects_a_physical_ordinal_gap() -> None:
    records = (
        CameraFrameRecord(np.zeros((2, 2), dtype=np.uint16), 0),
        CameraFrameRecord(np.zeros((2, 2), dtype=np.uint16), 2),
    )
    with pytest.raises(RuntimeError, match="not contiguous"):
        _strict_cycle_ordinals(
            records,
            expected_start=0,
            frames_per_cycle=2,
        )


def test_finite_capture_rejects_incomplete_terminal_evidence() -> None:
    with pytest.raises(RuntimeError, match="did not prove"):
        _strict_terminal(
            CameraCaptureTerminalRecord(2, True, True, True),
            expected_frames=3,
        )
