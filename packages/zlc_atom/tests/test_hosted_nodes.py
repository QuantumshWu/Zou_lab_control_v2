"""A NodeHost can drive domain nodes without conflating role and data extent."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from threading import Event

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
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_runtime import SelectionRange, SelectionState
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane

from tests.pulse_fixture import build_calibration_pulse


class _DetachRecordingPlane(SignalDataPlane):
    def __init__(self) -> None:
        super().__init__()
        self.detached: list[object] = []

    def detach_live(self, node: object) -> None:
        self.detached.append(node)
        super().detach_live(node)


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

    assert patch == {
        "roi_x": 17,
        "roi_y": 1,
        "roi_width": 12,
        "roi_height": 20,
    }
    assert all(type(value) is int for value in patch.values())
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
        program, metadata = build_calibration_pulse(sequencer.describe())
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)
        windows = int(metadata["camera_windows"])

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
        host = NodeHost(node, plane, wake.set)
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
        assert tuple(value.name for value in declarations) == tuple(
            f"frame_{index}" for index in range(windows)
        )
        publication = plane.latest_publication(host.signal_key("frame_0"))
        assert publication is not None
        frames = []
        for index in range(windows):
            value = publication.value(host.signal_key(f"frame_{index}"))
            assert value is not None
            assert tuple(
                axis.role for axis in value.snapshot.block.schema.cell_schema.data_axes
            ) == (SPATIAL_Y, SPATIAL_X)
            frame = np.asarray(value.snapshot.block.values)
            assert frame.shape[:2] == (1, 1)
            frames.append(frame)
        record = publication.run_record
        assert all(
            publication.value(host.signal_key(f"frame_{index}")).run_record == record
            for index in range(windows)
        )
        assert record["node"] == "cm"
        assert record["parameters"] == {
            "exposure_seconds": 0.02,
            "roi_xywh": None,
            "repeat": 1,
            "frames_per_cycle": windows,
        }
        assert record["named_devices"] == {"camera": "camera"}
        actual = record["device_snapshots"]["camera"]
        assert actual["exposure_seconds"] == 0.02
        assert tuple(actual["frame_shape_yx"]) == tuple(frames[0].shape[-2:])
        assert actual["dtype"] == frames[0].dtype.str
        json.dumps(dict(record))
    finally:
        if host is not None:
            host.shutdown()
        installation.close()
        plane.close()


def test_a_node_host_runs_and_stops_repeat_zero_camera_measurement() -> None:
    plane = _DetachRecordingPlane()
    installation = create_installation("virtual")
    host = None
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build_calibration_pulse(sequencer.describe())
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)

        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=0,
                frames_per_cycle=int(metadata["camera_windows"]),
            ),
            signal_plane=plane,
            producer="cm-live",
        )
        wake = Event()
        host = NodeHost(node, plane, wake.set)
        host.start()

        deadline = time.monotonic() + 5.0
        while not camera.capture_state() and time.monotonic() < deadline:
            host.poll()
            time.sleep(0.005)
        assert camera.capture_state(), "hosted repeat-zero worker did not arm the camera"

        sequencer.fire()
        sequencer.wait_done(1.0)
        signal_key = host.signal_key("frame_0")
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
        json.dumps(dict(live_record))

        host.cancel("test completed")
        deadline = time.monotonic() + 5.0
        while not host.observation.terminal and time.monotonic() < deadline:
            host.poll()
            time.sleep(0.005)
        assert host.observation.terminal
        assert camera.capture_state() is False
        assert plane.latest_publication(signal_key) is None
        assert plane.detached == [host]
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
    assert "read_frame_records" not in execute
    assert "snapshot_from_array" not in execute
