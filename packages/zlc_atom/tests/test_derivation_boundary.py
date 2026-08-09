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
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_atom.nodes.occupancy.logic_node import LOGIC_NODE as OCCUPANCY_LOGIC_NODE
from zlc_atom.nodes.occupancy.processor import OccupancyProcessor
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane

from tests.pulse_fixture import build_calibration_pulse


@pytest.fixture
def bench():
    plane = SignalDataPlane()
    installation = create_installation("virtual")
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build_calibration_pulse(sequencer.describe())
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)
        yield plane, camera, sequencer, int(metadata["camera_windows"])
    finally:
        installation.close()
        plane.close()


def _finished_shot(plane, camera, sequencer, windows):
    node = CameraMeasurementNode(
        camera=camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 1, windows, 2.0),
        signal_plane=plane,
        producer="cm",
    )
    capture = node.prepare()
    sequencer.fire()
    sequencer.wait_done(1.0)
    return node, capture.collect()


def test_a_finished_measurement_carries_no_coverage_and_that_is_correct(bench) -> None:
    plane, camera, sequencer, windows = bench
    node, result = _finished_shot(plane, camera, sequencer, windows)
    value = result.publication.value(node.signal_key("frames"))
    assert value.coverage is None, "a finished dataset has nothing left to keep up with"


def test_hosting_a_processor_on_a_finished_signal_derives_once(bench, tmp_path: Path) -> None:
    plane, camera, sequencer, windows = bench
    node, result = _finished_shot(plane, camera, sequencer, windows)
    source_name = node.signal_key("frames")
    source = result.publication.value(source_name)
    assert source is not None
    height, width = source.values.shape[-2:]
    actual = node.actual_working_point
    assert actual is not None
    roi_y, roi_x = actual.roi_origin_yx
    roi_height, roi_width = actual.roi_shape_yx
    frame_contract = FrameContract(
        (height, width),
        sensor_shape=actual.sensor_shape_yx,
        roi_xywh=(roi_x, roi_y, roi_width, roi_height),
        binning_yx=actual.binning_yx,
        exposure_seconds=actual.exposure_seconds,
        camera_id=node.camera_key,
        readout_mode=actual.readout_mode,
    )
    site_ids = ("site_0000",)
    calibration = TrapCalibration(
        SiteMap(
            site_ids,
            np.asarray([[width // 2, height // 2]], dtype=float),
            [True],
            [1.0],
        ),
        (ReadoutModel(site_ids, [0.0], [True], [1.0]),),
        ReadoutModelKind.BOX,
        frame_contract,
    )
    calibration_path = calibration.save(tmp_path / "calibration.json")
    processor = OCCUPANCY_LOGIC_NODE.instantiate(
        calibration_path=str(calibration_path),
        source_signal=source_name,
        signal_plane=plane,
    )
    assert isinstance(processor, OccupancyProcessor)
    assert processor.calibration_path == calibration_path.resolve()
    incompatible = TrapCalibration(
        calibration.site_map,
        calibration.models,
        calibration.default_model_kind,
        FrameContract(
            frame_contract.image_shape,
            sensor_shape=frame_contract.sensor_shape,
            roi_xywh=frame_contract.roi_xywh,
            binning_yx=frame_contract.binning_yx,
            exposure_seconds=frame_contract.exposure_seconds * 2.0,
            camera_id=frame_contract.camera_id,
            readout_mode=frame_contract.readout_mode,
        ),
    )
    incompatible_path = incompatible.save(tmp_path / "incompatible.json")
    incompatible_processor = OCCUPANCY_LOGIC_NODE.instantiate(
        calibration_path=str(incompatible_path),
        source_signal=source_name,
        signal_plane=plane,
    )
    with pytest.raises(ValueError, match="exposure"):
        incompatible_processor.evaluate(source)
    wake = Event()
    host = NodeHost(processor, plane, wake.set)

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
            "@logic/occupancy/valid",
            "@logic/occupancy/rate",
            "@logic/occupancy/frame_judged",
        }
        np.testing.assert_array_equal(
            publication.value("@logic/occupancy/frame_judged").values,
            source.values,
        )
        assert np.all(publication.value("@logic/occupancy/valid").values)
        assert all(
            value.coverage is None and value.transient is False
            for value in publication.signals.values()
        )
        expected_record = {
            "node": "occupancy",
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
        request=CameraMeasurementRequest("camera", 0.02, None, 0, 1, 2.0),
        signal_plane=plane,
        producer="cm",
    )
    monitor = node.monitor(buffer_frames=2)
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
