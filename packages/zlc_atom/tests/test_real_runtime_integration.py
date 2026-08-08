from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zlc_pulse
from zlc_pulse.wire import build_fingerprint
from zlc_runtime import SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera_measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
    MonitorCapture,
)
from zlc_atom.nodes.calibration import CalibrationRequest, CalibrationTask
from zlc_atom.nodes.occupancy import OccupancyProcessor
from zlc_atom.nodes.calibration.pulse import resolve_pulse
from zlc_atom.nodes.calibration.calibration import FrameContract, calibrate
from tests.pulse_fixture import PULSE_ROOT

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


FIXTURES = Path(__file__).parent / "fixtures"


def _oracle() -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "main_readout_oracle.npz", allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _calibration_request() -> CalibrationRequest:
    return CalibrationRequest(
        camera_key="camera",
        sequencer_key="sequencer",
        pulse_template="imaging_template.json",
        repeats=30,
        reference_exposure_seconds=0.02,
        readout_exposure_seconds=0.005,
        roi_xywh=None,
        integration_method="box",
        threshold_method="empirical",
        integration_half_width=1,
        reducer="mean",
        detection_spot_sigma=1.0,
        detection_min_distance=3,
        detection_sigma=6.0,
        timeout_seconds=2.0,
    )


def test_editable_runtime_and_pulse_packages_run_the_virtual_chain_to_frozen_oracle(
    tmp_path: Path,
) -> None:
    assert callable(build_fingerprint)
    manifest = json.loads((FIXTURES / "main_readout_oracle.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "main-readout-oracle"
    assert hashlib.sha256((FIXTURES / "main_readout_oracle.npz").read_bytes()).hexdigest() == "ec0194edbe0ea55cad64c70d780939c3cd5f4a3b419e20997e337359965386aa"

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        measurement = CameraMeasurementNode(
            camera=installation.device("camera"),
            request=CameraMeasurementRequest("camera", 0.02, None, 0, 1, 1.0),
            signal_plane=plane,
        )
        monitor = measurement.monitor(buffer_frames=1)
        assert isinstance(monitor, MonitorCapture)
        sequencer = installation.device("sequencer")
        sequencer.load(
            resolve_pulse(
                "imaging_template.json",
                search_paths=(PULSE_ROOT,),
                slot_values={
                    "reference_before": 0.02,
                    "readout": 0.005,
                    "reference_after": 0.02,
                },
            ).program
        )
        sequencer.fire()
        sequencer.wait_done(1.0)
        assert monitor.poll() is not None
        monitor_front = plane.freeze()
        assert measurement.signal_key("frames") in monitor_front.signals
        monitor.close()

        task_result = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=sequencer,
            request=_calibration_request(),
            pulse_search_paths=(PULSE_ROOT,),
            artifact_directory=tmp_path,
        ).run()
        occupancy_node = OccupancyProcessor(
            task_result.calibration,
            signal_plane=plane,
        )
        occupancy = occupancy_node.process(
            task_result.short,
            generation="calibration-task",
            revision=1,
        )
        assert occupancy.counts.shape == (30, 6)
        np.testing.assert_allclose(occupancy.artifacts["rate"].block.values[:, 0, 0], occupancy.rate)

        oracle = _oracle()
        result = calibrate(
            oracle["input_reference_frames"],
            oracle["input_short_frames"],
            frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
        )
        np.testing.assert_array_equal(result.report["predictions"], oracle["pred_box"])
        assert int(np.count_nonzero(result.report["predictions"] != oracle["input_latent_occupancy"])) == 29
    finally:
        plane.close()
        installation.close()
