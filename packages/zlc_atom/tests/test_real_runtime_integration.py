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
from zlc_atom.nodes.calibration import (
    CalibrationRequest,
    CalibrationTask,
    ReadoutModelKind,
)
from zlc_atom.nodes.occupancy import OccupancyProcessor
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from zlc_atom.nodes.calibration.calibration import FrameContract, calibrate
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE

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
        default_model_kind=ReadoutModelKind.BOX,
        threshold_method="empirical",
        box_half_width=1,
        box_reducer="mean",
        psf_half_width=3,
        psf_padding=3,
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
        pulse = resolve_pulse(
            IMAGING_PULSE_RESOURCE.value,
            path=IMAGING_PULSE_RESOURCE.path,
            board=sequencer.describe(),
            api_values={
                "reference_probe_duration_before": 0.02,
                "readout_probe_duration": 0.005,
                "reference_probe_duration_after": 0.02,
            },
        )
        arm_sequencer(sequencer, pulse)
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
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            artifact_directory=tmp_path,
        ).run()
        report_directory = task_result.artifact_path.with_suffix("") / "report"
        report_images = tuple(sorted(report_directory.glob("*.png")))
        assert tuple(path.name for path in report_images) == (
            "box.png",
            "psf.png",
            "site_map.png",
            "uniform_psf.png",
        )
        assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in report_images)
        occupancy_node = OccupancyProcessor(
            task_result.calibration,
            signal_plane=plane,
        )
        occupancy = occupancy_node.process(
            task_result.short,
            generation="calibration-task",
            revision=1,
        )
        assert occupancy.counts.shape == (30, 35)
        np.testing.assert_allclose(occupancy.artifacts["rate"].block.values[:, 0, 0], occupancy.rate)

        oracle = _oracle()
        result = calibrate(
            oracle["input_reference_frames"],
            oracle["input_short_frames"],
            frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
        )
        box_report = result.report["models"]["box"]
        np.testing.assert_array_equal(box_report["predictions"], oracle["pred_box"])
        assert int(np.count_nonzero(box_report["predictions"] != oracle["input_latent_occupancy"])) == 29
    finally:
        plane.close()
        installation.close()
