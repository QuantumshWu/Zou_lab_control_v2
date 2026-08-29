from __future__ import annotations

import json
from pathlib import Path

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
from tests.fakes import camera_cycle_snapshot
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


FIXTURES = Path(__file__).parent / "fixtures"


def _calibration_request() -> CalibrationRequest:
    return CalibrationRequest(
        camera_key="camera",
        sequencer_key="sequencer",
        pulse_template="imaging_template.json",
        repeats=30,
        reference_exposure_seconds=0.02,
        readout_exposure_seconds=0.005,
        reference_before_slot=1,
        readout_slot=2,
        reference_after_slot=3,
        default_model_kind=ReadoutModelKind.BOX,
        threshold_method="gaussian",
        box_half_width=1,
        box_reducer="mean",
        psf_half_width=3,
        psf_padding=3,
        detection_spot_sigma=1.0,
        detection_sigma=6.0,
    )


def test_editable_runtime_and_pulse_packages_run_the_virtual_chain_to_frozen_oracle(
    tmp_path: Path,
) -> None:
    assert callable(build_fingerprint)
    manifest = json.loads((FIXTURES / "main_readout_oracle.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "readout-known-truth-run"

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        measurement = CameraMeasurementNode(
            camera=installation.device("camera"),
            request=CameraMeasurementRequest("camera", 0.02, None, 0, 1),
            signal_plane=plane,
        )
        monitor = measurement.monitor()
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
            signal_plane=SignalDataPlane(),
        ).run(tmp_path)
        figure_directory = task_result.artifact_path.parents[1] / "figures"
        report_images = tuple(sorted(figure_directory.glob("*.png")))
        assert tuple(path.name for path in report_images) == (
            "actual_fidelity.png",
            "box.png",
            "gaussian_fidelity.png",
            "psf.png",
            "psf_kernels.png",
            "site_map.png",
            "uniform_psf.png",
        )
        assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in report_images)
        assert tuple(
            path.name for path in sorted(figure_directory.glob("*.npz"))
        ) == tuple(path.with_suffix(".npz").name for path in report_images)
        occupancy_node = OccupancyProcessor(
            task_result.calibration,
        )
        occupancy = occupancy_node.process(
            camera_cycle_snapshot(
                [(record,) for record in task_result.capture.short]
            ),
        )
        assert occupancy.counts.shape == (
            30,
            1,
            task_result.calibration.n_sites,
        )

        # What this test is for ends here: that the INSTALLED packages
        # resolve, arm, fire, publish and calibrate.  How well the readout
        # then recovers the atoms is a different question, asked on the same
        # frozen frames by test_readout_against_known_truth.py against a
        # floor that moved when the answer did -- and this second copy of it
        # did not, so it went on demanding 0.90 of a threshold fit that had
        # stopped reading the truth labels and settled at 0.869.  One claim,
        # one owner.
    finally:
        plane.close()
        installation.close()
