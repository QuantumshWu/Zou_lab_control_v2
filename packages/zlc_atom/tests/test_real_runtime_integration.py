"""The runtime, pulse and atom packages run the virtual chain together.

The installed-product evidence lane (``zou_lab_control.__main__``) runs this
file against a fresh wheel: the packages resolve, arm, fire, publish and
calibrate.  How well the readout then recovers the atoms is asked on frozen
frames by test_readout_against_known_truth.py.
"""

from __future__ import annotations

from pathlib import Path

from zlc_runtime import SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera_measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
    MonitorCapture,
)
from zlc_atom.nodes.calibration import CalibrationTask
from zlc_atom.nodes.occupancy import OccupancyProcessor
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from tests.fakes import camera_cycle_snapshot
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE, calibration_request


def test_the_packages_run_the_virtual_chain_from_fire_to_calibration(
    tmp_path: Path,
) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    task_plane = SignalDataPlane()
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
            sequencer=sequencer,
            api_values={
                "duration:long_before": 0.02,
                "duration:short": 0.005,
                "duration:long_after": 0.02,
            },
        )
        arm_sequencer(sequencer, pulse)
        sequencer.fire(run_repeats=1, scan_repeats=1)
        sequencer.wait_done(1.0)
        assert monitor.poll() is not None
        monitor_front = plane.freeze()
        assert measurement.signal_key("frames") in monitor_front.signals
        monitor.close()

        task_result = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=sequencer,
            request=calibration_request(),
            pulse_sequence=IMAGING_PULSE_RESOURCE.value,
            pulse_path=IMAGING_PULSE_RESOURCE.path,
            signal_plane=task_plane,
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
    finally:
        task_plane.close()
        plane.close()
        installation.close()
