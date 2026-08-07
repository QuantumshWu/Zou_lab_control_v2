from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zlc_pulse
from zlc_pulse.wire import build_fingerprint
from zlc_runtime import SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera_measurement import CameraMeasurementNode, MonitorCapture
from zlc_atom.nodes.calibration import CalibrationTask
from zlc_atom.nodes.occupancy import OccupancyProcessor
from zlc_atom.nodes._framework.pulse_source import resolve_pulse
from zlc_atom.nodes.calibration.calibration import FrameContract, calibrate

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


FIXTURES = Path(__file__).parent / "fixtures"


def _oracle() -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "main_readout_oracle.npz", allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _short_stamps(result: object) -> dict[str, object]:
    """The run the short frames came from, which a derived signal inherits."""

    from zlc_atom.nodes.occupancy.processor import inherited_stamps

    publication = result.short.publication
    name = next(iter(publication.signals))
    return inherited_stamps(publication.signals[name].snapshot)


def test_editable_runtime_and_pulse_packages_run_the_virtual_chain_to_frozen_oracle() -> None:
    assert callable(build_fingerprint)
    manifest = json.loads((FIXTURES / "main_readout_oracle.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "main-readout-oracle"
    assert hashlib.sha256((FIXTURES / "main_readout_oracle.npz").read_bytes()).hexdigest() == "ec0194edbe0ea55cad64c70d780939c3cd5f4a3b419e20997e337359965386aa"

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        measurement = CameraMeasurementNode(
            camera=installation.device("camera"),
            signal_plane=plane,
            timeout=1.0,
        )
        monitor = measurement.monitor(buffer_frames=1)
        assert isinstance(monitor, MonitorCapture)
        sequencer = installation.device("sequencer")
        sequencer.load(resolve_pulse("calibration", search_paths=(REPO_ROOT / "pulses",)).program)
        sequencer.fire()
        sequencer.wait_done(1.0)
        assert monitor.poll() is not None
        monitor_front = plane.freeze()
        assert measurement.signal_key("frames") in monitor_front.signals
        monitor.close()

        task_result = CalibrationTask(
            camera=installation.device("camera"),
            sequencer=sequencer,
            signal_plane=plane,
            pulse_search_paths=(REPO_ROOT / "pulses",),
            expected_centers_xy=installation.world.geometry.site_centers_xy,
        ).run()
        occupancy_node = OccupancyProcessor(
            task_result.calibration,
            signal_plane=plane,
        )
        occupancy = occupancy_node.process(
            task_result.short.frames, **_short_stamps(task_result)
        )
        assert occupancy.counts.shape == (30, 6)
        np.testing.assert_allclose(occupancy.artifacts["rate"].block.values[:, 0, 0], occupancy.rate)

        oracle = _oracle()
        result = calibrate(
            oracle["input_reference_frames"],
            oracle["input_short_frames"],
            frame_contract=FrameContract((34, 40), exposure_seconds=0.005),
            grid_shape=(2, 3),
            expected_centers_xy=oracle["centers_row_major"],
        )
        np.testing.assert_array_equal(result.report["predictions"], oracle["pred_box"])
        assert int(np.count_nonzero(result.report["predictions"] != oracle["input_latent_occupancy"])) == 29
    finally:
        plane.close()
        installation.close()
