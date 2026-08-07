from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_atom.devices.camera.world import SimulationWorld
from zlc_atom.install import create_installation
from zlc_atom.nodes._framework.pulse_source import resolve_pulse
from zlc_atom.nodes.camera_measurement import CameraMeasurementNode
from zlc_atom.nodes.calibration.calibration import extract_box_signals
from zlc_atom.devices.sequencer.virtual import DictRegisterTransport, VirtualPulseStreamer
from zlc_runtime import SignalDataPlane

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _program_opening(windows: int) -> SimpleNamespace:
    """A stand-in program that answers the way a compiled one does.

    A real CompiledProgram is asked how many camera windows it opens and works
    it out from its own edges; a double that hands over a bare number instead
    lets the twin accept a shape no program has.
    """

    return SimpleNamespace(camera_window_count=lambda _channel: windows)


def test_qcmos_parameters_and_derived_poisson_signal_are_single_world_physics() -> None:
    world = SimulationWorld(seed=4)
    assert world.atom_sigma_px == pytest.approx(0.7)
    assert world.background_rate == pytest.approx(300.0)
    assert world.atom_rate == pytest.approx(1100.0)
    assert world.conversion_e_per_count == pytest.approx(0.107)
    assert world.read_noise_e == pytest.approx(0.43)
    assert world.offset_counts == pytest.approx(200.0)

    exposure = 0.02
    assert world.atom_rate * exposure == pytest.approx(22.0)
    world.set_occupancy(np.zeros(6, dtype=bool))
    dark = np.asarray([world.render_frame(index, exposure_seconds=exposure) for index in range(24)])
    world.set_occupancy(np.ones(6, dtype=bool))
    bright = np.asarray([world.render_frame(index, exposure_seconds=exposure) for index in range(24)])
    assert dark.dtype == np.dtype("<u2")
    assert bright.dtype == np.dtype("<u2")
    assert float(np.mean(bright)) > float(np.mean(dark))
    assert float(np.var(dark)) > 0.0


def test_virtual_pulse_fire_uses_loaded_camera_window_count() -> None:
    world = SimulationWorld(seed=1)
    streamer = VirtualPulseStreamer(
        transport=DictRegisterTransport(),
        world=world,
        camera_trigger_channel="camera",
    )
    streamer.open()
    streamer.load(_program_opening(3))
    streamer.fire()
    streamer.wait_done(1.0)
    assert world.fire_count == 1
    streamer.close()


def test_calibration_bracket_keeps_one_shot_occupancy_and_exposure_scaling() -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        camera = installation.device("camera")
        sequencer = installation.device("sequencer")
        measurement = CameraMeasurementNode(camera=camera, signal_plane=plane)
        repeats = 30
        pulse = resolve_pulse("calibration", search_paths=(REPO_ROOT / "pulses",))
        capture = measurement.prepare(repeat=repeats, frames_per_cycle=3)
        sequencer.load(pulse.program)
        expected = []
        for _ in range(repeats):
            expected.append(installation.world.occupancy)
            sequencer.fire()
            sequencer.wait_done(1.0)
        result = capture.collect()
        labels = np.asarray(expected, dtype=bool)
        centers = installation.world.geometry.site_centers_xy
        values = np.asarray(
            [
                [extract_box_signals(record.image, centers, radius=1) for record in cycle]
                for cycle in result.cycles
            ],
            dtype=float,
        )

        def accuracy(window: np.ndarray) -> float:
            bright = float(np.mean(window[labels]))
            dark = float(np.mean(window[~labels]))
            return float(np.mean((window > (bright + dark) * 0.5) == labels))

        long_before, short, long_after = (values[:, index, :] for index in range(3))
        assert accuracy(long_before) >= 0.95
        assert accuracy(short) >= 0.90
        assert accuracy(long_after) >= 0.95
        np.testing.assert_allclose(
            np.mean(long_before, axis=0),
            np.mean(long_after, axis=0),
            atol=10.0,
        )
        long_contrast = 0.5 * (
            float(np.mean(long_before[labels]) - np.mean(long_before[~labels]))
            + float(np.mean(long_after[labels]) - np.mean(long_after[~labels]))
        )
        short_contrast = float(np.mean(short[labels]) - np.mean(short[~labels]))
        assert short_contrast / long_contrast == pytest.approx(0.25, rel=0.20)
        assert installation.world.fire_count == repeats
    finally:
        plane.close()
        installation.close()
