from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest

from zlc_atom.devices.simulation import SimulationWorld, VirtualPulseStreamer
from zlc_atom.install import create_installation
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from zlc_atom.nodes.camera_measurement import CameraMeasurementNode, CameraMeasurementRequest
from zlc_atom.nodes.calibration.calibration import extract_box_signals
from zlc_runtime import SignalDataPlane
from tests.pulse_fixture import (
    CAMERA_CHANNEL,
    IMAGING_PULSE_RESOURCE,
    build_calibration_pulse,
)

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    site_count = len(world.geometry.site_centers_xy)
    world.set_occupancy(np.zeros(site_count, dtype=bool))
    dark = np.asarray([world.render_frame(index, exposure_seconds=exposure) for index in range(24)])
    world.set_occupancy(np.ones(site_count, dtype=bool))
    bright = np.asarray([world.render_frame(index, exposure_seconds=exposure) for index in range(24)])
    assert dark.dtype == np.dtype("<u2")
    assert bright.dtype == np.dtype("<u2")
    assert float(np.mean(bright)) > float(np.mean(dark))
    assert float(np.var(dark)) > 0.0


def test_virtual_sites_have_repeatable_efficiency_and_psf_diversity() -> None:
    world = SimulationWorld(seed=4)
    efficiency = world.site_efficiency
    sigma_xy = world.site_psf_sigma_xy
    angles = world.site_psf_angle_radians
    skew = world.site_psf_skew

    assert efficiency.shape == (35,)
    assert float(np.max(efficiency) / np.min(efficiency)) == pytest.approx(
        2.0, rel=0.10
    )
    assert sigma_xy.shape == (35, 2)
    assert float(np.ptp(sigma_xy[:, 0])) > 0.05
    assert float(np.ptp(sigma_xy[:, 1])) > 0.05
    assert float(np.ptp(angles)) > 0.05
    assert float(np.ptp(skew)) > 0.05
    np.testing.assert_allclose(
        SimulationWorld(seed=4).site_efficiency,
        efficiency,
    )


def test_virtual_shots_randomly_reload_instead_of_alternating_two_patterns() -> None:
    world = SimulationWorld(seed=11)
    patterns: list[np.ndarray] = []
    for _ in range(8):
        world.fire()
        patterns.append(world.occupancy)

    assert len({pattern.tobytes() for pattern in patterns}) > 2
    assert any(
        not np.array_equal(right, np.logical_not(left))
        for left, right in zip(patterns, patterns[1:], strict=True)
    )


def test_virtual_trap_off_time_removes_loaded_atoms() -> None:
    world = SimulationWorld(seed=3)
    world.set_occupancy(np.ones(35, dtype=bool))
    world.fire(trap_off_seconds=1.0)
    assert not np.any(world.occupancy)


def test_virtual_pulse_fire_uses_loaded_camera_window_count() -> None:
    world = SimulationWorld(seed=1)
    streamer = VirtualPulseStreamer(
        world=world,
        camera_trigger_channel=CAMERA_CHANNEL,
    )
    streamer.open()
    try:
        program, _metadata = build_calibration_pulse(streamer.describe())
        streamer.load(program)
        started = time.monotonic()
        streamer.fire()
        time.sleep(program.duration_seconds * 0.25)
        assert streamer.wait_done(0.0) is None
        assert streamer.wait_done(1.0) is not None
        assert time.monotonic() - started >= program.duration_seconds * 0.8
        streamer.fire(forever=True)
        deadline = time.monotonic() + 0.5
        while world.fire_count < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert world.fire_count >= 3
        streamer.safe()
        stopped_at = world.fire_count
        time.sleep(0.08)
        assert world.fire_count == stopped_at
    finally:
        streamer.close()


def test_calibration_bracket_keeps_one_shot_occupancy_and_exposure_scaling() -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        camera = installation.device("camera")
        sequencer = installation.device("sequencer")
        repeats = 30
        measurement = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest("camera", 0.02, None, repeats, 3),
            signal_plane=plane,
        )
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
        capture = measurement.prepare()
        arm_sequencer(sequencer, pulse)
        expected = []
        for _ in range(repeats):
            sequencer.fire()
            expected.append(installation.world.occupancy)
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
