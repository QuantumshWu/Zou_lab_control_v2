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
from zlc_pulse import (
    AnalogStep,
    PulsePeriod,
    PulseSequence,
    compile_sequence,
    load_streamer_config,
)
from tests.pulse_fixture import (
    CAMERA_CHANNEL,
    IMAGING_PULSE_RESOURCE,
)

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _world_pulse(
    *,
    duration: float = 0.02,
    cooling: bool = False,
    probe: bool = False,
    trap: bool = False,
    camera: bool = False,
    da_x: int = 0,
    da_y: int = 0,
    da_z: int = 0,
) -> PulseSequence:
    target = IMAGING_PULSE_RESOURCE.value.target
    lane_index = {lane: index for index, lane in enumerate(target.raw_lanes)}
    states = [0] * len(target.raw_lanes)
    for name, active in (
        ("cooling", cooling),
        ("probe", probe),
        ("trap", trap),
        ("emCCD", camera),
    ):
        if active:
            states[lane_index[target.by_key[name].lanes[0]]] = 1
    return PulseSequence(
        name="world_physics",
        target=target,
        time_step_ns=20.0,
        periods=(
            PulsePeriod(
                "state",
                duration,
                "s",
                tuple(states),
                (
                    AnalogStep("da_bias_x", "edge", da_x),
                    AnalogStep("da_bias_y", "edge", da_y),
                    AnalogStep("da_bias_z", "edge", da_z),
                ),
            ),
        ),
    )


def _fire_world(world: SimulationWorld, pulse: PulseSequence) -> None:
    config = load_streamer_config()
    world.fire(
        compile_sequence(pulse, config["params"], float(config["clock_hz"]))
    )


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
    dark = np.asarray(
        [
            world.render_frame(index, exposure_seconds=exposure, probe_seconds=0.0)
            for index in range(24)
        ]
    )
    world.set_occupancy(np.ones(site_count, dtype=bool))
    bright = np.asarray(
        [
            world.render_frame(
                index,
                exposure_seconds=exposure,
                probe_seconds=exposure,
            )
            for index in range(24)
        ]
    )
    assert dark.dtype == np.dtype("<u2")
    assert bright.dtype == np.dtype("<u2")
    assert float(np.mean(bright)) > float(np.mean(dark))
    assert float(np.var(dark)) > 0.0


def test_mot_frame_is_uint8_with_a_windowed_separable_spot() -> None:
    """The MOT monitor renders like the Basler it stands in for: Mono8.

    Fired AT the planted optimum, because that is what a compensated MOT looks
    like: the net field is zero, so the spot sits at frame centre and at full
    brightness.  The spot is separable and Poisson samples are drawn only
    inside the +/-8 sigma window, so everything outside it must be pure
    offset-plus-read-noise.
    """

    from zlc_atom.devices.simulation import DEFAULT_MOT_FIELD_OPTIMUM_DAC

    opt_x, opt_y, opt_z = DEFAULT_MOT_FIELD_OPTIMUM_DAC
    world = SimulationWorld(seed=7)
    _fire_world(
        world,
        _world_pulse(cooling=True, trap=True, da_x=opt_x, da_y=opt_y, da_z=opt_z),
    )
    frame = world.render_mot_frame(0, frame_shape_yx=(400, 640))
    assert frame.dtype == np.dtype("|u1")
    assert frame.shape == (400, 640)

    spot = frame[197:204, 317:324].astype(float)
    assert float(np.mean(spot)) > 60.0

    # 8 sigma_x is ~136 px and 8 sigma_y is ~68 px: the left margin and the
    # top margin are outside the sampling window, so they carry only the
    # offset-7 read-noise floor (truncation shifts its mean to ~6.5).
    for margin in (frame[:, :150], frame[:120, :]):
        values = margin.astype(float)
        assert float(np.mean(values)) == pytest.approx(6.5, abs=0.3)
        assert float(np.std(values)) == pytest.approx(1.5, abs=0.3)
        assert int(np.max(margin)) < 20

    world.safe()
    empty = world.render_mot_frame(0, frame_shape_yx=(400, 640))
    assert float(np.mean(empty.astype(float))) == pytest.approx(6.5, abs=0.3)
    assert int(np.max(empty)) < 20


def test_mot_follows_the_net_field_and_is_best_at_the_planted_optimum() -> None:
    """Position and brightness both follow (dac - optimum), the NET field.

    The world plants a non-zero optimum -- the ambient field the bias coils
    exist to cancel -- so an optimiser that starts from zero has somewhere real
    to go.  At the optimum the spot is centred and brightest; away from it the
    spot moves AND dims; at dac zero (nothing compensated) it is measurably
    both off-centre and dimmer.  That gradient is what a field scan climbs.
    """

    from zlc_atom.devices.simulation import DEFAULT_MOT_FIELD_OPTIMUM_DAC

    opt_x, opt_y, opt_z = DEFAULT_MOT_FIELD_OPTIMUM_DAC
    assert (opt_x, opt_y, opt_z) != (0, 0, 0), (
        "a zero optimum makes every optimiser pass trivially"
    )

    def frames(seed: int, *, da_x: int, da_y: int, da_z: int) -> np.ndarray:
        world = SimulationWorld(seed=seed)
        _fire_world(
            world,
            _world_pulse(cooling=True, trap=True, da_x=da_x, da_y=da_y, da_z=da_z),
        )
        return world.render_mot_frame(0, frame_shape_yx=(200, 320))

    at_optimum = frames(5, da_x=opt_x, da_y=opt_y, da_z=opt_z)
    again = frames(5, da_x=opt_x, da_y=opt_y, da_z=opt_z)
    np.testing.assert_array_equal(at_optimum, again)

    def centroid_y(frame: np.ndarray) -> float:
        weights = np.clip(frame.astype(float) - 12.0, 0.0, None)
        rows = np.arange(frame.shape[0], dtype=float)
        return float(np.sum(rows * np.sum(weights, axis=1)) / np.sum(weights))

    def centroid_x(frame: np.ndarray) -> float:
        weights = np.clip(frame.astype(float) - 12.0, 0.0, None)
        columns = np.arange(frame.shape[1], dtype=float)
        return float(np.sum(columns * np.sum(weights, axis=0)) / np.sum(weights))

    assert centroid_x(at_optimum) == pytest.approx((320 - 1) * 0.5, abs=0.5)
    assert centroid_y(at_optimum) == pytest.approx((200 - 1) * 0.5, abs=0.5)

    # A bias field walks the spot only within its OWN size -- the quadrupole
    # zero moves a little -- while the main effect is fewer atoms.  Half the
    # DAC range is half a sigma-ish of walk, never a spot-width of it.
    fwhm_x, fwhm_y = 40.0, 20.0
    shifted = frames(5, da_x=opt_x, da_y=opt_y + 256, da_z=opt_z)
    walk_y = centroid_y(shifted) - centroid_y(at_optimum)
    assert 1.5 < walk_y < fwhm_y, walk_y
    assert float(np.sum(shifted)) < float(np.sum(at_optimum)), (
        "brightness, not position, is the MAIN effect of a field"
    )

    shifted = frames(5, da_x=opt_x + 256, da_y=opt_y, da_z=opt_z)
    walk_x = centroid_x(shifted) - centroid_x(at_optimum)
    assert 3.0 < walk_x < fwhm_x, walk_x
    assert float(np.sum(shifted)) < float(np.sum(at_optimum))

    # Even the FULL range keeps the spot inside its own feature size.
    extreme = frames(5, da_x=511, da_y=opt_y, da_z=opt_z)
    assert abs(centroid_x(extreme) - centroid_x(at_optimum)) < fwhm_x

    defocused = frames(5, da_x=opt_x, da_y=opt_y, da_z=opt_z + 256)
    assert float(np.sum(defocused)) < float(np.sum(at_optimum))

    # Nothing compensated: the state an optimisation starts from -- dimmer,
    # and nudged off centre by a couple of pixels at most.
    uncompensated = frames(5, da_x=0, da_y=0, da_z=0)
    assert float(np.sum(uncompensated)) < float(np.sum(at_optimum))
    assert 1.0 < abs(centroid_x(uncompensated) - centroid_x(at_optimum)) < fwhm_x



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
    pulse = _world_pulse(cooling=True, trap=True)
    patterns: list[np.ndarray] = []
    for _ in range(8):
        _fire_world(world, pulse)
        patterns.append(world.occupancy)

    assert len({pattern.tobytes() for pattern in patterns}) > 2
    assert any(
        not np.array_equal(right, np.logical_not(left))
        for left, right in zip(patterns, patterns[1:], strict=True)
    )


def test_virtual_trap_off_time_removes_loaded_atoms() -> None:
    world = SimulationWorld(seed=3)
    _fire_world(world, _world_pulse(trap=True))
    assert not np.any(world.occupancy), "a pulse without cooling cannot load atoms"
    world.set_occupancy(np.ones(35, dtype=bool))
    _fire_world(world, _world_pulse(duration=1.0))
    assert not np.any(world.occupancy)


def test_virtual_pulse_fire_uses_loaded_camera_window_count() -> None:
    world = SimulationWorld(seed=1)
    streamer = VirtualPulseStreamer(
        world=world,
        camera_trigger_channel=CAMERA_CHANNEL,
    )
    streamer.open()
    try:
        pulse = resolve_pulse(
            IMAGING_PULSE_RESOURCE.value,
            path=IMAGING_PULSE_RESOURCE.path,
            board=streamer.describe(),
            api_values={
                "reference_probe_duration_before": 0.02,
                "readout_probe_duration": 0.005,
                "reference_probe_duration_after": 0.02,
            },
        )
        program = pulse.program
        streamer.load(program, source=pulse.sequence)
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
