from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest
from scipy import fft

from zlc_atom.devices.simulation import SimulationWorld, VirtualPulseStreamer
from zlc_atom.devices.slm import canonical_phase
from zlc_atom.devices.slm.solver import (
    imported_target,
    load_phase,
    load_target,
    preset_checkerboard,
    preset_ellipse,
    preset_grid,
    preset_rectangle,
    preset_ring,
    save_phase,
    save_target,
    solve_phase,
    validate_target,
)
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



def test_virtual_sites_have_small_detector_nuisance_and_psf_diversity() -> None:
    world = SimulationWorld(seed=4)
    efficiency = world.detector_efficiency
    sigma_xy = world.site_psf_sigma_xy
    angles = world.site_psf_angle_radians
    skew = world.site_psf_skew

    assert efficiency.shape == (35,)
    assert float(np.max(efficiency) / np.min(efficiency)) <= 1.03
    assert sigma_xy.shape == (35, 2)
    assert float(np.ptp(sigma_xy[:, 0])) > 0.05
    assert float(np.ptp(sigma_xy[:, 1])) > 0.05
    assert float(np.ptp(angles)) > 0.05
    assert float(np.ptp(skew)) > 0.05
    np.testing.assert_allclose(
        SimulationWorld(seed=4).detector_efficiency,
        efficiency,
    )


def test_slm_coherent_plant_owns_the_twofold_site_error_and_caches_propagation() -> None:
    ratios: list[float] = []
    correctable_fractions: list[float] = []
    for seed in (0, 1, 4, 11, 23, 37, 91):
        world = SimulationWorld(seed=seed)
        # Hidden plant truth is intentionally private.  Only this acceptance
        # oracle may inspect it; a solver or Task can reach the plant only via
        # the SLM command and qCMOS publications.
        sites = world._site_trap_intensities
        initial_ratio = float(np.max(sites) / np.min(sites))
        ratios.append(initial_ratio)
        assert world.propagation_count == 1
        initial_loading = world._site_loading_probabilities()
        order = np.argsort(sites)
        assert np.all(np.diff(initial_loading[order]) >= 0.0)
        survival = world._site_survival_probabilities(16e-6)
        assert np.all(np.diff(survival[order]) >= 0.0)

        class FixedDraws:
            def random(self, size: int) -> np.ndarray:
                return np.full(size, 0.5)

        stochastic_rng = world.rng
        world.rng = FixedDraws()
        np.testing.assert_array_equal(world._load_shot(), initial_loading > 0.5)
        world.set_occupancy(np.ones(len(sites), dtype=bool))
        world._lose_atoms(16e-6)
        np.testing.assert_array_equal(world.occupancy, survival > 0.5)
        world.rng = stochastic_rng

        # Camera noise and occupancy draws never re-run the coherent FFT for
        # the same explicit SLM command.
        world.set_occupancy(np.ones(len(sites), dtype=bool))
        first = world.render_frame(0, exposure_seconds=0.005)
        second = world.render_frame(1, exposure_seconds=0.005)
        assert not np.array_equal(first, second)
        assert world.propagation_count == 1

        # Only this acceptance oracle reads the planted hidden wavefront.  The
        # solver and every future feedback Task receive no such reference.
        before = world.slm_phase_revision
        world.apply_slm_phase(
            world.commanded_phase - world._hidden_slm_aberration
        )
        assert world.slm_phase_revision == before + 1
        world._ensure_slm_propagation()
        corrected = world._site_trap_intensities
        assert world.propagation_count == 2
        _ = world._trap_plane_intensity
        assert world.propagation_count == 2
        corrected_loading = world._site_loading_probabilities()
        assert not np.allclose(initial_loading, corrected_loading)
        corrected_order = np.argsort(corrected)
        assert np.all(np.diff(corrected_loading[corrected_order]) >= 0.0)
        corrected_ratio = float(np.max(corrected) / np.min(corrected))
        correctable_fractions.append(
            1.0 - (corrected_ratio - 1.0) / (initial_ratio - 1.0)
        )

    assert min(ratios) >= 1.8
    assert max(ratios) <= 2.2
    assert min(correctable_fractions) >= 0.90
    assert not hasattr(SimulationWorld, "trap_plane_intensity")
    assert not hasattr(SimulationWorld, "site_trap_intensities")


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


def test_slm_presets_are_one_continuous_target_truth() -> None:
    shape = (64, 80)
    grid = preset_grid(shape, (3, 4), spacing_yx=(12, 14), intensity=0.7)
    checkerboard = preset_checkerboard(
        shape,
        (3, 4),
        spacing_yx=(12, 14),
        intensity_a=1.0,
        intensity_b=0.25,
    )
    rectangle = preset_rectangle(shape, (20, 28), intensity=0.8, edge=3)
    ellipse = preset_ellipse(shape, (10, 16), intensity=0.6, edge=2)
    ring = preset_ring(shape, radius=14, width=4, intensity=0.9, edge=2)

    for target in (grid, checkerboard, rectangle, ellipse, ring):
        assert target.shape == shape
        assert target.dtype == np.dtype("<f4")
        assert np.all(np.isfinite(target))
        assert np.all(target >= 0.0)
    assert np.count_nonzero(grid) == 12
    assert set(np.unique(checkerboard)) == {0.0, 0.25, 1.0}
    assert np.any((rectangle > 0.0) & (rectangle < 0.8))
    assert np.any((ellipse > 0.0) & (ellipse < 0.6))
    assert np.any((ring > 0.0) & (ring < 0.9))

    imported = imported_target([[0.0, 2.0], [1.0, 4.0]])
    np.testing.assert_allclose(imported, [[0.0, 0.5], [0.25, 1.0]])
    np.testing.assert_allclose(
        canonical_phase(
            [[-0.5, 2.0 * np.pi, 2.0 * np.pi + 0.25]],
            (1, 3),
        ),
        [[2.0 * np.pi - 0.5, 0.0, 0.25]],
        rtol=0.0,
        atol=2e-6,
    )
    edge_phase = canonical_phase(
        [[-np.finfo(np.float32).tiny, np.float32(2.0 * np.pi)]],
        (1, 2),
    )
    assert np.all((edge_phase >= 0.0) & (edge_phase < 2.0 * np.pi))
    np.testing.assert_array_equal(canonical_phase(edge_phase, (1, 2)), edge_phase)
    with pytest.raises(ValueError):
        edge_phase.setflags(write=True)
    for invalid in (
        np.zeros((2, 2, 2)),
        np.asarray([[0.0, -1.0]]),
        np.asarray([[0.0, np.nan]]),
    ):
        with pytest.raises((TypeError, ValueError)):
            validate_target(invalid)
    for invalid_phase in (
        np.asarray([[0.0, np.nan]]),
        np.asarray([[0.0, np.inf]]),
    ):
        with pytest.raises(ValueError, match="finite"):
            canonical_phase(invalid_phase, (1, 2))


def _ideal_slm_intensity(phase: np.ndarray) -> np.ndarray:
    height, width = phase.shape
    yy, xx = np.ogrid[-1.0:1.0:height * 1j, -1.0:1.0:width * 1j]
    pupil = (xx * xx + yy * yy <= 0.9**2).astype(np.float32)
    field = pupil * np.exp(1j * phase).astype(np.complex64)
    far = fft.fftshift(
        fft.fft2(fft.ifftshift(field), norm="ortho")
    )
    return np.abs(far) ** 2


def test_one_slm_solver_selects_sparse_wgs_and_dense_mraf() -> None:
    sparse = preset_grid((64, 64), (3, 5), spacing_yx=(10, 9))
    phase, metadata = solve_phase(sparse, iterations=80, seed=17)
    repeated, repeated_metadata = solve_phase(sparse, iterations=80, seed=17)
    np.testing.assert_array_equal(phase, repeated)
    assert metadata == repeated_metadata
    assert metadata["method"] == "weighted-gs"
    assert phase.dtype == np.dtype("<f4")
    assert np.all((phase >= 0.0) & (phase < 2.0 * np.pi))
    site_values = _ideal_slm_intensity(phase)[sparse > 0.0]
    assert float(np.max(site_values) / np.min(site_values)) <= 1.01

    graded = preset_checkerboard(
        (64, 64),
        (3, 4),
        spacing_yx=(10, 9),
        intensity_a=1.0,
        intensity_b=0.25,
    )
    graded_phase, graded_metadata = solve_phase(graded, iterations=80, seed=17)
    assert graded_metadata["method"] == "weighted-gs"
    graded_intensity = _ideal_slm_intensity(graded_phase)
    measured_ratio = float(
        np.mean(graded_intensity[graded == 1.0])
        / np.mean(graded_intensity[graded == 0.25])
    )
    assert measured_ratio == pytest.approx(4.0, rel=0.01)

    dense = preset_rectangle((64, 64), (22, 26), edge=4)
    dense_phase, dense_metadata = solve_phase(dense, seed=9)
    assert dense_metadata["method"] == "mraf"
    assert dense_metadata["iterations"] == 300
    intensity = _ideal_slm_intensity(dense_phase)
    interior = dense >= 0.999
    values = intensity[interior]
    assert float(np.percentile(values, 95) / np.percentile(values, 5)) <= 1.01

    warmed, warm_metadata = solve_phase(
        sparse,
        initial_phase=phase,
        iterations=8,
        seed=999,
    )
    assert warm_metadata["method"] == "weighted-gs"
    assert warmed.shape == sparse.shape
    calls = 0

    def stop_requested() -> bool:
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(InterruptedError):
        solve_phase(sparse, iterations=80, stop_requested=stop_requested)
    with pytest.raises(ValueError, match="positive intensity"):
        solve_phase(np.zeros((16, 16), dtype=np.float32))


def test_slm_target_json_and_phase_npz_are_strict_plain_artifacts(tmp_path: Path) -> None:
    target = preset_ellipse((24, 32), (7, 11), edge=2)
    target_path = save_target(tmp_path / "target.json", target)
    np.testing.assert_array_equal(load_target(target_path), target)

    phase, metadata = solve_phase(
        preset_grid((24, 32), (2, 3), spacing_yx=(7, 8)),
        iterations=30,
        seed=4,
    )
    metadata = {**metadata, "note": "plain", "shape": [24, 32]}
    phase_path = save_phase(tmp_path / "phase.npz", phase, metadata)
    loaded_phase, loaded_metadata = load_phase(phase_path)
    np.testing.assert_array_equal(loaded_phase, phase)
    assert loaded_metadata == metadata
    with np.load(phase_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"phase", "metadata"}
        assert archive["phase"].dtype == np.dtype("<f4")
        assert archive["metadata"].shape == ()
        assert archive["metadata"].dtype.kind == "U"

    malformed_target = tmp_path / "bad-target.json"
    malformed_target.write_text(
        '{"format":"zlc.slm.target","format":"zlc.slm.target",'
        '"version":1,"shape":[1,1],"intensity":[[1.0]]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_target(malformed_target)

    for name, payload in (
        (
            "bad-version.json",
            '{"format":"zlc.slm.target","version":true,'
            '"shape":[2,2],"intensity":[[1,0],[0,1]]}',
        ),
        (
            "bad-shape.json",
            '{"format":"zlc.slm.target","version":1,'
            '"shape":[2.0,2],"intensity":[[1,0],[0,1]]}',
        ),
        (
            "bad-value.json",
            '{"format":"zlc.slm.target","version":1,'
            '"shape":[2,2],"intensity":[["1",0],[0,1]]}',
        ),
    ):
        malformed = tmp_path / name
        malformed.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError):
            load_target(malformed)

    malformed_phase = tmp_path / "bad-phase.npz"
    with malformed_phase.open("wb") as stream:
        np.savez(
            stream,
            phase=phase,
            metadata=np.asarray("{}"),
            unexpected=np.asarray(1),
        )
    with pytest.raises(ValueError, match="members"):
        load_phase(malformed_phase)

    wrong_dtype = tmp_path / "wrong-dtype.npz"
    with wrong_dtype.open("wb") as stream:
        np.savez(
            stream,
            phase=np.asarray(phase, dtype=np.float64),
            metadata=np.asarray("{}"),
        )
    with pytest.raises(ValueError, match="float32"):
        load_phase(wrong_dtype)
