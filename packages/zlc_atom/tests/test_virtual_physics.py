from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import fft
from scipy.ndimage import maximum_filter

import zlc_atom.devices.simulation.world as simulation_world
import zlc_atom.devices.slm.solver as slm_solver
from zlc_atom.devices.simulation import (
    SimulationWorld,
    SimulationWorldConfig,
    VirtualCamera,
    VirtualCameraConfig,
    VirtualPulseStreamer,
)
from zlc_atom.devices.slm import canonical_phase
from zlc_atom.devices.slm.solver import (
    compose_science_phase,
    freeze_pattern_phase,
    imported_target,
    load_science_context,
    load_target,
    preset_checkerboard,
    preset_flat_top,
    preset_gaussian,
    preset_grid,
    preset_text,
    save_science_context,
    save_target,
    science_operator_wavefront,
    science_pupil_fields,
    solve_phase,
    validate_target,
)
from zlc_atom.install import create_installation
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from zlc_atom.nodes.camera_measurement import CameraMeasurementNode, CameraMeasurementRequest
from zlc_atom.nodes.camera_measurement.measurement import CameraCycleSource
from zlc_atom.nodes.calibration.calibration import extract_box_signals
from zlc_runtime import SignalDataPlane
from zlc_pulse import (
    AnalogStep,
    OutputDelay,
    PulseFieldRef,
    PulsePeriod,
    PulseSequence,
    PulseSlot,
    compile_sequence,
    load_streamer_config,
    resolve_api_parameters,
)
from zlc_pulse.wire import CtrlWords, STATUS_DONE, STATUS_RUNNING
from tests.pulse_fixture import (
    CAMERA_CHANNEL,
    IMAGING_PULSE_RESOURCE,
)

#: The repository this test belongs to.  Anchored to the file rather than to
#: the working directory, so a suite run from anywhere still finds pulses/.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _world(**config_changes: object) -> SimulationWorld:
    return SimulationWorld(replace(SimulationWorldConfig(), **config_changes))


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


def _timeline_pulse(
    states: tuple[tuple[bool, bool, bool], ...],
    *,
    period_seconds: float = 0.001,
    delays: tuple[OutputDelay, ...] = (),
) -> PulseSequence:
    """Author cooling/trap/camera edges without bypassing pulse compilation."""

    target = IMAGING_PULSE_RESOURCE.value.target
    lane_index = {lane: index for index, lane in enumerate(target.raw_lanes)}
    periods = []
    for index, (cooling, trap, camera) in enumerate(states):
        digital = [0] * len(target.raw_lanes)
        for name, active in (
            ("cooling", cooling),
            ("trap", trap),
            ("emCCD", camera),
        ):
            if active:
                digital[lane_index[target.by_key[name].lanes[0]]] = 1
        periods.append(
            PulsePeriod(
                f"state_{index}",
                period_seconds,
                "s",
                tuple(digital),
                (),
            )
        )
    return PulseSequence(
        name="world_timeline",
        target=target,
        time_step_ns=20.0,
        periods=tuple(periods),
        delays=delays,
    )


def test_qcmos_parameters_and_derived_poisson_signal_are_single_world_physics() -> None:
    world = _world(seed=4)
    assert world.atom_sigma_px == pytest.approx(0.7)
    assert world.background_rate == pytest.approx(300.0)
    assert world.atom_rate == pytest.approx(145_000.0)
    assert world.probe_detuning_linewidths == pytest.approx(-1.9)
    assert world.trap_light_shift_linewidths == pytest.approx(1.6)
    assert world.conversion_e_per_count == pytest.approx(0.107)
    assert world.read_noise_e == pytest.approx(0.43)
    assert world.offset_counts == pytest.approx(200.0)

    exposure = 0.02
    assert world.atom_rate * exposure == pytest.approx(2_900.0)
    assert -world.fluorescence_lifetime_seconds * np.expm1(
        -exposure / world.fluorescence_lifetime_seconds
    ) < exposure
    site_count = len(world.geometry.site_centers_xy)
    empty = np.zeros(site_count, dtype=bool)
    dark = np.asarray(
        [
            world.render_frame(
                index,
                exposure_seconds=exposure,
                probe_seconds=0.0,
                occupancy=empty,
            )
            for index in range(24)
        ]
    )
    loaded = np.ones(site_count, dtype=bool)
    bright = np.asarray(
        [
            world.render_frame(
                index,
                exposure_seconds=exposure,
                probe_seconds=exposure,
                occupancy=loaded,
            )
            for index in range(24)
        ]
    )
    assert dark.dtype == np.dtype("<u2")
    assert bright.dtype == np.dtype("<u2")
    assert float(np.mean(bright)) > float(np.mean(dark))
    assert float(np.var(dark)) > 0.0

    for field, invalid in (
        ("probe_saturation", 0.0),
        ("probe_saturation", np.inf),
        ("probe_detuning_linewidths", np.nan),
        ("trap_light_shift_linewidths", np.inf),
        ("trap_light_shift_linewidths", -0.1),
        ("fluorescence_lifetime_seconds", 0.0),
    ):
        with pytest.raises(ValueError, match=field):
            replace(SimulationWorldConfig(), seed=4, **{field: invalid})

    configured = _world(seed=4, loading_probability=0.5, atom_rate=10_000.0)
    assert configured.config.loading_probability == 0.5
    assert configured.atom_rate == 10_000.0
    with pytest.raises(AttributeError):
        configured.loading_probability = 1.0
    with pytest.raises(FrozenInstanceError):
        configured.config.atom_rate = 1.0


def test_direct_world_profile_is_resolved_before_virtual_device_init(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "simulation-world.json"
    profile.write_text(
        json.dumps(
            {
                "format": "zlc.simulation.world_profile",
                "offset_counts": 123.0,
                "conversion_e_per_count": 0.2,
                "loading_probability": 0.5,
                "atom_rate": 10_000.0,
                "mot_field_optimum_dac": [11, -12, 13],
            }
        ),
        encoding="utf-8",
    )
    installation = create_installation(
        (
            {
                "key": "camera",
                "type_id": "camera.virtual",
                "config": {},
            },
        ),
        simulation={"world_profile": str(profile)},
    )
    try:
        world = installation.world
        assert world.loading_probability == 0.5
        assert world.atom_rate == 10_000.0
        assert world.config.mot_field_optimum_dac == (11, -12, 13)
        assert installation.device("camera").photoelectron_conversion == (123.0, 0.2)
    finally:
        installation.close()

    duplicate = tmp_path / "duplicate-world.json"
    duplicate.write_text(
        '{"format":"zlc.simulation.world_profile",'
        '"atom_rate":1,"atom_rate":2}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict JSON"):
        SimulationWorldConfig.from_profile(
            duplicate,
            geometry=SimulationWorldConfig().geometry,
            seed=0,
        )

    unknown = tmp_path / "unknown-world-field.json"
    unknown.write_text(
        '{"format":"zlc.simulation.world_profile","unexpected":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        SimulationWorldConfig.from_profile(
            unknown,
            geometry=SimulationWorldConfig().geometry,
            seed=0,
        )


def test_loading_probability_is_a_half_probability_ceiling() -> None:
    assert SimulationWorldConfig(loading_probability=0.5).loading_probability == 0.5
    for invalid in (0.500_001, 1.0):
        with pytest.raises(ValueError, match="loading_probability"):
            SimulationWorldConfig(loading_probability=invalid)

    world = _world(loading_probability=0.5)
    scale = world._loading_intensity_scale
    assert scale is not None
    depths = scale * np.concatenate(((-1.0, 0.0), np.geomspace(1e-9, 1e9, 1000)))
    probabilities = world._loading_probabilities(depths)
    assert np.all((0.0 <= probabilities) & (probabilities <= 0.5))
    assert probabilities[-1] == pytest.approx(0.5)


def test_qcmos_reuses_byte_exact_fixed_site_psfs(monkeypatch) -> None:
    """Rendering must not rebuild the same 35 fixed optical spots per frame."""

    world = _world(seed=41)
    reference = _world(seed=41)
    site_count = len(world.geometry.site_centers_xy)
    occupancies = (
        np.ones(site_count, dtype=bool),
        np.arange(site_count) % 3 == 0,
        np.zeros(site_count, dtype=bool),
    )
    for ordinal, occupancy in enumerate(occupancies):
        actual = world.render_frame(
            ordinal,
            exposure_seconds=0.02,
            probe_seconds=0.005,
            occupancy=occupancy,
        )
        expected = reference.render_frame(
            ordinal,
            exposure_seconds=0.02,
            probe_seconds=0.005,
            occupancy=occupancy,
        )
        np.testing.assert_array_equal(actual, expected)

    def rebuilt_psf(*_args, **_kwargs):
        raise AssertionError("render_frame rebuilt a fixed site PSF")

    propagation_count = world._propagation_count
    monkeypatch.setattr(simulation_world.np, "exp", rebuilt_psf)
    for ordinal in (3, 4):
        world.render_frame(
            ordinal,
            exposure_seconds=0.02,
            probe_seconds=0.005,
            occupancy=occupancies[ordinal % len(occupancies)],
        )
    assert world._propagation_count == propagation_count


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
    world = _world(seed=7)
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
    """Zero field is the sole centred, brightest MOT operating point."""

    from zlc_atom.devices.simulation import DEFAULT_MOT_FIELD_OPTIMUM_DAC

    opt_x, opt_y, opt_z = DEFAULT_MOT_FIELD_OPTIMUM_DAC
    assert (opt_x, opt_y, opt_z) == (0, 0, 0)

    def frames(seed: int, *, da_x: int, da_y: int, da_z: int) -> np.ndarray:
        world = _world(seed=seed)
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

    defocused = frames(5, da_x=opt_x, da_y=opt_y, da_z=opt_z + 256)
    assert float(np.sum(defocused)) < float(np.sum(at_optimum))


def test_virtual_traps_share_one_aberrated_psf_without_per_site_nuisance() -> None:
    world = _world(seed=4)
    assert not hasattr(world, "_detector_efficiency")
    assert not hasattr(world, "_site_psf_sigma_xy")
    assert not hasattr(world, "_site_psf_angle_radians")
    assert not hasattr(world, "_site_psf_skew")
    psf = np.asarray(world._camera_psf)
    assert float(np.max(np.abs(psf - np.flip(psf, axis=0)))) > 0.05
    np.testing.assert_array_equal(
        world._trap_psf_spots,
        world._camera_spots(world._trap_centers_xy),
    )
    np.testing.assert_array_equal(_world(seed=4)._camera_psf, psf)


def test_slm_coherent_plant_owns_the_twofold_site_error_and_caches_propagation() -> None:
    ratios: list[float] = []
    correctable_fractions: list[float] = []
    for seed in (0, 1, 4, 11, 23, 37, 91):
        world = _world(seed=seed)
        # Hidden plant truth is intentionally private.  Only this acceptance
        # oracle may inspect it; a solver or Task can reach the plant only via
        # the SLM command and qCMOS publications.
        sites = world._trap_intensities
        initial_ratio = float(np.max(sites) / np.min(sites))
        ratios.append(initial_ratio)
        assert world._propagation_count == 1
        initial_loading = world._site_loading_probabilities()
        order = np.argsort(sites)
        assert np.all(np.diff(initial_loading[order]) >= 0.0)
        assert np.count_nonzero(initial_loading == 0.0) >= int(
            np.ceil(0.10 * len(initial_loading))
        )
        assert float(np.max(initial_loading)) > 0.0
        initial_fluorescence = world._fluorescence_scales(sites)
        assert np.all(np.diff(initial_fluorescence[order]) <= 0.0)
        initial_depth_interval = np.linspace(np.min(sites), np.max(sites), 128)
        assert np.all(
            np.diff(world._fluorescence_scales(initial_depth_interval)) < 0.0
        )
        fixed_probe_depths = world._loading_intensity_scale * np.linspace(
            0.70, 1.55, 128
        )
        fixed_probe_response = world._fluorescence_scales(fixed_probe_depths)
        survival = world._site_survival_probabilities(16e-6)
        assert np.all(np.diff(survival[order]) >= 0.0)

        # Camera noise and occupancy draws never re-run the coherent FFT for
        # the same explicit SLM command.
        loaded = np.ones(len(sites), dtype=bool)
        first = world.render_frame(
            0, exposure_seconds=0.005, occupancy=loaded
        )
        second = world.render_frame(
            1, exposure_seconds=0.005, occupancy=loaded
        )
        assert not np.array_equal(first, second)
        assert world._propagation_count == 1

        # Only this acceptance oracle reads the planted hidden wavefront.  The
        # solver and every future feedback Task receive no such reference.
        before = world._slm_phase_revision
        world.apply_slm_phase(
            world.commanded_phase - world._hidden_slm_aberration
        )
        assert world._slm_phase_revision == before + 1
        world._ensure_slm_propagation()
        corrected = world._trap_intensities
        assert world._propagation_count == 2
        _ = world._trap_plane_intensity
        assert world._propagation_count == 2
        corrected_loading = world._site_loading_probabilities()
        corrected_order = np.argsort(corrected)
        assert np.all(np.diff(corrected_loading[corrected_order]) >= 0.0)
        assert float(np.ptp(corrected_loading)) < float(np.ptp(initial_loading))
        corrected_loading_ratio = float(
            np.max(corrected_loading) / np.min(corrected_loading)
        )
        assert float(np.min(corrected_loading)) > 0.0
        assert corrected_loading_ratio == pytest.approx(1.0)
        np.testing.assert_array_equal(
            world._fluorescence_scales(fixed_probe_depths),
            fixed_probe_response,
        )
        corrected_survival = world._site_survival_probabilities(16e-6)
        assert np.all(np.diff(corrected_survival[corrected_order]) >= 0.0)
        corrected_ratio = float(np.max(corrected) / np.min(corrected))
        correctable_fractions.append(
            1.0 - (corrected_ratio - 1.0) / (initial_ratio - 1.0)
        )

    assert min(ratios) >= 1.8
    assert max(ratios) <= 2.2
    assert min(correctable_fractions) >= 0.90
    assert not hasattr(SimulationWorld, "trap_plane_intensity")
    assert not hasattr(SimulationWorld, "site_trap_intensities")


def test_every_trap_is_one_raw_local_peak_on_one_fixed_scale() -> None:
    """The propagated peaks are the only trap roster."""

    def assert_traps_match_plane(world: SimulationWorld) -> None:
        plane = np.asarray(world._trap_plane_intensity)
        local_peaks = maximum_filter(
            plane,
            size=simulation_world._TRAP_PEAK_NEIGHBORHOOD,
            mode="constant",
            cval=-np.inf,
        )
        peaks = np.asarray(world._trap_indices_yx)
        assert len({tuple(index) for index in peaks}) == len(peaks)
        rows, columns = peaks.T
        np.testing.assert_array_equal(plane[rows, columns], local_peaks[rows, columns])
        np.testing.assert_array_equal(
            world._trap_intensities,
            plane[rows, columns],
        )

    world = _world(seed=23)
    assert_traps_match_plane(world)
    assert len(world._trap_indices_yx) == 35
    assert world._loading_intensity_scale == pytest.approx(
        float(np.mean(world._trap_intensities))
    )

    target = np.zeros(world.slm_shape_yx, dtype=np.float32)
    for index in ((32, 24), (64, 72), (96, 104)):
        target[index] = 1.0
    phase, _metadata = solve_phase(target, seed=0)
    world.apply_slm_phase(phase)
    world._ensure_slm_propagation()

    assert_traps_match_plane(world)
    assert len(world._trap_indices_yx) == 3
    squared_distance = np.sum(
        (world._trap_indices_yx[:, None, :] - np.asarray(((32, 24), (64, 72), (96, 104)))[None, :, :]) ** 2,
        axis=2,
    )
    np.testing.assert_array_equal(np.sum(squared_distance <= 9, axis=0), np.ones(3))
    loading = world._site_loading_probabilities()
    order = np.argsort(world._trap_intensities)
    assert np.all(loading > 0.0)
    assert np.all(np.diff(loading[order]) >= 0.0)

    first_plane = np.array(world._trap_plane_intensity, copy=True)
    first_indices = np.array(world._trap_indices_yx, copy=True)
    first_depths = np.array(world._trap_intensities, copy=True)
    world.apply_slm_phase(phase)
    world._ensure_slm_propagation()
    np.testing.assert_array_equal(world._trap_plane_intensity, first_plane)
    np.testing.assert_array_equal(world._trap_indices_yx, first_indices)
    np.testing.assert_array_equal(world._trap_intensities, first_depths)


def test_slm_topology_is_exactly_the_dominant_propagated_peaks() -> None:
    """No hidden nominal roster exists beside the current physical traps."""

    reference = _world(seed=0)
    nominal = np.asarray(reference._reference_slm_indices_yx)
    shape_yx = reference.slm_shape_yx
    base = np.array(preset_grid(shape_yx, (5, 7)), copy=True)
    checker = np.array(base, copy=True)
    for index, site in enumerate(nominal):
        checker[tuple(site)] = 1.0 if index % 2 == 0 else 0.25
    removed = len(nominal) // 2
    new_site = np.asarray((32, 24))
    add = np.array(base, copy=True)
    add[tuple(new_site)] = 1.0
    remove = np.array(base, copy=True)
    remove[tuple(nominal[removed])] = 0.0
    move = np.array(remove, copy=True)
    move[tuple(new_site)] = 1.0
    three_extra = np.zeros(shape_yx, dtype=np.float32)
    extra_sites = np.asarray(((32, 24), (64, 72), (96, 104)))
    three_extra[tuple(extra_sites.T)] = 1.0
    three_nominal = np.zeros(shape_yx, dtype=np.float32)
    selected = np.asarray((0, len(nominal) // 2, len(nominal) - 1))
    three_nominal[tuple(nominal[selected].T)] = 1.0

    phases = [
        solve_phase(target, seed=0)[0]
        for target in (checker, remove, add, move, three_extra, three_nominal)
    ]
    authored_sites = (
        nominal,
        np.delete(nominal, removed, axis=0),
        np.vstack((nominal, new_site)),
        np.vstack((np.delete(nominal, removed, axis=0), new_site)),
        extra_sites,
        nominal[selected],
    )
    required_sites = (
        nominal[np.arange(len(nominal)) % 2 == 0],
        authored_sites[1],
        authored_sites[2],
        authored_sites[3],
        authored_sites[4],
        authored_sites[5],
    )

    for seed in range(256):
        world = _world(seed=seed)
        for phase, authored, required in zip(
            phases, authored_sites, required_sites, strict=True
        ):
            world.apply_slm_phase(phase)
            world._ensure_slm_propagation()
            actual = world._trap_indices_yx
            squared_distance = np.sum(
                (actual[:, None, :] - authored[None, :, :]) ** 2,
                axis=2,
            )
            np.testing.assert_array_equal(
                np.sum(squared_distance <= 9, axis=1),
                np.ones(len(actual), dtype=np.intp),
            )
            required_distance = np.sum(
                (actual[:, None, :] - required[None, :, :]) ** 2,
                axis=2,
            )
            np.testing.assert_array_equal(
                np.sum(required_distance <= 9, axis=0),
                np.ones(len(required), dtype=np.intp),
            )


def test_arbitrary_grid_spacing_uses_one_fourier_to_camera_map() -> None:
    world = _world(seed=0)
    target = preset_grid(
        world.slm_shape_yx,
        (5, 7),
        spacing_yx=(15, 15),
    )
    authored = np.argwhere(target > 0.0)
    phase, _metadata = solve_phase(target, seed=0)
    world.apply_slm_phase(phase)
    world._ensure_slm_propagation()

    squared_distance = np.sum(
        (world._trap_indices_yx[:, None, :] - authored[None, :, :]) ** 2,
        axis=2,
    )
    assert len(world._trap_indices_yx) == len(authored) == 35
    np.testing.assert_array_equal(
        np.sum(squared_distance <= 9, axis=0), np.ones(len(authored), dtype=int)
    )
    np.testing.assert_allclose(
        world._trap_centers_xy,
        world._camera_centers(world._trap_indices_yx),
        rtol=0.0,
        atol=0.0,
    )


def test_a_removed_trap_cannot_resurrect_its_atom(monkeypatch) -> None:
    installation = create_installation(
        "virtual",
        world=_world(loading_probability=0.5, atom_rate=100_000.0),
    )
    world = installation.world
    camera = installation.device("camera")
    try:
        nominal_phase = world.commanded_phase
        nominal_sites = np.asarray(world._reference_slm_indices_yx)
        kept = np.asarray((0, 2, 4))
        removed = len(nominal_sites) // 2
        target = np.zeros(world.slm_shape_yx, dtype=np.float32)
        target[tuple(nominal_sites[kept].T)] = 1.0
        sparse_phase, _metadata = solve_phase(target, seed=0)

        world._occupancy[:] = True
        assert np.all(world._occupancy)

        snapshots: list[np.ndarray] = []
        original_render = world.render_frame

        def record_render(
            ordinal: int,
            *,
            exposure_seconds: float,
            probe_seconds: float,
            occupancy: object,
        ) -> np.ndarray:
            snapshots.append(np.asarray(occupancy, dtype=bool).copy())
            return original_render(
                ordinal,
                exposure_seconds=exposure_seconds,
                probe_seconds=probe_seconds,
                occupancy=occupancy,
            )

        monkeypatch.setattr(world, "render_frame", record_render)

        def triggered_snapshot() -> np.ndarray:
            camera.arm(
                1,
                source_group_sizes=(1,),
                buffer_frame_count=1,
                timeout=1.0,
            )
            _fire_world(
                world,
                _world_pulse(trap=True, probe=True, camera=True),
            )
            record = camera.read_frame_records(1, timeout=1.0, exact=True)[0]
            terminal = camera.finish_record_capture()
            assert terminal.produced_count == 1
            return record.image

        world.apply_slm_phase(sparse_phase)
        triggered_snapshot()
        assert len(snapshots[-1]) == len(kept)
        assert np.all(snapshots[-1])
        assert np.all(world._occupancy)

        world.apply_slm_phase(nominal_phase)
        restored_frame = triggered_snapshot()
        restored_distance = np.sum(
            np.square(world._trap_indices_yx - nominal_sites[removed]), axis=1
        )
        restored = int(np.argmin(restored_distance))
        assert restored_distance[restored] <= 9
        assert not snapshots[-1][restored]
        assert not world._occupancy[restored]

        kept_center = tuple(world._camera_centers(nominal_sites[kept[:1]])[0])
        removed_center = tuple(world._camera_centers(nominal_sites[removed : removed + 1])[0])
        background_center = (15.0, 15.0)
        kept_signal, removed_signal, background = extract_box_signals(
            restored_frame,
            (kept_center, removed_center, background_center),
            radius=1,
        )
        assert kept_signal > background
        assert removed_signal < kept_signal
    finally:
        installation.close()


def test_occupied_qcmos_box_brightness_tracks_physical_trap_depth() -> None:
    """BOX means follow the shared Stark-shifted response to local depth."""

    world = _world(seed=23)
    sites = world._trap_indices_yx
    loading = world._site_loading_probabilities()
    loading_order = np.argsort(world._trap_intensities)
    assert np.all(np.diff(loading[loading_order]) >= 0.0)
    assert len(sites) == 35

    frame = np.mean(
        [
            world.render_frame(
                ordinal,
                exposure_seconds=0.02,
                probe_seconds=0.005,
                occupancy=np.ones(len(sites), dtype=bool),
            )
            for ordinal in range(32)
        ],
        axis=0,
    )
    extracted = extract_box_signals(
        frame,
        (*world._trap_centers_xy, (15.0, 15.0)),
        radius=1,
    )
    box_means = extracted[:-1] - extracted[-1]
    depths = world._trap_intensities / world._loading_intensity_scale
    fluorescence = world._fluorescence_scales(world._trap_intensities)

    assert float(np.corrcoef(fluorescence, box_means)[0, 1]) > 0.90
    assert float(np.corrcoef(depths, fluorescence)[0, 1]) < -0.99
    assert float(np.max(box_means) / np.min(box_means)) > 1.5


def test_add_remove_and_move_change_the_next_triggered_qcmos_frame(
    monkeypatch,
) -> None:
    """Each public phase command must change the next physical atom image."""

    installation = create_installation(
        "virtual",
        world=_world(loading_probability=0.5, atom_rate=100_000.0),
    )
    world = installation.world
    camera = installation.device("camera")
    sequencer = installation.device("sequencer")
    slm = installation.device("slm")
    try:
        def load_every_present_trap() -> np.ndarray:
            world._ensure_slm_propagation()
            world._occupancy = np.ones(len(world._trap_intensities), dtype=bool)
            return np.array(world._occupancy, copy=True)

        monkeypatch.setattr(world, "_load_shot", load_every_present_trap)
        base_target = np.array(preset_grid(slm.shape_yx, (5, 7)), copy=True)
        nominal_indices = np.argwhere(base_target > 0.0)
        old_index = nominal_indices[len(nominal_indices) // 2]
        new_index = np.asarray((32, 24))
        pulse = resolve_pulse(
            IMAGING_PULSE_RESOURCE.value,
            path=IMAGING_PULSE_RESOURCE.path,
            sequencer=sequencer,
            api_values={},
        )
        new_center = tuple(world._camera_centers(new_index.reshape(1, 2))[0])
        old_center = tuple(world._camera_centers(old_index.reshape(1, 2))[0])
        background_center = (15.0, 15.0)

        def capture(target: np.ndarray) -> tuple[float, float]:
            phase, _metadata = solve_phase(target, seed=0)
            slm.apply_phase(phase)
            world._ensure_slm_propagation()
            if target[tuple(new_index)] > 0.0:
                distance = np.sum(
                    np.square(world._trap_indices_yx - new_index), axis=1
                )
                matched = int(np.argmin(distance))
                assert distance[matched] <= 9
                measured_new_center = tuple(world._trap_centers_xy[matched])
            else:
                measured_new_center = new_center
            camera.arm(
                3,
                source_group_sizes=(3,),
                buffer_frame_count=3,
                timeout=1.0,
            )
            arm_sequencer(sequencer, pulse)
            sequencer.fire(run_repeats=1, scan_repeats=1)
            records = camera.read_frame_records(3, timeout=2.0, exact=True)
            assert sequencer.wait_done(1.0) is not None
            terminal = camera.finish_record_capture()
            assert terminal.produced_count == 3
            old_signal, new_signal, background = extract_box_signals(
                records[1].image,
                (old_center, measured_new_center, background_center),
                radius=1,
            )
            assert world._propagated_revision == world._slm_phase_revision
            return old_signal - background, new_signal - background

        add_target = np.array(base_target, copy=True)
        add_target[tuple(new_index)] = 1.0
        remove_target = np.array(base_target, copy=True)
        remove_target[tuple(old_index)] = 0.0
        move_target = np.array(remove_target, copy=True)
        move_target[tuple(new_index)] = 1.0

        base_old, base_new = capture(base_target)
        add_old, add_new = capture(add_target)
        remove_old, remove_new = capture(remove_target)
        move_old, move_new = capture(move_target)

        assert base_old > 20.0 and base_new < 0.35 * base_old
        assert add_old > 20.0 and add_new > 0.40 * add_old
        assert remove_old < 0.35 * base_old and remove_new < 0.35 * base_old
        assert move_new > 0.40 * base_old and move_old < 0.35 * move_new
    finally:
        installation.close()


def test_virtual_shots_randomly_reload_instead_of_alternating_two_patterns() -> None:
    world = _world(seed=11)
    target = np.zeros(world.slm_shape_yx, dtype=np.float32)
    nominal_sites = np.asarray(world._reference_slm_indices_yx)
    selected = np.asarray((0, 17, 34))
    target[tuple(nominal_sites[selected].T)] = 1.0
    phase, _metadata = solve_phase(target, seed=0)
    world.apply_slm_phase(phase)
    probabilities = world._site_loading_probabilities()
    assert len(world._trap_indices_yx) == 3
    order = np.argsort(world._trap_intensities)
    assert np.all(probabilities > 0.0)
    assert np.all(np.diff(probabilities[order]) >= 0.0)

    pulse = _world_pulse(cooling=True, trap=True)
    patterns: list[np.ndarray] = []
    for _ in range(32):
        _fire_world(world, pulse)
        occupancy = np.array(world._occupancy, copy=True)
        patterns.append(occupancy)

    assert len({pattern.tobytes() for pattern in patterns}) > 2
    assert any(
        not np.array_equal(right, np.logical_not(left))
        for left, right in zip(patterns, patterns[1:], strict=True)
    )


def test_atom_qcmos_and_mot_draws_are_independent() -> None:
    reference = _world(seed=31)
    after_qcmos = _world(seed=31)
    after_mot = _world(seed=31)
    np.testing.assert_array_equal(
        reference._camera_psf, after_qcmos._camera_psf
    )
    np.testing.assert_array_equal(
        reference._camera_psf, after_mot._camera_psf
    )

    empty = np.zeros(35, dtype=bool)
    for ordinal in range(4):
        after_qcmos.render_frame(
            ordinal,
            exposure_seconds=0.005,
            occupancy=empty,
        )
        after_mot._mot_population = 1.0
        after_mot.render_mot_frame(
            ordinal,
            exposure_seconds=0.01,
            frame_shape_yx=(96, 128),
        )

    loaded = [world._load_shot() for world in (reference, after_qcmos, after_mot)]
    np.testing.assert_array_equal(loaded[0], loaded[1])
    np.testing.assert_array_equal(loaded[0], loaded[2])

    for world in (reference, after_qcmos, after_mot):
        world._occupancy[:] = True
    after_qcmos.render_frame(5, exposure_seconds=0.005)
    after_mot.render_mot_frame(
        5,
        exposure_seconds=0.01,
        frame_shape_yx=(96, 128),
    )
    for world in (reference, after_qcmos, after_mot):
        world._lose_atoms(16e-6)
    np.testing.assert_array_equal(reference._occupancy, after_qcmos._occupancy)
    np.testing.assert_array_equal(reference._occupancy, after_mot._occupancy)

    qcmos_reference = _world(seed=37)
    qcmos_after_mot = _world(seed=37)
    loaded = np.ones(35, dtype=bool)
    qcmos_after_mot.render_mot_frame(
        0,
        exposure_seconds=0.01,
        occupancy=loaded,
        frame_shape_yx=(48, 64),
    )
    np.testing.assert_array_equal(
        qcmos_reference.render_frame(
            0,
            exposure_seconds=0.005,
            occupancy=loaded,
        ),
        qcmos_after_mot.render_frame(
            0,
            exposure_seconds=0.005,
            occupancy=loaded,
        ),
    )

    mot_reference = _world(seed=43)
    mot_after_qcmos = _world(seed=43)
    mot_after_qcmos.render_frame(
        0,
        exposure_seconds=0.005,
        occupancy=loaded,
    )
    np.testing.assert_array_equal(
        mot_reference.render_mot_frame(
            0,
            exposure_seconds=0.01,
            occupancy=loaded,
            frame_shape_yx=(48, 64),
        ),
        mot_after_qcmos.render_mot_frame(
            0,
            exposure_seconds=0.01,
            occupancy=loaded,
            frame_shape_yx=(48, 64),
        ),
    )


def test_fire_processes_every_cooling_rise_and_whole_trap_off_episode(
    monkeypatch,
) -> None:
    installation = create_installation("virtual")
    world = installation.world
    camera = installation.device("camera")
    events: list[tuple[str, float | None]] = []

    def record_load() -> np.ndarray:
        events.append(("load", None))
        return np.zeros(35, dtype=bool)

    def record_loss(seconds: float) -> None:
        if seconds > 0.0:
            events.append(("loss", float(seconds)))

    def record_camera(
        _ordinal: int,
        *,
        exposure_seconds: float,
        probe_seconds: float,
        occupancy: object,
    ) -> np.ndarray:
        assert exposure_seconds > 0.0
        assert probe_seconds >= 0.0
        assert np.asarray(occupancy).shape == (35,)
        events.append(("camera", None))
        return np.zeros(world.geometry.image_shape_yx, dtype=np.uint16)

    monkeypatch.setattr(world, "_load_shot", record_load)
    monkeypatch.setattr(world, "_lose_atoms", record_loss)
    monkeypatch.setattr(world, "render_frame", record_camera)
    pulse = _timeline_pulse(
        (
            (False, True, False),
            (True, True, False),   # load
            (False, True, False),
            (True, False, False),  # cooling rise while trap is off
            (False, False, True),  # camera must not split this release
            (True, False, False),  # another rejected cooling rise
            (False, True, False),  # one 3 ms release ends
            (True, True, False),   # load
            (False, False, False),
            (True, True, True),    # loss, load, then camera at one tick
        )
    )
    try:
        camera.arm(
            2,
            source_group_sizes=(2,),
            buffer_frame_count=2,
            timeout=1.0,
        )
        _fire_world(world, pulse)
        camera.read_frame_records(2, timeout=1.0, exact=True)
        camera.finish_record_capture()

        assert events == [
            ("load", None),
            ("camera", None),
            ("loss", pytest.approx(0.003, abs=1e-9)),
            ("load", None),
            ("loss", pytest.approx(0.001, abs=1e-9)),
            ("load", None),
            ("camera", None),
        ]
    finally:
        installation.close()


def test_fire_extends_release_to_the_delayed_physical_horizon(
    monkeypatch,
) -> None:
    installation = create_installation("virtual")
    world = installation.world
    camera = installation.device("camera")
    events: list[tuple[str, float | None]] = []

    def record_loss(seconds: float) -> None:
        events.append(("loss", float(seconds)))

    def record_camera(
        _ordinal: int,
        *,
        exposure_seconds: float,
        probe_seconds: float,
        occupancy: object,
    ) -> np.ndarray:
        assert exposure_seconds > 0.0
        assert probe_seconds >= 0.0
        assert np.asarray(occupancy).shape == (35,)
        events.append(("camera", None))
        return np.zeros(world.geometry.image_shape_yx, dtype=np.uint16)

    monkeypatch.setattr(world, "_lose_atoms", record_loss)
    monkeypatch.setattr(world, "render_frame", record_camera)
    pulse = _timeline_pulse(
        (
            (False, True, False),
            (False, False, True),
            (False, False, False),
        ),
        delays=(
            OutputDelay("trap", 0.001, "s"),
            OutputDelay("emCCD", 0.003, "s"),
        ),
    )
    try:
        camera.arm(
            1,
            source_group_sizes=(1,),
            buffer_frame_count=1,
            timeout=1.0,
        )
        _fire_world(world, pulse)
        camera.read_frame_records(1, timeout=1.0, exact=True)
        camera.finish_record_capture()

        assert events == [
            ("loss", pytest.approx(0.001, abs=1e-9)),
            ("camera", None),
            ("loss", pytest.approx(0.003, abs=1e-9)),
        ]
    finally:
        installation.close()


def test_release_does_not_deplete_the_mot_population() -> None:
    world = _world(seed=17)
    world._mot_population = 1.0
    world._occupancy[:] = True
    world._lose_atoms(16e-6)
    assert world._mot_population == 1.0


def test_safe_has_no_persistent_test_only_occupancy_mode() -> None:
    world = _world(seed=2)
    assert not hasattr(world, "set_occupancy")
    target = np.zeros(world.slm_shape_yx, dtype=np.float32)
    for index in ((32, 24), (64, 72), (96, 104)):
        target[index] = 1.0
    phase, _metadata = solve_phase(target, seed=0)
    world.apply_slm_phase(phase)
    world._ensure_slm_propagation()
    assert len(world._occupancy) == 3
    world._occupancy[:] = True
    world._mot_population = 1.0
    world._dac_values.update(da_bias_x=17, da_bias_y=-23, da_bias_z=31)
    world.safe()
    assert not np.any(world._occupancy)
    assert world._mot_population == 0.0
    assert world._dac_values == {
        "da_bias_x": 0,
        "da_bias_y": 0,
        "da_bias_z": 0,
    }


def test_virtual_trap_off_time_removes_loaded_atoms() -> None:
    world = _world(seed=3, loading_probability=0.5)
    _fire_world(world, _world_pulse(trap=True))
    assert not np.any(world._occupancy), "a pulse without cooling cannot load atoms"
    _fire_world(world, _world_pulse(cooling=True, trap=True))
    assert np.any(world._occupancy)
    _fire_world(world, _world_pulse(duration=1.0))
    assert not np.any(world._occupancy)


def test_virtual_pulse_fire_uses_loaded_camera_window_count() -> None:
    world = _world(seed=1)
    streamer = VirtualPulseStreamer(
        world=world,
        camera_trigger_channel=CAMERA_CHANNEL,
    )
    streamer.open()
    try:
        sequence = resolve_api_parameters(
            IMAGING_PULSE_RESOURCE.value,
            {
                "reference_probe_duration_before": 0.02,
                "readout_probe_duration": 0.005,
                "reference_probe_duration_after": 0.02,
            },
        )
        board = streamer.describe()
        program = compile_sequence(sequence, board.geometry, board.clock_hz)
        streamer.load(program, source=sequence)
        started = time.monotonic()
        streamer.fire(run_repeats=1, scan_repeats=1)
        assert streamer.snapshot()["status"] == STATUS_RUNNING
        assert streamer.transport.read_word(CtrlWords.STATUS) == STATUS_RUNNING
        time.sleep(program.duration_seconds * 0.25)
        assert streamer.wait_done(0.0) is None
        first_report = streamer.wait_done(1.0)
        assert first_report is not None
        assert first_report.status == STATUS_DONE
        assert first_report.cursor == 0
        assert streamer.snapshot()["status"] == STATUS_DONE
        assert streamer.snapshot()["firing"] is False
        assert time.monotonic() - started >= program.duration_seconds * 0.8
        streamer.fire(run_repeats=0, scan_repeats=1)
        deadline = time.monotonic() + 0.5
        while world._fire_count < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert world._fire_count >= 3
        streamer.safe()
        stopped_at = world._fire_count
        time.sleep(0.08)
        assert world._fire_count == stopped_at

        streamer.fire(run_repeats=20, scan_repeats=1)
        deadline = time.monotonic() + 0.5
        while world._fire_count < stopped_at + 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        streamer.safe()
        interrupted_at = world._fire_count
        safe_cursor = streamer.transport.read_word(CtrlWords.CURSOR)
        assert stopped_at + 2 <= interrupted_at < stopped_at + 20
        time.sleep(0.08)
        assert world._fire_count == interrupted_at
        assert streamer.transport.read_word(CtrlWords.STATUS) == 0
        assert streamer.transport.read_word(CtrlWords.CURSOR) == safe_cursor

        # A scan row is visited once only after all of its whole-Pulse Run
        # repeats.  Extend this existing lifecycle acceptance rather than
        # creating a second virtual scheduler oracle.
        slotted = replace(
            _world_pulse(duration=0.03, cooling=True, trap=True),
            slots=(
                PulseSlot(
                    "dac",
                    PulseFieldRef("dac", "state", "da_bias_x"),
                    "value",
                ),
            ),
        )
        program = compile_sequence(slotted, board.geometry, board.clock_hz)
        rows = ((-16,), (24,))
        streamer.load(program, source=slotted, rows=rows)
        seen_rows: list[tuple[int, ...]] = []
        original_fire = world.fire

        def record_row(*args, table=None, **kwargs) -> None:
            seen_rows.append(tuple(table))
            original_fire(*args, table=table, **kwargs)

        world.fire = record_row  # type: ignore[method-assign]
        streamer.fire(run_repeats=2, scan_repeats=3)
        deadline = time.monotonic() + 0.5
        while len(seen_rows) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert streamer.transport.read_word(CtrlWords.CURSOR) == 0
        while len(seen_rows) < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert streamer.transport.read_word(CtrlWords.CURSOR) == 1
        report = streamer.wait_done(1.0)
        assert report is not None
        assert report.status == STATUS_DONE
        assert report.cursor == len(rows) * 3 - 1
        assert streamer.snapshot()["status"] == STATUS_DONE
        assert streamer.snapshot()["cursor"] == len(rows) * 3 - 1
        assert streamer.snapshot()["firing"] is False
        assert seen_rows == [rows[0], rows[0], rows[1], rows[1]] * 3

        seen_rows.clear()
        streamer.fire(run_repeats=1, scan_repeats=0)
        deadline = time.monotonic() + 0.5
        while len(seen_rows) < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert seen_rows[:3] == [rows[0], rows[1], rows[0]]
        assert streamer.transport.read_word(CtrlWords.CURSOR) >= 2
        streamer.safe()
        scan_stopped_at = len(seen_rows)
        scan_safe_cursor = streamer.transport.read_word(CtrlWords.CURSOR)
        time.sleep(0.08)
        assert len(seen_rows) == scan_stopped_at
        assert streamer.transport.read_word(CtrlWords.STATUS) == 0
        assert streamer.transport.read_word(CtrlWords.CURSOR) == scan_safe_cursor
    finally:
        streamer.close()


def test_unslotted_cycles_are_independent_three_frame_shots(monkeypatch) -> None:
    """Each empty-row point reloads once; its three camera windows share it."""

    installation = create_installation("virtual")
    world = installation.world
    camera = installation.device("camera")
    sequencer = installation.device("sequencer")
    cycles = 5
    loaded: list[np.ndarray] = []
    rendered: list[np.ndarray] = []
    loaded_at: list[float] = []
    original_load = world._load_shot
    original_render = world.render_frame

    def record_load() -> np.ndarray:
        loaded_at.append(time.monotonic())
        occupancy = original_load()
        loaded.append(np.array(occupancy, copy=True))
        return occupancy

    def record_render(
        ordinal: int,
        *,
        exposure_seconds: float,
        probe_seconds: float | None = None,
        occupancy: object | None = None,
    ) -> np.ndarray:
        rendered.append(np.array(occupancy, dtype=bool, copy=True))
        return original_render(
            ordinal,
            exposure_seconds=exposure_seconds,
            probe_seconds=probe_seconds,
            occupancy=occupancy,
        )

    monkeypatch.setattr(world, "_load_shot", record_load)
    monkeypatch.setattr(world, "render_frame", record_render)
    try:
        pulse = resolve_pulse(
            IMAGING_PULSE_RESOURCE.value,
            path=IMAGING_PULSE_RESOURCE.path,
            sequencer=sequencer,
            api_values={
                "reference_probe_duration_before": 0.02,
                "readout_probe_duration": 0.005,
                "reference_probe_duration_after": 0.02,
            },
        )
        assert pulse.program.slot_count == 0
        camera.arm(
            cycles * 3,
            source_group_sizes=(3,) * cycles,
            buffer_frame_count=cycles * 3,
            timeout=1.0,
        )
        arm_sequencer(sequencer, pulse)
        sequencer.fire(run_repeats=cycles, scan_repeats=1)
        records = camera.read_frame_records(cycles * 3, timeout=2.0, exact=True)
        assert sequencer.wait_done(1.0) is not None
        terminal = camera.finish_record_capture()

        assert len(records) == cycles * 3
        assert [record.source_ordinal for record in records] == list(
            range(cycles * 3)
        )
        assert terminal.produced_count == cycles * 3
        assert world._fire_count == cycles
        assert len(loaded) == cycles
        assert len(rendered) == cycles * 3
        assert np.all(
            np.diff(loaded_at) >= pulse.program.duration_seconds * 0.8
        ), "virtual cycles must arrive at the compiled wall cadence"
        for shot, occupancy in enumerate(loaded):
            for frame_occupancy in rendered[shot * 3 : (shot + 1) * 3]:
                np.testing.assert_array_equal(frame_occupancy, occupancy)
    finally:
        installation.close()


def test_camera_cycle_source_does_not_interpret_pulse_windows_or_exposure() -> None:
    world = _world(seed=5)
    camera = VirtualCamera(frame_source=world.render_frame)
    streamer = VirtualPulseStreamer(world=world)
    streamer.open()
    try:
        sequence = resolve_api_parameters(
            IMAGING_PULSE_RESOURCE.value,
            {
                "reference_probe_duration_before": 0.02,
                "readout_probe_duration": 0.005,
                "reference_probe_duration_after": 0.02,
            },
        )
        board = streamer.describe()
        program = compile_sequence(sequence, board.geometry, board.clock_hz)
        context = SimpleNamespace(
            generation=object(), cancel_requested=lambda: False
        )
        prepared: list[dict[str, object]] = []

        def node(*, frames_per_cycle: int, exposure: float = 0.02):
            camera.set_exposure_seconds(exposure)
            return SimpleNamespace(
                request=SimpleNamespace(
                    repeat=2,
                    frames_per_cycle=frames_per_cycle,
                    camera_key="camera",
                ),
                actual_working_point=camera.working_point(),
                _configure_capture=camera.working_point,
                _arm_configured=lambda **kwargs: prepared.append(kwargs)
                or SimpleNamespace(close=lambda: None),
            )

        source = CameraCycleSource(node(frames_per_cycle=2, exposure=0.1))
        source.open(context, cycles=2)
        source.validate(program, np.empty((2, 0), dtype=np.int64))
        source.arm()
        assert len(prepared) == 1
    finally:
        streamer.close()


def test_virtual_camera_counts_collected_frames_not_pulse_edges() -> None:
    target = IMAGING_PULSE_RESOURCE.value.target
    lane = target.raw_lanes.index(target.by_key[CAMERA_CHANNEL].lanes[0])

    def states(high: bool) -> tuple[int, ...]:
        values = [0] * len(target.raw_lanes)
        values[lane] = int(high)
        return tuple(values)

    sequence = PulseSequence(
        "busy_camera",
        target,
        20.0,
        (
            PulsePeriod("first", 1.0, "ms", states(True)),
            PulsePeriod("gap", 4.0, "ms", states(False)),
            PulsePeriod("second", 1.0, "ms", states(True)),
            PulsePeriod("tail", 1.0, "ms", states(False)),
        ),
    )
    config = load_streamer_config()
    program = compile_sequence(
        sequence,
        config["params"],
        float(config["clock_hz"]),
    )
    world = _world(seed=7)
    camera = VirtualCamera(frame_source=world.render_frame)
    world.register_camera(camera)
    camera.set_exposure_seconds(0.1)
    camera.arm(2, source_group_sizes=(2,), buffer_frame_count=2, timeout=1.0)
    world.fire(
        program,
        camera_channel=CAMERA_CHANNEL,
    )
    records = camera.read_frame_records(2, timeout=1.0, exact=True)
    terminal = camera.finish_record_capture()
    assert [record.source_ordinal for record in records] == [0, 1]
    assert terminal.produced_count == 2


def test_calibration_bracket_keeps_one_shot_occupancy_and_exposure_scaling() -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        camera = installation.device("camera")
        sequencer = installation.device("sequencer")
        repeats = 90
        measurement = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest("camera", 0.02, None, repeats, 3),
            signal_plane=plane,
        )
        pulse = resolve_pulse(
            IMAGING_PULSE_RESOURCE.value,
            path=IMAGING_PULSE_RESOURCE.path,
            sequencer=sequencer,
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
            sequencer.fire(run_repeats=1, scan_repeats=1)
            sequencer.wait_done(1.0)
            expected.append(np.array(installation.world._occupancy, copy=True))
        result = capture.collect()
        labels = np.asarray(expected, dtype=bool)
        centers = installation.world._trap_centers_xy
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
        long_before_accuracy = accuracy(long_before)
        short_accuracy = accuracy(short)
        long_after_accuracy = accuracy(long_after)
        assert long_before_accuracy >= 0.95
        # The readout frame is short in LIGHT, not in integration: an
        # edge-triggered sensor integrates its configured exposure whatever
        # the pulse window does.  At the authored photon budget both windows
        # can classify occupancy perfectly; the contrast-ratio assertion below
        # is the physical duration guard rather than an artificial accuracy gap.
        assert short_accuracy >= 0.75
        assert long_after_accuracy >= 0.95
        np.testing.assert_allclose(
            np.mean(long_before, axis=0),
            np.mean(long_after, axis=0),
            rtol=0.01,
            atol=10.0,
        )
        long_contrast = 0.5 * (
            float(np.mean(long_before[labels]) - np.mean(long_before[~labels]))
            + float(np.mean(long_after[labels]) - np.mean(long_after[~labels]))
        )
        short_contrast = float(np.mean(short[labels]) - np.mean(short[~labels]))
        assert short_contrast / long_contrast == pytest.approx(0.25, rel=0.20)
        assert installation.world._fire_count == repeats
    finally:
        plane.close()
        installation.close()


def test_public_repeat_reduction_conflates_loading_and_bright_dark_contrast() -> None:
    """Fifty all-shot means are visible but are not the Feedback observable.

    Each cycle is a real fresh load.  The reduced image therefore contains the
    physical product the operator sees: depth-dependent capture times the
    occupied atom's Stark-shifted fluorescence.  The two factors move in
    opposite directions, which is why Feedback must fit bright-dark rather
    than treating this repeat mean as trap response.
    """

    installation = create_installation("virtual")
    plane = SignalDataPlane()
    try:
        world = installation.world
        camera = installation.device("camera")
        sequencer = installation.device("sequencer")
        slm = installation.device("slm")
        repeats = 50
        loading = world._site_loading_probabilities()
        fluorescence = world._fluorescence_scales(world._trap_intensities)
        expected_site_signal = loading * fluorescence
        order = np.argsort(world._trap_intensities)
        assert np.all(np.diff(loading[order]) >= 0.0)
        assert np.count_nonzero(loading == 0.0) >= int(
            np.ceil(0.10 * len(loading))
        )
        nominal_loading = float(
            world._loading_probabilities(
                np.asarray([world._loading_intensity_scale])
            )[0]
        )
        assert 0.0 < nominal_loading < world.loading_probability
        depth_ratio = float(
            np.max(world._trap_intensities)
            / np.min(world._trap_intensities)
        )
        assert 1.8 <= depth_ratio <= 2.2

        measurement = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                "camera", 0.02, None, repeats, 3, photoelectrons=False
            ),
            signal_plane=plane,
        )
        pulse = resolve_pulse(
            IMAGING_PULSE_RESOURCE.value,
            path=IMAGING_PULSE_RESOURCE.path,
            sequencer=sequencer,
            api_values={
                "reference_probe_duration_before": 0.02,
                "readout_probe_duration": 0.005,
                "reference_probe_duration_after": 0.02,
            },
        )

        def capture_frames() -> np.ndarray:
            capture = measurement.prepare()
            arm_sequencer(sequencer, pulse)
            for _ in range(repeats):
                sequencer.fire(run_repeats=1, scan_repeats=1)
                assert sequencer.wait_done(1.0) is not None
            result = capture.collect()
            return np.asarray(
                plane.current_dataset(
                    measurement.signal_key("frames"), result.publication
                ).block.values
            )

        frames = capture_frames()
        assert frames.shape == (repeats, 3, 96, 128)
        assert int(np.max(frames)) < np.iinfo(np.uint16).max

        # This is the public Image plot's Reduce repeat -> mean projection of
        # the authored 20 ms sensor / 5 ms probe readout frame.
        reduced = np.mean(frames[:, 1], axis=0)
        site_boxes = extract_box_signals(
            reduced, world._trap_centers_xy, radius=1
        )
        observed_raw_count_ratio = float(
            np.max(site_boxes) / np.min(site_boxes)
        )
        assert float(
            np.corrcoef(fluorescence, world._trap_intensities)[0, 1]
        ) < -0.99
        assert float(
            np.corrcoef(expected_site_signal, world._trap_intensities)[0, 1]
        ) > 0.50
        assert np.count_nonzero(expected_site_signal == 0.0) >= int(
            np.ceil(0.10 * len(expected_site_signal))
        )
        assert observed_raw_count_ratio > 1.3

        # The acceptance oracle may remove the planted coherent screen, but it
        # still observes the result only through the same public camera path.
        slm.apply_phase(world.commanded_phase - world._hidden_slm_aberration)
        corrected_frames = capture_frames()
        assert int(np.max(corrected_frames)) < np.iinfo(np.uint16).max
        corrected_reduced = np.mean(corrected_frames[:, 1], axis=0)
        corrected_site_boxes = extract_box_signals(
            corrected_reduced,
            world._trap_centers_xy,
            radius=1,
        )
        corrected_raw_count_ratio = float(
            np.max(corrected_site_boxes) / np.min(corrected_site_boxes)
        )
        assert corrected_raw_count_ratio - 1.0 < 0.35 * (
            observed_raw_count_ratio - 1.0
        )
    finally:
        plane.close()
        installation.close()


def test_slm_presets_are_one_continuous_target_truth() -> None:
    shape = (64, 80)
    grid = preset_grid(shape, (3, 4), spacing_yx=(5, 7), intensity=0.7)
    checkerboard = preset_checkerboard(
        shape,
        (4, 4),
        spacing_yx=(10, 6),
        intensity=0.75,
    )
    gaussian = preset_gaussian(shape, (6, 10), intensity=0.9)
    flat_top = preset_flat_top(shape, (10, 16), intensity=0.6, edge=2)

    for target in (grid, checkerboard, gaussian, flat_top):
        assert target.shape == shape
        assert target.dtype == np.dtype("<f4")
        assert np.all(np.isfinite(target))
        assert np.all(target >= 0.0)
    assert np.count_nonzero(grid) == 12
    grid_rows = np.unique(np.argwhere(grid > 0.0)[:, 0])
    grid_columns = np.unique(np.argwhere(grid > 0.0)[:, 1])
    np.testing.assert_array_equal(np.diff(grid_rows), [5, 5])
    np.testing.assert_array_equal(np.diff(grid_columns), [7, 7, 7])
    assert grid_rows[0] == (shape[0] - 2 * 5) // 2
    assert grid_columns[0] == (shape[1] - 3 * 7) // 2

    checker_rows = np.unique(np.argwhere(checkerboard > 0.0)[:, 0])
    checker_columns = [
        np.flatnonzero(checkerboard[row] > 0.0) for row in checker_rows
    ]
    assert [len(columns) for columns in checker_columns] == [4, 3, 4, 3]
    np.testing.assert_array_equal(np.diff(checker_rows), [10, 10, 10])
    np.testing.assert_array_equal(np.diff(checker_columns[0]), [12, 12, 12])
    np.testing.assert_array_equal(np.diff(checker_columns[1]), [12, 12])
    np.testing.assert_array_equal(
        checker_columns[1],
        (checker_columns[0][:-1] + checker_columns[0][1:]) // 2,
    )
    np.testing.assert_array_equal(checker_columns[2], checker_columns[0])
    np.testing.assert_array_equal(checker_columns[3], checker_columns[1])
    np.testing.assert_array_equal(
        np.unique(checkerboard), np.asarray([0.0, 0.75], dtype=np.float32)
    )

    center_y, center_x = shape[0] // 2, shape[1] // 2
    assert gaussian[center_y, center_x] == pytest.approx(0.9)
    assert gaussian[center_y + 6, center_x] == pytest.approx(
        0.9 * np.exp(-2.0), rel=1e-6
    )
    assert gaussian[center_y, center_x + 10] == pytest.approx(
        0.9 * np.exp(-2.0), rel=1e-6
    )
    assert gaussian[0, 0] == 0.0
    assert np.any(gaussian == 0.0)
    assert flat_top[center_y, center_x] == pytest.approx(0.6)
    assert flat_top[0, 0] == 0.0
    assert np.any((flat_top > 0.0) & (flat_top < 0.6))

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


def test_slm_text_preset_is_centered_budgeted_spaced_and_byte_deterministic(
    monkeypatch,
) -> None:
    font_root = Path(r"C:\Windows\Fonts")
    fonts = tuple(
        font_root / name
        for name in (
            "msyh.ttc",
            "msyhbd.ttc",
            "simhei.ttf",
            "Dengb.ttf",
            "Deng.ttf",
            "simsun.ttc",
        )
    )
    font = next((path for path in fonts if path.is_file()), None)
    if font is None:
        pytest.skip("no supported Windows CJK font is installed")

    shape, spacing = (256, 384), 3
    text = "USTC 中科大"
    target = preset_text(
        shape, text, spacing=spacing, atom_budget=200, font_path=font
    )
    repeated = preset_text(
        shape, text, spacing=spacing, atom_budget=200, font_path=font
    )
    np.testing.assert_array_equal(repeated, target)
    assert repeated.tobytes() == target.tobytes()
    assert target.dtype == np.dtype("<f4")
    coordinates = np.argwhere(target > 0.0)
    assert 0 < len(coordinates) <= 200
    assert np.all(np.diff(np.unique(coordinates[:, 0])) % spacing == 0)
    assert np.all(np.diff(np.unique(coordinates[:, 1])) % spacing == 0)
    assert abs(int(coordinates[:, 0].min() + coordinates[:, 0].max()) - shape[0]) <= 1
    assert abs(int(coordinates[:, 1].min() + coordinates[:, 1].max()) - shape[1]) <= 1
    _text_phase, text_metadata = solve_phase(
        target, objective_kind="spots", iterations=1
    )
    assert text_metadata["transform"] == "selected-dft"

    smaller = preset_text(
        shape, text, spacing=spacing, atom_budget=80, font_path=font
    )
    assert 0 < np.count_nonzero(smaller) <= 80
    assert np.count_nonzero(target) >= np.count_nonzero(smaller)

    monkeypatch.setenv("WINDIR", r"C:\Windows")
    assert slm_solver._text_font_path(None, text) == font
    discovered = preset_text(shape, text, spacing=spacing, atom_budget=200)
    np.testing.assert_array_equal(discovered, target)

    with pytest.raises(ValueError, match="text"):
        preset_text(shape, "   ", spacing=spacing, atom_budget=20, font_path=font)
    with pytest.raises(ValueError, match="letters, digits, CJK"):
        preset_text(shape, "USTC!", spacing=spacing, atom_budget=20, font_path=font)
    with pytest.raises(FileNotFoundError, match="font"):
        preset_text(
            shape,
            "USTC",
            spacing=spacing,
            atom_budget=20,
            font_path=font_root / "definitely-missing-font.ttf",
        )
    with pytest.raises(ValueError, match="does not contain"):
        preset_text(
            shape,
            "\U00020000",
            spacing=spacing,
            atom_budget=20,
            font_path=font,
        )
    with pytest.raises(ValueError, match="does not fit"):
        preset_text(
            (8, 8), "中科大中科大", spacing=4, atom_budget=1, font_path=font
        )

    def nonmonotonic_mask(_text: str, _font: Path, size: int) -> np.ndarray:
        if size == 1:
            return np.ones((1, 2), dtype=bool)
        if size == 2:
            return np.ones((1, 1), dtype=bool)
        return np.ones((9, 1), dtype=bool)

    monkeypatch.setattr(slm_solver, "_rasterized_text", nonmonotonic_mask)
    recovered = preset_text(
        (8, 8), "A", spacing=1, atom_budget=1, font_path=font
    )
    assert np.count_nonzero(recovered) == 1


def test_slm_text_preset_uses_compact_filled_yahei_glyphs() -> None:
    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font.is_file():
        pytest.skip("Microsoft YaHei Regular is not installed")

    shape, spacing, budget = (256, 384), 4, 120

    def logical_support(text: str) -> tuple[np.ndarray, int]:
        target = preset_text(
            shape,
            text,
            spacing=spacing,
            atom_budget=budget,
            font_path=font,
        )
        repeated = preset_text(
            shape,
            text,
            spacing=spacing,
            atom_budget=budget,
            font_path=font,
        )
        np.testing.assert_array_equal(repeated, target)
        coordinates = np.argwhere(target > 0.0)
        assert 0 < len(coordinates) <= budget
        rows = (coordinates[:, 0] - coordinates[:, 0].min()) // spacing
        columns = (coordinates[:, 1] - coordinates[:, 1].min()) // spacing
        support = np.zeros(
            (int(rows.max()) + 1, int(columns.max()) + 1), dtype=bool
        )
        support[rows, columns] = True
        return support, len(coordinates)

    latin, _ = logical_support("ZLab")
    latin_columns = np.flatnonzero(np.any(latin, axis=0))
    latin_runs = np.split(
        latin_columns, np.flatnonzero(np.diff(latin_columns) > 1) + 1
    )
    assert len(latin_runs) == 4
    gaps = [
        int(right[0] - left[-1] - 1)
        for left, right in zip(latin_runs, latin_runs[1:])
    ]
    assert all(1 <= gap <= 3 for gap in gaps)
    baselines = [
        int(np.flatnonzero(np.any(latin[:, run[0]:run[-1] + 1], axis=1))[-1])
        for run in latin_runs
    ]
    assert max(baselines) - min(baselines) <= 1
    assert len({len(run) for run in latin_runs}) > 1

    letter_b = latin[:, latin_runs[-1][0]:latin_runs[-1][-1] + 1]
    stem = np.count_nonzero(letter_b, axis=0)
    assert int(np.argmax(stem)) == 0
    assert int(stem[0]) >= letter_b.shape[0] - 1
    remaining = set(map(tuple, np.argwhere(letter_b)))
    frontier = [remaining.pop()]
    while frontier:
        row, column = frontier.pop()
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                neighbor = (row + delta_row, column + delta_column)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    assert not remaining

    hanzi, hanzi_count = logical_support("神芯")
    assert hanzi_count < budget
    hanzi_columns = np.flatnonzero(np.any(hanzi, axis=0))
    hanzi_runs = np.split(
        hanzi_columns, np.flatnonzero(np.diff(hanzi_columns) > 1) + 1
    )
    assert len(hanzi_runs) == 2
    assert 1 <= int(hanzi_runs[1][0] - hanzi_runs[0][-1] - 1) <= 3
    assert all(len(run) >= 8 for run in hanzi_runs)
    assert all(
        np.count_nonzero(np.any(hanzi[:, run[0]:run[-1] + 1], axis=1))
        >= hanzi.shape[0] - 1
        for run in hanzi_runs
    )
    dense_blocks = (
        hanzi[:-1, :-1]
        & hanzi[1:, :-1]
        & hanzi[:-1, 1:]
        & hanzi[1:, 1:]
    )
    assert not np.any(dense_blocks)

    with pytest.raises(ValueError, match="does not fit"):
        preset_text(shape, "神芯", spacing=spacing, atom_budget=2, font_path=font)


def _ideal_slm_intensity(
    phase: np.ndarray,
    pupil_amplitude: np.ndarray | None = None,
) -> np.ndarray:
    height, width = phase.shape
    if pupil_amplitude is None:
        yy, xx = np.ogrid[-1.0:1.0:height * 1j, -1.0:1.0:width * 1j]
        pupil = (xx * xx + yy * yy <= 0.9**2).astype(np.float32)
    else:
        pupil = np.asarray(pupil_amplitude, dtype=np.float32)
    field = pupil * np.exp(1j * phase).astype(np.complex64)
    far = fft.fftshift(
        fft.fft2(fft.ifftshift(field), norm="ortho")
    )
    return np.abs(far) ** 2


def _shifted_wgs_kim_reference(
    target: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """The direct full-frame/shifted formulation used as a quality oracle."""

    desired = np.asarray(target, dtype=np.float32)
    support = desired > 0.0
    height, width = desired.shape
    yy, xx = np.ogrid[-1.0:1.0:height * 1j, -1.0:1.0:width * 1j]
    pupil = (xx * xx + yy * yy <= 0.9**2).astype(np.float32)
    phase = np.random.default_rng(seed).uniform(
        0.0, 2.0 * np.pi, desired.shape
    ).astype(np.float32)
    field = pupil.astype(np.complex64) * np.exp(1j * phase).astype(np.complex64)
    target_amplitude = np.sqrt(desired[support]).astype(np.float32)
    target_amplitude /= np.linalg.norm(target_amplitude)
    weights = np.array(target_amplitude, copy=True)
    fixed_phase: np.ndarray | None = None
    epsilon = np.finfo(np.float32).eps

    for iteration in range(iterations):
        far = fft.fftshift(fft.fft2(fft.ifftshift(field), norm="ortho"))
        selected = far[support]
        magnitude = np.abs(selected).astype(np.float32)
        measured = magnitude / max(float(np.linalg.norm(magnitude)), epsilon)
        weights *= np.clip(
            target_amplitude / np.maximum(measured, epsilon), 0.2, 5.0
        ) ** np.float32(0.8)
        weights /= max(float(np.linalg.norm(weights)), epsilon)
        current_phase = np.divide(
            selected,
            magnitude,
            out=np.ones_like(selected),
            where=magnitude > epsilon,
        )
        if fixed_phase is None:
            selected_phase = current_phase
            if iteration + 1 == 12:
                fixed_phase = np.array(current_phase, copy=True)
        else:
            selected_phase = fixed_phase
        constrained = np.zeros_like(far)
        constrained[support] = weights * selected_phase
        back = fft.fftshift(
            fft.ifft2(fft.ifftshift(constrained), norm="ortho")
        )
        back_magnitude = np.abs(back).astype(np.float32)
        field = pupil.astype(np.complex64) * np.divide(
            back,
            back_magnitude,
            out=np.ones_like(back),
            where=back_magnitude > epsilon,
        )
    return canonical_phase(np.angle(field), desired.shape)


def _support_quality(
    phase: np.ndarray,
    target: np.ndarray,
    pupil_amplitude: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    intensity = _ideal_slm_intensity(phase, pupil_amplitude)
    support = target > 0.0
    measured = intensity[support]
    measured /= np.sum(measured)
    expected = target[support] / np.sum(target[support])
    normalized = measured / expected
    ratio = float(np.max(normalized) / np.min(normalized))
    efficiency = float(np.sum(intensity[support]) / np.sum(intensity))
    return normalized, ratio, efficiency


def _full_fft_returned_quality(
    phase: np.ndarray,
    target: np.ndarray,
    pupil_amplitude: np.ndarray,
) -> tuple[float, float]:
    desired = fft.ifftshift(target)
    support = desired > 0.0
    field = fft.ifftshift(
        pupil_amplitude.astype(np.complex64)
        * np.exp(np.complex64(1j) * phase).astype(np.complex64)
    )
    selected = fft.fft2(field, norm="ortho")[support]
    magnitude = np.abs(selected).astype(np.float32)
    desired_spots = desired[support]
    relative = np.square(magnitude, dtype=np.float32) / desired_spots
    ratio = float(np.max(relative) / np.min(relative))
    efficiency = float(
        np.sum(np.square(magnitude, dtype=np.float32))
        / np.sum(np.square(pupil_amplitude, dtype=np.float32))
    )
    return ratio, efficiency


def test_slm_solver_validates_authored_pupil_amplitude() -> None:
    target = preset_grid((32, 40), (2, 3), spacing_yx=(9, 8))
    invalid = (
        (np.ones((31, 40), dtype=np.float32), "shape"),
        (np.full(target.shape, np.nan, dtype=np.float32), "finite"),
        (-np.ones(target.shape, dtype=np.float32), "non-negative"),
        (np.zeros(target.shape, dtype=np.float32), "positive"),
    )
    for pupil, message in invalid:
        with pytest.raises(ValueError, match=message):
            solve_phase(
                target,
                pupil_amplitude=pupil,
                objective_kind="spots",
                iterations=1,
            )


def test_slm_solver_uses_authored_pupil_in_every_full_resolution_path() -> None:
    shape = (48, 64)
    cartesian = preset_grid(shape, (3, 4), spacing_yx=(11, 12))
    irregular = np.array(cartesian, copy=True)
    irregular[tuple(np.argwhere(irregular > 0.0)[-1])] = 0.0
    image = preset_flat_top(shape, (9, 12), edge=3)
    yy, xx = np.ogrid[-1.0:1.0:shape[0] * 1j, -1.0:1.0:shape[1] * 1j]
    pupil = (
        ((xx / 0.72) ** 2 + (yy / 0.86) ** 2 <= 1.0)
        * (0.35 + 0.65 * (xx + 1.0) * 0.5)
    ).astype(np.float32)

    for target, objective_kind, method, transform in (
        (cartesian, "spots", "wgs-kim", "selected-dft"),
        (irregular, "spots", "wgs-kim", "selected-dft"),
        (image, "image", "mraf", "fft"),
    ):
        default_phase, default_metadata = solve_phase(
            target,
            objective_kind=objective_kind,
            iterations=20,
            seed=41,
        )
        phase, metadata = solve_phase(
            target,
            pupil_amplitude=pupil,
            objective_kind=objective_kind,
            iterations=20,
            seed=41,
        )
        repeated, repeated_metadata = solve_phase(
            target,
            pupil_amplitude=pupil,
            objective_kind=objective_kind,
            iterations=20,
            seed=41,
        )

        assert not np.array_equal(phase, default_phase)
        np.testing.assert_array_equal(phase, repeated)
        assert metadata == repeated_metadata
        assert default_metadata["pupil_source"] == "default"
        assert metadata["pupil_source"] == "provided"
        assert metadata["method"] == method
        assert metadata["transform"] == transform
        _normalized, ratio, efficiency = _support_quality(phase, target, pupil)
        if objective_kind == "spots":
            assert metadata["support_intensity_ratio"] == pytest.approx(
                ratio, rel=2e-5
            )
        else:
            assert "support_intensity_ratio" not in metadata
            assert metadata["signal_relative_rms"] >= 0.0
            assert metadata["background_power_fraction"] >= 0.0
        assert metadata["diffraction_efficiency"] == pytest.approx(
            efficiency, rel=2e-5
        )

    default_pupil = (
        xx * xx + yy * yy <= 0.9**2
    ).astype(np.float32)
    default_phase, default_metadata = solve_phase(
        cartesian,
        objective_kind="spots",
        iterations=20,
        seed=41,
    )
    explicit_phase, explicit_metadata = solve_phase(
        cartesian,
        pupil_amplitude=default_pupil,
        objective_kind="spots",
        iterations=20,
        seed=41,
    )
    np.testing.assert_array_equal(explicit_phase, default_phase)
    assert {
        key: value
        for key, value in explicit_metadata.items()
        if key != "pupil_source"
    } == {
        key: value
        for key, value in default_metadata.items()
        if key != "pupil_source"
    }


def test_slm_solver_keeps_one_percent_gate_with_authored_pupil() -> None:
    target = preset_grid((48, 64), (2, 3), spacing_yx=(13, 15))
    yy, xx = np.ogrid[-1.0:1.0:48j, -1.0:1.0:64j]
    pupil = (
        ((xx / 0.76) ** 2 + (yy / 0.84) ** 2 <= 1.0)
        * (0.6 + 0.4 * (xx + 1.0) * 0.5)
    ).astype(np.float32)
    phase, metadata = solve_phase(
        target,
        pupil_amplitude=pupil,
        objective_kind="spots",
        seed=31,
    )
    _normalized, ratio, _efficiency = _support_quality(phase, target, pupil)

    if metadata["early_stopped"]:
        assert metadata["support_intensity_ratio"] <= 1.01
        assert ratio <= 1.01
    else:
        assert metadata["iterations_run"] == metadata["max_iterations"]


def test_slm_solver_reuses_small_plain_state_without_relaxing_quality() -> None:
    target = preset_grid((64, 80), (3, 4), spacing_yx=(13, 14))
    yy, xx = np.ogrid[-1.0:1.0:64j, -1.0:1.0:80j]
    pupil = (
        (xx * xx + yy * yy <= 0.82**2) * (0.75 + 0.25 * (xx + 1.0) * 0.5)
    ).astype(np.float32)
    state: dict[str, object] = {}
    _base_phase, base_metadata = solve_phase(
        target,
        pupil_amplitude=pupil,
        objective_kind="spots",
        iterations=30,
        seed=23,
        spot_optimizer_state=state,
    )
    assert base_metadata["optimizer_state_status"] == "created"
    assert state["shape_yx"] == [64, 80]
    assert len(state["support_yx"]) == 12
    assert len(state["fixed_farfield_phase"]) == 12
    assert len(state["site_weights"]) == 12
    assert len(state["target_amplitudes"]) == 12
    encoded_state = json.dumps(state, allow_nan=False)
    assert "optimizer_state" not in base_metadata

    changed = np.zeros_like(target)
    changed[target > 0.0] = np.linspace(0.8, 1.2, 12, dtype=np.float32)
    candidate_state = json.loads(encoded_state)
    candidate, candidate_metadata = solve_phase(
        changed,
        pupil_amplitude=pupil,
        objective_kind="spots",
        spot_optimizer_state=candidate_state,
        iterations=1,
        seed=999,
    )
    repeated_state = json.loads(encoded_state)
    repeated, repeated_metadata = solve_phase(
        changed,
        pupil_amplitude=pupil,
        objective_kind="spots",
        spot_optimizer_state=repeated_state,
        iterations=1,
        seed=1,
    )
    np.testing.assert_array_equal(candidate, repeated)
    assert {
        key: value for key, value in candidate_metadata.items() if key != "seed"
    } == {
        key: value for key, value in repeated_metadata.items() if key != "seed"
    }
    assert candidate_state == repeated_state
    assert candidate_metadata["optimizer_state_status"] == "reused"
    assert candidate_metadata["hot_start_used"] is True
    assert candidate_metadata["iterations_run"] == 1
    assert candidate_metadata["early_stopped"] is False
    assert candidate_metadata["support_intensity_ratio"] > 1.01
    candidate_ratio, candidate_efficiency = _full_fft_returned_quality(
        candidate, changed, pupil
    )
    assert candidate_metadata["transform"] == "selected-dft"
    assert candidate_metadata["support_intensity_ratio"] == pytest.approx(
        candidate_ratio, rel=2e-5
    )
    assert candidate_metadata["diffraction_efficiency"] == pytest.approx(
        candidate_efficiency, rel=2e-5
    )
    old_amplitudes = np.asarray(
        json.loads(encoded_state)["target_amplitudes"], dtype=np.float32
    )
    new_amplitudes = np.sqrt(
        np.asarray(
            [changed[tuple(yx)] for yx in candidate_state["support_yx"]],
            dtype=np.float32,
        )
    )
    new_amplitudes /= np.linalg.norm(new_amplitudes)
    initialized_weights = np.asarray(
        json.loads(encoded_state)["site_weights"], dtype=np.float32
    ) * (new_amplitudes / old_amplitudes)
    initialized_weights /= np.linalg.norm(initialized_weights)
    assert not np.allclose(
        candidate_state["site_weights"], initialized_weights, rtol=1e-6, atol=1e-7
    )
    continued_state = json.loads(json.dumps(candidate_state, allow_nan=False))
    continued, continued_metadata = solve_phase(
        changed,
        pupil_amplitude=pupil,
        objective_kind="spots",
        spot_optimizer_state=continued_state,
        iterations=1,
    )
    assert continued_metadata["iterations_run"] == 1
    assert not np.array_equal(continued, candidate)

    accepted_state = json.loads(encoded_state)
    accepted, accepted_metadata = solve_phase(
        changed,
        pupil_amplitude=pupil,
        objective_kind="spots",
        spot_optimizer_state=accepted_state,
    )
    _normalized, ratio, _efficiency = _support_quality(accepted, changed, pupil)
    assert accepted_metadata["optimizer_state_status"] == "reused"
    assert accepted_metadata["hot_start_used"] is True
    assert accepted_metadata["early_stopped"] is True
    assert accepted_metadata["iterations_run"] >= 1
    assert accepted_metadata["support_intensity_ratio"] <= 1.01
    assert accepted_metadata["support_tolerance"] == 1.01
    assert accepted_metadata["minimum_iterations"] == 1
    assert ratio <= 1.01

    # A caller may tighten the gate: no early stop before the minimum passes,
    # and none until the support ratio is inside the requested tolerance.
    tightened_state = json.loads(encoded_state)
    _tightened, tightened_metadata = solve_phase(
        changed,
        pupil_amplitude=pupil,
        objective_kind="spots",
        spot_optimizer_state=tightened_state,
        support_tolerance=1.002,
        minimum_iterations=5,
    )
    assert tightened_metadata["support_tolerance"] == 1.002
    assert tightened_metadata["minimum_iterations"] == 5
    assert tightened_metadata["iterations_run"] >= 5
    if tightened_metadata["early_stopped"]:
        assert tightened_metadata["support_intensity_ratio"] <= 1.002
    else:
        assert tightened_metadata["iterations_run"] == tightened_metadata["max_iterations"]
    with pytest.raises(ValueError, match="support_tolerance"):
        solve_phase(changed, objective_kind="spots", support_tolerance=0.99)
    with pytest.raises(ValueError, match="minimum_iterations"):
        solve_phase(changed, objective_kind="spots", minimum_iterations=0)

    stopped_state = json.loads(encoded_state)
    unchanged = json.dumps(stopped_state, sort_keys=True)
    stop_calls = 0

    def stop_during_hot_update() -> bool:
        nonlocal stop_calls
        stop_calls += 1
        return stop_calls == 3

    with pytest.raises(InterruptedError):
        solve_phase(
            changed,
            pupil_amplitude=pupil,
            objective_kind="spots",
            spot_optimizer_state=stopped_state,
            stop_requested=stop_during_hot_update,
        )
    assert json.dumps(stopped_state, sort_keys=True) == unchanged


def test_slm_selected_dft_hot_gate_matches_full_fft_returned_phase() -> None:
    target = preset_grid((64, 80), (3, 4), spacing_yx=(13, 14))
    yy, xx = np.ogrid[-1.0:1.0:64j, -1.0:1.0:80j]
    pupil = (
        (xx * xx + yy * yy <= 0.82**2) * (0.75 + 0.25 * (xx + 1.0) * 0.5)
    ).astype(np.float32)
    state: dict[str, object] = {}
    solve_phase(
        target,
        pupil_amplitude=pupil,
        objective_kind="spots",
        iterations=30,
        seed=23,
        spot_optimizer_state=state,
    )
    encoded_state = json.dumps(state, allow_nan=False)
    changed = np.zeros_like(target)
    changed[target > 0.0] = np.linspace(0.8, 1.2, 12, dtype=np.float32)
    accepted_state = json.loads(encoded_state)
    accepted, metadata = solve_phase(
        changed,
        pupil_amplitude=pupil,
        objective_kind="spots",
        spot_optimizer_state=accepted_state,
    )
    ratio, efficiency = _full_fft_returned_quality(accepted, changed, pupil)
    assert metadata["transform"] == "selected-dft"
    assert metadata["support_intensity_ratio"] == pytest.approx(
        ratio, rel=2e-5
    )
    assert metadata["diffraction_efficiency"] == pytest.approx(
        efficiency, rel=2e-5
    )
    assert metadata["early_stopped"] is True

    accepted_iteration = metadata["iterations_run"]
    exact_state = json.loads(encoded_state)
    exact, exact_metadata = solve_phase(
        changed,
        pupil_amplitude=pupil,
        objective_kind="spots",
        iterations=accepted_iteration,
        spot_optimizer_state=exact_state,
    )
    assert exact_metadata["iterations_run"] == accepted_iteration
    np.testing.assert_array_equal(exact, accepted)
    if accepted_iteration > 1:
        previous_state = json.loads(encoded_state)
        previous, _previous_metadata = solve_phase(
            changed,
            pupil_amplitude=pupil,
            objective_kind="spots",
            iterations=accepted_iteration - 1,
            spot_optimizer_state=previous_state,
        )
        assert _full_fft_returned_quality(previous, changed, pupil)[0] > 1.01


def test_slm_optimizer_state_explicitly_invalidates_on_changed_physics() -> None:
    target = preset_grid((64, 80), (3, 4), spacing_yx=(13, 14))
    state: dict[str, object] = {}
    _phase, metadata = solve_phase(
        target,
        objective_kind="spots",
        iterations=30,
        seed=23,
        spot_optimizer_state=state,
    )
    assert metadata["optimizer_state_status"] == "created"
    encoded_state = json.dumps(state, allow_nan=False)

    moved = preset_grid((64, 80), (3, 4), spacing_yx=(12, 15))
    moved_state = json.loads(encoded_state)
    _moved_phase, moved_metadata = solve_phase(
        moved,
        objective_kind="spots",
        spot_optimizer_state=moved_state,
        iterations=20,
        seed=23,
    )
    assert moved_metadata["optimizer_state_status"] == "support-changed"
    assert moved_metadata["hot_start_used"] is False

    yy, xx = np.ogrid[-1.0:1.0:64j, -1.0:1.0:80j]
    changed_pupil = (
        (xx * xx + yy * yy <= 0.75**2) * (0.7 + 0.3 * (xx + 1.0) * 0.5)
    ).astype(np.float32)
    pupil_state = json.loads(encoded_state)
    _pupil_phase, pupil_metadata = solve_phase(
        target,
        pupil_amplitude=changed_pupil,
        objective_kind="spots",
        spot_optimizer_state=pupil_state,
        iterations=20,
        seed=23,
    )
    assert pupil_metadata["optimizer_state_status"] == "pupil-changed"
    assert pupil_metadata["hot_start_used"] is False

    image_state = json.loads(encoded_state)
    _image_phase, image_metadata = solve_phase(
        target,
        objective_kind="image",
        spot_optimizer_state=image_state,
        iterations=1,
        seed=23,
    )
    assert image_metadata["optimizer_state_status"] == "objective-changed"
    assert image_metadata["hot_start_used"] is False
    assert "optimizer_state" not in image_metadata
    assert image_state == {}


def test_slm_solver_uses_authored_spot_or_image_objective() -> None:
    adjacent = np.zeros((64, 64), dtype=np.float32)
    adjacent[32, 30:35] = 1.0
    grid = preset_grid((64, 64), (5, 7), spacing_yx=(7, 6))
    for target in (adjacent, grid):
        _phase, metadata = solve_phase(
            target,
            objective_kind="spots",
            iterations=4,
            seed=17,
        )
        assert metadata["method"] == "wgs-kim"
    _auto_phase, auto_metadata = solve_phase(
        adjacent,
        objective_kind="auto",
        iterations=4,
        seed=17,
    )
    assert auto_metadata["method"] == "wgs-kim"

    dense = preset_flat_top((64, 64), (11, 13), edge=4)
    _dense_phase, dense_metadata = solve_phase(
        dense,
        objective_kind="image",
        iterations=4,
        seed=9,
    )
    assert dense_metadata["method"] == "mraf"
    with pytest.raises(ValueError, match="objective_kind"):
        solve_phase(grid, objective_kind="unknown", iterations=1)


@pytest.mark.parametrize("cartesian", [True, False])
def test_sparse_solver_matches_full_shifted_wgs_kim_quality(cartesian: bool) -> None:
    target = np.array(
        preset_grid((64, 80), (3, 4), spacing_yx=(12, 13)), copy=True
    )
    if not cartesian:
        row, column = np.argwhere(target > 0.0)[-1]
        target[row, column] = 0.0
    phase, metadata = solve_phase(
        target,
        objective_kind="spots",
        iterations=30,
        seed=23,
    )
    reference = _shifted_wgs_kim_reference(target, iterations=30, seed=23)
    normalized, ratio, efficiency = _support_quality(phase, target)
    reference_normalized, reference_ratio, reference_efficiency = _support_quality(
        reference, target
    )

    assert metadata["transform"] == "selected-dft"
    np.testing.assert_allclose(
        normalized,
        reference_normalized,
        rtol=0.01,
        atol=1e-4,
    )
    assert ratio == pytest.approx(reference_ratio, rel=0.01)
    assert efficiency == pytest.approx(reference_efficiency, rel=0.01)


def test_selected_dft_active_mask_preserves_spot_order_and_hot_state() -> None:
    target = np.array(
        preset_checkerboard(
            (48, 64), (3, 4), spacing_yx=(9, 5), intensity=1.0
        ),
        copy=True,
    )
    authored_sites = np.argwhere(target > 0.0)
    target[tuple(authored_sites.T)] = np.linspace(
        0.6, 1.4, len(authored_sites), dtype=np.float32
    )
    state: dict[str, object] = {}
    _phase, metadata = solve_phase(
        target,
        objective_kind="spots",
        iterations=12,
        seed=29,
        spot_optimizer_state=state,
    )
    assert metadata["transform"] == "selected-dft"
    support_yx = np.asarray(state["support_yx"], dtype=np.intp)
    expected_amplitudes = np.sqrt(target[tuple(support_yx.T)])
    expected_amplitudes /= np.linalg.norm(expected_amplitudes)
    np.testing.assert_allclose(
        state["target_amplitudes"], expected_amplitudes, rtol=1e-6, atol=1e-7
    )

    changed = np.zeros_like(target)
    changed[tuple(authored_sites.T)] = np.linspace(
        1.4, 0.6, len(authored_sites), dtype=np.float32
    )
    _hot, hot_metadata = solve_phase(
        changed,
        objective_kind="spots",
        iterations=1,
        spot_optimizer_state=state,
    )
    assert hot_metadata["transform"] == "selected-dft"
    assert hot_metadata["optimizer_state_status"] == "reused"
    assert hot_metadata["hot_start_used"] is True

    diagonal = np.zeros((320, 320), dtype=np.float32)
    indices = np.arange(256)
    diagonal[indices, indices] = 1.0
    _fallback, fallback_metadata = solve_phase(
        diagonal, objective_kind="spots", iterations=1
    )
    assert fallback_metadata["transform"] == "fft"


@pytest.mark.parametrize("irregular", (False, True))
def test_sparse_solver_early_stops_only_after_exact_returned_phase_quality(
    irregular: bool,
) -> None:
    target = preset_grid((64, 64), (2, 3), spacing_yx=(14, 13))
    if irregular:
        target = np.array(target, copy=True)
        target[tuple(np.argwhere(target > 0.0)[-1])] = 0.0
    calls = 0

    def keep_running() -> bool:
        nonlocal calls
        calls += 1
        return False

    _exact, exact_metadata = solve_phase(
        target,
        objective_kind="spots",
        iterations=24,
        seed=31,
        stop_requested=keep_running,
    )
    assert calls == 24
    assert exact_metadata["iterations"] == 24
    assert exact_metadata["iterations_run"] == 24
    assert exact_metadata["early_stopped"] is False
    assert exact_metadata["stop_reason"] == "iteration-limit"

    phase, metadata = solve_phase(target, objective_kind="spots", seed=31)
    _normalized, ratio, _efficiency = _support_quality(phase, target)
    assert metadata["early_stopped"] is True
    assert metadata["stop_reason"] == "support-ratio"
    assert metadata["transform"] == "selected-dft"
    assert metadata["support_intensity_ratio"] <= 1.01
    assert ratio <= 1.01
    assert metadata["iterations"] < metadata["max_iterations"]


def test_one_slm_solver_selects_sparse_wgs_and_dense_mraf() -> None:
    sparse = preset_grid((64, 64), (3, 5), spacing_yx=(10, 9))
    phase, metadata = solve_phase(
        sparse, objective_kind="spots", iterations=80, seed=17
    )
    repeated, repeated_metadata = solve_phase(
        sparse, objective_kind="spots", iterations=80, seed=17
    )
    np.testing.assert_array_equal(phase, repeated)
    assert metadata == repeated_metadata
    assert metadata["method"] == "wgs-kim"
    assert phase.dtype == np.dtype("<f4")
    assert np.all((phase >= 0.0) & (phase < 2.0 * np.pi))
    site_values = _ideal_slm_intensity(phase)[sparse > 0.0]
    assert float(np.max(site_values) / np.min(site_values)) <= 1.01

    checkerboard = preset_checkerboard(
        (64, 64),
        (3, 4),
        spacing_yx=(10, 9),
        intensity=0.7,
    )
    checker_phase, checker_metadata = solve_phase(
        checkerboard, objective_kind="spots", iterations=80, seed=17
    )
    assert checker_metadata["method"] == "wgs-kim"
    assert checker_metadata["transform"] == "selected-dft"
    checker_values = _ideal_slm_intensity(checker_phase)[checkerboard > 0.0]
    assert float(np.max(checker_values) / np.min(checker_values)) <= 1.01

    dense = preset_flat_top((64, 64), (11, 13), edge=4)
    dense_phase, dense_metadata = solve_phase(
        dense, objective_kind="image", seed=9
    )
    assert dense_metadata["method"] == "mraf"
    assert dense_metadata["early_stopped"] is True
    # A target with a real flat interior now stops on the physical interior
    # uniformity gate (the image analogue of the spots support gate), not on
    # merit stagnation -- the quality assertions below are what it guarantees.
    assert dense_metadata["stop_reason"] == "interior-uniformity"
    assert dense_metadata["iterations"] < dense_metadata["max_iterations"]
    assert dense_metadata["signal_pixels"] > 0
    assert dense_metadata["noise_pixels"] > 0
    assert dense_metadata["figure_of_merit"] >= 0.0
    assert dense_metadata["signal_relative_rms"] >= 0.0
    assert 0.0 <= dense_metadata["background_power_fraction"] <= 1.0
    assert dense_metadata["signal_roughness"] >= 0.0
    intensity = _ideal_slm_intensity(dense_phase)
    interior = dense >= 0.999
    values = intensity[interior]
    assert float(np.percentile(values, 95) / np.percentile(values, 5)) <= 1.01

    _exact_dense, exact_dense_metadata = solve_phase(
        dense, objective_kind="image", iterations=37, seed=9
    )
    assert exact_dense_metadata["iterations_run"] == 37
    assert exact_dense_metadata["early_stopped"] is False
    assert exact_dense_metadata["stop_reason"] == "iteration-limit"

    warmed, warm_metadata = solve_phase(
        sparse,
        initial_phase=phase,
        objective_kind="spots",
        iterations=8,
        seed=999,
    )
    assert warm_metadata["method"] == "wgs-kim"
    assert warmed.shape == sparse.shape
    calls = 0

    def stop_requested() -> bool:
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(InterruptedError):
        solve_phase(
            sparse,
            objective_kind="spots",
            iterations=80,
            stop_requested=stop_requested,
        )
    with pytest.raises(ValueError, match="positive intensity"):
        solve_phase(np.zeros((16, 16), dtype=np.float32))


def test_slm_target_json_is_a_strict_objective_bearing_artifact(tmp_path: Path) -> None:
    target = preset_flat_top((24, 32), (7, 11), edge=2)
    target_path = save_target(
        tmp_path / "target.json", target, objective_kind="image"
    )
    loaded_target, loaded_objective = load_target(target_path)
    np.testing.assert_array_equal(loaded_target, target)
    assert loaded_objective == "image"
    assert set(json.loads(target_path.read_text(encoding="utf-8"))) == {
        "format", "shape", "intensity", "objective_kind",
    }

    malformed_target = tmp_path / "bad-target.json"
    malformed_target.write_text(
        '{"format":"zlc.slm.target","format":"zlc.slm.target",'
        '"shape":[1,1],"intensity":[[1.0]],'
        '"objective_kind":"spots"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_target(malformed_target)

    for name, payload in (
        (
            "unknown-field.json",
            '{"format":"zlc.slm.target","unexpected":2,'
            '"shape":[2,2],"intensity":[[1,0],[0,1]],'
            '"objective_kind":"spots"}',
        ),
        (
            "bad-shape.json",
            '{"format":"zlc.slm.target",'
            '"shape":[2.0,2],"intensity":[[1,0],[0,1]],'
            '"objective_kind":"spots"}',
        ),
        (
            "bad-value.json",
            '{"format":"zlc.slm.target",'
            '"shape":[2,2],"intensity":[["1",0],[0,1]],'
            '"objective_kind":"spots"}',
        ),
        (
            "bad-format.json",
            '{"format":"zlc.slm.other",'
            '"shape":[2,2],"intensity":[[1,0],[0,1]],'
            '"objective_kind":"spots"}',
        ),
    ):
        malformed = tmp_path / name
        malformed.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError):
            load_target(malformed)

def test_science_context_roundtrip_freezes_layers_pupil_receipt_and_correction(
    tmp_path: Path,
) -> None:
    shape = (8, 10)
    target = preset_grid(shape, (2, 3))
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    pattern = canonical_phase(np.broadcast_to(0.1 + xx / 5.0, shape), shape)
    frozen_pattern = freeze_pattern_phase(pattern, shape)
    pupil = {
        "enabled": True,
        "center_xy": [4.5, 3.5],
        "diameter_xy": [7.0, 6.0],
    }
    operator_metadata = {
        "enabled": True,
        "carrier_waves_xy": [1.0, -0.5],
        "zernike_noll_waves_rms": {"defocus": 0.125},
    }
    support, amplitude = science_pupil_fields(shape, pupil)
    wavefront = science_operator_wavefront(shape, pupil, operator_metadata)
    phase = compose_science_phase(frozen_pattern, wavefront)
    receipt = {
        "transport": "usb",
        "identity": "hamamatsu-x15213:usb:LSH0804382",
        "profile": "LSH0804382",
        "model": "X15213",
        "serial": "LSH0804382",
        "wavelength_nm": 852.0,
        "flip_x": False,
        "flip_y": True,
        "correction_path": "CAL_LSH0804382_852nm.bmp",
        "correction_enabled": True,
        "mapping_revision": 3,
        "settle_seconds": 0.05,
        "phase_curve_source": "workspace/profile.json",
        "outcome": "known-new",
        "command_revision": 7,
        "stage": "settled",
        "readback": "exact-frame-memory",
    }
    correction = {
        "kind": "pupil_phase_map",
        "reference": "workspace/corrections/my_correction.npz",
        "wavelength_nm": 852.0,
        "pupil": {
            "enabled": True,
            "center_xy": [4.5, 3.5],
            "diameter_xy": [7.0, 6.0],
        },
        "coordinate_system": "slm-pixel-xy",
        "valid_region": "frozen pupil support",
        "measurement_method": "dense-grid interferometry",
    }
    path = save_science_context(
        tmp_path / "context.npz",
        pattern,
        target_intensity=target,
        objective_kind="spots",
        pupil=pupil,
        system_correction=correction,
        command_receipt=receipt,
        pattern_metadata={"method": "wgs-kim", "iterations": 80},
        operator_metadata=operator_metadata,
    )
    context = load_science_context(path)
    assert slm_solver.SCIENCE_CONTEXT_ARTIFACT_CONTRACT == (
        "zlc.slm.science-context"
    )
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "pattern_phase_delta", "target_intensity", "metadata",
        }
        assert archive["pattern_phase_delta"].dtype == np.dtype("<u2")
        assert json.loads(str(archive["metadata"].item()))["format"] == (
            "zlc.slm.science-context"
        )
    for key, expected in (
        ("phase", phase),
        ("pattern_phase", frozen_pattern),
        ("operator_wavefront", wavefront),
        ("pupil_amplitude", amplitude),
        ("pupil_support", support),
        ("target_intensity", target),
    ):
        np.testing.assert_array_equal(context[key], expected)
        assert not context[key].flags.writeable
    assert context["objective_kind"] == "spots"
    assert context["system_correction"] == correction
    assert context["command_receipt"] == receipt
    assert context["pupil"]["center_xy"] == [4.5, 3.5]

    missing_target_path = tmp_path / "missing-target-context.npz"
    with np.load(path, allow_pickle=False) as archive:
        np.savez(
            missing_target_path,
            **{
                key: archive[key]
                for key in archive.files if key != "target_intensity"
            },
        )
    with pytest.raises(ValueError, match="wrong members"):
        load_science_context(missing_target_path)

    unknown_path = tmp_path / "unknown-context-field.npz"
    with np.load(path, allow_pickle=False) as archive:
        unknown_metadata = json.loads(str(archive["metadata"].item()))
        unknown_metadata["unexpected"] = 2
        np.savez(
            unknown_path,
            **{
                key: archive[key]
                for key in archive.files if key != "metadata"
            },
            metadata=np.asarray(json.dumps(unknown_metadata)),
        )
    with pytest.raises(ValueError, match="metadata has the wrong fields"):
        load_science_context(unknown_path)

    save_science_context(
        tmp_path / "response-context.npz",
        pattern,
        target_intensity=target,
        objective_kind="spots",
        pupil=pupil,
        system_correction={
            **correction,
            "kind": "target_response_map",
            "reference": "workspace/corrections/site_response.npz",
            "measurement_method": "bright-dark fluorescence",
        },
        command_receipt={
            **receipt,
            "transport": "virtual",
            "identity": "virtual-simulation:slm",
            "profile": "virtual",
            "wavelength_nm": None,
        },
        pattern_metadata={},
        operator_metadata={
            "enabled": False,
            "carrier_waves_xy": [0.0, 0.0],
            "zernike_noll_waves_rms": {},
        },
    )
    with pytest.raises(ValueError, match="pupil_phase_map or target_response_map"):
        save_science_context(
            tmp_path / "ambiguous.npz",
            pattern,
            target_intensity=target,
            objective_kind="spots",
            pupil=pupil,
            system_correction={
                **correction,
                "kind": "both",
                "reference": "ambiguous.npz",
            },
            command_receipt=receipt,
            pattern_metadata={},
            operator_metadata={
                "enabled": False,
                "carrier_waves_xy": [0.0, 0.0],
                "zernike_noll_waves_rms": {},
            },
        )
