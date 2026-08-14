"""Installation-owned atom/imaging world shared by simulation devices."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import threading
from typing import Any, Callable

import numpy as np

from zlc_pulse.compile import CompiledProgram
from zlc_pulse.schedule import run_duration_seconds, trigger_windows


DEFAULT_SIMULATION_GRID_SHAPE_YX = (5, 7)
DEFAULT_SIMULATION_IMAGE_SHAPE_YX = (96, 128)
DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX = (1200, 1920)
#: Where the MOT is actually best, in DAC codes -- the ground truth a field
#: optimisation is supposed to FIND.  Non-zero on purpose: a real bench has a
#: stray ambient field and the bias coils exist to cancel it, so an optimiser
#: that starts at zero must genuinely move.  The world models the ambient as
#: exactly -optimum, making the net field (dac - optimum) / 512 per axis.
DEFAULT_MOT_FIELD_OPTIMUM_DAC = (96, -144, 48)
DEFAULT_SIMULATION_SITE_SPACING_PIXELS = 9.0
DEFAULT_SIMULATION_SLM_SHAPE_YX = (128, 128)

#: 87-Rb, the atom this bench traps: 86.909 180 5 u.
RB87_MASS_KG = 1.443160648e-25
BOLTZMANN_J_PER_K = 1.380649e-23

#: What the traps hold, and the trap that holds it.  A release loses atoms
#: because atoms MOVE: switch the light off and every atom flies at the speed
#: it already had, and the ones that walk out of the trap's reach before it
#: comes back are gone.  So the loss is set by how fast the atoms are (their
#: temperature), how far they may go (the trap's reach), and how fast an atom
#: may be and still be held at all (the trap's depth, quoted as a temperature
#: the way traps are).  Those are apparatus facts and they live here, on the
#: apparatus -- a measurement asks the bench what happened, it does not tell
#: the bench what the answer should be.
DEFAULT_ATOM_TEMPERATURE_K = 2.0e-5
DEFAULT_TRAP_DEPTH_K = 1.0e-3
DEFAULT_TRAP_WAIST_M = 1.0e-6


def _maxwell_boltzmann_below(speed: float, most_probable_speed: float) -> float:
    """The fraction of a thermal gas slower than ``speed``.

    The Maxwell-Boltzmann speed distribution, integrated: erf(x) minus the
    2 x exp(-x^2) / sqrt(pi) that the v^2 weight adds back.
    """

    if speed <= 0.0:
        return 0.0
    x = float(speed) / float(most_probable_speed)
    return math.erf(x) - 2.0 * x * math.exp(-x * x) / math.sqrt(math.pi)


def _readonly(values: object) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=float)
    array.setflags(write=False)
    return array


def _immutable(values: object, dtype: object) -> np.ndarray:
    """Own an array through immutable bytes, so callers cannot re-enable writes."""

    contiguous = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@lru_cache(maxsize=None)
def _nominal_slm_command(
    shape_yx: tuple[int, int],
    grid_shape_yx: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve each apparatus geometry once; the nominal target is seed-independent."""

    from zlc_atom.devices.slm.solver import preset_grid, solve_phase

    target = preset_grid(shape_yx, grid_shape_yx)
    indices = np.argwhere(target > 0.0)
    phase, _metadata = solve_phase(target, seed=0)
    return _immutable(indices, np.intp), _immutable(phase, "<f4")


@dataclass(frozen=True)
class SimulationGeometry:
    """One source of truth for virtual site geometry and authoring defaults."""

    grid_shape_yx: tuple[int, int] = DEFAULT_SIMULATION_GRID_SHAPE_YX
    image_shape_yx: tuple[int, int] = DEFAULT_SIMULATION_IMAGE_SHAPE_YX
    site_spacing_pixels: float = DEFAULT_SIMULATION_SITE_SPACING_PIXELS

    def __post_init__(self) -> None:
        grid = tuple(int(item) for item in self.grid_shape_yx)
        image = tuple(int(item) for item in self.image_shape_yx)
        if len(grid) != 2 or len(image) != 2 or any(item <= 0 for item in (*grid, *image)):
            raise ValueError("simulation geometry dimensions must be positive")
        object.__setattr__(self, "grid_shape_yx", grid)
        object.__setattr__(self, "image_shape_yx", image)
        spacing = float(self.site_spacing_pixels)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("site_spacing_pixels must be positive and finite")
        if (grid[0] - 1) * spacing >= image[0] or (grid[1] - 1) * spacing >= image[1]:
            raise ValueError("simulation site grid does not fit inside the image")
        object.__setattr__(self, "site_spacing_pixels", spacing)

    @property
    def site_centers_xy(self) -> np.ndarray:
        rows, columns = self.grid_shape_yx
        height, width = self.image_shape_yx
        spacing = self.site_spacing_pixels
        y0 = (height - (rows - 1) * spacing) * 0.5
        x0 = (width - (columns - 1) * spacing) * 0.5
        y_values = y0 + np.arange(rows, dtype=float) * spacing
        x_values = x0 + np.arange(columns, dtype=float) * spacing
        return np.asarray([(x, y) for y in y_values for x in x_values], dtype="<f8")


@dataclass(frozen=True)
class SimulationWorldConfig:
    """Resolved apparatus contribution used to construct one shared world."""

    geometry: SimulationGeometry
    seed: int = 0
    mot_field_optimum_dac: tuple[int, int, int] = DEFAULT_MOT_FIELD_OPTIMUM_DAC

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, SimulationGeometry):
            raise TypeError("geometry must be SimulationGeometry")
        object.__setattr__(self, "seed", int(self.seed))
        optimum = tuple(int(value) for value in self.mot_field_optimum_dac)
        if len(optimum) != 3 or any(abs(value) > 511 for value in optimum):
            raise ValueError(
                "mot_field_optimum_dac must be three DAC codes within the bus range"
            )
        object.__setattr__(self, "mot_field_optimum_dac", optimum)


class SimulationWorld:
    """Explicit state and trigger routing for all virtual devices."""

    def __init__(
        self,
        geometry: SimulationGeometry | None = None,
        *,
        seed: int = 0,
        mot_field_optimum_dac: tuple[int, int, int] = DEFAULT_MOT_FIELD_OPTIMUM_DAC,
    ) -> None:
        self.geometry = SimulationGeometry() if geometry is None else geometry
        seed = int(seed)
        optimum = tuple(int(value) for value in mot_field_optimum_dac)
        if len(optimum) != 3 or any(abs(value) > 511 for value in optimum):
            raise ValueError(
                "mot_field_optimum_dac must be three DAC codes within the bus range"
            )
        self._mot_field_optimum = dict(
            zip(("da_bias_x", "da_bias_y", "da_bias_z"), optimum)
        )
        self.rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self._cameras: list[tuple[Any, Callable[..., np.ndarray] | None]] = []
        self._fire_count = 0
        self.offset_counts = 200.0
        self.conversion_e_per_count = 0.107
        self.read_noise_e = 0.43
        self.background_rate = 300.0
        self.atom_rate = 1_100.0
        self.atom_sigma_px = 0.7
        self.loading_probability = 0.5
        self.atom_temperature_k = DEFAULT_ATOM_TEMPERATURE_K
        self.trap_depth_k = DEFAULT_TRAP_DEPTH_K
        self.trap_waist_m = DEFAULT_TRAP_WAIST_M
        self.dark_current_e_per_s = 0.0
        site_count = len(self.geometry.site_centers_xy)
        self._slm_shape_yx = DEFAULT_SIMULATION_SLM_SHAPE_YX
        self._slm_site_indices_yx, nominal_phase = _nominal_slm_command(
            self._slm_shape_yx,
            self.geometry.grid_shape_yx,
        )
        if len(self._slm_site_indices_yx) != site_count:
            raise RuntimeError("nominal SLM target differs from simulation site geometry")
        self._commanded_phase = _immutable(nominal_phase, "<f4")
        self._slm_phase_revision = 0
        self._propagated_revision = -1
        self._trap_plane_intensity: np.ndarray | None = None
        self._site_trap_intensities: np.ndarray | None = None
        self._propagation_count = 0
        self._loading_intensity_scale: float | None = None
        self._slm_pupil_amplitude, self._hidden_slm_aberration = (
            self._slm_plant(seed)
        )
        efficiency_log = self.rng.normal(0.0, 1.0, site_count)
        efficiency_log -= float(np.min(efficiency_log))
        span = float(np.ptp(efficiency_log))
        if span:
            efficiency_log *= np.log(1.02) / span
        efficiency_log -= 0.5 * np.log(1.02)
        self._detector_efficiency = _readonly(np.exp(efficiency_log))
        aspect = np.sqrt(1.25)
        base_sigma = np.asarray(
            (self.atom_sigma_px / aspect, self.atom_sigma_px * aspect),
            dtype=float,
        )
        self._site_psf_sigma_xy = _readonly(
            base_sigma[np.newaxis, :]
            * np.exp(self.rng.normal(0.0, 0.10, (site_count, 2)))
        )
        self._site_psf_angle_radians = _readonly(
            np.deg2rad(18.0 + self.rng.normal(0.0, 5.0, site_count))
        )
        self._site_psf_skew = _readonly(
            np.clip(0.45 + self.rng.normal(0.0, 0.08, site_count), 0.15, 0.75)
        )
        height, width = self.geometry.image_shape_yx
        yy, xx = np.mgrid[:height, :width]
        site_psf_spots = []
        for (x, y), gain, sigma_xy, angle, skew in zip(
            self.geometry.site_centers_xy,
            self._detector_efficiency,
            self._site_psf_sigma_xy,
            self._site_psf_angle_radians,
            self._site_psf_skew,
            strict=True,
        ):
            sigma_x, sigma_y = (float(value) for value in sigma_xy)
            cosine, sine = np.cos(angle), np.sin(angle)
            dx = (xx - x) * cosine + (yy - y) * sine
            dy = -(xx - x) * sine + (yy - y) * cosine
            core = np.exp(
                -0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2)
            )
            spot = np.clip(
                core * (1.0 + float(skew) * dx / sigma_x), 0.0, None
            )
            site_psf_spots.append(spot)
        self._site_psf_spots = _readonly(site_psf_spots)
        self._occupancy = np.zeros(site_count, dtype=bool)
        self._forced_occupancy: np.ndarray | None = None
        self._mot_population = 0.0
        self._dac_values = {"da_bias_x": 0, "da_bias_y": 0, "da_bias_z": 0}
        #: Read-only pixel coordinate vectors per MOT frame shape.  A frame
        #: shape is a configuration fact, so this holds one or two entries.
        self._mot_axis_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

        # Establish the physical intensity scale once from the nominal command.
        # Later commands are compared against that fixed bench scale, rather
        # than renormalized per phase (which would erase diffraction loss).
        self._ensure_slm_propagation()
        self._loading_intensity_scale = float(np.mean(self._site_trap_intensities))

    def _slm_plant(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Materialize fixed illumination and hidden low-order wavefront error."""

        height, width = self._slm_shape_yx
        yy, xx = np.ogrid[
            -1.0 : 1.0 : height * 1j,
            -1.0 : 1.0 : width * 1j,
        ]
        radius_squared = xx * xx + yy * yy
        pupil = radius_squared <= 0.9**2
        # A slightly decentered Gaussian beam plus mild edge vignetting.  These
        # are fixed apparatus optics, not per-shot gain or solver knowledge.
        decenter_x, decenter_y = 0.055, -0.04
        illumination = np.exp(
            -0.34 * ((xx - decenter_x) ** 2 + 1.12 * (yy - decenter_y) ** 2)
        )
        illumination *= np.clip(1.0 - 0.16 * radius_squared, 0.0, None)
        amplitude = np.where(pupil, illumination, 0.0)

        # Seed changes the hidden bench, not the nominal command.  A fixed
        # low-order component ensures every supported seed is meaningfully
        # uncorrected; bounded jitter prevents a one-seed-only simulation.
        coefficients = np.asarray((2.34, -1.56, 1.885, -1.17), dtype=float)
        coefficients += np.random.default_rng(seed ^ 0x5A17).uniform(
            -0.02, 0.02, coefficients.shape
        )
        defocus = 2.0 * radius_squared - 1.0
        astigmatism = xx * xx - yy * yy
        coma_x = (3.0 * radius_squared - 2.0) * xx
        coma_y = (3.0 * radius_squared - 2.0) * yy
        aberration = (
            coefficients[0] * defocus
            + coefficients[1] * astigmatism
            + coefficients[2] * coma_x
            + coefficients[3] * coma_y
        )
        return (
            _immutable(amplitude, "<f4"),
            _immutable(np.where(pupil, aberration, 0.0), "<f4"),
        )

    @property
    def slm_shape_yx(self) -> tuple[int, int]:
        return self._slm_shape_yx

    @property
    def commanded_phase(self) -> np.ndarray:
        with self._lock:
            return self._commanded_phase

    @property
    def slm_phase_revision(self) -> int:
        with self._lock:
            return self._slm_phase_revision

    def apply_slm_phase(self, radians: object) -> np.ndarray:
        """Atomically accept one explicit command and invalidate propagation."""

        from zlc_atom.devices.slm import canonical_phase

        commanded = canonical_phase(radians, self._slm_shape_yx)
        with self._lock:
            self._commanded_phase = commanded
            self._slm_phase_revision += 1
            self._propagated_revision = -1
            return self._commanded_phase

    def _ensure_slm_propagation(self) -> None:
        if self._propagated_revision == self._slm_phase_revision:
            return
        from scipy import fft

        # The simulated panel has 256 phase levels.  Quantization affects the
        # optical field only; last-commanded remains canonical radians.
        levels = np.remainder(
            np.rint(self._commanded_phase * (256.0 / (2.0 * np.pi))),
            256.0,
        )
        phase = levels * (2.0 * np.pi / 256.0)
        field = self._slm_pupil_amplitude * np.exp(
            1j * (phase + self._hidden_slm_aberration)
        )
        far_field = fft.fftshift(
            fft.fft2(fft.ifftshift(field), norm="ortho")
        )
        intensity = np.abs(far_field) ** 2
        rows, columns = self._slm_site_indices_yx.T
        sites = intensity[rows, columns]
        self._trap_plane_intensity = _immutable(intensity, "<f4")
        self._site_trap_intensities = _immutable(sites, "<f4")
        self._propagated_revision = self._slm_phase_revision
        self._propagation_count += 1

    @property
    def propagation_count(self) -> int:
        with self._lock:
            return self._propagation_count

    def _site_loading_probabilities(self) -> np.ndarray:
        self._ensure_slm_propagation()
        scale = self._loading_intensity_scale
        if scale is None or not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("nominal SLM command produced no site intensity")
        relative_depth = np.clip(self._site_trap_intensities / scale, 0.0, None)
        base = float(self.loading_probability)
        if not 0.0 <= base <= 1.0:
            raise ValueError("loading_probability must be between zero and one")
        if base == 1.0:
            return np.ones_like(relative_depth)
        return 1.0 - np.power(1.0 - base, relative_depth)

    @property
    def detector_efficiency(self) -> np.ndarray:
        """Small fixed fluorescence-readout nuisance, never trap-depth truth."""

        return np.array(self._detector_efficiency, copy=True)

    @property
    def site_psf_sigma_xy(self) -> np.ndarray:
        return np.array(self._site_psf_sigma_xy, copy=True)

    @property
    def site_psf_angle_radians(self) -> np.ndarray:
        return np.array(self._site_psf_angle_radians, copy=True)

    @property
    def site_psf_skew(self) -> np.ndarray:
        return np.array(self._site_psf_skew, copy=True)

    def register_camera(
        self,
        camera: Any,
        renderer: Callable[..., np.ndarray] | None = None,
    ) -> None:
        with self._lock:
            if not any(existing is camera for existing, _renderer in self._cameras):
                self._cameras.append((camera, renderer))

    def set_occupancy(self, occupancy: object) -> None:
        values = np.asarray(occupancy, dtype=bool).reshape(-1)
        if values.size != len(self.geometry.site_centers_xy):
            raise ValueError("occupancy size differs from simulation site map")
        with self._lock:
            self._forced_occupancy = np.array(values, copy=True)
            self._occupancy = np.array(values, copy=True)

    @property
    def occupancy(self) -> np.ndarray:
        with self._lock:
            return np.array(self._occupancy, copy=True)

    @property
    def fire_count(self) -> int:
        with self._lock:
            return self._fire_count

    def _load_shot(self) -> np.ndarray:
        if self._forced_occupancy is None:
            shot = self.rng.random(len(self.geometry.site_centers_xy)) < self._site_loading_probabilities()
        else:
            shot = np.array(self._forced_occupancy, copy=True)
        self._occupancy = shot
        self._mot_population = 1.0
        return np.array(shot, copy=True)

    def _release_survival(self, trap_off_seconds: float, relative_depth: float) -> float:
        off_time = float(trap_off_seconds)
        depth_scale = float(relative_depth)
        if not np.isfinite(off_time) or off_time < 0:
            raise ValueError("trap_off_seconds must be finite and non-negative")
        if not np.isfinite(depth_scale) or depth_scale < 0:
            raise ValueError("relative trap depth must be finite and non-negative")
        if depth_scale == 0.0:
            return 0.0
        most_probable = math.sqrt(
            2.0 * BOLTZMANN_J_PER_K * self.atom_temperature_k / RB87_MASS_KG
        )
        escape = math.sqrt(
            2.0
            * BOLTZMANN_J_PER_K
            * self.trap_depth_k
            * depth_scale
            / RB87_MASS_KG
        )
        bound = _maxwell_boltzmann_below(escape, most_probable)
        if bound <= 0.0:
            return 0.0
        if off_time == 0.0:
            return 1.0
        # A shallower local trap has both a lower escape speed and a smaller
        # effective recapture reach.  At the nominal depth this is exactly the
        # established release-recapture curve.
        reach_speed = self.trap_waist_m * math.sqrt(depth_scale) / off_time
        recaptured = min(escape, reach_speed)
        return _maxwell_boltzmann_below(recaptured, most_probable) / bound

    def release_survival(self, trap_off_seconds: float) -> float:
        """The chance one trapped atom is still trapped after a release.

        The trap is off for ``trap_off_seconds`` and every atom flies at the
        speed it had.  It is back inside the trap's reach at the end only if
        it was slower than reach/time; it was bound in the first place only
        if it was slower than the depth allows.  So the survivors are the
        slow tail of the thermal distribution, and the release time alone
        decides how slow that tail has to be.

        This is what makes release-recapture a THERMOMETER: the shape of
        this curve against the release time is the atoms' temperature, and
        nothing else about the bench enters it.
        """

        return self._release_survival(trap_off_seconds, 1.0)

    def _site_survival_probabilities(self, trap_off_seconds: float) -> np.ndarray:
        self._ensure_slm_propagation()
        scale = self._loading_intensity_scale
        if scale is None or not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("nominal SLM command produced no site intensity")
        relative_depth = np.clip(self._site_trap_intensities / scale, 0.0, None)
        return np.asarray(
            [
                self._release_survival(trap_off_seconds, float(value))
                for value in relative_depth
            ],
            dtype=float,
        )

    def _lose_atoms(self, trap_off_seconds: float) -> None:
        off_time = float(trap_off_seconds)
        if not np.isfinite(off_time) or off_time < 0:
            raise ValueError("trap_off_seconds must be finite and non-negative")
        if off_time:
            survival = self._site_survival_probabilities(off_time)
            self._occupancy &= self.rng.random(self._occupancy.size) < survival
            self._mot_population *= float(np.mean(survival))

    def safe(self) -> None:
        """Return the simulated apparatus to the board target's safe outputs."""

        with self._lock:
            self._occupancy[:] = False
            self._mot_population = 0.0
            self._dac_values.update(da_bias_x=0, da_bias_y=0, da_bias_z=0)

    def render_frame(
        self,
        ordinal: int,
        *,
        exposure_seconds: float = 0.005,
        probe_seconds: float | None = None,
        occupancy: object | None = None,
    ) -> np.ndarray:
        with self._lock:
            height, width = self.geometry.image_shape_yx
            exposure = float(exposure_seconds)
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError("exposure_seconds must be positive and finite")
            probe = exposure if probe_seconds is None else float(probe_seconds)
            if not np.isfinite(probe) or probe < 0 or probe > exposure:
                raise ValueError("probe_seconds must be finite and between zero and exposure")
            floor_e = (self.background_rate + self.dark_current_e_per_s) * exposure
            expected_electrons = np.full((height, width), floor_e, dtype=float)
            if occupancy is None:
                shot_occupancy = np.array(self._occupancy, copy=True)
            else:
                shot_occupancy = np.asarray(occupancy, dtype=bool).reshape(-1)
                if shot_occupancy.size != len(self.geometry.site_centers_xy):
                    raise ValueError("occupancy size differs from simulation site map")
            base_area = self.atom_sigma_px**2
            for occupied, gain, sigma_xy, spot in zip(
                shot_occupancy,
                self._detector_efficiency,
                self._site_psf_sigma_xy,
                self._site_psf_spots,
                strict=True,
            ):
                if occupied:
                    sigma_x, sigma_y = (float(value) for value in sigma_xy)
                    expected_electrons += (
                        self.atom_rate
                        * probe
                        * float(gain)
                        * base_area
                        / (sigma_x * sigma_y)
                        * spot
                    )
            electrons = self.rng.poisson(np.clip(expected_electrons, 0.0, None))
            counts = electrons / self.conversion_e_per_count + self.offset_counts
            counts += self.rng.normal(
                0.0,
                self.read_noise_e / self.conversion_e_per_count,
                counts.shape,
            )
            return np.clip(counts, 0, np.iinfo(np.uint16).max).astype("<u2")

    def _mot_axes(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """Cached read-only pixel coordinate vectors for one MOT frame shape."""

        key = (height, width)
        axes = self._mot_axis_cache.get(key)
        if axes is None:
            axes = (
                _readonly(np.arange(width, dtype=float)),
                _readonly(np.arange(height, dtype=float)),
            )
            self._mot_axis_cache[key] = axes
        return axes

    def render_mot_frame(
        self,
        ordinal: int,
        *,
        exposure_seconds: float = 0.1,
        occupancy: object | None = None,
        frame_shape_yx: tuple[int, int] = DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX,
    ) -> np.ndarray:
        """Render one broad MOT fluorescence spot with the shared loading state.

        The spot is an axis-aligned separable Gaussian, so its expected-photon
        plane is the outer product of two 1-D profiles, and Poisson samples are
        drawn only inside the +/-8 sigma window: outside it the rate is below
        ``exp(-32)`` of the peak, an expected total well under one count over
        the whole frame.  The read-noise plane is generated in float32.  The
        real MOT monitor is a Basler Mono8 sensor, so the frame is uint8.
        """

        with self._lock:
            height, width = (int(value) for value in frame_shape_yx)
            if height <= 0 or width <= 0:
                raise ValueError("MOT frame shape must be positive")
            exposure = float(exposure_seconds)
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError("exposure_seconds must be positive and finite")
            if occupancy is None:
                loading = self._mot_population
            else:
                loading = float(np.mean(np.asarray(occupancy, dtype=bool)))
            # The NET field: what the coils add minus the ambient they exist
            # to cancel.  Position and brightness both follow it -- at the
            # optimum the spot is centred AND brightest, which is what a
            # compensated MOT looks like on the monitor.
            scale = 1.0 / 512.0
            optimum = self._mot_field_optimum
            field_x = (self._dac_values["da_bias_x"] - optimum["da_bias_x"]) * scale
            field_y = (self._dac_values["da_bias_y"] - optimum["da_bias_y"]) * scale
            field_z = (self._dac_values["da_bias_z"] - optimum["da_bias_z"]) * scale
            sigma_x = 40.0 / 2.354820045
            sigma_y = 20.0 / 2.354820045
            # A bias field moves the quadrupole zero only a LITTLE: the spot
            # walks within its own size (one sigma per unit net field, so at
            # the full DAC range it stays inside its own FWHM), and the main
            # effect of an uncompensated field is fewer atoms -- the exp() on
            # the peak below, not the position.
            center_x = 0.5 * (width - 1) + sigma_x * field_x
            center_y = 0.5 * (height - 1) + sigma_y * field_y
            x_axis, y_axis = self._mot_axes(height, width)
            counts = self.rng.standard_normal((height, width), dtype=np.float32)
            counts *= 1.5
            counts += 7.0
            field_distance_sq = field_x * field_x + field_y * field_y + field_z * field_z
            # -6, not -1.5: atom number falls fast with net field, which is the
            # physics AND what makes a coarse scan able to tell neighbouring
            # grid points apart above the frame's read noise.
            peak = 93.0 * loading * np.exp(-6.0 * field_distance_sq) * (exposure / 0.1)
            if peak > 0.0:
                x_low = max(0, int(np.floor(center_x - 8.0 * sigma_x)))
                x_high = min(width, int(np.ceil(center_x + 8.0 * sigma_x)) + 1)
                y_low = max(0, int(np.floor(center_y - 8.0 * sigma_y)))
                y_high = min(height, int(np.ceil(center_y + 8.0 * sigma_y)) + 1)
                if x_low < x_high and y_low < y_high:
                    profile_x = np.exp(
                        -0.5 * ((x_axis[x_low:x_high] - center_x) / sigma_x) ** 2
                    )
                    profile_y = np.exp(
                        -0.5 * ((y_axis[y_low:y_high] - center_y) / sigma_y) ** 2
                    )
                    window = counts[y_low:y_high, x_low:x_high]
                    window += self.rng.poisson(
                        peak * np.outer(profile_y, profile_x)
                    ).astype(np.float32)
            np.clip(counts, 0.0, float(np.iinfo(np.uint8).max), out=counts)
            return counts.astype(np.uint8)

    def fire(
        self,
        program: CompiledProgram,
        *,
        table: object | None = None,
        camera_channel: str = "emCCD",
    ) -> None:
        """Play one applied board point through the shared physical world."""

        if not isinstance(program, CompiledProgram):
            raise TypeError("program must be CompiledProgram")
        row = None if table is None else np.asarray(table, dtype=np.int64).reshape(1, -1)
        clock = float(program.clock_hz)
        cooling = trigger_windows(program, "cooling", row)
        probe = trigger_windows(program, "probe", row)
        trap = trigger_windows(program, "trap", row)
        camera_windows = trigger_windows(program, str(camera_channel), row)
        duration_ticks = run_duration_seconds(program, row) * clock

        with self._lock:
            ordinal = self._fire_count
            self._fire_count += 1
            # Atoms come from the MOT, so a cycle loads them while its cooling
            # light is on -- and it loads them when that light COMES ON.  The
            # moment used to be the light going OFF, which put the load after
            # any picture taken while the cooling was still on: that frame then
            # showed the PREVIOUS cycle's atoms, and nothing anywhere said so.
            load_tick = float(cooling[0][0]) if cooling else None
            loaded = False
            cursor = 0.0

            for start_tick, end_tick in camera_windows:
                start = float(start_tick)
                if load_tick is not None and not loaded and load_tick <= start:
                    self._lose_atoms(_low_time(cursor, load_tick, trap, clock))
                    self._load_shot()
                    cursor = load_tick
                    loaded = True
                self._lose_atoms(_low_time(cursor, start, trap, clock))
                cursor = start
                shot_occupancy = np.array(self._occupancy, copy=True)
                for device, registered_renderer in tuple(self._cameras):
                    if not device.capture_state():
                        continue
                    point = device.working_point()
                    configured = float(point.exposure_seconds)
                    # How long the sensor integrates is the CAMERA's fact, and
                    # which fact depends on how it is triggered.  Externally
                    # triggered means edge triggered here -- the qCMOS driver
                    # sets and asserts rising-edge mode -- and an accepted
                    # edge integrates the camera's own exposure however long
                    # the pulse holds its window open.  Taking the shorter of
                    # the two was level-trigger physics, so the simulated
                    # bench and the real one disagreed by construction: a
                    # readout frame gated 5 ms long really collects 5 ms of
                    # probe light on top of the full exposure's background,
                    # and only the simulation pretended otherwise.
                    free_running = str(point.acquisition_mode).upper().endswith(
                        "FREE_RUNNING"
                    )
                    exposure = (
                        min(configured, (end_tick - start_tick) / clock)
                        if free_running
                        else configured
                    )
                    integration_end = start + exposure * clock
                    probe_seconds = _overlap_ticks(start, integration_end, probe) / clock
                    if registered_renderer is None:
                        frame = self.render_frame(
                            ordinal,
                            exposure_seconds=exposure,
                            probe_seconds=probe_seconds,
                            occupancy=shot_occupancy,
                        )
                    else:
                        frame = registered_renderer(
                            ordinal,
                            exposure_seconds=exposure,
                            occupancy=shot_occupancy,
                        )
                    device.trigger(1, frame=frame)

            if load_tick is not None and not loaded:
                self._lose_atoms(_low_time(cursor, load_tick, trap, clock))
                self._load_shot()
                cursor = load_tick
            self._lose_atoms(_low_time(cursor, duration_ticks, trap, clock))
            self._dac_values.update(_final_dac_values(program, row))


def _overlap_ticks(
    start: float,
    end: float,
    windows: tuple[tuple[int, int], ...],
) -> float:
    return sum(max(0.0, min(end, stop) - max(start, begin)) for begin, stop in windows)


def _low_time(
    start: float,
    end: float,
    high_windows: tuple[tuple[int, int], ...],
    clock_hz: float,
) -> float:
    return max(0.0, end - start - _overlap_ticks(start, end, high_windows)) / clock_hz


def _final_dac_values(
    program: CompiledProgram,
    table: np.ndarray | None,
) -> dict[str, int]:
    point = () if table is None else tuple(int(value) for value in table.reshape(-1))
    values = {name: 0 for name in program.bus_names}
    for segment in program.bus_segments:
        selector = segment.stop_value_select or segment.value_select
        code = point[selector - 1] if selector else segment.stop_value
        values[segment.bus_name] = int(code) - int(program.bus_safe_values[segment.bus_index])
    return values


__all__ = [
    "DEFAULT_SIMULATION_GRID_SHAPE_YX",
    "DEFAULT_SIMULATION_IMAGE_SHAPE_YX",
    "DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX",
    "DEFAULT_SIMULATION_SITE_SPACING_PIXELS",
    "DEFAULT_SIMULATION_SLM_SHAPE_YX",
    "SimulationGeometry",
    "SimulationWorld",
    "SimulationWorldConfig",
]
