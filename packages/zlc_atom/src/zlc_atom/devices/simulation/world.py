"""Installation-owned atom/imaging world shared by simulation devices."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np

from zlc_pulse.compile import CompiledProgram, evaluate_affine_tick
from zlc_pulse.schedule import run_duration_seconds, trigger_windows


DEFAULT_SIMULATION_GRID_SHAPE_YX = (5, 7)
DEFAULT_SIMULATION_IMAGE_SHAPE_YX = (96, 128)
DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX = (1200, 1920)
#: Zero authored bias is the one compensated MOT operating point.
DEFAULT_MOT_FIELD_OPTIMUM_DAC = (0, 0, 0)
DEFAULT_SIMULATION_SITE_SPACING_PIXELS = 9.0
DEFAULT_SIMULATION_SLM_SHAPE_YX = (128, 128)
# Every physical trap is one dominant local maximum of the propagated field.
# There is no second nominal-site roster beside those peaks.
_TRAP_PEAK_FRACTION = 0.20
_TRAP_PEAK_NEIGHBORHOOD = 7

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
DEFAULT_TRAP_DEPTH_K = 5.2e-4
DEFAULT_TRAP_WAIST_M = 1.0e-6
_WORLD_PROFILE_FORMAT = "zlc.simulation.world_profile"
_SIMULATION_FLOAT_FIELDS = (
    "offset_counts",
    "conversion_e_per_count",
    "read_noise_e",
    "background_rate",
    "atom_rate",
    "atom_sigma_px",
    "loading_probability",
    "probe_saturation",
    "probe_detuning_linewidths",
    "trap_light_shift_linewidths",
    "fluorescence_lifetime_seconds",
    "atom_temperature_k",
    "trap_depth_k",
    "trap_waist_m",
    "dark_current_e_per_s",
)


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
def _initial_slm_command(
    shape_yx: tuple[int, int],
    grid_shape_yx: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the configured startup target once; it is not a second site roster."""

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
    """Resolved immutable physics configuration for one shared world."""

    geometry: SimulationGeometry = field(default_factory=SimulationGeometry)
    seed: int = 0
    mot_field_optimum_dac: tuple[int, int, int] = DEFAULT_MOT_FIELD_OPTIMUM_DAC
    offset_counts: float = 200.0
    conversion_e_per_count: float = 0.107
    read_noise_e: float = 0.43
    background_rate: float = 300.0
    atom_rate: float = 145_000.0
    atom_sigma_px: float = 0.7
    loading_probability: float = 0.5
    probe_saturation: float = 0.1
    probe_detuning_linewidths: float = -1.9
    trap_light_shift_linewidths: float = 1.6
    fluorescence_lifetime_seconds: float = 0.05
    atom_temperature_k: float = DEFAULT_ATOM_TEMPERATURE_K
    trap_depth_k: float = DEFAULT_TRAP_DEPTH_K
    trap_waist_m: float = DEFAULT_TRAP_WAIST_M
    dark_current_e_per_s: float = 0.0

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
        for name in _SIMULATION_FLOAT_FIELDS:
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be numeric") from error
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in (
            "conversion_e_per_count",
            "atom_sigma_px",
            "probe_saturation",
            "fluorescence_lifetime_seconds",
            "atom_temperature_k",
            "trap_depth_k",
            "trap_waist_m",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "offset_counts",
            "read_noise_e",
            "background_rate",
            "atom_rate",
            "dark_current_e_per_s",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.loading_probability <= 0.5:
            raise ValueError("loading_probability must be in [0.0, 0.5] (maximum 0.5)")
        if self.trap_light_shift_linewidths < 0.0:
            raise ValueError("trap_light_shift_linewidths must be non-negative")

    @classmethod
    def from_profile(
        cls,
        path: str | Path,
        *,
        geometry: SimulationGeometry,
        seed: int,
    ) -> SimulationWorldConfig:
        """Resolve one strict workspace profile before constructing the world."""

        resolved = Path(path).expanduser().resolve()

        def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate simulation profile field {key!r}")
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise ValueError(f"non-finite simulation profile value {value!r}")

        try:
            payload = json.loads(
                resolved.read_text(encoding="utf-8"),
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"simulation world profile {resolved} is not strict JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("simulation world profile must be a JSON object")
        allowed = {
            "format",
            "mot_field_optimum_dac",
            *_SIMULATION_FLOAT_FIELDS,
        }
        if set(payload) - allowed:
            raise ValueError("simulation world profile has unknown fields")
        if payload.get("format") != _WORLD_PROFILE_FORMAT:
            raise ValueError("simulation world profile has an unsupported format")
        changes: dict[str, object] = {}
        for name in _SIMULATION_FLOAT_FIELDS:
            if name not in payload:
                continue
            value = payload[name]
            if type(value) not in (int, float):
                raise TypeError(f"simulation world profile {name} must be numeric")
            changes[name] = value
        if "mot_field_optimum_dac" in payload:
            optimum = payload["mot_field_optimum_dac"]
            if (
                not isinstance(optimum, list)
                or len(optimum) != 3
                or any(type(value) is not int for value in optimum)
            ):
                raise TypeError(
                    "simulation world profile mot_field_optimum_dac must be three integers"
                )
            changes["mot_field_optimum_dac"] = tuple(optimum)
        return cls(
            geometry=geometry,
            seed=seed,
            **changes,
        )


class SimulationWorld:
    """Explicit state and trigger routing for all virtual devices."""

    def __init__(
        self,
        config: SimulationWorldConfig | None = None,
    ) -> None:
        resolved = SimulationWorldConfig() if config is None else config
        if not isinstance(resolved, SimulationWorldConfig):
            raise TypeError("config must be SimulationWorldConfig or None")
        self._config = resolved
        seed = resolved.seed
        optimum = resolved.mot_field_optimum_dac
        self._mot_field_optimum = dict(
            zip(("da_bias_x", "da_bias_y", "da_bias_z"), optimum)
        )
        _static_seed, atom_seed, qcmos_seed, mot_seed = np.random.SeedSequence(
            seed
        ).spawn(4)
        self._atom_rng = np.random.default_rng(atom_seed)
        self._qcmos_rng = np.random.default_rng(qcmos_seed)
        self._mot_rng = np.random.default_rng(mot_seed)
        self._lock = threading.RLock()
        self._cameras: list[tuple[Any, Callable[..., np.ndarray] | None]] = []
        self._fire_count = 0
        site_count = len(self.geometry.site_centers_xy)
        self._slm_shape_yx = DEFAULT_SIMULATION_SLM_SHAPE_YX
        self._reference_slm_indices_yx, initial_phase = _initial_slm_command(
            self._slm_shape_yx,
            self.geometry.grid_shape_yx,
        )
        if len(self._reference_slm_indices_yx) != site_count:
            raise RuntimeError("initial SLM target differs from simulation site geometry")
        self._commanded_phase = _immutable(initial_phase, "<f4")
        self._slm_phase_revision = 0
        self._propagated_revision = -1
        self._propagation_count = 0
        self._trap_plane_intensity: np.ndarray | None = None
        self._trap_indices_yx = _immutable(np.empty((0, 2), dtype=np.intp), np.intp)
        self._trap_intensities = _immutable(np.empty(0), "<f4")
        self._loading_intensity_scale: float | None = None
        self._slm_pupil_amplitude, self._hidden_slm_aberration = (
            self._slm_plant(seed)
        )
        imaging_y, imaging_x = np.ogrid[
            -1.0 : 1.0 : self._slm_shape_yx[0] * 1j,
            -1.0 : 1.0 : self._slm_shape_yx[1] * 1j,
        ]
        imaging_radius_squared = imaging_x * imaging_x + imaging_y * imaging_y
        imaging_amplitude = np.exp(-2.5 * imaging_radius_squared)
        imaging_coma_x = (3.0 * imaging_radius_squared - 2.0) * imaging_x
        imaging_astigmatism = imaging_x * imaging_x - imaging_y * imaging_y
        imaging_aberration = 0.30 * imaging_coma_x + 0.12 * imaging_astigmatism
        imaging_field = imaging_amplitude * np.exp(1j * imaging_aberration)
        imaging_psf = np.abs(
            np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(imaging_field), norm="ortho")
            )
        ) ** 2
        imaging_psf /= float(np.max(imaging_psf))
        self._camera_psf = _readonly(imaging_psf)
        height, width = self.geometry.image_shape_yx
        self._trap_centers_xy = _readonly(np.empty((0, 2), dtype=float))
        self._trap_psf_spots = _readonly(
            np.empty((0, height, width), dtype=float)
        )
        self._occupancy = np.zeros(0, dtype=bool)
        self._mot_population = 1.0
        self._dac_values = {"da_bias_x": 0, "da_bias_y": 0, "da_bias_z": 0}
        #: Read-only pixel coordinate vectors per MOT frame shape.  A frame
        #: shape is a configuration fact, so this holds one or two entries.
        self._mot_axis_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        # Establish the physical intensity scale once from the startup command.
        # Later commands are compared against that fixed bench scale, rather
        # than renormalized per phase (which would erase diffraction loss).
        self._ensure_slm_propagation()

    @property
    def config(self) -> SimulationWorldConfig:
        return self._config

    @property
    def geometry(self) -> SimulationGeometry:
        return self._config.geometry

    @property
    def offset_counts(self) -> float:
        return self._config.offset_counts

    @property
    def conversion_e_per_count(self) -> float:
        return self._config.conversion_e_per_count

    @property
    def read_noise_e(self) -> float:
        return self._config.read_noise_e

    @property
    def background_rate(self) -> float:
        return self._config.background_rate

    @property
    def atom_rate(self) -> float:
        return self._config.atom_rate

    @property
    def atom_sigma_px(self) -> float:
        return self._config.atom_sigma_px

    @property
    def loading_probability(self) -> float:
        return self._config.loading_probability

    @property
    def probe_saturation(self) -> float:
        return self._config.probe_saturation

    @property
    def probe_detuning_linewidths(self) -> float:
        return self._config.probe_detuning_linewidths

    @property
    def trap_light_shift_linewidths(self) -> float:
        return self._config.trap_light_shift_linewidths

    @property
    def fluorescence_lifetime_seconds(self) -> float:
        return self._config.fluorescence_lifetime_seconds

    @property
    def atom_temperature_k(self) -> float:
        return self._config.atom_temperature_k

    @property
    def trap_depth_k(self) -> float:
        return self._config.trap_depth_k

    @property
    def trap_waist_m(self) -> float:
        return self._config.trap_waist_m

    @property
    def dark_current_e_per_s(self) -> float:
        return self._config.dark_current_e_per_s

    def _slm_plant(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Materialize one common pupil illumination and low-order aberration."""

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
            -0.45 * ((xx - decenter_x) ** 2 + 1.12 * (yy - decenter_y) ** 2)
        )
        illumination *= np.clip(1.0 - 0.20 * radius_squared, 0.0, None)
        amplitude = np.where(pupil, illumination, 0.0)

        coefficients = np.asarray((2.34, -1.56, 2.135, 0.955), dtype=float)
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

    def apply_slm_phase(self, radians: object) -> np.ndarray:
        """Atomically accept one explicit command and invalidate propagation."""

        from zlc_atom.devices.slm import canonical_phase

        commanded = canonical_phase(radians, self._slm_shape_yx)
        with self._lock:
            self._commanded_phase = commanded
            self._slm_phase_revision += 1
            self._propagated_revision = -1
            return self._commanded_phase

    def _camera_centers(self, indices_yx: np.ndarray) -> np.ndarray:
        """Project every physical Fourier peak through one camera geometry."""

        indices = np.asarray(indices_yx, dtype=float).reshape(-1, 2)
        if not len(indices):
            return np.empty((0, 2), dtype=float)
        anchor_rows, anchor_columns = self._reference_slm_indices_yx.T
        camera = np.asarray(self.geometry.site_centers_xy, dtype=float)
        scale_x = float(np.ptp(camera[:, 0]) / np.ptp(anchor_columns))
        scale_y = float(np.ptp(camera[:, 1]) / np.ptp(anchor_rows))
        return np.column_stack(
            (
                np.mean(camera[:, 0])
                + (indices[:, 1] - np.mean(anchor_columns)) * scale_x,
                np.mean(camera[:, 1])
                + (indices[:, 0] - np.mean(anchor_rows)) * scale_y,
            )
        )

    def _camera_spots(self, centers_xy: np.ndarray) -> np.ndarray:
        """Translate one shared aberrated optical PSF to every physical site."""

        from scipy.ndimage import map_coordinates

        height, width = self.geometry.image_shape_yx
        centers = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
        if not len(centers):
            return np.empty((0, height, width), dtype=float)
        yy, xx = np.mgrid[:height, :width]
        psf = np.asarray(self._camera_psf, dtype=float)
        origin_y, origin_x = np.unravel_index(int(np.argmax(psf)), psf.shape)
        scale = float(self.atom_sigma_px) / 0.7
        return np.asarray(
            [
                map_coordinates(
                    psf,
                    (
                        origin_y + (yy - float(y)) / scale,
                        origin_x + (xx - float(x)) / scale,
                    ),
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                for x, y in centers
            ],
            dtype=float,
        )

    @staticmethod
    def _resolved_traps(
        intensity: np.ndarray,
        local_peak_plane: np.ndarray,
        cutoff: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the one physical trap roster from dominant local peaks."""

        peaks = np.argwhere(
            (intensity == local_peak_plane) & (intensity >= cutoff)
        )
        if not len(peaks):
            return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=float)
        rows, columns = peaks.T
        return peaks, np.asarray(intensity[rows, columns], dtype=float)

    def _ensure_slm_propagation(self) -> None:
        if self._propagated_revision == self._slm_phase_revision:
            return
        from scipy import fft
        from scipy.ndimage import maximum_filter

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
        local_peak_plane = maximum_filter(
            intensity,
            size=_TRAP_PEAK_NEIGHBORHOOD,
            mode="constant",
            cval=-np.inf,
        )
        peak_reference = max(float(np.max(intensity)), self._loading_intensity_scale or 0.0)
        trap_indices, trap_intensities = self._resolved_traps(
            intensity,
            local_peak_plane,
            _TRAP_PEAK_FRACTION * peak_reference,
        )
        if self._loading_intensity_scale is None:
            if not len(trap_intensities):
                raise RuntimeError("reference SLM command produced no trap intensity")
            self._loading_intensity_scale = float(np.mean(trap_intensities))
        scale = self._loading_intensity_scale
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("reference SLM command produced no trap intensity")
        occupancy = np.zeros(len(trap_indices), dtype=bool)
        if len(self._trap_indices_yx) and len(trap_indices):
            squared_distance = np.sum(
                (
                    trap_indices[:, np.newaxis, :]
                    - self._trap_indices_yx[np.newaxis, :, :]
                )
                ** 2,
                axis=2,
            )
            pairs = np.argwhere(squared_distance <= 9)
            used_new = np.zeros(len(trap_indices), dtype=bool)
            used_old = np.zeros(len(self._trap_indices_yx), dtype=bool)
            if len(pairs):
                distances = squared_distance[pairs[:, 0], pairs[:, 1]]
                for new_index, old_index in pairs[np.argsort(distances, kind="stable")]:
                    if used_new[new_index] or used_old[old_index]:
                        continue
                    occupancy[new_index] = self._occupancy[old_index]
                    used_new[new_index] = True
                    used_old[old_index] = True
        trap_centers = self._camera_centers(trap_indices)
        self._trap_plane_intensity = _immutable(intensity, "<f4")
        self._trap_indices_yx = _immutable(trap_indices, np.intp)
        self._trap_intensities = _immutable(trap_intensities, "<f4")
        self._trap_centers_xy = _readonly(trap_centers)
        self._trap_psf_spots = _readonly(self._camera_spots(trap_centers))
        self._occupancy = occupancy
        self._propagated_revision = self._slm_phase_revision
        self._propagation_count += 1

    def _loading_probabilities(self, intensities: np.ndarray) -> np.ndarray:
        base = float(self.loading_probability)
        scale = self._loading_intensity_scale
        if scale is None or not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("reference SLM command produced no trap intensity")
        relative_depth = np.clip(np.asarray(intensities) / scale, 0.0, None)
        depth_k = self.trap_depth_k * relative_depth
        cooling_temperature = self.atom_temperature_k
        minimum_loading_depth = 25.0 * cooling_temperature
        excess_depth = np.maximum(depth_k - minimum_loading_depth, 0.0)
        capture = -np.expm1(-excess_depth / cooling_temperature)
        return base * capture

    def _fluorescence_scales(self, intensities: np.ndarray) -> np.ndarray:
        """Occupied-atom scattering from one shared Stark-shifted probe law."""

        scale = self._loading_intensity_scale
        if scale is None or not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("reference SLM command produced no trap intensity")
        relative_depth = np.clip(np.asarray(intensities) / scale, 0.0, None)
        saturation = float(self.probe_saturation)
        probe_detuning = float(self.probe_detuning_linewidths)
        light_shift = float(self.trap_light_shift_linewidths)
        # The probe is fixed at the apparatus working point selected for the
        # ideal uniform reference array.  It never follows a candidate phase or
        # the current array mean, so absolute depth remains observable.
        # On the experiment's red-detuned probe working point, trap light can
        # only move the atomic transition farther to the red.  The configured
        # light shift is therefore a positive magnitude, not a signed knob.
        detuning = probe_detuning - light_shift * relative_depth
        scattering = saturation / (
            1.0 + saturation + np.square(2.0 * detuning)
        )
        reference_detuning = probe_detuning - light_shift
        reference_scattering = saturation / (
            1.0 + saturation + (2.0 * reference_detuning) ** 2
        )
        return np.where(
            relative_depth > 0.0,
            scattering / reference_scattering,
            0.0,
        )

    def _site_loading_probabilities(self) -> np.ndarray:
        self._ensure_slm_propagation()
        return self._loading_probabilities(self._trap_intensities)

    def register_camera(
        self,
        camera: Any,
        renderer: Callable[..., np.ndarray] | None = None,
    ) -> None:
        with self._lock:
            if not any(existing is camera for existing, _renderer in self._cameras):
                self._cameras.append((camera, renderer))

    def _load_shot(self) -> np.ndarray:
        self._ensure_slm_propagation()
        shot = (
            self._atom_rng.random(len(self._trap_intensities))
            < self._site_loading_probabilities()
        )
        self._occupancy = np.asarray(shot, dtype=bool)
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

    def _site_survival_probabilities(self, trap_off_seconds: float) -> np.ndarray:
        self._ensure_slm_propagation()
        return self._survival_probabilities(self._trap_intensities, trap_off_seconds)

    def _survival_probabilities(
        self,
        intensities: np.ndarray,
        trap_off_seconds: float,
    ) -> np.ndarray:
        scale = self._loading_intensity_scale
        if scale is None or not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("reference SLM command produced no trap intensity")
        relative_depth = np.clip(np.asarray(intensities) / scale, 0.0, None)
        survival = np.asarray(
            [
                self._release_survival(trap_off_seconds, float(value))
                for value in relative_depth
            ],
            dtype=float,
        )
        return np.where(np.asarray(intensities) > 0.0, survival, 0.0)

    def _lose_atoms(self, trap_off_seconds: float) -> None:
        off_time = float(trap_off_seconds)
        if not np.isfinite(off_time) or off_time < 0:
            raise ValueError("trap_off_seconds must be finite and non-negative")
        if off_time:
            survival = self._site_survival_probabilities(off_time)
            occupied = np.flatnonzero(self._occupancy)
            self._occupancy[occupied] &= (
                self._atom_rng.random(len(occupied)) < survival[occupied]
            )

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
            # A trigger observes the latest command, never a stale lazy cache.
            self._ensure_slm_propagation()
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
                if shot_occupancy.size != len(self._trap_intensities):
                    raise ValueError("occupancy size differs from current trap roster")
            fluorescence_lifetime = float(self.fluorescence_lifetime_seconds)
            fluorescence_seconds = -fluorescence_lifetime * math.expm1(
                -probe / fluorescence_lifetime
            )
            fluorescence_scales = self._fluorescence_scales(
                self._trap_intensities
            )
            for occupied, brightness, spot in zip(
                shot_occupancy,
                fluorescence_scales,
                self._trap_psf_spots,
                strict=True,
            ):
                if occupied:
                    expected_electrons += (
                        self.atom_rate
                        * fluorescence_seconds
                        * float(brightness)
                        * spot
                    )
            electrons = self._qcmos_rng.poisson(
                np.clip(expected_electrons, 0.0, None)
            )
            counts = electrons / self.conversion_e_per_count + self.offset_counts
            counts += self._qcmos_rng.normal(
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
            counts = self._mot_rng.standard_normal(
                (height, width), dtype=np.float32
            )
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
                    window += self._mot_rng.poisson(
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
        base_duration_ticks = int(
            round(run_duration_seconds(program, row) * clock)
        )
        duration_ticks = max(
            (
                base_duration_ticks,
                *(
                    end
                    for windows in (cooling, probe, trap, camera_windows)
                    for _start, end in windows
                ),
            )
        )

        trap_off: list[tuple[int, int]] = []
        trap_cursor = 0
        for start_tick, end_tick in trap:
            if start_tick > trap_cursor:
                trap_off.append((trap_cursor, start_tick))
            trap_cursor = end_tick
        if trap_cursor < duration_ticks:
            trap_off.append((trap_cursor, duration_ticks))

        cooling_rises = {start for start, _end in cooling}
        release_ends = {end: start for start, end in trap_off}
        cameras_by_tick: dict[int, list[tuple[int, int]]] = {}
        for start_tick, end_tick in camera_windows:
            cameras_by_tick.setdefault(start_tick, []).append(
                (start_tick, end_tick)
            )
        event_ticks = sorted(
            cooling_rises | set(release_ends) | set(cameras_by_tick)
        )

        with self._lock:
            self._ensure_slm_propagation()
            ordinal = self._fire_count
            self._fire_count += 1
            for tick in event_ticks:
                self._dac_values.update(
                    _dac_values_at_tick(program, row, tick)
                )
                release_start = release_ends.get(tick)
                if release_start is not None:
                    self._lose_atoms((tick - release_start) / clock)

                if tick in cooling_rises:
                    self._mot_population = 1.0
                    if any(start <= tick < end for start, end in trap):
                        self._load_shot()

                for start_tick, end_tick in cameras_by_tick.get(tick, ()):
                    start = float(start_tick)
                    shot_occupancy = np.array(self._occupancy, copy=True)
                    for device, registered_renderer in tuple(self._cameras):
                        if not device.capture_state():
                            continue
                        point = device.working_point()
                        configured = float(point.exposure_seconds)
                        # An external rising edge starts the camera's authored
                        # integration; a free-running camera remains bounded by
                        # the high window that was actually played.
                        free_running = str(point.acquisition_mode).upper().endswith(
                            "FREE_RUNNING"
                        )
                        exposure = (
                            min(configured, (end_tick - start_tick) / clock)
                            if free_running
                            else configured
                        )
                        integration_end = start + exposure * clock
                        probe_seconds = (
                            _overlap_ticks(start, integration_end, probe) / clock
                        )
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
                        device.trigger(
                            1,
                            frame=frame,
                        )

            self._dac_values.update(_final_dac_values(program, row))


def _overlap_ticks(
    start: float,
    end: float,
    windows: tuple[tuple[int, int], ...],
) -> float:
    return sum(max(0.0, min(end, stop) - max(start, begin)) for begin, stop in windows)


def _dac_values_at_tick(
    program: CompiledProgram,
    table: np.ndarray | None,
    tick: int,
) -> dict[str, int]:
    """Project the compiled DAC buses at one physical playback tick."""

    point = () if table is None else tuple(int(value) for value in table.reshape(-1))
    delays = {
        int(item.bus_index): int(item.delay_ticks)
        for item in program.bus_delays
    }
    values: dict[str, int] = {}
    for bus_index, bus_name in enumerate(program.bus_names):
        safe = int(program.bus_safe_values[bus_index])
        phase = int(tick) - delays.get(bus_index, 0)
        code = safe
        if phase >= 0:
            segments = [
                segment
                for segment in program.bus_segments
                if int(segment.bus_index) == bus_index
            ]

            def effective(base: int, coefficients: tuple[int, ...]) -> int:
                return evaluate_affine_tick(
                    int(base),
                    coefficients,
                    point,
                    program.scan_coeff_frac_bits,
                )

            def endpoint(selector: int, literal: int) -> int:
                return int(point[selector - 1]) if selector else int(literal)

            segments.sort(
                key=lambda segment: effective(
                    segment.start_tick,
                    segment.start_tick_coeffs,
                )
            )
            chosen = None
            ramp_start = safe
            for segment in segments:
                start = effective(
                    segment.start_tick,
                    segment.start_tick_coeffs,
                )
                if start < phase or start == 0:
                    if chosen is not None:
                        selector = int(chosen.stop_value_select)
                        ramp_start = endpoint(selector, chosen.stop_value)
                    chosen = segment
                else:
                    break
            if chosen is not None:
                start = effective(
                    chosen.start_tick,
                    chosen.start_tick_coeffs,
                )
                stop = effective(
                    chosen.stop_tick,
                    chosen.stop_tick_coeffs,
                )
                selector = int(chosen.stop_value_select)
                target = endpoint(selector, chosen.stop_value)
                start_selector = int(chosen.value_select)
                if start_selector:
                    ramp_start = endpoint(start_selector, chosen.start_value)
                if chosen.mode == "ramp" and stop > start:
                    if phase <= start:
                        code = ramp_start
                    elif phase > stop:
                        code = target
                    else:
                        span = stop - start
                        distance = abs(target - ramp_start)
                        elapsed = (phase - 1) - start
                        moved = elapsed * distance // span
                        code = (
                            ramp_start + moved
                            if target >= ramp_start
                            else ramp_start - moved
                        )
                else:
                    code = target
        values[bus_name] = code - safe
    return values


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
