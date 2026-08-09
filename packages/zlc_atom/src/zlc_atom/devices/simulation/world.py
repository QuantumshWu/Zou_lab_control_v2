"""Installation-owned atom/imaging world shared by simulation devices."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable

import numpy as np


DEFAULT_SIMULATION_GRID_SHAPE_YX = (5, 7)
DEFAULT_SIMULATION_IMAGE_SHAPE_YX = (96, 128)
DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX = (1200, 1920)
DEFAULT_SIMULATION_SITE_SPACING_PIXELS = 9.0


def _readonly(values: object) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=float)
    array.setflags(write=False)
    return array


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

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, SimulationGeometry):
            raise TypeError("geometry must be SimulationGeometry")
        object.__setattr__(self, "seed", int(self.seed))


class SimulationWorld:
    """Explicit state and trigger routing for all virtual devices."""

    def __init__(self, geometry: SimulationGeometry | None = None, *, seed: int = 0) -> None:
        self.geometry = SimulationGeometry() if geometry is None else geometry
        self.rng = np.random.default_rng(int(seed))
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
        self.trap_off_lifetime_s = 2.0e-3
        self.dark_current_e_per_s = 0.0
        site_count = len(self.geometry.site_centers_xy)
        efficiency_log = self.rng.normal(0.0, 1.0, site_count)
        efficiency_log -= float(np.min(efficiency_log))
        span = float(np.ptp(efficiency_log))
        if span:
            efficiency_log *= np.log(2.0) / span
        efficiency_log -= 0.5 * np.log(2.0)
        self._site_efficiency = _readonly(np.exp(efficiency_log))
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
        self._occupancy = self.rng.random(site_count) < self.loading_probability
        self._forced_occupancy: np.ndarray | None = None

    @property
    def site_efficiency(self) -> np.ndarray:
        return np.array(self._site_efficiency, copy=True)

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

    def _load_shot(self, trap_off_seconds: float) -> np.ndarray:
        if self._forced_occupancy is None:
            shot = self.rng.random(len(self.geometry.site_centers_xy)) < self.loading_probability
        else:
            shot = np.array(self._forced_occupancy, copy=True)
        off_time = float(trap_off_seconds)
        if not np.isfinite(off_time) or off_time < 0:
            raise ValueError("trap_off_seconds must be finite and non-negative")
        if off_time:
            survival = np.exp(-off_time / self.trap_off_lifetime_s)
            shot &= self.rng.random(shot.size) < survival
        self._occupancy = shot
        return np.array(shot, copy=True)

    def render_frame(
        self,
        ordinal: int,
        *,
        exposure_seconds: float = 0.005,
        occupancy: object | None = None,
    ) -> np.ndarray:
        with self._lock:
            height, width = self.geometry.image_shape_yx
            exposure = float(exposure_seconds)
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError("exposure_seconds must be positive and finite")
            floor_e = (self.background_rate + self.dark_current_e_per_s) * exposure
            expected_electrons = np.full((height, width), floor_e, dtype=float)
            yy, xx = np.mgrid[:height, :width]
            if occupancy is None:
                shot_occupancy = np.array(self._occupancy, copy=True)
            else:
                shot_occupancy = np.asarray(occupancy, dtype=bool).reshape(-1)
                if shot_occupancy.size != len(self.geometry.site_centers_xy):
                    raise ValueError("occupancy size differs from simulation site map")
            base_area = self.atom_sigma_px**2
            for occupied, (x, y), gain, sigma_xy, angle, skew in zip(
                shot_occupancy,
                self.geometry.site_centers_xy,
                self._site_efficiency,
                self._site_psf_sigma_xy,
                self._site_psf_angle_radians,
                self._site_psf_skew,
                strict=True,
            ):
                if occupied:
                    sigma_x, sigma_y = (float(value) for value in sigma_xy)
                    cosine, sine = np.cos(angle), np.sin(angle)
                    dx = (xx - x) * cosine + (yy - y) * sine
                    dy = -(xx - x) * sine + (yy - y) * cosine
                    core = np.exp(
                        -0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2)
                    )
                    spot = np.clip(core * (1.0 + float(skew) * dx / sigma_x), 0.0, None)
                    expected_electrons += (
                        self.atom_rate
                        * exposure
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

    def render_mot_frame(
        self,
        ordinal: int,
        *,
        exposure_seconds: float = 0.05,
        occupancy: object | None = None,
        frame_shape_yx: tuple[int, int] = DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX,
    ) -> np.ndarray:
        """Render one broad MOT fluorescence spot with the shared loading state."""

        with self._lock:
            height, width = (int(value) for value in frame_shape_yx)
            if height <= 0 or width <= 0:
                raise ValueError("MOT frame shape must be positive")
            exposure = float(exposure_seconds)
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError("exposure_seconds must be positive and finite")
            atoms = self._occupancy if occupancy is None else np.asarray(occupancy, dtype=bool)
            loading = float(np.mean(np.asarray(atoms, dtype=bool)))
            center_x = 0.5 * width + 0.015 * width * np.sin(0.31 * int(ordinal))
            center_y = 0.5 * height + 0.015 * height * np.cos(0.23 * int(ordinal))
            sigma_x = 40.0 / 2.354820045
            sigma_y = 20.0 / 2.354820045
            yy, xx = np.mgrid[:height, :width]
            spot = np.exp(
                -0.5
                * (
                    ((xx - center_x) / sigma_x) ** 2
                    + ((yy - center_y) / sigma_y) ** 2
                )
            )
            signal = self.rng.poisson(
                93.0 * loading * (exposure / 0.05) * spot
            )
            counts = 7.0 + signal + self.rng.normal(0.0, 1.5, spot.shape)
            return np.clip(counts, 0, np.iinfo(np.uint16).max).astype("<u2")

    def fire(
        self,
        count: int = 1,
        *,
        frame_exposures: object | None = None,
        trap_off_seconds: float = 0.0,
    ) -> None:
        """Route one shot's camera windows using one shared atom occupancy.

        ``count`` is the number of camera windows in the loaded pulse, not the
        number of independent shots.  The sequencer calls this once per
        ``fire``; all windows rendered during that call therefore share the
        occupancy selected for that shot.  A camera working point owns the
        maximum integration exposure.  An explicitly compiled external gate
        may shorten an individual window, as the calibration long/readout/long
        protocol does, but it cannot extend the configured camera exposure.
        """

        windows = int(count)
        if windows <= 0:
            raise ValueError("camera window count must be positive")
        if frame_exposures is None:
            gates: tuple[float, ...] | None = None
        else:
            gates = tuple(float(value) for value in frame_exposures)  # type: ignore[arg-type]
            if len(gates) != windows or any(
                not np.isfinite(value) or value <= 0 for value in gates
            ):
                raise ValueError(
                    "frame_exposures must contain one positive finite value per camera window"
                )
        with self._lock:
            ordinal = self._fire_count
            shot_occupancy = self._load_shot(trap_off_seconds)
            self._fire_count += 1
            # Devices own their protocol; the world only performs explicit
            # trigger routing and never exposes private backend channels.
            for camera, registered_renderer in tuple(self._cameras):
                if not camera.capture_state():
                    continue
                configured = float(camera.capture_working_point().exposure_seconds)
                effective = (
                    (configured,) * windows
                    if gates is None
                    else tuple(min(configured, gate) for gate in gates)
                )
                renderer = self.render_frame if registered_renderer is None else registered_renderer
                for exposure in effective:
                    frame = renderer(
                        ordinal,
                        exposure_seconds=exposure,
                        occupancy=shot_occupancy,
                    )
                    camera.trigger(1, frame=frame)


__all__ = [
    "DEFAULT_SIMULATION_GRID_SHAPE_YX",
    "DEFAULT_SIMULATION_IMAGE_SHAPE_YX",
    "DEFAULT_SIMULATION_MOT_IMAGE_SHAPE_YX",
    "DEFAULT_SIMULATION_SITE_SPACING_PIXELS",
    "SimulationGeometry",
    "SimulationWorld",
    "SimulationWorldConfig",
]
