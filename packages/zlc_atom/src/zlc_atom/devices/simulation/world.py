"""Installation-owned atom/imaging world shared by simulation devices."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np


DEFAULT_SIMULATION_GRID_SHAPE_YX = (5, 7)
DEFAULT_SIMULATION_IMAGE_SHAPE_YX = (96, 128)
DEFAULT_SIMULATION_SITE_SPACING_PIXELS = 9.0


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
        self._cameras: list[Any] = []
        self._occupancy_override: np.ndarray | None = None
        self._fire_count = 0
        self.offset_counts = 200.0
        self.conversion_e_per_count = 0.107
        self.read_noise_e = 0.43
        self.background_rate = 300.0
        self.atom_rate = 1_100.0
        self.atom_sigma_px = 0.7
        self.dark_current_e_per_s = 0.0

    def register_camera(self, camera: Any) -> None:
        with self._lock:
            if camera not in self._cameras:
                self._cameras.append(camera)

    def set_occupancy(self, occupancy: object) -> None:
        values = np.asarray(occupancy, dtype=bool).reshape(-1)
        if values.size != len(self.geometry.site_centers_xy):
            raise ValueError("occupancy size differs from simulation site map")
        with self._lock:
            self._occupancy_override = np.array(values, copy=True)

    @property
    def occupancy(self) -> np.ndarray:
        with self._lock:
            return self._occupancy_for_ordinal(self._fire_count)

    @property
    def fire_count(self) -> int:
        with self._lock:
            return self._fire_count

    def _occupancy_for_ordinal(self, ordinal: int) -> np.ndarray:
        if self._occupancy_override is not None:
            return np.array(self._occupancy_override, copy=True)
        return (int(ordinal) + np.arange(len(self.geometry.site_centers_xy))) % 2 == 0

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
                shot_occupancy = self._occupancy_for_ordinal(ordinal)
            else:
                shot_occupancy = np.asarray(occupancy, dtype=bool).reshape(-1)
                if shot_occupancy.size != len(self.geometry.site_centers_xy):
                    raise ValueError("occupancy size differs from simulation site map")
            for occupied, (x, y) in zip(shot_occupancy, self.geometry.site_centers_xy, strict=True):
                if occupied:
                    expected_electrons += self.atom_rate * exposure * np.exp(
                        -((xx - x) ** 2 + (yy - y) ** 2)
                        / (2.0 * self.atom_sigma_px**2)
                    )
            electrons = self.rng.poisson(np.clip(expected_electrons, 0.0, None))
            counts = electrons / self.conversion_e_per_count + self.offset_counts
            counts += self.rng.normal(
                0.0,
                self.read_noise_e / self.conversion_e_per_count,
                counts.shape,
            )
            return np.clip(counts, 0, np.iinfo(np.uint16).max).astype("<u2")

    def fire(self, count: int = 1, *, frame_exposures: object | None = None) -> None:
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
            shot_occupancy = self._occupancy_for_ordinal(ordinal)
            self._fire_count += 1
            # Devices own their protocol; the world only performs explicit
            # trigger routing and never exposes private backend channels.
            for camera in tuple(self._cameras):
                configured = float(camera.capture_working_point().exposure_seconds)
                effective = (
                    (configured,) * windows
                    if gates is None
                    else tuple(min(configured, gate) for gate in gates)
                )
                for exposure in effective:
                    frame = self.render_frame(
                        ordinal,
                        exposure_seconds=exposure,
                        occupancy=shot_occupancy,
                    )
                    camera.trigger(1, frame=frame)


__all__ = [
    "DEFAULT_SIMULATION_GRID_SHAPE_YX",
    "DEFAULT_SIMULATION_IMAGE_SHAPE_YX",
    "DEFAULT_SIMULATION_SITE_SPACING_PIXELS",
    "SimulationGeometry",
    "SimulationWorld",
    "SimulationWorldConfig",
]
