"""Headless calibration values and the single box/PSF dispatch point."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from zlc_durable import write_readable_json

from .bimodal import fit_bimodal, finite_mean, gaussian_fidelity, optimal_gaussian_threshold, per_site_fidelity
from .psf import extract_psf_window, gaussian_psf_kernel


class GridOrder(str, Enum):
    ROW_MAJOR = "row_major"
    SERPENTINE = "serpentine"
    COLUMN_MAJOR = "column_major"
    COLUMN_SERPENTINE = "column_serpentine"


def _shape(value: object, field_name: str) -> tuple[int, int]:
    try:
        result = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a two-integer tuple") from exc
    if len(result) != 2 or any(item <= 0 for item in result):
        raise ValueError(f"{field_name} must contain two positive integers")
    return result


def site_grid_positions_yx(grid_shape: tuple[int, int], ordering: GridOrder = GridOrder.ROW_MAJOR) -> np.ndarray:
    """Return normalized ``(row, column)`` grid indices in authored order."""

    rows, columns = _shape(grid_shape, "grid_shape")
    ordering = GridOrder(ordering)
    positions: list[tuple[int, int]] = []
    if ordering in (GridOrder.ROW_MAJOR, GridOrder.SERPENTINE):
        for row in range(rows):
            cols = range(columns) if ordering is GridOrder.ROW_MAJOR or row % 2 == 0 else range(columns - 1, -1, -1)
            positions.extend((row, column) for column in cols)
    else:
        for column in range(columns):
            row_range = range(rows) if ordering is GridOrder.COLUMN_MAJOR or column % 2 == 0 else range(rows - 1, -1, -1)
            positions.extend((row, column) for row in row_range)
    return np.asarray(positions, dtype="<i8")


@dataclass(frozen=True)
class FrameContract:
    """Physical image facts on which a calibration is valid."""

    image_shape: tuple[int, int]
    sensor_shape: tuple[int, int] | None = None
    roi_xywh: tuple[int, int, int, int] | None = None
    exposure_seconds: float | None = None
    camera_id: str | None = None
    readout_mode: str | None = None

    def __post_init__(self) -> None:
        image = _shape(self.image_shape, "image_shape")
        object.__setattr__(self, "image_shape", image)
        if self.sensor_shape is not None:
            sensor = _shape(self.sensor_shape, "sensor_shape")
            object.__setattr__(self, "sensor_shape", sensor)
            if sensor[0] < image[0] or sensor[1] < image[1]:
                raise ValueError("sensor_shape cannot be smaller than image_shape")
        if self.roi_xywh is not None:
            roi = tuple(int(item) for item in self.roi_xywh)
            if len(roi) != 4 or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0:
                raise ValueError("roi_xywh must be (x, y, width, height) with positive size")
            if roi[2:] != (image[1], image[0]):
                raise ValueError("roi_xywh size must match image_shape")
            if self.sensor_shape is not None and (roi[0] + roi[2] > self.sensor_shape[1] or roi[1] + roi[3] > self.sensor_shape[0]):
                raise ValueError("roi_xywh lies outside sensor_shape")
            object.__setattr__(self, "roi_xywh", roi)
        if self.exposure_seconds is not None and (not np.isfinite(self.exposure_seconds) or self.exposure_seconds <= 0):
            raise ValueError("exposure_seconds must be finite and positive")
        for name in ("camera_id", "readout_mode"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} cannot be blank")

    def assert_image(self, image: object) -> np.ndarray:
        payload = image.values if hasattr(image, "values") else image.image if hasattr(image, "image") else image
        array = np.asarray(payload)
        if array.shape != self.image_shape:
            raise ValueError(f"image shape {array.shape} differs from calibration {self.image_shape}")
        return array

    @property
    def fingerprint(self) -> str:
        payload = {
            "image_shape": self.image_shape,
            "sensor_shape": self.sensor_shape,
            "roi_xywh": self.roi_xywh,
            "exposure_seconds": self.exposure_seconds,
            "camera_id": self.camera_id,
            "readout_mode": self.readout_mode,
        }
        return hashlib.blake2b(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            digest_size=16,
        ).hexdigest()


@dataclass(frozen=True)
class AtomDetection:
    """Named result returned by :meth:`TrapCalibration.detect`."""

    counts: np.ndarray
    occupied: np.ndarray
    occupied_indices: tuple[int, ...]
    thresholds: np.ndarray

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts, dtype="<f8")
        occupied = np.asarray(self.occupied, dtype=bool)
        thresholds = np.asarray(self.thresholds, dtype="<f8")
        if counts.shape != occupied.shape or thresholds.shape != counts.shape:
            raise ValueError("detection arrays must share one site shape")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "occupied_indices", tuple(int(item) for item in self.occupied_indices))

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        return np.asarray(self.occupied, dtype=dtype)


def classify_threshold(counts: object, thresholds: object, *, bright_above: bool = True) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    boundary = np.asarray(thresholds, dtype=float)
    if boundary.ndim == 1 and values.ndim >= 1 and values.shape[-1] == boundary.size:
        boundary = np.broadcast_to(boundary, values.shape)
    elif values.shape != boundary.shape:
        raise ValueError("counts and thresholds must have the same shape or site-axis broadcasting")
    if bright_above:
        return np.isfinite(values) & np.isfinite(boundary) & (values > boundary)
    return np.isfinite(values) & np.isfinite(boundary) & (values < boundary)


def _immutable_array(value: object, dtype: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array shape {array.shape} differs from expected {shape}")
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _box_bounds(center: tuple[float, float], radius: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y = (int(round(float(center[0]))), int(round(float(center[1]))))
    width, height = 2 * int(radius) + 1, 2 * int(radius) + 1
    x0, y0 = x - int(radius), y - int(radius)
    if x0 < 0 or y0 < 0 or x0 + width > image_shape[1] or y0 + height > image_shape[0]:
        raise ValueError(f"site center {center!r} with radius {radius} lies outside image")
    return x0, y0, width, height


def extract_box_signals(image: object, centers_xy: object, *, radius: int = 1, reducer: str = "mean") -> np.ndarray:
    """Extract one box statistic per site."""

    array = np.asarray(image.values if hasattr(image, "values") else image, dtype=float)
    if array.ndim != 2:
        raise ValueError("image must be two-dimensional")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    reducer = str(reducer).lower()
    if reducer not in {"mean", "sum", "median", "max"}:
        raise ValueError("reducer must be mean, sum, median, or max")
    output = np.full(len(np.asarray(centers_xy)), np.nan, dtype="<f8")
    for index, center in enumerate(np.asarray(centers_xy, dtype=float).reshape(-1, 2)):
        x, y, width, height = _box_bounds(tuple(center), radius, array.shape)
        values = array[y : y + height, x : x + width]
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        output[index] = {"mean": np.mean, "sum": np.sum, "median": np.median, "max": np.max}[reducer](finite)
    return output


def extract_psf_signals(
    image: object,
    centers_xy: object,
    *,
    kernels: np.ndarray | None = None,
    boxes_xywh: np.ndarray | None = None,
    background: str = "annulus",
    radius: int = 2,
    padding: int = 3,
) -> np.ndarray:
    """Extract one matched-filter statistic per site."""

    array = np.asarray(image.values if hasattr(image, "values") else image)
    centers = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    size = 2 * int(radius) + 1
    if kernels is None:
        kernels = np.broadcast_to(gaussian_psf_kernel(1.0, radius), (len(centers), size, size))
    kernels = np.asarray(kernels, dtype=float)
    if kernels.shape != (len(centers), size, size):
        raise ValueError("kernels must have shape (N, 2*radius+1, 2*radius+1)")
    if boxes_xywh is None:
        boxes = np.asarray([_box_bounds(tuple(center), int(radius), array.shape) for center in centers], dtype=int)
    else:
        boxes = np.asarray(boxes_xywh, dtype=int)
    if boxes.shape != (len(centers), 4):
        raise ValueError("boxes_xywh must have shape (N, 4)")
    output = np.full(len(centers), np.nan, dtype="<f8")
    for index, (box, kernel) in enumerate(zip(boxes, kernels, strict=True)):
        x, y, width, height = (int(value) for value in box)
        if kernel.shape != (height, width):
            raise ValueError("PSF kernel shape differs from box")
        cut = array[y : y + height, x : x + width]
        if cut.shape != kernel.shape or not np.isfinite(cut).all():
            continue
        output[index] = extract_psf_window(array, (x, y, width, height), kernel, background=background, padding=int(padding))
    return output


@dataclass(frozen=True)
class TrapCalibration:
    """Immutable site map and thresholds with explicit box/PSF dispatch."""

    centers: np.ndarray
    thresholds: np.ndarray | float
    frame_contract: FrameContract | None = None
    method: str = "box"
    roi_radius: int = 1
    reducer: str = "mean"
    psf_weights: np.ndarray | None = None
    psf_boxes: np.ndarray | None = None
    background: str = "none"
    psf_padding: int = 3
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        centers = _immutable_array(self.centers, "<f8")
        if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers):
            raise ValueError("centers must have shape (N, 2)")
        thresholds = np.asarray(self.thresholds, dtype="<f8")
        if thresholds.ndim == 0:
            thresholds = np.full(len(centers), float(thresholds), dtype="<f8")
        thresholds = _immutable_array(thresholds.reshape(-1), "<f8", (len(centers),))
        if self.frame_contract is not None and not isinstance(self.frame_contract, FrameContract):
            raise TypeError("frame_contract must be FrameContract or None")
        method = str(self.method).lower()
        if method not in {"box", "psf", "uniform_psf"}:
            raise ValueError("method must be 'box', 'psf', or 'uniform_psf'")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "method", method)
        radius = int(self.roi_radius)
        if radius < 0:
            raise ValueError("roi_radius must be non-negative")
        object.__setattr__(self, "roi_radius", radius)
        object.__setattr__(self, "reducer", str(self.reducer).lower())
        background = str(self.background).lower()
        if background not in {"none", "annulus"}:
            raise ValueError("background must be 'none' or 'annulus'")
        object.__setattr__(self, "background", background)
        padding = int(self.psf_padding)
        if padding <= 0:
            raise ValueError("psf_padding must be positive")
        object.__setattr__(self, "psf_padding", padding)
        if method in {"psf", "uniform_psf"}:
            if self.psf_weights is None or self.psf_boxes is None:
                raise ValueError("PSF calibration requires psf_weights and psf_boxes")
            weights = _immutable_array(self.psf_weights, "<f8")
            boxes = _immutable_array(self.psf_boxes, "<i8")
            if weights.ndim != 3 or weights.shape[0] != len(centers) or boxes.shape != (len(centers), 4):
                raise ValueError("PSF arrays have incompatible shapes")
            object.__setattr__(self, "psf_weights", weights)
            object.__setattr__(self, "psf_boxes", boxes)
        else:
            object.__setattr__(self, "psf_weights", None)
            object.__setattr__(self, "psf_boxes", None)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_sites(self) -> int:
        return int(self.centers.shape[0])

    @property
    def fingerprint(self) -> str:
        payload = {
            "centers": self.centers.tolist(),
            "thresholds": self.thresholds.tolist(),
            "frame_contract": None if self.frame_contract is None else self.frame_contract.fingerprint,
            "method": self.method,
            "roi_radius": self.roi_radius,
            "reducer": self.reducer,
            "psf_weights": None if self.psf_weights is None else self.psf_weights.tolist(),
            "psf_boxes": None if self.psf_boxes is None else self.psf_boxes.tolist(),
            "background": self.background,
            "psf_padding": self.psf_padding,
            "metadata": self.metadata,
        }
        return hashlib.blake2b(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            digest_size=16,
        ).hexdigest()

    def signals(self, image: object, *, method: str | None = None) -> np.ndarray:
        array = np.asarray(image.values if hasattr(image, "values") else image.image if hasattr(image, "image") else image)
        if self.frame_contract is not None:
            array = self.frame_contract.assert_image(array)
        selected = self.method if method in (None, "") else str(method).lower()
        if selected != self.method:
            raise ValueError(f"calibration owns method {self.method!r}, not {selected!r}")
        if selected == "box":
            return extract_box_signals(array, self.centers, radius=self.roi_radius, reducer=self.reducer)
        return extract_psf_signals(
            array,
            self.centers,
            kernels=self.psf_weights,
            boxes_xywh=self.psf_boxes,
            background=self.background,
            radius=(self.psf_weights.shape[1] - 1) // 2,
            padding=self.psf_padding,
        )

    def detect(self, image: object, *, method: str | None = None) -> AtomDetection:
        values = self.signals(image, method=method)
        occupied = classify_threshold(values, self.thresholds)
        return AtomDetection(values, occupied, tuple(np.flatnonzero(occupied)), self.thresholds)

    def readout_exposure(self, fallback: float | None = None) -> float | None:
        if self.frame_contract is None or self.frame_contract.exposure_seconds is None:
            return fallback
        return float(self.frame_contract.exposure_seconds)

    def thresholds_for(self, method: str | None = None) -> np.ndarray:
        selected = self.method if method in (None, "") else str(method).lower()
        if selected != self.method:
            raise ValueError(f"calibration owns method {self.method!r}, not {selected!r}")
        return self.thresholds

    def with_thresholds(self, thresholds: object, **metadata: Any) -> "TrapCalibration":
        return replace(self, thresholds=thresholds, metadata={**self.metadata, **metadata})

    def to_dict(self) -> dict[str, Any]:
        return {
            "centers": self.centers.tolist(),
            "thresholds": self.thresholds.tolist(),
            "frame_contract": {
                "image_shape": None if self.frame_contract is None else self.frame_contract.image_shape,
                "sensor_shape": None if self.frame_contract is None else self.frame_contract.sensor_shape,
                "roi_xywh": None if self.frame_contract is None else self.frame_contract.roi_xywh,
                "exposure_seconds": None if self.frame_contract is None else self.frame_contract.exposure_seconds,
                "camera_id": None if self.frame_contract is None else self.frame_contract.camera_id,
                "readout_mode": None if self.frame_contract is None else self.frame_contract.readout_mode,
            } if self.frame_contract is not None else None,
            "method": self.method,
            "roi_radius": self.roi_radius,
            "reducer": self.reducer,
            "psf_weights": None if self.psf_weights is None else self.psf_weights.tolist(),
            "psf_boxes": None if self.psf_boxes is None else self.psf_boxes.tolist(),
            "background": self.background,
            "psf_padding": self.psf_padding,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrapCalibration":
        frame = payload["frame_contract"]
        contract = None if frame is None else FrameContract(**frame)
        calibration = cls(
            np.asarray(payload["centers"]),
            np.asarray(payload["thresholds"]),
            contract,
            method=payload.get("method", "box"),
            roi_radius=payload.get("roi_radius", 1),
            reducer=payload.get("reducer", "mean"),
            psf_weights=None if payload.get("psf_weights") is None else np.asarray(payload["psf_weights"]),
            psf_boxes=None if payload.get("psf_boxes") is None else np.asarray(payload["psf_boxes"]),
            background=payload.get("background", "none"),
            psf_padding=payload.get("psf_padding", 3),
            metadata=payload.get("metadata", {}),
        )
        if payload.get("fingerprint") not in (None, calibration.fingerprint):
            raise ValueError("calibration fingerprint mismatch")
        return calibration

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        # The same reader-facing layout every saved document in this project
        # uses, written atomically.  The two json.dumps above are DIGESTS and
        # stay compact and sorted on purpose: a hash must not move because a
        # line-wrap rule changed its mind.
        write_readable_json(target, self.to_dict())
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TrapCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CalibrationResult:
    calibration: TrapCalibration
    report: Mapping[str, Any]


def _regular_centers(image_shape: tuple[int, int], grid_shape: tuple[int, int], ordering: GridOrder) -> np.ndarray:
    rows, columns = _shape(grid_shape, "grid_shape")
    height, width = image_shape
    ys = np.linspace(0.18 * height, 0.82 * height, rows)
    xs = np.linspace(0.18 * width, 0.82 * width, columns)
    indices = site_grid_positions_yx((rows, columns), ordering)
    return np.asarray([(xs[column], ys[row]) for row, column in indices], dtype="<f8")


def _gaussian_2d(coords: object, offset: float, amplitude: float, x0: float, y0: float, sigma_x: float, sigma_y: float) -> np.ndarray:
    x, y = coords
    return (
        float(offset)
        + float(amplitude)
        * np.exp(-0.5 * (((x - float(x0)) / float(sigma_x)) ** 2 + ((y - float(y0)) / float(sigma_y)) ** 2))
    ).ravel()


def _fit_gaussian_spot_2d(
    data: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    x0: float,
    y0: float,
    offset0: float,
    amplitude: float,
    sigma0: float = 0.9,
) -> tuple[float, float, float, float, bool]:
    from scipy.optimize import curve_fit

    amplitude = float(amplitude)
    try:
        initial = [float(offset0), max(amplitude, 1e-6), float(x0), float(y0), sigma0, sigma0]
        lower = [float(np.nanmin(data)) - abs(amplitude) - 1, 0.0, float(xx.min()) - 0.5, float(yy.min()) - 0.5, 0.2, 0.2]
        upper = [float(np.nanmax(data)) + abs(amplitude) + 1, max(amplitude * 5, 1.0), float(xx.max()) + 0.5, float(yy.max()) + 0.5, 4.0, 4.0]
        fitted, _ = curve_fit(
            _gaussian_2d,
            (xx.ravel(), yy.ravel()),
            data.ravel(),
            p0=initial,
            bounds=(lower, upper),
            maxfev=5000,
        )
        _offset, _amplitude, x_fit, y_fit, sigma_x, sigma_y = fitted
        return float(x_fit), float(y_fit), float(abs(sigma_x)), float(abs(sigma_y)), True
    except Exception:
        values = np.clip(data - np.nanpercentile(data, 20), 0, None)
        total = float(np.sum(values))
        if total <= 0:
            return float(x0), float(y0), float(sigma0), float(sigma0), False
        return (
            float(np.sum(xx * values) / total),
            float(np.sum(yy * values) / total),
            float(sigma0),
            float(sigma0),
            False,
        )


def _refine_center_subpixel(image: np.ndarray, x: float, y: float, half: int = 2) -> tuple[float, float]:
    height, width = image.shape
    x_int, y_int = int(round(x)), int(round(y))
    x0, x1 = max(0, x_int - half), min(width, x_int + half + 1)
    y0, y1 = max(0, y_int - half), min(height, y_int + half + 1)
    cut = image[y0:y1, x0:x1]
    if cut.size < 9 or not np.isfinite(cut).any():
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    background = float(np.nanmedian(cut))
    amplitude = float(np.nanmax(cut) - background)
    x_fit, y_fit, _sigma_x, _sigma_y, _ok = _fit_gaussian_spot_2d(
        cut,
        yy,
        xx,
        x0=float(x),
        y0=float(y),
        offset0=background,
        amplitude=amplitude,
    )
    return x_fit, y_fit


def _sort_centers_grid(centers: np.ndarray, grid_shape: tuple[int, int], ordering: GridOrder) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    rows, columns = grid_shape
    if centers.shape != (rows * columns, 2):
        raise ValueError("center count does not match grid_shape")
    positions = site_grid_positions_yx(grid_shape, ordering)
    grid = np.empty((rows, columns, 2), dtype=float)
    if ordering in (GridOrder.ROW_MAJOR, GridOrder.SERPENTINE):
        for row_index, chunk in enumerate(np.array_split(centers[np.argsort(centers[:, 1])], rows)):
            grid[row_index, :, :] = chunk[np.argsort(chunk[:, 0])]
    else:
        for column_index, chunk in enumerate(np.array_split(centers[np.argsort(centers[:, 0])], columns)):
            grid[:, column_index, :] = chunk[np.argsort(chunk[:, 1])]
    return np.stack(tuple(grid[row, column] for row, column in positions))


def _robust_axis_lattice(anchors: np.ndarray, count: int) -> np.ndarray:
    anchors = np.asarray(anchors, dtype=float)
    if count <= 1:
        return anchors.copy()
    indices = np.arange(count, dtype=float)
    slopes = [(anchors[j] - anchors[i]) / (j - i) for i in range(count) for j in range(i + 1, count)]
    pitch = float(np.median(slopes)) if slopes else 0.0
    origin = float(np.median(anchors - pitch * indices))
    return origin + pitch * indices


def _regularize_grid(row_major_centers: np.ndarray, grid_shape: tuple[int, int], image_shape: tuple[int, int]) -> np.ndarray:
    rows, columns = grid_shape
    grid = np.asarray(row_major_centers, dtype=float).reshape(rows, columns, 2)
    height, width = image_shape
    row_y = _robust_axis_lattice(np.median(grid[:, :, 1], axis=1), rows)
    column_x = _robust_axis_lattice(np.median(grid[:, :, 0], axis=0), columns)
    pitches = []
    if rows > 1:
        pitches.append(abs(float(row_y[1] - row_y[0])))
    if columns > 1:
        pitches.append(abs(float(column_x[1] - column_x[0])))
    pitches = [value for value in pitches if math.isfinite(value) and value > 0]
    pitch = float(min(pitches)) if pitches else float(min(height, width))
    lattice_x = np.broadcast_to(column_x[None, :], (rows, columns))
    lattice_y = np.broadcast_to(row_y[:, None], (rows, columns))
    off_node = np.hypot(grid[:, :, 0] - lattice_x, grid[:, :, 1] - lattice_y) > 0.5 * pitch
    grid[off_node, 0] = lattice_x[off_node]
    grid[off_node, 1] = lattice_y[off_node]
    margin = max(1.0, 0.25 * pitch)
    grid[:, :, 0] = np.clip(grid[:, :, 0], margin, max(margin, width - 1 - margin))
    grid[:, :, 1] = np.clip(grid[:, :, 1], margin, max(margin, height - 1 - margin))
    return grid.reshape(rows * columns, 2)


def find_site_centers(
    image: object,
    grid_shape: tuple[int, int],
    *,
    min_distance: int | None = None,
    threshold_rel: float = 0.35,
    ordering: GridOrder = GridOrder.ROW_MAJOR,
    refine_half: int = 2,
) -> np.ndarray:
    """Detect a bright lattice with subpixel refinement and lattice repair."""

    from scipy import ndimage

    array = np.asarray(image.values if hasattr(image, "values") else image, dtype=float)
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).any():
        raise ValueError("image must be a non-empty finite 2D array")
    grid = _shape(grid_shape, "grid_shape")
    rows, columns = grid
    needed = rows * columns
    if min_distance is None:
        min_distance = max(3, int(min(array.shape) / max(rows, columns, 1) / 2))
    min_distance = int(min_distance)
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    smooth = ndimage.gaussian_filter(array, sigma=1.0)
    cutoff = float(np.nanmin(smooth) + float(threshold_rel) * (np.nanmax(smooth) - np.nanmin(smooth)))
    local_max = ndimage.maximum_filter(smooth, size=min_distance)
    is_peak = smooth == local_max
    candidates_yx = np.argwhere(is_peak & (smooth >= cutoff))
    if len(candidates_yx) < needed:
        peaks_yx = np.argwhere(is_peak)
        if len(peaks_yx) < needed:
            raise ValueError(f"only {len(peaks_yx)} local maxima for a {rows}x{columns} grid")
        weights = smooth[peaks_yx[:, 0], peaks_yx[:, 1]]
        candidates_yx = peaks_yx[np.argsort(weights)[::-1][:needed]]
    weights = smooth[candidates_yx[:, 0], candidates_yx[:, 1]]
    selected = candidates_yx[np.argsort(weights)[::-1]][:needed]
    centers = np.asarray(
        [_refine_center_subpixel(array, float(column), float(row), half=int(refine_half)) for row, column in selected],
        dtype=float,
    )
    row_major = _sort_centers_grid(centers, grid, GridOrder.ROW_MAJOR)
    repaired = _regularize_grid(row_major, grid, array.shape)
    return _sort_centers_grid(repaired, grid, GridOrder(ordering))


def _annulus_background(image: np.ndarray, box: tuple[int, int, int, int], padding: int) -> float:
    x, y, width, height = (int(value) for value in box)
    y0, y1 = max(0, y - int(padding)), min(image.shape[0], y + height + int(padding))
    x0, x1 = max(0, x - int(padding)), min(image.shape[1], x + width + int(padding))
    region = np.asarray(image[y0:y1, x0:x1], dtype=float)
    ring = np.array(region, copy=True)
    ring[y - y0 : y - y0 + height, x - x0 : x - x0 + width] = np.nan
    return float(np.nanmedian(ring)) if np.isfinite(ring).any() else 0.0


def _fit_psf_features(
    reference_average: np.ndarray,
    centers_xy: np.ndarray,
    *,
    radius: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit per-site kernels and their uniform sibling from the reference image."""

    from scipy import ndimage

    radius = int(radius)
    padding = int(padding)
    if radius < 0 or padding <= 0:
        raise ValueError("PSF radius must be non-negative and padding must be positive")
    boxes: list[tuple[int, int, int, int]] = []
    kernels: list[np.ndarray] = []
    fit_centers: list[tuple[float, float]] = []
    fit_sigmas: list[tuple[float, float]] = []
    fit_ok: list[bool] = []
    for center in np.asarray(centers_xy, dtype=float).reshape(-1, 2):
        box = _box_bounds(tuple(center), radius, reference_average.shape)
        x, y, width, height = box
        boxes.append(box)
        cut = reference_average[y : y + height, x : x + width]
        background = _annulus_background(reference_average, box, padding)
        subtracted = cut - background
        yy, xx = np.mgrid[y : y + height, x : x + width]
        amplitude = float(np.nanmax(subtracted)) if np.isfinite(subtracted).any() else 0.0
        x_fit, y_fit, sigma_x, sigma_y, ok = _fit_gaussian_spot_2d(
            subtracted,
            yy,
            xx,
            x0=float(center[0]),
            y0=float(center[1]),
            offset0=0.0,
            amplitude=amplitude,
        )
        positive = ndimage.gaussian_filter(np.clip(subtracted, 0, None), 0.35)
        total = float(np.sum(positive))
        if total > 0:
            kernel = positive / total
        else:
            kernel = gaussian_psf_kernel(float(np.mean((sigma_x, sigma_y))), radius)
        kernels.append(np.ascontiguousarray(kernel, dtype="<f8"))
        fit_centers.append((x_fit, y_fit))
        fit_sigmas.append((sigma_x, sigma_y))
        fit_ok.append(bool(ok))
    per_site = np.stack(kernels, axis=0)
    uniform = np.mean(per_site, axis=0)
    uniform = uniform / float(np.sum(uniform))
    return (
        per_site,
        uniform,
        np.asarray(boxes, dtype="<i8"),
        np.asarray(fit_centers, dtype="<f8"),
        np.asarray(fit_sigmas, dtype="<f8"),
        np.asarray(fit_ok, dtype=bool),
    )


def _coerce_reference_stack(reference_frames: object, frame_contract: FrameContract) -> np.ndarray:
    if isinstance(reference_frames, np.ndarray):
        raw = np.asarray(reference_frames)
        if raw.ndim == 4:
            return np.asarray(
                [[frame_contract.assert_image(frame) for frame in group] for group in raw],
                dtype=float,
            )
        if raw.ndim == 3:
            return np.asarray(
                [[frame_contract.assert_image(frame)] for frame in raw],
                dtype=float,
            )
        if raw.ndim == 2:
            return np.asarray([[frame_contract.assert_image(raw)]], dtype=float)
        raise ValueError("reference frames must have shape (groups, shots, y, x)")
    groups = list(reference_frames)  # type: ignore[arg-type]
    if not groups:
        raise ValueError("calibration requires non-empty reference frames")
    nested = isinstance(groups[0], (tuple, list))
    grouped = groups if nested else [[item] for item in groups]
    return np.asarray(
        [[frame_contract.assert_image(frame) for frame in group] for group in grouped],
        dtype=float,
    )


def _coerce_short_stack(short_frames: object, frame_contract: FrameContract) -> np.ndarray:
    if isinstance(short_frames, np.ndarray):
        raw = np.asarray(short_frames)
        if raw.ndim != 3:
            raise ValueError("short frames must have shape (groups, y, x)")
        return np.asarray([frame_contract.assert_image(frame) for frame in raw], dtype=float)
    frames = list(short_frames)  # type: ignore[arg-type]
    if not frames:
        raise ValueError("calibration requires non-empty short frames")
    return np.asarray([frame_contract.assert_image(frame) for frame in frames], dtype=float)


def _common_bin_edges(values: np.ndarray, mask: np.ndarray, *, bins: int) -> np.ndarray:
    samples = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        return np.linspace(-1.0, 1.0, int(bins) + 1)
    lower, upper = np.quantile(samples, (0.001, 0.999))
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        lower, upper = float(np.min(samples)), float(np.max(samples))
    span = max(float(upper - lower), 1.0)
    return np.linspace(lower - 0.04 * span, upper + 0.04 * span, int(bins) + 1)


def _empirical_threshold(
    dark: object,
    bright: object,
    edges: object,
    *,
    bright_above: bool,
    tie_target: float,
) -> float:
    dark_values = np.asarray(dark, dtype=float).reshape(-1)
    bright_values = np.asarray(bright, dtype=float).reshape(-1)
    dark_values = dark_values[np.isfinite(dark_values)]
    bright_values = bright_values[np.isfinite(bright_values)]
    if not dark_values.size or not bright_values.size:
        return float("nan")
    edges = np.asarray(edges, dtype=float)
    dark_histogram, _ = np.histogram(dark_values, bins=edges)
    bright_histogram, _ = np.histogram(bright_values, bins=edges)
    dark_cumulative = np.concatenate([[0], np.cumsum(dark_histogram)])
    bright_cumulative = np.concatenate([[0], np.cumsum(bright_histogram)])
    if bright_above:
        dark_fidelity = dark_cumulative / dark_values.size
        bright_fidelity = (bright_values.size - bright_cumulative) / bright_values.size
    else:
        dark_fidelity = (dark_values.size - dark_cumulative) / dark_values.size
        bright_fidelity = bright_cumulative / bright_values.size
    fidelity = 0.5 * (dark_fidelity + bright_fidelity)
    best_value = float(np.nanmax(fidelity))
    best = np.flatnonzero(np.isclose(fidelity, best_value, rtol=0.0, atol=1e-12))
    index = int(best[np.argmin(np.abs(edges[best] - float(tie_target)))]) if best.size > 1 else int(best[0])
    return float(edges[index])


def _seeded_train_test(occupied: np.ndarray, valid: np.ndarray, *, train_fraction: float = 0.9, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    fraction = float(train_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    occupied = np.asarray(occupied, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if occupied.shape != valid.shape or occupied.ndim != 2:
        raise ValueError("occupied and valid labels must share a (groups, sites) shape")
    train = np.zeros_like(valid)
    test = np.zeros_like(valid)
    rng = np.random.default_rng(int(seed))
    for site in range(occupied.shape[1]):
        for state in (False, True):
            indices = np.where(valid[:, site] & (occupied[:, site] == state))[0]
            if not indices.size:
                continue
            permutation = rng.permutation(indices)
            train_count = int(round(fraction * indices.size))
            train_count = min(max(train_count, 1), indices.size - 1) if indices.size >= 2 else 1
            train[permutation[:train_count], site] = True
            if train_count < indices.size:
                test[permutation[train_count:], site] = True
    return train, test


def calibrate(
    reference_frames: object,
    short_frames: object,
    *,
    frame_contract: FrameContract,
    grid_shape: tuple[int, int],
    method: str = "box",
    roi_radius: int = 1,
    reducer: str = "mean",
    expected_centers_xy: object | None = None,
    train_fraction: float = 0.9,
    split_seed: int = 0,
    histogram_bins: int = 120,
    max_drop: int = 5,
    maximum_site_residual_px: float = 0.1,
    psf_padding: int = 3,
) -> CalibrationResult:
    """Calibrate using grouped reference labels and held-out short frames."""

    references = _coerce_reference_stack(reference_frames, frame_contract)
    shorts = _coerce_short_stack(short_frames, frame_contract)
    if references.shape[0] != shorts.shape[0] or not references.shape[0] or not references.shape[1]:
        raise ValueError("reference and short frames must share non-empty group counts")
    reference_average = finite_mean(references, axis=(0, 1))
    centers = find_site_centers(reference_average, grid_shape)
    if expected_centers_xy is not None:
        expected = np.asarray(expected_centers_xy, dtype=float).reshape(-1, 2)
        residual_limit = float(maximum_site_residual_px)
        if not np.isfinite(residual_limit) or residual_limit <= 0.0:
            raise ValueError("maximum_site_residual_px must be finite and positive")
        if expected.shape != centers.shape or np.max(np.linalg.norm(centers - expected, axis=1)) > residual_limit:
            raise ValueError("detected site centers differ from expected_centers_xy")
        centers = expected.copy()
    method = str(method).lower()
    if method == "box":
        extractor = lambda frame: extract_box_signals(frame, centers, radius=roi_radius, reducer=reducer)
        calibration_kwargs: dict[str, Any] = {}
    elif method in {"psf", "uniform_psf"}:
        per_site_weights, uniform_weight, boxes, psf_fit_centers, psf_fit_sigmas, psf_fit_ok = _fit_psf_features(
            reference_average,
            centers,
            radius=roi_radius,
            padding=psf_padding,
        )
        weights = per_site_weights if method == "psf" else np.broadcast_to(uniform_weight, per_site_weights.shape).copy()
        extractor = lambda frame: extract_psf_signals(
            frame,
            centers,
            kernels=weights,
            boxes_xywh=boxes,
            background="annulus",
            radius=roi_radius,
            padding=psf_padding,
        )
        calibration_kwargs = {
            "psf_weights": weights,
            "psf_boxes": boxes,
            "background": "annulus",
            "psf_padding": psf_padding,
        }
    else:
        raise ValueError("method must be 'box' or 'psf'")
    reference_signals = np.asarray([[extractor(frame) for frame in group] for group in references], dtype=float)
    short_signals = np.asarray([extractor(frame) for frame in shorts], dtype=float)
    reference_valid = np.isfinite(reference_signals)
    fits = tuple(fit_bimodal(reference_signals[:, :, site][reference_valid[:, :, site]]) for site in range(len(centers)))
    bright = np.zeros_like(reference_signals, dtype=bool)
    fit_ok = np.zeros(len(centers), dtype=bool)
    for site, fit in enumerate(fits):
        if fit.ok and fit.bright_above and np.isfinite(fit.threshold):
            reference_values = reference_signals[:, :, site]
            bright[:, :, site] = classify_threshold(
                reference_values,
                np.full(reference_values.shape, fit.threshold, dtype=float),
                bright_above=True,
            )
            fit_ok[site] = True
    every_reference_valid = np.all(reference_valid, axis=1)
    all_bright = np.all(bright, axis=1)
    all_dark = np.all(~bright, axis=1)
    labels_valid = every_reference_valid & (all_bright | all_dark) & fit_ok[np.newaxis, :]
    labels_occupied = all_bright & labels_valid
    labels_dark = all_dark & labels_valid
    train, test = _seeded_train_test(labels_occupied, labels_valid, train_fraction=train_fraction, seed=split_seed)
    common_validity = labels_valid & np.isfinite(short_signals)
    edges = _common_bin_edges(short_signals, common_validity, bins=histogram_bins)
    thresholds = np.full(len(centers), np.nan, dtype=float)
    predictions = np.zeros_like(short_signals, dtype=bool)
    site_fidelity = np.full(len(centers), np.nan, dtype=float)
    site_fidelity_dark = np.full(len(centers), np.nan, dtype=float)
    site_fidelity_bright = np.full(len(centers), np.nan, dtype=float)
    site_model_fidelity = np.full(len(centers), np.nan, dtype=float)
    n_test = np.zeros(len(centers), dtype=int)
    n_train_dark = np.zeros(len(centers), dtype=int)
    n_train_bright = np.zeros(len(centers), dtype=int)
    for site in range(len(centers)):
        finite = np.isfinite(short_signals[:, site])
        train_mask = train[:, site] & labels_valid[:, site] & finite
        test_mask = test[:, site] & labels_valid[:, site] & finite
        dark = short_signals[train_mask & ~labels_occupied[:, site], site]
        bright_values = short_signals[train_mask & labels_occupied[:, site], site]
        n_train_dark[site], n_train_bright[site] = dark.size, bright_values.size
        if dark.size >= 2 and bright_values.size >= 2:
            dark_mean, bright_mean = float(np.mean(dark)), float(np.mean(bright_values))
            dark_sigma = max(float(np.std(dark, ddof=1)), 1e-12)
            bright_sigma = max(float(np.std(bright_values, ddof=1)), 1e-12)
            gaussian_threshold, bright_above = optimal_gaussian_threshold(dark_mean, dark_sigma, bright_mean, bright_sigma)
            if not bright_above:
                # An atom scatters photons: bright is above dark, and a site
                # whose training data says otherwise has not been calibrated,
                # it has been fitted to noise.  Carrying the direction per site
                # let this loop classify one way while per_site_fidelity and
                # every later TrapCalibration.detect() classified the other --
                # three answers to "is this site bright?" on the number a
                # readout is judged by.  Refusing the site is one answer.
                n_train_dark[site] = n_train_bright[site] = 0
                continue
            threshold = _empirical_threshold(dark, bright_values, edges, bright_above=True, tie_target=gaussian_threshold)
            if not np.isfinite(threshold):
                threshold = gaussian_threshold
            site_model_fidelity[site] = gaussian_fidelity(dark_mean, dark_sigma, bright_mean, bright_sigma, threshold, True)[2]
            thresholds[site] = threshold
            short_values = short_signals[:, site]
            predictions[:, site] = classify_threshold(
                short_values,
                np.full(short_values.shape, threshold, dtype=float),
                bright_above=True,
            )
    # One confusion, computed once, reported whole.  The loop used to compute
    # its own balanced figure and have it thrown away by this call, while the
    # dark and bright halves it also computed -- under a different validity
    # mask -- were kept.  So the reported balanced fidelity was not the mean of
    # the reported halves.
    confusion = per_site_fidelity(
        short_signals,
        labels_occupied,
        thresholds,
        test_mask=test,
        valid_mask=labels_valid,
    )
    site_fidelity = confusion.balanced
    site_fidelity_dark = confusion.dark
    site_fidelity_bright = confusion.bright
    n_test = confusion.tested
    calibration = TrapCalibration(centers, thresholds, frame_contract, method=method, roi_radius=roi_radius, reducer=reducer, **calibration_kwargs)
    report = {
        "reference_average": reference_average,
        "reference_signals": reference_signals,
        "labels_occupied": labels_occupied,
        "labels_dark": labels_dark,
        "labels_valid": labels_valid,
        "short_signals": short_signals,
        "thresholds": thresholds,
        "predictions": predictions,
        "site_fidelity": site_fidelity,
        "site_fidelity_dark": site_fidelity_dark,
        "site_fidelity_bright": site_fidelity_bright,
        "site_model_fidelity": site_model_fidelity,
        "site_n_test": n_test,
        "site_n_train_dark": n_train_dark,
        "site_n_train_bright": n_train_bright,
        "split_train": train,
        "split_test": test,
        "fits": fits,
        "reference_fit_threshold": np.asarray([fit.threshold for fit in fits]),
        "reference_fit_fidelity": np.asarray([fit.fidelity for fit in fits]),
        "reference_fit_dark_mean": np.asarray([fit.dark_mean for fit in fits]),
        "reference_fit_dark_sigma": np.asarray([fit.dark_sigma for fit in fits]),
        "reference_fit_bright_mean": np.asarray([fit.bright_mean for fit in fits]),
        "reference_fit_bright_sigma": np.asarray([fit.bright_sigma for fit in fits]),
        "reference_fit_bright_above": np.asarray([fit.bright_above for fit in fits]),
        "reference_fit_ok": np.asarray([fit.ok for fit in fits]),
    }
    if method in {"psf", "uniform_psf"}:
        report.update(
            {
                "psf_fit_centers_xy": psf_fit_centers,
                "psf_fit_sigma_xy": psf_fit_sigmas,
                "psf_fit_ok": psf_fit_ok,
                "psf_boxes_xywh": boxes,
                "psf_kernels": per_site_weights,
                "uniform_kernel": uniform_weight,
            }
        )
    return CalibrationResult(calibration, report)


def signals(calibration: TrapCalibration, image: object, *, method: str | None = None) -> np.ndarray:
    if not isinstance(calibration, TrapCalibration):
        raise TypeError("calibration must be TrapCalibration")
    return calibration.signals(image, method=method)


def detect(calibration: TrapCalibration, image: object, *, method: str | None = None) -> AtomDetection:
    if not isinstance(calibration, TrapCalibration):
        raise TypeError("calibration must be TrapCalibration")
    return calibration.detect(image, method=method)


__all__ = [
    "AtomDetection",
    "CalibrationResult",
    "classify_threshold",
    "FrameContract",
    "GridOrder",
    "TrapCalibration",
    "calibrate",
    "detect",
    "extract_box_signals",
    "extract_psf_signals",
    "find_site_centers",
    "site_grid_positions_yx",
    "signals",
]
