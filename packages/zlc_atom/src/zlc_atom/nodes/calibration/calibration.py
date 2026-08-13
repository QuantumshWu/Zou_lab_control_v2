"""Headless calibration values and the single box/PSF dispatch point."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from math import sqrt
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from zlc_data import SITE, AxisId, AxisSpec
from zlc_durable import write_readable_json

from .bimodal import fit_bimodal, finite_mean, gaussian_fidelity, optimal_gaussian_threshold, per_site_fidelity
from .psf import extract_psf_window, gaussian_psf_kernel


class ReadoutModelKind(str, Enum):
    """The closed set of readout feature models stored by a calibration."""

    BOX = "box"
    PER_SITE_PSF = "psf"
    UNIFORM_PSF = "uniform_psf"


DEFAULT_READOUT_MODEL_CHOICE = "default"
READOUT_MODEL_CHOICES = (
    DEFAULT_READOUT_MODEL_CHOICE,
    *(kind.value for kind in ReadoutModelKind),
)


def readout_model_kind_from_choice(
    value: object,
) -> ReadoutModelKind | None:
    """Resolve the authored request for the artifact-selected default."""

    if value == DEFAULT_READOUT_MODEL_CHOICE:
        return None
    try:
        return ReadoutModelKind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown readout model choice {value!r}") from exc


def _shape(value: object, field_name: str) -> tuple[int, int]:
    try:
        result = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a two-integer tuple") from exc
    if len(result) != 2 or any(item <= 0 for item in result):
        raise ValueError(f"{field_name} must contain two positive integers")
    return result


@dataclass(frozen=True)
class FrameContract:
    """Physical image facts on which a calibration is valid."""

    image_shape: tuple[int, int]
    sensor_shape: tuple[int, int] | None = None
    roi_xywh: tuple[int, int, int, int] | None = None
    binning_yx: tuple[int, int] = (1, 1)
    exposure_seconds: float | None = None
    camera_id: str | None = None
    readout_mode: str | None = None

    def __post_init__(self) -> None:
        image = _shape(self.image_shape, "image_shape")
        object.__setattr__(self, "image_shape", image)
        binning = _shape(self.binning_yx, "binning_yx")
        object.__setattr__(self, "binning_yx", binning)
        if self.sensor_shape is not None:
            sensor = _shape(self.sensor_shape, "sensor_shape")
            object.__setattr__(self, "sensor_shape", sensor)
            if sensor[0] < image[0] * binning[0] or sensor[1] < image[1] * binning[1]:
                raise ValueError("sensor_shape cannot be smaller than image_shape")
        if self.roi_xywh is not None:
            roi = tuple(int(item) for item in self.roi_xywh)
            if len(roi) != 4 or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0:
                raise ValueError("roi_xywh must be (x, y, width, height) with positive size")
            if roi[2:] != (image[1] * binning[1], image[0] * binning[0]):
                raise ValueError("roi_xywh size and binning must match image_shape")
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_shape": self.image_shape,
            "sensor_shape": self.sensor_shape,
            "roi_xywh": self.roi_xywh,
            "binning_yx": self.binning_yx,
            "exposure_seconds": self.exposure_seconds,
            "camera_id": self.camera_id,
            "readout_mode": self.readout_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrameContract":
        return cls(**dict(payload))


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


def _site_ids(value: object) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)  # type: ignore[arg-type]
    if not result or any(not item.strip() for item in result):
        raise ValueError("site_ids must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("site_ids must be unique")
    return result


def _nullable_floats(value: object) -> list[float | None]:
    return [float(item) if np.isfinite(item) else None for item in np.asarray(value, dtype=float).reshape(-1)]


def _floats_from_json(value: object) -> np.ndarray:
    return np.asarray([np.nan if item is None else float(item) for item in value], dtype="<f8")  # type: ignore[arg-type]


@dataclass(frozen=True)
class SiteMap:
    """Measured site identities and image-pixel positions."""

    site_ids: tuple[str, ...]
    centers_xy: np.ndarray
    valid_sites: np.ndarray
    quality: np.ndarray
    coordinate_frame: str = "image_pixel_xy"
    topology: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        site_ids = _site_ids(self.site_ids)
        centers = _immutable_array(self.centers_xy, "<f8", (len(site_ids), 2))
        if not np.isfinite(centers).all():
            raise ValueError("site centers must be finite")
        valid = _immutable_array(self.valid_sites, "?", (len(site_ids),))
        quality = _immutable_array(self.quality, "<f8", (len(site_ids),))
        frame = str(self.coordinate_frame).strip()
        if not frame:
            raise ValueError("coordinate_frame must be non-empty")
        if self.topology is not None and not isinstance(self.topology, Mapping):
            raise TypeError("topology must be a mapping or None")
        object.__setattr__(self, "site_ids", site_ids)
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "valid_sites", valid)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(self, "topology", None if self.topology is None else dict(self.topology))

    @property
    def n_sites(self) -> int:
        return len(self.site_ids)

    @cached_property
    def site_axis(self) -> AxisSpec:
        """THE declaration of site identity, built where the fact lives.

        Every site-axed signal in the tree -- occupancy's counts, the
        calibration report's fidelity, samples and PSF kernels -- names its
        sites through this one object, so two of them cannot disagree about
        which site is which.

        A site array is CELL data: it is one image resampled onto the trap
        lattice, and nobody scanned over sites.  So this is an ``AxisSpec``,
        not a point column.

        The coordinates are the ordinals 1..n.  Every projection in zlc_plot
        plots a coordinate as a number, so a text coordinate is refused at
        build time -- which is exactly why every site-axed signal used to
        raise ``DataViewError`` the moment anyone tried to draw it, and why a
        parallel "site ordinal" axis had to be invented to dodge the wall.

        The site ids are NOT the labels of this axis.  An id is how a record
        names a site to another record; a label is what a person reads on a
        picture.  Handing the id over as the label printed "Site=site_0001"
        across every facet title -- the same fact as the coordinate beside it,
        spelled in the form nobody can read and in the width nothing has room
        for.  The ids stay where identity belongs: in the site map, and in
        what it saves.
        """

        return AxisSpec(
            AxisId("calibration.site"),
            "Site",
            SITE,
            self.n_sites,
            coordinates=tuple(range(1, self.n_sites + 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_ids": list(self.site_ids),
            "centers_xy": self.centers_xy.tolist(),
            "valid_sites": self.valid_sites.tolist(),
            "quality": _nullable_floats(self.quality),
            "coordinate_frame": self.coordinate_frame,
            "topology": self.topology,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SiteMap":
        return cls(
            tuple(payload["site_ids"]),
            np.asarray(payload["centers_xy"]),
            np.asarray(payload["valid_sites"]),
            _floats_from_json(payload["quality"]),
            str(payload["coordinate_frame"]),
            payload["topology"],
        )


@dataclass(frozen=True)
class ReadoutModel:
    """Per-site integration features, response levels, and classification."""

    site_ids: tuple[str, ...]
    thresholds: np.ndarray
    dark_mean: np.ndarray
    bright_mean: np.ndarray
    usable_sites: np.ndarray
    quality: np.ndarray
    kind: ReadoutModelKind = ReadoutModelKind.BOX
    integration_half_width: int = 1
    reducer: str | None = "mean"
    threshold_method: str = "empirical"
    psf_weights: np.ndarray | None = None
    psf_boxes: np.ndarray | None = None
    background: str | None = None
    psf_padding: int | None = None

    def __post_init__(self) -> None:
        site_ids = _site_ids(self.site_ids)
        thresholds = np.asarray(self.thresholds, dtype="<f8")
        if thresholds.ndim == 0:
            thresholds = np.full(len(site_ids), float(thresholds), dtype="<f8")
        thresholds = _immutable_array(thresholds.reshape(-1), "<f8", (len(site_ids),))
        dark_mean = _immutable_array(self.dark_mean, "<f8", (len(site_ids),))
        bright_mean = _immutable_array(self.bright_mean, "<f8", (len(site_ids),))
        usable = _immutable_array(self.usable_sites, "?", (len(site_ids),))
        quality = _immutable_array(self.quality, "<f8", (len(site_ids),))
        with np.errstate(invalid="ignore", over="ignore"):
            response = bright_mean - dark_mean
        if np.any(
            usable
            & (
                ~np.isfinite(dark_mean)
                | ~np.isfinite(bright_mean)
                | ~np.isfinite(response)
                | (response <= 0.0)
            )
        ):
            raise ValueError("usable sites require finite bright_mean > dark_mean")
        if not isinstance(self.kind, ReadoutModelKind):
            raise TypeError("kind must be ReadoutModelKind")
        half_width = int(self.integration_half_width)
        if half_width < 0:
            raise ValueError("integration_half_width must be non-negative")
        threshold_method = str(self.threshold_method).lower()
        if threshold_method not in {"empirical", "gaussian"}:
            raise ValueError("threshold_method must be 'empirical' or 'gaussian'")
        reducer: str | None
        background: str | None
        padding: int | None
        weights = boxes = None
        if self.kind is ReadoutModelKind.BOX:
            reducer = str(self.reducer).lower()
            if reducer not in {"mean", "sum", "median", "max"}:
                raise ValueError("box reducer must be mean, sum, median, or max")
            if self.psf_weights is not None or self.psf_boxes is not None:
                raise ValueError("box model cannot carry PSF features")
            if self.background is not None or self.psf_padding is not None:
                raise ValueError("box model cannot carry PSF background parameters")
            background = None
            padding = None
        else:
            if self.reducer is not None:
                raise ValueError("PSF model cannot carry a box reducer")
            if self.psf_weights is None or self.psf_boxes is None:
                raise ValueError("PSF calibration requires psf_weights and psf_boxes")
            weights = _immutable_array(self.psf_weights, "<f8")
            boxes = _immutable_array(self.psf_boxes, "<i8")
            if weights.ndim != 3 or weights.shape[0] != len(site_ids) or boxes.shape != (len(site_ids), 4):
                raise ValueError("PSF arrays have incompatible shapes")
            background = str(self.background).lower()
            if background not in {"none", "annulus"}:
                raise ValueError("PSF background must be 'none' or 'annulus'")
            if self.psf_padding is None:
                raise ValueError("PSF model requires psf_padding")
            padding = int(self.psf_padding)
            if padding <= 0:
                raise ValueError("psf_padding must be positive")
            reducer = None
        object.__setattr__(self, "site_ids", site_ids)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "dark_mean", dark_mean)
        object.__setattr__(self, "bright_mean", bright_mean)
        object.__setattr__(self, "usable_sites", usable)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "integration_half_width", half_width)
        object.__setattr__(self, "reducer", reducer)
        object.__setattr__(self, "threshold_method", threshold_method)
        object.__setattr__(self, "psf_weights", weights)
        object.__setattr__(self, "psf_boxes", boxes)
        object.__setattr__(self, "background", background)
        object.__setattr__(self, "psf_padding", padding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "site_ids": list(self.site_ids),
            "thresholds": _nullable_floats(self.thresholds),
            "dark_mean": _nullable_floats(self.dark_mean),
            "bright_mean": _nullable_floats(self.bright_mean),
            "usable_sites": self.usable_sites.tolist(),
            "quality": _nullable_floats(self.quality),
            "threshold_method": self.threshold_method,
            "integration": {
                "half_width": self.integration_half_width,
                "reducer": self.reducer,
                "psf_weights": None if self.psf_weights is None else self.psf_weights.tolist(),
                "psf_boxes": None if self.psf_boxes is None else self.psf_boxes.tolist(),
                "background": self.background,
                "padding": self.psf_padding,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReadoutModel":
        integration = payload["integration"]
        return cls(
            tuple(payload["site_ids"]),
            _floats_from_json(payload["thresholds"]),
            _floats_from_json(payload["dark_mean"]),
            _floats_from_json(payload["bright_mean"]),
            np.asarray(payload["usable_sites"]),
            _floats_from_json(payload["quality"]),
            kind=ReadoutModelKind(payload["kind"]),
            integration_half_width=integration["half_width"],
            reducer=integration["reducer"],
            threshold_method=payload["threshold_method"],
            psf_weights=None if integration["psf_weights"] is None else np.asarray(integration["psf_weights"]),
            psf_boxes=None if integration["psf_boxes"] is None else np.asarray(integration["psf_boxes"]),
            background=integration["background"],
            psf_padding=integration["padding"],
        )


@dataclass(frozen=True)
class TrapCalibration:
    """One SiteMap and the aligned readout models trained from one capture."""

    site_map: SiteMap
    models: tuple[ReadoutModel, ...]
    default_model_kind: ReadoutModelKind
    frame_contract: FrameContract
    report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.site_map, SiteMap):
            raise TypeError("site_map must be SiteMap")
        models = tuple(self.models)
        if not models or any(not isinstance(model, ReadoutModel) for model in models):
            raise TypeError("models must contain at least one ReadoutModel")
        if not isinstance(self.frame_contract, FrameContract):
            raise TypeError("frame_contract must be FrameContract")
        kinds = tuple(model.kind for model in models)
        if len(set(kinds)) != len(kinds):
            raise ValueError("models must contain at most one model of each kind")
        if kinds != tuple(kind for kind in ReadoutModelKind if kind in kinds):
            raise ValueError("models must follow ReadoutModelKind order")
        if not isinstance(self.default_model_kind, ReadoutModelKind):
            raise TypeError("default_model_kind must be ReadoutModelKind")
        if self.default_model_kind not in kinds:
            raise ValueError("default_model_kind must name a stored model")
        if any(self.site_map.site_ids != model.site_ids for model in models):
            raise ValueError("SiteMap and every ReadoutModel site_ids must align")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "report", dict(self.report))

    @property
    def n_sites(self) -> int:
        return self.site_map.n_sites

    def select_model(
        self, kind: ReadoutModelKind | None = None
    ) -> ReadoutModel:
        selected = self.default_model_kind if kind is None else kind
        if not isinstance(selected, ReadoutModelKind):
            raise TypeError("kind must be ReadoutModelKind or None")
        for model in self.models:
            if model.kind is selected:
                return model
        raise KeyError(selected)

    def signals(
        self,
        image: object,
        *,
        model_kind: ReadoutModelKind | None = None,
    ) -> np.ndarray:
        array = np.asarray(image.values if hasattr(image, "values") else image.image if hasattr(image, "image") else image)
        array = self.frame_contract.assert_image(array)
        model = self.select_model(model_kind)
        if model.kind is ReadoutModelKind.BOX:
            values = extract_box_signals(
                array,
                self.site_map.centers_xy,
                radius=model.integration_half_width,
                reducer=model.reducer,  # type: ignore[arg-type]
            )
        else:
            values = extract_psf_signals(
                array,
                self.site_map.centers_xy,
                kernels=model.psf_weights,
                boxes_xywh=model.psf_boxes,
                background=model.background,
                radius=model.integration_half_width,
                padding=model.psf_padding,
            )
        return np.where(self.site_map.valid_sites & model.usable_sites, values, np.nan)

    def detect(
        self,
        image: object,
        *,
        model_kind: ReadoutModelKind | None = None,
    ) -> AtomDetection:
        model = self.select_model(model_kind)
        values = self.signals(image, model_kind=model.kind)
        thresholds = model.thresholds
        occupied = classify_threshold(values, thresholds)
        return AtomDetection(values, occupied, tuple(np.flatnonzero(occupied)), thresholds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_map": self.site_map.to_dict(),
            "models": [model.to_dict() for model in self.models],
            "default_model_kind": self.default_model_kind.value,
            "frame_contract": self.frame_contract.to_dict(),
            "report": dict(self.report),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrapCalibration":
        return cls(
            SiteMap.from_dict(payload["site_map"]),
            tuple(ReadoutModel.from_dict(model) for model in payload["models"]),
            ReadoutModelKind(payload["default_model_kind"]),
            FrameContract.from_dict(payload["frame_contract"]),
            payload["report"],
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        write_readable_json(target, self.to_dict())
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TrapCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CalibrationResult:
    calibration: TrapCalibration
    report: Mapping[str, Any]


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


def _stable_site_order(centers_xy: np.ndarray, row_tolerance: float) -> np.ndarray:
    """Order measured sites top-to-bottom, then left-to-right within a row."""

    rows: list[list[int]] = []
    for index in np.argsort(centers_xy[:, 1], kind="stable"):
        if not rows or centers_xy[index, 1] - float(np.mean(centers_xy[rows[-1], 1])) > row_tolerance:
            rows.append([int(index)])
        else:
            rows[-1].append(int(index))
    return np.asarray(
        [index for row in rows for index in sorted(row, key=lambda item: centers_xy[item, 0])],
        dtype=int,
    )


def detect_sites(
    frames: object,
    *,
    spot_sigma: float = 1.0,
    min_distance: int = 3,
    detection_sigma: float = 4.0,
) -> SiteMap:
    """Discover every resolvable site from how often it lights up.

    A site is not a place that is bright on average, and not a place that is
    bright in a high quantile either: both of those are statements about how
    OFTEN a trap is loaded as much as about whether a trap is there.  A
    lattice does not load uniformly -- a corner of it loading a fifth as often
    as the middle is ordinary -- so any single brightness cut over a run holds
    either the well-loaded traps or the poorly-loaded ones, never both, and
    the dim corner of the array simply never appeared.

    What every trap shares, whatever its loading, is that a loaded shot is
    unmistakable in the shot itself.  So each frame is thresholded on its own
    noise, which turns it into a map of where an atom was seen, and those maps
    are added up.  A place with a hundred sightings and a place with eight are
    then the same kind of evidence, differing only in loading -- and eight is
    a great many more than the noise of a run produces at one pixel.

    That last number is not a guess: thresholding a frame at k sigma admits a
    noise pixel with the Gaussian tail probability of k, so the count at a
    background pixel is binomial with that rate, and the count that a whole
    image of background will not reach follows from it.  ``detection_sigma``
    sets k -- how sure a single sighting must be -- and the arithmetic below
    converts that into how many sightings make a site.
    """

    from scipy import ndimage
    from scipy.special import erfc
    from scipy.stats import binom

    stack = np.asarray(
        frames.values if hasattr(frames, "values") else frames, dtype=float
    )
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3 or 0 in stack.shape or not np.isfinite(stack).all():
        raise ValueError("frames must be a non-empty finite stack of 2D images")
    spot_sigma = float(spot_sigma)
    detection_sigma = float(detection_sigma)
    min_distance = int(min_distance)
    if spot_sigma <= 0 or not np.isfinite(spot_sigma):
        raise ValueError("spot_sigma must be positive and finite")
    if detection_sigma <= 0 or not np.isfinite(detection_sigma):
        raise ValueError("detection_sigma must be positive and finite")
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")

    background_sigma = max(4.0 * spot_sigma, spot_sigma + 2.0)
    hits = np.zeros(stack.shape[1:], dtype=np.int64)
    lit_response = np.zeros(stack.shape[1:], dtype=float)
    for frame in stack:
        # Each frame is judged against its own noise: an exposure that came out
        # dim, or a run whose background drifted, changes what "bright" means
        # in that frame and in no other.
        smooth = ndimage.gaussian_filter(frame, sigma=spot_sigma)
        response = smooth - ndimage.gaussian_filter(frame, sigma=background_sigma)
        baseline = float(np.median(response))
        lower = response[response <= baseline]
        noise = 1.4826 * float(np.median(np.abs(lower - baseline)))
        noise = max(
            noise,
            np.finfo(float).eps * max(1.0, float(np.max(np.abs(response)))),
        )
        lit = response >= baseline + detection_sigma * noise
        hits += lit
        lit_response += np.where(lit, response - baseline, 0.0)

    shots = int(stack.shape[0])
    pixels = int(hits.size)
    # How often noise alone clears the per-frame cut, and therefore how many
    # sightings an image of pure background will not reach anywhere in it.
    false_rate = max(float(0.5 * erfc(detection_sigma / sqrt(2.0))), 1e-12)
    expected_false_sites = 0.5
    required = 1
    for count in range(1, shots + 1):
        if pixels * float(binom.sf(count - 1, shots, false_rate)) < expected_false_sites:
            required = count
            break
    else:
        required = shots

    # Where a site is, refined on how bright the place is WHEN it is lit: the
    # one image in a run whose contrast does not depend on loading.
    conditional = lit_response / np.maximum(hits, 1)
    local_maxima = hits == ndimage.maximum_filter(hits, size=3, mode="nearest")
    candidates = np.argwhere(local_maxima & (hits >= required))
    ranked = sorted(
        candidates,
        key=lambda item: (
            -int(hits[tuple(item)]),
            -float(conditional[tuple(item)]),
            int(item[0]),
            int(item[1]),
        ),
    )
    selected: list[tuple[int, int]] = []
    for row, column in ranked:
        point = (int(row), int(column))
        if all(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 >= min_distance**2
            for other in selected
        ):
            selected.append(point)
    if not selected:
        raise ValueError("calibration frames contain no detectable sites")

    refine_half = max(2, int(np.ceil(2.0 * spot_sigma)))
    centers = np.asarray(
        [
            _refine_center_subpixel(
                conditional, float(column), float(row), half=refine_half
            )
            for row, column in selected
        ],
        dtype="<f8",
    )
    # How far a site's count stands out of the noise count, in its own sigmas.
    spread = sqrt(max(shots * false_rate * (1.0 - false_rate), np.finfo(float).tiny))
    quality = np.asarray(
        [(float(hits[row, column]) - shots * false_rate) / spread for row, column in selected],
        dtype="<f8",
    )
    order = _stable_site_order(centers, float(min_distance))
    centers = centers[order]
    quality = quality[order]
    site_ids = tuple(f"site_{index:04d}" for index in range(len(centers)))
    return SiteMap(
        site_ids,
        centers,
        np.ones(len(centers), dtype=bool),
        quality,
        "image_pixel_xy",
        None,
    )


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


def _empirical_threshold(
    dark: object,
    bright: object,
    *,
    bright_above: bool,
    tie_target: float,
) -> float:
    """The cut that classifies THIS run's labelled shots best.

    The labels come from the long frames, where an atom is unmistakable; the
    values are the short ones a runtime readout will actually see.  So the
    best cut is a fact about the data in hand, found by trying every place a
    cut can go -- between one observed value and the next -- and keeping the
    one that classifies the labelled shots best, with ties settled towards the
    fitted Gaussian crossing.

    Every place a cut can go, rather than every histogram bin edge: a cut is
    not a bin, and rounding it to the display's binning moved the operating
    point of the readout by up to half a bin for no reason but the picture.
    """

    dark_values = np.asarray(dark, dtype=float).reshape(-1)
    bright_values = np.asarray(bright, dtype=float).reshape(-1)
    dark_values = dark_values[np.isfinite(dark_values)]
    bright_values = bright_values[np.isfinite(bright_values)]
    if not dark_values.size or not bright_values.size:
        return float("nan")
    values = np.unique(np.concatenate([dark_values, bright_values]))
    if values.size == 1:
        return float(values[0])
    # A cut sits between two observed values, plus one outside each end.
    cuts = np.concatenate(
        [
            [values[0] - 0.5 * float(values[1] - values[0])],
            0.5 * (values[:-1] + values[1:]),
            [values[-1] + 0.5 * float(values[-1] - values[-2])],
        ]
    )
    dark_below = np.searchsorted(np.sort(dark_values), cuts, side="left") / dark_values.size
    bright_below = (
        np.searchsorted(np.sort(bright_values), cuts, side="left") / bright_values.size
    )
    if bright_above:
        fidelity = 0.5 * (dark_below + (1.0 - bright_below))
    else:
        fidelity = 0.5 * ((1.0 - dark_below) + bright_below)
    best_value = float(np.max(fidelity))
    best = np.flatnonzero(np.isclose(fidelity, best_value, rtol=0.0, atol=1e-12))
    index = int(best[np.argmin(np.abs(cuts[best] - float(tie_target)))])
    return float(cuts[index])

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


def _train_readout_model(
    *,
    kind: ReadoutModelKind,
    site_map: SiteMap,
    short_signals: np.ndarray,
    labels_occupied: np.ndarray,
    labels_valid: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    threshold_method: str,
    model_parameters: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
) -> tuple[ReadoutModel, dict[str, Any]]:
    """Train one feature model against the shared labels and split."""

    centers = site_map.centers_xy
    thresholds = np.full(len(centers), np.nan, dtype=float)
    predictions = np.zeros_like(short_signals, dtype=bool)
    site_model_fidelity = np.full(len(centers), np.nan, dtype=float)
    gaussian_thresholds = np.full(len(centers), np.nan, dtype=float)
    dark_means = np.full(len(centers), np.nan, dtype=float)
    bright_means = np.full(len(centers), np.nan, dtype=float)
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
            dark_means[site], bright_means[site] = dark_mean, bright_mean
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
            gaussian_thresholds[site] = gaussian_threshold
            if threshold_method == "gaussian":
                threshold = gaussian_threshold
            else:
                threshold = _empirical_threshold(
                    dark,
                    bright_values,
                    bright_above=True,
                    tie_target=gaussian_threshold,
                )
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
    usable_sites = site_map.valid_sites & np.isfinite(thresholds)
    readout_model = ReadoutModel(
        site_map.site_ids,
        thresholds,
        dark_means,
        bright_means,
        usable_sites,
        site_fidelity,
        kind=kind,
        threshold_method=threshold_method,
        **dict(model_parameters),
    )
    report = {
        "kind": kind.value,
        "short_signals": short_signals,
        "thresholds": thresholds,
        "gaussian_thresholds": gaussian_thresholds,
        "predictions": predictions,
        "site_usable": usable_sites,
        "site_fidelity": site_fidelity,
        "site_fidelity_dark": confusion.dark,
        "site_fidelity_bright": confusion.bright,
        "site_model_fidelity": site_model_fidelity,
        "site_n_test": confusion.tested,
        "site_n_train_dark": n_train_dark,
        "site_n_train_bright": n_train_bright,
    }
    report.update(dict(diagnostics or {}))
    return readout_model, report


def calibrate(
    reference_frames: object,
    short_frames: object,
    *,
    frame_contract: FrameContract,
    default_model_kind: ReadoutModelKind = ReadoutModelKind.BOX,
    threshold_method: str = "empirical",
    box_half_width: int = 1,
    box_reducer: str = "mean",
    psf_half_width: int = 3,
    psf_padding: int = 3,
    detection_spot_sigma: float = 1.0,
    detection_min_distance: int = 3,
    detection_sigma: float = 6.0,
    train_fraction: float = 0.9,
    split_seed: int = 0,
) -> CalibrationResult:
    """Discover sites once and train all readout models from one capture."""

    if not isinstance(default_model_kind, ReadoutModelKind):
        raise TypeError("default_model_kind must be ReadoutModelKind")
    threshold_method = str(threshold_method).lower()
    if threshold_method not in {"empirical", "gaussian"}:
        raise ValueError("threshold_method must be 'empirical' or 'gaussian'")
    box_half_width = int(box_half_width)
    psf_half_width = int(psf_half_width)
    psf_padding = int(psf_padding)
    if box_half_width < 0 or psf_half_width < 0:
        raise ValueError("integration half-widths must be non-negative")
    if psf_padding <= 0:
        raise ValueError("psf_padding must be positive")
    box_reducer = str(box_reducer).lower()
    if box_reducer not in {"mean", "sum", "median", "max"}:
        raise ValueError("box_reducer must be mean, sum, median, or max")

    references = _coerce_reference_stack(reference_frames, frame_contract)
    shorts = _coerce_short_stack(short_frames, frame_contract)
    if references.shape[0] != shorts.shape[0] or not references.shape[0] or not references.shape[1]:
        raise ValueError("reference and short frames must share non-empty group counts")
    reference_average = finite_mean(references, axis=(0, 1))
    # Sites are found from every reference frame of the run, one at a time:
    # what makes a place a site is being seen there repeatedly, and no summary
    # of the run keeps that -- an average or a quantile mixes "how bright when
    # loaded" with "how often loaded", and traps differ in the second.
    site_map = detect_sites(
        references.reshape(-1, *references.shape[2:]),
        spot_sigma=detection_spot_sigma,
        min_distance=detection_min_distance,
        detection_sigma=detection_sigma,
    )
    centers = site_map.centers_xy

    box_extractor: Callable[[np.ndarray], np.ndarray] = lambda frame: extract_box_signals(
        frame,
        centers,
        radius=box_half_width,
        reducer=box_reducer,
    )
    reference_label_signals = np.asarray(
        [[box_extractor(frame) for frame in group] for group in references],
        dtype=float,
    )
    reference_valid = np.isfinite(reference_label_signals)
    fits = tuple(
        fit_bimodal(
            reference_label_signals[:, :, site][reference_valid[:, :, site]]
        )
        for site in range(len(centers))
    )
    bright = np.zeros_like(reference_label_signals, dtype=bool)
    fit_ok = np.zeros(len(centers), dtype=bool)
    for site, fit in enumerate(fits):
        if fit.ok and fit.bright_above and np.isfinite(fit.threshold):
            reference_values = reference_label_signals[:, :, site]
            bright[:, :, site] = classify_threshold(
                reference_values,
                np.full(reference_values.shape, fit.threshold, dtype=float),
                bright_above=True,
            )
            fit_ok[site] = True
    every_reference_valid = np.all(reference_valid, axis=1)
    all_bright = np.all(bright, axis=1)
    all_dark = np.all(~bright, axis=1)
    labels_valid = (
        every_reference_valid
        & (all_bright | all_dark)
        & fit_ok[np.newaxis, :]
    )
    labels_occupied = all_bright & labels_valid
    labels_dark = all_dark & labels_valid
    train, test = _seeded_train_test(
        labels_occupied,
        labels_valid,
        train_fraction=train_fraction,
        seed=split_seed,
    )

    (
        per_site_weights,
        uniform_weight,
        psf_boxes,
        psf_fit_centers,
        psf_fit_sigmas,
        psf_fit_ok,
    ) = _fit_psf_features(
        reference_average,
        centers,
        radius=psf_half_width,
        padding=psf_padding,
    )
    uniform_weights = np.broadcast_to(
        uniform_weight, per_site_weights.shape
    ).copy()

    def psf_extractor(weights: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda frame: extract_psf_signals(
            frame,
            centers,
            kernels=weights,
            boxes_xywh=psf_boxes,
            background="annulus",
            radius=psf_half_width,
            padding=psf_padding,
        )

    feature_specs: tuple[
        tuple[
            ReadoutModelKind,
            Callable[[np.ndarray], np.ndarray],
            Mapping[str, Any],
            Mapping[str, Any],
        ],
        ...,
    ] = (
        (
            ReadoutModelKind.BOX,
            box_extractor,
            {
                "integration_half_width": box_half_width,
                "reducer": box_reducer,
            },
            {},
        ),
        (
            ReadoutModelKind.PER_SITE_PSF,
            psf_extractor(per_site_weights),
            {
                "integration_half_width": psf_half_width,
                "reducer": None,
                "psf_weights": per_site_weights,
                "psf_boxes": psf_boxes,
                "background": "annulus",
                "psf_padding": psf_padding,
            },
            {
                "psf_fit_centers_xy": psf_fit_centers,
                "psf_fit_sigma_xy": psf_fit_sigmas,
                "psf_fit_ok": psf_fit_ok,
                "psf_boxes_xywh": psf_boxes,
                "psf_kernels": per_site_weights,
            },
        ),
        (
            ReadoutModelKind.UNIFORM_PSF,
            psf_extractor(uniform_weights),
            {
                "integration_half_width": psf_half_width,
                "reducer": None,
                "psf_weights": uniform_weights,
                "psf_boxes": psf_boxes,
                "background": "annulus",
                "psf_padding": psf_padding,
            },
            {
                "psf_fit_centers_xy": psf_fit_centers,
                "psf_fit_sigma_xy": psf_fit_sigmas,
                "psf_fit_ok": psf_fit_ok,
                "psf_boxes_xywh": psf_boxes,
                "uniform_kernel": uniform_weight,
            },
        ),
    )
    models: list[ReadoutModel] = []
    model_reports: dict[str, dict[str, Any]] = {}
    for kind, extractor, parameters, diagnostics in feature_specs:
        short_signals = np.asarray(
            [extractor(frame) for frame in shorts], dtype=float
        )
        model, model_report = _train_readout_model(
            kind=kind,
            site_map=site_map,
            short_signals=short_signals,
            labels_occupied=labels_occupied,
            labels_valid=labels_valid,
            train=train,
            test=test,
            threshold_method=threshold_method,
            model_parameters=parameters,
            diagnostics=diagnostics,
        )
        models.append(model)
        model_reports[kind.value] = model_report

    calibration_report = {
        "models": {
            model.kind.value: {
                "site_n_test": [
                    int(value)
                    for value in model_reports[model.kind.value]["site_n_test"]
                ],
                "site_n_train_dark": [
                    int(value)
                    for value in model_reports[model.kind.value]["site_n_train_dark"]
                ],
                "site_n_train_bright": [
                    int(value)
                    for value in model_reports[model.kind.value]["site_n_train_bright"]
                ],
            }
            for model in models
        },
    }
    calibration = TrapCalibration(
        site_map,
        tuple(models),
        default_model_kind,
        frame_contract,
        calibration_report,
    )
    report = {
        "site_ids": site_map.site_ids,
        "site_centers_xy": site_map.centers_xy,
        "site_valid": site_map.valid_sites,
        "reference_average": reference_average,
        "reference_label_model_kind": ReadoutModelKind.BOX.value,
        "reference_label_signals": reference_label_signals,
        "labels_occupied": labels_occupied,
        "labels_dark": labels_dark,
        "labels_valid": labels_valid,
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
        "models": model_reports,
    }
    return CalibrationResult(calibration, report)


def signals(
    calibration: TrapCalibration,
    image: object,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> np.ndarray:
    if not isinstance(calibration, TrapCalibration):
        raise TypeError("calibration must be TrapCalibration")
    return calibration.signals(image, model_kind=model_kind)


def detect(
    calibration: TrapCalibration,
    image: object,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> AtomDetection:
    if not isinstance(calibration, TrapCalibration):
        raise TypeError("calibration must be TrapCalibration")
    return calibration.detect(image, model_kind=model_kind)


__all__ = [
    "AtomDetection",
    "CalibrationResult",
    "DEFAULT_READOUT_MODEL_CHOICE",
    "READOUT_MODEL_CHOICES",
    "classify_threshold",
    "detect_sites",
    "FrameContract",
    "ReadoutModel",
    "ReadoutModelKind",
    "SiteMap",
    "TrapCalibration",
    "calibrate",
    "detect",
    "extract_box_signals",
    "extract_psf_signals",
    "readout_model_kind_from_choice",
    "signals",
]
