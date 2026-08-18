"""Headless calibration values and the single box/PSF dispatch point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from math import sqrt

from scipy import ndimage
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


def reads_photoelectrons(calibration: object) -> bool:
    """Whether this calibration's thresholds are numbers of photoelectrons.

    Read from the run it was fitted on.  Every node that classifies frames
    against a calibration has to take its own frames in the SAME numbers --
    the conversion is affine, so a mismatch does not read a little wrong, it
    reads every site the same way -- and asking here is how they agree
    without each keeping its own switch.
    """

    report = getattr(calibration, "report", None)
    if not isinstance(report, Mapping):
        return False
    record = report.get("run_record")
    if not isinstance(record, Mapping):
        return False
    request = record.get("request")
    if not isinstance(request, Mapping):
        return False
    return bool(request.get("photoelectrons", False))


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


def box_fits(
    center: tuple[float, float],
    radius: int,
    image_shape: tuple[int, int],
) -> bool:
    """Whether a box of this radius round this centre is wholly in the picture.

    THE rule, asked by the detector before it publishes a centre and by the
    extractor before it reads one.  Two copies of it is how a run died at the
    readout for a site the detector had been happy to report.
    """

    x, y = (int(round(float(center[0]))), int(round(float(center[1]))))
    radius = int(radius)
    return (
        x - radius >= 0
        and y - radius >= 0
        and x + radius < int(image_shape[1])
        and y + radius < int(image_shape[0])
    )


def _box_bounds(center: tuple[float, float], radius: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if not box_fits(center, radius, image_shape):
        raise ValueError(f"site center {center!r} with radius {radius} lies outside image")
    x, y = (int(round(float(center[0]))), int(round(float(center[1]))))
    width, height = 2 * int(radius) + 1, 2 * int(radius) + 1
    return x - int(radius), y - int(radius), width, height


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
    if array.ndim != 2:
        raise ValueError("image must be two-dimensional")
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

    # The normal calibrated path has equal, complete PSF boxes and complete
    # annuli.  Gather those windows once instead of rebuilding 35 slices and
    # annulus masks for every frame.  Keep the final matched-filter reduction
    # site-by-site: changing that reduction order produces small float64
    # differences, while window gathering and the median are exactly the same
    # operations as the scalar path below.
    pad = int(padding)
    heights = boxes[:, 3]
    widths = boxes[:, 2]
    complete_boxes = bool(
        len(boxes)
        and np.all(widths == size)
        and np.all(heights == size)
        and np.all(boxes[:, 0] >= 0)
        and np.all(boxes[:, 1] >= 0)
        and np.all(boxes[:, 0] + widths <= array.shape[1])
        and np.all(boxes[:, 1] + heights <= array.shape[0])
    )
    complete_annuli = bool(
        background == "annulus"
        and pad >= 0
        and complete_boxes
        and np.all(boxes[:, 0] >= pad)
        and np.all(boxes[:, 1] >= pad)
        and np.all(boxes[:, 0] + widths + pad <= array.shape[1])
        and np.all(boxes[:, 1] + heights + pad <= array.shape[0])
    )
    if (background == "none" and complete_boxes) or complete_annuli:
        yy = boxes[:, 1, None, None] + np.arange(size)[None, :, None]
        xx = boxes[:, 0, None, None] + np.arange(size)[None, None, :]
        cuts = array[yy, xx]
        finite_cuts = np.isfinite(cuts).all(axis=(1, 2))
        offsets = np.zeros(len(boxes), dtype="<f8")
        if complete_annuli and pad:
            padded_size = size + 2 * pad
            padded_yy = (
                boxes[:, 1, None, None]
                - pad
                + np.arange(padded_size)[None, :, None]
            )
            padded_xx = (
                boxes[:, 0, None, None]
                - pad
                + np.arange(padded_size)[None, None, :]
            )
            padded = np.asarray(array[padded_yy, padded_xx], dtype=float)
            ring_mask = np.ones((padded_size, padded_size), dtype=bool)
            ring_mask[pad : pad + size, pad : pad + size] = False
            rings = padded[:, ring_mask]
            finite_rings = np.isfinite(rings).any(axis=1)
            if np.any(finite_rings):
                offsets[finite_rings] = np.nanmedian(
                    rings[finite_rings], axis=1
                )
        for index in np.flatnonzero(finite_cuts):
            output[index] = float(
                np.sum(
                    np.asarray(kernels[index], dtype=float)
                    * (np.asarray(cuts[index], dtype=float) - offsets[index])
                )
            )
        return output

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

    def rebased(
        self,
        roi_xywh: tuple[int, int, int, int] | None,
        binning_yx: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> "TrapCalibration":
        """This calibration, read against a different crop of the same sensor.

        Where a trap is, is a fact about the SENSOR; where it appears in an
        array is a fact about the crop the camera happened to take.  A run
        that moves the ROI therefore does not invalidate a calibration -- it
        just numbers the same places differently, and the translation is
        arithmetic this class can do because it knows both crops.

        Binning is not translatable: it changes what one pixel means, so the
        integration boxes and PSF kernels measured under it would be measuring
        something else.  A site the new crop does not cover is refused by
        name, because reading a box that runs off the edge would return a
        number that looks like a measurement.
        """

        contract = self.frame_contract
        binning = tuple(int(value) for value in binning_yx)
        if binning != tuple(contract.binning_yx):
            raise ValueError(
                f"binning {binning} differs from the calibration's "
                f"{tuple(contract.binning_yx)}: one pixel does not mean the "
                "same thing, so its thresholds do not either"
            )
        shape = tuple(int(value) for value in image_shape)
        if contract.roi_xywh is None or roi_xywh is None:
            if shape != tuple(contract.image_shape):
                raise ValueError(
                    f"frame shape {shape} differs from calibration "
                    f"{tuple(contract.image_shape)} and neither run records "
                    "the crop it came from, so the sites cannot be placed"
                )
            return self
        old_x, old_y, _old_w, _old_h = (int(value) for value in contract.roi_xywh)
        new_x, new_y, new_w, new_h = (int(value) for value in roi_xywh)
        if (new_h // binning[0], new_w // binning[1]) != shape:
            raise ValueError(
                f"frame shape {shape} does not match the crop {roi_xywh} it "
                "is said to come from"
            )
        shift_x = (old_x - new_x) / float(binning[1])
        shift_y = (old_y - new_y) / float(binning[0])
        if shift_x == 0.0 and shift_y == 0.0 and shape == tuple(contract.image_shape):
            return self
        centers = np.asarray(self.site_map.centers_xy, dtype=float) + (shift_x, shift_y)
        radius = max(model.integration_half_width for model in self.models)
        outside = [
            self.site_map.site_ids[index]
            for index, (x, y) in enumerate(centers)
            if not (
                radius <= x <= shape[1] - 1 - radius
                and radius <= y <= shape[0] - 1 - radius
            )
        ]
        if outside:
            raise ValueError(
                f"this crop does not cover {len(outside)} calibrated site(s) "
                f"({', '.join(outside[:4])}{'...' if len(outside) > 4 else ''}): "
                "move the ROI back over them or calibrate again"
            )
        moved_models = []
        for model in self.models:
            boxes = model.psf_boxes
            if boxes is not None:
                boxes = np.asarray(boxes, dtype=int) + (
                    int(round(shift_x)),
                    int(round(shift_y)),
                    0,
                    0,
                )
            moved_models.append(replace(model, psf_boxes=boxes))
        return replace(
            self,
            site_map=replace(self.site_map, centers_xy=centers),
            models=tuple(moved_models),
            frame_contract=replace(
                contract,
                image_shape=shape,
                roi_xywh=(new_x, new_y, new_w, new_h),
            ),
        )

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


def _background_scatter(
    image: np.ndarray,
    exclusion: int,
) -> tuple[float, float, np.ndarray]:
    """The level and the scatter of an image's BACKGROUND.

    Measured where the sources are not.  A robust spread over the whole
    picture is not the background's: an array of traps contributes its own
    peaks to the upper half and, once the picture has been band-passed, the
    dark rings around them to the lower half, so the spread grows with how
    many traps there are and how bright they got.  On a real run that read
    fourteen times the true scatter and buried every dim trap; estimated over
    the whole picture there is no threshold that is right for both a sparse
    array and a dense one.

    So the sources are found provisionally -- anything standing clear of the
    picture's own median -- and set aside with a spot's width around them.
    What is left is background, and its median and MAD are what a candidate is
    judged against.  If a picture is nothing but sources the estimate falls
    back to the whole of it rather than to nothing at all.
    """

    from scipy import ndimage

    finite = np.asarray(image, dtype=float)
    median = float(np.median(finite))
    quartile = float(np.quantile(finite, 0.25))
    provisional = max(1.4826 * (median - quartile), np.finfo(float).tiny)
    # Three sigma of the provisional estimate: generous, because the point is
    # to REMOVE sources rather than to detect them, and an over-generous mask
    # only costs background pixels there are plenty of.
    sources = finite >= median + 3.0 * provisional
    if sources.any():
        # Dilated by the BAND-PASS's reach, not by the spot's: a source in a
        # band-passed picture is a peak with a dark ring around it, and the
        # ring is as fixed as the peak.
        sources = ndimage.binary_dilation(
            sources, structure=np.ones((int(exclusion), int(exclusion)), dtype=bool)
        )
    background = finite[~sources]
    if background.size < max(16, finite.size // 20):
        background = finite.reshape(-1)
    level = float(np.median(background))
    # The LOWER half of what is left: masking the sources truncates the upper
    # tail, and a two-sided spread of a truncated sample understates the
    # scatter.  The lower half is untouched by sources and by masking alike.
    scatter = 1.4826 * (level - float(np.quantile(background, 0.25)))
    return (
        level,
        max(
            scatter,
            float(np.finfo(float).eps * max(1.0, float(np.max(np.abs(finite))))),
        ),
        sources,
    )


def _seen_in_both_halves(
    half_hits: Sequence[np.ndarray],
    half_response: Sequence[np.ndarray],
    half_frames: Sequence[int],
    *,
    source_footprint: int,
) -> np.ndarray:
    """Where both halves of the run agree that something is there.

    Each half is judged on its own terms -- its own background level and
    scatter -- and a place has to stand clear of them in both.  The bar per
    half is deliberately low: half a run is half the evidence, and the point
    is not to detect the trap twice over but to refuse the coincidence that
    happened once.

    A run that cannot be halved -- one frame, or an already-averaged picture
    handed in as one -- carries no evidence about repetition either way, so
    this says nothing about it rather than refusing everything.  What admits
    a site there is the thresholds alone, which is all such a picture can
    support.
    """

    from scipy import ndimage

    agreed = np.ones(np.asarray(half_hits[0]).shape, dtype=bool)
    if min(int(count) for count in half_frames) < 1:
        return agreed
    for hits_half, response_half, frames in zip(
        half_hits, half_response, half_frames
    ):
        average_half = np.asarray(response_half, dtype=float) / float(frames)
        level, scatter, _sources = _background_scatter(
            average_half, source_footprint
        )
        z_half = (average_half - level) / scatter
        # Either kind of evidence, at half strength: a sighting in this half,
        # or a place that stands three sigma clear of this half's background.
        agreed &= (np.asarray(hits_half) > 0) | (z_half >= _HALF_RUN_SIGMA)
    return agreed


#: How far clear of its own background half a run must put a place, when that
#: half saw no single-shot sighting of it.
_HALF_RUN_SIGMA = 3.0


def _one_hill(
    evidence: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    *,
    saddle_fraction: float,
) -> bool:
    """Whether two peaks are two bumps on ONE hill rather than two hills.

    Distance cannot answer this.  A lattice four pixels apart and one nine
    pixels apart are both perfectly resolvable, while a single wide spot can
    carry two integer maxima three pixels apart -- so any exclusion radius
    chosen for one is wrong for the other, and every radius this detector has
    had was wrong for something (four sigma merged real traps six pixels
    apart; two pixels left duplicates on one spot).

    What separates two traps is the VALLEY between them.  Walk the segment
    joining the two peaks and take the lowest evidence on it: two distinct
    traps have background between them, so the walk drops most of the way to
    nothing, while two maxima on one spot never leave the spot and the walk
    barely dips.  This is the persistence criterion, and it needs no length
    scale at all -- which is why it holds for any spacing and any spot width.
    """

    (row_a, column_a), (row_b, column_b) = first, second
    peak = min(float(evidence[row_a, column_a]), float(evidence[row_b, column_b]))
    if peak <= 0.0:
        return True
    steps = max(3, int(np.ceil(np.hypot(row_b - row_a, column_b - column_a))) * 2)
    rows = np.linspace(row_a, row_b, steps)
    columns = np.linspace(column_a, column_b, steps)
    walk = ndimage.map_coordinates(
        evidence, np.vstack((rows, columns)), order=1, mode="nearest"
    )
    return float(np.min(walk)) >= saddle_fraction * peak


def _refine_center_subpixel(image: np.ndarray, x: float, y: float, half: int = 2) -> tuple[float, float]:
    """Where the spot centred on this peak actually is, or the peak itself.

    A peak is an integer pixel; the trap is somewhere inside it.  A fit tells
    you where -- WHEN it converges on the spot it was given, and a fit is only
    worth its answer under both conditions:

    * it succeeded.  The solver's own verdict used to be discarded, so a fit
      that diverged handed back whatever it had reached and that became a
      published site coordinate.
    * it stayed home.  A trap cannot be more than a pixel from its own peak --
      that is what being the peak means -- so a larger move means the fit
      walked onto a neighbour.  On a lattice whose rows are five pixels apart,
      a window that reaches two pixels already sees a neighbour's flank, and
      the answer it gives is that neighbour's.

    Failing either, the integer peak is the honest answer: it is accurate to
    half a pixel and it is certainly this spot.
    """

    height, width = image.shape
    x_int, y_int = int(round(x)), int(round(y))
    half = max(1, int(half))
    x0, x1 = max(0, x_int - half), min(width, x_int + half + 1)
    y0, y1 = max(0, y_int - half), min(height, y_int + half + 1)
    cut = image[y0:y1, x0:x1]
    if cut.size < 9 or not np.isfinite(cut).any():
        return float(x_int), float(y_int)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    background = float(np.nanmedian(cut))
    amplitude = float(np.nanmax(cut) - background)
    x_fit, y_fit, _sigma_x, _sigma_y, ok = _fit_gaussian_spot_2d(
        cut,
        yy,
        xx,
        x0=float(x_int),
        y0=float(y_int),
        offset0=background,
        amplitude=amplitude,
    )
    if (
        ok
        and np.isfinite(x_fit)
        and np.isfinite(y_fit)
        and abs(x_fit - x_int) <= 1.0
        and abs(y_fit - y_int) <= 1.0
    ):
        return float(x_fit), float(y_fit)
    return _core_centroid(image, x_int, y_int)


def _brightest_within(
    image: np.ndarray,
    row: int,
    column: int,
    reach: int = 1,
) -> tuple[int, int]:
    """The brightest pixel within a pixel or two of a candidate peak."""

    height, width = image.shape
    y0, y1 = max(0, row - reach), min(height, row + reach + 1)
    x0, x1 = max(0, column - reach), min(width, column + reach + 1)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=float)
    if not patch.size or not np.isfinite(patch).any():
        return int(row), int(column)
    offset_y, offset_x = np.unravel_index(int(np.nanargmax(patch)), patch.shape)
    return int(y0 + offset_y), int(x0 + offset_x)


def _core_centroid(image: np.ndarray, x_int: int, y_int: int) -> tuple[float, float]:
    """The spot's centre of light over its own core, used when a fit will not.

    A rejected fit does not make the integer peak the best answer available --
    it only makes the FIT unavailable.  The first moment of the three pixels
    round the peak, with the patch's own floor taken off, is unbiased for a
    symmetric spot and costs nothing; measured against it, the integer peak it
    replaces was up to half a pixel out on a tenth of a real array's sites.

    Bounded to the peak's own pixel for the same reason the fit is: the centre
    of THIS spot is inside it, and anything further is another spot's light.
    """

    height, width = image.shape
    y0, y1 = max(0, y_int - 1), min(height, y_int + 2)
    x0, x1 = max(0, x_int - 1), min(width, x_int + 2)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=float)
    if patch.size < 4 or not np.isfinite(patch).all():
        return float(x_int), float(y_int)
    weights = patch - float(np.min(patch))
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(x_int), float(y_int)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    x_centre = float(np.sum(weights * xx) / total)
    y_centre = float(np.sum(weights * yy) / total)
    if abs(x_centre - x_int) > 1.0 or abs(y_centre - y_int) > 1.0:
        return float(x_int), float(y_int)
    return x_centre, y_centre


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


#: How far the evidence must fall between two peaks for them to be two traps.
#: A Gaussian pair resolvable at all dips below half its peaks between them --
#: that is what "resolvable" means -- so half is the threshold that admits
#: exactly the pairs an eye would call two spots.
_SADDLE_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class _RunEvidence:
    """Everything one pass over the frames can say, before it means anything."""

    hits: np.ndarray
    lit_response: np.ndarray
    total_response: np.ndarray
    half_hits: list
    half_response: list
    half_frames: list
    background_sigma: float


def _accumulate_run(
    stack: np.ndarray,
    *,
    spot_sigma: float,
    detection_sigma: float,
) -> _RunEvidence:
    """Read the run once, keeping every quantity a later step could want.

    One pass, because it is one pass: a run of two thousand large frames is
    not walked again for the sake of tidiness.  What it keeps is where an atom
    was SEEN, the light on the shots where it was, the light on every shot,
    and all of that again for each interleaved half of the run.
    """

    from scipy import ndimage


    background_sigma = max(4.0 * spot_sigma, spot_sigma + 2.0)
    hits = np.zeros(stack.shape[1:], dtype=np.int64)
    lit_response = np.zeros(stack.shape[1:], dtype=float)
    #: Every shot added up, lit or not: what a trap too dim to clear the
    #: per-shot cut still writes into the picture.
    total_response = np.zeros(stack.shape[1:], dtype=float)
    #: The same two quantities over each interleaved half of the run.
    half_hits = [np.zeros(stack.shape[1:], dtype=np.int64) for _ in range(2)]
    half_response = [np.zeros(stack.shape[1:], dtype=float) for _ in range(2)]
    half_frames = [0, 0]
    # Frames are filtered many at a time, in blocks sized by memory rather than
    # by frame count: one SciPy call over a block costs a fraction of one call
    # per frame, and filtering makes two more copies of whatever it is given,
    # which for a run of 2048-square frames must not be the whole run.
    per_frame_bytes = max(1, int(np.prod(stack.shape[1:]))) * 8
    block = max(1, min(int(stack.shape[0]), int(64_000_000 // per_frame_bytes)))
    for start in range(0, int(stack.shape[0]), block):
        frames = np.asarray(stack[start : start + block], dtype=float)
        response = ndimage.gaussian_filter(
            frames, sigma=(0.0, spot_sigma, spot_sigma)
        ) - ndimage.gaussian_filter(
            frames, sigma=(0.0, background_sigma, background_sigma)
        )
        # Each frame is judged against its own noise: an exposure that came out
        # dim, or a run whose background drifted, changes what "bright" means
        # in that frame and in no other.
        # The noise of a frame is read from its lower half, where sites are
        # not: the median distance below the median.  That is the quarter
        # quantile of the whole frame, which is one call per block rather than
        # a masked median per frame.
        baseline, lower_quartile = np.quantile(
            response, (0.5, 0.25), axis=(1, 2), keepdims=True
        )
        noise = 1.4826 * (baseline - lower_quartile)
        noise = np.maximum(
            noise,
            np.finfo(float).eps
            * np.maximum(1.0, np.max(np.abs(response), axis=(1, 2), keepdims=True)),
        )
        lit = response >= baseline + detection_sigma * noise
        hits += np.count_nonzero(lit, axis=0)
        lit_response += np.sum(np.where(lit, response - baseline, 0.0), axis=0)
        total_response += np.sum(response - baseline, axis=0)
        # The run, also kept as two interleaved halves.  A trap is in both of
        # them; a noise peak is in one.  Interleaved rather than split in the
        # middle so a drift over the run cannot land in one half alone.
        odd = (start + np.arange(frames.shape[0])) % 2 == 1
        for half, mask in ((0, ~odd), (1, odd)):
            if not mask.any():
                continue
            half_hits[half] += np.count_nonzero(lit[mask], axis=0)
            half_response[half] += np.sum(
                (response - baseline)[mask], axis=0
            )
            half_frames[half] += int(np.count_nonzero(mask))
    return _RunEvidence(
        hits,
        lit_response,
        total_response,
        half_hits,
        half_response,
        half_frames,
        background_sigma,
    )


@dataclass(frozen=True, slots=True)
class _Admission:
    """How high a place must stand, in each of the two statistics."""

    shots: int
    pixels: int
    average: np.ndarray
    source_footprint: int
    sources: np.ndarray
    false_rate: float
    required: int
    average_z: np.ndarray
    average_cut: float


def _admission_thresholds(
    stack: np.ndarray,
    evidence: _RunEvidence,
    *,
    detection_sigma: float,
) -> _Admission:
    """The bars, measured on this run's own background rather than assumed.

    A band-passed picture breaks assumed distributions in the direction that
    invents traps, so what noise reaches here is counted where the sources
    are not.
    """

    from scipy.special import erfc, erfcinv
    from scipy.stats import binom

    hits = evidence.hits
    total_response = evidence.total_response
    background_sigma = evidence.background_sigma


    shots = int(stack.shape[0])
    pixels = int(hits.size)

    # The second admission: the run's average, judged the same way.  Its
    # noise is read the same robust way, and the significance a place must
    # reach is set by the same rule the sighting count uses -- an image of
    # pure background must not be expected to produce even half a site
    # anywhere in it.  Averaging is what makes this sensitive to a trap whose
    # single shots are unremarkable: N shots of a persistent signal add up as
    # N while their noise adds up as sqrt(N).
    average = total_response / float(shots)
    # The average's noise, measured on the average itself.  Photon noise over
    # the root of the frame count is what a perfect sensor would leave; fixed
    # pattern, warm pixels and background structure survive averaging
    # unchanged, and they are what a real picture's scatter is made of.
    # A source's footprint in a band-passed picture is the background kernel's
    # reach, peak and dark ring together.  Every background estimate in this
    # function excludes that much, or it is measuring the sources.
    source_footprint = 2 * int(np.ceil(background_sigma)) + 1
    average_baseline, average_noise, sources = _background_scatter(
        average, source_footprint
    )
    background = ~sources

    # How often noise alone clears the per-frame cut: counted where the
    # sources are not.  A band-passed frame's pixels are correlated, so its
    # robust scatter understates the real spread and the Gaussian tail of the
    # requested sigma understates this rate; that tail stays as a floor, so a
    # quiet stretch of background cannot license a lower bar than the physics.
    background_pixels = int(np.count_nonzero(background))
    observed_rate = (
        float(np.sum(hits[background])) / float(shots * background_pixels)
        if background_pixels
        else 0.0
    )
    false_rate = max(
        observed_rate,
        float(0.5 * erfc(detection_sigma / sqrt(2.0))),
        1e-12,
    )
    expected_false_sites = 0.5
    required = 1
    for count in range(1, shots + 1):
        if pixels * float(binom.sf(count - 1, shots, false_rate)) < expected_false_sites:
            required = count
            break
    else:
        required = shots

    average_z = (average - average_baseline) / average_noise
    # How high a place must stand in the average: the per-pixel Gaussian tail
    # for an image of this many pixels.  A floor rather than a guarantee --
    # the map is band-passed, so its local maxima come from a heavier
    # distribution than any single pixel of it.  False positives are refused
    # by reproducibility across the run's two halves, below.
    average_cut = float(sqrt(2.0) * erfcinv(1.0 / pixels))
    return _Admission(
        shots,
        pixels,
        average,
        source_footprint,
        sources,
        false_rate,
        required,
        average_z,
        average_cut,
    )


def _candidate_peaks(
    evidence: _RunEvidence,
    admission: _Admission,
    *,
    spot_sigma: float,
    peak_window: int,
    measurement_radius: int,
) -> tuple[list, np.ndarray]:
    """Every place that both admissions offer and both halves of the run keep.

    Returns the candidates strongest first, and the sighting significance the
    published quality is read from.
    """

    from scipy import ndimage

    hits = evidence.hits
    lit_response = evidence.lit_response
    half_hits = evidence.half_hits
    half_response = evidence.half_response
    half_frames = evidence.half_frames
    shots = admission.shots
    average = admission.average
    average_z = admission.average_z
    average_cut = admission.average_cut
    required = admission.required
    false_rate = admission.false_rate
    source_footprint = admission.source_footprint


    conditional = lit_response / np.maximum(hits, 1)
    # A candidate must be a peak of BOTH maps, over the same distance.
    #
    # The count alone is not enough: in every frame where two neighbouring
    # traps are both loaded, the pixels between them are lifted by both, clear
    # that frame's cut, and collect a sighting.  Those sightings are as real
    # as any other and they accumulate into a genuine local maximum of the
    # count, sitting in the gap.  What the gap cannot do is be as bright as
    # its neighbours WHEN LIT -- it is their tails, and they are their peaks.
    # Conditional brightness is also loading-independent, so requiring it
    # costs a dim trap nothing: a trap loaded a twentieth of the time is just
    # as bright on the shots where it is loaded.
    local_maxima = hits == ndimage.maximum_filter(
        hits, size=peak_window, mode="nearest"
    )
    brightest = conditional == ndimage.maximum_filter(
        conditional, size=peak_window, mode="nearest"
    )
    # Adjacent, not identical: noise puts the count's peak and the
    # brightness's peak on neighbouring pixels of the same spot often enough
    # that demanding one pixel lost real sites -- measured, two of thirty-eight
    # on a dense lattice, each with a brightness peak one pixel from its count
    # peak.  A spot is one place; a gap between two traps is four or more
    # pixels from either of their peaks and is excluded either way.
    local_maxima &= ndimage.binary_dilation(brightest, structure=np.ones((3, 3)))
    # A site has to be a place the picture actually shows, and a place that
    # can be MEASURED: within a spot's reach of the border there is no
    # background to compare against -- the filters invent one by extending the
    # edge, which biases every frame the same way -- and within the readout's
    # own box of the border there is no room to read it.  ``measurement_radius``
    # is that box, told by whoever will read it; the two reaches are the same
    # kind of fact and take the larger.
    margin = max(2, int(np.ceil(2.0 * spot_sigma)), int(measurement_radius))
    inside = np.zeros(hits.shape, dtype=bool)
    if 2 * margin < min(hits.shape):
        inside[margin:-margin, margin:-margin] = True
    # A place gets in either way.  The average path needs its own peak test --
    # a gap between two traps is a maximum of neither map -- and needs no
    # conditional-brightness check, because a place with no sightings has no
    # conditional brightness to speak of.
    spread = sqrt(max(shots * false_rate * (1.0 - false_rate), np.finfo(float).tiny))
    count_z = (hits - shots * false_rate) / spread
    persistent = average == ndimage.maximum_filter(
        average, size=peak_window, mode="nearest"
    )
    counted = local_maxima & (hits >= required)
    averaged = persistent & (average_z >= average_cut)
    # A place is a trap only if BOTH halves of the run show it.  This is the
    # one test that assumes nothing about a distribution: a trap is in the
    # first half of the run and in the second; a coincidence is not.
    repeated = _seen_in_both_halves(
        half_hits,
        half_response,
        half_frames,
        source_footprint=source_footprint,
    )
    candidates = np.argwhere((counted | averaged) & inside & repeated)
    ranked = sorted(
        candidates,
        key=lambda item: (
            -max(float(count_z[tuple(item)]), float(average_z[tuple(item)])),
            int(item[0]),
            int(item[1]),
        ),
    )
    return ranked, count_z


def _place_candidates(
    ranked: list,
    *,
    average: np.ndarray,
    average_z: np.ndarray,
    frame_shape: tuple,
    spot_sigma: float,
    measurement_radius: int,
) -> tuple[list, list]:
    """Where each admitted place is, and one answer per trap.

    Both questions are settled on the average -- the only map linear in the
    light -- and neither is settled by a distance.
    """


    refine_half = max(2, int(np.ceil(2.0 * spot_sigma)))
    selected: list[tuple[int, int]] = []
    centers_list: list[np.ndarray] = []
    dropped_at_border = 0
    for row, column in ranked:
        # Refined on the AVERAGE, whichever statistic admitted the place.
        # Loading scales a spot; it does not move it, so the map that uses
        # every photon is the one that says where the trap is.  (Conditional
        # brightness -- the light on the shots where a pixel was lit -- is a
        # biased picture of the shape: a skirt pixel is counted only on the
        # shots where it happened to be brightest.)
        # Starting from the average's OWN maximum: admission says a trap is
        # here, and where it is is answered on one map from end to end.  A
        # sighting count saturates, so its argmax can sit a pixel off the
        # light, and a fit windowed there sees a lopsided, truncated spot.
        row, column = _brightest_within(average, int(row), int(column))
        centre = _refine_center_subpixel(
            average, float(column), float(row), half=refine_half
        )
        # Checked on the PUBLISHED centre: refinement moves it by up to a
        # pixel, so the margin its integer peak passed is not the margin it
        # keeps.  What cannot be measured is not a site.
        if measurement_radius and not box_fits(
            (float(centre[0]), float(centre[1])), int(measurement_radius), frame_shape
        ):
            dropped_at_border += 1
            continue
        if all(
            not _one_hill(
                average_z,
                (int(row), int(column)),
                other_peak,
                saddle_fraction=_SADDLE_FRACTION,
            )
            for other_peak in selected
        ):
            selected.append((int(row), int(column)))
            centers_list.append(centre)
    if not selected:
        raise ValueError(
            "calibration frames contain no detectable sites"
            + (
                f" ({dropped_at_border} candidate(s) were too close to the "
                "border to be measured)"
                if dropped_at_border
                else ""
            )
        )
    return selected, centers_list


def detect_sites(
    frames: object,
    *,
    spot_sigma: float = 1.0,
    detection_sigma: float = 4.0,
    measurement_radius: int = 0,
) -> SiteMap:
    """Find the traps in a run of frames, and say where each one is.

    A trap announces itself in two independent ways, and a lattice contains
    both kinds at once, so the detector admits either:

    * IN ONE SHOT.  A loaded shot is unmistakable against that frame's own
      noise, so each frame becomes a map of where an atom was seen and the
      maps are added up.  This is what finds a trap that loads rarely: a
      hundred sightings and eight are the same evidence, differing only in
      loading, and eight is far more than a run's noise puts at one pixel.
    * IN THE AVERAGE.  A trap too dim for any single shot to clear the cut
      leaves no sightings at all, however often it loads, and is still plain
      in the average of the run.

    Neither alone is enough: an average-only cut holds the well-loaded traps
    and loses the rare ones; a sightings-only cut holds the bright ones and
    loses the dim ones.

    Every threshold either admission uses is measured on this run's own
    background rather than assumed from a distribution, and a place is
    admitted only if BOTH interleaved halves of the run show it -- a trap is
    in the first half and in the second, and a coincidence is not.

    ``spot_sigma`` is the optics: how wide one trap's image is.
    ``detection_sigma`` is how sure a single sighting must be.
    ``measurement_radius`` is the box the readout will later ask for round
    each site, which a published site must be able to carry.
    """

    stack = np.asarray(
        frames.values if hasattr(frames, "values") else frames, dtype=float
    )
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3 or 0 in stack.shape or not np.isfinite(stack).all():
        raise ValueError("frames must be a non-empty finite stack of 2D images")
    spot_sigma = float(spot_sigma)
    detection_sigma = float(detection_sigma)
    if spot_sigma <= 0 or not np.isfinite(spot_sigma):
        raise ValueError("spot_sigma must be positive and finite")
    if detection_sigma <= 0 or not np.isfinite(detection_sigma):
        raise ValueError("detection_sigma must be positive and finite")
    # The scale over which a spot is ONE peak: its full width at half
    # maximum, 2.355 sigma, rounded up to an odd pixel count.  The only length
    # in the detector -- whether two peaks are two traps is decided by the
    # light between them, not by a distance.
    peak_window = max(3, 2 * int(np.ceil(1.1775 * spot_sigma)) + 1)
    # How near in y two sites must be to be READ as one row of the array.
    # Ordering only: it decides the order site ids are handed out in, never
    # which places are sites.
    row_tolerance = max(2.0, float(spot_sigma))

    evidence = _accumulate_run(
        stack, spot_sigma=spot_sigma, detection_sigma=detection_sigma
    )
    admission = _admission_thresholds(
        stack, evidence, detection_sigma=detection_sigma
    )
    ranked, count_z = _candidate_peaks(
        evidence,
        admission,
        spot_sigma=spot_sigma,
        peak_window=peak_window,
        measurement_radius=int(measurement_radius),
    )
    selected, centers_list = _place_candidates(
        ranked,
        average=admission.average,
        average_z=admission.average_z,
        frame_shape=evidence.hits.shape,
        spot_sigma=spot_sigma,
        measurement_radius=int(measurement_radius),
    )
    average_z = admission.average_z

    centers = np.asarray(centers_list, dtype="<f8")
    # How far a site stands out of the noise, in its own sigmas -- by whichever
    # evidence admitted it.  Reporting only the count would call a trap that is
    # certain in the average a marginal one.
    quality = np.asarray(
        [
            max(float(count_z[row, column]), float(average_z[row, column]))
            for row, column in selected
        ],
        dtype="<f8",
    )
    order = _stable_site_order(centers, row_tolerance)
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


@dataclass(frozen=True)
class _MeasuredSpots:
    """What an atom does to the picture, per site, and what follows from it."""

    #: The weighting each site's readout applies, over its own box.
    weights: np.ndarray
    #: The one weighting shared by every site, for the uniform model.
    uniform: np.ndarray
    boxes: np.ndarray
    #: The difference an atom makes, over the box AND the ring around it.
    templates: np.ndarray
    #: Of all the light an atom adds, the share that lands inside the box.
    box_light_fraction: np.ndarray
    fit_centers: np.ndarray
    fit_sigmas: np.ndarray
    fit_ok: np.ndarray


def _measure_readout_weights(
    references: np.ndarray,
    centers_xy: np.ndarray,
    labels_occupied: np.ndarray,
    labels_valid: np.ndarray,
    train: np.ndarray,
    *,
    radius: int,
    padding: int,
    fallback_sigma: float,
) -> _MeasuredSpots:
    """The weighting a readout should apply, MEASURED on this run.

    What a readout has to detect is the difference an atom makes: the average
    frame where the site was loaded, minus the average frame where it was
    not.  That difference is the atom's own image and nothing else -- the
    pedestal, the fixed pattern, the neighbours' skirts, every fixed thing in
    the picture cancels -- and weighting each pixel by it is the matched
    filter, which is the best any linear readout can do against white noise.

    Measured on the LONG frames, where an atom is unmistakable and there are
    two per cycle, and on the TRAIN half of the run only, so the fidelity
    reported on the held-out half is a statement about a readout that never
    saw those shots.

    What this replaces was a shape ASSUMED rather than measured: a Gaussian
    fitted to the reference AVERAGE -- loaded and empty shots together, over
    a pedestal, with the neighbours' skirts in it -- smoothed, clipped at
    zero, and normalised.  Measured on the bench's own 35-trap run, held out:
    6.46 separations for the box, 7.27 for that kernel, 7.53 for this one.

    There is also no per-shot background subtraction here, and that is not an
    omission.  An annulus in a DENSE lattice is not background: it holds the
    neighbouring traps, whose loading changes shot to shot, so subtracting it
    injects their noise into this site's answer.  Measured on the same run:
    6.45 without it, 6.29 subtracting the ring's median, 5.97 its mean.  A
    background level common to the whole frame is absorbed by the threshold,
    which is measured in the same counts as the signal.
    """

    radius = int(radius)
    padding = int(padding)
    if radius < 0 or padding <= 0:
        raise ValueError("PSF radius must be non-negative and padding must be positive")
    centers = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    frames = references.reshape(-1, *references.shape[2:])
    per_cycle = int(references.shape[1])
    picked = np.asarray(train, dtype=bool) & np.asarray(labels_valid, dtype=bool)
    occupied = np.asarray(labels_occupied, dtype=bool)

    outer = radius + padding
    size = 2 * radius + 1
    core = slice(padding, padding + size)
    boxes: list[tuple[int, int, int, int]] = []
    templates: list[np.ndarray] = []
    measured: list[bool] = []
    for index, center in enumerate(centers):
        boxes.append(_box_bounds(tuple(center), radius, frames.shape[-2:]))
        window = _box_bounds(tuple(center), outer, frames.shape[-2:])
        x, y, width, height = window
        lit = np.repeat(picked[:, index] & occupied[:, index], per_cycle)
        dark = np.repeat(picked[:, index] & ~occupied[:, index], per_cycle)
        if (
            int(lit.sum()) < 2
            or int(dark.sum()) < 2
            or (height, width) != (2 * outer + 1, 2 * outer + 1)
        ):
            templates.append(np.zeros((2 * outer + 1, 2 * outer + 1)))
            measured.append(False)
            continue
        cut = frames[:, y : y + height, x : x + width]
        templates.append(
            np.asarray(cut[lit], dtype=float).mean(axis=0)
            - np.asarray(cut[dark], dtype=float).mean(axis=0)
        )
        measured.append(True)

    template_stack = np.stack(templates, axis=0)
    weights = np.array(template_stack[:, core, core], dtype="<f8")
    totals = weights.sum(axis=(1, 2))
    usable = np.asarray(measured, dtype=bool) & (totals > 0.0)
    # A site whose own atoms were never seen enough to measure a shape gets
    # the shape the OTHER sites agreed on -- one optic images them all -- and
    # a run where no site could be measured falls back to the spot size
    # detection was told to look for.
    normalised = np.zeros_like(weights)
    normalised[usable] = weights[usable] / totals[usable, np.newaxis, np.newaxis]
    if np.any(usable):
        uniform = normalised[usable].mean(axis=0)
        uniform = uniform / float(np.sum(uniform))
    else:
        uniform = np.asarray(gaussian_psf_kernel(float(fallback_sigma), radius))
    normalised[~usable] = uniform

    window_light = template_stack.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(window_light > 0.0, totals / window_light, np.nan)

    fit_centers: list[tuple[float, float]] = []
    fit_sigmas: list[tuple[float, float]] = []
    fit_ok: list[bool] = []
    for index, center in enumerate(centers):
        template = template_stack[index]
        if not usable[index]:
            fit_centers.append((float(center[0]), float(center[1])))
            fit_sigmas.append((float(fallback_sigma), float(fallback_sigma)))
            fit_ok.append(False)
            continue
        half = (template.shape[0] - 1) // 2
        anchor_x = int(round(float(center[0]))) - half
        anchor_y = int(round(float(center[1]))) - half
        yy, xx = np.mgrid[
            anchor_y : anchor_y + template.shape[0],
            anchor_x : anchor_x + template.shape[1],
        ]
        x_fit, y_fit, sigma_x, sigma_y, ok = _fit_gaussian_spot_2d(
            template,
            yy,
            xx,
            x0=float(center[0]),
            y0=float(center[1]),
            offset0=0.0,
            amplitude=float(np.nanmax(template)),
            sigma0=float(fallback_sigma),
        )
        fit_centers.append((x_fit, y_fit))
        fit_sigmas.append((sigma_x, sigma_y))
        fit_ok.append(bool(ok))

    return _MeasuredSpots(
        np.ascontiguousarray(normalised, dtype="<f8"),
        np.ascontiguousarray(uniform, dtype="<f8"),
        np.asarray(boxes, dtype="<i8"),
        np.ascontiguousarray(template_stack, dtype="<f8"),
        np.asarray(fraction, dtype="<f8"),
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
        detection_sigma=detection_sigma,
        # Every box this calibration will read out of a site: the integration
        # box, and the PSF box with the background ring round it.  A place that
        # cannot carry all of them is not a site THIS calibration can use.
        measurement_radius=max(box_half_width, psf_half_width + psf_padding),
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

    spots = _measure_readout_weights(
        references,
        centers,
        labels_occupied,
        labels_valid,
        train,
        radius=psf_half_width,
        padding=psf_padding,
        fallback_sigma=detection_spot_sigma,
    )
    per_site_weights = spots.weights
    psf_boxes = spots.boxes
    psf_fit_centers = spots.fit_centers
    psf_fit_sigmas = spots.fit_sigmas
    psf_fit_ok = spots.fit_ok
    uniform_weights = np.broadcast_to(
        spots.uniform, per_site_weights.shape
    ).copy()

    def psf_extractor(weights: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda frame: extract_psf_signals(
            frame,
            centers,
            kernels=weights,
            boxes_xywh=psf_boxes,
            background="none",
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
                "background": "none",
                "psf_padding": psf_padding,
            },
            {
                "psf_fit_centers_xy": psf_fit_centers,
                "psf_fit_sigma_xy": psf_fit_sigmas,
                "psf_fit_ok": psf_fit_ok,
                "psf_boxes_xywh": psf_boxes,
                "psf_kernels": per_site_weights,
                "psf_templates": spots.templates,
                "psf_box_light_fraction": spots.box_light_fraction,
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
                "background": "none",
                "psf_padding": psf_padding,
            },
            {
                "psf_fit_centers_xy": psf_fit_centers,
                "psf_fit_sigma_xy": psf_fit_sigmas,
                "psf_fit_ok": psf_fit_ok,
                "psf_boxes_xywh": psf_boxes,
                "uniform_kernel": spots.uniform,
                "psf_templates": spots.templates,
                "psf_box_light_fraction": spots.box_light_fraction,
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
    "extract_box_signals",
    "extract_psf_signals",
    "readout_model_kind_from_choice",
]
