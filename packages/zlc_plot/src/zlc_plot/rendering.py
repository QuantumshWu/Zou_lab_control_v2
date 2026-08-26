"""Persistent Matplotlib artists for every public ZLC plot kind.

This module owns no GUI toolkit and never calls ``pyplot``.  A surface can be
attached to Agg or a Qt5 canvas.  Data/display edits mutate persistent
artists; fixed-size changes rebuild layout within the same Figure.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
from numbers import Real
from pathlib import Path
import re
from enum import Enum
from time import perf_counter
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ._image_raster import ImageFrontStore, PreparedImageFront
from ._fit_scene import FitOverlay, FitPolyline
from .data_view import aligned_histogram_edges
from ._kinds import handler_for
from ._rendering.pulse import update_pulse_timeline
from ._pulse_time import pulse_time_scale
from ._selector_scene import (
    ColorLimitCandidate,
    ColorLimitState,
    SceneKind,
    SelectorItemContext,
    SelectorLine,
    SelectorMarkers,
    SelectorPrimitive,
    SelectorScene,
    SelectorSceneOwner,
    SelectorSceneKind,
    SelectorSceneStyle,
    SelectorTarget,
)
from .layout import SurfacePlan, facet_focus_box, fitted_facet_cell_title
from .parameters import RenderEffect
from .primitives import ImagePointOverlay, PointStatus, PulseTimelineData
from .selectors import (
    CrosshairPoint,
    NumericRange,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
)
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSpec,
    PulseTimelinePlot,
    RollingPlot,
    semantic_spec,
)
from .state import DisplayState
from .style import PlotStyleConfig, style_context
from .ticks import apply_declared_ticks, apply_smart_ticks
from ._validation import readonly_copy


@dataclass(frozen=True, slots=True)
class _PreparedSeries:
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    label: str
    identity: tuple[tuple[str, str | None, str], ...]
    #: Display names for the x positions of a labelled categorical axis.
    x_labels: tuple[str, ...] | None = None
    #: (low, high) display-unit bounds of the standard-error band, or None.
    #: Bounds, not sigma: converting y+/-sem keeps affine display units
    #: honest where converting a difference would not be.
    band: tuple[np.ndarray, np.ndarray] | None = None


def _display_array(value: Any) -> np.ndarray:
    raw = getattr(value, "display", value)
    return np.asarray(raw)


def _series_band(item: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Display-unit y+/-sem bounds of one projected series, if it has sem."""

    sem = getattr(item, "sem", None)
    if sem is None:
        return None
    quantity = item.y
    canonical = np.asarray(quantity.canonical, dtype=float).reshape(-1)
    spread = np.asarray(sem, dtype=float).reshape(-1)
    unit = quantity.canonical_unit
    low = unit.convert_value_to(canonical - spread, quantity.display_unit)
    high = unit.convert_value_to(canonical + spread, quantity.display_unit)
    return (
        np.asarray(low, dtype=float).reshape(-1),
        np.asarray(high, dtype=float).reshape(-1),
    )


def _valid_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    raw = getattr(value, "valid", None)
    if raw is None:
        return np.ones(shape, dtype=bool)
    return np.asarray(raw, dtype=bool)


def _unit_symbol(value: Any) -> str:
    unit = getattr(value, "display_unit", None)
    return "" if unit is None else str(getattr(unit, "symbol", unit))


def _series_identity(item: Any) -> tuple[tuple[str, str | None, str], ...]:
    return tuple((value.ref.domain.value, value.ref.axis_id, repr(value.canonical))
                 for value in getattr(item, "group_key", ()))


def _series_slot(identity: object, count: int) -> int:
    digest = hashlib.blake2s(repr(identity).encode(), digest_size=2).digest()
    return 0 if not identity else int.from_bytes(digest, "big") % count


_EXPLICIT_UNIT_SUFFIX = re.compile(r"(?:\[[^\[\]]+\]|\([^()]+\))\s*$")
def _quantity_label(value: Any, fallback: str, explicit: str | None = None) -> str:
    if explicit is not None:
        label = str(explicit)
        if not label:
            return ""
    else:
        label = str(getattr(value, "label", fallback) or fallback)
    unit = _unit_symbol(value)
    if not unit or unit == "1":
        return label
    if explicit is not None and _EXPLICIT_UNIT_SUFFIX.search(label):
        return label
    suffix = f"({unit})"
    return label if label.rstrip().endswith(suffix) else f"{label} {suffix}"


def _state_label(
    state: DisplayState,
    name: str,
    fallback: str | None,
) -> str | None:
    value = state.values.get(name)
    return fallback if value is None else str(value)


def _curve_x_limits(values: np.ndarray) -> tuple[float, float] | None:
    """Return the authored curve coordinate span without autoscale padding."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low == high:
        half_span = max(0.5, 0.05 * abs(low))
        return low - half_span, high + half_span
    return low, high


def _data_limits(values: np.ndarray) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _scalar_close(left: float, right: float) -> bool:
    """The same closeness np.allclose(rtol=1e-12, atol=1e-15) answers,
    without the per-call array wrapping."""

    return abs(left - right) <= 1e-15 + 1e-12 * abs(right)


def _relim_retains(mode: str) -> bool:
    """Whether an autoscaled axis may keep the limits it already shows.

    THE answer to that question, for every autoscaled axis there is.  TIGHT
    means tight: the picture takes the data's range, every time, so a step
    the operator can see moves the scale.  The other modes ask for a steady
    view and hold what they have until the data leaves it -- that hysteresis
    is why a scale, and all the chrome keyed to it, does not flicker from
    one shot to the next.  FIXED never reaches here at all: its limits are
    authored, so nothing is autoscaled to retain.

    The colour scale and the count axis each carried a copy of this sentence
    and disagreed about it for as long as both existed: the count axis
    re-fitted on tight, the colour scale sat behind a 35% deadband, and an
    operator who chose tight watched an image's scale ignore a peak that had
    moved by a fifth.
    """

    return mode != "tight"


def _autoscaled_limits(
    data_range: tuple[float, float],
    current: tuple[float, float] | None,
    *,
    padding_fraction: float,
    deadband_fraction: float,
    zero_based: bool = False,
    retain: bool = True,
) -> tuple[float, float]:
    """The limits one autoscaled axis takes for this data.

    Two questions, and they are not the same question: ``zero_based`` is the
    SHAPE the axis wants -- a count read against zero, anything else against
    its own span -- and ``retain`` is whether the limits already on screen
    may be kept.  A single ``mode`` argument answered both, so the colour
    rail, which passed "tight" to ask for the padded shape, gave up its
    hysteresis in the same breath without ever asking to.
    """

    low, high = map(float, data_range)
    if zero_based and low >= 0.0:
        target = (0.0, high * 1.2 if high else 1.0)
    else:
        data_span = (high - low) or (abs(high) or 1.0)
        padding = padding_fraction * data_span
        target = low - padding, high + padding
    if current is None or not retain:
        return target
    current_low, current_high = map(float, current)
    span = current_high - current_low
    if span <= 0.0:
        return target
    # Retention is decided against the view ON SCREEN, whatever shape
    # derived it.  Restoring the zero-anchored shape whenever the data
    # happened to be non-negative made an axis whose minimum wanders
    # around zero flip between two limit shapes forever -- and every flip
    # re-captured all the chrome keyed to the scale.  The zero-anchored
    # keep-rule applies exactly when the current view IS zero-anchored.
    clips = low < current_low or high > current_high
    if zero_based and low >= 0.0 and current_low == 0.0:
        too_empty = not (0.7 * current_high <= high) or current_high <= 0.0
    else:
        too_empty = (
            high < current_high - deadband_fraction * span
            or low > current_low + deadband_fraction * span
        )
    return target if clips or too_empty else (current_low, current_high)


def _select_display_limits(
    mode: str,
    automatic: tuple[float, float] | None,
    state: DisplayState,
    quantity: str,
    *,
    allow_partial: bool = False,
) -> tuple[float, float]:
    low_name, high_name = f"{quantity}_min", f"{quantity}_max"
    authored_low, authored_high = state[low_name], state[high_name]
    if mode == "fixed" and (authored_low is None or authored_high is None):
        raise RuntimeError("committed fixed limits are incomplete")
    if mode != "fixed" and not allow_partial:
        return tuple(map(float, automatic or (0.0, 1.0)))
    base = (0.0, 1.0) if automatic is None else automatic
    low = base[0] if authored_low is None else float(authored_low)
    high = base[1] if authored_high is None else float(authored_high)
    if low >= high:
        if authored_low is not None and authored_high is not None:
            raise RuntimeError("committed display limits are not increasing")
        span = max(abs(low), abs(high), 1.0) * 1e-12
        low, high = (low, low + span) if authored_low is not None else (
            high - span,
            high,
        )
    return float(low), float(high)


def _image_data_range(
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float] | None:
    """Return the exact finite range without promoting a megapixel image."""

    values = np.asarray(values)
    validity = np.broadcast_to(np.asarray(valid, dtype=np.bool_), values.shape)
    all_valid = bool(np.all(validity))
    if values.dtype.kind in "biu":
        if not all_valid and not bool(np.any(validity)):
            return None
        if all_valid:
            native_low = np.min(values)
            native_high = np.max(values)
        elif values.dtype.kind == "b":
            native_low = np.min(values, where=validity, initial=True)
            native_high = np.max(values, where=validity, initial=False)
        else:
            info = np.iinfo(values.dtype)
            native_low = np.min(values, where=validity, initial=info.max)
            native_high = np.max(values, where=validity, initial=info.min)
        low, high = float(native_low), float(native_high)
        if native_low != native_high and low == high:
            raise TypeError(
                "integer image range is not distinguishable in float display space"
            )
        return low, high

    # Keep temporary finite masks bounded.  This matters for float camera
    # frames whose validity is sparse, while the common all-valid path still
    # scans contiguous source chunks without selecting/copying the image.
    low = math.inf
    high = -math.inf
    chunk_size = 262_144
    operands = (values,) if all_valid else (values, validity)
    iterator = np.nditer(
        operands,
        flags=("external_loop", "buffered", "zerosize_ok"),
        op_flags=tuple(("readonly",) for _operand in operands),
        order="K",
        buffersize=chunk_size,
    )
    for chunks in iterator:
        if all_valid:
            chunk = np.asarray(chunks)
            valid_chunk = None
        else:
            value_chunk, mask_chunk = chunks
            chunk = np.asarray(value_chunk)
            valid_chunk = np.asarray(mask_chunk, dtype=np.bool_)
        finite = np.isfinite(chunk)
        if valid_chunk is not None:
            np.logical_and(finite, valid_chunk, out=finite)
        if not bool(np.any(finite)):
            continue
        low = min(low, float(np.min(chunk, where=finite, initial=np.inf)))
        high = max(high, float(np.max(chunk, where=finite, initial=-np.inf)))
    return (low, high) if math.isfinite(low) and math.isfinite(high) else None


def _image_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z)
    if z.ndim != 2 or z.dtype.kind not in "biuf":
        raise TypeError("image values must be a real numeric two-dimensional array")
    if z.shape != (y.size, x.size):
        raise ValueError("image values must match the y/x coordinate grid")
    validity = (
        np.broadcast_to(np.asarray(True, dtype=np.bool_), z.shape)
        if valid is None
        else np.asarray(valid, dtype=np.bool_)
    )
    try:
        validity = np.broadcast_to(validity, z.shape)
    except ValueError as error:
        raise ValueError("image validity cannot broadcast to its values") from error
    x0, x1, y0, y1 = _centers_extent(x, y)
    return z, validity, (x0, x1, y1, y0)


def _bounded_image_distribution_values(
    values: np.ndarray,
    valid: np.ndarray,
    sample_target: int,
) -> np.ndarray:
    values = np.asarray(values)
    valid = np.broadcast_to(np.asarray(valid, dtype=bool), values.shape)
    order = "F" if values.flags.f_contiguous and not values.flags.c_contiguous else "C"
    flat_values = np.ravel(values, order=order)
    flat_valid = np.ravel(valid, order=order)
    if flat_values.size > sample_target:
        step = flat_values.size // sample_target + 1
        sampled = flat_values[::step]
        sampled_valid = flat_valid[::step]
        if sampled.dtype.kind == "f":
            sampled_valid = sampled_valid & np.isfinite(sampled)
        if bool(np.any(sampled_valid)):
            return np.asarray(sampled[sampled_valid], dtype=float)
    full_valid = flat_valid
    if flat_values.dtype.kind == "f":
        full_valid = full_valid & np.isfinite(flat_values)
    if bool(np.all(full_valid)):
        return np.asarray(flat_values, dtype=float)
    return np.asarray(flat_values[full_valid], dtype=float)


def _point_ring_radius(
    points: np.ndarray,
    *,
    fraction: float,
    fallback: float,
) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or not bool(np.all(np.isfinite(points))):
        return float(fallback)
    from scipy.spatial import cKDTree

    distances, _indices = cKDTree(points).query(points, k=2)
    nearest = np.asarray(distances[:, 1], dtype=float)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if nearest.size == 0:
        return float(fallback)
    return float(fraction) * float(np.median(nearest))


def _centers_extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    def edge(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=float).reshape(-1)
        if not values.size or not bool(np.all(np.isfinite(values))):
            raise ValueError("image coordinates must be non-empty and finite")
        if values.size == 1:
            return (float(values[0] - 0.5), float(values[0] + 0.5))
        steps = np.diff(values)
        if not (bool(np.all(steps > 0.0)) or bool(np.all(steps < 0.0))):
            raise ValueError("image coordinates must be strictly monotonic")
        return (
            float(values[0] - steps[0] / 2.0),
            float(values[-1] + steps[-1] / 2.0),
        )

    x0, x1 = edge(x)
    y0, y1 = edge(y)
    return (x0, x1, y0, y1)


def _square_image_limits(
    extent: tuple[float, float, float, float],
    *,
    coordinate_aspect: float = 1.0,
) -> tuple[float, float, float, float]:
    """Pad the shorter physical span without stretching image pixels."""

    left, right, upper, lower = map(float, extent)
    x_span = abs(right - left)
    y_span = abs(lower - upper) * float(coordinate_aspect)
    if (
        x_span <= 0.0
        or y_span <= 0.0
        or not math.isfinite(float(coordinate_aspect))
        or float(coordinate_aspect) <= 0.0
    ):
        raise ValueError("image extent spans must be positive")
    if x_span > y_span:
        padding = (x_span - y_span) / (2.0 * float(coordinate_aspect))
        direction = 1.0 if lower > upper else -1.0
        upper -= direction * padding
        lower += direction * padding
    elif y_span > x_span:
        padding = (y_span - x_span) / 2.0
        direction = 1.0 if right > left else -1.0
        left -= direction * padding
        right += direction * padding
    return left, right, upper, lower


def _image_coordinate_aspect(x: Any, y: Any) -> float | None:
    """Return canonical y/x scale, or ``None`` for unrelated quantities.

    An image whose axes represent different physical dimensions has no
    meaningful isotropic aspect.  Treating that case as ``1`` silently pads
    one axis and changes the authored geometry; the renderer must leave it in
    Matplotlib's normal ``auto`` mode instead.
    """

    x_unit = getattr(x, "display_unit", None)
    y_unit = getattr(y, "display_unit", None)
    if (
        x_unit is None
        or y_unit is None
        or not x_unit.compatible_with(y_unit)
    ):
        return None
    return abs(float(y_unit.scale) / float(x_unit.scale))


def _histogram_vertices(edges: np.ndarray, counts: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float).reshape(-1)
    counts = np.asarray(counts, dtype=float).reshape(-1)
    if edges.size != counts.size + 1:
        raise ValueError("histogram edges must contain one more value than counts")
    vertices = np.empty((counts.size, 4, 2), dtype=float)
    vertices[:, 0] = np.column_stack((edges[:-1], np.zeros(counts.size)))
    vertices[:, 1] = np.column_stack((edges[:-1], counts))
    vertices[:, 2] = np.column_stack((edges[1:], counts))
    vertices[:, 3] = np.column_stack((edges[1:], np.zeros(counts.size)))
    return vertices


def _compact_engineering(value: float, *, length: int | None = None) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    numeric = float(value)

    def normalized(text: str) -> str:
        if "e" not in text.lower():
            return text
        mantissa, exponent = text.lower().split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"

    if length is None:
        general = f"{numeric:.4g}"
        return general if "e" not in general.lower() else normalized(f"{numeric:.1e}")
    maximum = max(1, int(length))
    for significant in range(4, 0, -1):
        general = normalized(f"{numeric:.{significant}g}")
        if len(general) <= maximum:
            return general
        scientific = normalized(f"{numeric:.{significant - 1}e}")
        if len(scientific) <= maximum:
            return scientific
    return normalized(f"{numeric:.0e}")


_ENVELOPE_MIN_POINTS_PER_COLUMN = 4
_ENVELOPE_MIN_COLUMNS = 64
_ENVELOPE_MAX_COLUMNS = 4096


def _envelope_decimated(
    x: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
    columns: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-column min/max envelope of a dense polyline, or None to draw raw.

    Past a few samples per pixel column a stroked polyline is visually
    defined by each column's extremes alone; the envelope hands Agg exactly
    those extremes (three vertices per column: minimum, maximum, and a
    separator that becomes NaN wherever the column holds an invalid sample,
    so gaps stay gaps at pixel resolution) instead of one vertex per sample.
    Sparse windows -- a deep zoom, a short trace -- return None and the
    caller draws every point, which keeps the picture exact where the
    envelope has nothing to save.  ``x`` must be finite and sorted; gaps
    travel in ``y`` as NaN.
    """

    low, high = float(window[0]), float(window[1])
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return None
    start = int(np.searchsorted(x, low, side="left"))
    stop = int(np.searchsorted(x, high, side="right"))
    if stop - start < columns * _ENVELOPE_MIN_POINTS_PER_COLUMN:
        return None
    x_view = x[start:stop]
    y_view = y[start:stop]
    edges = np.linspace(low, high, columns + 1)
    starts = np.searchsorted(x_view, edges[:-1], side="left")
    counts = np.diff(np.append(starts, x_view.size))
    finite = np.isfinite(y_view)
    guarded_min = np.where(finite, y_view, np.inf)
    guarded_max = np.where(finite, y_view, -np.inf)
    with np.errstate(invalid="ignore"):
        col_min = np.minimum.reduceat(guarded_min, np.minimum(starts, x_view.size - 1))
        col_max = np.maximum.reduceat(guarded_max, np.minimum(starts, x_view.size - 1))
        finite_counts = np.add.reduceat(
            finite.astype(np.int64), np.minimum(starts, x_view.size - 1)
        )
    empty = counts == 0
    finite_counts = np.where(empty, 0, finite_counts)
    gap = finite_counts < counts
    blank = finite_counts == 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    out_x = np.repeat(centers, 3)
    out_y = np.empty(columns * 3, dtype=np.float64)
    out_y[0::3] = np.where(blank, np.nan, col_min)
    out_y[1::3] = np.where(blank, np.nan, col_max)
    # The separator repeats the maximum (a zero-length segment) except where
    # the column carried an invalid sample: there it breaks the stroke.
    out_y[2::3] = np.where(gap, np.nan, out_y[1::3])
    # One raw neighbour on each side keeps the entering/leaving segment at
    # the window edge instead of clipping it a column early.
    prefix_x = x[start - 1 : start]
    prefix_y = y[start - 1 : start]
    suffix_x = x[stop : stop + 1]
    suffix_y = y[stop : stop + 1]
    return (
        np.concatenate((prefix_x, out_x, suffix_x)),
        np.concatenate((prefix_y, out_y, suffix_y)),
    )


def _isolated_curve_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Valid curve vertices with no valid neighbour on either side."""

    finite = np.isfinite(x) & np.isfinite(y)
    connected = np.zeros(finite.shape, dtype=bool)
    if finite.size > 1:
        adjacent = finite[:-1] & finite[1:]
        connected[:-1] |= adjacent
        connected[1:] |= adjacent
    return finite & ~connected


def _bounded_isolated_curve_points(
    x: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Visible singleton glyphs, bounded to two per display column."""

    low, high = sorted(map(float, window))
    start = max(0, int(np.searchsorted(x, low, side="left")) - 1)
    stop = min(x.size, int(np.searchsorted(x, high, side="right")) + 1)
    visible_x = x[start:stop]
    visible_y = y[start:stop]
    isolated = _isolated_curve_mask(visible_x, visible_y)
    if not bool(np.any(isolated)):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    isolated &= (visible_x >= low) & (visible_x <= high)
    isolated_x = visible_x[isolated]
    isolated_y = visible_y[isolated]
    if isolated_x.size <= 2 * columns or not high > low:
        return isolated_x, isolated_y

    bins = np.floor((isolated_x - low) * columns / (high - low)).astype(np.int64)
    np.clip(bins, 0, columns - 1, out=bins)
    starts = np.flatnonzero(np.concatenate(([True], bins[1:] != bins[:-1])))
    occupied = bins[starts]
    minimum = np.minimum.reduceat(isolated_y, starts)
    maximum = np.maximum.reduceat(isolated_y, starts)
    centers = low + (occupied.astype(float) + 0.5) * (high - low) / columns
    bounded_x = np.repeat(centers, 2)
    bounded_y = np.column_stack((minimum, maximum)).reshape(-1)
    keep = np.ones(bounded_y.shape, dtype=bool)
    keep[1::2] = maximum != minimum
    return bounded_x[keep], bounded_y[keep]


def _facet_cell_title(cell: Any, fallback: int) -> str:
    """Use the projected facet identity with a compact numeric coordinate."""

    label = getattr(cell, "label", None)
    if label is None:
        label = str(getattr(cell, "facet_value_display", fallback))
    title = str(label)
    value = getattr(cell, "facet_value_display", None)
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            raw = str(value)
            if raw in title:
                title = title.replace(raw, _compact_engineering(numeric), 1)
    return title


def _fit_parameter_value_text(parameter: Any) -> str:
    """Format one fitted value for renderer-owned text."""

    return _compact_engineering(float(parameter.value))


class _FitAnnotationDetail(str, Enum):
    NONE = "none"
    HEADLINE = "headline"
    FULL = "full"


_FIT_DIAGNOSTIC_SINGLE_MAX_CHARS = 72
_FIT_DIAGNOSTIC_FACET_MAX_CHARS = 24


def _truncate_fit_diagnostic(message: str, maximum: int) -> str:
    """Keep fit diagnostics bounded so they cannot change the layout."""

    text = str(message).strip() or "fit failed"
    if len(text) <= maximum:
        return text
    return text[: maximum - 3] + "..."


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """Complete immutable input for one renderer presentation transaction."""

    payload: Any
    state: DisplayState
    effects: RenderEffect
    data_revision: int | None = None
    fit_overlays: tuple[FitOverlay, ...] = ()
    fit_model_id: str | None = None
    classifier_overlays: tuple[FitOverlay, ...] = ()
    classifier_thresholds: tuple[float | None, ...] = ()
    classifier_labels: tuple[str, ...] = ()
    image_overlay: ImagePointOverlay | None = None
    selectors: SelectorSnapshot = SelectorSnapshot(())
    facet_index: int | None = None
    facet_focus_index: int | None = None
    view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        overlays = tuple(self.classifier_overlays)
        thresholds = tuple(self.classifier_thresholds)
        labels = tuple(map(str, self.classifier_labels))
        if len(overlays) != len(thresholds) or len(labels) != len(thresholds):
            raise ValueError("classifier overlays, thresholds, and labels must align")
        if any(
            value is not None and not math.isfinite(float(value))
            for value in thresholds
        ):
            raise ValueError("classifier thresholds must be finite or None")
        object.__setattr__(self, "classifier_overlays", overlays)
        object.__setattr__(self, "classifier_thresholds", thresholds)
        object.__setattr__(self, "classifier_labels", labels)


class MatplotlibRenderer:
    """One fixed-layout Figure with persistent artists and selector overlays."""

    def __init__(
        self,
        spec: PlotSpec,
        plan: SurfacePlan,
        *,
        style: PlotStyleConfig,
    ) -> None:
        if not isinstance(plan, SurfacePlan):
            raise TypeError("plan must be SurfacePlan")
        if not isinstance(style, PlotStyleConfig):
            raise TypeError("style must be PlotStyleConfig")
        self.spec = spec
        self.plan = plan
        self.style = style
        self._figure: Any = None
        self._axes: dict[str, list[Any]] = {}
        self._artists: dict[str, Any] = {}
        self._selector_artists: dict[SceneKind, tuple[Any, ...]] = {}
        self._selector_topologies: dict[SceneKind, tuple[object, ...]] = {}
        self._selector_candidate: SelectorState | None = None
        self._color_limit_candidate: ColorLimitCandidate | None = None
        self._height_bars_calls: dict[str, tuple] = {}
        self._selector_gesture_kind: SceneKind | None = None
        self._last_selectors = SelectorSnapshot(())
        self._fit_artists: list[Any] = []
        self._fit_slots: dict[str, Any] = {}
        self._facet_fit_topologies: dict[
            int,
            tuple[Any, str, str | None, dict[str, Any], tuple[Any, ...]],
        ] = {}
        self._last_fit_overlay: FitOverlay | None = None
        self._last_fit_overlays: tuple[FitOverlay, ...] = ()
        self._classifier_artists: dict[int, tuple[tuple[Any, ...], Any, Any]] = {}
        self._classifier_labels: tuple[str, ...] = ()
        self._fit_source_scatter: Any | None = None
        self._fit_hidden_source_lines: tuple[tuple[Any, bool], ...] = ()
        self._fit_axis: Any | None = None
        self._fit_family: str | None = None
        self._fit_model_id: str | None = None
        self._data_revision: int | None = None
        #: Cached Agg chrome region (everything except renderer-owned dynamic
        #: artists) and the canvas signature it was captured for.  Payload-only
        #: presents restore it and repaint just the dynamic artists; any
        #: chrome-dirty axes, layout/text/chrome effect or canvas change forces
        #: a fresh full draw and recapture, so a stale background can never be
        #: published.
        self._background_region: Any = None
        self._background_signature: tuple[object, ...] | None = None
        #: The same cut, one artist deeper: everything a compose paints BELOW
        #: the gesture's own artists, captured while a gesture is in flight.
        #: A pointer move then costs a region restore and the handful of
        #: artists above the cut, instead of the whole figure -- measured on a
        #: 512x512 live image panel, 0.06 ms against 5.4 ms.  Without it a
        #: drag is composed like a data frame, and interaction latency is the
        #: whole scene's cost plus whatever frame is being composed.
        self._gesture_region: Any = None
        self._gesture_overlay: tuple[tuple[tuple[int, float, int], Any], ...] = ()
        self._gesture_selector_ids: frozenset[int] = frozenset()
        #: Per painted surface: the finite range of the image drawn on it, and
        #: the arrays it was measured from.  The range is a function of the
        #: data alone -- it cannot change when the viewport does -- yet it was
        #: rescanned on every pan step and every wheel tick: 0.85 ms over a
        #: 1920x1200 frame, at the pointer-motion rate.  The source arrays are
        #: held so a freed array cannot have its id recycled into a stale hit.
        self._image_ranges: dict[
            str,
            tuple[tuple[int, int, int | None], tuple[float, float] | None, tuple[object, object]],
        ] = {}
        self._last_payload: Any = None
        self._last_state: DisplayState | None = None
        self._home_limits: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._requested_view_limits: tuple[
            tuple[float, float], tuple[float, float]
        ] | None = None
        self._chrome_dirty_axes: set[Any] = set()
        #: Per-axes boundary chrome (ticks, gridlines, spines) collected for
        #: the dynamic compose, cached under the same two facts the chrome
        #: itself depends on: the axes' view limits (chrome-dirty tracking)
        #: and the canvas signature (size/DPR).  ``_update_ticks`` runs the
        #: locator and formatter, and a 35-cell facet paid for seventy such
        #: runs per steady frame in which nothing had moved.
        self._boundary_chrome_cache: dict[
            int, tuple[tuple[Any, Any, float], ...]
        ] = {}
        self._boundary_chrome_signature: tuple[object, ...] | None = None
        #: Canonical raw data behind each displayed line.  The artist may hold
        #: a display-resolution envelope, but fit-source presentation and any
        #: future redraw read this one source truth.  Artist retirement and
        #: relayout retire the matching entry in the same owner.
        self._line_sources: dict[
            int, tuple[Any, Any, np.ndarray, np.ndarray, bool]
        ] = {}
        self._series_lines: dict[int, tuple[tuple[Any, object, str], ...]] = {}
        #: Error-bar artists BY SERIES, parallel to _series_lines: the bars
        #: are part of the series, so focus dims and raises them with their
        #: line instead of leaving them behind at full strength.
        self._series_bars: dict[int, dict[object, tuple[Any, ...]]] = {}
        #: Pixel-space polylines for hover hit tests, per line, keyed by the
        #: view/canvas signature.  Transforming every point of every series
        #: on every motion event was the hover lag; the data only changes on
        #: a series mutation, which clears this.
        self._series_hit_cache: dict[int, tuple[tuple, tuple]] = {}
        self._series_indices: dict[int, dict[object, int]] = {}
        self._series_hover: tuple[int, object, str, float, float] | None = None
        self._series_locked: tuple[int, object, str, float, float] | None = None
        self._series_press: tuple[float, float, object | None] | None = None
        self._series_annotations: dict[int, Any] = {}
        self._raster_generation = 0
        #: The generation whose pixels currently sit in the Agg buffer,
        #: or -1 while artist state has mutated since the last compose.
        #: ``rgba`` reuses the buffer when they match instead of paying
        #: a full draw -- which also nuked the compose background and
        #: forced the NEXT frame to rebuild it: one stray full draw per
        #: publication, twice over.
        self._composed_generation = -1
        self._focused_facet_index: int | None = None
        self._facet_focus_index: int | None = None
        self._height_bars_dragging = False
        self._height_bars_rendered_camera = None
        self._height_bars_scene = False
        self._visible_facet_count = 0
        #: Which cell currently owns the focused image side chrome (the
        #: distribution/colorbar axes over ``plan.facet_focus_axes``), or
        #: None while no cell does.  A per-presentation fact, never stored
        #: on the axes: the chrome is destroyed with its axes on unfocus.
        self._facet_focus_chrome_index: int | None = None
        self._compose_figure()

    @property
    def figure(self) -> Any:
        return self._figure

    @property
    def axes(self) -> Mapping[str, tuple[Any, ...]]:
        return MappingProxyType({key: tuple(value) for key, value in self._axes.items()})

    @property
    def semantic_spec(self) -> PlotSpec:
        """The spec that decides what this renderer DRAWS on one surface.

        A FacetGrid is a layout; each of its cells draws the cell spec.  Every
        per-plot facility gates on this, never on the outer spec: re-typing it
        by hand is what left colour-limit dragging, the point overlay and the
        crosshair value rail working for a standalone image and dead inside a
        facet cell of the very same picture.
        """

        return semantic_spec(self.spec)

    @property
    def painted_surfaces(self) -> tuple[tuple[str, Any, int | None], ...]:
        """Every data surface this presentation paints: (artist key, axes, cell).

        One plot paints ONE surface under its kind's name; a FacetGrid paints
        its visible cells under ``facet:<i>``, or only the focused cell.  This
        is the single answer to "what is on screen right now": the render
        dispatch, the requested view, the home limits, the point overlay and
        every image gesture read it instead of re-deriving their own.
        """

        if self.semantic_spec is self.spec:
            return (self._whole_surface(),)
        cells = self._axes.get("facet_cell", ())
        if self._facet_focus_index is not None:
            index = self._facet_focus_index
            return ((f"facet:{index}", cells[index], index),)
        return tuple(
            (f"facet:{index}", cells[index], index)
            for index in range(min(self._visible_facet_count, len(cells)))
        )

    def _whole_surface(self) -> tuple[str, Any, None]:
        """The single-surface answer: this kind's artist key and its axes."""

        key = handler_for(self.spec).kind.value
        for role in ("main", "history", "image", "facet_cell"):
            values = self._axes.get(role)
            if values:
                return (key, values[0], None)
        return (key, next(iter(self._axes.values()))[0], None)

    @property
    def primary_surface(self) -> tuple[str, Any, int | None]:
        """The one painted surface a pointer gesture and its chrome act on.

        The ONE spelling of "which cell is active".  Two spellings is how the
        colour-limit preview edited a different artist than the one the rail
        was measured from.
        """

        surfaces = self.painted_surfaces
        if not surfaces:
            # A grid with no cells paints nothing, yet the pointer and chrome
            # questions still need an axes to resolve against.
            return self._whole_surface()
        index = self._focused_facet_index or 0
        return surfaces[min(index, len(surfaces) - 1)]

    @property
    def primary_axes(self) -> Any:
        return self.primary_surface[1]

    def facet_index_for_axes(self, axes: Any) -> int | None:
        if not isinstance(self.spec, FacetGridPlot):
            return None
        for index, candidate in enumerate(self._axes.get("facet_cell", ())):
            if axes is candidate and index < self._visible_facet_count:
                return index
        return None

    def interactive_axes_at(self, event: Any) -> Any | None:
        """Resolve pointer ownership from visible geometry, not backend hints."""

        candidate = getattr(event, "inaxes", None)
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        visible = tuple(axis for axis in self._figure.axes if axis.get_visible())
        if x is None or y is None:
            return candidate if candidate in visible else None
        point = (float(x), float(y))
        if candidate in visible and bool(candidate.bbox.contains(*point)):
            return candidate
        # Focused Facet axes overlap the hidden overview geometry exactly.
        # Resolve against current visibility and put the semantic primary axes
        # first so a stale browser-canvas ``inaxes`` value cannot steal the event.
        ordered = (self.primary_axes, *reversed(visible))
        for axis in dict.fromkeys(ordered):
            if axis in visible and bool(axis.bbox.contains(*point)):
                return axis
        return None

    def _consume_facet_presentation(
        self,
        *,
        facet_index: int | None,
        facet_focus_index: int | None,
        visible_count: int | None = None,
    ) -> None:
        """Mirror the session-owned facet state for the current render pass."""

        for value, name in (
            (facet_index, "facet_index"),
            (facet_focus_index, "facet_focus_index"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        if not isinstance(self.spec, FacetGridPlot):
            if facet_index is not None or facet_focus_index is not None:
                raise TypeError("facet presentation state requires FacetGridPlot")
            self._focused_facet_index = None
            self._facet_focus_index = None
            self._visible_facet_count = 0
            return
        count = (
            len(self._axes.get("facet_cell", ()))
            if visible_count is None
            else visible_count
        )
        if count < 0 or count > len(self._axes.get("facet_cell", ())):
            raise ValueError("visible facet count is outside the rendered grid")
        if facet_index is None:
            if facet_focus_index is not None:
                raise ValueError("an open facet requires a selected facet")
            if visible_count is not None and count:
                raise ValueError("a non-empty FacetGrid requires a selected facet")
        else:
            if facet_index < 0 or facet_index >= count:
                raise IndexError("facet index is outside the current grid")
            if facet_focus_index is not None and facet_focus_index != facet_index:
                raise ValueError("the open facet must equal the selected facet")
        self._focused_facet_index = facet_index
        self._facet_focus_index = facet_focus_index
        self._visible_facet_count = count

    def _compose_figure(self) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        with style_context(
            self.style,
            {
                "figure.dpi": self.plan.logical_dpi,
                "figure.figsize": self.plan.figure_size_inches,
            },
        ):
            figure = Figure(
                figsize=self.plan.figure_size_inches,
                dpi=self.plan.logical_dpi,
                layout=None,
            )
            FigureCanvasAgg(figure)
            # Matplotlib native canvases derive physical DPI from this logical
            # baseline.  Materialise the Agg front at the requested screen DPR
            # without allowing a later frontend canvas to multiply it again.
            figure._original_dpi = self.plan.logical_dpi
            figure._set_dpi(self.plan.dpi, forward=False)
            self._figure = figure
            self._axes = self._create_axes(figure, self.plan)

    @staticmethod
    def _create_axes(figure: Any, plan: SurfacePlan) -> dict[str, list[Any]]:
        axes: dict[str, list[Any]] = {}
        for axes_plan in plan.axes:
            axis = figure.add_axes(axes_plan.box.matplotlib_bounds())
            axis.set_gid(
                f"{axes_plan.role}:{axes_plan.cell_index}"
                if axes_plan.cell_index is not None
                else axes_plan.role
            )
            axes.setdefault(axes_plan.role, []).append(axis)
        return axes

    def relayout(
        self,
        plan: SurfacePlan,
        *,
        facet_index: int | None = None,
        facet_focus_index: int | None = None,
    ) -> None:
        """Rebuild a named layout while preserving the attached canvas model."""

        if not isinstance(plan, SurfacePlan):
            raise TypeError("plan must be SurfacePlan")
        figure = self._figure
        with style_context(
            self.style,
            {
                "figure.dpi": plan.logical_dpi,
                "figure.figsize": plan.figure_size_inches,
            },
        ):
            figure.clear()
            figure.set_size_inches(plan.figure_size_inches, forward=False)
            figure._original_dpi = plan.logical_dpi
            figure._set_dpi(plan.dpi, forward=False)
            self.plan = plan
            self._axes = self._create_axes(figure, plan)
        self._artists.clear()
        self._line_sources.clear()
        self._series_lines.clear(); self._series_indices.clear(); self._series_annotations.clear()
        self._series_bars.clear()
        self._series_hit_cache.clear()
        self._series_hover = self._series_locked = self._series_press = None
        self._boundary_chrome_cache.clear()
        self._boundary_chrome_signature = None
        self._selector_artists.clear()
        self._selector_topologies.clear()
        self._forget_gesture_region()
        self._selector_candidate = None
        self._color_limit_candidate = None
        self._last_selectors = SelectorSnapshot(())
        self._fit_artists.clear()
        self._fit_slots.clear()
        self._facet_fit_topologies.clear()
        self._last_fit_overlay = None
        self._last_fit_overlays = ()
        self._classifier_artists.clear()
        self._classifier_labels = ()
        self._fit_source_scatter = None
        self._fit_hidden_source_lines = ()
        self._fit_axis = None
        self._fit_family = None
        self._fit_model_id = None
        self._image_ranges.clear()
        self._data_revision = None
        self._last_payload = None
        self._last_state = None
        self._home_limits.clear()
        self._requested_view_limits = None
        self._chrome_dirty_axes.clear()
        self._facet_focus_chrome_index = None
        self._consume_facet_presentation(
            facet_index=facet_index,
            facet_focus_index=facet_focus_index,
        )

    def present(self, frame: RenderFrame) -> None:
        """Mutate all layers and publish exactly one complete canvas front."""

        if not isinstance(frame, RenderFrame):
            raise TypeError("frame must be RenderFrame")
        payload = frame.payload
        state = frame.state
        selected_effects = frame.effects
        self._composed_generation = -1
        previous_payload = self._last_payload
        previous_data_revision = self._data_revision
        state_changed = self._last_state is None or state.revision != self._last_state.revision
        payload_changed = (
            payload is not previous_payload
            or frame.data_revision != previous_data_revision
        )
        visible_facet_count = (
            len(tuple(getattr(payload, "cells", ())))
            if isinstance(self.spec, FacetGridPlot)
            else None
        )
        self._consume_facet_presentation(
            facet_index=frame.facet_index,
            facet_focus_index=frame.facet_focus_index,
            visible_count=visible_facet_count,
        )
        base_effects = (
            RenderEffect.VIEW_PROJECTION
            | RenderEffect.PAYLOAD_PROJECTION
            | RenderEffect.BASE_GEOMETRY
            | RenderEffect.LAYOUT
        )
        base_changed = payload_changed or bool(selected_effects & base_effects)
        if (
            state_changed
            and selected_effects & RenderEffect.AXIS_TRANSFORM
        ):
            base_changed = True
        if (
            selected_effects & RenderEffect.AXIS_TRANSFORM
            and isinstance(self.semantic_spec, ImagePlot)
        ):
            # The display front is viewport-sized, so a view edit changes its
            # crop even though the immutable source payload is unchanged.
            base_changed = True
        style_only = (
            not base_changed
            and state_changed
            and bool(selected_effects & RenderEffect.BASE_STYLE)
        )
        overview = isinstance(self.spec, FacetGridPlot) and self._facet_focus_index is None
        painted_selectors = SelectorSnapshot(()) if overview else frame.selectors
        painted_fit_overlays = tuple(frame.fit_overlays)
        if not overview:
            painted_fit_overlays = self._focused_fit_overlays(painted_fit_overlays)
        with style_context(self.style):
            self._requested_view_limits = frame.view_limits
            self._last_payload = payload
            self._last_state = state
            self._data_revision = frame.data_revision
            self._last_fit_overlays = painted_fit_overlays
            self._last_fit_overlay = (
                painted_fit_overlays[0] if len(painted_fit_overlays) == 1 else None
            )
            painted = self.painted_surfaces
            if base_changed:
                self._update_plot(payload, state)
                for _key, axes, _index in painted:
                    self._capture_home_limits(axes)
            elif style_only:
                self._update_base_style(state)
            if state_changed and selected_effects & RenderEffect.TEXT:
                self._update_text_artists(payload, state)
            if state_changed and selected_effects & RenderEffect.CHROME:
                self._update_chrome_artists(state)
            # A height-bar scene owns its whole data region: 2D overlays --
            # selectors, fit polylines, classifier guides, the point
            # overlay -- speak data coordinates that do not exist on the
            # projected scene, so their artists hide while the COMMITTED
            # state stays in the session and returns with the heatmap.
            scene_3d = self._height_bars_active(
                self.primary_surface[0], state
            )
            self._height_bars_scene = scene_3d
            # Every painted surface honours the requested view, not just the
            # selected one: a FacetGrid overview shows N cells of the same
            # picture, and zooming one of them alone is not a view of anything.
            if not scene_3d:
                for _key, axes, _index in painted:
                    self._apply_requested_view(axes, frame.view_limits)
            self._classifier_labels = frame.classifier_labels
            self._update_classifier(
                () if scene_3d else frame.classifier_overlays,
                frame.classifier_thresholds,
                frame.classifier_labels,
            )
            self._last_selectors = (
                SelectorSnapshot(()) if scene_3d else painted_selectors
            )
            self._update_selectors(self._last_selectors)
            if scene_3d:
                self._update_height_bars_cage(painted_selectors)
            else:
                self._update_height_bars_cage(SelectorSnapshot(()))
            self._set_fit_mode(
                bool(painted_fit_overlays) and not overview and not scene_3d
            )
            self._update_fit(
                () if scene_3d else painted_fit_overlays,
                overview=overview,
                model_id=frame.fit_model_id,
            )
            cells = tuple(getattr(payload, "cells", ()))
            for key, axes, index in painted:
                cell = None if index is None else cells[index]
                self._update_image_point_overlay(
                    axes,
                    payload if cell is None else getattr(cell, "payload", cell),
                    None if scene_3d else frame.image_overlay,
                    state,
                    key,
                    None
                    if cell is None
                    else getattr(cell, "facet_value_canonical", None),
                )
            self._compose_frame(
                chrome_stable=not bool(
                    selected_effects
                    & (
                        RenderEffect.LAYOUT
                        | RenderEffect.TEXT
                        | RenderEffect.CHROME
                    )
                )
            )

    def _capture_home_limits(self, axis: Any) -> None:
        if isinstance(self.semantic_spec, ImagePlot) and id(axis) in self._home_limits:
            # Image mutation registers the full source extent before applying
            # the frame viewport.  Never replace it with a zoomed front.
            return
        self._home_limits[id(axis)] = (
            tuple(map(float, axis.get_xlim())),
            tuple(map(float, axis.get_ylim())),
        )

    def _apply_requested_view(
        self,
        axis: Any,
        requested: tuple[tuple[float, float], tuple[float, float]] | None,
    ) -> None:
        if requested is None:
            requested = self._home_limits.get(id(axis))
        if requested is None:
            return
        self._set_xlim(axis, *requested[0])
        self._set_ylim(axis, *requested[1])

    def _update_base_style(self, state: DisplayState) -> None:
        """Apply property-only edits without rebuilding data geometry."""

        # Plot-specific mapping edits are still in-place artist mutations; the
        # plot updater owns their one canonical implementation.
        self._update_plot(self._last_payload, state)
        for _key, axes, _index in self.painted_surfaces:
            self._capture_home_limits(axes)

    def _apply_line_data(
        self,
        axes: Any,
        line: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        isolated_glyphs: bool = False,
    ) -> None:
        """Hand a polyline to its artist, enveloped when denser than pixels."""

        self._line_sources[id(line)] = (line, axes, x, y, isolated_glyphs)
        self._set_enveloped_line(
            axes, line, x, y, isolated_glyphs=isolated_glyphs
        )

    def _set_enveloped_line(
        self,
        axes: Any,
        line: Any,
        x: np.ndarray,
        y: np.ndarray,
        *,
        isolated_glyphs: bool,
    ) -> None:
        columns = int(
            min(
                _ENVELOPE_MAX_COLUMNS,
                max(_ENVELOPE_MIN_COLUMNS, float(axes.bbox.width) * 2.0),
            )
        )
        window = tuple(map(float, axes.get_xlim()))
        enveloped = (
            _envelope_decimated(x, y, window, columns)
            if x.size >= columns * _ENVELOPE_MIN_POINTS_PER_COLUMN
            else None
        )
        drawn_x, drawn_y = (x, y) if enveloped is None else enveloped
        if isolated_glyphs:
            point_x, point_y = _bounded_isolated_curve_points(
                x, y, window, columns
            )
        else:
            point_x = point_y = np.asarray([], dtype=float)
        if point_x.size:
            extra_x = np.empty(point_x.size * 2, dtype=float)
            extra_y = np.empty(point_y.size * 2, dtype=float)
            extra_x[0::2] = np.nan
            extra_y[0::2] = np.nan
            extra_x[1::2] = point_x
            extra_y[1::2] = point_y
            marker_start = drawn_x.size + 1
            line.set_data(
                np.concatenate((drawn_x, extra_x)),
                np.concatenate((drawn_y, extra_y)),
            )
            line.set_marker("_")
            line.set_markersize(self.style.artists.curve_marker_size_pt)
            line.set_markeredgewidth(line.get_linewidth())
            line.set_markevery(
                np.arange(marker_start, marker_start + extra_x.size, 2)
            )
        else:
            line.set_data(drawn_x, drawn_y)
            marker = self.style.artists.curve.marker
            line.set_marker("None" if marker is None else marker)
            line.set_markersize(self.style.artists.curve_marker_size_pt)
            line.set_markevery(None)

    def _refresh_enveloped_lines(self, axis: Any) -> None:
        for line_id, (line, owner, x, y, isolated_glyphs) in tuple(
            self._line_sources.items()
        ):
            attached = getattr(line, "axes", None)
            if attached is None:
                del self._line_sources[line_id]
            elif owner is axis and attached is axis:
                self._set_enveloped_line(
                    axis,
                    line,
                    x,
                    y,
                    isolated_glyphs=isolated_glyphs,
                )

    def _set_xlim(self, axis: Any, low: float, high: float) -> None:
        # Scalar closeness: np.allclose costs ~25 us of array wrapping per
        # call, and a 64-cell grid asks this question hundreds of times a
        # frame.
        previous_low, previous_high = axis.get_xlim()
        if not (
            _scalar_close(float(previous_low), float(low))
            and _scalar_close(float(previous_high), float(high))
        ):
            axis.set_xlim(float(low), float(high))
            self._mark_axes_chrome_dirty(axis)
            self._refresh_enveloped_lines(axis)

    def _set_ylim(self, axis: Any, low: float, high: float) -> None:
        previous_low, previous_high = axis.get_ylim()
        if not (
            _scalar_close(float(previous_low), float(low))
            and _scalar_close(float(previous_high), float(high))
        ):
            axis.set_ylim(float(low), float(high))
            self._mark_axes_chrome_dirty(axis)

    def _mark_axes_chrome_dirty(self, *axes: Any) -> None:
        self._chrome_dirty_axes.update(axes)

    @property
    def raster_generation(self) -> int:
        """Monotonic count of composed Agg frames; ties publishes to pixels."""

        return self._raster_generation

    def draw(self) -> None:
        """Compose one complete Agg frame from the current artist state."""

        with style_context(self.style):
            self._native_draw(self._figure.canvas)
            self._chrome_dirty_axes.clear()
            # A direct full draw bakes dynamic artists into the buffer, so any
            # previously captured region is no longer current.
            self._background_region = None
            self._forget_gesture_region()
            self._raster_generation += 1
            self._composed_generation = self._raster_generation

    def _dynamic_artists(self) -> list[tuple[tuple[int, float, int], Any]]:
        """Every artist this renderer owns, keyed for full-draw-exact stacking.

        Axes-owned text chrome (tick labels, titles, axis labels, colorbar
        internals) is deliberately absent: it forms the cached background.
        Each entry's key is ``(figure axes index, effective zorder, insertion
        sequence)``: a full draw composes one axes at a time, sorts within it
        by zorder with child order breaking ties, and paints tick marks and
        grid lines when their owning Axis draws — at the Axis' zorder, not
        their own.
        """

        from matplotlib.artist import Artist

        figure_axes = list(self._figure.axes)
        axes_order = {id(axes): index for index, axes in enumerate(figure_axes)}
        fallback_order = len(figure_axes)
        collected: list[tuple[tuple[int, float, int], Any]] = []
        seen: set[int] = set()

        def keyed(artist: Any, owner: Any, zorder: float) -> None:
            if id(artist) in seen:
                return
            seen.add(id(artist))
            collected.append(
                (
                    (
                        axes_order.get(id(owner), fallback_order),
                        float(zorder),
                        len(collected),
                    ),
                    artist,
                )
            )

        def add(value: Any) -> None:
            if isinstance(value, dict):
                # The height-bar chrome keeps its artists in a dict; its
                # lines and TEXTS move with the camera, so they must be
                # dynamic -- baked into the background they could only
                # stay correct while something forced a full redraw
                # between frames.
                for item in value.values():
                    add(item)
                return
            if isinstance(value, (tuple, list, set)):
                for item in value:
                    add(item)
                return
            # Mirror the full-draw contract at the one collection point: a
            # full figure draw skips an INVISIBLE AXES together with
            # everything on it, and never draws an axes that is no longer in
            # the figure at all.  A focused FacetGrid is the layout that
            # hides axes -- collecting their artists here made every hidden
            # cell (and, through the tick loop below, its tick marks) ghost
            # into the focused frame at the old cell boxes; a REMOVED axes
            # (the focused image side chrome dies with its focus) reports
            # visible forever, so figure membership is part of the mirror.
            if (
                isinstance(value, Artist)
                and getattr(value, "axes", None) is not None
                and id(value.axes) in axes_order
                and value.axes.get_visible()
            ):
                keyed(value, value.axes, value.get_zorder())

        for value in self._artists.values():
            add(value)
        # Error bars are part of their series and their focus alpha moves
        # per hover: dynamic, or the change sits baked into the cached
        # background until the next full draw -- the line dimmed instantly
        # while its bars answered one publication later.
        for bars in self._series_bars.values():
            for artists in bars.values():
                add(artists)
        for values in self._selector_artists.values():
            add(values)
        add(self._fit_artists)
        for lines, threshold_line, label in self._classifier_artists.values():
            add(lines)
            add(threshold_line)
            add(label)
        if self._fit_source_scatter is not None:
            add(self._fit_source_scatter)
        for axes in figure_axes:
            add(axes.get_legend())
        # Boundary chrome draws ABOVE low-z data in a full draw, so it must
        # be composed beside the data rather than captured: tick children
        # carry ``axes=None`` in Matplotlib, hence ownership and stacking
        # position are supplied here.  Outside-the-box text (tick labels,
        # titles) stays in the background and is painted exactly once.
        from matplotlib.axes import Axes
        from matplotlib.axis import Axis

        dynamic_axis_ids = {
            id(artist)
            for _key, artist in collected
            if isinstance(artist, Axis)
        }
        dynamic_full_axes_ids = {
            id(artist)
            for _key, artist in collected
            if isinstance(artist, Axes)
        }
        canvas = self._figure.canvas
        signature = (
            id(canvas),
            int(round(float(self._figure.bbox.width))),
            int(round(float(self._figure.bbox.height))),
        )
        if signature != self._boundary_chrome_signature:
            self._boundary_chrome_cache.clear()
            self._boundary_chrome_signature = signature
        for axes in {entry[1].axes for entry in tuple(collected)}:
            if not axes.get_visible():
                continue
            if not getattr(axes, "axison", True):
                # ``set_axis_off`` (the height-bar scene) removes ticks and
                # spines from a full draw; composing their cached artists
                # anyway framed the 3D scene with 2D chrome.
                continue
            if id(axes) in dynamic_full_axes_ids:
                continue
            cached = (
                None
                if axes in self._chrome_dirty_axes
                else self._boundary_chrome_cache.get(id(axes))
            )
            if cached is not None:
                for artist, owner, zorder in cached:
                    keyed(artist, owner, zorder)
                continue
            entries: list[tuple[Any, Any, float]] = []
            for axis in (axes.xaxis, axes.yaxis):
                if id(axis) in dynamic_axis_ids:
                    continue
                axis_z = float(axis.get_zorder())
                # The same ticks a full Axis.draw would paint: positions
                # refreshed and clipped to the current view interval.  The
                # raw ``majorTicks`` list keeps stale instances parked at
                # out-of-view locations after a limit change, and painting
                # those leaks mark segments outside the axes box.  Tick
                # geometry is a function of the view limits and canvas size
                # alone, so the refresh runs when one of those moved (the
                # axes is chrome-dirty, or the signature above changed) and
                # the collected artists are replayed verbatim in between.
                # An INVISIBLE artist paints nothing in a full draw, and
                # visibility is as stable as the cached geometry itself:
                # dropping them here (gridlines are off everywhere, most
                # cells hide one tick side) removes a thousand no-op draw
                # calls per composed frame without touching a pixel.
                for tick in axis._update_ticks():
                    for artist in (
                        tick.gridline, tick.tick1line, tick.tick2line
                    ):
                        if artist.get_visible():
                            entries.append((artist, axes, axis_z))
            for spine in axes.spines.values():
                if spine.get_visible():
                    entries.append((spine, axes, spine.get_zorder()))
            self._boundary_chrome_cache[id(axes)] = tuple(entries)
            for artist, owner, zorder in entries:
                keyed(artist, owner, zorder)
        return collected

    def _compose_frame(self, *, chrome_stable: bool) -> None:
        """Compose one complete frame, reusing the cached chrome background.

        The published buffer is always complete: either a plain full draw, or
        the captured chrome region with every dynamic artist repainted in
        z-order.  Limit moves, text/chrome/layout effects and colorbar edits
        all mark axes chrome-dirty or drop ``chrome_stable``, which forces the
        capture path, so a stale background can never reach a front.
        """

        canvas = self._figure.canvas
        restore = getattr(canvas, "restore_region", None)
        capture = getattr(canvas, "copy_from_bbox", None)
        get_renderer = getattr(canvas, "get_renderer", None)
        if not (callable(restore) and callable(capture) and callable(get_renderer)):
            self.draw()
            return
        signature = (
            id(canvas),
            int(round(float(self._figure.bbox.width))),
            int(round(float(self._figure.bbox.height))),
        )
        reusable = (
            chrome_stable
            and self._background_region is not None
            and self._background_signature == signature
            and not self._chrome_dirty_axes
        )
        if not reusable:
            # Anything that invalidates the background (layout, text, chrome
            # effects, limit moves) may also have moved tick geometry through
            # a locator or formatter change, which the per-axes dirty set does
            # not see.  The boundary cache is only ever trusted between two
            # consecutive reusable frames.
            self._boundary_chrome_cache.clear()
        dynamics = self._dynamic_artists()
        ordered = sorted(dynamics, key=lambda entry: entry[0])
        # Where the gesture's own artists begin, in the one z-order a full
        # draw uses.  The frame below that point is captured on the way past,
        # so a pointer move repaints only the tail.  Splitting the SEQUENCE
        # rather than partitioning by ownership is what keeps the compose
        # full-draw-exact: anything that legitimately draws above a selector
        # stays above it, and is simply repainted with it.
        selector_ids = self._selector_artist_ids()
        split = None
        if self._selector_gesture_kind is not None and selector_ids:
            split = next(
                (
                    index
                    for index, (_key, artist) in enumerate(ordered)
                    if id(artist) in selector_ids
                ),
                None,
            )
        # Every owner call enters the renderer's style once around mutation
        # and compose.  Re-entering here copied the full rcParams mapping for
        # every frame without changing a property on any existing artist.
        if not reusable:
            visibility = [
                (artist, artist.get_visible()) for _key, artist in dynamics
            ]
            try:
                for artist, _visible in visibility:
                    artist.set_visible(False)
                self._native_draw(canvas)
            finally:
                for artist, visible in visibility:
                    artist.set_visible(visible)
            self._background_region = capture(self._figure.bbox)
            self._background_signature = signature
            self._chrome_dirty_axes.clear()
        restore(self._background_region)
        renderer = get_renderer()
        for index, (_key, artist) in enumerate(ordered):
            if index == split:
                self._gesture_region = capture(self._figure.bbox)
                self._gesture_overlay = tuple(ordered[split:])
                self._gesture_selector_ids = selector_ids
            if artist.get_visible():
                if not self._blit_exact_rgba_image(artist, canvas):
                    artist.draw(renderer)
        if split is None:
            self._forget_gesture_region()
        self._raster_generation += 1
        self._composed_generation = self._raster_generation

    def _selector_artist_ids(self) -> frozenset[int]:
        """Every artist the selector scene owns right now, by identity."""

        ids: set[int] = set()

        def add(value: Any) -> None:
            if isinstance(value, (tuple, list, set)):
                for item in value:
                    add(item)
                return
            ids.add(id(value))

        for values in self._selector_artists.values():
            add(values)
        return frozenset(ids)

    def _forget_gesture_region(self) -> None:
        self._gesture_region = None
        self._gesture_overlay = ()
        self._gesture_selector_ids = frozenset()

    def _paint_gesture_overlay(self) -> bool:
        """Repaint only what sits above the captured frame, or refuse.

        The caller holds the style context: this is the pointer-motion path,
        and rc_context is not free.

        Refuses whenever the capture cannot still be the frame below the
        gesture: no capture, no gesture, or a selector scene whose artists
        were rebuilt (the drag changed the scene's shape).  The caller then
        composes, which recaptures.  Everything else that could invalidate
        the capture -- a relayout, an axes removed, a full draw, the end of
        the gesture -- drops it at the source.
        """

        canvas = self._figure.canvas
        restore = getattr(canvas, "restore_region", None)
        get_renderer = getattr(canvas, "get_renderer", None)
        if (
            self._gesture_region is None
            or self._selector_gesture_kind is None
            or not callable(restore)
            or not callable(get_renderer)
            or self._gesture_selector_ids != self._selector_artist_ids()
        ):
            return False
        restore(self._gesture_region)
        renderer = get_renderer()
        for _key, artist in self._gesture_overlay:
            if artist.get_visible():
                if not self._blit_exact_rgba_image(artist, canvas):
                    artist.draw(renderer)
        self._raster_generation += 1
        self._composed_generation = self._raster_generation
        return True

    def _blit_exact_rgba_image(self, artist: Any, canvas: Any) -> bool:
        """Copy a 1:1 precomposed RGBA front straight into the Agg buffer.

        Even an identity nearest resample pays Matplotlib's full image
        machinery per draw.  When the artist holds this renderer's own
        composed uint8 RGBA, fills its axes' whole data view, and its box
        lands on integer pixels at exactly the array's size, the draw is a
        row-aligned copy.  Every other case falls back to the artist.
        """

        from matplotlib.image import AxesImage

        if not isinstance(artist, AxesImage):
            return False
        axes = artist.axes
        if axes is None or artist.get_alpha() is not None:
            return False
        shown = artist.get_array()
        # Matplotlib stores EVERY image array as a MaskedArray -- set_data
        # runs it through safe_masked_invalid -- so refusing the type refused
        # every image there is, and this copy has never once run since it was
        # written.  What matters is whether anything is actually masked: a
        # fully unmasked array's data is the RGBA we composed.
        if isinstance(shown, np.ma.MaskedArray):
            if np.ma.getmask(shown) is not np.ma.nomask:
                return False
            shown = np.ma.getdata(shown)
        if not (
            isinstance(shown, np.ndarray)
            and shown.dtype == np.uint8
            and shown.ndim == 3
            and shown.shape[2] == 4
        ):
            return False
        # The copy OVERWRITES; a full draw alpha-BLENDS.  They agree only
        # when every pixel is opaque -- a translucent front (the 3D
        # scene outside its pane, a NaN-holed image) must take the real
        # draw or it would punch its transparency into the buffer.
        if shown[..., 3].min() != 255:
            return False
        if artist.get_interpolation() != "nearest":
            return False
        left, right, bottom, top = (float(v) for v in artist.get_extent())
        x_low, x_high = sorted(map(float, axes.get_xlim()))
        y_low, y_high = sorted(map(float, axes.get_ylim()))
        if not (
            math.isclose(x_low, min(left, right), rel_tol=1e-12, abs_tol=1e-9)
            and math.isclose(x_high, max(left, right), rel_tol=1e-12, abs_tol=1e-9)
            and math.isclose(y_low, min(bottom, top), rel_tol=1e-12, abs_tol=1e-9)
            and math.isclose(y_high, max(bottom, top), rel_tol=1e-12, abs_tol=1e-9)
        ):
            return False
        bbox = axes.bbox
        rows, columns = shown.shape[:2]
        x0, y0 = float(bbox.x0), float(bbox.y0)
        x1, y1 = float(bbox.x1), float(bbox.y1)
        if any(
            abs(value - round(value)) > 1e-6 for value in (x0, y0, x1, y1)
        ):
            return False
        if round(x1) - round(x0) != columns or round(y1) - round(y0) != rows:
            return False
        try:
            buffer = np.asarray(canvas.buffer_rgba())
            if not buffer.flags.writeable:
                return False
            height = buffer.shape[0]
            row_start = height - int(round(y1))
            buffer[
                row_start : row_start + rows,
                int(round(x0)) : int(round(x0)) + columns,
            ] = shown
        except Exception:
            return False
        return True

    @staticmethod
    def _native_draw(canvas: Any) -> None:
        """Compose one complete frame through the concrete Matplotlib canvas."""

        draw = getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("the native canvas has no callable draw method")
        draw()

    @contextmanager
    def raster_transaction(self) -> Iterator[None]:
        """Group session mutations without exposing partial raster state."""

        yield

    def begin_selector_gesture(self, kind: SelectorKind) -> bool:
        """Start a native selector gesture with complete-frame redraws."""

        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        self._selector_gesture_kind = kind
        self._selector_candidate = None
        # The COMPOSE is inside the style too, not just the artist update.
        # A frame drawn outside it is drawn under matplotlib's defaults --
        # its fonts, its sizes -- which is a different picture from every
        # other frame, and is what made the first drag on an image search
        # for font families this machine has never had.
        with style_context(self.style):
            self._update_selectors(self._last_selectors)
            self._compose_frame(chrome_stable=True)
        return True

    def begin_color_limit_gesture(self, candidate: ColorLimitCandidate) -> bool:
        """Start a native color-limit gesture with complete-frame redraws."""

        if not isinstance(candidate, ColorLimitCandidate):
            raise TypeError("candidate must be ColorLimitCandidate")
        self._selector_gesture_kind = SelectorSceneKind.COLOR_LIMITS
        self._color_limit_candidate = candidate
        with style_context(self.style):
            self._update_selectors(self._last_selectors)
            if not self._paint_gesture_overlay():
                self._compose_frame(chrome_stable=True)
        return True

    def preview_selector(self, state: SelectorState) -> bool:
        """Paint the current selector candidate as one complete frame."""

        if not isinstance(state, SelectorState):
            raise TypeError("state must be SelectorState")
        if self._selector_gesture_kind is not state.kind:
            return False
        self._selector_candidate = state
        # ONE style context for the whole move.  Entering it copies the whole
        # RcParams mapping (measured 0.4 ms), which is affordable once a frame
        # and not four times a pointer motion.
        with style_context(self.style):
            self._update_selectors(self._last_selectors)
            if not self._paint_gesture_overlay():
                self._compose_frame(chrome_stable=True)
        return True

    def preview_color_limit_candidate(self, candidate: ColorLimitCandidate) -> bool:
        """Paint the current color-limit candidate as one complete frame."""

        self._composed_generation = -1
        if not isinstance(candidate, ColorLimitCandidate):
            raise TypeError("candidate must be ColorLimitCandidate")
        if self._selector_gesture_kind is not SelectorSceneKind.COLOR_LIMITS:
            return False
        self._color_limit_candidate = candidate
        with style_context(self.style):
            self._update_selectors(self._last_selectors)
            self._compose_frame(chrome_stable=True)
        return True

    def _update_plot(self, payload: Any, state: DisplayState) -> None:
        """Update semantic data artists in place for one complete frame."""
        key, axes, _index = self.primary_surface
        handler_for(self.spec).render(self, payload, state, axes=axes, key=key)

    def end_selector_gesture(self) -> None:
        self._forget_gesture_region()
        self._selector_candidate = None
        self._color_limit_candidate = None
        self._selector_gesture_kind = None

    @staticmethod
    def _curve_labels(
        semantic: Any,
        source: tuple[Any, ...],
        state: DisplayState,
    ) -> tuple[str, str]:
        """The x and y labels one curve shows, from one rule.

        Written out twice, once per effect channel: BASE_GEOMETRY and TEXT run
        separately, so the two copies had to agree or the plot showed one label
        after a data update and a different one after a label edit.
        """

        labels = getattr(semantic, "labels", None)
        explicit_x = _state_label(
            state,
            "x_label",
            labels.x if labels and labels.x else None,
        )
        explicit_y = _state_label(state, "y_label", None)
        if explicit_y is None:
            explicit_y = _state_label(
                state,
                "value_label",
                None if labels is None else labels.y or labels.value,
            )
        x_label = (
            _quantity_label(source[0].x, "x", explicit_x)
            if source
            else ("x" if explicit_x is None else explicit_x)
        )
        y_label = (
            _quantity_label(source[0].y, "value", explicit_y)
            if source
            else ("value" if explicit_y is None else explicit_y)
        )
        return x_label, y_label

    def _effective_labels(
        self,
        payload: Any,
        state: DisplayState,
    ) -> tuple[str, str, str | None]:
        semantic = self.semantic_spec
        if isinstance(self.spec, FacetGridPlot):
            cells = tuple(getattr(payload, "cells", ()))
            if cells:
                index = self._focused_facet_index or 0
                index = min(max(index, 0), len(cells) - 1)
                payload = getattr(cells[index], "payload", cells[index])
        labels = getattr(semantic, "labels", None)
        if isinstance(semantic, CurvePlot):
            x_label, y_label = self._curve_labels(
                semantic, self._series(payload), state
            )
            return x_label, y_label, None
        if isinstance(semantic, HistogramPlot):
            explicit_x = _state_label(state, "x_label", None)
            if explicit_x is None:
                explicit_x = _state_label(
                    state,
                    "value_label",
                    None if labels is None else labels.x or labels.value,
                )
            quantity = getattr(payload, "edges", None)
            if quantity is None:
                quantity = getattr(payload, "values", payload)
            y_label = _state_label(
                state,
                "y_label",
                labels.y if labels and labels.y else None,
            )
            if y_label is None:
                y_label = "density" if bool(state["density"]) else "Shots"
            return _quantity_label(quantity, "value", explicit_x), y_label, None
        if isinstance(semantic, ImagePlot):
            explicit_x = _state_label(
                state,
                "x_label",
                labels.x if labels and labels.x else None,
            )
            explicit_y = _state_label(
                state,
                "y_label",
                labels.y if labels and labels.y else None,
            )
            explicit_value = _state_label(
                state,
                "value_label",
                labels.value if labels and labels.value else None,
            )
            value_label = (
                explicit_value
                if explicit_value and _EXPLICIT_UNIT_SUFFIX.search(explicit_value)
                else _quantity_label(payload.z, "value", explicit_value)
            )
            return (
                _quantity_label(payload.x, "x", explicit_x),
                _quantity_label(payload.y, "y", explicit_y),
                value_label,
            )
        if isinstance(semantic, RollingPlot):
            source = self._series(payload)
            # The rolling payload owns its x-axis meaning (absolute shot
            # index); rendering only falls back to it, never restates it.
            payload_x = source[0].x.label if source else "Shot"
            explicit_x = _state_label(state, "x_label", labels.x or payload_x)
            explicit_y = _state_label(state, "y_label", None)
            if explicit_y is None:
                explicit_y = _state_label(
                    state,
                    "value_label",
                    labels.y or labels.value,
                )
            y_label = (
                _quantity_label(source[0].y, "value", explicit_y)
                if source
                else ("value" if explicit_y is None else explicit_y)
            )
            return payload_x if explicit_x is None else explicit_x, y_label, None
        if isinstance(semantic, PulseTimelinePlot):
            _factor, unit = pulse_time_scale(payload, state["x_display_unit"])
            x_label = _state_label(state, "x_label", f"Time ({unit})")
            if x_label and not _EXPLICIT_UNIT_SUFFIX.search(x_label):
                x_label = f"{x_label} ({unit})"
            y_label = _state_label(state, "y_label", "")
            return x_label or "", y_label or "", None
        raise TypeError(f"unsupported plot specification {type(semantic).__name__}")

    def _update_text_artists(self, payload: Any, state: DisplayState) -> None:
        """Update semantic labels and titles without touching data artists."""

        x_label, y_label, value_label = self._effective_labels(payload, state)
        if isinstance(self.spec, FacetGridPlot):
            for name, value in (("x", x_label), ("y", y_label)):
                artist = self._artists.get(f"facet:outer_{name}")
                if artist is not None and artist.get_text() != value:
                    artist.set_text(value)
            if self._facet_focus_index is not None:
                axis = self._axes["facet_cell"][self._facet_focus_index]
                if axis.get_xlabel() != x_label:
                    axis.set_xlabel(x_label, fontsize=self.style.fonts.axis_label_pt)
                if axis.get_ylabel() != y_label:
                    axis.set_ylabel(y_label, fontsize=self.style.fonts.axis_label_pt)
        else:
            if self.primary_axes.get_xlabel() != x_label:
                self.primary_axes.set_xlabel(x_label)
            if self.primary_axes.get_ylabel() != y_label:
                self.primary_axes.set_ylabel(y_label)
        if value_label is not None:
            for key, value in self._artists.items():
                if key.endswith(":colorbar") and hasattr(value, "set_label"):
                    value.set_label(value_label)
        self._update_title_artist(state)

    def _update_chrome_artists(self, state: DisplayState) -> None:
        """Apply visibility/grid chrome without revisiting semantic text."""

        self._apply_colorbar_visibility(state)
        self._apply_grid(state)

    def _apply_colorbar_visibility(self, state: DisplayState) -> None:
        if "show_colorbar" not in state.values:
            return
        visible = bool(state["show_colorbar"])
        if isinstance(self.spec, FacetGridPlot) and self._facet_focus_index is None:
            # The overview owns no colorbar surface; the parameter applies
            # again the moment a cell is focused.
            visible = False
        for axis in self._axes.get("colorbar", ()):
            if bool(axis.get_visible()) != visible:
                axis.set_visible(visible)
                self._mark_axes_chrome_dirty(axis)

    def _series(self, payload: Any) -> tuple[Any, ...]:
        values = getattr(payload, "series", None)
        if values is None:
            raise TypeError("curve payload must expose a series tuple")
        return tuple(values)

    @staticmethod
    def _prepare_curve_series(source: Sequence[Any]) -> tuple[_PreparedSeries, ...]:
        prepared = []
        for item in source:
            x = np.asarray(_display_array(item.x), dtype=float).reshape(-1)
            y = np.asarray(_display_array(item.y), dtype=float).reshape(-1)
            valid = _valid_array(item, x.shape) & np.isfinite(x) & np.isfinite(y)
            label = getattr(item, "label", None)
            if label is None:
                group_key = getattr(item, "group_key", ())
                label = ", ".join(str(value) for value in group_key) if group_key else ""
            prepared.append(
                _PreparedSeries(
                    x,
                    y,
                    valid,
                    str(label),
                    _series_identity(item),
                    x_labels=getattr(item, "x_labels", None),
                    band=_series_band(item),
                )
            )
        return tuple(prepared)

    def _ensure_lines(self, axes: Any, count: int, key: str) -> list[Any]:
        lines: list[Any] = self._artists.setdefault(key, [])
        while len(lines) < count:
            index = len(lines)
            (line,) = axes.plot(
                [],
                [],
                color=self.style.palette.line_color(index),
                linewidth=self.style.artists.curve.linewidth,
                alpha=self.style.artists.curve.alpha,
                linestyle=self.style.artists.curve.linestyle,
                marker=self.style.artists.curve.marker,
                markersize=self.style.artists.curve_marker_size_pt,
            )
            lines.append(line)
        for index, line in enumerate(lines):
            visible = index < count
            if line.get_visible() != visible:
                line.set_visible(visible)
            if not visible:
                self._line_sources.pop(id(line), None)
        return lines

    def _update_curve(
        self,
        axes: Any,
        payload: Any,
        state: DisplayState,
        key: str,
        *,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
        prepared_series: tuple[_PreparedSeries, ...] | None = None,
        paint_labels: bool = True,
    ) -> None:
        source = self._series(payload)
        series = (
            self._prepare_curve_series(source)
            if prepared_series is None
            else prepared_series
        )
        x_label, y_label = self._curve_labels(self.semantic_spec, source, state)
        self._mutate_series_artists(
            axes,
            series,
            state,
            key,
            x_label=x_label,
            y_label=y_label,
            limits=limits,
            paint_labels=paint_labels,
            isolated_glyphs=True,
        )

    def _mutate_series_artists(
        self,
        axes: Any,
        series: tuple[_PreparedSeries, ...],
        state: DisplayState,
        key: str,
        *,
        x_label: str,
        y_label: str,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
        paint_labels: bool = True,
        isolated_glyphs: bool = False,
    ) -> None:
        lines = self._ensure_lines(axes, len(series), key)
        # The lines' data is about to change; every cached hover polyline is
        # of the old data.
        self._series_hit_cache.clear()
        # Limits need only the valid extremes, and min/max are indifferent to
        # element order: reducing each series in place is the same numbers as
        # gathering and concatenating every valid sample (two full copies of
        # a million-point trace per frame) and asking for the extremes then.
        extremes = np.array([np.inf, -np.inf, np.inf, -np.inf])
        series_lines: list[tuple[Any, object, str]] = []
        cycle = self.style.palette.line_cycle
        # Bars are rebuilt per update (uncertainty panels are scan-point
        # sized, never the million-point envelope path) and kept BY SERIES,
        # so focus can move them with their line.
        stale_bars = self._series_bars.pop(id(axes), {})
        for artists in stale_bars.values():
            for artist in artists:
                artist.remove()
        bars_by_series: dict[object, tuple[Any, ...]] = {}
        for index, item in enumerate(series):
            colour = cycle[_series_slot(item.identity, len(cycle))]
            # NaNs preserve invalid runs as gaps instead of joining neighbours.
            plotted_y = np.where(item.valid, item.y, np.nan)
            self._apply_line_data(
                axes,
                lines[index],
                item.x,
                plotted_y,
                isolated_glyphs=isolated_glyphs,
            )
            lines[index].set_color(colour)
            lines[index].set_linewidth(self.style.artists.curve.linewidth)
            lines[index].set_alpha(self.style.artists.curve.alpha)
            if lines[index].get_label() != item.label:
                lines[index].set_label(item.label)
            series_lines.append((lines[index], item.identity, item.label))
            band_low = band_high = None
            if item.band is not None:
                band_low, band_high = item.band
                band_where = (
                    item.valid
                    & np.isfinite(band_low)
                    & np.isfinite(band_high)
                )
                if bool(np.any(band_where)):
                    policy = self.style.render
                    x_marked = item.x[band_where]
                    y_marked = item.y[band_where]
                    container = axes.errorbar(
                        x_marked,
                        y_marked,
                        # Asymmetric on purpose: the bounds were converted
                        # to display units, and an affine display unit
                        # makes the two arms differ.
                        yerr=(
                            y_marked - band_low[band_where],
                            band_high[band_where] - y_marked,
                        ),
                        fmt="none",
                        ecolor=colour,
                        alpha=policy.uncertainty_bar_alpha,
                        elinewidth=policy.uncertainty_bar_linewidth,
                        capsize=policy.uncertainty_bar_capsize_pt,
                        capthick=policy.uncertainty_bar_linewidth,
                        zorder=lines[index].get_zorder() - 0.1,
                    )
                    _marker, caplines, barlinecols = container.lines
                    bars_by_series[item.identity] = (*caplines, *barlinecols)
                    # Keep the artists on the axes; drop only the
                    # container bookkeeping so revisions do not accumulate.
                    if container in axes.containers:
                        axes.containers.remove(container)
            if limits is None and bool(np.any(item.valid)):
                extremes[0] = min(
                    extremes[0],
                    float(np.min(item.x, where=item.valid, initial=np.inf)),
                )
                extremes[1] = max(
                    extremes[1],
                    float(np.max(item.x, where=item.valid, initial=-np.inf)),
                )
                low_source = item.y if band_low is None else np.where(
                    np.isfinite(band_low), band_low, item.y
                )
                high_source = item.y if band_high is None else np.where(
                    np.isfinite(band_high), band_high, item.y
                )
                extremes[2] = min(
                    extremes[2],
                    float(np.min(low_source, where=item.valid, initial=np.inf)),
                )
                extremes[3] = max(
                    extremes[3],
                    float(
                        np.max(high_source, where=item.valid, initial=-np.inf)
                    ),
                )
        self._series_lines[id(axes)] = tuple(series_lines)
        if bars_by_series:
            self._series_bars[id(axes)] = bars_by_series
        self._series_indices[id(axes)] = {
            identity: index for index, (_line, identity, _label) in enumerate(series_lines)
        }
        if limits is not None:
            self._set_xlim(axes, *limits[0])
            self._set_ylim(axes, *limits[1])
        else:
            xlim = (
                _curve_x_limits(extremes[0:2])
                if math.isfinite(extremes[0])
                else None
            )
            y_range = (
                _data_limits(extremes[2:4])
                if math.isfinite(extremes[2])
                else None
            )
            if xlim is not None:
                self._set_xlim(axes, *xlim)
            self._set_ylim(
                axes,
                *self._resolve_curve_y_limits(key, y_range, state),
            )
        # Grouped series are told apart by the palette cycle; no legend is
        # drawn for them -- it was never an authored behaviour, and on a
        # panel-sized plot it covers the data it names.
        if axes.get_legend() is not None:
            axes.get_legend().remove()
        if paint_labels:
            if axes.get_xlabel() != x_label:
                axes.set_xlabel(x_label)
            if axes.get_ylabel() != y_label:
                axes.set_ylabel(y_label)
            # A facet cell (paint_labels=False) gets its tick policy from
            # the GRID loop, at the grid's typography.  Installing the
            # standalone policy here too made the two signatures thrash:
            # every cell re-installed both locators on every frame.
            apply_smart_ticks(axes, label_pt=self.style.fonts.tick_pt)
        labelled = next(
            (item for item in series if item.x_labels is not None), None
        )
        if labelled is not None:
            # A labelled categorical axis ticks BY NAME, one tick per
            # declared coordinate -- the same names the legend, hover and
            # scope rows already use.
            axes.set_xticks(
                np.asarray(labelled.x, dtype=float),
                labels=labelled.x_labels,
            )
        self._apply_series_focus()

    def _series_hit(self, axes: Any | None, px: float, py: float, radius: float
                    ) -> tuple[int, object, str, float, float] | None:
        if axes is None or not axes.get_visible():
            return None
        point = np.asarray((px, py), dtype=float)
        best = (float(radius) * self.plan.device_pixel_ratio) ** 2
        hit = None
        entries = self._series_lines.get(id(axes), ())
        current = None if self._series_hover is None else self._series_hover[1]
        if current is not None:
            entries = tuple(item for item in entries if item[1] == current) + tuple(
                item for item in entries if item[1] != current)
        for line, identity, label in entries:
            if not line.get_visible():
                continue
            signature = (
                tuple(map(float, axes.get_xlim())),
                tuple(map(float, axes.get_ylim())),
                int(round(float(self._figure.bbox.width))),
                int(round(float(self._figure.bbox.height))),
            )
            cached = self._series_hit_cache.get(id(line))
            if cached is not None and cached[0] == signature:
                x, y, pixels, finite, isolated_glyphs = cached[1]
            else:
                registered = self._line_sources.get(id(line))
                isolated_glyphs = False
                if registered is not None and registered[0] is line:
                    _line, _owner, raw_x, raw_y, isolated_glyphs = registered
                    x = np.asarray(raw_x, dtype=float).reshape(-1)
                    y = np.asarray(raw_y, dtype=float).reshape(-1)
                else:
                    x = np.asarray(line.get_xdata(), dtype=float).reshape(-1)
                    y = np.asarray(line.get_ydata(), dtype=float).reshape(-1)
                if x.size > _ENVELOPE_MAX_COLUMNS * 4:
                    low, high = sorted(map(float, axes.get_xlim()))
                    start = max(0, int(np.searchsorted(x, low)) - 1)
                    stop = min(x.size, int(np.searchsorted(x, high, side="right")) + 1)
                    x, y = x[start:stop], y[start:stop]
                finite = np.isfinite(x) & np.isfinite(y)
                pixels = np.full((x.size, 2), np.nan)
                if np.any(finite):
                    pixels[finite] = axes.transData.transform(
                        np.column_stack((x[finite], y[finite]))
                    )
                self._series_hit_cache[id(line)] = (
                    signature,
                    (x, y, pixels, finite, isolated_glyphs),
                )
            if not np.any(finite):
                continue
            adjacent = finite[:-1] & finite[1:]
            starts, ends = pixels[:-1][adjacent], pixels[1:][adjacent]
            if starts.size:
                delta = ends - starts
                length = np.einsum("ij,ij->i", delta, delta)
                t = np.zeros(length.shape)
                usable = length > 0
                t[usable] = np.einsum("ij,ij->i", point - starts[usable],
                                      delta[usable]) / length[usable]
                np.clip(t, 0.0, 1.0, out=t)
                closest = starts + t[:, None] * delta - point
                distance = np.einsum("ij,ij->i", closest, closest)
                local = int(np.argmin(distance))
                if float(distance[local]) <= best:
                    indices = np.flatnonzero(adjacent)
                    source = int(indices[local])
                    best = float(distance[local])
                    hit = (id(axes), identity, label,
                           float(x[source] + t[local] * (x[source + 1] - x[source])),
                           float(y[source] + t[local] * (y[source + 1] - y[source])))
                    if identity == current:
                        return hit
            singleton = (
                _isolated_curve_mask(x, y)
                if isolated_glyphs
                else finite if not starts.size and finite.sum() == 1
                else np.zeros(finite.shape, dtype=bool)
            )
            if bool(np.any(singleton)):
                sources = np.flatnonzero(singleton)
                delta = pixels[sources] - point
                distance = np.einsum("ij,ij->i", delta, delta)
                local = int(np.argmin(distance))
                if float(distance[local]) <= best:
                    source = int(sources[local])
                    best = float(distance[local])
                    hit = (
                        id(axes),
                        identity,
                        label,
                        float(x[source]),
                        float(y[source]),
                    )
                    if identity == current:
                        return hit
        return hit

    def _apply_series_focus(self) -> None:
        locked = self._series_locked
        active = locked or self._series_hover
        # Focus styling is a pure function of (focus state, the exact line
        # artists alive).  Data revisions reuse their artists, so replaying
        # the whole property walk over 64 cells' series on every frame was
        # 17 ms of setting every value to itself.
        token = (
            locked,
            self._series_hover,
            tuple(
                id(line)
                for entries in self._series_lines.values()
                for line, _series_id, _label in entries
            ),
            # Error-bar artists are rebuilt per revision; fresh ones carry
            # default properties until this walk styles them.
            tuple(
                id(artist)
                for bars in self._series_bars.values()
                for artists in bars.values()
                for artist in artists
            ),
        )
        if getattr(self, "_series_focus_applied", None) == token:
            return
        self._series_focus_applied = token
        identity = None if active is None else active[1]
        focus_line = None
        bar_alpha = self.style.render.uncertainty_bar_alpha
        for axis_id, entries in self._series_lines.items():
            axis_bars = self._series_bars.get(axis_id, {})
            for line, series_id, _label in entries:
                focused = identity is not None and series_id == identity
                if locked is not None:
                    line.set_linewidth(
                        self.style.artists.curve.linewidth * (2 if focused else 1)
                    )
                    line.set_alpha(1.0 if focused else 0.18)
                    line.set_zorder(4.0 if focused else 2.0)
                elif active is not None:
                    line.set_linewidth(
                        self.style.artists.curve.linewidth * (1.45 if focused else 1)
                    )
                    line.set_alpha(
                        1.0 if focused else self.style.artists.curve.alpha
                    )
                    line.set_zorder(3.0 if focused else 2.0)
                else:
                    line.set_linewidth(self.style.artists.curve.linewidth)
                    line.set_alpha(self.style.artists.curve.alpha)
                    line.set_zorder(2.0)
                # The bars are part of the series: alpha, weight and depth
                # all move with their line -- dimming to near-nothing behind
                # a locked focus, thickening with a focused line, and always
                # sitting just under it.
                bar_linewidth = self.style.render.uncertainty_bar_linewidth
                if locked is not None:
                    series_bar_alpha = bar_alpha if focused else 0.06
                    series_bar_width = bar_linewidth * (1.6 if focused else 1.0)
                elif active is not None:
                    series_bar_alpha = (
                        min(1.0, bar_alpha * 1.4) if focused else bar_alpha
                    )
                    series_bar_width = bar_linewidth * (1.3 if focused else 1.0)
                else:
                    series_bar_alpha = bar_alpha
                    series_bar_width = bar_linewidth
                for artist in axis_bars.get(series_id, ()):
                    artist.set_alpha(series_bar_alpha)
                    artist.set_zorder(line.get_zorder() - 0.1)
                    if hasattr(artist, "set_markeredgewidth"):
                        # A capline is a marker-only Line2D: its visible
                        # weight is the marker edge.
                        artist.set_markeredgewidth(series_bar_width)
                    else:
                        artist.set_linewidth(series_bar_width)
                if line.get_marker() == "_":
                    line.set_markeredgewidth(line.get_linewidth())
                if focused and active is not None and axis_id == active[0]:
                    focus_line = line
        for annotation in self._series_annotations.values():
            annotation.set_visible(False)
        if active is None or focus_line is None:
            return
        axis_id = active[0]
        annotation = self._series_annotations.get(axis_id)
        if annotation is None:
            annotation = focus_line.axes.text(
                0.98,
                0.98,
                "",
                transform=focus_line.axes.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=self.style.fonts.annotation_pt,
                clip_on=True,
                zorder=12,
            )
            self._series_annotations[axis_id] = annotation
            self._artists[f"series-inspector:{axis_id}"] = annotation
        annotation.set_text(
            f"{'* ' if locked is not None else ''}{active[2] or 'Series'}"
        )
        annotation.set_color(focus_line.get_color())
        annotation.set_visible(True)

    def series_focus(self, action: str, axes: Any | None, px: float, py: float, *,
                     hit_radius: float, click_radius: float = 0.0, redraw: bool = True) -> bool:
        before = self._series_locked or self._series_hover
        before_state = (
            "locked" if self._series_locked is not None else
            "hover" if self._series_hover is not None else "none",
            None if before is None else before[1],
        )
        handled = False
        if action == "press":
            hit = self._series_hit(axes, px, py, hit_radius)
            self._series_press = (px, py, None if hit is None else hit[1])
            return False
        if action == "move":
            if self._series_locked is not None:
                return False
            hit = self._series_hit(axes, px, py, hit_radius)
            if (None if before is None else before[1]) == (None if hit is None else hit[1]):
                return False
            self._series_hover = hit
        elif action == "release":
            press, self._series_press = self._series_press, None
            if press is None or math.hypot(px - press[0], py - press[1]) > (
                click_radius * self.plan.device_pixel_ratio
            ):
                return False
            hit = self._series_hit(axes, px, py, hit_radius)
            if press[2] != (None if hit is None else hit[1]):
                return False
            handled = hit is not None or before is not None
            self._series_hover = None
            same = hit is not None and self._series_locked is not None
            self._series_locked = None if hit is None or (same and self._series_locked[1] == hit[1]) else hit
        elif action == "leave":
            self._series_press = None
            if self._series_locked is not None:
                return False
            self._series_hover = None
        elif action == "clear":
            self._series_press = self._series_hover = self._series_locked = None
        after = self._series_locked or self._series_hover
        after_state = (
            "locked" if self._series_locked is not None else
            "hover" if self._series_hover is not None else "none",
            None if after is None else after[1],
        )
        if before_state == after_state:
            return handled
        self._apply_series_focus()
        if redraw:
            with style_context(self.style):
                self._compose_frame(chrome_stable=True)
        return handled if action == "release" else True

    def series_focus_scroll(self, axes: Any | None, step: float) -> bool:
        locked = self._series_locked
        if (
            locked is None or axes is None or id(axes) != locked[0]
            or not locked[1]
            or not isinstance(self.semantic_spec, (CurvePlot, RollingPlot))
            or (isinstance(self.spec, FacetGridPlot) and self._facet_focus_index is None)
        ):
            return False
        entries = self._series_lines.get(locked[0], ())
        current = self._series_indices.get(locked[0], {}).get(locked[1])
        if current is None or not entries:
            return False
        target = max(0, min(len(entries) - 1, current + (-1 if step > 0 else 1)))
        if target == current:
            return True
        line, identity, label = entries[target]
        x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        anchor = (locked[3], locked[4])
        if x.size and y.size and np.isfinite((x[0], y[0])).all():
            anchor = (float(x[0]), float(y[0]))
        self._series_locked = (locked[0], identity, label, *anchor)
        self._apply_series_focus()
        with style_context(self.style):
            self._compose_frame(chrome_stable=True)
        return True

    def _histogram_arrays(
        self, payload: Any, state: DisplayState
    ) -> tuple[np.ndarray, np.ndarray]:
        edges = getattr(payload, "edges", None)
        counts = getattr(payload, "counts", None)
        if edges is not None and counts is not None:
            edge_values = np.asarray(_display_array(edges), dtype=float).reshape(-1)
            count_values = np.asarray(counts, dtype=float).reshape(-1)
        else:
            values = getattr(payload, "values", payload)
            values = np.asarray(_display_array(values), dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            count_values, edge_values = np.histogram(
                values,
                bins=aligned_histogram_edges(values, int(state["bin_count"])),
            )
            count_values = count_values.astype(float)
        density = bool(state["density"])
        cumulative = bool(state["cumulative"])
        if cumulative:
            count_values = np.cumsum(count_values)
            if density and count_values.size and count_values[-1] > 0.0:
                count_values = count_values / count_values[-1]
        elif density:
            total = float(np.sum(count_values))
            widths = np.diff(edge_values)
            if total > 0:
                count_values = count_values / (total * widths)
        return edge_values, count_values

    def _update_histogram(
        self,
        axes: Any,
        payload: Any,
        state: DisplayState,
        key: str,
        *,
        arrays: tuple[np.ndarray, np.ndarray] | None = None,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
        paint_labels: bool = True,
    ) -> None:
        from matplotlib.collections import PolyCollection

        edges, counts = self._histogram_arrays(payload, state) if arrays is None else arrays
        collection = self._artists.get(key)
        alpha = self.style.artists.histogram_fill_alpha
        if collection is None:
            collection = PolyCollection(
                _histogram_vertices(edges, counts),
                facecolors=self.style.palette.hist_fill,
                edgecolors="none",
                alpha=alpha,
            )
            axes.add_collection(collection)
            self._artists[key] = collection
        else:
            collection.set_verts(_histogram_vertices(edges, counts))
        self._artists[f"{key}:projection"] = (edges, counts)
        if limits is not None:
            selected_x = limits[0]
            selected_y = limits[1]
        else:
            selected_x = (
                (float(edges[0]), float(edges[-1]))
                if edges.size >= 2
                else None
            )
            selected_y = self._resolve_histogram_y_limits(key, counts, state)

        scale = "log" if bool(state["log_y"]) else "linear"
        if scale == "log" and selected_y[0] <= 0.0:
            raise RuntimeError("resolved logarithmic limits are not positive")
        if axes.get_yscale() != scale:
            if scale == "log":
                # Matplotlib cannot switch a linear axis whose current lower
                # bound is zero directly to log.  Install a valid temporary
                # positive span first; the authored limits are applied below
                # in the same render transaction.
                old_low, old_high = sorted(map(float, axes.get_ylim()))
                if old_low <= 0.0:
                    axes.set_ylim(
                        float(selected_y[0]),
                        max(float(selected_y[1]), float(selected_y[0]) * 10.0),
                    )
            axes.set_yscale(scale)
            self._mark_axes_chrome_dirty(axes)

        if limits is not None:
            self._set_xlim(axes, *selected_x)
            self._set_ylim(axes, *selected_y)
        else:
            if selected_x is not None:
                self._set_xlim(axes, *selected_x)
            self._set_ylim(axes, *selected_y)
        labels = getattr(self.semantic_spec, "labels", None)
        explicit_x = _state_label(state, "x_label", None)
        if explicit_x is None:
            explicit_x = _state_label(
                state,
                "value_label",
                None if labels is None else labels.x or labels.value,
            )
        quantity = getattr(payload, "edges", None)
        if quantity is None:
            quantity = getattr(payload, "values", payload)
        x_label = _quantity_label(quantity, "value", explicit_x)
        y_label = _state_label(
            state,
            "y_label",
            labels.y if labels and labels.y else None,
        )
        if y_label is None:
            y_label = "density" if bool(state["density"]) else "Shots"
        if paint_labels:
            if axes.get_xlabel() != x_label:
                axes.set_xlabel(x_label)
            if axes.get_ylabel() != y_label:
                axes.set_ylabel(y_label)
        if paint_labels:
            apply_smart_ticks(axes, label_pt=self.style.fonts.tick_pt)

    def _resolve_histogram_y_limits(
        self,
        key: str,
        counts: np.ndarray,
        state: DisplayState,
    ) -> tuple[float, float]:
        mode = str(state["relim_mode"])
        logarithmic = bool(state["log_y"])
        # Everything that changes what one count MEANS belongs in the
        # retention signature: toggling density/cumulative or re-binning is a
        # representation change the axis must re-fit to cleanly, not
        # shot-to-shot jitter for the expand/shrink hysteresis to damp.
        count_semantics = (
            mode,
            logarithmic,
            bool(state["density"]),
            bool(state["cumulative"]),
            int(state["bin_count"]),
        )
        shrink_ratio = self.style.render.distribution_count_shrink_ratio
        state_key = f"{key}:histogram_limits"
        mode_key = f"{key}:histogram_mode"
        previous = self._artists.get(state_key)
        previous_mode = self._artists.get(mode_key)
        if mode == "fixed":
            automatic = None
        else:
            finite = np.asarray(counts, dtype=float).reshape(-1)
            finite = finite[np.isfinite(finite)]
            if logarithmic:
                positive = finite[finite > 0.0]
                if positive.size:
                    target = (
                        max(float(np.min(positive)) * 0.8, np.finfo(float).tiny),
                        max(float(np.max(positive)) * 1.2, 1.0),
                    )
                else:
                    target = (0.8, 1.2)
            else:
                peak = float(np.max(finite)) if finite.size else 0.0
                # The 1.0 fallback guards a degenerate axis over empty data
                # only; it must not act as a counts floor, which pinned
                # density histograms (peaks ~1e-2) to the bottom of the axis.
                target = (0.0, peak * 1.08 if peak > 0.0 else 1.0)
            force = (
                previous is None
                or previous_mode != count_semantics
                or not _relim_retains(mode)
            )
            if force:
                automatic = target
            else:
                prior_low, prior_high = tuple(map(float, previous))
                # Expansion triggers on the DATA breaching the held bound,
                # not on the padded target exceeding it: a rising sample
                # maximum otherwise ratchets the axis (and every dependent
                # cell's chrome) on each revision.  An actual breach expands
                # with extra headroom so ordinary jitter fits under the new
                # bound for many revisions.
                if logarithmic:
                    breach_low = float(np.min(positive)) if positive.size else prior_low
                    breach_high = float(np.max(positive)) if positive.size else prior_high
                    expand = breach_low < prior_low or breach_high > prior_high
                    expanded = (
                        min(target[0], prior_low),
                        max(breach_high * 1.5, prior_high),
                    )
                else:
                    expand = peak > prior_high
                    expanded = (0.0, max(1.0, peak * 1.25))
                shrink = (
                    target[1] < shrink_ratio * prior_high
                    or (
                        logarithmic
                        and target[0]
                        > prior_low / shrink_ratio
                    )
                )
                if expand:
                    automatic = expanded
                elif shrink:
                    automatic = target
                else:
                    automatic = (prior_low, prior_high)
        selected = _select_display_limits(mode, automatic, state, "y")
        if logarithmic and selected[0] <= 0.0:
            raise RuntimeError("resolved logarithmic limits are not positive")
        self._artists[state_key] = selected
        self._artists[mode_key] = count_semantics
        return selected

    def _cached_image_range(
        self,
        key: str,
        source_values: object,
        source_valid: object,
        values: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[float, float] | None:
        """The finite range of one image surface, measured once per revision.

        Keyed on the arrays the caller was handed rather than the normalised
        pair: normalisation rebuilds a broadcast view every call, so a key made
        of those is a fresh object each time and never matches.  Cached per
        surface ``key``, so a FacetGrid overview measures each cell once even
        though the pooled colour scale asks for every cell's range before any
        cell is drawn.
        """

        revision_key = (id(source_values), id(source_valid), self._data_revision)
        cached = self._image_ranges.get(key)
        if cached is not None and cached[0] == revision_key:
            return cached[1]
        measured = _image_data_range(values, valid)
        #: The source arrays are held so their ids cannot be recycled.
        self._image_ranges[key] = (revision_key, measured, (source_values, source_valid))
        return measured

    def _resolve_image_limits(
        self,
        key: str,
        data_range: tuple[float, float] | None,
        state: DisplayState,
    ) -> tuple[float, float]:
        return self._resolve_data_limits(
            key, data_range, state, "color", allow_partial=True
        )

    def _resolve_data_limits(
        self,
        key: str,
        data_range: tuple[float, float] | None,
        state: DisplayState,
        quantity: str,
        *,
        allow_partial: bool = False,
        cache_selected: bool = False,
    ) -> tuple[float, float]:
        limits_key = f"{key}:automatic_{quantity}_limits"
        mode_key = f"{key}:relim_mode"
        mode = str(state["relim_mode"])
        current = self._artists.get(limits_key)
        if mode == "fixed":
            automatic = None
        elif data_range is None:
            automatic = current if current is not None else (0.0, 1.0)
        else:
            policy = self.style.render
            automatic = _autoscaled_limits(
                data_range,
                current,
                padding_fraction=policy.image_color_padding_fraction,
                deadband_fraction=policy.image_color_deadband_fraction,
                zero_based=mode == "normal",
                # Changing the mode is a new question, so the answer is
                # re-fitted rather than held over from the old one.
                retain=_relim_retains(mode) and self._artists.get(mode_key) == mode,
            )
        selected = _select_display_limits(
            mode, automatic, state, quantity, allow_partial=allow_partial
        )
        cached = selected if cache_selected else automatic
        if cached is not None:
            self._artists[limits_key] = cached
        self._artists[mode_key] = mode
        return selected

    def _resolve_distribution_limits(
        self,
        key: str,
        data_range: tuple[float, float] | None,
        color_limits: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Resolve histogram and rail domains without clipping the color handles."""

        limits_key = f"{key}:distribution_data_limits"
        current = self._artists.get(limits_key)
        if data_range is None:
            selected = current if current is not None else (0.0, 1.0)
        else:
            policy = self.style.render
            # The rail shows where this data sits under the colour scale.
            # It is not one of the panel's relim-mode axes: its domain is the
            # data's own span, damped, so the handles the operator drags do
            # not shift under the pointer between revisions.
            selected = _autoscaled_limits(
                data_range,
                current,
                padding_fraction=policy.image_color_padding_fraction,
                deadband_fraction=policy.image_color_deadband_fraction,
            )
        histogram_limits = tuple(map(float, selected))
        self._artists[limits_key] = histogram_limits
        rail_limits = (
            min(histogram_limits[0], float(color_limits[0])),
            max(histogram_limits[1], float(color_limits[1])),
        )
        return histogram_limits, rail_limits

    def _resolve_curve_y_limits(
        self,
        key: str,
        data_range: tuple[float, float] | None,
        state: DisplayState,
    ) -> tuple[float, float]:
        return self._resolve_data_limits(
            key, data_range, state, "y", cache_selected=True
        )

    def _update_image_artist(
        self,
        axes: Any,
        values: np.ndarray,
        valid: np.ndarray,
        extent: tuple[float, float, float, float],
        state: DisplayState,
        key: str,
        color_limits: tuple[float, float] | None,
        *,
        square_view: bool,
        coordinate_aspect: float | None,
        valid_identity: object = None,
    ) -> tuple[Any, Any]:
        import matplotlib

        policy = self.style.render
        cmap_name = str(state["colormap"])
        cmap_cache_key = (cmap_name,)
        cached_cmap = self._artists.get("image:cmap_cache")
        if cached_cmap is not None and cached_cmap[0] == cmap_cache_key:
            cmap = cached_cmap[1]
        else:
            cmap = matplotlib.colormaps[cmap_name].copy()
            # "No data here" is the surface showing through, not a colour of
            # its own.  A grey of its own is how one image panel came to say
            # it twice -- grey inside the extent, white in the square band
            # beside it -- and how a facet cell with nothing in it looked
            # like a different fact from an empty image plot.
            cmap.set_bad("none")
            self._artists["image:cmap_cache"] = (cmap_cache_key, cmap)
        mapping_state = (cmap_name, color_limits)
        mapping_key = f"{key}:mapping_state"
        previous_mapping = self._artists.get(mapping_key)

        if self._height_bars_active(key, state):
            self._artists[mapping_key] = mapping_state
            return self._update_height_bars_artist(
                axes,
                values,
                valid,
                extent,
                state,
                key,
                color_limits,
                cmap_name,
                cmap,
                valid_identity=valid_identity,
            )
        if not axes.axison:
            # Returning from the height-bar presentation restores the 2D
            # chrome this artist path owns and hides the scene's.
            axes.set_axis_on()
            self._hide_height_bars_chrome(key)
            self._mark_axes_chrome_dirty(axes)

        home_extent = (
            _square_image_limits(
                extent,
                coordinate_aspect=coordinate_aspect,
            )
            if square_view and coordinate_aspect is not None
            else extent
        )
        self._home_limits[id(axes)] = (
            (float(home_extent[0]), float(home_extent[1])),
            (float(home_extent[2]), float(home_extent[3])),
        )
        requested = (
            self._requested_view_limits
            if any(axes is item for _key, item, _index in self.painted_surfaces)
            else None
        )
        if requested is None:
            x_limits = (home_extent[0], home_extent[1])
            y_limits = (home_extent[2], home_extent[3])
        else:
            x_limits, y_limits = requested
        self._set_xlim(axes, *x_limits)
        self._set_ylim(axes, *y_limits)
        if axes.get_anchor() != policy.image_anchor:
            axes.set_anchor(policy.image_anchor)
        wanted_aspect = "auto" if coordinate_aspect is None else coordinate_aspect
        if (
            axes.get_aspect() != wanted_aspect
            or axes.get_adjustable() != "box"
        ):
            axes.set_aspect(wanted_aspect, adjustable="box")
        # Equal aspect changes the actual drawable box.  Resolve that box
        # before choosing a source reduction so one prepared sample maps to at
        # roughly one physical output pixel at the current DPR.  The resolve
        # is a pure function of (aspect, limits, position); a 64-cell grid
        # re-asked it per cell per frame with none of them changed.
        aspect_signature = (
            wanted_aspect,
            tuple(map(float, x_limits)),
            tuple(map(float, y_limits)),
            tuple(axes.get_position(original=True).bounds),
        )
        aspect_key = f"{key}:aspect_box"
        aspect_cached = self._artists.get(aspect_key)
        if aspect_cached is not None and aspect_cached[0] == aspect_signature:
            display_pixel_shape = aspect_cached[1]
        else:
            axes.apply_aspect()
            display_pixel_shape = (
                max(1, round(float(axes.bbox.width))),
                max(1, round(float(axes.bbox.height))),
            )
            self._artists[aspect_key] = (aspect_signature, display_pixel_shape)
        store_key = f"{key}:front_store"
        store = self._artists.get(store_key)
        if not isinstance(store, ImageFrontStore):
            store = ImageFrontStore()
            self._artists[store_key] = store
        prepared: PreparedImageFront = store.prepare(
            values,
            valid,
            extent,
            x_limits=tuple(map(float, x_limits)),
            y_limits=tuple(map(float, y_limits)),
            display_pixel_shape=display_pixel_shape,
            policy=policy.image_front,
            revision_token=(
                self._data_revision,
                id(values),
                values.shape,
                values.strides,
                values.dtype.str,
            ),
        )

        # Precomposed RGBA sidesteps Matplotlib's per-draw normalize + LUT +
        # second mask-resample machinery: the same 256-level quantization is
        # applied once per (front, colormap, limits) here instead of on every
        # draw.  Nearest resampling commutes with colormapping exactly; masked
        # fronts and unresolved limits keep the scalar path.
        self._artists[f"{key}:prepared_current"] = prepared
        rgba_front = (
            self._image_rgba_front(key, prepared, cmap_name, cmap, color_limits)
            if color_limits is not None
            and not isinstance(prepared.values, np.ma.MaskedArray)
            else None
        )
        self._artists[f"{key}:color_mode"] = (
            "scalar" if rgba_front is None else "rgba"
        )
        if rgba_front is not None:
            rgba_front = self._box_sized_rgba_front(
                key,
                rgba_front,
                prepared.extent,
                x_limits,
                y_limits,
                axes,
            )
        shown: Any = prepared.values if rgba_front is None else rgba_front
        applied_key = f"{key}:applied_front"
        image = self._artists.get(key)
        if image is None:
            scalar_options = (
                {
                    "cmap": cmap,
                    "vmin": None if color_limits is None else color_limits[0],
                    "vmax": None if color_limits is None else color_limits[1],
                }
                if rgba_front is None
                else {}
            )
            image = axes.imshow(
                shown,
                origin=policy.image_origin,
                aspect="auto" if coordinate_aspect is None else coordinate_aspect,
                extent=prepared.extent,
                interpolation="nearest",
                **scalar_options,
            )
            if rgba_front is not None:
                # RGBA rendering ignores these, but the artist stays the
                # authority every painted-limits consumer reads.
                image.set_cmap(cmap)
                assert color_limits is not None
                image.set_clim(*color_limits)
            self._artists[key] = image
            self._artists[applied_key] = shown
        else:
            if self._artists.get(applied_key) is not shown:
                # ``set_data`` copies the front; skip it when the artist
                # already holds this exact composed object (cache hit).
                image.set_data(shown)
                extent_key = f"{key}:applied_extent"
                if self._artists.get(extent_key) != prepared.extent:
                    # ``set_extent`` rebuilds transforms and re-autoscales;
                    # a live feed re-sets the same extent per revision.
                    image.set_extent(prepared.extent)
                    self._artists[extent_key] = prepared.extent
                self._artists[applied_key] = shown
            # The artist's cmap/clim stay authoritative in both modes: RGBA
            # rendering ignores them, but selector handles, rail guides and
            # pointer snapshots all read the painted limits off the artist.
            if previous_mapping is None or previous_mapping[0] != cmap_name:
                image.set_cmap(cmap)
            if color_limits is not None:
                applied_low, applied_high = image.get_clim()
                if (
                    previous_mapping is None
                    or previous_mapping[1] != color_limits
                    or not (
                        _scalar_close(float(applied_low), float(color_limits[0]))
                        and _scalar_close(
                            float(applied_high), float(color_limits[1])
                        )
                    )
                ):
                    image.set_clim(*color_limits)
        self._artists[mapping_key] = mapping_state
        # ``imshow``/``set_extent`` may autoscale a new artist.  Reassert the
        # transaction's final transform after mutating the front.
        self._set_xlim(axes, *x_limits)
        self._set_ylim(axes, *y_limits)
        return image, cmap

    def _box_sized_rgba_front(
        self,
        key: str,
        rgba: np.ndarray,
        extent: tuple[float, float, float, float],
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        axes: Any,
    ) -> np.ndarray:
        """Upsample a small composed RGBA front to its axes' pixel box.

        A 10x10 heatmap cell paid Matplotlib's full image machinery on
        every draw of every cell; at the box size the compose's exact
        row-copy blit takes over, and a facet of 64 such cells stops
        re-rendering 64 images per frame.  Nearest-neighbour placement is
        computed here instead of inside Agg -- boundary rows may land one
        pixel from Agg's fixed-point choice, which changes no value or
        coordinate semantics.  Anything but a strict integer-box upsample
        of a view-filling front is returned unchanged.
        """

        rows, columns = rgba.shape[:2]
        left, right, bottom, top = (float(v) for v in extent)
        x_low, x_high = sorted(map(float, x_limits))
        y_low, y_high = sorted(map(float, y_limits))
        if not (
            math.isclose(x_low, min(left, right), rel_tol=1e-12, abs_tol=1e-9)
            and math.isclose(x_high, max(left, right), rel_tol=1e-12, abs_tol=1e-9)
            and math.isclose(y_low, min(bottom, top), rel_tol=1e-12, abs_tol=1e-9)
            and math.isclose(y_high, max(bottom, top), rel_tol=1e-12, abs_tol=1e-9)
        ):
            return rgba
        bbox = axes.bbox
        corners = (
            float(bbox.x0),
            float(bbox.y0),
            float(bbox.x1),
            float(bbox.y1),
        )
        if any(abs(v - round(v)) > 1e-6 for v in corners):
            return rgba
        box_w = int(round(corners[2])) - int(round(corners[0]))
        box_h = int(round(corners[3])) - int(round(corners[1]))
        if box_w <= columns or box_h <= rows or box_w < 1 or box_h < 1:
            return rgba
        cache_name = f"{key}:rgba_box"
        cache_key = (id(rgba), box_w, box_h)
        cached = self._artists.get(cache_name)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        row_map = np.minimum(
            ((np.arange(box_h) + 0.5) * (rows / box_h)).astype(np.intp),
            rows - 1,
        )
        column_map = np.minimum(
            ((np.arange(box_w) + 0.5) * (columns / box_w)).astype(np.intp),
            columns - 1,
        )
        sized = rgba[row_map][:, column_map]
        sized.setflags(write=False)
        self._artists[cache_name] = (cache_key, sized)
        return sized

    def _height_bars_active(self, key: str, state: DisplayState) -> bool:
        """Whether this image surface paints the height-bar presentation.

        The presentation is a property of the ONE examined image surface:
        the standalone image, or a focused facet cell.  A facet overview
        stays a heatmap grid -- sixty-four simultaneous 3D scenes would
        be neither readable nor affordable.
        """

        try:
            wanted = str(state["presentation"])
        except KeyError:
            return False
        if wanted != "height_bars":
            return False
        if key == "image":
            return True
        return key.startswith("facet:") and self._facet_focus_index is not None

    def set_height_bars_dragging(self, dragging: bool) -> None:
        """Say whether a hand is currently turning the scene.

        This is a RESOLUTION BUDGET and nothing else.  The camera itself
        has one owner -- the display parameters -- so a drag is not a
        second, transient place where the view can live: whatever moves
        the scene writes the parameters, and anything that rebuilds,
        replaces or re-mounts the surface therefore shows the view the
        hand is holding rather than the one it left.
        """

        dragging = bool(dragging)
        if dragging == self._height_bars_dragging:
            return
        self._composed_generation = -1
        self._height_bars_dragging = dragging

    @property
    def height_bars_camera(self) -> "HeightBarCamera | None":
        """The camera the LAST height-bar render actually used."""

        return getattr(self, "_height_bars_rendered_camera", None)

    def _update_height_bars_artist(
        self,
        axes: Any,
        values: np.ndarray,
        valid: np.ndarray,
        extent: tuple[float, float, float, float],
        state: DisplayState,
        key: str,
        color_limits: tuple[float, float] | None,
        cmap_name: str,
        cmap: Any,
        valid_identity: object = None,
    ) -> tuple[Any, Any]:
        """Paint the value grid as shaded outlined boxes via the raster.

        The scene replaces only the data region: the same artist, clim,
        colormap, colorbar and distribution rail keep their meanings, so
        a colour-limit drag recolours the bars exactly as it recolours
        the heatmap.
        """

        from ._height3d_raster import HeightBarCamera, render_height_bars

        # The colour-limit preview re-renders the scene with candidate
        # limits (a clim drag moves bar HEIGHTS too -- the z axis and the
        # colorbar are one scale), so the call that painted it is kept.
        self._height_bars_calls[key] = (
            axes, values, valid, extent, state, cmap_name, cmap,
            valid_identity,
        )
        policy = self.style.render
        if axes.axison:
            # The 2D ticks and spines say nothing about a 3D scene.
            axes.set_axis_off()
            self._mark_axes_chrome_dirty(axes)
        if axes.get_aspect() != "auto":
            axes.set_aspect("auto")
        # Everything here depends on (data revision, validity, colour
        # limits, colormap) alone -- never on the camera.  A camera
        # commit on a 1000x2000 scan spent more time REDERIVING these
        # (masking, code quantization, the LUT gather) than rendering,
        # so they cache under exactly the facts they derive from.  The
        # cached arrays double as identity anchors for the kernel's
        # pooling cache below.
        # ``valid`` itself is rebuilt (broadcast) on every render, so its
        # id would bust this key each frame; the caller hands the STABLE
        # identity of the validity it was given -- the same lesson the
        # image range cache already carries.
        input_key = (
            self._data_revision,
            id(values),
            values.shape,
            values.strides,
            values.dtype.str,
            valid_identity,
            None
            if color_limits is None
            else (float(color_limits[0]), float(color_limits[1])),
            cmap_name,
        )
        cached_inputs = self._artists.get("image:h3d_inputs")
        if cached_inputs is not None and cached_inputs[0] == input_key:
            heights, top_rgb, low, high, zero_rgb = cached_inputs[1]
        else:
            heights = np.asarray(values, dtype=np.float64)
            usable = (
                _valid_array(valid, heights.shape)
                if valid is not None
                else np.ones(heights.shape, dtype=bool)
            )
            heights = np.where(usable, heights, np.nan)

            if color_limits is not None and color_limits[1] > color_limits[0]:
                low, high = (float(value) for value in color_limits)
            else:
                finite = heights[np.isfinite(heights)]
                low = float(finite.min()) if finite.size else 0.0
                high = float(finite.max()) if finite.size else 1.0
                if high <= low:
                    high = low + 1.0
            lut = self._image_color_lut(cmap_name, cmap)
            with np.errstate(invalid="ignore"):
                codes = np.clip(
                    (heights - low) * (256.0 / (high - low)), 0.0, 255.0
                )
            codes = np.where(np.isfinite(codes), codes, 0.0).astype(np.uint8)
            top_rgb = lut[codes][..., :3].astype(np.float32) / np.float32(255.0)
            with np.errstate(invalid="ignore"):
                zero_code = int(
                    np.clip((0.0 - low) * (256.0 / (high - low)), 0.0, 255.0)
                )
            zero_rgb = tuple(
                float(v) for v in lut[zero_code][:3].astype(np.float32) / 255.0
            )
            self._artists["image:h3d_inputs"] = (
                input_key,
                (heights, top_rgb, low, high, zero_rgb),
            )

        camera = HeightBarCamera(
            azimuth_deg=float(state["camera_azimuth"]),
            elevation_deg=float(state["camera_elevation"]),
            zoom=float(state["camera_zoom"]),
        )
        self._height_bars_rendered_camera = camera

        box = axes.bbox
        box_w = max(int(round(float(box.width))), 8)
        box_h = max(int(round(float(box.height))), 8)
        divisor = 1
        if self._height_bars_dragging:
            # The drag's resolution is a BUDGET, not a constant.  A
            # fixed divisor renders whatever the data costs: on a 2048x2048
            # scan that is hundreds of milliseconds a frame, so the drag
            # collapsed to a few frames per second while the hand kept
            # moving.  The divisor now tracks what the last drag frame
            # actually cost, so an interactive frame rate survives any
            # grid size -- the committed frame is untouched, being drawn
            # at divisor 1 by definition.
            divisor = max(int(policy.height_bars_drag_resolution_divisor), 1)
        vector_outlines = (
            heights.size <= policy.height_bars_supersample_tiny_bars
            and divisor == 1
        )
        # Vertical anti-aliasing is ANALYTIC (exact coverage), so the
        # only sampling knob left is horizontal: three subcolumn taps on
        # every committed frame, whatever the grid size.  The camera
        # DRAG is the fast lane instead.
        supersample = 1 if divisor != 1 else 3
        frame, scene = render_height_bars(
            heights,
            top_rgb,
            camera=camera,
            value_limits=(low, high),
            zero_rgb=zero_rgb,
            width=max(box_w // divisor, 8),
            height=max(box_h // divisor, 8),
            supersample=supersample,
            pool_cache=self._artists.setdefault("image:h3d_pool_cache", {}),
            side_shades=policy.height_bars_side_shades,
            edge_darken=policy.height_bars_edge_darken,
            background_rgb=policy.height_bars_background_rgb,
            z_fraction=policy.height_bars_z_fraction,
            bar_edges=not vector_outlines,
            display_stretch=float(divisor),
            # Committed geometry decides the pooled grid, so a drag
            # frame reuses it instead of re-pooling at its own size.
            pool_reference_width=box_w * 3,
        )
        self._height_bars_scene_map = scene
        self._height_bars_values = heights
        self._height_bars_axes_id = id(axes)
        finite_heights = heights[np.isfinite(heights)]
        lowest = float(finite_heights.min()) if finite_heights.size else 0.0
        self._height_bars_floor_value = float(np.clip(min(lowest, 0.0), low, 0.0))
        self._height_bars_data_frame = (
            tuple(float(v) for v in extent),
            int(heights.shape[1]),
            int(heights.shape[0]),
        )

        scene_extent = (0.0, 1.0, 0.0, 1.0)
        image = self._artists.get(key)
        if image is None:
            image = axes.imshow(
                frame,
                origin=policy.image_origin,
                aspect="auto",
                extent=scene_extent,
                interpolation="nearest",
            )
            self._artists[key] = image
        else:
            image.set_data(frame)
            extent_key = f"{key}:applied_extent"
            if self._artists.get(extent_key) != scene_extent:
                image.set_extent(scene_extent)
                self._artists[extent_key] = scene_extent
        self._artists[f"{key}:applied_front"] = frame
        self._artists[f"{key}:color_mode"] = "rgba"
        # The artist stays the clim/cmap authority every consumer reads.
        image.set_cmap(cmap)
        image.set_clim(low, high)
        self._home_limits[id(axes)] = ((0.0, 1.0), (0.0, 1.0))
        self._set_xlim(axes, 0.0, 1.0)
        self._set_ylim(axes, 0.0, 1.0)
        self._update_height_bars_chrome(axes, key, scene, box_w, box_h)
        self._update_height_bars_outlines(
            axes, key, scene, heights if vector_outlines else None
        )
        return image, cmap

    def _height_bars_fraction(
        self, scene: Any, x: float, y: float
    ) -> tuple[float, float]:
        """Scene pixel (top-origin) -> axes-fraction coordinates."""

        return x / max(scene.width, 1), 1.0 - y / max(scene.height, 1)

    def _update_height_bars_chrome(
        self, axes: Any, key: str, scene: Any, box_w: int, box_h: int
    ) -> None:
        """The scene's axis chrome: z ticks/labels and base coordinate labels.

        Text stays Matplotlib text at the style's fonts -- only positions
        come from the projection, so the 3D scene reads in the same
        typography as every 2D panel.
        """

        from matplotlib.ticker import MaxNLocator

        chrome_key = f"{key}:h3d_chrome"
        artists = self._artists.get(chrome_key)
        if artists is None:
            artists = {"lines": None, "grid": None, "texts": []}
            self._artists[chrome_key] = artists

        import matplotlib as _matplotlib

        segments_x: list[float] = []
        segments_y: list[float] = []

        def add_segment(p0, p1):
            f0 = self._height_bars_fraction(scene, *p0)
            f1 = self._height_bars_fraction(scene, *p1)
            segments_x.extend((f0[0], f1[0], np.nan))
            segments_y.extend((f0[1], f1[1], np.nan))

        wanted_texts: list[tuple[float, float, str, str, str]] = []
        # The SAME chrome metrics every 2D panel runs under: tick length,
        # tick pad and line width come from the style's rc context, in
        # points, converted at this figure's dpi.
        dots_per_point = float(axes.figure.dpi) / 72.0
        tick_length_px = (
            float(_matplotlib.rcParams["xtick.major.size"]) * dots_per_point
        )
        tick_pad_px = (
            float(_matplotlib.rcParams["xtick.major.pad"]) * dots_per_point
        )
        label_gap_px = tick_length_px + tick_pad_px
        # Point metrics are CANVAS pixels; fractions must divide by the
        # canvas box, not the scene raster -- a drag preview renders the
        # raster at a fraction of the box and stretches it, and dividing
        # by the reduced raster inflated every tick and label gap by the
        # divisor: the scene held still while its labels flew out.

        # The pane grid follows the reference (MATLAB) convention: RULES
        # sit at tick positions, and every rule runs the FULL display
        # limits.  The open-boundary look needs nothing special: the
        # limits themselves carry no tick, so no vertical rule ever
        # lands on the pane border -- horizontals reach the border and
        # end unclosed.  Grid rules, axis ticks and wall rules all read
        # the same tick lists: one authority, never two.
        grid_edges: list[tuple[tuple[float, float, float],
                               tuple[float, float, float]]] = []
        z_low, z_high = scene.value_low, scene.value_high
        wall_low = min(z_low, 0.0)
        wall_high = max(z_high, 0.0)
        # nbins=2 is the reference's tick sparseness: a [0, 1] scale
        # reads 0 / 0.5 / 1 exactly as the MATLAB panels do (nbins=3
        # picked a 0.4 step whose top tick fell outside the range).
        z_ticks = [
            float(tick)
            for tick in MaxNLocator(nbins=2).tick_values(z_low, z_high)
            if z_low - 1e-12 <= tick <= z_high + 1e-12
        ]
        for tick in z_ticks:
            grid_edges.append(
                ((0.0, float(scene.ny), tick),
                 (float(scene.nx), float(scene.ny), tick))
            )
            grid_edges.append(((0.0, 0.0, tick), (0.0, float(scene.ny), tick)))

        def outward(edge_point, inner_point):
            """Unit direction (axes fractions) pushing a label AWAY from
            the scene: the projected direction of the axis departing from
            its edge, the reference's tick convention."""

            f_edge = self._height_bars_fraction(scene, *edge_point)
            f_inner = self._height_bars_fraction(scene, *inner_point)
            vx = (f_edge[0] - f_inner[0]) * box_w
            vy = (f_edge[1] - f_inner[1]) * box_h
            length = float(np.hypot(vx, vy))
            if length <= 0.0:
                return (0.0, -1.0), f_edge
            return (vx / length, vy / length), f_edge

        def anchored(direction):
            ux, uy = direction
            ha = "left" if ux > 0.4 else ("right" if ux < -0.4 else "center")
            va = "bottom" if uy > 0.4 else ("top" if uy < -0.4 else "center")
            return ha, va

        # ---- z axis along the left pane's front edge, at ground (0, 0)
        base = scene.project(0.0, 0.0, wall_low)
        top = scene.project(0.0, 0.0, wall_high)
        add_segment(base, top)
        for tick in z_ticks:
            # The z labels leave their axis along the projected x
            # direction (the bench's ruling), exactly as the floor labels
            # leave theirs.
            (ux, uy), f = outward(
                scene.project(0.0, 0.0, float(tick)),
                scene.project(1.0, 0.0, float(tick)),
            )
            segments_x.extend(
                (f[0], f[0] + ux * tick_length_px / box_w, np.nan)
            )
            segments_y.extend(
                (f[1], f[1] + uy * tick_length_px / box_h, np.nan)
            )
            ha, va = anchored((ux, uy))
            wanted_texts.append(
                (
                    f[0] + ux * label_gap_px / box_w,
                    f[1] + uy * label_gap_px / box_h,
                    f"{tick:g}",
                    ha,
                    va,
                )
            )

        # ---- base coordinate labels along the two front edges
        data_frame = getattr(self, "_height_bars_data_frame", None)
        if data_frame is not None:
            (left, right, bottom, top_c), source_nx, source_ny = data_frame
            # Labels hug the lowest geometry actually drawn: the floor at
            # z=0, or the deepest hanging bar.  Anchoring at the colour
            # limit floated them mid-air below the scene whenever the
            # limits reach below the data.
            base_value = getattr(self, "_height_bars_floor_value", 0.0)

            def picks(count: int) -> list[int]:
                shown = min(count, 6)
                return sorted({
                    int(round(v))
                    for v in np.linspace(0, count - 1, shown)
                })

            # The two front floor edges are the scene's x/y axis lines,
            # carrying the tick marks exactly as a 2D panel's spines do.
            add_segment(
                scene.project(0.0, 0.0, base_value),
                scene.project(float(scene.nx), 0.0, base_value),
            )
            add_segment(
                scene.project(float(scene.nx), 0.0, base_value),
                scene.project(float(scene.nx), float(scene.ny), base_value),
            )

            def a_tick(centre: float, value: float) -> None:
                grid_edges.append(
                    ((centre, float(scene.ny), wall_low),
                     (centre, float(scene.ny), wall_high))
                )
                if not scene.dense:
                    grid_edges.append(
                        ((centre, 0.0, 0.0), (centre, float(scene.ny), 0.0))
                    )
                (ux, uy), f = outward(
                    scene.project(centre, 0.0, base_value),
                    scene.project(centre, 1.0, base_value),
                )
                segments_x.extend(
                    (f[0], f[0] + ux * tick_length_px / box_w, np.nan)
                )
                segments_y.extend(
                    (f[1], f[1] + uy * tick_length_px / box_h, np.nan)
                )
                ha, va = anchored((ux, uy))
                wanted_texts.append(
                    (
                        f[0] + ux * label_gap_px / box_w,
                        f[1] + uy * label_gap_px / box_h,
                        f"{value:g}",
                        ha,
                        va,
                    )
                )

            def b_tick(centre: float, value: float) -> None:
                grid_edges.append(
                    ((0.0, centre, wall_low), (0.0, centre, wall_high))
                )
                if not scene.dense:
                    grid_edges.append(
                        ((0.0, centre, 0.0), (float(scene.nx), centre, 0.0))
                    )
                (ux, uy), f = outward(
                    scene.project(float(scene.nx), centre, base_value),
                    scene.project(float(scene.nx) - 1.0, centre, base_value),
                )
                segments_x.extend(
                    (f[0], f[0] + ux * tick_length_px / box_w, np.nan)
                )
                segments_y.extend(
                    (f[1], f[1] + uy * tick_length_px / box_h, np.nan)
                )
                ha, va = anchored((ux, uy))
                wanted_texts.append(
                    (
                        f[0] + ux * label_gap_px / box_w,
                        f[1] + uy * label_gap_px / box_h,
                        f"{value:g}",
                        ha,
                        va,
                    )
                )

            # The rot90 fold hands each source axis to a DIFFERENT front
            # edge on odd quadrants: whichever folded coordinate a source
            # cell varies along decides which edge carries its labels.
            even = scene.quadrant % 2 == 0
            x_picks = picks(source_nx)
            y_picks = picks(source_ny)
            trimmed = y_picks if even else x_picks
            if len(trimmed) > 2:
                # The shared far corner keeps one label, not two.
                del trimmed[-1]
            for column in x_picks:
                a, b = scene.fold_cell(0, column * scene.pool_x)
                value = left + (column + 0.5) * (right - left) / source_nx
                if even:
                    a_tick(a + 0.5, value)
                else:
                    b_tick(b + 0.5, value)
            for row in y_picks:
                a, b = scene.fold_cell(row * scene.pool_y, 0)
                value = top_c + (row + 0.5) * (bottom - top_c) / source_ny
                if even:
                    b_tick(b + 0.5, value)
                else:
                    a_tick(a + 0.5, value)

        # ---- the pane/floor grid, occluded like real geometry
        grid = artists["grid"]
        if grid is None:
            (grid,) = axes.plot(
                [], [],
                transform=axes.transAxes,
                color=self.style.render.height_bars_grid_rgb,
                alpha=self.style.render.height_bars_grid_alpha,
                linewidth=self.style.render.height_bars_grid_line_pt,
                solid_capstyle="butt",
                zorder=5,
                clip_on=True,
            )
            artists["grid"] = grid
        if grid_edges:
            edge_array = np.asarray(grid_edges, dtype=np.float64)
            grid_x, grid_y = self._height_bars_occluded_polyline(
                scene, edge_array
            )
            grid.set_data(grid_x, grid_y)
        else:
            grid.set_data([], [])
        grid.set_visible(True)

        line = artists["lines"]
        if line is None:
            (line,) = axes.plot(
                [], [],
                transform=axes.transAxes,
                color=self.style.render.height_bars_axis_color,
                linewidth=float(_matplotlib.rcParams["axes.linewidth"]),
                zorder=6,
                clip_on=True,
            )
            artists["lines"] = line
        line.set_data(segments_x, segments_y)
        line.set_visible(True)

        region = self._height_bars_chrome_region(axes)
        texts = artists["texts"]
        for index, (fx, fy, content, ha, va) in enumerate(wanted_texts):
            if index < len(texts):
                text = texts[index]
            else:
                text = axes.text(
                    0, 0, "",
                    transform=axes.transAxes,
                    fontsize=self.style.fonts.tick_pt,
                    zorder=6,
                )
                texts.append(text)
            text.set_position((fx, fy))
            text.set_text(content)
            text.set_horizontalalignment(ha)
            text.set_verticalalignment(va)
            text.set_visible(True)
            # A 3D label goes where the projection puts it, which is not a
            # place any layout reserved.  It gets the room the scene owns --
            # out to whatever sits on its right, down from under the title --
            # and is cut off at that edge instead of printing across a
            # neighbour that means something else.
            text.set_clip_box(region)
            text.set_clip_on(True)
        for text in texts[len(wanted_texts):]:
            text.set_visible(False)
        self._thin_overlapping_chrome(texts[: len(wanted_texts)])

    def _height_bars_chrome_region(self, axes: Any) -> Any:
        """The room a 3D scene's own labels may occupy, in device pixels.

        A 2D panel reserves margins for its chrome because it knows where
        its chrome goes.  A rotated 3D scene does not: a tick that sat
        below the floor a moment ago is beside the colorbar now.  So the
        scene is given a REGION rather than a margin -- its own box grown
        to whatever bounds it (the rail or colorbar on the right, the
        title above) and never past it.  Inside, labels may go anywhere
        the projection puts them; at the edge they are cut off, which is
        the honest thing for a label with nowhere left to go.
        """

        from matplotlib.transforms import Bbox

        figure = axes.figure
        box = axes.get_window_extent()
        left, bottom = 0.0, 0.0
        right = float(figure.bbox.width)
        top = float(figure.bbox.height)
        for role in ("distribution", "colorbar"):
            for neighbour in self._axes.get(role, ()):
                if neighbour is axes or not neighbour.get_visible():
                    continue
                other = neighbour.get_window_extent()
                if other.x0 >= box.x1:
                    right = min(right, float(other.x0))
                elif other.x1 <= box.x0:
                    left = max(left, float(other.x1))
        title = self._artists.get("figure:title")
        if title is not None and title.get_visible():
            try:
                extent = title.get_window_extent(
                    figure.canvas.get_renderer()
                )
            except (AttributeError, RuntimeError, ValueError):
                extent = None
            if extent is not None and extent.y0 >= box.y1:
                top = min(top, float(extent.y0))
        return Bbox.from_extents(
            min(left, box.x0),
            min(bottom, box.y0),
            max(right, box.x1),
            max(top, box.y1),
        )

    def _thin_overlapping_chrome(self, texts: list) -> None:
        """Drop 3D labels that would print across one already kept.

        Rotation is continuous and the labels move with it, so any fixed
        stride is wrong at some angle: two ticks that were a centimetre
        apart meet when the axis turns edge-on.  What can be measured is
        whether two labels actually collide, so that is what decides --
        and the ENDS are kept first, because an axis whose extremes are
        legible still says what it spans, while one thinned from the
        outside in says nothing at all.
        """

        visible = [text for text in texts if text.get_visible()]
        if len(visible) < 2:
            return
        renderer = getattr(visible[0].figure.canvas, "get_renderer", None)
        if renderer is None:
            return
        renderer = renderer()
        order = [0, len(visible) - 1] + list(range(1, len(visible) - 1))
        kept: list[Any] = []
        for position in order:
            text = visible[position]
            try:
                extent = text.get_window_extent(renderer)
            except (RuntimeError, ValueError):
                continue
            if any(extent.overlaps(other) for other in kept):
                text.set_visible(False)
                continue
            kept.append(extent)

    def _hide_height_bars_chrome(self, key: str) -> None:
        artists = self._artists.get(f"{key}:h3d_chrome")
        if artists:
            if artists["lines"] is not None:
                artists["lines"].set_visible(False)
            if artists["grid"] is not None:
                artists["grid"].set_visible(False)
            for text in artists["texts"]:
                text.set_visible(False)
        cage = self._artists.get(f"{key}:h3d_cage")
        if cage is not None:
            cage.set_visible(False)
        outline = self._artists.get(f"{key}:h3d_outlines")
        if outline is not None:
            outline.set_visible(False)

    def height_bars_pick(
        self, canvas_x: float, canvas_y: float
    ) -> tuple[float, float] | None:
        """Canvas pixel -> the picked bar's DATA coordinates, or None."""

        scene = getattr(self, "_height_bars_scene_map", None)
        data_frame = getattr(self, "_height_bars_data_frame", None)
        if scene is None or data_frame is None:
            return None
        axes = self.primary_axes
        box = axes.bbox
        local_x = float(canvas_x) - float(box.x0)
        local_y = float(box.y1) - float(canvas_y)
        picked = scene.pick(local_x, local_y)
        if picked is None:
            return None
        row, column = picked
        (left, right, bottom, top), source_nx, source_ny = data_frame
        x_value = left + (column + 0.5) * (right - left) / source_nx
        y_value = top + (row + 0.5) * (bottom - top) / source_ny
        return x_value, y_value

    def _height_bars_cell_of(
        self, x_value: float, y_value: float
    ) -> tuple[int, int] | None:
        data_frame = getattr(self, "_height_bars_data_frame", None)
        if data_frame is None:
            return None
        (left, right, bottom, top), source_nx, source_ny = data_frame
        if right == left or bottom == top:
            return None
        column = int(np.floor((x_value - left) / (right - left) * source_nx))
        row = int(np.floor((y_value - top) / (bottom - top) * source_ny))
        if not (0 <= column < source_nx and 0 <= row < source_ny):
            return None
        return row, column

    def _height_bars_occluded_polyline(
        self,
        scene: Any,
        edges: "np.ndarray",
        samples_per_edge: int = 24,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample 3D edges against the scene, NaN-ing occluded samples.

        ``edges`` is (E, 2, 3) folded (a, b, value) endpoints.  The
        occlusion test is EXACT and purely local.  The occluder is the
        whole BOX whose face the sample's pixel shows, and this camera's
        view rays rise toward the viewer within one pixel column, so:
        the box hides the sample iff it lies AHEAD of the sample along
        the ray and the ray is still BELOW the box top where it enters
        the box footprint.  The floor and the panes hide nothing.

        No centre-depth proxy: a cell's centre is up to half a cell
        nearer than its own boundary, and that approximation carved
        dashes into every edge along a neighbouring cell.  And no
        per-face plane test: a face is just where the ray was measured,
        the body behind it is what blocks.

        ``samples_per_edge`` is a floor: the density adapts to the
        LONGEST projected edge (a pane-wide grid rule needs its occlusion
        breaks at pixel granularity, not at 1/24 of its span), bounded by
        a total-sample budget so huge edge sets stay cheap.
        """

        E = edges.shape[0]
        end_x = edges[:, :, 0] * scene.ca + edges[:, :, 1] * scene.sa
        end_y = (
            -edges[:, :, 0] * scene.sa * scene.se
            + edges[:, :, 1] * scene.ca * scene.se
            + edges[:, :, 2] * scene.z_unit * scene.ce
        )
        longest = float(
            np.max(
                np.hypot(
                    np.diff(end_x, axis=1), np.diff(end_y, axis=1)
                )
            )
            * scene.scale
        ) if E else 0.0
        samples = int(min(
            max(samples_per_edge, longest / 2.0 + 2.0),
            max(samples_per_edge, 2_000_000 / max(E, 1)),
            512,
        ))
        from ._height3d_raster import _scanline_selected

        if E and samples > 1 and _scanline_selected():
            # The numba mirror of everything below -- pure float64 with
            # integer lookups, bit-identical by construction; the code
            # underneath remains the specification and the fallback.
            from ._height3d_scanline import _occlusion_samples

            xs = np.empty(E * (samples + 1), dtype=np.float64)
            ys = np.empty(E * (samples + 1), dtype=np.float64)
            _occlusion_samples(
                np.ascontiguousarray(edges),
                scene.id_plane,
                np.ascontiguousarray(scene.top_values),
                float(scene.ca),
                float(scene.sa),
                float(scene.se),
                float(scene.z_unit),
                float(scene.ce),
                float(scene.x_low),
                float(scene.y_high),
                float(scene.scale),
                np.int64(samples),
                xs,
                ys,
            )
            return xs, ys
        fractions = np.linspace(0.0, 1.0, samples)[None, :, None]
        points = edges[:, 0:1, :] + (
            edges[:, 1:2, :] - edges[:, 0:1, :]
        ) * fractions  # (E, N, 3)
        ga = points[..., 0]
        gb = points[..., 1]
        gz = points[..., 2]
        sx = ga * scene.ca + gb * scene.sa
        sy = (
            -ga * scene.sa * scene.se
            + gb * scene.ca * scene.se
            + gz * scene.z_unit * scene.ce
        )
        px = (sx - scene.x_low) * scene.scale
        py = (scene.y_high - sy) * scene.scale
        columns = np.clip(px.astype(np.int64), 0, scene.width - 1)
        rows = np.clip(py.astype(np.int64), 0, scene.height - 1)
        face = scene.id_plane[rows, columns]
        shown_cell = (face - 4) >> 2
        shown_a = shown_cell % scene.nx
        shown_b = shown_cell // scene.nx
        shown_top = scene.top_values[
            np.clip(shown_b, 0, scene.ny - 1),
            np.clip(shown_a, 0, scene.nx - 1),
        ]
        # Half a pixel of height tolerance absorbs the raster rounding of
        # which face a borderline pixel shows.
        z_slack = 0.5 / max(scene.z_unit * scene.ce * scene.scale, 1e-9)
        # Toward the viewer the ray walks (+sa, -ca) over the ground and
        # RISES by this many value units per ground unit walked.
        rise = scene.se / (scene.z_unit * scene.ce)
        enter = np.maximum(
            np.maximum(
                (shown_a - ga) / scene.sa,
                (gb - shown_b - 1) / scene.ca,
            ),
            0.0,
        )
        leave = np.minimum(
            (shown_a + 1 - ga) / scene.sa,
            (gb - shown_b) / scene.ca,
        )
        hidden = (
            (face >= 4)
            & (leave > 1e-9)
            & (gz + enter * rise < shown_top - z_slack)
        )
        # A sample INSIDE solid geometry -- within some cell's footprint
        # and below that cell's top -- is hidden whatever its pixel
        # shows.  This is pixel-independent, so it also catches seam
        # samples whose rounded pixel falls on the floor or a pane
        # (the pixel test alone leaked dots there), and it conceals a
        # lower bar's rim where a taller neighbour's body covers it.
        # A sample exactly ON a cell boundary belongs to the VIEWER-side
        # cell -- that is the only body that can stand in front of it.
        # The ray walks (+a, -b) toward the viewer, so that is floor()
        # along a but ceil()-1 along b (a plain floor put a b-boundary
        # sample in the far cell and leaked dots along the pane seam).
        cell_a = np.floor(ga).astype(np.int64)
        cell_b = np.ceil(gb).astype(np.int64) - 1
        inside = (
            (cell_a >= 0) & (cell_a < scene.nx)
            & (cell_b >= 0) & (cell_b < scene.ny)
        )
        inside_top = np.where(
            inside,
            scene.top_values[
                np.clip(cell_b, 0, scene.ny - 1),
                np.clip(cell_a, 0, scene.nx - 1),
            ],
            -np.inf,
        )
        hidden |= gz < inside_top - z_slack
        # A visible sample with hidden samples on BOTH sides is pixel
        # rounding poking through at a corner -- a stray dot, not a line
        # segment.  Erode it.
        visible = ~hidden
        alone = visible.copy()
        alone[:, 1:] &= hidden[:, :-1]
        alone[:, :-1] &= hidden[:, 1:]
        hidden |= alone
        fx = px / max(scene.width, 1)
        fy = 1.0 - py / max(scene.height, 1)
        fx = np.where(hidden, np.nan, fx)
        fy = np.where(hidden, np.nan, fy)
        # One NaN column between edges keeps them separate polylines.
        pad = np.full((E, 1), np.nan)
        xs = np.concatenate([fx, pad], axis=1).ravel()
        ys = np.concatenate([fy, pad], axis=1).ravel()
        return xs, ys

    def _height_bars_box_edges(
        self, scene: Any, cells: "list[tuple[int, int]]",
        z_low: "np.ndarray", z_high: "np.ndarray",
    ) -> np.ndarray:
        """The box edges of each cell -> (E, 2, 3) folded points."""

        edges = []
        pixel_z = 1.0 / max(scene.z_unit * scene.ce * scene.scale, 1e-9)
        for index, (row, column) in enumerate(cells):
            a, b = scene.fold_cell(row, column)
            low = float(z_low[index])
            high = float(z_high[index])
            corners = ((0, 0), (1, 0), (1, 1), (0, 1))
            bottom = [(a + da, b + db, low) for da, db in corners]
            top = [(a + da, b + db, high) for da, db in corners]
            # A box flatter than a pixel keeps only its top rectangle.
            flat_box = (high - low) < pixel_z
            for i in range(4):
                edges.append((top[i], top[(i + 1) % 4]))
                if not flat_box:
                    edges.append((bottom[i], bottom[(i + 1) % 4]))
                    edges.append((bottom[i], top[i]))
        return np.asarray(edges, dtype=np.float64)

    @staticmethod
    def _height_bars_node_verticals(
        folded: "np.ndarray",
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """The EXACT vertical edge set of the drawn surface, per node.

        Two kinds of vertical line exist at a grid node, both derived
        from the four surrounding quadrant heights (absent cells count
        as floor-height zero) -- no thresholds, no heuristics:

        * CREASES of the box-union surface: at a height where the set
          of occupied quadrants forms a corner (one quadrant, three
          quadrants, or two diagonal ones), the surface turns and owns
          a vertical edge.  Two ADJACENT occupied quadrants make a
          straight wall through the node -- no edge.
        * SEAMS between coplanar side faces (the bar3 convention that
          also draws every cell boundary ON the surface): where the two
          half-walls on one wall plane both expose faces in the SAME
          direction, their overlap carries the seam.

        Emitting per NODE keeps every vertical single-stroked: boxes
        share these edges, and per-box copies used to double-draw the
        overlap into a visibly fatter line.
        """

        ny, nx = folded.shape

        def quadrant(a: int, b: int) -> float | None:
            if 0 <= a < nx and 0 <= b < ny:
                value = folded[b, a]
                if np.isfinite(value):
                    return float(value)
            return None

        segments: list[
            tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = []
        for b0 in range(ny + 1):
            for a0 in range(nx + 1):
                cells = {
                    "mm": quadrant(a0 - 1, b0 - 1),
                    "pm": quadrant(a0, b0 - 1),
                    "mp": quadrant(a0 - 1, b0),
                    "pp": quadrant(a0, b0),
                }
                if all(v is None for v in cells.values()):
                    continue
                spans: list[tuple[float, float]] = []
                for sign in (1.0, -1.0):
                    # sign=+1 handles the tops above the floor, sign=-1
                    # mirrors the analysis for hanging (negative) boxes.
                    h = {
                        k: (0.0 if v is None else max(sign * v, 0.0))
                        for k, v in cells.items()
                    }
                    levels = sorted({v for v in h.values() if v > 0.0})
                    lo = 0.0
                    for hi in levels:
                        occupied = {
                            k for k, v in h.items() if v >= hi - 1e-12
                        }
                        corner = (
                            len(occupied) in (1, 3)
                            or occupied in ({"mm", "pp"}, {"pm", "mp"})
                        )
                        if corner:
                            spans.append((sign * lo, sign * hi))
                        lo = hi
                    for near, far in (
                        (("mm", "pm"), ("mp", "pp")),  # wall a = a0
                        (("mm", "mp"), ("pm", "pp")),  # wall b = b0
                    ):
                        d_near = h[near[1]] - h[near[0]]
                        d_far = h[far[1]] - h[far[0]]
                        if (
                            d_near == 0.0
                            or d_far == 0.0
                            or (d_near > 0.0) != (d_far > 0.0)
                        ):
                            continue
                        seam_lo = max(
                            min(h[near[0]], h[near[1]]),
                            min(h[far[0]], h[far[1]]),
                        )
                        seam_hi = min(
                            max(h[near[0]], h[near[1]]),
                            max(h[far[0]], h[far[1]]),
                        )
                        if seam_hi > seam_lo:
                            spans.append((sign * seam_lo, sign * seam_hi))
                if not spans:
                    continue
                # Merge overlapping spans so every vertical strokes once.
                ordered = sorted(
                    (min(z0, z1), max(z0, z1)) for z0, z1 in spans
                )
                merged = [list(ordered[0])]
                for z0, z1 in ordered[1:]:
                    if z0 <= merged[-1][1] + 1e-12:
                        merged[-1][1] = max(merged[-1][1], z1)
                    else:
                        merged.append([z0, z1])
                for z0, z1 in merged:
                    segments.append((
                        (float(a0), float(b0), z0),
                        (float(a0), float(b0), z1),
                    ))
        return segments

    def _height_bars_dedupe_edges(self, edges: "np.ndarray") -> np.ndarray:
        """Draw every coincident edge ONCE.

        Neighbouring equal-height boxes share edges; each box emitting
        its own copy double-strokes the antialiased line into a heavier
        one.  Occlusion is a local test, so which copy survives cannot
        matter.
        """

        if edges.shape[0] < 2:
            return edges
        rounded = np.round(edges, 9)
        p0, p1 = rounded[:, 0, :], rounded[:, 1, :]
        flip = (
            (p1[:, 0] < p0[:, 0])
            | ((p1[:, 0] == p0[:, 0]) & (p1[:, 1] < p0[:, 1]))
            | (
                (p1[:, 0] == p0[:, 0])
                & (p1[:, 1] == p0[:, 1])
                & (p1[:, 2] < p0[:, 2])
            )
        )
        low = np.where(flip[:, None], p1, p0)
        high = np.where(flip[:, None], p0, p1)
        key = np.concatenate([low, high], axis=1)
        order = np.lexsort((
            key[:, 5], key[:, 4], key[:, 3],
            key[:, 2], key[:, 1], key[:, 0],
        ))
        ordered = key[order]
        first = np.ones(order.shape[0], dtype=bool)
        first[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
        return edges[order[first]]

    def _update_height_bars_outlines(
        self, axes: Any, key: str, scene: Any, heights: "np.ndarray | None"
    ) -> None:
        """Vector bar outlines for small grids: anti-aliased artist lines
        at the style's axes line width, occluded like real geometry."""

        import matplotlib as _matplotlib

        outline = self._artists.get(f"{key}:h3d_outlines")
        if heights is None:
            if outline is not None:
                outline.set_visible(False)
            return
        finite = np.isfinite(heights)
        rows_grid, cols_grid = np.nonzero(finite)
        if rows_grid.size == 0:
            if outline is not None:
                outline.set_visible(False)
            return
        clipped = np.where(
            finite,
            np.clip(heights, scene.value_low, scene.value_high),
            np.nan,
        )
        # Outline geometry quantizes to PIXEL resolution -- heights the
        # display cannot distinguish become exactly equal, so their
        # coincident lines dedupe to one.  The resolution of the screen
        # is the only "threshold" anywhere in this pass; everything else
        # is the exact geometry of the drawn surface.
        # The edge SET depends on (heights identity, that quantum, the
        # fold quadrant) alone -- an azimuth orbit inside one quadrant
        # replays it, so only the occlusion SAMPLING reruns per camera.
        pixel_z = 1.0 / max(scene.z_unit * scene.ce * scene.scale, 1e-9)
        geometry_key = (
            id(heights), heights.shape, repr(pixel_z), scene.quadrant
        )
        cached_geometry = self._artists.get(f"{key}:h3d_outline_geometry")
        if cached_geometry is not None and cached_geometry[0] == geometry_key:
            edges = cached_geometry[1]
            xs, ys = self._height_bars_occluded_polyline(scene, edges)
            outline = self._height_bars_outline_artist(axes, key)
            outline.set_data(xs, ys)
            outline.set_visible(True)
            return
        quantized = np.round(clipped / pixel_z) * pixel_z
        # The scene works in FOLDED orientation; flip the height grid to
        # match so neighbourhoods read directly.
        folded = (
            np.rot90(quantized, scene.quadrant)
            if scene.quadrant
            else quantized
        )
        if folded.shape != (scene.ny, scene.nx):
            # A pooled scene's outline cells are the POOLED boxes; the
            # generic per-cell path handles the fold-plus-pool mapping.
            cells = list(zip(rows_grid.tolist(), cols_grid.tolist()))
            values = quantized[rows_grid, cols_grid]
            edges = self._height_bars_box_edges(
                scene, cells, np.minimum(values, 0.0),
                np.maximum(values, 0.0),
            )
            edges = self._height_bars_dedupe_edges(edges)
            self._artists[f"{key}:h3d_outline_geometry"] = (
                geometry_key, edges
            )
            xs, ys = self._height_bars_occluded_polyline(scene, edges)
            outline = self._height_bars_outline_artist(axes, key)
            outline.set_data(xs, ys)
            outline.set_visible(True)
            return
        padded = np.full(
            (folded.shape[0] + 2, folded.shape[1] + 2), np.nan
        )
        padded[1:-1, 1:-1] = folded
        edge_list: list[tuple[tuple[float, float, float],
                              tuple[float, float, float]]] = []
        grid_ny, grid_nx = folded.shape
        corners = ((0, 0), (1, 0), (1, 1), (0, 1))
        for b in range(grid_ny):
            for a in range(grid_nx):
                h = folded[b, a]
                if not np.isfinite(h):
                    continue
                low, high = min(h, 0.0), max(h, 0.0)
                bottom = [(a + da, b + db, low) for da, db in corners]
                top = [(a + da, b + db, high) for da, db in corners]
                # rim i runs corner[i] -> corner[i+1]; its neighbour:
                neighbours = (
                    padded[b, a + 1],      # rim 0: constant b side
                    padded[b + 1, a + 2],  # rim 1: constant a+1 side
                    padded[b + 2, a + 1],  # rim 2: constant b+1 side
                    padded[b + 1, a],      # rim 3: constant a side
                )
                for i in range(4):
                    other = neighbours[i]
                    # Both horizontal creases of every boundary are real
                    # geometry: the top ring stays CLOSED at the box's
                    # own height (coincident rims of equal boxes dedupe
                    # to one), and where the surface drops to the floor
                    # -- an absent neighbour -- the foot line closes the
                    # face against the floor.
                    edge_list.append((top[i], top[(i + 1) % 4]))
                    if not np.isfinite(other) and high > low:
                        edge_list.append(
                            (bottom[i], bottom[(i + 1) % 4])
                        )
        edge_list.extend(self._height_bars_node_verticals(folded))
        if not edge_list:
            if outline is not None:
                outline.set_visible(False)
            return
        edges = np.asarray(edge_list, dtype=np.float64)
        edges = self._height_bars_dedupe_edges(edges)
        self._artists[f"{key}:h3d_outline_geometry"] = (geometry_key, edges)
        xs, ys = self._height_bars_occluded_polyline(scene, edges)
        outline = self._height_bars_outline_artist(axes, key)
        outline.set_data(xs, ys)
        outline.set_visible(True)

    def _height_bars_outline_artist(self, axes: Any, key: str) -> Any:
        import matplotlib as _matplotlib

        outline = self._artists.get(f"{key}:h3d_outlines")
        if outline is None:
            (outline,) = axes.plot(
                [], [],
                transform=axes.transAxes,
                color=self.style.render.height_bars_axis_color,
                linewidth=float(_matplotlib.rcParams["axes.linewidth"]),
                solid_capstyle="butt",
                zorder=6,
                clip_on=True,
            )
            self._artists[f"{key}:h3d_outlines"] = outline
        return outline

    def _update_height_bars_cage(self, snapshot: Any) -> None:
        """A wireframe cage over the crosshair-selected bar, full z span.

        Semi-transparent grey, and OCCLUDED like real geometry: every
        edge is sampled against the scene's id plane, and a sample whose
        pixel shows a strictly NEARER bar drops out of the polyline --
        so the cage passes behind the bars that stand in front of it.
        """

        key = self.primary_surface[0]
        cage = self._artists.get(f"{key}:h3d_cage")
        scene = getattr(self, "_height_bars_scene_map", None)
        crosshair = next(
            (
                state
                for state in getattr(snapshot, "states", ())
                if getattr(state.kind, "value", None) == "crosshair"
            ),
            None,
        )
        readout = self._artists.get(f"{key}:h3d_cage_readout")
        if scene is None or crosshair is None:
            if cage is not None:
                cage.set_visible(False)
            if readout is not None:
                readout.set_visible(False)
            return
        cell = self._height_bars_cell_of(
            float(crosshair.value.x), float(crosshair.value.y)
        )
        if cell is None:
            if cage is not None:
                cage.set_visible(False)
            if readout is not None:
                readout.set_visible(False)
            return
        z_low = np.asarray([min(scene.value_low, 0.0)])
        z_high = np.asarray([max(scene.value_high, 0.0)])
        edges = self._height_bars_box_edges(scene, [cell], z_low, z_high)
        xs, ys = self._height_bars_occluded_polyline(
            scene, edges, samples_per_edge=48
        )

        axes = self.primary_axes
        if cage is None:
            (cage,) = axes.plot(
                [], [],
                transform=axes.transAxes,
                color=self.style.render.height_bars_grid_rgb,
                alpha=self.style.render.height_bars_cage_alpha,
                linewidth=self.style.render.height_bars_grid_line_pt,
                solid_capstyle="round",
                zorder=7,
                clip_on=True,
            )
            self._artists[f"{key}:h3d_cage"] = cage
        cage.set_data(xs, ys)
        cage.set_visible(True)
        self._update_height_bars_readout(key, axes, crosshair, cell)

    def _update_height_bars_readout(
        self, key: str, axes: Any, crosshair: Any, cell: tuple[int, int]
    ) -> None:
        """The selected bar's numbers, where the heatmap crosshair puts
        them: (x, y, z) in the axes corner, selector typography."""

        from ._selector_scene import _selector_number, _selector_precision

        values = getattr(self, "_height_bars_values", None)
        frame = getattr(self, "_height_bars_data_frame", None)
        scene = getattr(self, "_height_bars_scene_map", None)
        if values is None or frame is None or scene is None:
            return
        extent = frame[0]
        row, column = cell
        z = float(values[row, column])
        xp = _selector_precision(abs(extent[1] - extent[0]))
        yp = _selector_precision(abs(extent[3] - extent[2]))
        zp = _selector_precision(abs(scene.value_high - scene.value_low))
        text = (
            f"({_selector_number(float(crosshair.value.x), xp)}, "
            f"{_selector_number(float(crosshair.value.y), yp)}, "
            f"{_selector_number(z, zp)})"
        )
        readout = self._artists.get(f"{key}:h3d_cage_readout")
        policy = self.style.render
        inset = policy.axes_text_inset_fraction
        if readout is None:
            from matplotlib.colors import to_rgba

            readout = axes.text(
                1.0 - inset,
                1.0 - inset,
                text,
                transform=axes.transAxes,
                ha="right",
                va="top",
                fontsize=self.style.fonts.annotation_pt,
                fontfamily=self.style.fonts.resolved_family,
                color=to_rgba(self.style.palette.line_single),
                zorder=8,
            )
            self._artists[f"{key}:h3d_cage_readout"] = readout
        elif readout.get_text() != text:
            readout.set_text(text)
        readout.set_visible(True)

    def _image_color_lut(self, cmap_name: str, cmap: Any) -> np.ndarray:
        """The colormap's 256-entry uint8 RGBA table, cached per colormap."""

        lut_key = (cmap_name,)
        cached = self._artists.get("image:lut_cache")
        if cached is not None and cached[0] == lut_key:
            return cached[1]
        # Midpoint sampling reads back exactly the colormap's own internal
        # 256-slot table, so index ``floor(norm * 256)`` reproduces what
        # Matplotlib's scalar draw would have picked.
        lut = cmap((np.arange(256, dtype=float) + 0.5) / 256.0, bytes=True)
        lut.setflags(write=False)
        self._artists["image:lut_cache"] = (lut_key, lut)
        return lut

    def _image_rgba_front(
        self,
        key: str,
        prepared: PreparedImageFront,
        cmap_name: str,
        cmap: Any,
        color_limits: tuple[float, float],
    ) -> np.ndarray | None:
        """Compose the front's uint8 RGBA once per (front, colormap, limits)."""

        vmin, vmax = (float(value) for value in color_limits)
        if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
            return None
        cache_key = (id(prepared), cmap_name, vmin, vmax)
        cache_name = f"{key}:rgba_front"
        cached = self._artists.get(cache_name)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        lut = self._image_color_lut(cmap_name, cmap)
        values = np.asarray(prepared.values)
        if values.dtype.kind == "u" and values.dtype.itemsize <= 2:
            # Raw unsigned fronts color through one direct table over the
            # dtype's whole domain: a single integer gather, no float pass.
            table_key = (cmap_name, vmin, vmax, values.dtype.str)
            table_cached = self._artists.get("image:direct_color_table")
            if table_cached is not None and table_cached[0] == table_key:
                table = table_cached[1]
            else:
                domain = np.arange(
                    np.iinfo(values.dtype).max + 1, dtype=np.float32
                )
                table = lut[
                    np.clip(
                        (domain - np.float32(vmin))
                        * np.float32(256.0 / (vmax - vmin)),
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                ]
                self._artists["image:direct_color_table"] = (table_key, table)
            rgba = table[values]
        else:
            # In-place float32 passes; boundary pixels may differ from
            # Matplotlib's float64 normalize by one 256-level step, which is
            # the same quantization the colormap applies anyway.
            scaled = values.astype(np.float32, copy=True)
            scaled -= np.float32(vmin)
            scaled *= np.float32(256.0 / (vmax - vmin))
            np.clip(scaled, 0.0, 255.0, out=scaled)
            rgba = lut[scaled.astype(np.uint8)]
        rgba.setflags(write=False)
        self._artists[cache_name] = (cache_key, rgba)
        return rgba

    def _update_horizontal_histogram(
        self,
        axes: Any,
        key: str,
        edges: np.ndarray,
        counts: np.ndarray,
        y_limits: tuple[float, float],
        *,
        projection_changed: bool,
        tick_profile: str,
    ) -> None:
        from matplotlib.collections import PolyCollection

        policy = self.style.render
        collection = self._artists.get(key)
        if collection is None:
            collection = PolyCollection(
                _histogram_vertices(edges, counts)[..., ::-1],
                facecolors=self.style.palette.hist_fill,
                edgecolors="none",
                alpha=policy.side_distribution_fill_alpha,
            )
            axes.add_collection(collection)
            self._artists[key] = collection
            # This rail's numbers are a bound, not a coordinate: counts per
            # bin, always from zero.  Two ticks say all of it, at every
            # preset -- the crowding ladder gave it two on a wide panel and
            # none on a narrow one, which is a layout accident.
            apply_declared_ticks(
                axes,
                "x",
                policy.distribution_tick_count,
                label_pt=self.style.fonts.tick_pt,
            )
            self._artists[f"{key}:dynamic_axes"] = (axes.xaxis, axes.yaxis)
            if tick_profile == "image":
                axes.tick_params(
                    axis="y", left=True, right=False, labelleft=False, labelright=False
                )
            else:
                axes.set_xlabel("")
                axes.set_ylabel("")
                axes.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                )
                axes.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    top=False,
                )
        elif projection_changed:
            collection.set_verts(_histogram_vertices(edges, counts)[..., ::-1])
        peak = float(np.max(counts)) if counts.size else 0.0
        wanted = float(
            max(
                policy.distribution_count_floor,
                int(
                    max(
                        peak + policy.distribution_count_headroom,
                        peak * policy.distribution_count_growth,
                    )
                ),
            )
        )
        ceiling_key = f"{key}:count_ceiling"
        ceiling = float(self._artists.get(ceiling_key, 0.0))
        if (
            ceiling <= 0.0
            or wanted > ceiling
            or wanted < policy.distribution_count_shrink_ratio * ceiling
        ):
            ceiling = wanted
            self._artists[ceiling_key] = ceiling
        if tuple(map(float, axes.get_xlim())) != (0.0, ceiling):
            axes.set_xlim(0.0, ceiling)
        if tuple(map(float, axes.get_ylim())) != tuple(map(float, y_limits)):
            axes.set_ylim(*y_limits)

    def _update_image(
        self,
        axes: Any,
        payload: Any,
        state: DisplayState,
        key: str,
        *,
        color_limits: tuple[float, float] | None = None,
        paint_labels: bool = True,
    ) -> None:
        labels = getattr(self.semantic_spec, "labels", None)
        explicit_x = _state_label(
            state,
            "x_label",
            labels.x if labels and labels.x else None,
        )
        explicit_y = _state_label(
            state,
            "y_label",
            labels.y if labels and labels.y else None,
        )
        explicit_value = _state_label(
            state,
            "value_label",
            labels.value if labels and labels.value else None,
        )
        self._mutate_image_artists(
            axes,
            np.asarray(_display_array(payload.x)),
            np.asarray(_display_array(payload.y)),
            np.asarray(_display_array(payload.z)),
            getattr(payload, "valid", None),
            state,
            key,
            x_label=_quantity_label(payload.x, "x", explicit_x),
            y_label=_quantity_label(payload.y, "y", explicit_y),
            value_label=(
                explicit_value
                if explicit_value and _EXPLICIT_UNIT_SUFFIX.search(explicit_value)
                else _quantity_label(payload.z, "value", explicit_value)
            ),
            coordinate_aspect=_image_coordinate_aspect(payload.x, payload.y),
            color_limits=color_limits,
            paint_labels=paint_labels,
        )

    def _mutate_image_artists(
        self,
        axes: Any,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        valid: np.ndarray | None,
        state: DisplayState,
        key: str,
        *,
        x_label: str,
        y_label: str,
        value_label: str,
        coordinate_aspect: float | None,
        color_limits: tuple[float, float] | None = None,
        paint_labels: bool = True,
    ) -> None:
        source_values, source_valid = z, valid
        z, valid, extent = _image_arrays(x, y, z, valid)
        # Keyed on the arrays the CALLER was handed.  _image_arrays rebuilds
        # its broadcast view on every call, so keying on what it returns meant
        # the key was a fresh object each time and the cache never hit -- the
        # whole point of it being a function of the data alone.
        data_range = self._cached_image_range(
            key, source_values, source_valid, z, valid
        )
        # ``color_limits`` is the GRID's pooled scale when a FacetGrid overview
        # paints this cell: every cell of one grid must be comparable, so the
        # scale is a fact about the grid and this surface is told it.
        vmin, vmax = (
            self._resolve_image_limits(key, data_range, state)
            if color_limits is None
            else color_limits
        )
        if self._color_limit_candidate is not None:
            vmin = self._color_limit_candidate.value.low
            vmax = self._color_limit_candidate.value.high
        image, cmap = self._update_image_artist(
            axes,
            z,
            valid,
            extent,
            state,
            key,
            (vmin, vmax),
            square_view=coordinate_aspect is not None,
            coordinate_aspect=coordinate_aspect,
            valid_identity=(
                None
                if source_valid is None
                else (
                    id(source_valid),
                    np.shape(source_valid),
                )
            ),
        )
        if paint_labels:
            if axes.get_xlabel() != x_label:
                axes.set_xlabel(x_label)
            if axes.get_ylabel() != y_label:
                axes.set_ylabel(y_label)
        self._update_image_chrome(
            axes,
            key,
            z,
            valid,
            state,
            data_range,
            (vmin, vmax),
            cmap,
            value_label,
        )

    def _update_image_chrome(
        self,
        axes: Any,
        key: str,
        z: np.ndarray,
        valid: np.ndarray | None,
        state: DisplayState,
        data_range: tuple[float, float] | None,
        color_limits: tuple[float, float],
        cmap: Any,
        value_label: str,
    ) -> None:
        """Distribution, colorbar and spatial-tick chrome for ONE image axes.

        The standalone Image kind and a focused FacetGrid image cell both
        call this single authority, so the focused cell can never drift
        from the standalone chrome; ``key`` namespaces the cached artists
        (``image`` / ``facet:<i>:image``).
        """

        policy = self.style.render
        vmin, vmax = color_limits
        histogram_limits, distribution_limits = self._resolve_distribution_limits(
            key,
            data_range,
            (vmin, vmax),
        )
        cmap_name = str(state["colormap"])

        distribution_axes = self._axes.get("distribution", [])
        if distribution_axes:
            distribution = distribution_axes[0]
            bin_count = max(
                policy.image_distribution_min_bins,
                min(
                    max(int(z.size), 1) // policy.distribution_bin_divisor,
                    policy.distribution_max_bins,
                ),
            )
            cache_key = (
                self._data_revision,
                bin_count,
                histogram_limits,
            )
            cache_name = f"{key}:distribution_cache"
            cached = self._artists.get(cache_name)
            if cached is not None and cached[0] == cache_key:
                counts, edges = cached[1], cached[2]
                distribution_changed = False
            else:
                samples = _bounded_image_distribution_values(
                    z,
                    valid,
                    policy.image_distribution_sample_target,
                )
                if not samples.size:
                    samples = np.asarray([histogram_limits[0]], dtype=float)
                counts, edges = np.histogram(
                    samples,
                    bins=aligned_histogram_edges(
                        samples,
                        bin_count,
                        limits=histogram_limits,
                    ),
                )
                self._artists[cache_name] = (cache_key, counts, edges)
                distribution_changed = True

            distribution_key = f"{key}:distribution"
            self._update_horizontal_histogram(
                distribution,
                distribution_key,
                edges,
                counts,
                distribution_limits,
                projection_changed=distribution_changed,
                tick_profile="image",
            )

            guide_key = f"{key}:guides"
            guides = self._artists.get(guide_key)
            if guides is None:
                guides = tuple(
                    distribution.axhline(
                        value,
                        color=self.style.palette.guide,
                        linewidth=self.style.artists.side_distribution_linewidth,
                        alpha=policy.distribution_guide_alpha,
                    )
                    for value in (vmin, vmax)
                )
                self._artists[guide_key] = guides
            show_guides = data_range is not None
            for index, guide in enumerate(guides):
                if guide.get_visible() != show_guides:
                    guide.set_visible(show_guides)
                if show_guides:
                    assert data_range is not None
                    value = data_range[index]
                    guide.set_ydata((value, value))

        colorbar_axes = self._axes.get("colorbar", [])
        if colorbar_axes:
            colorbar_key = f"{key}:colorbar"
            colorbar = self._artists.get(colorbar_key)
            mappable_key = f"{key}:colorbar_mappable"
            mappable = self._artists.get(mappable_key)
            if colorbar is None:
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize

                # The colorbar reads a dedicated proxy mappable: the image
                # artist may hold precomposed RGBA, which carries no norm or
                # colormap for a colorbar to describe.
                mappable = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap)
                self._artists[mappable_key] = mappable
                colorbar = self._figure.colorbar(mappable, cax=colorbar_axes[0])
                self._artists[colorbar_key] = colorbar
            colorbar_state = (
                cmap_name,
                (vmin, vmax),
                value_label,
            )
            state_key = f"{key}:colorbar_state"
            previous_colorbar_state = self._artists.get(state_key)
            if colorbar_state != previous_colorbar_state:
                if mappable is not None:
                    if (
                        previous_colorbar_state is None
                        or previous_colorbar_state[0] != cmap_name
                    ):
                        mappable.set_cmap(cmap)
                    if (
                        previous_colorbar_state is None
                        or previous_colorbar_state[1] != (vmin, vmax)
                    ):
                        mappable.set_clim(vmin, vmax)
                if (
                    previous_colorbar_state is None
                    or previous_colorbar_state[2] != value_label
                ):
                    colorbar.set_label(
                        value_label,
                        labelpad=policy.colorbar_endpoint_label_pad_pt,
                    )
                if (
                    previous_colorbar_state is None
                    or previous_colorbar_state[1] != (vmin, vmax)
                ):
                    colorbar.set_ticks((vmin, vmax))
                    label_chars = policy.colorbar_endpoint_label_chars
                    colorbar.set_ticklabels(
                        (
                            _compact_engineering(vmin, length=label_chars),
                            _compact_engineering(vmax, length=label_chars),
                        )
                    )
                self._artists[state_key] = colorbar_state
            # Only the colorbar parts whose pixels can change belong above
            # the cached background.  Repainting its whole Axes repeated the
            # patch, hidden short Axis and empty title children every frame.
            self._artists[f"{key}:colorbar_dynamic_artists"] = (
                colorbar.solids,
                colorbar.dividers,
                colorbar.outline,
                colorbar.long_axis,
            )
        self._apply_colorbar_visibility(state)
        # A tick configuration has ONE owner per axes: it writes the axis'
        # ``_zlc_tick_signature`` and installs its locator only when that
        # signature changes.  In a FacetGrid OVERVIEW the grid owns cell ticks
        # (one shared 3-tick locator, boundary-gated labels, applied below the
        # cell loop); the image kind's own spatial budget owns them whenever
        # this surface is the whole plot -- standalone, or a focused cell.
        # With both writing, each frame reinstalled two locators per cell and
        # reset their tick artists, undoing the "install once" guarantee.
        if self._facet_focus_index is not None or self.semantic_spec is self.spec:
            # An image's data axes has the distribution rail beside it, not a
            # margin: an x edge label would print over the rail's own first
            # one.  Above and below it there is only the figure's margin, so
            # the y axis keeps the ends of its range.
            apply_smart_ticks(
                axes, "x", label_pt=self.style.fonts.tick_pt, prune_edges=True
            )
            apply_smart_ticks(axes, "y", label_pt=self.style.fonts.tick_pt)

    def _update_rolling(
        self,
        axes: Any,
        payload: Any,
        state: DisplayState,
        key: str,
    ) -> None:
        # A rolling plot's one surface IS its history axes.
        history = axes
        series = self._series(payload)
        sliced: list[_PreparedSeries] = []
        for item in series:
            y_values = np.asarray(_display_array(item.y), dtype=float).reshape(-1)
            x_values = np.asarray(_display_array(item.x), dtype=float).reshape(-1)
            valid = (
                _valid_array(
                    item,
                    _display_array(item.x).reshape(-1).shape,
                )
                & np.isfinite(x_values)
                & np.isfinite(y_values)
            )
            label = getattr(item, "label", "")
            if label is None:
                label = ""
            sliced.append(_PreparedSeries(
                x_values,
                y_values,
                valid,
                str(label),
                _series_identity(item),
                band=_series_band(item),
            ))

        labels = self.spec.labels
        explicit_y = _state_label(state, "y_label", None)
        if explicit_y is None:
            explicit_y = _state_label(
                state,
                "value_label",
                labels.y or labels.value,
            )
        payload_x = series[0].x.label if series else "Shot"
        explicit_x = _state_label(state, "x_label", labels.x or payload_x)
        y_label = (
            _quantity_label(series[0].y, "value", explicit_y)
            if series
            else ("value" if explicit_y is None else explicit_y)
        )
        self._mutate_series_artists(
            history,
            tuple(sliced),
            state,
            f"{key}:history",
            x_label=payload_x if explicit_x is None else explicit_x,
            y_label=y_label,
        )
        # The shot axis frames the FULL configured window from the first
        # revision on: it opens at shots [0, window-1] and slides only once
        # the trace has filled it, so the window parameter is what you see
        # and the axis never names a negative shot.
        shot_values = np.concatenate(
            [item.x[item.valid] for item in sliced]
        ) if sliced else np.asarray([], dtype=float)
        if shot_values.size:
            last_shot = float(np.max(shot_values))
            window = int(state["window"])
            low = max(0.0, last_shot - window + 1)
            frame = _curve_x_limits(np.asarray([low, low + window - 1]))
            if frame is not None:
                self._set_xlim(history, *frame)
        latest = None
        if sliced:
            usable = sliced[0].y[sliced[0].valid]
            if usable.size:
                latest = float(usable[-1])
        latest_text = self._artists.get(f"{key}:latest")
        if latest_text is None:
            latest_text = history.text(
                0.97,
                0.95,
                "",
                transform=history.transAxes,
                color=self.style.palette.readout,
                ha="right",
                va="top",
                fontsize=self.style.fonts.annotation_pt,
            )
            self._artists[f"{key}:latest"] = latest_text
        latest_text.set_text("" if latest is None else f"{latest:.6g}")

        distribution_axes = self._axes.get("distribution", [])
        if distribution_axes:
            policy = self.style.render
            samples = [item.y[item.valid] for item in sliced]
            values = np.concatenate(samples) if samples else np.asarray([], dtype=float)
            history_length = max((item.size for item in samples), default=0)
            requested_bins = int(state["bin_count"])
            bin_count = max(
                policy.rolling_distribution_min_bins,
                min(
                    requested_bins,
                    max(
                        history_length // policy.distribution_bin_divisor,
                        policy.rolling_distribution_min_bins,
                    ),
                    policy.distribution_max_bins,
                ),
            )
            y_limits = tuple(float(value) for value in history.get_ylim())
            counts, edges = np.histogram(
                values[np.isfinite(values)] if values.size else np.asarray([y_limits[0]]),
                bins=bin_count,
                range=tuple(sorted(y_limits)),
            )
            side = distribution_axes[0]
            self._update_horizontal_histogram(
                side,
                f"{key}:distribution",
                edges,
                counts,
                y_limits,
                projection_changed=True,
                tick_profile="rolling",
            )

    #: Side-chrome artist keys the focused image cell creates under its
    #: ``facet:<i>`` surface namespace.  Purged together with the side axes so
    #: a later focus rebuilds them on the new axes instead of ghosting.
    _FACET_FOCUS_CHROME_SUFFIXES = (
        "distribution",
        "distribution:dynamic_axes",
        "distribution_cache",
        "distribution:count_ceiling",
        "distribution_data_limits",
        "guides",
        "colorbar",
        "colorbar_mappable",
        "colorbar_state",
        "colorbar_dynamic_artists",
    )

    def _sync_facet_focus_chrome(self, index: int | None) -> None:
        """Create or destroy the focused image cell's side chrome axes.

        The distribution and colorbar axes exist exactly while one image
        cell is focused: they are facts of the presentation, not of the
        figure, so unfocusing destroys them (and their cached artists)
        rather than parking stale chrome behind a visibility flag.
        """

        previous = self._facet_focus_chrome_index
        if previous == index:
            return
        if previous is not None:
            key = f"facet:{previous}"
            for suffix in self._FACET_FOCUS_CHROME_SUFFIXES:
                self._artists.pop(f"{key}:{suffix}", None)
            removed: list[Any] = []
            for role in ("distribution", "colorbar"):
                removed.extend(self._axes.pop(role, ()))
            removed_ids = {id(axis) for axis in removed}
            # Selector scenes (the color rail and its handles) cache their
            # artists per scene kind; a scene living on a removed axes must
            # die with it or the compose path repaints it at the dead box.
            for kind, artists in tuple(self._selector_artists.items()):
                if any(
                    getattr(item, "axes", None) is not None
                    and id(item.axes) in removed_ids
                    for item in artists
                ):
                    self._remove_artists(artists)
                    self._selector_artists.pop(kind, None)
                    self._selector_topologies.pop(kind, None)
            for axis in removed:
                self._chrome_dirty_axes.discard(axis)
                axis.remove()
            self._background_region = None
            self._forget_gesture_region()
        if index is not None:
            assert self.plan.facet_focus_axes is not None
            for item in self.plan.facet_focus_axes:
                if item.role == "image":
                    continue
                axis = self._figure.add_axes(item.box.matplotlib_bounds())
                axis.set_gid(item.role)
                self._axes.setdefault(item.role, []).append(axis)
            self._background_region = None
            self._forget_gesture_region()
        self._facet_focus_chrome_index = index

    def _pixel_quantized_bounds(
        self,
        axis: Any,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Snap one cell's box to whole physical pixels.

        A cell box on fractional pixels denies the compose's exact image
        blit to every heatmap cell and smears its tick marks; the plan's
        normalized fractions land wherever the figure size puts them.
        Snapping moves an edge by under a pixel.  An aspect-locked cell
        additionally keeps its pixel box exactly ON the aspect (in the
        exactly-representable case, e.g. square data), so ``apply_aspect``
        preserves the snapped box instead of re-deriving a fractional one;
        an aspect this cannot represent keeps the plan's box unchanged.
        """

        figure_box = self._figure.bbox
        width = float(figure_box.width)
        height = float(figure_box.height)
        if width <= 1.0 or height <= 1.0:
            return bounds
        # Grow-only snapping: the box floor/ceils outward into the grid
        # gap, so the tick policy never sees LESS room than the plan gave
        # it (a half-pixel shrink at a pricing threshold dropped a cell
        # from five labels to three).
        x0 = math.floor(bounds[0] * width)
        y0 = math.floor(bounds[1] * height)
        x1 = math.ceil((bounds[0] + bounds[2]) * width)
        y1 = math.ceil((bounds[1] + bounds[3]) * height)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return bounds
        aspect = axis.get_aspect()
        if isinstance(aspect, (int, float)) and not isinstance(aspect, bool):
            x_limits = axis.get_xlim()
            y_limits = axis.get_ylim()
            x_span = abs(float(x_limits[1]) - float(x_limits[0]))
            y_span = abs(float(y_limits[1]) - float(y_limits[0]))
            if not (x_span > 0.0 and y_span > 0.0):
                return bounds
            ratio = float(aspect) * y_span / x_span
            box_w = x1 - x0
            box_h = y1 - y0
            wanted_h = box_w * ratio
            if abs(wanted_h - round(wanted_h)) < 1e-9 and round(wanted_h) >= 2:
                if round(wanted_h) <= box_h:
                    y1 = y0 + int(round(wanted_h))
                else:
                    wanted_w = box_h / ratio
                    if abs(wanted_w - round(wanted_w)) < 1e-9 and round(
                        wanted_w
                    ) >= 2:
                        x1 = x0 + int(round(wanted_w))
                    else:
                        return bounds
            else:
                return bounds
        return (
            x0 / width,
            y0 / height,
            (x1 - x0) / width,
            (y1 - y0) / height,
        )

    def _position_facet_axes_for_frame(self, axes: Sequence[Any]) -> None:
        """Apply final cell boxes before artists resolve pixel-dependent work."""

        focus_plans = (
            self.plan.facet_focus_axes
            if self._facet_focus_index is not None
            and isinstance(self.semantic_spec, ImagePlot)
            else None
        )
        self._sync_facet_focus_chrome(
            self._facet_focus_index if focus_plans is not None else None
        )
        if self._facet_focus_index is None:
            for index, axis in enumerate(axes):
                bounds = self._pixel_quantized_bounds(
                    axis, self.plan.axes[index].box.matplotlib_bounds()
                )
                # ``set_position`` invalidates the axes transform stack even
                # for an identical box; skip the per-frame no-op.  The
                # comparison needs a tolerance: Bbox round-trips width as
                # (x0+w)-x0, one ulp off the stored fraction, and an exact
                # compare re-positioned (and now re-captured chrome for)
                # every cell on every frame.
                current = tuple(axis.get_position().bounds)
                if any(
                    abs(now - wanted) > 1e-12
                    for now, wanted in zip(current, bounds)
                ):
                    axis.set_position(bounds)
                    # A moved box invalidates the captured chrome behind it.
                    self._mark_axes_chrome_dirty(axis)
                visible = index < self._visible_facet_count
                if axis.get_visible() != visible:
                    axis.set_visible(visible)
            return
        selected_index = self._facet_focus_index
        if focus_plans is not None:
            # A focused image cell IS the standalone Image surface: the cell
            # takes the split's image box, the side axes carry its chrome.
            bounds = next(
                item.box for item in focus_plans if item.role == "image"
            ).matplotlib_bounds()
        else:
            bounds = facet_focus_box(self.plan).matplotlib_bounds()
        for index, axis in enumerate(axes):
            visible = index == selected_index
            if axis.get_visible() != visible:
                axis.set_visible(visible)
        if tuple(axes[selected_index].get_position().bounds) != tuple(bounds):
            axes[selected_index].set_position(bounds)

    def _update_facets(self, payload: Any, state: DisplayState) -> None:

        cells = tuple(getattr(payload, "cells", ()))
        axes = self._axes.get("facet_cell", [])
        if self._visible_facet_count != len(cells):
            raise RuntimeError("FacetGrid payload and rendered topology are inconsistent")
        semantic = self.semantic_spec
        handler = handler_for(semantic)
        focused = self._facet_focus_index is not None
        # Facet focus changes the physical cell box.  Establish the complete
        # frame geometry before an image cell chooses its display raster.
        self._position_facet_axes_for_frame(axes)

        # Everything resolved below the loop is a fact about the GRID, not
        # about any cell: one colour scale, one x span, one histogram binning,
        # so that N cells of one measurement are comparable at a glance.  Each
        # is handed to the cell renders as the SAME keyword argument a
        # standalone plot of that kind already accepts.  It stays an explicit
        # three-branch on purpose: a per-kind hook would hide that the pooling
        # is the grid's decision, not the cell's.
        #
        # It is the grid's decision while ONE cell is open too.  Opening a cell
        # is looking closer at the same measurement, not opening another plot,
        # and skipping the pooling here let the cell resolve limits under its
        # own key: the same data in the same cell was painted two ways --
        # measured at 573 of 768 samples different, colour scale (-7.8, 85.8)
        # in the overview against (-2.6, 28.6) opened -- so what the operator
        # compared before double-clicking was not what they then examined, and
        # a saved figure's scale depended on which cell happened to be open.
        # The focused cell's own rail and colorbar are what show where it sits
        # inside the shared scale; authored limits are what override it.
        # Pooling costs one scan per cell per revision, cached under each
        # cell's own key -- the same price the overview already pays.
        cell_options: tuple[dict[str, Any], ...] = tuple({} for _ in cells)
        curve_series: tuple[tuple[_PreparedSeries, ...], ...] = ()
        curve_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        histogram_arrays: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
        histogram_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        if isinstance(semantic, CurvePlot):
            curve_series = tuple(
                self._prepare_curve_series(
                    self._series(getattr(cell, "payload", cell))
                )
                for cell in cells
            )
            x_groups: list[np.ndarray] = []
            y_groups: list[np.ndarray] = []
            for series in curve_series:
                for item in series:
                    if bool(np.any(item.valid)):
                        x_groups.append(item.x[item.valid])
                        y_groups.append(item.y[item.valid])
            if x_groups:
                x_target = _curve_x_limits(np.concatenate(x_groups))
                y_range = _data_limits(np.concatenate(y_groups))
                assert x_target is not None
                curve_limits = (
                    x_target,
                    self._resolve_curve_y_limits(
                        "facet:curve",
                        y_range,
                        state,
                    ),
                )
            cell_options = tuple(
                {"limits": curve_limits, "prepared_series": series}
                for series in curve_series
            )
        elif isinstance(semantic, HistogramPlot):
            histogram_arrays = tuple(
                self._histogram_arrays(getattr(cell, "payload", cell), state)
                for cell in cells
            )
            usable = tuple(item for item in histogram_arrays if item[0].size >= 2)
            if usable:
                shared_edges = usable[0][0]
                if any(
                    edges.shape != shared_edges.shape
                    or not np.array_equal(edges, shared_edges)
                    for edges, _counts in usable[1:]
                ):
                    raise ValueError("FacetGrid histogram cells must share one bin projection")
                # The shared edges already carry relim_mode's retention: the
                # session's histogram projection holds them under ``normal``
                # and ``fixed`` and recomputes them under ``tight``, so their
                # span applies directly — a second damping layer here would
                # override the authored mode.
                x_limits = (float(shared_edges[0]), float(shared_edges[-1]))
                pooled_counts = np.concatenate(
                    tuple(np.asarray(counts, dtype=float) for _edges, counts in usable)
                )
                histogram_limits = (
                    x_limits,
                    self._resolve_histogram_y_limits(
                        "facet:histogram",
                        pooled_counts,
                        state,
                    ),
                )
            cell_options = tuple(
                {"arrays": arrays, "limits": histogram_limits}
                for arrays in histogram_arrays
            )
        elif isinstance(semantic, ImagePlot):
            pooled_low: float | None = None
            pooled_high: float | None = None
            for index, cell in enumerate(cells):
                cell_payload = getattr(cell, "payload", cell)
                source_values = np.asarray(_display_array(cell_payload.z))
                source_valid = getattr(cell_payload, "valid", None)
                values, valid, _extent = _image_arrays(
                    np.asarray(_display_array(cell_payload.x)),
                    np.asarray(_display_array(cell_payload.y)),
                    source_values,
                    source_valid,
                )
                # Measured under the cell surface's OWN key, so the render
                # below reads this exact answer back out of the cache instead
                # of rescanning every cell a second time.
                cell_range = self._cached_image_range(
                    f"facet:{index}", source_values, source_valid, values, valid
                )
                if cell_range is not None:
                    pooled_low = (
                        cell_range[0]
                        if pooled_low is None
                        else min(pooled_low, cell_range[0])
                    )
                    pooled_high = (
                        cell_range[1]
                        if pooled_high is None
                        else max(pooled_high, cell_range[1])
                    )
            pooled_range = (
                None
                if pooled_low is None or pooled_high is None
                else (pooled_low, pooled_high)
            )
            image_limits = self._resolve_image_limits(
                "facet:image", pooled_range, state
            )
            cell_options = tuple({"color_limits": image_limits} for _cell in cells)

        cell_options = tuple(
            {**options, "paint_labels": False} for options in cell_options
        )
        outer_x, outer_y, _value_label = self._effective_labels(payload, state)
        visible_axes: list[tuple[int, Any]] = []
        # ONE call draws a cell, and it is the same call that draws the
        # standalone plot of that kind.  The hand-copied per-kind chain that
        # used to live here re-implemented the render half and omitted
        # the interaction half, so every facility a cell should inherit had to
        # REMEMBER to delegate -- and the ones that forgot (colour-limit
        # dragging, square cells, the point overlay, the crosshair value rail)
        # were user-visible bugs.
        for key, axis, index in self.painted_surfaces:
            cell = cells[index]
            handler.render(
                self,
                getattr(cell, "payload", cell),
                state,
                axes=axis,
                key=key,
                **cell_options[index],
            )
            if not focused:
                if axis.get_xlabel():
                    axis.set_xlabel("")
                if axis.get_ylabel():
                    axis.set_ylabel("")
            visible_axes.append((index, axis))
        typography = self.plan.facet_typography
        rows, columns = self.plan.facet_shape or (1, max(len(cells), 1))
        for index, axis in visible_axes:
            cell = cells[index]
            label = str(_facet_cell_title(cell, index))
            if typography is not None:
                # The plan knows each cell's exclusive title room; a title
                # wider than it shrinks (then truncates) rather than
                # overlapping its neighbour into one unreadable line.
                title_text, title_pt = fitted_facet_cell_title(
                    label, typography, self.style.fonts
                )
            else:
                title_text, title_pt = label, self.style.fonts.tick_pt
            if (
                axis.get_title() != title_text
                or axis.title.get_fontsize() != title_pt
            ):
                axis.set_title(
                    title_text,
                    fontsize=title_pt,
                    pad=self.style.render.compact_axes_title_pad_pt,
                )
            # The tick MARKS are the grid's; their label SIZE belongs to the
            # tick policy below, which may shrink it to keep two labels
            # apart and must be the last writer.
            if any(
                item.get_tick_params().get("length") != 2
                for item in (axis.xaxis, axis.yaxis)
            ):
                axis.tick_params(axis="both", length=2)
            row, column = divmod(index, columns)
            if focused:
                # The focused cell's ticks belong to the standalone-kind
                # policy applied below; installing the overview locators in
                # between would reset the tick artists twice per frame.
                continue
            # EVERY cell carries the SAME tick marks (one shared locator and
            # formatter on both axes); only tick LABELS are boundary-gated,
            # so the cells read against one coordinate frame while the grid
            # interior stays uncluttered.  Locator installs reset the axis'
            # tick artists, so each branch runs only when its configuration
            # actually changed (a reset tick is unpositioned until the next
            # full Axis draw) -- and the label flag is PART of the signature,
            # so a cell whose boundary classification changes as a live grid
            # grows re-fires its label gating.
            label_left = column == 0
            label_bottom = row == rows - 1 or index + columns >= len(cells)
            # A cell is a small panel, and it gets the same tick policy as a
            # large one -- which, being measured against the width it is
            # actually painted at, spends a cell's inch differently from a
            # panel's six.  A separate MaxNLocator(3) here meant the cells
            # were the one surface the shared policy never reached: no
            # compact offset, and three labels whether they fitted or not.
            # Only WHICH cells show their labels is the grid's business.
            apply_smart_ticks(
                axis,
                prune_edges=True,
                label_pt=(
                    typography.tick_pt
                    if typography is not None
                    else self.style.fonts.tick_pt
                ),
            )
            if axis.yaxis.get_tick_params().get("labelleft") != label_left:
                axis.tick_params(axis="y", labelleft=label_left)
            if axis.xaxis.get_tick_params().get("labelbottom") != label_bottom:
                axis.tick_params(axis="x", labelbottom=label_bottom)
            # The cells share one x span and one y span, so they share
            # whatever offset the tick policy took out of their labels: it is
            # written once, by the corner cell that carries both sets of tick
            # labels, into the FIGURE's corners -- the far end of the shared x
            # axis and the top of the shared y axis.  (The tick policy places
            # it; where a cell's own margin is, its neighbour is.)
            corner = label_bottom and label_left
            if axis.xaxis.get_offset_text().get_visible() != corner:
                axis.xaxis.get_offset_text().set_visible(corner)
            if axis.yaxis.get_offset_text().get_visible() != corner:
                axis.yaxis.get_offset_text().set_visible(corner)

        outer_labels = (("x", outer_x, 0.5, 0.012, 0.0), ("y", outer_y, 0.008, 0.5, 90.0))
        for name, value, x_pos, y_pos, rotation in outer_labels:
            artist_key = f"facet:outer_{name}"
            artist = self._artists.get(artist_key)
            if artist is None:
                artist = self._figure.text(
                    x_pos,
                    y_pos,
                    "",
                    ha="center" if name == "x" else "left",
                    va="bottom" if name == "x" else "center",
                    rotation=rotation,
                    fontsize=(
                        typography.outer_axis_label_pt
                        if typography is not None
                        else self.style.fonts.axis_label_pt
                    ),
                )
                self._artists[artist_key] = artist
            if artist.get_text() != value:
                artist.set_text(value)
            visible = self._facet_focus_index is None
            if artist.get_visible() != visible:
                artist.set_visible(visible)

        if self._facet_focus_index is None:
            return

        selected_index = self._facet_focus_index
        selected = axes[selected_index]
        if selected.get_xlabel() != outer_x:
            selected.set_xlabel(outer_x, fontsize=self.style.fonts.axis_label_pt)
        if selected.get_ylabel() != outer_y:
            selected.set_ylabel(outer_y, fontsize=self.style.fonts.axis_label_pt)
        focused_title = str(_facet_cell_title(cells[selected_index], selected_index))
        if (
            selected.get_title() != focused_title
            or selected.title.get_fontsize() != self.style.fonts.figure_title_pt
        ):
            selected.set_title(
                # The focused cell owns the whole panel: its title is the FULL
                # label again, not whatever fitted the grid cell's room.
                focused_title,
                fontsize=self.style.fonts.figure_title_pt,
                pad=self.style.render.compact_axes_title_pad_pt,
            )
        if isinstance(semantic, ImagePlot):
            # The chrome authority already applied the standalone image
            # kind's spatial tick budget; restating it keeps the signature
            # stable instead of re-installing default-budget locators.
            apply_smart_ticks(
                selected, "x", label_pt=self.style.fonts.tick_pt, prune_edges=True
            )
            apply_smart_ticks(selected, "y", label_pt=self.style.fonts.tick_pt)
        if (
            not selected.xaxis.get_tick_params().get("labelbottom", False)
            or not selected.yaxis.get_tick_params().get("labelleft", False)
        ):
            selected.tick_params(
                axis="both",
                labelbottom=True,
                labelleft=True,
            )

    def _update_image_point_overlay(
        self,
        axis: Any,
        payload: Any,
        overlay: ImagePointOverlay | None,
        state: DisplayState,
        key: str,
        facet_value: object | None,
    ) -> None:
        """Mutate ONE image surface's point layer, independently of its raster.

        Takes the surface it paints on, like every other per-plot painter, and
        gates on the SEMANTIC spec, so a FacetGrid of image cells carries the
        site overlay on each cell instead of dropping it at the outer spec.

        The overlay keeps the same repeat/point carrier as the image.  Its one
        resolution rule applies this PlotSpec's scopes and the current facet;
        a surface that still pools either leading axis has no single-shot
        status and paints UNKNOWN rather than an invented consensus.
        """

        if not isinstance(self.semantic_spec, ImagePlot):
            return
        x_quantity = getattr(payload, "x", None)
        y_quantity = getattr(payload, "y", None)
        if x_quantity is None or y_quantity is None:
            raise TypeError("Image point overlays require image coordinate quantities")
        signature = (
            None if overlay is None else overlay.revision,
            None if overlay is None else id(overlay),
            facet_value,
            bool(state["show_point_labels"]),
            str(getattr(x_quantity, "display_unit", "")),
            str(getattr(y_quantity, "display_unit", "")),
        )
        signature_key = f"{key}:points-signature"
        if signature == self._artists.get(signature_key):
            return
        self._artists[signature_key] = signature
        collection = self._artists.get(f"{key}:points")
        labels: list[Any] = self._artists.setdefault(f"{key}:point-labels", [])
        if overlay is None or overlay.count == 0:
            if collection is not None:
                collection.set_visible(False)
            for label in labels:
                label.set_visible(False)
            return

        from matplotlib.collections import EllipseCollection
        from matplotlib.colors import to_rgba

        canonical = np.asarray(overlay.coordinates, dtype=float)
        x_display = np.asarray(
            x_quantity.canonical_unit.convert_value_to(
                canonical[:, 0], x_quantity.display_unit
            ),
            dtype=float,
        )
        y_display = np.asarray(
            y_quantity.canonical_unit.convert_value_to(
                canonical[:, 1], y_quantity.display_unit
            ),
            dtype=float,
        )
        points = np.column_stack((x_display, y_display))
        x_domain = np.asarray(getattr(x_quantity, "canonical", ()), dtype=float)
        y_domain = np.asarray(getattr(y_quantity, "canonical", ()), dtype=float)
        spans = tuple(
            float(np.ptp(values[np.isfinite(values)]))
            for values in (x_domain, y_domain)
            if bool(np.any(np.isfinite(values)))
        )
        finite_spans = tuple(span for span in spans if span > 0.0)
        singleton_radius = (
            min(finite_spans) * self.style.artists.point_single_radius_fraction
            if finite_spans
            else self.style.artists.point_single_radius_fraction
        )
        canonical_radius = _point_ring_radius(
            canonical,
            fraction=self.style.artists.point_auto_radius_fraction,
            fallback=singleton_radius,
        )

        def display_radius(quantity: Any) -> float:
            converted = np.asarray(
                quantity.canonical_unit.convert_value_to(
                    (0.0, canonical_radius), quantity.display_unit
                ),
                dtype=float,
            )
            return abs(float(converted[1] - converted[0]))

        radius_x = display_radius(x_quantity)
        radius_y = display_radius(y_quantity)
        statuses = overlay.statuses_for(self.spec, facet_value) or (
            PointStatus.UNKNOWN,
        ) * overlay.count
        tokens = {
            PointStatus.UNKNOWN: self.style.artists.point_unknown,
            PointStatus.EMPTY: self.style.artists.point_empty,
            PointStatus.OCCUPIED: self.style.artists.point_occupied,
            PointStatus.INVALID: self.style.artists.point_invalid,
        }
        edgecolors = tuple(
            to_rgba(tokens[status].color, tokens[status].alpha)
            for status in statuses
        )
        linewidths = tuple(tokens[status].linewidth for status in statuses)
        # Matplotlib treats a tuple as one custom ``(offset, dash)``
        # specification; use a list for per-element styles instead.
        linestyles = [
            "--" if status is PointStatus.INVALID else "-" for status in statuses
        ]
        if collection is None:
            collection = EllipseCollection(
                widths=np.full(overlay.count, 2.0 * radius_x),
                heights=np.full(overlay.count, 2.0 * radius_y),
                angles=np.zeros(overlay.count),
                units="xy",
                offsets=points,
                transOffset=axis.transData,
                facecolors="none",
                edgecolors=edgecolors,
                linewidths=linewidths,
                linestyles=linestyles,
                zorder=self.style.artists.point_zorder,
                clip_on=True,
            )
            axis.add_collection(collection)
            self._artists[f"{key}:points"] = collection
        else:
            collection.set_offsets(points)
            collection.set_widths(np.full(overlay.count, 2.0 * radius_x))
            collection.set_heights(np.full(overlay.count, 2.0 * radius_y))
            collection.set_angles(np.zeros(overlay.count))
            collection.set_edgecolors(edgecolors)
            collection.set_linewidths(linewidths)
            collection.set_linestyles(linestyles)
            collection.set_visible(True)

        while len(labels) < overlay.count:
            labels.append(
                axis.text(
                    0.0,
                    0.0,
                    "",
                    ha="right",
                    va="bottom",
                    fontsize=self.style.fonts.fit_annotation_pt,
                    zorder=self.style.artists.point_label_zorder,
                    clip_on=True,
                )
            )
        show_labels = bool(state["show_point_labels"])
        point_ids = overlay.point_ids
        point_labels = overlay.labels
        label_x = points[:, 0] + (
            radius_x if axis.xaxis_inverted() else -radius_x
        )
        label_y = points[:, 1] + (
            -radius_y if axis.yaxis_inverted() else radius_y
        )
        for index, label in enumerate(labels):
            visible = show_labels and index < overlay.count
            label.set_visible(visible)
            if not visible:
                continue
            label.set_position((label_x[index], label_y[index]))
            label.set_color(edgecolors[index])
            explicit = None if point_labels is None else point_labels[index]
            point_id = None if point_ids is None else point_ids[index]
            label.set_text(explicit or point_id or "")

    def _update_pulse_timeline(
        self,
        axes: Any,
        payload: Any,
        state: DisplayState,
        key: str,
    ) -> None:
        update_pulse_timeline(
            axes,
            payload,
            state,
            self.style,
            self._artists,
        )
        x_label = _state_label(state, "x_label", None)
        if x_label is not None:
            if x_label:
                _factor, unit = pulse_time_scale(
                    payload,
                    state["x_display_unit"],
                )
                if not _EXPLICIT_UNIT_SUFFIX.search(x_label):
                    x_label = f"{x_label} ({unit})"
            axes.set_xlabel(x_label)
        y_label = _state_label(state, "y_label", None)
        if y_label is not None:
            axes.set_ylabel(y_label)
        signature = (
            payload.time_unit,
            state["x_display_unit"],
            bool(state["show_grid"]),
            tuple((channel.channel_id, channel.label) for channel in payload.channels),
            tuple((trace.name, trace.label) for trace in payload.analog_traces),
            tuple(axes.get_xlim()),
            tuple(axes.get_ylim()),
        )
        signature_key = f"{key}:chrome_signature"
        previous = self._artists.get(signature_key)
        self._artists[signature_key] = signature
        if previous != signature:
            self._mark_axes_chrome_dirty(axes)

    def _update_title_artist(self, state: DisplayState) -> None:
        title = _state_label(state, "title", self.spec.labels.title) or ""
        if isinstance(self.spec, FacetGridPlot):
            title_artist = self._artists.get("figure:title")
            if title_artist is None:
                title_artist = self._figure.text(
                    0.5,
                    self.style.render.figure_title_y,
                    "",
                    ha="center",
                    va="top",
                    fontsize=self.style.fonts.figure_title_pt,
                )
                self._artists["figure:title"] = title_artist
            title_artist.set_text(title)
            # The authored figure title stays up in BOTH presentations: the
            # focused cell shows its cell-value title alongside it, exactly
            # like a standalone plot shows its own title.
            title_artist.set_visible(bool(title))
        else:
            owner = self.primary_axes
            owner.set_title(
                title,
                fontsize=self.style.fonts.figure_title_pt,
                pad=self.style.render.axes_title_pad_pt,
                y=1.0,
            )

    def _apply_grid(self, state: DisplayState) -> None:
        show_grid = bool(state["show_grid"])
        data_axes = (
            self._axes.get("facet_cell", [])
            if isinstance(self.spec, FacetGridPlot)
            else [self.primary_axes]
        )
        for axis in data_axes:
            if isinstance(self.spec, PulseTimelinePlot):
                if show_grid:
                    axis.grid(
                        True,
                        axis="x",
                        color=self.style.palette.pulse_grid,
                        linewidth=self.style.pulse.grid_linewidth,
                    )
                else:
                    axis.grid(False, axis="x")
            else:
                axis.grid(show_grid)

    def _remove_artists(self, artists: Iterable[Any]) -> None:
        for artist in tuple(artists):
            self._line_sources.pop(id(artist), None)
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                artist.set_visible(False)

    def _selector_axis(self, state: SelectorState) -> Any:
        if isinstance(self.spec, FacetGridPlot):
            axes = self._axes.get("facet_cell", [])
            index = self._focused_facet_index if state.facet_index is None else state.facet_index
            if index is None or index < 0 or index >= self._visible_facet_count:
                raise IndexError("selector facet index is outside the current grid")
            return axes[index]
        return self.primary_axes

    def _image_selector_artist(self, state: SelectorState) -> Any | None:
        if not isinstance(self.semantic_spec, ImagePlot):
            return None
        axes = self._selector_axis(state)
        for key, candidate, _index in self.painted_surfaces:
            if candidate is axes:
                return self._artists.get(key)
        return None

    def _active_image_artist(self) -> Any | None:
        """The image artist of the surface a gesture acts on, or None.

        One spelling of "which cell is active" -- ``primary_surface`` -- so a
        colour-limit preview can never edit a different artist than the rail
        it was measured from.
        """

        if not isinstance(self.semantic_spec, ImagePlot):
            return None
        return self._artists.get(self.primary_surface[0])

    def resolved_color_limits(self) -> tuple[float, float]:
        """Return the clim painted by the active image artist."""

        image = self._active_image_artist()
        if image is None:
            raise TypeError("color limits require an active Image artist")
        return tuple(map(float, image.get_clim()))

    def _image_selector_payload(self, state: SelectorState) -> Any | None:
        if not isinstance(self.semantic_spec, ImagePlot):
            return None
        if self.semantic_spec is self.spec:
            return self._last_payload
        cells = tuple(getattr(self._last_payload, "cells", ()))
        index = self._focused_facet_index if state.facet_index is None else state.facet_index
        if index is not None and 0 <= index < len(cells):
            return getattr(cells[index], "payload", cells[index])
        return None

    def _image_cross_value(self, state: SelectorState) -> float | str | None:
        if not isinstance(state.value, CrosshairPoint):
            return None
        payload = self._image_selector_payload(state)
        if payload is None:
            return None
        x = np.asarray(_display_array(payload.x), dtype=float).reshape(-1)
        y = np.asarray(_display_array(payload.y), dtype=float).reshape(-1)
        z = np.asarray(_display_array(payload.z))
        if not x.size or not y.size or z.shape != (y.size, x.size):
            return None
        x_index = int(np.argmin(np.abs(x - state.value.x)))
        y_index = int(np.argmin(np.abs(y - state.value.y)))
        valid = np.asarray(
            getattr(payload, "valid", np.ones(z.shape, dtype=bool)), dtype=bool
        )
        valid = np.broadcast_to(valid, z.shape)
        value = z[y_index, x_index]
        if not bool(valid[y_index, x_index]) or not bool(np.isfinite(value)):
            return "NaN"
        return float(value)

    def _histogram_threshold_text(self, state: SelectorState) -> str:
        """Describe one effective threshold from the already-painted bins."""

        if state.kind is not SelectorKind.THRESHOLD:
            return ""
        if not isinstance(self.semantic_spec, HistogramPlot):
            return ""
        if self._classifier_labels:
            index = 0 if state.facet_index is None else state.facet_index
            if 0 <= index < len(self._classifier_labels):
                return self._classifier_labels[index]
        payload = self._last_payload
        if isinstance(self.spec, FacetGridPlot):
            cells = tuple(getattr(payload, "cells", ()))
            index = self._focused_facet_index if state.facet_index is None else state.facet_index
            if index is None or index < 0 or index >= len(cells):
                return ""
            payload = getattr(cells[index], "payload", cells[index])
        edges = getattr(getattr(payload, "edges", None), "display", None)
        counts = getattr(payload, "counts", None)
        if edges is None or counts is None:
            return ""
        edges = np.asarray(edges, dtype=float).reshape(-1)
        weights = np.asarray(counts, dtype=float).reshape(-1)
        if edges.size != weights.size + 1 or not weights.size:
            return ""
        total = float(np.sum(weights))
        threshold = float(state.value)
        if total <= 0.0:
            left_fraction = 0.0
        else:
            index = int(np.clip(
                np.searchsorted(edges, threshold, side="right") - 1,
                -1,
                len(weights),
            ))
            if index < 0:
                left_fraction = 0.0
            elif index >= len(weights):
                left_fraction = 1.0
            else:
                width = max(float(edges[index + 1] - edges[index]), 1.0e-300)
                fraction = float(np.clip(
                    (threshold - edges[index]) / width,
                    0.0,
                    1.0,
                ))
                left_fraction = float(
                    (np.sum(weights[:index]) + weights[index] * fraction) / total
                )
        return (
            f"th={_compact_engineering(threshold)}\n"
            f"L/R={100.0 * left_fraction:.1f}%/"
            f"{100.0 * (1.0 - left_fraction):.1f}%"
        )

    def _threshold_uses_x_axis(self) -> bool:
        return isinstance(self.semantic_spec, HistogramPlot)

    def preview_color_limits(self, low: float, high: float) -> None:
        """Preview image normalization without committing display state.

        The preview repaints exactly what the gesture is about: the image
        pixels and the artist clim every painted-limits consumer reads.
        The colorbar is deliberately untouched -- its gradient is a fixed
        proxy that clim changes never recolor, and its endpoint labels are
        axes chrome whose rewrite forces a full background recapture per
        drag step, which is what made large-image drags crawl.  The
        committed path reapplies colorbar state once on release through
        its own ``colorbar_state`` comparison.
        """

        selected = np.asarray((low, high), dtype=float)
        if not np.all(np.isfinite(selected)) or selected[0] >= selected[1]:
            raise ValueError("preview color limits must be finite and increasing")
        key = self.primary_surface[0]
        image = self._active_image_artist()
        if image is None:
            raise TypeError("color-limit preview requires an Image")
        if self._height_bars_scene:
            # The 2D fast path below re-gathers HEATMAP pixels, which for
            # the 3D scene would paint a stale heatmap over it (and with
            # no cached planes would change nothing until release).  The
            # scene's own render is the recolour: candidate limits move
            # both the colours and the bar heights they anchor.
            stashed = self._height_bars_calls.get(key)
            if stashed is None:
                # No stashed scene call yet: update the clim authority and
                # wait for the next scene render.  Falling through to the
                # 2D path would paint a HEATMAP front over the scene.
                image.set_clim(float(selected[0]), float(selected[1]))
                return
            axes, values, valid, extent, state, cmap_name, cmap, vid = (
                stashed
            )
            with style_context(self.style):
                self._update_height_bars_artist(
                    axes, values, valid, extent, state, key,
                    (float(selected[0]), float(selected[1])),
                    cmap_name, cmap, valid_identity=vid,
                )
                image.set_clim(float(selected[0]), float(selected[1]))
            return
        with style_context(self.style):
            limits = (float(selected[0]), float(selected[1]))
            prepared = self._artists.get(f"{key}:prepared_current")
            mapping = self._artists.get(f"{key}:mapping_state")
            cmap_cache = self._artists.get("image:cmap_cache")
            rgba = None
            if (
                self._artists.get(f"{key}:color_mode") == "rgba"
                and prepared is not None
                and mapping is not None
                and cmap_cache is not None
            ):
                rgba = self._image_rgba_front(
                    key, prepared, mapping[0], cmap_cache[1], limits
                )
            if rgba is not None:
                image.set_data(rgba)
                self._artists[f"{key}:applied_front"] = rgba
            # The artist clim stays authoritative for selector geometry and
            # snapshots in both modes.
            image.set_clim(*limits)

    @staticmethod
    def _selector_target_for_axis(axis: Any) -> SelectorTarget:
        role, separator, suffix = str(axis.get_gid() or "main").partition(":")
        return SelectorTarget(
            role or "main",
            int(suffix) if separator and suffix.isdigit() else None,
        )

    def _axis_for_selector_target(self, target: SelectorTarget) -> Any:
        for axis in self._figure.axes:
            if self._selector_target_for_axis(axis) == target:
                return axis
        raise KeyError(f"selector target is not present: {target!r}")

    def _resolved_color_limit_state(self) -> ColorLimitState | None:
        if not self._axes.get("distribution"):
            return None
        image = self._active_image_artist()
        if image is None:
            return None
        low, high = sorted(map(float, image.get_clim()))
        if not np.all(np.isfinite((low, high))) or low >= high:
            return None
        return ColorLimitState(NumericRange(low, high))

    def _make_selector_scene_owner(
        self,
        snapshot: SelectorSnapshot,
    ) -> SelectorSceneOwner:
        from matplotlib.colors import to_rgba

        if not isinstance(snapshot, SelectorSnapshot):
            raise TypeError("snapshot must be SelectorSnapshot")

        policy = self.style.render
        pulse_factor = 1.0
        if isinstance(self.spec, PulseTimelinePlot) and isinstance(
            self._last_payload, PulseTimelineData
        ):
            pulse_factor, _unit = pulse_time_scale(
                self._last_payload,
                self._last_state["x_display_unit"],
            )
        threshold = self.style.artists.threshold_line
        threshold_rgba = tuple(map(float, to_rgba(threshold.color, threshold.alpha)))
        threshold_label_rgba = tuple(map(float, to_rgba(self.style.palette.threshold)))
        contexts = []
        for state in snapshot.states:
            axis = self._selector_axis(state)
            image = self._image_selector_artist(state)
            cmap = None if image is None else image.get_cmap()
            selector_rgba = tuple(map(float, to_rgba(
                self.style.palette.line_single
                if cmap is None
                else cmap(policy.colormap_high_fraction)
            )))
            sample_value = self._image_cross_value(state) if image is not None else None
            sample_axis = (
                self._axes["distribution"][0]
                if isinstance(self.semantic_spec, ImagePlot)
                and state.kind is SelectorKind.CROSSHAIR
                and self._axes.get("distribution")
                else None
            )
            clim = None if image is None else tuple(map(float, image.get_clim()))
            contexts.append(SelectorItemContext(
                kind=state.kind,
                target=self._selector_target_for_axis(axis),
                label_target=self._selector_target_for_axis(axis),
                x_limits=tuple(map(float, axis.get_xlim())),
                y_limits=tuple(map(float, axis.get_ylim())),
                selector_rgba=selector_rgba,
                color_limit_rgba=(selector_rgba, selector_rgba),
                threshold_rgba=threshold_rgba,
                threshold_label_rgba=threshold_label_rgba,
                threshold_uses_x=self._threshold_uses_x_axis(),
                threshold_text=self._histogram_threshold_text(state),
                x_label_factor=pulse_factor,
                sample_value=sample_value,
                sample_span=(None if clim is None else abs(clim[1] - clim[0])),
                sample_target=(
                    None if sample_axis is None else self._selector_target_for_axis(sample_axis)
                ),
                sample_x_limits=(
                    None if sample_axis is None else tuple(map(float, sample_axis.get_xlim()))
                ),
                sample_rgba=(
                    None
                    if cmap is None or sample_axis is None
                    else tuple(map(float, to_rgba(
                        cmap(policy.selector_sample_colormap_fraction),
                        policy.selector_sample_alpha,
                    )))
                ),
            ))
        color_context = None
        distribution = self._axes.get("distribution", ())
        image = self._active_image_artist()
        if distribution and image is not None:
            color_axis = distribution[0]
            cmap = image.get_cmap()
            color_limit_rgba = tuple(
                tuple(map(float, to_rgba(color)))
                for color in (
                    cmap(policy.colormap_low_fraction),
                    cmap(policy.colormap_high_fraction),
                )
            )
            label_axis = self.primary_axes
            color_context = SelectorItemContext(
                kind=SelectorSceneKind.COLOR_LIMITS,
                target=self._selector_target_for_axis(color_axis),
                label_target=self._selector_target_for_axis(label_axis),
                x_limits=tuple(map(float, color_axis.get_xlim())),
                y_limits=tuple(map(float, color_axis.get_ylim())),
                selector_rgba=tuple(map(float, to_rgba(self.style.palette.line_single))),
                color_limit_rgba=color_limit_rgba,
                threshold_rgba=threshold_rgba,
                threshold_label_rgba=threshold_label_rgba,
                threshold_uses_x=False,
            )
        visual = self.style.artists
        return SelectorSceneOwner(
            tuple(contexts),
            SelectorSceneStyle(
                line_width_pt=visual.selector_line_width,
                line_alpha=visual.selector_alpha,
                handle_size_pt=visual.selector_handle_size_pt,
                handle_edge_width_pt=visual.selector_handle_edge_width,
                crosshair_size_pt=visual.crosshair_marker_size_pt,
                annotation_pt=self.style.fonts.annotation_pt,
                side_distribution_width_pt=visual.side_distribution_linewidth,
                selector_zorder=visual.selector_zorder,
                font_family=self.style.fonts.resolved_family,
                label_inset_fraction=policy.axes_text_inset_fraction,
                label_line_fraction=policy.selector_label_line_fraction,
                threshold_width_pt=threshold.linewidth,
                threshold_linestyle=threshold.linestyle,
            ),
            color_context,
        )

    @staticmethod
    def _selector_primitive_topology(primitive: SelectorPrimitive) -> object:
        return type(primitive), primitive.target, getattr(primitive, "shape", None)

    def _new_selector_artist(self, primitive: SelectorPrimitive) -> Any:
        axis = self._axis_for_selector_target(primitive.target)
        if isinstance(primitive, SelectorLine):
            (artist,) = axis.plot((), (), clip_on=True)
            return artist
        if isinstance(primitive, SelectorMarkers):
            return axis.scatter(
                (),
                (),
                marker={"square": "s", "circle": "o"}[primitive.shape],
                clip_on=True,
            )
        return axis.text(
            0.0,
            0.0,
            "",
            transform=axis.transAxes,
            va="top",
            clip_on=True,
        )

    @staticmethod
    def _mutate_selector_artist(artist: Any, primitive: SelectorPrimitive) -> None:
        if isinstance(primitive, SelectorLine):
            points = np.asarray(primitive.points, dtype=float)
            artist.set_data(points[:, 0], points[:, 1])
            artist.set_color(primitive.color)
            artist.set_linewidth(primitive.width_pt)
            artist.set_linestyle(primitive.linestyle)
        elif isinstance(primitive, SelectorMarkers):
            artist.set_offsets(np.asarray(primitive.points, dtype=float))
            artist.set_sizes(np.full(len(primitive.points), primitive.size_pt**2))
            artist.set_facecolors((primitive.facecolor,))
            artist.set_edgecolors(
                "none" if primitive.edgecolor is None else (primitive.edgecolor,)
            )
            artist.set_linewidths(primitive.edge_width_pt)
        else:
            artist.set_text(primitive.text)
            artist.set_position(primitive.position)
            artist.set_horizontalalignment(primitive.horizontal_alignment)
            artist.set_color(primitive.color)
            artist.set_fontfamily(primitive.font_family)
            artist.set_fontsize(primitive.font_size_pt)
        artist.set_clip_on(True)
        artist.set_visible(True)
        artist.set_zorder(primitive.zorder)

    def _update_selectors(self, snapshot: SelectorSnapshot) -> None:
        if not isinstance(snapshot, SelectorSnapshot):
            raise TypeError("selector update requires SelectorSnapshot")
        if self._selector_candidate is not None:
            snapshot = SelectorSnapshot(
                snapshot.committed,
                self._selector_candidate,
            )
        owner = self._make_selector_scene_owner(snapshot)
        scene = owner.build(
            snapshot,
            color_state=(
                self._color_limit_candidate
                if self._color_limit_candidate is not None
                else self._resolved_color_limit_state()
            ),
        )
        active_kinds = set(scene.selector_kinds)
        for kind in tuple(self._selector_artists):
            if kind in active_kinds:
                continue
            self._remove_artists(self._selector_artists.pop(kind))
            self._selector_topologies.pop(kind, None)
        for kind, primitives in scene.groups:
            topology = tuple(map(self._selector_primitive_topology, primitives))
            artists = self._selector_artists.get(kind)
            if artists is None or self._selector_topologies.get(kind) != topology:
                if artists is not None:
                    self._remove_artists(artists)
                artists = tuple(self._new_selector_artist(item) for item in primitives)
                self._selector_artists[kind] = artists
                self._selector_topologies[kind] = topology
            for artist, primitive in zip(artists, primitives, strict=True):
                self._mutate_selector_artist(artist, primitive)

    def _fit_annotation_text(self, overlay: FitOverlay) -> str:
        lines: list[str] = []
        if overlay.formula:
            lines.append(overlay.formula)
        for parameter in overlay.parameter_display:
            lines.append(self._fit_parameter_line(parameter))
        return "\n".join(lines)

    @staticmethod
    def _fit_parameter_line(parameter: Any) -> str:
        uncertainty = (
            "n/a"
            if parameter.standard_error is None
            else _compact_engineering(parameter.standard_error)
        )
        unit = f" {parameter.unit}" if parameter.unit else ""
        return (
            f"{parameter.label} = {_fit_parameter_value_text(parameter)} "
            f"± {uncertainty}{unit}"
        )

    def _fit_headline_annotation_text(self, overlay: FitOverlay) -> str:
        parameter = overlay.headline_parameter
        return "" if parameter is None else self._fit_parameter_line(parameter)

    def _restore_fit_source_lines(self) -> None:
        for line, was_visible in self._fit_hidden_source_lines:
            if getattr(line, "axes", None) is not None:
                line.set_visible(was_visible)
        self._fit_hidden_source_lines = ()
        if self._fit_source_scatter is not None:
            self._fit_source_scatter.set_offsets(np.empty((0, 2), dtype=float))
            self._fit_source_scatter.set_visible(False)

    def _show_fit_source_scatter(
        self,
        axis: Any,
        source_lines: Sequence[Any],
    ) -> None:
        hidden = tuple(
            (line, was_visible)
            for line, was_visible in self._fit_hidden_source_lines
            if getattr(line, "axes", None) is axis
            and any(line is candidate for candidate in source_lines)
        )
        visible = tuple(
            line
            for line in source_lines
            if getattr(line, "axes", None) is axis and line.get_visible()
        )
        if visible:
            active_lines = visible
            self._fit_hidden_source_lines = tuple((line, True) for line in visible)
        else:
            active_lines = tuple(line for line, was_visible in hidden if was_visible)
        # Count before materializing: past the cap the scatter is refused
        # anyway, and a large source (a million-point trace) should not pay
        # for column stacks it will never draw.
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        point_count = 0
        cap = self.style.render.fit_source_scatter_max_points
        for line in active_lines:
            registered = self._line_sources.get(id(line))
            if registered is not None and registered[0] is line:
                _line, _axis, raw_x, raw_y, _isolated_glyphs = registered
                x = np.asarray(raw_x, dtype=float).reshape(-1)
                y = np.asarray(raw_y, dtype=float).reshape(-1)
            else:
                x = np.asarray(line.get_xdata(), dtype=float).reshape(-1)
                y = np.asarray(line.get_ydata(), dtype=float).reshape(-1)
            if x.shape != y.shape:
                continue
            pairs.append((x, y))
            point_count += int(np.count_nonzero(np.isfinite(x) & np.isfinite(y)))
            if point_count >= cap:
                self._restore_fit_source_lines()
                return
        point_groups: list[np.ndarray] = []
        for x, y in pairs:
            finite = np.isfinite(x) & np.isfinite(y)
            if bool(np.any(finite)):
                point_groups.append(np.column_stack((x[finite], y[finite])))
        if not point_groups:
            self._restore_fit_source_lines()
            return
        scatter = self._fit_source_scatter
        if scatter is not None and getattr(scatter, "axes", None) is not axis:
            self._remove_artists((scatter,))
            scatter = None
        if scatter is None:
            scatter = axis.scatter(
                (),
                (),
                s=self.style.artists.fit_source_scatter_area_pt2,
                color=self.style.palette.data_scatter,
                edgecolors="none",
                clip_on=True,
                zorder=self.style.artists.fit_source_scatter_zorder,
            )
            self._fit_source_scatter = scatter
        scatter.set_offsets(np.concatenate(point_groups, axis=0))
        scatter.set_visible(True)
        for line, _was_visible in self._fit_hidden_source_lines:
            line.set_visible(False)

    def _set_fit_mode(self, active: bool) -> None:
        """Keep source presentation stable while fit-result topology changes."""

        if not active:
            self._restore_fit_source_lines()
            return
        if not isinstance(self.semantic_spec, (CurvePlot, RollingPlot)):
            self._restore_fit_source_lines()
            return
        if isinstance(self.spec, FacetGridPlot):
            index = self._focused_facet_index
            axes = self._axes.get("facet_cell", ())
            if index is None or index < 0 or index >= len(axes):
                self._restore_fit_source_lines()
                return
            axis = axes[index]
        else:
            axis = self.primary_axes
        source_lines: list[Any] = []
        for key, value in self._artists.items():
            if key not in {"curve", "rolling:history"} and not key.startswith("facet:"):
                continue
            if isinstance(value, list):
                source_lines.extend(
                    item
                    for item in value
                    if getattr(item, "axes", None) is axis
                    and hasattr(item, "get_xdata")
                )
        self._show_fit_source_scatter(axis, source_lines[:1])

    def _focused_fit_overlays(
        self,
        overlays: tuple[FitOverlay, ...],
    ) -> tuple[FitOverlay, ...]:
        """Select the one overlay painted by a focused surface."""

        if not isinstance(self.spec, FacetGridPlot):
            return overlays[:1]
        selected = self._focused_facet_index
        if selected is None:
            return ()
        matching = tuple(
            overlay
            for overlay in overlays
            if overlay.facet_index in {None, selected}
        )
        return matching[:1]

    def _fit_target(self, overlay: FitOverlay) -> tuple[Any, PlotSpec]:
        if isinstance(self.spec, FacetGridPlot):
            index = (
                self._focused_facet_index
                if overlay.facet_index is None
                else overlay.facet_index
            )
            if index is None or index < 0 or index >= self._visible_facet_count:
                raise IndexError("fit facet index is outside the current grid")
            return self._axes["facet_cell"][index], self.semantic_spec
        return self.primary_axes, self.semantic_spec

    @staticmethod
    def _fit_family_for(semantic: PlotSpec, overlay: FitOverlay) -> str:
        if not overlay.success:
            return "failure"
        if overlay.ellipse_glyph is not None:
            return "ellipse"
        if any(polyline.role == "component" for polyline in overlay.polylines):
            return f"components:{len(overlay.polylines)}"
        if overlay.polylines:
            return "histogram" if isinstance(semantic, HistogramPlot) else "curve"
        return "annotation"

    def _clear_fit_topology(self) -> None:
        self._remove_artists(self._fit_artists)
        self._fit_artists.clear()
        self._fit_slots.clear()
        self._facet_fit_topologies.clear()
        self._fit_axis = None
        self._fit_family = None
        self._fit_model_id = None

    def _fit_polyline_token(self, semantic: PlotSpec, polyline: FitPolyline) -> Any:
        if polyline.role == "component":
            component_tokens = self.style.artists.bimodal_fit_lines[:2]
            return component_tokens[polyline.component_index % len(component_tokens)]
        if polyline.role == "total" or isinstance(semantic, HistogramPlot):
            return self.style.artists.bimodal_fit_lines[2]
        return self.style.artists.curve_fit_line

    def _build_fit_topology(
        self,
        axis: Any,
        family: str,
        semantic: PlotSpec,
        overlay: FitOverlay,
        *,
        annotation: _FitAnnotationDetail = _FitAnnotationDetail.FULL,
    ) -> None:
        from matplotlib.patches import Ellipse

        self._fit_axis = axis
        self._fit_family = family
        if family == "failure":
            if annotation is _FitAnnotationDetail.NONE:
                return
            diagnostic = axis.text(
                self.style.render.axes_text_inset_fraction,
                1.0 - self.style.render.axes_text_inset_fraction,
                "",
                transform=axis.transAxes,
                va="top",
                color=self.style.artists.fit_failure_color,
                fontsize=(
                    self.style.fonts.facet_fit_annotation_pt
                    if annotation is _FitAnnotationDetail.HEADLINE
                    else self.style.fonts.fit_annotation_pt
                ),
                clip_on=True,
                zorder=self.style.artists.fit_annotation_zorder,
            )
            self._fit_slots["diagnostic"] = diagnostic
            self._fit_artists.append(diagnostic)
            return
        if family == "ellipse":
            ring_token = self.style.artists.point_occupied
            center = axis.scatter(
                (),
                (),
                color=self.style.artists.fit_ellipse_color,
                s=self.style.artists.fit_ellipse_center_area_pt2,
                clip_on=True,
                zorder=self.style.artists.fit_ellipse_zorder,
            )
            ring = Ellipse(
                (0.0, 0.0),
                width=0.0,
                height=0.0,
                edgecolor=ring_token.color,
                facecolor="none",
                linewidth=ring_token.linewidth,
                alpha=ring_token.alpha,
                clip_on=True,
                zorder=self.style.artists.fit_ellipse_zorder,
            )
            center.set_visible(False)
            ring.set_visible(False)
            axis.add_patch(ring)
            self._fit_slots.update(center=center, ring=ring)
            self._fit_artists.extend((center, ring))
        elif family.startswith("components:"):
            lines = []
            for polyline in overlay.polylines:
                token = self._fit_polyline_token(semantic, polyline)
                (line,) = axis.plot((), (), **token.kwargs())
                lines.append(line)
            self._fit_slots["lines"] = tuple(lines)
            self._fit_artists.extend(lines)
        elif family in {"histogram", "curve"}:
            token = self._fit_polyline_token(semantic, overlay.polylines[0])
            (line,) = axis.plot((), (), **token.kwargs())
            self._fit_slots["line"] = line
            self._fit_artists.append(line)
        if annotation is _FitAnnotationDetail.NONE:
            return
        annotation = axis.text(
            self.style.render.axes_text_inset_fraction,
            1.0 - self.style.render.axes_text_inset_fraction,
            "",
            transform=axis.transAxes,
            ha="left",
            va="top",
            clip_on=True,
            zorder=self.style.artists.fit_annotation_zorder,
            fontsize=(
                self.style.fonts.facet_fit_annotation_pt
                if annotation is _FitAnnotationDetail.HEADLINE
                else self.style.fonts.fit_annotation_pt
            ),
            color=self.style.palette.fit_text,
        )
        annotation.set_visible(False)
        self._fit_slots["annotation"] = annotation
        self._fit_artists.append(annotation)

    def _update_fit_annotation(
        self,
        overlay: FitOverlay,
    ) -> None:
        annotation = self._fit_slots["annotation"]
        content = self._fit_annotation_text(overlay)
        annotation.set_text(content)
        annotation.set_visible(bool(content))

    def _update_fit_headline_annotation(self, overlay: FitOverlay) -> None:
        annotation = self._fit_slots["annotation"]
        content = self._fit_headline_annotation_text(overlay)
        annotation.set_text(content)
        annotation.set_visible(bool(content))

    def _set_fit_line(self, line: Any, polyline: FitPolyline) -> None:
        order = np.argsort(polyline.x)
        self._apply_line_data(line.axes, line, polyline.x[order], polyline.y[order])
        line.set_visible(True)

    def _update_fit_primitives(self, family: str, overlay: FitOverlay) -> None:
        if family == "ellipse":
            center = self._fit_slots["center"]
            ring = self._fit_slots["ring"]
            glyph = overlay.ellipse_glyph
            if glyph is None:
                raise RuntimeError("ellipse fit family requires one ellipse glyph")
            center.set_offsets(
                np.asarray(((glyph.center_x, glyph.center_y),), dtype=float)
            )
            ring.set_center((glyph.center_x, glyph.center_y))
            ring.set_width(2.0 * glyph.radius_x)
            ring.set_height(2.0 * glyph.radius_y)
            center.set_visible(True)
            ring.set_visible(True)
            return
        if family.startswith("components:"):
            for line, polyline in zip(
                self._fit_slots["lines"],
                overlay.polylines,
                strict=True,
            ):
                self._set_fit_line(line, polyline)
            return
        if family in {"histogram", "curve"}:
            self._set_fit_line(self._fit_slots["line"], overlay.polylines[0])

    def _update_single_fit(
        self,
        overlay: FitOverlay | None,
        model_id: str | None,
    ) -> None:
        if overlay is None:
            self._clear_fit_topology()
            return
        axis, semantic = self._fit_target(overlay)
        family = self._fit_family_for(semantic, overlay)
        if (
            self._fit_axis is not axis
            or self._fit_family != family
            or self._fit_model_id != model_id
        ):
            self._clear_fit_topology()
            self._build_fit_topology(axis, family, semantic, overlay)
            self._fit_model_id = model_id
        if not overlay.success:
            diagnostic_text = overlay.diagnostic.strip() or "fit failed"
            diagnostic_text = _truncate_fit_diagnostic(
                diagnostic_text,
                _FIT_DIAGNOSTIC_SINGLE_MAX_CHARS,
            )
            diagnostic = self._fit_slots["diagnostic"]
            diagnostic.set_text(f"fit: {diagnostic_text}")
            diagnostic.set_visible(True)
            return

        self._update_fit_primitives(family, overlay)
        self._update_fit_annotation(overlay)

    def _update_facet_fit_overview(
        self,
        overlays: tuple[FitOverlay, ...],
        model_id: str | None,
    ) -> None:
        """Paint all cell fit curves without per-cell parameter annotations."""

        if not isinstance(self.spec, FacetGridPlot):
            raise TypeError("facet fit overview requires FacetGridPlot")
        if self._fit_axis is not None:
            self._clear_fit_topology()
        axes = self._axes.get("facet_cell", ())
        semantic = self.semantic_spec
        active_indices = {
            overlay.facet_index
            for overlay in overlays
            if overlay.facet_index is not None
            and 0 <= overlay.facet_index < self._visible_facet_count
        }
        for index in tuple(self._facet_fit_topologies):
            if index in active_indices:
                continue
            _axis, _family, _model, _slots, artists = (
                self._facet_fit_topologies.pop(index)
            )
            self._remove_artists(artists)
            removed_ids = {id(removed) for removed in artists}
            self._fit_artists[:] = [
                artist
                for artist in self._fit_artists
                if id(artist) not in removed_ids
            ]
        for overlay in overlays:
            index = overlay.facet_index
            if index is None or index < 0 or index >= self._visible_facet_count:
                continue
            family = self._fit_family_for(semantic, overlay)
            axis = axes[index]
            topology = self._facet_fit_topologies.get(index)
            if (
                topology is None
                or topology[0] is not axis
                or topology[1] != family
                or topology[2] != model_id
            ):
                if topology is not None:
                    old_artists = topology[4]
                    self._remove_artists(old_artists)
                    removed_ids = {id(removed) for removed in old_artists}
                    self._fit_artists[:] = [
                        artist
                        for artist in self._fit_artists
                        if id(artist) not in removed_ids
                    ]
                self._fit_slots = {}
                first = len(self._fit_artists)
                self._build_fit_topology(
                    axis,
                    family,
                    semantic,
                    overlay,
                    annotation=_FitAnnotationDetail.HEADLINE,
                )
                topology = (
                    axis,
                    family,
                    model_id,
                    dict(self._fit_slots),
                    tuple(self._fit_artists[first:]),
                )
                self._facet_fit_topologies[index] = topology
            self._fit_slots = topology[3]
            if family == "failure":
                diagnostic = self._fit_slots["diagnostic"]
                diagnostic.set_text(
                    _truncate_fit_diagnostic(
                        overlay.diagnostic,
                        _FIT_DIAGNOSTIC_FACET_MAX_CHARS,
                    )
                )
                diagnostic.set_visible(True)
            else:
                self._update_fit_primitives(family, overlay)
                self._update_fit_headline_annotation(overlay)
        self._fit_slots = {}
        self._fit_axis = None
        self._fit_family = None
        self._fit_model_id = None

    def _update_fit(
        self,
        overlays: tuple[FitOverlay, ...],
        *,
        overview: bool,
        model_id: str | None,
    ) -> None:
        if overview:
            self._update_facet_fit_overview(overlays, model_id)
            return
        self._update_single_fit(overlays[0] if overlays else None, model_id)

    def _annotation_size_that_fits(self, axis: Any, content: str) -> float:
        """The size this annotation must shrink to in order to stay inside.

        The classifier writes three numbers into the corner of the surface it
        annotates.  In a panel they fit; in a facet cell they are fifteen
        characters against an inch, and at a fixed size they were drawn over
        the distribution they describe and then clipped by the cell's own
        edge -- "hreshold 323.2" beside a histogram it hid.  Shortening the
        text loses numbers an operator asked for; the room is what has to be
        respected, so the text is measured against it, exactly as a cell
        title is.
        """

        from .layout import _text_width_pt

        size = (
            self.style.fonts.facet_fit_annotation_pt
            if isinstance(self.spec, FacetGridPlot)
            else self.style.fonts.annotation_pt
        )
        if not content:
            return size
        figure = getattr(getattr(axis, "figure", None), "dpi", None)
        if figure is None or float(figure) <= 0.0:
            return size
        inset = 2.0 * self.style.render.axes_text_inset_fraction
        room = float(axis.bbox.width) / (float(figure) / 72.0) * (1.0 - inset)
        families = self.style.fonts.sans_serif
        widest = max(
            _text_width_pt(line, families, size) for line in content.splitlines()
        )
        if widest <= room or widest <= 0.0:
            return size
        return max(size * room / widest, self.style.fonts.facet_title_min_pt)

    def _facet_mark_scale(self) -> float:
        """How much of its full weight a mark keeps inside a facet cell."""

        typography = self.plan.facet_typography
        if not isinstance(self.spec, FacetGridPlot) or typography is None:
            return 1.0
        return float(typography.scale)

    def _update_classifier(
        self,
        overlays: tuple[FitOverlay, ...],
        thresholds: tuple[float | None, ...],
        labels: tuple[str, ...],
    ) -> None:
        """Paint Distribution classifier curves separately from ordinary fits."""

        active: dict[int, tuple[FitOverlay, float, str]] = {}
        if isinstance(self.semantic_spec, HistogramPlot):
            for fallback, (overlay, threshold, label) in enumerate(
                zip(overlays, thresholds, labels, strict=True)
            ):
                index = fallback if overlay.facet_index is None else overlay.facet_index
                if (
                    overlay.success
                    and threshold is not None
                    and len(overlay.polylines) == 3
                    and (
                        not isinstance(self.spec, FacetGridPlot)
                        or index < self._visible_facet_count
                    )
                ):
                    active[index] = (overlay, float(threshold), label)
        for index in tuple(self._classifier_artists):
            if index in active:
                continue
            lines, threshold_line, label = self._classifier_artists.pop(index)
            self._remove_artists((*lines, threshold_line, label))

        axes = self._axes.get("facet_cell", ())
        for index, (overlay, threshold, content) in active.items():
            axis = axes[index] if isinstance(self.spec, FacetGridPlot) else self.primary_axes
            artists = self._classifier_artists.get(index)
            if artists is None:
                lines = tuple(
                    axis.plot((), (), clip_on=True, **token.kwargs())[0]
                    for token in self.style.artists.bimodal_fit_lines
                )
                threshold_line = axis.plot(
                    (),
                    (),
                    clip_on=True,
                    **self.style.artists.classifier_threshold_line.kwargs(),
                )[0]
                label = axis.text(
                    1.0 - self.style.render.axes_text_inset_fraction,
                    1.0 - self.style.render.axes_text_inset_fraction,
                    "",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    clip_on=True,
                    zorder=self.style.artists.fit_annotation_zorder,
                    fontsize=(
                        self.style.fonts.facet_fit_annotation_pt
                        if isinstance(self.spec, FacetGridPlot)
                        else self.style.fonts.annotation_pt
                    ),
                    color=self.style.palette.threshold,
                )
                artists = (lines, threshold_line, label)
                self._classifier_artists[index] = artists
            lines, threshold_line, label = artists
            for line, polyline in zip(lines, overlay.polylines, strict=True):
                self._set_fit_line(line, polyline)
            y_low, y_high = axis.get_ylim()
            threshold_line.set_data((threshold, threshold), (y_low, y_high))
            # A cell's marks are drawn at the cell's scale, like its text.
            # The threshold's own weight is a GRAB target -- wide enough to
            # put a mouse on -- and nobody drags a cell of a 35-cell report:
            # at full weight it covered a fifth of the distribution it cuts.
            threshold_line.set_linewidth(
                self.style.artists.classifier_threshold_line.linewidth
                * self._facet_mark_scale()
            )
            interactive = (
                not isinstance(self.spec, FacetGridPlot)
                or self._facet_focus_index == index
            )
            threshold_line.set_visible(not interactive)
            label.set_text(content)
            label.set_fontsize(self._annotation_size_that_fits(axis, content))
            label.set_visible(not interactive and bool(content))

    def _rgba_buffer(self) -> np.ndarray:
        """Return the already-painted canvas buffer in the surface contract."""

        source = np.asarray(self._figure.canvas.buffer_rgba(), dtype=np.uint8)
        target_width, target_height = self.plan.raster_size
        actual_height, actual_width = source.shape[:2]
        if (actual_width, actual_height) == (target_width, target_height):
            return readonly_copy(source)

        # Fractional DPR can put Agg's floor allocation one trailing pixel away
        # from the rounded frontend contract.  Preserve artist transforms and
        # adjust only the right and bottom handoff edges.
        pixels = np.empty((target_height, target_width, 4), dtype=np.uint8)
        background = np.rint(
            np.clip(np.asarray(self._figure.get_facecolor()), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        pixels[...] = background
        copy_width = min(target_width, actual_width)
        copy_height = min(target_height, actual_height)
        pixels[:copy_height, :copy_width] = source[:copy_height, :copy_width]
        pixels.setflags(write=False)
        return pixels

    def capture_rgba(
        self,
        *,
        redraw: bool = False,
    ) -> np.ndarray:
        """Capture the already composed front without changing artist state."""

        if redraw:
            self.draw()
        return self._rgba_buffer()

    def capture_rgba_bytes(self, *, redraw: bool = False) -> tuple[bytes, int, int]:
        """The composed front as raw bytes, with the size they are in.

        For a caller that wants bytes -- a raster front does, and immediately
        drops the array it was handed.  Going through an ndarray copied the
        whole frame once to make it read-only and again to serialise it: at the
        2x2 preset with DPR 2 that is 9 MB copied twice per published front,
        about 6 ms where 3 will do, on the worker that has to keep up with a
        live camera.
        """

        if redraw:
            self.draw()
        source = np.asarray(self._figure.canvas.buffer_rgba(), dtype=np.uint8)
        target_width, target_height = self.plan.raster_size
        actual_height, actual_width = source.shape[:2]
        if (actual_width, actual_height) == (target_width, target_height):
            return source.tobytes(order="C"), target_height, target_width
        # Fractional DPR takes the padded path, which has to build an array
        # anyway; one copy there is unavoidable and the shape is already right.
        padded = self._rgba_buffer()
        return padded.tobytes(order="C"), target_height, target_width

    def rgba(self) -> np.ndarray:
        """Return an immutable RGBA snapshot of the current scene.

        When the Agg buffer already holds the latest composed frame the
        snapshot is a copy of it -- pixel-identical to a full draw by the
        compose contract.  Anything that mutates artist state without
        composing resets the freshness mark and lands here as a full
        draw, so a stale frame can never be published.
        """

        if self._composed_generation != self._raster_generation or (
            self._composed_generation < 0
        ):
            self.draw()
        return self._rgba_buffer()

    def save(self, path: str | Path | BytesIO, *, dpi: float | None = None, **kwargs: Any) -> None:
        locked, hover = self._series_locked, self._series_hover
        try:
            self._series_locked = self._series_hover = None
            self._apply_series_focus()
            with style_context(self.style):
                self._figure.savefig(path, dpi=dpi or self.plan.dpi, **kwargs)
        finally:
            self._series_locked, self._series_hover = locked, hover
            self._apply_series_focus()
            self.draw()


__all__ = ["MatplotlibRenderer", "RenderFrame"]
