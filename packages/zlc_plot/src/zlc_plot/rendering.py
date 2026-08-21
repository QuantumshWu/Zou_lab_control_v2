"""Persistent Matplotlib artists for every public ZLC plot kind.

This module owns no GUI toolkit and never calls ``pyplot``.  A surface can be
attached to Agg or a Qt5 canvas.  Data/display edits mutate persistent
artists; fixed-size changes rebuild layout within the same Figure.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import math
from numbers import Real
from pathlib import Path
import re
from enum import Enum
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


def _display_array(value: Any) -> np.ndarray:
    raw = getattr(value, "display", value)
    return np.asarray(raw)


def _valid_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    raw = getattr(value, "valid", None)
    if raw is None:
        return np.ones(shape, dtype=bool)
    return np.asarray(raw, dtype=bool)


def _unit_symbol(value: Any) -> str:
    unit = getattr(value, "display_unit", None)
    return "" if unit is None else str(getattr(unit, "symbol", unit))


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
    if zero_based and low >= 0.0:
        if (
            current_low == 0.0
            and current_high > 0.0
            and 0.7 * current_high <= high <= current_high
        ):
            return current_low, current_high
        return target
    span = current_high - current_low
    if span <= 0.0:
        return target
    clips = low < current_low or high > current_high
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
        self._raster_generation = 0
        self._focused_facet_index: int | None = None
        self._facet_focus_index: int | None = None
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
            # Every painted surface honours the requested view, not just the
            # selected one: a FacetGrid overview shows N cells of the same
            # picture, and zooming one of them alone is not a view of anything.
            for _key, axes, _index in painted:
                self._apply_requested_view(axes, frame.view_limits)
            self._classifier_labels = frame.classifier_labels
            self._update_classifier(
                frame.classifier_overlays,
                frame.classifier_thresholds,
                frame.classifier_labels,
            )
            self._last_selectors = painted_selectors
            self._update_selectors(self._last_selectors)
            self._set_fit_mode(bool(painted_fit_overlays) and not overview)
            self._update_fit(
                painted_fit_overlays,
                overview=overview,
                model_id=frame.fit_model_id,
            )
            cells = tuple(getattr(payload, "cells", ()))
            for key, axes, index in painted:
                cell = None if index is None else cells[index]
                self._update_image_point_overlay(
                    axes,
                    payload if cell is None else getattr(cell, "payload", cell),
                    frame.image_overlay,
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

    def _set_xlim(self, axis: Any, low: float, high: float) -> None:
        previous = np.asarray(axis.get_xlim(), dtype=float)
        wanted = np.asarray((low, high), dtype=float)
        if not np.allclose(previous, wanted, rtol=1e-12, atol=1e-15):
            axis.set_xlim(float(low), float(high))
            self._mark_axes_chrome_dirty(axis)

    def _set_ylim(self, axis: Any, low: float, high: float) -> None:
        previous = np.asarray(axis.get_ylim(), dtype=float)
        wanted = np.asarray((low, high), dtype=float)
        if not np.allclose(previous, wanted, rtol=1e-12, atol=1e-15):
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
        for axes in {entry[1].axes for entry in tuple(collected)}:
            if not axes.get_visible():
                continue
            if id(axes) in dynamic_full_axes_ids:
                continue
            for axis in (axes.xaxis, axes.yaxis):
                if id(axis) in dynamic_axis_ids:
                    continue
                axis_z = float(axis.get_zorder())
                # The same ticks a full Axis.draw would paint: positions
                # refreshed and clipped to the current view interval.  The
                # raw ``majorTicks`` list keeps stale instances parked at
                # out-of-view locations after a limit change, and painting
                # those leaks mark segments outside the axes box.
                for tick in axis._update_ticks():
                    keyed(tick.gridline, axes, axis_z)
                    keyed(tick.tick1line, axes, axis_z)
                    keyed(tick.tick2line, axes, axis_z)
            for spine in axes.spines.values():
                keyed(spine, axes, spine.get_zorder())
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
        dynamics = self._dynamic_artists()
        reusable = (
            chrome_stable
            and self._background_region is not None
            and self._background_signature == signature
            and not self._chrome_dirty_axes
        )
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
        if artist.get_interpolation() not in ("nearest", "antialiased", "auto"):
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
            prepared.append(_PreparedSeries(x, y, valid, str(label)))
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
    ) -> None:
        lines = self._ensure_lines(axes, len(series), key)
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for index, item in enumerate(series):
            # NaNs preserve invalid runs as gaps instead of joining neighbours.
            plotted_x = np.where(item.valid, item.x, np.nan)
            plotted_y = np.where(item.valid, item.y, np.nan)
            lines[index].set_data(plotted_x, plotted_y)
            if lines[index].get_label() != item.label:
                lines[index].set_label(item.label)
            all_x.append(item.x[item.valid])
            all_y.append(item.y[item.valid])
        if limits is not None:
            self._set_xlim(axes, *limits[0])
            self._set_ylim(axes, *limits[1])
        else:
            usable_x = tuple(value for value in all_x if value.size)
            usable_y = tuple(value for value in all_y if value.size)
            xlim = (
                _curve_x_limits(np.concatenate(usable_x))
                if usable_x
                else None
            )
            y_range = (
                _data_limits(np.concatenate(usable_y))
                if usable_y
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
        apply_smart_ticks(axes, label_pt=self.style.fonts.tick_pt)

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
        interpolation = str(state["interpolation"])
        mapping_state = (cmap_name, interpolation, color_limits)
        mapping_key = f"{key}:mapping_state"
        previous_mapping = self._artists.get(mapping_key)

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
        # roughly one physical output pixel at the current DPR.
        axes.apply_aspect()
        display_pixel_shape = (
            max(1, round(float(axes.bbox.width))),
            max(1, round(float(axes.bbox.height))),
        )
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
        # draw.  Nearest resampling commutes with colormapping exactly; the
        # default antialiased/auto kernels act on the prepared front's ≤1.25x
        # residual only, where filtering composed colors is indistinguishable
        # from filtering scalars for the closed colormap set.  Explicitly
        # smooth kernels, masked fronts and unresolved limits keep the scalar
        # path.
        self._artists[f"{key}:prepared_current"] = prepared
        rgba_front = (
            self._image_rgba_front(key, prepared, cmap_name, cmap, color_limits)
            if color_limits is not None
            and interpolation in ("nearest", "antialiased", "auto")
            and not isinstance(prepared.values, np.ma.MaskedArray)
            else None
        )
        self._artists[f"{key}:color_mode"] = (
            "scalar" if rgba_front is None else "rgba"
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
                interpolation=interpolation,
                interpolation_stage=prepared.interpolation_stage,
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
                image.set_extent(prepared.extent)
                self._artists[applied_key] = shown
            if previous_mapping is None or previous_mapping[1] != interpolation:
                image.set_interpolation(interpolation)
            if image.get_interpolation_stage() != prepared.interpolation_stage:
                image.set_interpolation_stage(prepared.interpolation_stage)
            # The artist's cmap/clim stay authoritative in both modes: RGBA
            # rendering ignores them, but selector handles, rail guides and
            # pointer snapshots all read the painted limits off the artist.
            if previous_mapping is None or previous_mapping[0] != cmap_name:
                image.set_cmap(cmap)
            if (
                color_limits is not None
                and (
                    previous_mapping is None
                    or previous_mapping[2] != color_limits
                    or not np.allclose(
                        np.asarray(image.get_clim(), dtype=float),
                        np.asarray(color_limits, dtype=float),
                        rtol=1.0e-12,
                        atol=1.0e-15,
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
            sliced.append(_PreparedSeries(x_values, y_values, valid, str(label)))

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
                bounds = self.plan.axes[index].box.matplotlib_bounds()
                # ``set_position`` invalidates the axes transform stack even
                # for an identical box; skip the per-frame no-op.
                if tuple(axis.get_position().bounds) != tuple(bounds):
                    axis.set_position(bounds)
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
        # used to live here re-implemented the render half and never migrated
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
        point_groups: list[np.ndarray] = []
        point_count = 0
        for line in active_lines:
            x = np.asarray(line.get_xdata(), dtype=float).reshape(-1)
            y = np.asarray(line.get_ydata(), dtype=float).reshape(-1)
            if x.shape != y.shape:
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            if bool(np.any(finite)):
                points = np.column_stack((x[finite], y[finite]))
                point_groups.append(points)
                point_count += len(points)
        if (
            not point_groups
            or point_count >= self.style.render.fit_source_scatter_max_points
        ):
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
                edgecolor=self.style.artists.fit_ellipse_color,
                facecolor="none",
                linewidth=self.style.artists.fit_ellipse_ring_linewidth,
                alpha=self.style.artists.fit_ellipse_ring_alpha,
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

    @staticmethod
    def _set_fit_line(line: Any, polyline: FitPolyline) -> None:
        order = np.argsort(polyline.x)
        line.set_data(polyline.x[order], polyline.y[order])
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
        """Draw the current scene and return an immutable RGBA snapshot."""

        self.draw()
        return self._rgba_buffer()

    def save(self, path: str | Path | BytesIO, *, dpi: float | None = None, **kwargs: Any) -> None:
        with style_context(self.style):
            self._figure.savefig(path, dpi=dpi or self.plan.dpi, **kwargs)
        self.draw()


__all__ = ["MatplotlibRenderer", "RenderFrame"]
