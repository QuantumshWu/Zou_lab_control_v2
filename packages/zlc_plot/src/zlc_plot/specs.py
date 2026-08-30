"""Typed semantic plot specifications and their editable display parameters.

The specification describes *what* data coordinates a plot uses.  It is
immutable for the lifetime of a session.  Frequently edited presentation
values live in :class:`~zlc_plot.state.DisplayStateStore`, so a GUI slider can
change (for example) ``bin_count`` without constructing another Figure.
"""

from __future__ import annotations

import math

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import ClassVar, TypeAlias

from zlc_data import (
    CoordinateScalar,
    CoordinateSelector,
    LATEST_COORDINATE,
    canonical_coordinate_scalar,
)

from ._validation import finite_real, integer, text
from .kinds import AxisRef, PlotKind
from .parameters import ParameterSchema, ParameterSpec, RenderEffect
from .style import PlotStyleConfig


_TEXT_EFFECTS = RenderEffect.TEXT
_DISPLAY_UNIT_EFFECTS = (
    RenderEffect.VIEW_PROJECTION
    | RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.TEXT
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.INTERACTION_REPROJECT
)
_FACET_UNIT_EFFECTS = (
    RenderEffect.VIEW_PROJECTION
    | RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.TEXT
)
_PULSE_TIME_UNIT_EFFECTS = (
    RenderEffect.AXIS_TRANSFORM
    | RenderEffect.TEXT
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.INTERACTION_REPROJECT
)
_AXIS_LIMIT_EFFECTS = (
    RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.INTERACTION_REPROJECT
)
#: The histogram's VALUE axis, unlike every other limit pair, is not
#: resolved at render time: x_relim_mode/x_min/x_max are read only inside
#: _histogram_bins, which runs when the payload is built.  Without
#: PAYLOAD_PROJECTION the bins the operator asked for arrived at the next
#: data revision -- and never at all once acquisition stopped.
_HISTOGRAM_VALUE_AXIS_EFFECTS = (
    _AXIS_LIMIT_EFFECTS | RenderEffect.PAYLOAD_PROJECTION
)
_HISTOGRAM_REPRESENTATION_EFFECTS = (
    RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.FIT_SELECTION
    | RenderEffect.INTERACTION_REPROJECT
)
_HISTOGRAM_PROJECTION_EFFECTS = (
    RenderEffect.PAYLOAD_PROJECTION | _HISTOGRAM_REPRESENTATION_EFFECTS
)
_IMAGE_COLOR_EFFECTS = (
    RenderEffect.BASE_STYLE
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.INTERACTION_REPROJECT
)
_ROLLING_DISTRIBUTION_BIN_EFFECTS = (
    RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
)
#: Flipping the band re-projects the payload (sem appears), moves the y
#: range (the band must fit), and re-selects any fit (sigma weighting).
_UNCERTAINTY_EFFECTS = (
    RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.FIT_SELECTION
)

_ROLLING_WINDOW_EFFECTS = (
    RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.FIT_SELECTION
    | RenderEffect.INTERACTION_REPROJECT
)

FACET_FIT_PARAMETER = "facet_fit_parameter"


class Reduction(str, Enum):
    """How repeated samples that share a plotted coordinate are combined."""

    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    FIRST = "first"


class RelimMode(str, Enum):
    TIGHT = "tight"
    NORMAL = "normal"
    FIXED = "fixed"


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be text")
    return value


_normalize_nonempty_text = partial(text, field="value")
_normalize_integer = partial(integer, field="value")


def _finite_float(value: object) -> float:
    return finite_real(value, "value")


def _finite_or_none(value: object) -> float | None:
    return None if value is None else _finite_float(value)


def _relim_mode(value: object) -> str:
    if isinstance(value, RelimMode):
        return value.value
    return _text(value).strip().lower()


def _unit_text_or_none(value: object) -> str | None:
    if value is None:
        return None
    result = _text(value).strip()
    return result or None


def _label_text_or_none(value: object) -> str | None:
    if value is None:
        return None
    return _text(value)


@dataclass(frozen=True, slots=True)
class PlotLabels:
    title: str | None = None
    x: str | None = None
    y: str | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        for name in ("title", "x", "y", "value"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"label {name!r} must be non-empty text or None")


#: One axis pinned to one of its coordinates.  A scope is a tuple of these:
#: the panel plots the data where every named axis holds the named value, and
#: everything downstream -- fits, selectors, the distribution -- sees only
#: that.  It is a fate an axis can be given, exactly like being x or being
#: grouped by, which is why it lives on the specification and not among the
#: display parameters: it changes WHAT is plotted, not how it looks.
ScopeTerm: TypeAlias = tuple[AxisRef, CoordinateScalar | CoordinateSelector]


def _require_distinct_axes(
    roles: tuple[tuple[str, AxisRef | None], ...],
    scope: tuple[ScopeTerm, ...],
) -> None:
    used: dict[tuple[object, str | None], str] = {}
    for name, axis in roles:
        if axis is None:
            continue
        identity = axis.physical_identity
        previous = used.get(identity)
        if previous is not None:
            raise ValueError(
                f"one physical axis cannot be both {previous} and {name}"
            )
        used[identity] = name
    for axis, _coordinate in scope:
        identity = axis.physical_identity
        previous = used.get(identity)
        if previous is not None:
            raise ValueError(
                f"one physical axis cannot be both {previous} and scope"
            )
        used[identity] = "scope"


def _validated_scope(value: object) -> tuple[ScopeTerm, ...]:
    if not isinstance(value, tuple):
        raise TypeError("scope must be a tuple of (AxisRef, value) pairs")
    terms: list[ScopeTerm] = []
    seen: set[tuple[object, str | None]] = set()
    for term in value:
        if not isinstance(term, tuple) or len(term) != 2:
            raise TypeError("each scope term must be an (AxisRef, value) pair")
        axis, coordinate = term
        if not isinstance(axis, AxisRef):
            raise TypeError("scope term axis must be AxisRef")
        coordinate = (
            LATEST_COORDINATE
            if coordinate is LATEST_COORDINATE
            else canonical_coordinate_scalar(coordinate, "scope coordinate")
        )
        identity = axis.physical_identity
        if identity in seen:
            raise ValueError(f"axis {axis!r} is scoped twice")
        seen.add(identity)
        terms.append((axis, coordinate))
    return tuple(terms)


@dataclass(frozen=True, slots=True)
class CurvePlot:
    x: AxisRef
    group: AxisRef | None = None
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[ScopeTerm, ...] = ()
    kind: ClassVar[PlotKind] = PlotKind.CURVE

    def __post_init__(self) -> None:
        if not isinstance(self.x, AxisRef):
            raise TypeError("CurvePlot.x must be AxisRef")
        if self.group is not None and not isinstance(self.group, AxisRef):
            raise TypeError("CurvePlot.group must be AxisRef or None")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("CurvePlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("CurvePlot.labels must be PlotLabels")
        scope = _validated_scope(self.scope)
        _require_distinct_axes((("x", self.x), ("group", self.group)), scope)
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True, slots=True)
class ImagePlot:
    x: AxisRef
    y: AxisRef
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[ScopeTerm, ...] = ()
    kind: ClassVar[PlotKind] = PlotKind.IMAGE

    def __post_init__(self) -> None:
        if not isinstance(self.x, AxisRef) or not isinstance(self.y, AxisRef):
            raise TypeError("ImagePlot x and y must be AxisRef")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("ImagePlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("ImagePlot.labels must be PlotLabels")
        scope = _validated_scope(self.scope)
        _require_distinct_axes((("x", self.x), ("y", self.y)), scope)
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True, slots=True)
class HistogramPlot:
    """Distribution of the acquired values.

    Pooling is the DEFAULT fate, not the only one: an axis may instead be
    collapsed under the reduction before the values are binned, which is
    the difference between "the distribution of every shot" and "the
    distribution of each site's mean over shots" -- two different
    measurements of the same data.  There is no group fate: a histogram
    draws one distribution, and several belong in a facet grid.
    """

    reduction: Reduction = Reduction.MEAN
    #: Axes collapsed under the reduction before binning; every other axis
    #: pools into the one distribution.
    reduced: tuple[AxisRef, ...] = ()
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[ScopeTerm, ...] = ()
    kind: ClassVar[PlotKind] = PlotKind.HISTOGRAM

    def __post_init__(self) -> None:
        if not isinstance(self.reduction, Reduction):
            raise TypeError("HistogramPlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("HistogramPlot.labels must be PlotLabels")
        reduced = tuple(self.reduced)
        if any(not isinstance(ref, AxisRef) for ref in reduced):
            raise TypeError("HistogramPlot.reduced must contain AxisRef values")
        identities = [ref.physical_identity for ref in reduced]
        if len(set(identities)) != len(identities):
            raise ValueError("HistogramPlot.reduced must name distinct axes")
        scope = _validated_scope(self.scope)
        pinned = {term[0].physical_identity for term in scope}
        if pinned & set(identities):
            raise ValueError("an axis cannot be both reduced and pinned")
        object.__setattr__(self, "reduced", reduced)
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True, slots=True)
class RollingPlot:
    group: AxisRef | None = None
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[ScopeTerm, ...] = ()
    kind: ClassVar[PlotKind] = PlotKind.ROLLING

    def __post_init__(self) -> None:
        if self.group is not None and not isinstance(self.group, AxisRef):
            raise TypeError("RollingPlot.group must be AxisRef or None")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("RollingPlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("RollingPlot.labels must be PlotLabels")
        scope = _validated_scope(self.scope)
        _require_distinct_axes((("group", self.group),), scope)
        object.__setattr__(self, "scope", scope)


CellPlot: TypeAlias = CurvePlot | ImagePlot | HistogramPlot


@dataclass(frozen=True, slots=True)
class FacetGridPlot:
    """One cell per value of a single facet axis.

    The facet axis determines the cell count; how those cells pack into
    rows and columns is a layout decision the fixed layout optimizes, never
    an authored semantic.
    """

    facet: AxisRef
    cell: CellPlot
    labels: PlotLabels = field(default_factory=PlotLabels)
    scope: tuple[ScopeTerm, ...] = ()
    kind: ClassVar[PlotKind] = PlotKind.FACET_GRID

    def __post_init__(self) -> None:
        if not isinstance(self.facet, AxisRef):
            raise TypeError("FacetGridPlot.facet must be AxisRef")
        if not isinstance(self.cell, (CurvePlot, ImagePlot, HistogramPlot)):
            raise TypeError("FacetGrid cells must be CurvePlot, ImagePlot or HistogramPlot")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("FacetGridPlot.labels must be PlotLabels")
        if self.cell.scope:
            raise ValueError(
                "FacetGrid cell.scope is invalid; scope belongs to FacetGridPlot"
            )
        scope = _validated_scope(self.scope)
        _require_distinct_axes(
            (
                ("facet", self.facet),
                ("x", getattr(self.cell, "x", None)),
                ("y", getattr(self.cell, "y", None)),
                ("group", getattr(self.cell, "group", None)),
            ),
            scope,
        )
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True, slots=True)
class PulseTimelinePlot:
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.PULSE_TIMELINE

    def __post_init__(self) -> None:
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("PulseTimelinePlot.labels must be PlotLabels")


PlotSpec: TypeAlias = (
    CurvePlot
    | ImagePlot
    | HistogramPlot
    | RollingPlot
    | FacetGridPlot
    | PulseTimelinePlot
)


def semantic_spec(spec: PlotSpec) -> PlotSpec:
    """Return the spec that decides what a drawn surface IS.

    A FacetGrid is a layout: every cell draws the cell spec, so every
    question about the plotted thing -- its kind, its axes, its labels, the
    gestures its surface accepts -- is a question about the cell.  This is
    the ONE place that unwraps it.

    Re-typing it inline is how a per-plot facility ends up honouring the
    cell in one half of the renderer and the grid in the other: colour-limit
    dragging, cell squareness, the point overlay and the crosshair value
    rail were each split that way, and each was a user-visible bug.
    ``test_semantic_spec_has_one_authority`` holds the line mechanically.
    """

    return spec.cell if isinstance(spec, FacetGridPlot) else spec


def _title_parameter(default: str | None) -> ParameterSpec[object]:
    del default
    return ParameterSpec(
        "title",
        str,
        _TEXT_EFFECTS,
        default=None,
        normalizer=_label_text_or_none,
        allow_none=True,
        label="Title",
    )


def _label_parameter(name: str, default: str | None, label: str) -> ParameterSpec[object]:
    del default
    return ParameterSpec(
        name,
        str,
        _TEXT_EFFECTS,
        default=None,
        normalizer=_label_text_or_none,
        allow_none=True,
        label=label,
    )


def _show_grid_parameter(*, default: bool = False) -> ParameterSpec[object]:
    return ParameterSpec(
        "show_grid",
        bool,
        RenderEffect.CHROME,
        default=default,
        label="Grid",
    )


def _unit_parameter(
    name: str,
    *,
    effects: RenderEffect = _DISPLAY_UNIT_EFFECTS,
) -> ParameterSpec[object]:
    return ParameterSpec(
        name,
        str,
        effects,
        default=None,
        normalizer=_unit_text_or_none,
        allow_none=True,
        label=name.removesuffix("_display_unit").replace("_", " ").title() + " unit",
    )


def _pulse_time_unit(value: object) -> str | None:
    unit = _unit_text_or_none(value)
    if unit is None:
        return None
    normalized = unit.lower().replace("μ", "µ")
    if normalized == "µs":
        normalized = "us"
    return normalized


_PULSE_X_UNIT_PARAMETER = ParameterSpec(
    "x_display_unit",
    str,
    _PULSE_TIME_UNIT_EFFECTS,
    default=None,
    normalizer=_pulse_time_unit,
    allow_none=True,
    label="Time unit",
    choices=("ns", "us", "ms", "s"),
)


def _curve_parameters() -> tuple[ParameterSpec[object], ...]:
    return (
        ParameterSpec(
            "relim_mode",
            str,
            _AXIS_LIMIT_EFFECTS,
            default=RelimMode.NORMAL.value,
            normalizer=_relim_mode,
            label="Limits",
            choices=tuple(item.value for item in RelimMode),
        ),
        ParameterSpec(
            "y_min",
            (int, float),
            _AXIS_LIMIT_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Y minimum",
        ),
        ParameterSpec(
            "y_max",
            (int, float),
            _AXIS_LIMIT_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Y maximum",
        ),
    )


def _bin_count_parameter(
    effects: RenderEffect = _HISTOGRAM_PROJECTION_EFFECTS,
) -> ParameterSpec[object]:
    return ParameterSpec(
        "bin_count",
        int,
        effects,
        default=60,
        normalizer=_normalize_integer,
        label="Bins",
        minimum=1,
        maximum=4096,
        step=1,
    )


def _window_parameter(default: int) -> ParameterSpec[object]:
    """How many shots of history this panel looks back over.

    Only the two kinds that HAVE a history offer it, and each consumes it its
    own fixed way: a rolling trace plots the window along x, a distribution
    pools it.  Which way is not the operator's to choose -- a distribution
    with a shot axis is not a distribution -- so the parameter carries only
    the size.
    """

    return ParameterSpec(
        "window",
        int,
        _ROLLING_WINDOW_EFFECTS,
        default=default,
        normalizer=_normalize_integer,
        label="Window",
        minimum=1,
        maximum=1_000_000,
        step=1,
    )


def _histogram_parameters() -> tuple[ParameterSpec[object], ...]:
    return (
        _bin_count_parameter(),
        ParameterSpec(
            "threshold_classifier",
            bool,
            RenderEffect.OVERLAY | RenderEffect.FIT_SELECTION,
            default=False,
            label="Threshold classifier",
        ),
        ParameterSpec(
            "density",
            bool,
            _HISTOGRAM_REPRESENTATION_EFFECTS | RenderEffect.TEXT,
            default=False,
            label="Density",
        ),
        ParameterSpec(
            "cumulative",
            bool,
            _HISTOGRAM_REPRESENTATION_EFFECTS,
            default=False,
            label="Cumulative",
        ),
        ParameterSpec(
            "relim_mode",
            str,
            _AXIS_LIMIT_EFFECTS,
            default=RelimMode.NORMAL.value,
            normalizer=_relim_mode,
            label="Count limits",
            choices=tuple(item.value for item in RelimMode),
        ),
        ParameterSpec(
            "y_min",
            (int, float),
            _AXIS_LIMIT_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Count minimum",
        ),
        ParameterSpec(
            "y_max",
            (int, float),
            _AXIS_LIMIT_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Count maximum",
        ),
        ParameterSpec(
            "log_y",
            bool,
            _AXIS_LIMIT_EFFECTS,
            default=False,
            label="Log count axis",
        ),
        # The VALUE axis is autoscaled too, and until it had a mode it was
        # the one axis in the product with no answer to "may this keep the
        # limits it already shows".  It binned between the raw minimum and
        # maximum every revision, so on a live camera its bin edges moved on
        # 90 of 94 frames: the bars were re-cut under the operator, and the
        # moving limits marked the chrome dirty, which took the panel out of
        # the composed-background path into a full redraw on 60 per cent of
        # frames.  Normal by default, for the same reason every other steady
        # view is: a distribution is read across revisions.
        ParameterSpec(
            "x_relim_mode",
            str,
            _HISTOGRAM_VALUE_AXIS_EFFECTS,
            default=RelimMode.NORMAL.value,
            normalizer=_relim_mode,
            label="Value limits",
            choices=tuple(item.value for item in RelimMode),
        ),
        ParameterSpec(
            "x_min",
            (int, float),
            _HISTOGRAM_VALUE_AXIS_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Value minimum",
        ),
        ParameterSpec(
            "x_max",
            (int, float),
            _HISTOGRAM_VALUE_AXIS_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Value maximum",
        ),
    )


from ._height3d_raster import HeightBarCamera as _HeightBarCamera

_HOME_CAMERA = _HeightBarCamera()


class ImagePresentation(str, Enum):
    """How the Image kind paints its one value grid."""

    HEATMAP = "heatmap"
    HEIGHT_BARS = "height_bars"


def _image_presentation(value: object) -> str:
    return ImagePresentation(str(value)).value


def _finite_number(value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError("camera parameters must be finite")
    return number


def _image_parameters(style: PlotStyleConfig) -> tuple[ParameterSpec[object], ...]:
    policy = style.render
    entries: list[ParameterSpec[object]] = [
        ParameterSpec(
            "relim_mode",
            str,
            _IMAGE_COLOR_EFFECTS,
            default=RelimMode.TIGHT.value,
            normalizer=_relim_mode,
            label="Color limits",
            choices=tuple(item.value for item in RelimMode),
        ),
        ParameterSpec(
            "colormap",
            str,
            _IMAGE_COLOR_EFFECTS,
            default=policy.image_default_colormap,
            normalizer=_normalize_nonempty_text,
            label="Colormap",
            choices=policy.image_colormaps,
        ),
        ParameterSpec(
            "color_min",
            (int, float),
            _IMAGE_COLOR_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Color minimum",
        ),
        ParameterSpec(
            "color_max",
            (int, float),
            _IMAGE_COLOR_EFFECTS,
            default=None,
            normalizer=_finite_or_none,
            allow_none=True,
            label="Color maximum",
        ),
    ]
    entries.append(
        ParameterSpec(
            "show_colorbar",
            bool,
            RenderEffect.CHROME,
            default=True,
            label="Colorbar",
        )
    )
    entries.extend(
        (
            ParameterSpec(
                "presentation",
                str,
                # LAYOUT because the two presentations are laid out
                # differently: a heatmap's chrome has reserved margins,
                # a 3D scene's labels move with the camera and share one
                # padded region with the scene itself.
                _IMAGE_COLOR_EFFECTS
                | RenderEffect.BASE_GEOMETRY
                | RenderEffect.LAYOUT,
                default=ImagePresentation.HEATMAP.value,
                normalizer=_image_presentation,
                label="Presentation",
                choices=tuple(item.value for item in ImagePresentation),
            ),
            ParameterSpec(
                "camera_azimuth",
                (int, float),
                RenderEffect.BASE_GEOMETRY,
                default=_HOME_CAMERA.azimuth_deg,
                normalizer=_finite_number,
                label="View azimuth",
            ),
            ParameterSpec(
                "camera_elevation",
                (int, float),
                RenderEffect.BASE_GEOMETRY,
                default=_HOME_CAMERA.elevation_deg,
                normalizer=_finite_number,
                label="View elevation",
            ),
            ParameterSpec(
                "camera_zoom",
                (int, float),
                RenderEffect.BASE_GEOMETRY,
                default=_HOME_CAMERA.zoom,
                normalizer=_finite_number,
                label="View zoom",
            ),
        )
    )
    return tuple(entries)


_LIMIT_PARAMETER_PAIRS = (
    ("relim_mode", "color_min", "color_max"),
    ("relim_mode", "y_min", "y_max"),
    ("x_relim_mode", "x_min", "x_max"),
)

#: Every relim mode there is, and the one place that knows it.  The editor
#: greys an authored limit out unless ITS axis is fixed, and it used to keep
#: a second copy of this list to do it.
LIMIT_MODE_NAMES = tuple(dict.fromkeys(mode for mode, _low, _high
                                       in _LIMIT_PARAMETER_PAIRS))


def limit_pairs() -> tuple[tuple[str, str, str], ...]:
    """Every authored limit pair, with the mode that governs it.

    The pairing was rediscovered by string surgery in three separate walks
    ("_min" -> name[:-4] + "_max"), each keyed on ``relim_mode`` alone.  That
    was survivable while there was one mode and two pairs; it stopped being
    survivable the moment the histogram's value axis got a mode of its own,
    because the walks would have handed the x pair the y axis's limits.
    """

    return _LIMIT_PARAMETER_PAIRS


def limit_pair_for(name: str) -> tuple[str, str, str] | None:
    """The (mode, low, high) triple this authored limit belongs to."""

    for mode, low, high in _LIMIT_PARAMETER_PAIRS:
        if name in (low, high):
            return mode, low, high
    return None


def _normalize_limit_transition(
    current: Mapping[str, object],
    updates: Mapping[str, object],
) -> Mapping[str, object]:
    normalized = dict(updates)
    # Per pair, and per the mode that governs THAT pair: leaving fixed on the
    # count axis must not throw away an authored value-axis limit.
    for mode_name, low_name, high_name in _LIMIT_PARAMETER_PAIRS:
        if mode_name not in updates:
            continue
        if updates[mode_name] == RelimMode.FIXED.value:
            continue
        if low_name in current and high_name in current:
            normalized.setdefault(low_name, None)
            normalized.setdefault(high_name, None)
    return normalized


def _validate_authored_conflicts(values: Mapping[str, object]) -> None:
    """The display states no host could EVER accept.

    A stored appearance may be INCOMPLETE -- fixed limits materialize on
    the next configure -- but never contradictory: an inverted pair or a
    non-positive log floor fails every future host start, so these are the
    rules a WRITE must already answer to, not only a running session.
    """

    mode = values.get("relim_mode")
    for _mode_name, low_name, high_name in _LIMIT_PARAMETER_PAIRS:
        if low_name not in values or high_name not in values:
            continue
        low = values[low_name]
        high = values[high_name]
        if low is not None and high is not None and float(low) >= float(high):
            raise ValueError(f"{low_name} must be smaller than {high_name}")
    if (
        values.get("log_y") is True
        and mode == RelimMode.FIXED.value
        and values.get("y_min") is not None
        and float(values["y_min"]) <= 0.0
    ):
        raise ValueError("log count limits require a positive y_min")
    if (
        values.get("threshold_classifier") is True
        and values.get("cumulative") is True
    ):
        raise ValueError("threshold classifier requires a non-cumulative histogram")


def _validate_limit_state(values: Mapping[str, object]) -> None:
    _validate_authored_conflicts(values)
    for mode_name, low_name, high_name in _LIMIT_PARAMETER_PAIRS:
        if low_name not in values or high_name not in values:
            continue
        if values.get(mode_name) == RelimMode.FIXED.value and (
            values[low_name] is None or values[high_name] is None
        ):
            raise ValueError(
                f"fixed {mode_name} requires {low_name} and {high_name}"
            )


def validate_authored_display(
    kind: "PlotKind | str",
    values: Mapping[str, object],
    *,
    style: "PlotStyleConfig",
    facet_cell_kind: "PlotKind | str | None" = None,
) -> None:
    """Refuse an authored display state no host could ever accept.

    Panel state stores appearance; a host applies it at START.  Storing a
    contradictory state therefore wedges every surface whose start applies
    it -- including the editor whose form is the one tool that could repair
    it.  Incomplete states pass (fixed limits materialize on the next
    configure); contradictory ones raise with the schema's own sentence.
    A facet grid whose cell kind is not authored yet has no display
    contract, so there is nothing to refuse against.
    """

    resolved = PlotKind(kind)
    if resolved is PlotKind.FACET_GRID and facet_cell_kind is None:
        return
    schema = parameter_schema_for_kind(
        resolved, style=style, facet_cell_kind=facet_cell_kind
    )
    state = dict(schema.initial_values(None))
    state.update(schema.prepare_updates(schema.declared_subset(values)))
    _validate_authored_conflicts(state)


def accepts_classifier_thresholds(
    spec: PlotSpec,
    display: Mapping[str, object],
) -> bool:
    """Whether this exact accepted plot vocabulary owns classifier levels."""

    if not isinstance(display, Mapping):
        raise TypeError("display must be a mapping")
    return bool(
        isinstance(semantic_spec(spec), HistogramPlot)
        and display.get("threshold_classifier") is True
    )


def paints_image_surface(spec: PlotSpec) -> bool:
    """Whether the resolved plot (including a FacetGrid cell) paints images."""

    return semantic_spec(spec).kind is PlotKind.IMAGE


def history_window_requirement(
    spec: PlotSpec,
    display: Mapping[str, object],
) -> int | None:
    """Bounded source-index history this accepted view actually requests.

    A fate on the materialized primary-index axis is an ordinary Dataset
    projection and never owns retention.  Resource demand comes only from a
    vocabulary that exposes a bounded history window.
    """

    if not isinstance(display, Mapping):
        raise TypeError("display must be a mapping")
    semantic = semantic_spec(spec)
    if not isinstance(semantic, (RollingPlot, HistogramPlot)):
        return None
    window = display.get("window")
    if type(window) is not int or window <= 0:
        return None
    if isinstance(semantic, HistogramPlot) and window == 1:
        return None
    if isinstance(semantic, RollingPlot):
        # A trailing mean reads shots the window may not show, so demand is
        # whichever reaches further back.  Sizing retention by the visible
        # window alone would leave the earliest points of a long trailing
        # mean averaging a history that had already been released.
        trailing = display.get("trailing")
        if type(trailing) is int and trailing > window:
            return trailing
    return window


@dataclass(frozen=True, slots=True)
class _ParameterSchemaContext:
    """The semantic facts that change a display-parameter schema."""

    kind: PlotKind
    semantic_kind: PlotKind
    labels: PlotLabels


def _parameter_schema_context(spec: PlotSpec) -> _ParameterSchemaContext:
    if not isinstance(
        spec,
        (
            CurvePlot,
            ImagePlot,
            HistogramPlot,
            RollingPlot,
            FacetGridPlot,
            PulseTimelinePlot,
        ),
    ):
        raise TypeError("unsupported plot specification")
    return _ParameterSchemaContext(
        spec.kind, semantic_spec(spec).kind, spec.labels
    )


def _parameter_schema_for_context(
    context: _ParameterSchemaContext,
    *,
    style: PlotStyleConfig,
) -> ParameterSchema:
    """Build the one schema used by bound sessions and unbound authoring UI."""

    if not isinstance(style, PlotStyleConfig):
        raise TypeError("style must be PlotStyleConfig")
    kind = context.kind
    semantic_kind = context.semantic_kind
    labels = context.labels
    entries: list[ParameterSpec[object]] = [
        _title_parameter(labels.title),
        _label_parameter("x_label", labels.x, "X label"),
        _label_parameter("y_label", labels.y, "Y label"),
        _label_parameter("value_label", labels.value, "Value label"),
        _show_grid_parameter(default=kind is PlotKind.PULSE_TIMELINE),
    ]
    if kind is PlotKind.PULSE_TIMELINE:
        entries.append(_PULSE_X_UNIT_PARAMETER)
    else:
        if semantic_kind in {PlotKind.CURVE, PlotKind.IMAGE}:
            entries.append(_unit_parameter("x_display_unit"))
        if semantic_kind is PlotKind.IMAGE:
            entries.append(_unit_parameter("y_display_unit"))
        if kind is PlotKind.FACET_GRID:
            entries.append(
                _unit_parameter(
                    "facet_display_unit",
                    effects=_FACET_UNIT_EFFECTS,
                )
            )
            entries.append(
                ParameterSpec(
                    FACET_FIT_PARAMETER,
                    str,
                    RenderEffect.OVERLAY,
                    default="model headline",
                    normalizer=_normalize_nonempty_text,
                    label="Cell fit value",
                )
            )
        entries.append(_unit_parameter("value_display_unit"))
    if semantic_kind in {PlotKind.CURVE, PlotKind.ROLLING}:
        entries.extend(_curve_parameters())
    if semantic_kind in {PlotKind.CURVE, PlotKind.ROLLING}:
        # A display choice, not a data declaration: the operator flips the
        # band on a live panel and the projection computes the MEAN's
        # standard error on demand.  On a curve that is the spread of what
        # each plotted point pooled; on a rolling trace it is each shot's
        # own pooled spread (or the trailing error when averaging).  On a
        # non-MEAN reduction the statistic does not exist and the switch
        # is inert.
        #
        # ON by default.  A mean drawn without its spread is a number
        # presented as if it were exact, and the operator had to know the
        # switch existed to find out otherwise.  Where the statistic does
        # not exist the switch is inert, so defaulting it on costs nothing
        # and never draws a band that is not there.
        entries.append(
            ParameterSpec(
                "uncertainty",
                bool,
                _UNCERTAINTY_EFFECTS,
                default=True,
                label="Uncertainty band",
            )
        )
    if semantic_kind is PlotKind.HISTOGRAM:
        entries.extend(_histogram_parameters())
        # One shot by default: a distribution of what was just measured.  A
        # larger window pools that many of the most recent shots into the same
        # picture, which is how a per-site histogram gets enough counts to
        # separate two peaks.
        entries.append(_window_parameter(1))
    if semantic_kind is PlotKind.IMAGE:
        # A FacetGrid whose cell is an image carries the FULL image surface:
        # the focused cell is the standalone Image kind, so its parameters
        # (colorbar included) must exist here too.  The overview keeps the
        # colorbar hidden through the renderer's visibility mechanism.
        entries.extend(_image_parameters(style))
    if kind is PlotKind.ROLLING:
        entries.extend(
            (
                _bin_count_parameter(_ROLLING_DISTRIBUTION_BIN_EFFECTS),
                _window_parameter(100),
                ParameterSpec(
                    "side_distribution",
                    bool,
                    RenderEffect.LAYOUT,
                    default=True,
                    label="Side distribution",
                ),
                # How many shots each drawn point averages: 1 is the shot
                # itself, N is the mean of the last N -- the live "rate
                # over the recent past" view, with the standard error of
                # those same N shots as its band.  It reads the retained
                # moments, so a shot that pooled more samples weighs more,
                # which is why it is inert on non-MEAN reductions (a mean
                # of maxima weighted by sample counts would be a number
                # about nothing).
                ParameterSpec(
                    "trailing",
                    int,
                    _UNCERTAINTY_EFFECTS,
                    default=1,
                    normalizer=_normalize_integer,
                    label="Trailing mean (shots)",
                    minimum=1,
                    maximum=1_000_000,
                    step=1,
                ),
            )
        )
    if semantic_kind is PlotKind.IMAGE:
        entries.append(
            ParameterSpec(
                "show_point_labels",
                bool,
                RenderEffect.OVERLAY,
                default=True,
                label="Point labels",
            )
        )
    if kind is PlotKind.PULSE_TIMELINE:
        entries.append(
            ParameterSpec(
                "show_scan_regions",
                bool,
                RenderEffect.BASE_STYLE | RenderEffect.TEXT,
                default=True,
                label="Scan regions",
            )
        )
    has_limits = any(entry.name in LIMIT_MODE_NAMES for entry in entries)
    def validate_state(values: Mapping[str, object]) -> None:
        if has_limits:
            _validate_limit_state(values)
        else:
            _validate_authored_conflicts(values)

    return ParameterSchema(
        entries,
        transition_normalizer=(
            _normalize_limit_transition if has_limits else None
        ),
        full_state_validator=validate_state,
    )


def parameter_schema_for(spec: PlotSpec, *, style: PlotStyleConfig) -> ParameterSchema:
    """Return the introspectable, complete UI parameter contract for ``spec``."""

    return _parameter_schema_for_context(
        _parameter_schema_context(spec),
        style=style,
    )


def parameter_schema_for_kind(
    kind: PlotKind | str,
    *,
    style: PlotStyleConfig,
    facet_cell_kind: PlotKind | str | None = None,
) -> ParameterSchema:
    """Return the data-independent display contract for an authored kind.

    Semantic axis choices and fit models require a dataset. Display controls
    do not, so a blank panel exposes the same surface before its signal is
    wired. A facet grid is the one exception: its display contract depends on
    the cell kind, which must therefore already be authored.
    """

    resolved = PlotKind(kind)
    semantic_kind = resolved
    if resolved is PlotKind.FACET_GRID:
        if facet_cell_kind is None:
            raise ValueError(
                "facet grid display parameters require a fixed cell kind"
            )
        semantic_kind = PlotKind(facet_cell_kind)
        if semantic_kind not in {
            PlotKind.CURVE,
            PlotKind.IMAGE,
            PlotKind.HISTOGRAM,
        }:
            raise ValueError(
                "facet grid cell kind must be curve, image, or histogram"
            )
    elif facet_cell_kind is not None:
        raise ValueError("facet_cell_kind is only valid for a facet grid")
    return _parameter_schema_for_context(
        _ParameterSchemaContext(resolved, semantic_kind, PlotLabels()),
        style=style,
    )


__all__ = [
    "FACET_FIT_PARAMETER",
    "accepts_classifier_thresholds",
    "CellPlot",
    "CurvePlot",
    "FacetGridPlot",
    "HistogramPlot",
    "history_window_requirement",
    "ImagePlot",
    "PlotLabels",
    "PlotSpec",
    "paints_image_surface",
    "PulseTimelinePlot",
    "Reduction",
    "RelimMode",
    "RollingPlot",
    "parameter_schema_for",
    "parameter_schema_for_kind",
    "semantic_spec",
]
