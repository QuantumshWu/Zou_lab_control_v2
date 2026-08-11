"""Typed semantic plot specifications and their editable display parameters.

The specification describes *what* data coordinates a plot uses.  It is
immutable for the lifetime of a session.  Frequently edited presentation
values live in :class:`~zlc_plot.state.DisplayStateStore`, so a GUI slider can
change (for example) ``bin_count`` without constructing another Figure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import ClassVar, TypeAlias

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
_HISTOGRAM_PROJECTION_EFFECTS = (
    RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.FIT_SELECTION
    | RenderEffect.INTERACTION_REPROJECT
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
_ROLLING_WINDOW_EFFECTS = (
    RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.FIT_SELECTION
    | RenderEffect.INTERACTION_REPROJECT
)


class Reduction(str, Enum):
    """How repeated samples that share a plotted coordinate are combined."""

    MEAN = "mean"
    MEDIAN = "median"
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


@dataclass(frozen=True, slots=True)
class CurvePlot:
    x: AxisRef
    group: AxisRef | None = None
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.CURVE

    def __post_init__(self) -> None:
        if not isinstance(self.x, AxisRef):
            raise TypeError("CurvePlot.x must be AxisRef")
        if self.group is not None and not isinstance(self.group, AxisRef):
            raise TypeError("CurvePlot.group must be AxisRef or None")
        if self.group == self.x:
            raise ValueError("CurvePlot.group must differ from x")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("CurvePlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("CurvePlot.labels must be PlotLabels")


@dataclass(frozen=True, slots=True)
class ImagePlot:
    x: AxisRef
    y: AxisRef
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.IMAGE

    def __post_init__(self) -> None:
        if not isinstance(self.x, AxisRef) or not isinstance(self.y, AxisRef):
            raise TypeError("ImagePlot x and y must be AxisRef")
        if self.x == self.y:
            raise ValueError("ImagePlot x and y must be different axes")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("ImagePlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("ImagePlot.labels must be PlotLabels")


@dataclass(frozen=True, slots=True)
class HistogramPlot:
    """Distribution of every acquired value.

    A histogram pools the whole (repeat, points, data axes) box by
    definition — that is the one fate a distribution gives every axis, so
    there is nothing to declare per axis.  Slicing belongs to selection and
    faceting, not to the histogram itself.
    """

    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.HISTOGRAM

    def __post_init__(self) -> None:
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("HistogramPlot.labels must be PlotLabels")


@dataclass(frozen=True, slots=True)
class RollingPlot:
    group: AxisRef | None = None
    reduction: Reduction = Reduction.MEAN
    labels: PlotLabels = field(default_factory=PlotLabels)
    kind: ClassVar[PlotKind] = PlotKind.ROLLING

    def __post_init__(self) -> None:
        if self.group is not None and not isinstance(self.group, AxisRef):
            raise TypeError("RollingPlot.group must be AxisRef or None")
        if not isinstance(self.reduction, Reduction):
            raise TypeError("RollingPlot.reduction must be Reduction")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("RollingPlot.labels must be PlotLabels")


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
    kind: ClassVar[PlotKind] = PlotKind.FACET_GRID

    def __post_init__(self) -> None:
        if not isinstance(self.facet, AxisRef):
            raise TypeError("FacetGridPlot.facet must be AxisRef")
        if not isinstance(self.cell, (CurvePlot, ImagePlot, HistogramPlot)):
            raise TypeError("FacetGrid cells must be CurvePlot, ImagePlot or HistogramPlot")
        if not isinstance(self.labels, PlotLabels):
            raise TypeError("FacetGridPlot.labels must be PlotLabels")
        used = {
            getattr(self.cell, "x", None),
            getattr(self.cell, "y", None),
            getattr(self.cell, "group", None),
        }
        if self.facet in used:
            raise ValueError("facet source cannot also be a cell axis or group")


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
            _HISTOGRAM_PROJECTION_EFFECTS | RenderEffect.TEXT,
            default=False,
            label="Density",
        ),
        ParameterSpec(
            "cumulative",
            bool,
            _HISTOGRAM_PROJECTION_EFFECTS,
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
    )


def _image_parameters(
    style: PlotStyleConfig,
    *,
    default_interpolation: str,
) -> tuple[ParameterSpec[object], ...]:
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
        ParameterSpec(
            "interpolation",
            str,
            RenderEffect.BASE_STYLE,
            default=default_interpolation,
            normalizer=_normalize_nonempty_text,
            label="Interpolation",
            choices=policy.image_interpolations,
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
    return tuple(entries)


_LIMIT_PARAMETER_PAIRS = (
    ("color_min", "color_max"),
    ("y_min", "y_max"),
)


def _normalize_limit_transition(
    current: Mapping[str, object],
    updates: Mapping[str, object],
) -> Mapping[str, object]:
    normalized = dict(updates)
    if "relim_mode" in updates and updates["relim_mode"] != RelimMode.FIXED.value:
        for low_name, high_name in _LIMIT_PARAMETER_PAIRS:
            if low_name in current and high_name in current:
                normalized.setdefault(low_name, None)
                normalized.setdefault(high_name, None)
    return normalized


def _validate_limit_state(values: Mapping[str, object]) -> None:
    mode = values["relim_mode"]
    for low_name, high_name in _LIMIT_PARAMETER_PAIRS:
        if low_name not in values or high_name not in values:
            continue
        low = values[low_name]
        high = values[high_name]
        if low is not None and high is not None and float(low) >= float(high):
            raise ValueError(f"{low_name} must be smaller than {high_name}")
        if mode == RelimMode.FIXED.value and (low is None or high is None):
            raise ValueError(
                f"fixed relim_mode requires {low_name} and {high_name}"
            )
    if (
        values.get("log_y") is True
        and mode == RelimMode.FIXED.value
        and values.get("y_min") is not None
        and float(values["y_min"]) <= 0.0
    ):
        raise ValueError("log count limits require a positive y_min")


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
    semantic = spec.cell if isinstance(spec, FacetGridPlot) else spec
    return _ParameterSchemaContext(spec.kind, semantic.kind, spec.labels)


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
        entries.append(_unit_parameter("value_display_unit"))
    if semantic_kind in {PlotKind.CURVE, PlotKind.ROLLING}:
        entries.extend(_curve_parameters())
    if semantic_kind is PlotKind.HISTOGRAM:
        entries.extend(_histogram_parameters())
    if semantic_kind is PlotKind.IMAGE:
        # A FacetGrid whose cell is an image carries the FULL image surface:
        # the focused cell is the standalone Image kind, so its parameters
        # (colorbar included) must exist here too.  The overview keeps the
        # colorbar hidden through the renderer's visibility mechanism.
        entries.extend(
            _image_parameters(
                style,
                default_interpolation=(
                    style.render.facet_image_interpolation
                    if kind is PlotKind.FACET_GRID
                    else style.render.image_default_interpolation
                ),
            )
        )
    if kind is PlotKind.ROLLING:
        entries.extend(
            (
                _bin_count_parameter(_ROLLING_DISTRIBUTION_BIN_EFFECTS),
                ParameterSpec(
                    "window",
                    int,
                    _ROLLING_WINDOW_EFFECTS,
                    default=100,
                    normalizer=_normalize_integer,
                    label="Window",
                    minimum=1,
                    maximum=1_000_000,
                    step=1,
                ),
                ParameterSpec(
                    "side_distribution",
                    bool,
                    RenderEffect.LAYOUT,
                    default=True,
                    label="Side distribution",
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
    has_limits = any(entry.name == "relim_mode" for entry in entries)
    def validate_state(values: Mapping[str, object]) -> None:
        if has_limits:
            _validate_limit_state(values)
        if (
            values.get("threshold_classifier") is True
            and values.get("cumulative") is True
        ):
            raise ValueError("threshold classifier requires a non-cumulative histogram")

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
    "CellPlot",
    "CurvePlot",
    "FacetGridPlot",
    "HistogramPlot",
    "ImagePlot",
    "PlotLabels",
    "PlotSpec",
    "PulseTimelinePlot",
    "Reduction",
    "RelimMode",
    "RollingPlot",
    "parameter_schema_for",
    "parameter_schema_for_kind",
]
