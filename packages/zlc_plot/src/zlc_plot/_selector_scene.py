"""Backend-neutral selector geometry, styling, and labels."""

from __future__ import annotations

from ._axis_scale import LINEAR, LOG, midpoint

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeAlias

from .selectors import (
    CrosshairPoint,
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
    _selector_number,
    _selector_precision,
)

RGBA: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]


class SelectorSceneKind(str, Enum):
    COLOR_LIMITS = "color_limits"


SceneKind: TypeAlias = SelectorKind | SelectorSceneKind


@dataclass(frozen=True, slots=True)
class ColorLimitState:
    """Effective image color scale rendered beside an Image plot."""

    value: NumericRange
    kind: ClassVar[SelectorSceneKind] = SelectorSceneKind.COLOR_LIMITS


@dataclass(frozen=True, slots=True)
class ColorLimitCandidate(ColorLimitState):
    """Transient color-scale drag state; never a data selector."""


@dataclass(frozen=True, slots=True)
class SelectorTarget:
    role: str
    cell_index: int | None = None


@dataclass(frozen=True, slots=True)
class SelectorLine:
    key: str
    target: SelectorTarget
    points: tuple[Point, ...]
    color: RGBA
    width_pt: float
    zorder: float
    linestyle: str = "-"


@dataclass(frozen=True, slots=True)
class SelectorMarkers:
    key: str
    target: SelectorTarget
    points: tuple[Point, ...]
    shape: str
    size_pt: float
    facecolor: RGBA
    edgecolor: RGBA | None
    edge_width_pt: float
    zorder: float


@dataclass(frozen=True, slots=True)
class SelectorText:
    key: str
    target: SelectorTarget
    text: str
    position: Point
    horizontal_alignment: str
    color: RGBA
    font_family: str
    font_size_pt: float
    zorder: float


SelectorPrimitive: TypeAlias = SelectorLine | SelectorMarkers | SelectorText


@dataclass(frozen=True, slots=True)
class SelectorScene:
    groups: tuple[tuple[SceneKind, tuple[SelectorPrimitive, ...]], ...]

    @property
    def selector_kinds(self) -> tuple[SceneKind, ...]:
        return tuple(kind for kind, _ in self.groups)

    def primitives_for(self, kind: SceneKind) -> tuple[SelectorPrimitive, ...]:
        return next((items for key, items in self.groups if key is kind), ())


@dataclass(frozen=True, slots=True)
class SelectorSceneStyle:
    line_width_pt: float
    line_alpha: float
    handle_size_pt: float
    handle_edge_width_pt: float
    crosshair_size_pt: float
    annotation_pt: float
    side_distribution_width_pt: float
    selector_zorder: float
    font_family: str
    label_inset_fraction: float
    label_line_fraction: float
    threshold_width_pt: float
    threshold_linestyle: str


@dataclass(frozen=True, slots=True)
class SelectorItemContext:
    kind: SceneKind
    target: SelectorTarget
    label_target: SelectorTarget
    x_limits: Point
    y_limits: Point
    selector_rgba: RGBA
    color_limit_rgba: tuple[RGBA, RGBA]
    threshold_rgba: RGBA
    threshold_label_rgba: RGBA
    threshold_uses_x: bool
    threshold_text: str = ""
    #: How each axis divides the space between its limits.  The scene
    #: draws in DATA coordinates, so Matplotlib places what it is given
    #: correctly whatever the scale -- but where a handle BELONGS is a
    #: screen question, and answering it from the limits alone put the
    #: side handles of a log box nowhere near its sides.
    x_scale: str = LINEAR
    y_scale: str = LINEAR
    x_label_factor: float = 1.0
    sample_value: float | str | None = None
    sample_span: float | None = None
    sample_target: SelectorTarget | None = None
    sample_x_limits: Point | None = None
    sample_rgba: RGBA | None = None


@dataclass(frozen=True, slots=True)
class SelectorSceneOwner:
    contexts: tuple[SelectorItemContext, ...]
    style: SelectorSceneStyle
    color_context: SelectorItemContext | None = None

    def build(
        self,
        snapshot: SelectorSnapshot,
        *,
        color_state: ColorLimitState | None = None,
    ) -> SelectorScene:
        if not isinstance(snapshot, SelectorSnapshot):
            raise TypeError("selector scene build requires SelectorSnapshot")
        contexts = {item.kind: item for item in self.contexts}
        slots: dict[tuple[SelectorTarget, str], int] = {}
        groups = []
        for state in snapshot.states:
            context = contexts.get(state.kind)
            if context is not None:
                items = _selector_primitives(state, context, self.style, slots)
                if items:
                    groups.append((state.kind, items))
        if color_state is not None and self.color_context is not None:
            items = _color_limit_primitives(
                color_state,
                self.color_context,
                self.style,
                slots,
            )
            if items:
                groups.append((color_state.kind, items))
        return SelectorScene(tuple(groups))


def _label(
    key: str,
    text: str,
    target: SelectorTarget,
    anchor: str,
    color: RGBA,
    style: SelectorSceneStyle,
    slots: dict[tuple[SelectorTarget, str], int],
) -> SelectorText:
    slot = slots.get((target, anchor), 0)
    slots[target, anchor] = slot + max(1, text.count("\n") + 1)
    inset = style.label_inset_fraction
    return SelectorText(
        key,
        target,
        text,
        (inset if anchor == "left" else 1 - inset,
         max(inset, 1 - inset - style.label_line_fraction * slot)),
        anchor,
        color,
        style.font_family,
        style.annotation_pt,
        style.selector_zorder,
    )


def _selector_primitives(
    state: SelectorState,
    context: SelectorItemContext,
    style: SelectorSceneStyle,
    slots: dict[tuple[SelectorTarget, str], int],
) -> tuple[SelectorPrimitive, ...]:
    x0, x1 = context.x_limits
    y0, y1 = context.y_limits
    xp = _selector_precision(abs(x1 - x0) * context.x_label_factor)
    yp = _selector_precision(abs(y1 - y0))

    # How many decimals a number deserves depends on what the axis can
    # RESOLVE there.  A linear axis resolves the same absolute step
    # everywhere, so one precision taken from the span serves all of it.
    # A log axis resolves a constant ratio instead: across counts from
    # 0.8 to 1200 the span-derived precision prints the bottom of the
    # range as a bare integer, which there is the whole reading.
    def _decimals(value: float, span_precision: int, scale: str) -> int:
        if scale != LOG:
            return span_precision
        return _selector_precision(abs(float(value)))

    def fx(value: float) -> str:
        scaled = value * context.x_label_factor
        return _selector_number(
            scaled, _decimals(scaled, xp, context.x_scale)
        )

    def fy(value: float) -> str:
        return _selector_number(
            value, _decimals(value, yp, context.y_scale)
        )
    target, zorder = context.target, style.selector_zorder
    line_color = (*context.selector_rgba[:3],
                  context.selector_rgba[3] * style.line_alpha)

    def line(
        key: str,
        points: tuple[Point, ...],
        color: RGBA = line_color,
        width: float = style.line_width_pt,
        linestyle: str = "-",
        line_target: SelectorTarget = target,
    ) -> SelectorLine:
        return SelectorLine(key, line_target, points, color, width, zorder, linestyle)

    def label(key: str, text: str, anchor: str, color: RGBA) -> SelectorText:
        return _label(
            key, text, context.label_target, anchor, color, style, slots
        )

    if state.kind is SelectorKind.X_RANGE:
        value = state.value
        assert isinstance(value, NumericRange)
        middle_y = midpoint(y0, y1, context.y_scale)
        return (
            line("low", ((value.low, y0), (value.low, y1))),
            line("high", ((value.high, y0), (value.high, y1))),
            SelectorMarkers(
                "handles", target, ((value.low, middle_y), (value.high, middle_y)),
                "square", style.handle_size_pt, (1, 1, 1, 1),
                context.selector_rgba, style.handle_edge_width_pt, zorder,
            ),
            label("label", f"({fx(value.low)}, {fx(value.high)})", "right",
                  context.selector_rgba),
        )

    if state.kind is SelectorKind.AREA:
        value = state.value
        assert isinstance(value, RectangleRange)
        low_x, high_x, low_y, high_y = (
            value.x.low, value.x.high, value.y.low, value.y.high
        )
        middle_x = midpoint(low_x, high_x, context.x_scale)
        middle_y = midpoint(low_y, high_y, context.y_scale)
        handles = (
            (low_x, low_y), (middle_x, low_y), (high_x, low_y),
            (high_x, middle_y), (high_x, high_y), (middle_x, high_y),
            (low_x, high_y), (low_x, middle_y),
        )
        return (
            line("outline", handles + (handles[0],)),
            SelectorMarkers(
                "handles", target, handles, "square", style.handle_size_pt,
                (1, 1, 1, 1), context.selector_rgba,
                style.handle_edge_width_pt, zorder,
            ),
            label(
                "label",
                f"({fx(low_x)}, {fy(low_y)})\n({fx(high_x)}, {fy(high_y)})",
                "left",
                context.selector_rgba,
            ),
        )

    if state.kind is SelectorKind.CROSSHAIR:
        value = state.value
        assert isinstance(value, CrosshairPoint)
        sample = context.sample_value
        sample_text = None
        if isinstance(sample, float):
            sample_text = _selector_number(
                sample, _selector_precision(context.sample_span or 0)
            )
        elif sample is not None:
            sample_text = sample
        items: list[SelectorPrimitive] = [
            line("vertical", ((value.x, y0), (value.x, y1))),
            line("horizontal", ((x0, value.y), (x1, value.y))),
            SelectorMarkers(
                "point", target, ((value.x, value.y),), "circle",
                style.crosshair_size_pt, line_color, None, 0, zorder,
            ),
            label(
                "label",
                f"({fx(value.x)}, {fy(value.y)}"
                + ("" if sample_text is None else f", {sample_text}") + ")",
                "right",
                context.selector_rgba,
            ),
        ]
        if (
            isinstance(sample, float)
            and context.sample_target is not None
            and context.sample_x_limits is not None
            and context.sample_rgba is not None
        ):
            sample_x0, sample_x1 = context.sample_x_limits
            items.append(line(
                "sample", ((sample_x0, sample), (sample_x1, sample)),
                context.sample_rgba, style.annotation_pt / 4,
                line_target=context.sample_target,
            ))
        return tuple(items)

    if state.kind is SelectorKind.THRESHOLD:
        value = float(state.value)
        points = (((value, y0), (value, y1)) if context.threshold_uses_x
                  else ((x0, value), (x1, value)))
        items: tuple[SelectorPrimitive, ...] = (line(
            "threshold", points, context.threshold_rgba,
            style.threshold_width_pt, style.threshold_linestyle,
        ),)
        if context.threshold_text:
            items += (label(
                "label",
                context.threshold_text,
                "right",
                context.threshold_label_rgba,
            ),)
        return items
    return ()


def _color_limit_primitives(
    state: ColorLimitState,
    context: SelectorItemContext,
    style: SelectorSceneStyle,
    slots: dict[tuple[SelectorTarget, str], int],
) -> tuple[SelectorPrimitive, ...]:
    value = state.value
    x0, x1 = context.x_limits
    y0, y1 = context.y_limits
    precision = _selector_precision(abs(y1 - y0))
    low_color, high_color = context.color_limit_rgba

    def line(key: str, level: float, color: RGBA) -> SelectorLine:
        return SelectorLine(
            key,
            context.target,
            ((x0, level), (x1, level)),
            color,
            style.side_distribution_width_pt,
            style.selector_zorder,
        )

    items: tuple[SelectorPrimitive, ...] = (
        line("low", value.low, low_color),
        line("high", value.high, high_color),
    )
    if not isinstance(state, ColorLimitCandidate):
        return items
    text = (
        f"H low={_selector_number(value.low, precision)}  "
        f"high={_selector_number(value.high, precision)}"
    )
    return items + (
        _label(
            "label",
            text,
            context.label_target,
            "left",
            context.threshold_label_rgba,
            style,
            slots,
        ),
    )


__all__ = [
    "ColorLimitCandidate", "ColorLimitState", "SceneKind", "SelectorItemContext",
    "SelectorLine", "SelectorMarkers",
    "SelectorPrimitive", "SelectorScene", "SelectorSceneOwner",
    "SelectorSceneKind", "SelectorSceneStyle", "SelectorTarget", "SelectorText",
]
