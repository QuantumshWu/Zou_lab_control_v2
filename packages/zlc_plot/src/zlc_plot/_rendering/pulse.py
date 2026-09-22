"""Pulse timeline artist updates."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .._pulse_time import pulse_content_bounds, pulse_time_scale
from ..primitives import PulseAnalogTrace, PulseTimelineData
from ..state import DisplayState
from ..style import PlotStyleConfig


def _sync_lines(axis: Any, artists: MutableMapping[str, Any], key: str, count: int) -> list[Any]:
    lines: list[Any] = artists.setdefault(key, [])
    while len(lines) < count:
        (line,) = axis.plot([], [])
        line.set_clip_on(True)
        lines.append(line)
    for index, line in enumerate(lines):
        line.set_visible(index < count)
    return lines


def _sync_rectangles(
    axis: Any,
    artists: MutableMapping[str, Any],
    key: str,
    count: int,
) -> list[Any]:
    from matplotlib.patches import Rectangle

    rectangles: list[Any] = artists.setdefault(key, [])
    while len(rectangles) < count:
        rectangle = Rectangle((0.0, 0.0), 0.0, 0.0)
        rectangle.set_clip_on(True)
        axis.add_patch(rectangle)
        rectangles.append(rectangle)
    for index, rectangle in enumerate(rectangles):
        rectangle.set_visible(index < count)
    return rectangles


def _sync_texts(axis: Any, artists: MutableMapping[str, Any], key: str, count: int) -> list[Any]:
    texts: list[Any] = artists.setdefault(key, [])
    while len(texts) < count:
        text = axis.text(0.0, 0.0, "")
        text.set_clip_on(True)
        texts.append(text)
    for index, text in enumerate(texts):
        text.set_visible(index < count)
    return texts


class SpanLabel(Text):
    """A name printed over a span of time, only where the span is wide
    enough ON SCREEN to hold it.

    Whether a block or a period can carry its name is a fact of the drawn
    picture, not of the document: the same 0.3 µs period hides its name in
    the home view and shows it once the operator zooms in.  So the decision
    is taken at draw time, against the span's pixel width under the axes'
    current transform and the text's rendered extent -- never against a
    fraction of the total duration, which no zoom could change.
    """

    #: The span the label names, in data (time) units.
    span: tuple[float, float] = (0.0, 0.0)
    #: Breathing room demanded on each side of the text, in points.
    pad_pt: float = 0.0
    #: Whether the last draw found room and printed the name.
    fitted: bool = False

    def draw(self, renderer: Any) -> None:
        self.fitted = False
        if not self.get_visible() or not self.get_text():
            self.stale = False
            return
        start, stop = self.span
        x0, x1 = self.get_transform().transform([(start, 0.0), (stop, 0.0)])[:, 0]
        width = float(self.get_window_extent(renderer).width)
        if width + 2.0 * renderer.points_to_pixels(self.pad_pt) > abs(float(x1) - float(x0)):
            self.stale = False
            return
        self.fitted = True
        super().draw(renderer)


class RepeatBracketSide(Line2D):
    """One side of a loop bracket: a rail from foot to foot, with feet whose
    length is a fraction of the axes' width ON SCREEN.

    Drawn in data units the feet grew with every zoom, until a loop that
    spanned the view had feet running across all of it.  So the foot is
    measured at draw time against the axes' pixel width under the current
    transform, and capped at a fraction of the loop's own span so the two
    feet of a short loop never cross.
    """

    #: The loop in data (time) units and which end this side stands at.
    start: float = 0.0
    stop: float = 0.0
    at_start: bool = True
    #: The rail's vertical extent in axis units, and how far above y_high
    #: the top rail is lifted, in points: nested loops stack by points so
    #: that each line clears the label of the one inside it at any size.
    y_low: float = 0.0
    y_high: float = 0.0
    lift_pt: float = 0.0
    #: The foot as a fraction of the axes' width, and its cap as a fraction
    #: of the loop's span.
    foot_fraction: float = 0.0
    max_foot_fraction: float = 0.0

    def draw(self, renderer: Any) -> None:
        axes = self.axes
        if axes is None or not self.get_visible():
            self.stale = False
            return
        pixels = axes.transData.transform([(self.start, 0.0), (self.stop, 0.0)])[:, 0]
        span_pixels = abs(float(pixels[1] - pixels[0]))
        foot_pixels = min(
            float(axes.get_window_extent(renderer).width) * self.foot_fraction,
            span_pixels * self.max_foot_fraction,
        )
        span = self.stop - self.start
        foot = span * foot_pixels / span_pixels if span_pixels > 0.0 else 0.0
        x = self.start if self.at_start else self.stop
        toe = x + foot if self.at_start else x - foot
        top = self.y_high
        if self.lift_pt:
            unit = axes.transData.transform([(0.0, 0.0), (0.0, 1.0)])[:, 1]
            pixels_per_unit = abs(float(unit[1] - unit[0]))
            if pixels_per_unit > 0.0:
                top += self.lift_pt * renderer.points_to_pixels(1.0) / pixels_per_unit
        self.set_data((toe, x, x, toe), (top, top, self.y_low, self.y_low))
        super().draw(renderer)


def _room_above(axis: Any, room_pt: float, below: float) -> float:
    """Axis units that put ``room_pt`` points above a line ``below`` units
    up from the bottom limit, once the limits are set so.

    The points are what the stacked loop lines and the outermost label
    need; the axes' height on screen is what they are measured against.
    An axes too short to hold them gives up under half of itself.
    """

    figure = axis.figure
    room_pixels = room_pt * figure.dpi / 72.0
    axes_pixels = axis.get_position().height * figure.get_figheight() * figure.dpi
    room_pixels = min(room_pixels, 0.45 * axes_pixels)
    return room_pixels * below / (axes_pixels - room_pixels)


def _sync_bracket_sides(
    axis: Any, artists: MutableMapping[str, Any], key: str, count: int
) -> list[RepeatBracketSide]:
    sides: list[RepeatBracketSide] = artists.setdefault(key, [])
    while len(sides) < count:
        side = RepeatBracketSide([], [])
        side.set_clip_on(True)
        axis.add_line(side)
        sides.append(side)
    for index, side in enumerate(sides):
        side.set_visible(index < count)
    return sides


def _sync_annotations(
    axis: Any, artists: MutableMapping[str, Any], key: str, count: int
) -> list[Any]:
    """Texts anchored at a data point and offset from it by points."""

    notes: list[Any] = artists.setdefault(key, [])
    while len(notes) < count:
        note = axis.annotate("", xy=(0.0, 0.0), xytext=(0.0, 0.0), textcoords="offset points")
        note.set_clip_on(True)
        notes.append(note)
    for index, note in enumerate(notes):
        note.set_visible(index < count)
    return notes


def _sync_span_labels(
    axis: Any, artists: MutableMapping[str, Any], key: str, count: int
) -> list[SpanLabel]:
    labels: list[SpanLabel] = artists.setdefault(key, [])
    while len(labels) < count:
        label = SpanLabel(0.0, 0.0, "", transform=axis.transData)
        label.set_clip_on(True)
        axis.add_artist(label)
        labels.append(label)
    for index, label in enumerate(labels):
        label.set_visible(index < count)
    return labels


def _analog_geometry(
    trace: PulseAnalogTrace,
    row_base: float,
    row_height: float,
) -> tuple[str, float, float, float, np.ndarray, np.ndarray]:
    span = trace.maximum - trace.minimum
    zero_fraction = float(np.clip((0.0 - trace.minimum) / span, 0.0, 1.0))
    zero_y = row_base + row_height * zero_fraction
    starts = np.asarray(trace.starts, dtype=float)
    values = np.asarray(trace.values, dtype=float)
    return trace.name, trace.minimum, span, zero_y, starts, values


def update_pulse_timeline(
    axis: Any,
    payload: PulseTimelineData,
    state: DisplayState,
    style: PlotStyleConfig,
    artists: MutableMapping[str, Any],
) -> None:
    """Update a pulse timeline without replacing its data artists."""

    if not isinstance(payload, PulseTimelineData):
        raise TypeError("pulse timeline requires PulseTimelineData")
    if not isinstance(state, DisplayState):
        raise TypeError("state must be DisplayState")
    if not isinstance(style, PlotStyleConfig):
        raise TypeError("style must be PlotStyleConfig")
    if not isinstance(artists, MutableMapping):
        raise TypeError("artists must be a mutable mapping")

    channels = payload.channels
    analog_traces = payload.analog_traces
    loop_markers = payload.loop_markers
    scan_dac_segments = payload.scan_dac_segments
    pulse = style.pulse

    analog_keys = tuple(
        f"analog:{index}:{trace.name}"
        for index, trace in enumerate(analog_traces)
    )
    row_keys = tuple(channel.channel_id for channel in channels) + analog_keys
    row_count = len(row_keys)
    row_index = {key: row_count - 1 - index for index, key in enumerate(row_keys)}
    row_colors = {
        key: style.palette.pulse_cycle[index % len(style.palette.pulse_cycle)]
        for index, key in enumerate(row_keys)
    }
    row_height = (
        pulse.row_height
        if row_count <= pulse.dense_row_threshold
        else max(pulse.dense_min_row_height, pulse.dense_total_height / row_count)
    )
    baseline_offset = row_height / 2.0

    start_min, stop_max = pulse_content_bounds(payload)
    span = stop_max - start_min
    margin = max(span * pulse.x_margin_fraction, np.finfo(float).eps * span)
    left_limit = start_min - margin
    right_limit = stop_max + margin

    baseline_y: dict[str, float] = {}
    baselines = _sync_lines(axis, artists, "pulse:baselines", len(channels))
    for index, channel in enumerate(channels):
        y = row_index[channel.channel_id] - baseline_offset
        baseline_y[channel.channel_id] = y
        line = baselines[index]
        line.set_data((left_limit, right_limit), (y, y))
        line.set_color(row_colors[channel.channel_id])
        line.set_linewidth(pulse.trace_linewidth)
        line.set_alpha(1.0)
        line.set_linestyle("-")
        line.set_zorder(pulse.base_zorder)

    blocks = _sync_rectangles(axis, artists, "pulse:blocks", len(payload.blocks))
    block_labels = _sync_span_labels(axis, artists, "pulse:block_labels", len(payload.blocks))
    label_style = {
        "fontsize": style.fonts.pulse_bar_label_pt,
        "color": style.palette.pulse_name,
    }
    for index, block in enumerate(payload.blocks):
        baseline = baseline_y[block.channel_id]
        color = row_colors[block.channel_id]
        rectangle = blocks[index]
        rectangle.set_xy((block.start, baseline))
        rectangle.set_width(block.stop - block.start)
        rectangle.set_height(row_height)
        rectangle.set_facecolor(color)
        rectangle.set_edgecolor("none")
        rectangle.set_linewidth(0.0)
        rectangle.set_alpha(1.0)
        rectangle.set_zorder(pulse.base_zorder)
        label = block_labels[index]
        label.set_position(((block.start + block.stop) / 2.0, row_index[block.channel_id]))
        label.set_text(block.label)
        label.set_ha("center")
        label.set_va("center")
        label.set_clip_on(True)
        label.set_zorder(pulse.base_zorder + 1.0)
        label.update(label_style)
        label.span = (block.start, block.stop)
        label.pad_pt = pulse.label_fit_pad_pt
        label.set_visible(bool(block.label))

    analog_zero_lines = _sync_lines(
        axis,
        artists,
        "pulse:analog_zero",
        len(analog_traces),
    )
    analog_value_lines = _sync_lines(
        axis,
        artists,
        "pulse:analog_values",
        len(analog_traces),
    )
    analog_ranges: dict[str, tuple[float, float, float]] = {}
    for index, (key, trace) in enumerate(zip(analog_keys, analog_traces)):
        row_base = row_index[key] - baseline_offset
        baseline_y[key] = row_base
        name, minimum, value_span, zero_y, trace_starts, trace_values = _analog_geometry(
            trace,
            row_base,
            row_height,
        )
        analog_ranges[name] = (row_base, minimum, value_span)
        color = row_colors[key]
        zero_line = analog_zero_lines[index]
        zero_line.set_data((left_limit, right_limit), (zero_y, zero_y))
        zero_line.set_color(color)
        zero_line.set_linewidth(pulse.trace_linewidth)
        zero_line.set_alpha(pulse.analog_zero_alpha)
        zero_line.set_linestyle(pulse.analog_zero_dash)
        zero_line.set_zorder(pulse.base_zorder + 1.0)
        value_line = analog_value_lines[index]
        count = min(trace_values.size, max(0, trace_starts.size - 1))
        if count:
            x = np.repeat(trace_starts[: count + 1], 2)[1:-1]
            y_values = row_base + row_height * np.clip(
                (trace_values[:count] - minimum) / value_span,
                0.0,
                1.0,
            )
            value_line.set_data(x, np.repeat(y_values, 2))
            value_line.set_visible(True)
        else:
            value_line.set_data((left_limit, right_limit), (zero_y, zero_y))
            value_line.set_visible(True)
        value_line.set_color(color)
        value_line.set_linewidth(pulse.trace_linewidth)
        value_line.set_alpha(1.0)
        value_line.set_linestyle("-")
        value_line.set_zorder(pulse.base_zorder + 2.0)

    show_scan = bool(state["show_scan_regions"])
    scan_rectangles = _sync_rectangles(
        axis,
        artists,
        "pulse:scan_regions",
        len(payload.scan_regions),
    )
    scan_labels = _sync_texts(
        axis,
        artists,
        "pulse:scan_labels",
        len(payload.scan_regions),
    )
    area_bottom = min(baseline_y.values(), default=-baseline_offset)
    area_top = max(baseline_y.values(), default=0.0) + row_height
    scan_text = {
        "fontsize": style.fonts.pulse_scan_annotation_pt,
        "color": style.artists.pulse_scan_annotation_color,
    }
    def _slot_color(record) -> str:
        """One colour per kind of slot, asked once and used by every artist."""

        return (
            style.artists.pulse_api_region_color
            if getattr(record, "kind", "scan") == "api"
            else style.artists.pulse_scan_region_color
        )

    def _badge(record) -> dict:
        return {
            "boxstyle": f"{'circle' if len(record.label) == 1 else 'round'},pad={pulse.scan_badge_pad:g}",
            "facecolor": _slot_color(record),
            "edgecolor": "none",
        }
    for index, region in enumerate(payload.scan_regions):
        rectangle = scan_rectangles[index]
        rectangle.set_xy((region.start, area_bottom))
        rectangle.set_width(region.stop - region.start)
        rectangle.set_height(area_top - area_bottom)
        rectangle.set_facecolor(_slot_color(region))
        rectangle.set_edgecolor("none")
        rectangle.set_alpha(pulse.scan_region_alpha)
        rectangle.set_zorder(pulse.scan_region_zorder)
        rectangle.set_visible(show_scan)
        text = scan_labels[index]
        text.set_position(((region.start + region.stop) / 2.0, area_top - row_height / 2.0))
        text.set_text(region.label)
        text.set_ha("center")
        text.set_va("center")
        text.set_zorder(pulse.annotation_zorder)
        text.set_bbox(_badge(region))
        text.update(scan_text)
        text.set_visible(show_scan)

    dac_lines = _sync_lines(
        axis,
        artists,
        "pulse:scan_dac",
        len(scan_dac_segments),
    )
    dac_labels = _sync_texts(
        axis,
        artists,
        "pulse:scan_dac_labels",
        len(scan_dac_segments),
    )
    for index, segment in enumerate(scan_dac_segments):
        geometry = analog_ranges[segment.trace_name]
        line = dac_lines[index]
        text = dac_labels[index]
        row_base, minimum, value_span = geometry
        start = segment.start
        stop = segment.stop
        value = segment.value
        y = row_base + row_height * float(np.clip((value - minimum) / value_span, 0.0, 1.0))
        line.set_data((start, stop), (y, y))
        line.set_color(_slot_color(segment))
        line.set_linewidth(pulse.scan_dac_linewidth)
        line.set_alpha(pulse.scan_dac_alpha)
        line.set_solid_capstyle("butt")
        line.set_zorder(pulse.scan_dac_zorder)
        line.set_visible(show_scan)
        text.set_position(((start + stop) / 2.0, row_base + row_height / 2.0))
        text.set_text(segment.label)
        text.set_ha("center")
        text.set_va("center")
        text.set_zorder(pulse.annotation_zorder)
        text.set_bbox(_badge(segment))
        text.update(scan_text)
        text.set_visible(show_scan and bool(segment.label))

    # THE PERIODS, NAMED.  A band above the top row carries each period's
    # name over its span and a rule at every boundary runs down the rows,
    # so the drawing reads the way the pulse was written -- by period --
    # and the brackets and the frame make room for the band.
    periods = payload.periods
    band = pulse.period_band_height if periods else 0.0
    row_top = row_count - 1 + row_height / 2.0
    edges = tuple(mark.start for mark in periods) + (
        (periods[-1].stop,) if periods else ()
    )
    boundaries = _sync_lines(axis, artists, "pulse:period_bounds", len(edges))
    for index, edge in enumerate(edges):
        line = boundaries[index]
        line.set_data((edge, edge), (pulse.ylim_bottom, row_top + band))
        line.set_color(style.palette.pulse_period)
        line.set_linewidth(pulse.period_boundary_linewidth)
        line.set_alpha(pulse.period_boundary_alpha)
        line.set_linestyle(pulse.period_boundary_dash)
        line.set_clip_on(True)
        line.set_zorder(pulse.base_zorder - 1.0)
    # A spacer is hatched across the rows and the band, and carries no name:
    # it is time given to a device, not a period anybody reads by name.
    spacers = tuple(mark for mark in periods if mark.spacer)
    hatches = _sync_rectangles(axis, artists, "pulse:spacers", len(spacers))
    for index, mark in enumerate(spacers):
        rectangle = hatches[index]
        rectangle.set_xy((mark.start, pulse.ylim_bottom))
        rectangle.set_width(mark.stop - mark.start)
        rectangle.set_height(row_top + band - pulse.ylim_bottom)
        rectangle.set_facecolor("none")
        rectangle.set_edgecolor(style.palette.pulse_period)
        rectangle.set_linewidth(0.0)
        rectangle.set_hatch(pulse.spacer_hatch)
        rectangle.set_alpha(pulse.spacer_alpha)
        rectangle.set_zorder(pulse.base_zorder - 1.0)
    period_labels = _sync_span_labels(axis, artists, "pulse:period_labels", len(periods))
    for index, mark in enumerate(periods):
        text = period_labels[index]
        text.set_position(((mark.start + mark.stop) / 2.0, row_top + band / 2.0))
        text.set_text("" if mark.spacer else mark.name)
        text.set_ha("center")
        text.set_va("center")
        text.set_clip_on(True)
        text.set_zorder(pulse.base_zorder + 1.0)
        text.update(
            {
                "fontsize": style.fonts.pulse_bar_label_pt,
                "color": style.palette.pulse_period,
            }
        )
        text.span = (mark.start, mark.stop)
        text.pad_pt = pulse.label_fit_pad_pt
        text.set_visible(bool(mark.name))
    left_brackets = _sync_bracket_sides(
        axis,
        artists,
        "pulse:loop_left",
        len(loop_markers),
    )
    right_brackets = _sync_bracket_sides(
        axis,
        artists,
        "pulse:loop_right",
        len(loop_markers),
    )
    bracket_labels = _sync_annotations(
        axis,
        artists,
        "pulse:loop_labels",
        len(loop_markers),
    )
    for index, marker in enumerate(loop_markers):
        start, stop, label_value = marker.start, marker.stop, marker.label
        color = style.palette.bracket_cycle[index % len(style.palette.bracket_cycle)]
        # Callers state nested loops from inner to outer.  Later markers must
        # therefore grow around earlier ones; reversing this made an internal
        # Bracket visually surround the complete Run loop.
        outer_depth = index
        y_low = pulse.repeat_bottom - pulse.repeat_bottom_step * outer_depth
        y_high = row_count + band + pulse.repeat_top_offset
        lift_pt = pulse.repeat_top_step_pt * outer_depth
        left_line = left_brackets[index]
        right_line = right_brackets[index]
        for line, at_start in ((left_line, True), (right_line, False)):
            line.start = start
            line.stop = stop
            line.at_start = at_start
            line.y_low = y_low
            line.y_high = y_high
            line.lift_pt = lift_pt
            line.foot_fraction = pulse.repeat_foot_axes_fraction
            line.max_foot_fraction = pulse.repeat_max_foot_fraction
            line.set_color(color)
            line.set_alpha(pulse.repeat_alpha)
            line.set_linewidth(pulse.repeat_linewidth)
            line.set_solid_capstyle("round")
            line.set_clip_on(True)
            line.set_zorder(pulse.repeat_bracket_zorder + index)
        text = bracket_labels[index]
        text.xy = (stop, y_high)
        offset_x, offset_y = pulse.repeat_label_offset_pt
        text.set_position((offset_x, offset_y + lift_pt))
        text.set_text(label_value)
        text.set_ha("right")
        text.set_va("bottom")
        text.set_alpha(pulse.repeat_alpha)
        text.set_clip_on(True)
        text.set_in_layout(False)
        text.set_zorder(pulse.repeat_label_zorder + index)
        text.update({"fontsize": style.fonts.pulse_repeat_pt, "color": color})
        text.set_visible(bool(label_value))

    axis.set_xlim(left_limit, right_limit)
    top_limit = row_count + band + pulse.ylim_top_offset
    bottom_limit = pulse.ylim_bottom
    if loop_markers:
        # Every bracket's foot stands INSIDE the axes, or its bottom rail
        # is clipped away at the edge.  The footer clears the second
        # bracket's foot by a margin; a deeper bracket keeps that same
        # margin under its own foot, so one or two brackets draw exactly
        # as they always have and a third is drawn complete.
        lowest_foot = pulse.repeat_bottom - pulse.repeat_bottom_step * (
            len(loop_markers) - 1
        )
        margin = pulse.repeat_bottom - pulse.repeat_bottom_step - pulse.ylim_bottom
        bottom_limit = min(bottom_limit, lowest_foot - margin)
        # The top rails stack by points above the innermost one and the
        # outermost carries its label: room for exactly that, on screen.
        innermost_top = row_count + band + pulse.repeat_top_offset
        room_pt = (
            pulse.repeat_top_step_pt * (len(loop_markers) - 1) + pulse.repeat_ylim_room_pt
        )
        top_limit = innermost_top + _room_above(axis, room_pt, innermost_top - bottom_limit)
    axis.set_ylim(bottom_limit, top_limit)
    axis.set_yticks([row_index[key] for key in row_keys])
    row_labels = [channel.label for channel in channels] + [
        trace.label for trace in analog_traces
    ]
    axis.set_yticklabels(
        row_labels,
        fontsize=max(
            pulse.ytick_font_floor_pt,
            style.fonts.tick_pt - pulse.ytick_font_delta_pt,
        ),
    )
    for tick, key in zip(axis.get_yticklabels(), row_keys):
        tick.set_color(row_colors[key])

    display_factor, display_unit = pulse_time_scale(
        payload,
        state["x_display_unit"],
        source_span=span,
    )
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    axis.set_xlabel(
        f"Time ({display_unit})",
        fontsize=style.fonts.axis_label_pt,
    )
    axis.set_ylabel("")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=pulse.xtick_count, prune="both"))
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: ""
            if value < 0.0
            else f"{value * display_factor:.4g}"
        )
    )
    axis.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
        pad=pulse.xtick_pad_pt,
    )
    axis.set_axisbelow(True)
    axis.grid(False, axis="y")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(True)
    axis.spines["left"].set_visible(True)
__all__ = ["update_pulse_timeline"]
