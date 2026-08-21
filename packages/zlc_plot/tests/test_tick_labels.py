"""A tick label says what distinguishes its tick, and nothing else.

The failure these guard against: an axis over 1000000.0 .. 1000000.4 printed
``10000000 10000001 10000002`` beside a ``x1e-1``, eight characters each to
say four tenths.  The offset was taken only when the labels did not FIT the
axis width -- a question that is never asked of a y axis, so a y axis never
took one at all, whatever it was showing.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import zlc_plot.ticks as ticks_module
from zlc_plot.ticks import SmartOffsetLocator, apply_smart_ticks


#: Ranges this bench actually produces.  Each is (name, low, high).
RANGES = (
    ("an absolute shot number late in a run", 1234500.0, 1234600.0),
    ("a wavelength window", 1000000.0, 1000000.4),
    ("counts about a pedestal", 5180.0, 5220.0),
    ("a control voltage", -1.0, 2.0),
    ("an optical frequency in Hz", 384227900000.0, 384228100000.0),
    ("a normalised rate", 0.0, 1.0),
    ("a pulse length in seconds", 1.0e-6, 5.0e-6),
)

#: What a label may cost.  Four significant characters plus a sign and a
#: decimal point: "-0.25" is a label, "10000001" is a coordinate that forgot
#: to take its common part out.
LONGEST_LABEL = 6


#: What the caller is about to draw these labels at.  The policy prices its
#: candidates against exactly this, and may shrink it when two labels can no
#: longer clear each other.
LABEL_PT = 6.5


def _drawn(low: float, high: float, *, surface: str, inches: float):
    figure = plt.figure(figsize=(inches, inches * 0.75), dpi=100)
    axes = figure.add_subplot(111)
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    apply_smart_ticks(axes, label_pt=LABEL_PT, prune_edges=surface == "cell")
    figure.canvas.draw()
    return figure, axes


def _labels(axis) -> list[str]:
    return [text.get_text() for text in axis.get_ticklabels() if text.get_text()]


@pytest.mark.parametrize("name,low,high", RANGES)
@pytest.mark.parametrize("surface,inches", (("panel", 6.0), ("cell", 1.2)))
def test_no_axis_prints_a_label_longer_than_its_own_range_needs(
    name: str, low: float, high: float, surface: str, inches: float
) -> None:
    figure, axes = _drawn(low, high, surface=surface, inches=inches)
    try:
        for which, axis in (("x", axes.xaxis), ("y", axes.yaxis)):
            labels = _labels(axis)
            assert labels, f"{name}: {which} drew no labels"
            longest = max(len(value) for value in labels)
            assert longest <= LONGEST_LABEL, (
                f"{name}: {which} labels {labels} beside "
                f"offset {axis.get_offset_text().get_text()!r}"
            )
    finally:
        plt.close(figure)


def test_a_y_axis_far_from_zero_takes_the_offset_too() -> None:
    """The regression itself: the width question is not asked of y."""

    figure, axes = _drawn(1000000.0, 1000000.4, surface="panel", inches=6.0)
    try:
        assert axes.yaxis.get_offset_text().get_text() == "+1000000"
        assert axes.xaxis.get_offset_text().get_text() == "+1000000"
    finally:
        plt.close(figure)


def test_labels_that_are_already_short_keep_their_coordinates() -> None:
    """An offset is a cost: it is paid only where it buys something."""

    figure, axes = _drawn(5180.0, 5220.0, surface="panel", inches=6.0)
    try:
        assert axes.yaxis.get_offset_text().get_text() == ""
        assert _labels(axes.yaxis)[0] == "5180"
    finally:
        plt.close(figure)


def test_a_scale_and_an_offset_stay_two_statements() -> None:
    """Run together, "x1e4+3.84e11" reads as arithmetic that means nothing."""

    figure, axes = _drawn(
        384227900000.0, 384228100000.0, surface="panel", inches=6.0
    )
    try:
        text = axes.yaxis.get_offset_text().get_text()
        assert "×1e" in text and "+" in text
        assert " " in text or "\n" in text, text
    finally:
        plt.close(figure)


@pytest.mark.parametrize("surface,inches", (("panel", 6.0), ("cell", 1.2)))
def test_the_offset_is_written_in_the_figures_own_corners(
    surface: str, inches: float
) -> None:
    """Both surfaces, both parts, inside the canvas.

    In axes fractions -- what this was, and what the reference GUI uses -- the
    x offset sits a tenth of the data box BELOW the axes.  That reference
    reserves a hundred-pixel bottom margin for it; a panel laid out to fit its
    own labels does not, so the two-line form was drawn off the canvas.
    """

    figure, axes = _drawn(
        384227900000.0, 384228100000.0, surface=surface, inches=inches
    )
    try:
        for axis, position in (
            (axes.xaxis, (0.995, 0.008)),
            (axes.yaxis, (0.008, 0.995)),
        ):
            offset = axis.get_offset_text()
            assert offset.get_text(), "a range this far from zero has an offset"
            assert offset.get_position() == position
            assert offset.get_transform() is figure.transFigure
            window = offset.get_window_extent(figure.canvas.get_renderer())
            assert figure.bbox.containsx(window.x0) and figure.bbox.containsx(
                window.x1
            ), f"{surface} {axis.axis_name} offset runs off the canvas: {window}"
            assert figure.bbox.containsy(window.y0) and figure.bbox.containsy(
                window.y1
            ), f"{surface} {axis.axis_name} offset runs off the canvas: {window}"
    finally:
        plt.close(figure)


def test_a_caller_says_where_it_draws_and_how_big_never_how_many() -> None:
    """The two facts a caller owns, and the one it does not.

    How many labels an axis carries follows from its extent and its labels;
    a caller supplies only the size they are drawn at (required, because the
    policy prices against it) and whether the room past the edge is its own.
    """

    figure = plt.figure(figsize=(4.0, 3.0), dpi=100)
    try:
        axes = figure.add_subplot(111)
        with pytest.raises(TypeError):
            apply_smart_ticks(axes)
        with pytest.raises(ValueError, match="label_pt"):
            apply_smart_ticks(axes, label_pt=0.0)
        with pytest.raises(ValueError, match="which"):
            apply_smart_ticks(axes, "diagonal", label_pt=LABEL_PT)
    finally:
        plt.close(figure)


def test_identical_tick_input_reuses_layout_but_range_and_extent_do_not(
    monkeypatch,
) -> None:
    figure = plt.figure(figsize=(4.0, 3.0), dpi=100)
    try:
        axes = figure.add_subplot(111)
        axes.set_xlim(0.0, 10.0)
        apply_smart_ticks(axes, "x", label_pt=LABEL_PT)
        locator = axes.xaxis.get_major_locator()
        assert isinstance(locator, SmartOffsetLocator)
        native = locator._unit
        calls = 0

        def counted(low: float, high: float):
            nonlocal calls
            calls += 1
            return native(low, high)

        monkeypatch.setattr(locator, "_unit", counted)
        forward = locator.tick_values(0.0, 10.0)
        assert locator.tick_values(0.0, 10.0) is forward
        assert calls == 1

        reverse = locator.tick_values(10.0, 0.0)
        assert reverse == list(reversed(forward))
        assert calls == 2

        axes.set_position((0.2, 0.2, 0.35, 0.6))
        locator.tick_values(10.0, 0.0)
        assert calls == 3
    finally:
        plt.close(figure)


def test_a_settled_fine_unit_is_rejected_before_a_large_range_is_enumerated(
    monkeypatch,
) -> None:
    """A representation-scale jump must not enumerate the old tick lattice."""

    locator = SmartOffsetLocator(max_ticks=8, label_pt=LABEL_PT)
    locator.tick_values(0.0, 1.0)
    native_range = range

    def bounded_range(*args: int):
        candidate = native_range(*args)
        assert len(candidate) <= locator.max_ticks + 1
        return candidate

    monkeypatch.setattr(ticks_module, "range", bounded_range, raising=False)
    values = locator.tick_values(0.0, 4_200_000.0)
    fresh = SmartOffsetLocator(max_ticks=8, label_pt=LABEL_PT)

    assert 2 <= len(values) <= locator.max_ticks
    assert values == fresh.tick_values(0.0, 4_200_000.0)


def _rendered_labels(renderer):
    """Every label a real render drew, with the box it drew it in."""

    figure = renderer.figure
    figure.canvas.draw()
    canvas = figure.canvas.get_renderer()
    for role, axes_list in renderer.axes.items():
        for index, axes in enumerate(axes_list):
            if not axes.get_visible():
                continue
            for name, axis in (("x", axes.xaxis), ("y", axes.yaxis)):
                drawn = [
                    (label.get_text(), float(label.get_fontsize()),
                     label.get_window_extent(canvas))
                    for label in axis.get_ticklabels()
                    if label.get_text() and label.get_visible()
                ]
                if drawn:
                    yield f"{role}:{index}:{name}", name, drawn


@pytest.mark.parametrize("preset", ("2x2", "8x8"))
@pytest.mark.parametrize("shape", ((83, 60), (2048, 2048)))
@pytest.mark.parametrize("frames", (1, 3))
def test_no_two_tick_labels_come_closer_than_one_digit(
    preset: str, shape: tuple[int, int], frames: int
) -> None:
    """The rule, measured on what was DRAWN, not on what was predicted.

    Every panel this bench opens: a camera frame or a grid of them, at the
    presets an operator picks, at the ROIs this camera takes.  Two labels may
    only come within a digit of each other when the axis is already at the
    bottom of the ladder -- two labels at the smallest readable size -- and no
    two may overlap at all, whichever axes drew them, because a label that
    prints across its neighbour's reads as one wrong number.
    """

    import numpy as np

    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
    from zlc_plot import AxisRef, FacetGridPlot, ImagePlot
    from zlc_plot.session import PlotSession
    from zlc_plot.ticks import MIN_TICK_LABEL_PT, _label_size_pt

    height, width = shape
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"frame": [float(i) for i in range(frames)]}),
        data_axes=(
            Axis.create("spatial-y", size=height),
            Axis.create("spatial-x", size=width),
        ),
        dtype=np.float32,
    )
    values = np.zeros((1, frames, height, width), dtype=np.float32)
    values[..., ::7] = 1.0
    cell = ImagePlot(AxisRef.data("spatial-x"), AxisRef.data("spatial-y"))
    spec = cell if frames == 1 else FacetGridPlot(AxisRef.point("frame"), cell)
    session = PlotSession(DatasetSnapshot(schema, values, 0), spec, size=preset)
    try:
        session.rgba()
        renderer = session._renderer
        dpi = float(renderer.figure.dpi)
        boxes = []
        rails = []
        for where, name, drawn in _rendered_labels(renderer):
            boxes.extend((where, text, box) for text, _pt, box in drawn)
            if where.startswith("distribution") and name == "x":
                rails.append(len(drawn))
            if len(drawn) < 2:
                continue
            size_pt = max(pt for _text, pt, _box in drawn)
            required = _label_size_pt("0", size_pt)[0]
            spans = sorted(
                (box.x0, box.x1) if name == "x" else (box.y0, box.y1)
                for _text, _pt, box in drawn
            )
            clear = min(
                (second[0] - first[1]) / dpi * 72.0
                for first, second in zip(spans, spans[1:])
            )
            if clear + 1e-6 < required:
                assert len(drawn) <= 2 and size_pt <= MIN_TICK_LABEL_PT + 1e-6, (
                    f"{where}: {len(drawn)} labels at {size_pt}pt, "
                    f"clear {clear:.2f} < {required:.2f}"
                )
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                one, two = boxes[first], boxes[second]
                if one[0] == two[0]:
                    continue
                assert not one[2].overlaps(two[2]), (
                    f"{one[0]} {one[1]!r} prints over {two[0]} {two[1]!r}"
                )
        # The rail states a bound, never a scale.
        assert all(count <= 2 for count in rails), rails
    finally:
        session.close()
