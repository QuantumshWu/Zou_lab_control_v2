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

from zlc_plot.ticks import apply_smart_ticks


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


def _drawn(low: float, high: float, *, surface: str, inches: float):
    figure = plt.figure(figsize=(inches, inches * 0.75), dpi=100)
    axes = figure.add_subplot(111)
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    apply_smart_ticks(axes, surface=surface)
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


def test_the_surface_is_the_only_thing_a_caller_chooses() -> None:
    figure = plt.figure(figsize=(4.0, 3.0), dpi=100)
    try:
        axes = figure.add_subplot(111)
        with pytest.raises(ValueError, match="surface"):
            apply_smart_ticks(axes, surface="thumbnail")
    finally:
        plt.close(figure)
