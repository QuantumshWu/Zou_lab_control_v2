"""A square field is square in EVERY view, not only the first one.

The image panel shows a square field -- an authored requirement -- and the
box it is drawn in is square in whole device pixels.  For those two to
agree, the VIEW has to be square too.

Nothing made it so.  The viewport snapper rounded each axis outward in that
axis's own pitch, independently, so a drag that was a whole sample in x and
a fraction in y grew the y span alone: measured on a live image zoomed in,
the view sat at 1090 by 1087 samples and the skew wandered between one and
three samples as the hand moved.

Something absorbs that mismatch, and both absorbers are visible.  Where the
layout owns the box the PICTURE is stretched.  Where Matplotlib owns it --
a focused FacetGrid image cell -- the BOX shrinks to the view's aspect and,
anchored West, takes the whole shrink off its RIGHT edge: straight out of
the gap between the picture and the distribution rail, which is the gap an
operator watched breathe while dragging.

Squaring the limits afterwards is not the answer: it takes the view off the
pixel grid and one image front in twenty-seven then loses the exact-copy
blit.  Both are requirements, so the snap satisfies both.
"""

from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, ImagePlot, PlotSession
from zlc_plot.selectors import NumericRange, RectangleRange


def _square_field_snapshot() -> DatasetSnapshot:
    """Same pitch on both axes in the same unit: a camera frame's shape."""

    x = np.arange(64, dtype=float)
    y = np.arange(48, dtype=float)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"sample": [0.0]}),
        data_axes=(
            Axis.create("x", values=x, canonical_unit="m"),
            Axis.create("y", values=y, canonical_unit="m"),
        ),
        dtype=np.float64,
        canonical_unit="1",
        generation="square-viewport",
    )
    values = np.zeros((1, 1, x.size, y.size), dtype=float)
    return DatasetSnapshot(schema, values, revision=0)


def _samples(session, viewport):
    snapped = session._image_viewport_on_pixel_grid(viewport)
    payload = session._payload
    counts = []
    for axis, span in ((payload.x, snapped.x), (payload.y, snapped.y)):
        values = np.asarray(
            getattr(axis, "display", axis), dtype=float
        ).reshape(-1)
        pitch = abs((values[-1] - values[0]) / (values.size - 1))
        counts.append((span.high - span.low) / pitch)
        # Whole samples, still: the crop the front is cut to and the limits
        # the axes carry have to be the same rectangle.
        assert abs(counts[-1] - round(counts[-1])) < 1e-6, counts
    return counts


def test_a_snapped_viewport_is_square_in_samples() -> None:
    """Every rectangle a hand can ask for comes back square."""

    session = PlotSession(
        _square_field_snapshot(),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
    )
    try:
        worst = 0.0
        for low_x, high_x, low_y, high_y in (
            (10.0, 30.0, 10.0, 30.0),
            (10.3, 30.1, 10.0, 30.0),
            (10.0, 30.0, 10.4, 29.6),
            (10.7, 29.2, 10.2, 30.8),
            (0.4, 63.1, 0.6, 47.2),
            (5.5, 6.5, 5.5, 6.5),
            (12.49, 34.51, 12.51, 34.49),
        ):
            x_count, y_count = _samples(
                session,
                RectangleRange(
                    NumericRange(low_x, high_x), NumericRange(low_y, high_y)
                ),
            )
            worst = max(worst, abs(x_count - y_count))
            assert x_count == y_count, (
                (low_x, high_x, low_y, high_y), x_count, y_count
            )
        assert worst == 0.0
    finally:
        session.close()


def test_the_snap_still_contains_what_was_asked_for() -> None:
    """Growing to square never loses part of the requested rectangle.

    Outward, and outward only: the requested view stays wholly visible,
    which is the reason the snap rounds the way it does in the first place.
    """

    session = PlotSession(
        _square_field_snapshot(),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
    )
    try:
        for low_x, high_x, low_y, high_y in (
            (10.3, 30.1, 10.0, 30.0),
            (10.7, 29.2, 10.2, 30.8),
            (12.49, 34.51, 12.51, 34.49),
        ):
            asked = RectangleRange(
                NumericRange(low_x, high_x), NumericRange(low_y, high_y)
            )
            got = session._image_viewport_on_pixel_grid(asked)
            assert got.x.low <= low_x + 1e-9 and got.x.high >= high_x - 1e-9
            assert got.y.low <= low_y + 1e-9 and got.y.high >= high_y - 1e-9
    finally:
        session.close()
