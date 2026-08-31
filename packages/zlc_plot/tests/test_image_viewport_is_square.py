"""Image viewports keep whole square cells and the source rows/columns ratio."""

from __future__ import annotations

import numpy as np
import pytest

from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
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


def test_a_snapped_viewport_preserves_the_source_cell_aspect() -> None:
    """Every rectangle keeps the source rows/columns ratio and whole cells."""

    session = PlotSession(
        _square_field_snapshot(),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
    )
    try:
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
            assert x_count * 48 == y_count * 64, (
                (low_x, high_x, low_y, high_y), x_count, y_count
            )
    finally:
        session.close()


def test_unequal_scan_steps_still_draw_square_cells_and_keep_the_zoom_box() -> None:
    short = np.linspace(0.0025, 0.0075, 50)
    long = np.linspace(0.01, 0.03, 50)
    short_axis = Axis.create("short", values=short, canonical_unit="s")
    long_axis = Axis.create("long", values=long, canonical_unit="s")
    point_table = PointTable.from_columns(
        {
            "short": np.tile(short, long.size),
            "long": np.repeat(long, short.size),
        }
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        point_table,
        point_topology=PointTopology.from_cartesian(
            (long_axis, short_axis), point_table=point_table
        ),
        dtype=np.float64,
        generation="unequal-scan-step-image",
    )
    session = PlotSession(
        DatasetSnapshot(schema, np.zeros((1, 2500)), 0),
        ImagePlot(
            AxisRef.point_dimension("short"),
            AxisRef.point_dimension("long"),
        ),
    )
    try:
        axes = session._renderer.primary_axes
        before = tuple(map(float, axes.bbox.bounds))
        x_step = float(short[1] - short[0])
        y_step = float(long[1] - long[0])
        origin = axes.transData.transform((short[0], long[0]))
        x_pixels = axes.transData.transform((short[1], long[0]))[0] - origin[0]
        y_pixels = axes.transData.transform((short[0], long[1]))[1] - origin[1]
        assert x_step != y_step
        assert abs(x_pixels) == pytest.approx(abs(y_pixels), rel=1.0e-9)

        session.set_viewport(
            NumericRange(short[10] - x_step / 2.0, short[30] + x_step / 2.0),
            NumericRange(long[10] - y_step / 2.0, long[30] + y_step / 2.0),
        )
        after = tuple(map(float, axes.bbox.bounds))
        assert after == pytest.approx(before, abs=1.0e-9)
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
