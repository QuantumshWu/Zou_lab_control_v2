"""Every Image keeps one square frame made of whole square cells."""

from __future__ import annotations

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, REPEAT, SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, ImagePlot, PlotSession
from zlc_plot.selectors import NumericRange, RectangleRange

def _square_field_snapshot() -> OwnedSnapshot:
    """Same pitch on both axes in the same unit: a camera frame's shape."""

    x = np.arange(64, dtype=float)
    y = np.arange(48, dtype=float)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"sample": [0.0]}),
        cell_axes=(
            axis("x", values=x, unit="m", role=SPATIAL_X),
            axis("y", values=y, unit="m", role=SPATIAL_Y),
        ),
        dtype=np.float64,
        value_unit="1",
    )
    values = np.zeros((1, 1, x.size, y.size), dtype=float)
    return make_snapshot(schema, values, revision=0)

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

def test_a_snapped_viewport_is_square_in_cell_units() -> None:
    """Every zoom keeps equal whole-cell spans inside the square frame."""

    session = PlotSession(
        _square_field_snapshot(),
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
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
            assert x_count == y_count, (
                (low_x, high_x, low_y, high_y), x_count, y_count
            )
    finally:
        session.close()

def test_unequal_scan_steps_still_draw_square_cells_and_keep_the_zoom_box() -> None:
    short = np.linspace(0.0025, 0.0075, 50)
    long = np.linspace(0.01, 0.03, 50)
    point_domain = mapped_domain_from_columns(
        {
            "short": np.tile(short, long.size),
            "long": np.repeat(long, short.size),
        },
        units={"short": "s", "long": "s"},
    )
    schema = make_dataset_schema(
        repeat_domain(size=1),
        point_domain,
        dtype=np.float64,
    )
    session = PlotSession(
        make_snapshot(schema, np.zeros((1, 2500)), 0),
        ImagePlot(
            AxisRef.point("short"),
            AxisRef.point("long"),
        ),
    )
    try:
        axes = session._renderer.primary_axes
        before = tuple(map(float, axes.bbox.bounds))
        assert before[2] == pytest.approx(before[3], abs=1.0e-9)
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
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
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
