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
from zlc_plot.primitives import ImagePointOverlay, PointStatus
from zlc_plot.selectors import NumericRange

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
    for axis, span in ((payload.x, snapped[0]), (payload.y, snapped[1])):
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
    """Every zoom keeps equal whole-cell spans inside the square frame.

    Growing to square never loses part of the requested rectangle:
    outward, and outward only, so the requested view stays wholly visible,
    which is the reason the snap rounds the way it does in the first place.
    """

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
            asked = (NumericRange(low_x, high_x), NumericRange(low_y, high_y))
            x_count, y_count = _samples(session, asked)
            assert x_count == y_count, (
                (low_x, high_x, low_y, high_y), x_count, y_count
            )
            got = session._image_viewport_on_pixel_grid(asked)
            assert got[0].low <= low_x + 1e-9 and got[0].high >= high_x - 1e-9
            assert got[1].low <= low_y + 1e-9 and got[1].high >= high_y - 1e-9
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

def _camera_window_snapshot(rows: int, columns: int) -> OwnedSnapshot:
    """A camera ROI: whole-pixel coordinates starting at the window's corner."""

    y = 1116.0 + np.arange(rows, dtype=float)
    x = 2076.0 + np.arange(columns, dtype=float)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"sample": [0.0]}),
        cell_axes=(
            axis("y", values=y, unit="m", role=SPATIAL_Y),
            axis("x", values=x, unit="m", role=SPATIAL_X),
        ),
        dtype=np.float64,
        value_unit="1",
    )
    # One bright spot on a noisy floor: something a radial fit finds.
    spot = np.exp(
        -(
            (x[None, :] - x[columns // 3]) ** 2
            + (y[:, None] - y[rows // 2]) ** 2
        )
        / (2.0 * 3.0**2)
    )
    noise = 0.05 * np.random.default_rng(7).random((rows, columns))
    return make_snapshot(schema, (spot + noise)[None, None], revision=0)

def _site_overlay(x: np.ndarray, y: np.ndarray) -> ImagePointOverlay:
    """Sites on the picture's corner pixels and two standing outside it."""

    sites = (
        (x[0], y[0]),
        (x[-1], y[-1]),
        (x[0] - 30.0, y[0] - 25.0),
        (x[-1] + 40.0, y[-1] + 35.0),
    )
    return ImagePointOverlay(
        revision=1,
        coordinates=np.asarray(sites, dtype=float),
        point_ids=tuple(f"s{index}" for index in range(len(sites))),
        static_statuses=(PointStatus.OCCUPIED,) * len(sites),
    )

@pytest.mark.parametrize(("rows", "columns"), ((80, 88), (64, 64)))
def test_a_site_overlay_never_moves_the_image_limits(rows: int, columns: int) -> None:
    """The limits are the picture's pixel edges, letterboxed to the square.

    The longer side ends exactly at its outermost pixel edges; the shorter
    one gains the same letterbox on both sides, and a square picture none.
    Site rings, their labels, sites outside the picture and a fitted ring
    are drawn on that frame and take no part in it -- neither in the limits
    the axes show nor in the data limits an autoscale would read.
    """

    snapshot = _camera_window_snapshot(rows, columns)
    spec = ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y"))
    x = 2076.0 + np.arange(columns, dtype=float)
    y = 1116.0 + np.arange(rows, dtype=float)
    span = float(max(rows, columns))
    x_pad = (span - columns) / 2.0
    y_pad = (span - rows) / 2.0
    expected_x = (x[0] - 0.5 - x_pad, x[-1] + 0.5 + x_pad)
    expected_y = (y[0] - 0.5 - y_pad, y[-1] + 0.5 + y_pad)

    bare = PlotSession(snapshot, spec, size="4x4")
    marked = PlotSession(snapshot, spec, size="4x4")
    try:
        bare.rgba()
        marked.set_parameter("show_point_labels", True)
        marked.update_image_overlay(_site_overlay(x, y))
        assert marked.fit("radial_gaussian_center", live=False).success
        marked.rgba()
        renderer = marked._renderer
        assert renderer._artists["image:points"].get_visible()
        assert renderer._fit_topologies[(None, ())][3]["ring"].get_visible()
        for session in (bare, marked):
            axes = session._renderer.primary_axes
            assert tuple(sorted(map(float, axes.get_xlim()))) == pytest.approx(
                expected_x, abs=1.0e-9
            )
            assert tuple(sorted(map(float, axes.get_ylim()))) == pytest.approx(
                expected_y, abs=1.0e-9
            )
            data = axes.dataLim
            assert expected_x[0] - 1.0e-9 <= data.x0 and data.x1 <= expected_x[1] + 1.0e-9
            assert expected_y[0] - 1.0e-9 <= data.y0 and data.y1 <= expected_y[1] + 1.0e-9
        bare_axes = bare._renderer.primary_axes
        marked_axes = marked._renderer.primary_axes
        assert marked_axes.get_xlim() == bare_axes.get_xlim()
        assert marked_axes.get_ylim() == bare_axes.get_ylim()
        assert marked_axes.bbox.bounds == bare_axes.bbox.bounds
    finally:
        marked.close()
        bare.close()
