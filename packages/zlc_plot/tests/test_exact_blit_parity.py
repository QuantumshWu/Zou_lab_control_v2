"""The exact blit is a SPEED path; Matplotlib's draw is the specification.

``_blit_exact_rgba_image`` copies a precomposed RGBA front straight into
the Agg buffer instead of letting the artist draw it.  That is only ever
allowed to be faster, never different, so every surface that takes the fast
path is composed twice here -- once with the copy, once with the copy
refused so the artist draws -- and the two frames are compared pixel by
pixel.

The device pixel ratios matter: a panel margin of 16.8 design pixels lands
on a whole pixel at ratio 1 and on 50.4 at ratio 3, and it is exactly the
fractional boxes where a copy could land somewhere the draw would not.
"""
from __future__ import annotations

import numpy as np
import pytest

from zlc_plot import AxisRef, CurvePlot, ImagePlot, PlotSession
from zlc_plot import rendering as rendering_module

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


def _camera_snapshot(height: int, width: int, revision: int = 1):
    rng = np.random.default_rng(4)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": np.asarray([0.0])}),
        data_axes=(
            Axis.create("y", values=[float(i) for i in range(height)]),
            Axis.create("x", values=[float(i) for i in range(width)]),
        ),
        dtype=np.uint16,
        generation="blit-parity",
    )
    yy, xx = np.mgrid[0:height, 0:width]
    blob = 3000.0 * np.exp(
        -(((yy - height / 2) ** 2) + ((xx - width / 2) ** 2))
        / (2 * (height / 6.0) ** 2)
    )
    values = np.clip(
        blob + rng.normal(scale=90.0, size=(height, width)) + 300.0, 0, 65535
    ).astype(np.uint16)[None, None]
    return DatasetSnapshot(schema, values, revision=revision)


def _compose_both_ways(session: PlotSession) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(with_blit, with_artist_draw)`` for the current scene."""

    renderer = rendering_module.MatplotlibRenderer
    fast = np.array(session.rgba(), copy=True)
    original = renderer._blit_exact_rgba_image
    try:
        renderer._blit_exact_rgba_image = lambda self, artist, canvas: False
        session._renderer._composed_generation = -1
        drawn = np.array(session.rgba(), copy=True)
    finally:
        renderer._blit_exact_rgba_image = original
        session._renderer._composed_generation = -1
    return fast, drawn


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("preset", ["2x2", "4x4"])
def test_the_exact_blit_paints_what_the_artist_would(ratio, preset) -> None:
    """A camera image, whose front is composed to the box and copied."""

    session = PlotSession(
        _camera_snapshot(256, 256),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        device_pixel_ratio=ratio,
    )
    try:
        session.set_size(preset)
        session.update_data(_camera_snapshot(256, 256, revision=2))
        fast, drawn = _compose_both_ways(session)
    finally:
        session.close()
    np.testing.assert_array_equal(fast, drawn)


@pytest.mark.parametrize("ratio", [1.0, 1.5, 3.0])
def test_the_height_bar_scene_is_copied_exactly(ratio) -> None:
    """The 3D scene: a full-panel front on a box that ratio 3 makes fractional.

    Its frame is finished over its own background, so the copy and the
    draw must agree -- and this is the surface the fractional box used to
    exclude from the fast path entirely.
    """

    session = PlotSession(
        _camera_snapshot(64, 64),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        device_pixel_ratio=ratio,
    )
    try:
        session.set_size("4x4")
        session.set_parameters({"presentation": "height_bars"})
        session.update_data(_camera_snapshot(64, 64, revision=2))
        fast, drawn = _compose_both_ways(session)
    finally:
        session.close()
    np.testing.assert_array_equal(fast, drawn)


@pytest.mark.parametrize("ratio", [1.0, 3.0])
def test_a_curve_panel_is_unaffected_by_the_fast_path(ratio) -> None:
    """A surface with no image front composes identically either way."""

    rng = np.random.default_rng(2)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=4),
        PointTable.from_columns({"x": np.arange(64.0)}),
        generation="blit-parity-curve",
    )
    values = rng.normal(size=(4, 64))
    session = PlotSession(
        DatasetSnapshot(schema, values, revision=1),
        CurvePlot(AxisRef.point("x")),
        device_pixel_ratio=ratio,
    )
    try:
        session.set_size("4x4")
        fast, drawn = _compose_both_ways(session)
    finally:
        session.close()
    np.testing.assert_array_equal(fast, drawn)
