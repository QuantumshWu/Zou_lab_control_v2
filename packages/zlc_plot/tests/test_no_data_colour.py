"""Nothing here looks the same everywhere it appears.

An image panel used to say "no data" in two colours at once: the pixels
inside the extent came out of the colormap's bad colour and the square
letterbox band beside them came out of the axes background.  Side by side in
one panel that is two different facts, and a facet cell with nothing in it
read as a grey block while an empty image plot read as white space.

There is one answer: no data is the surface showing through.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable

from zlc_plot import AxisRef, ImagePlot, PlotSession
from zlc_plot.config import DEFAULTS


def _empty_image_session(size: int = 48) -> PlotSession:
    """One image panel whose every sample is missing."""

    repeat = Axis.create("repeat", values=np.array([0], dtype=np.int64))
    y_axis = Axis.create("camera_y", values=np.arange(size, dtype=np.int32))
    x_axis = Axis.create("camera_x", values=np.arange(size, dtype=np.int32))
    points = PointTable.from_columns({"sample": np.array([0], dtype=np.int64)})
    schema = DatasetSchema.create(
        repeat,
        points,
        data_axes=(y_axis, x_axis),
        value_label="Counts",
        canonical_unit="1",
        display_unit="1",
        dtype=np.float64,
        generation="no-data-colour",
    )
    values = np.full((1, 1, size, size), np.nan, dtype=np.float64)
    snapshot = DatasetSnapshot(schema, values, revision=1)
    spec = ImagePlot(AxisRef.data("camera_x"), AxisRef.data("camera_y"))
    return PlotSession(snapshot, spec)


def _canvas(session: PlotSession) -> np.ndarray:
    renderer = session._renderer
    renderer.draw()
    return np.array(renderer.figure.canvas.buffer_rgba(), copy=True)


def _axes_pixel(session, canvas, fx: float, fy: float) -> tuple[int, ...]:
    """One pixel of the data axes, addressed in axes fractions."""

    axes = session._renderer.figure.axes[0]
    box = axes.get_window_extent()
    height = canvas.shape[0]
    x = int(round(box.x0 + fx * box.width))
    y = int(round(height - (box.y0 + fy * box.height)))
    x = min(max(x, int(box.x0) + 1), int(box.x1) - 2)
    y = min(max(y, int(height - box.y1) + 1), int(height - box.y0) - 2)
    return tuple(int(v) for v in canvas[y, x])


def test_a_missing_pixel_is_the_same_colour_as_the_space_beside_it() -> None:
    session = _empty_image_session()
    try:
        canvas = _canvas(session)
        # The square band above the data and the middle of the data itself:
        # both are "nothing here", and an operator must not be able to tell
        # them apart by colour.
        band = _axes_pixel(session, canvas, 0.5, 0.02)
        inside = _axes_pixel(session, canvas, 0.5, 0.5)
        assert inside == band, (
            "a missing sample and the empty space beside it are the same "
            f"absence: inside={inside} band={band}"
        )
        white = (255, 255, 255, 255)
        assert inside == white, (
            "the surface background is what shows through: "
            f"inside={inside}"
        )
    finally:
        session.close()


def test_the_style_declares_no_second_colour_for_absence() -> None:
    """The colormap's bad colour was that second answer; it is gone."""

    palette = DEFAULTS.style.palette
    assert not hasattr(palette, "bad"), (
        "a palette token for missing data is a second source of truth for "
        "what absence looks like"
    )
