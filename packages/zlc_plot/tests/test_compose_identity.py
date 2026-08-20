"""Composed frames are bit-identical to full draws, including tick churn.

The renderer's chrome-background compose repaints boundary tick marks and
grid lines as dynamics.  Those must be the exact position-fresh, view-clipped
subset a full ``Axis.draw`` paints: the raw ``majorTicks`` lists keep stale
instances parked at out-of-view locations after a limit change, and painting
those leaked mark segments outside the axes box (the "stray tick lines"
beside a 2D image's distribution pane).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable

from zlc_plot import AxisRef, ImagePlot, PlotLabels, PlotSession
from zlc_plot.rendering import MatplotlibRenderer
from zlc_plot.selectors import NumericRange


def _image_contract(size: int):
    repeat = Axis.create("repeat", values=np.array([0], dtype=np.int64))
    y_axis = Axis.create("camera_y", values=np.arange(size, dtype=np.int32))
    x_axis = Axis.create("camera_x", values=np.arange(size, dtype=np.int32))
    points = PointTable.from_columns({"sample": np.array([0], dtype=np.int64)})
    return DatasetSchema.create(
        repeat,
        points,
        data_axes=(y_axis, x_axis),
        value_label="Counts",
        canonical_unit="1",
        display_unit="1",
        dtype=np.uint16,
        generation="compose-identity",
    )


def _snapshot(schema, size: int, scale: float, revision: int, seed: int):
    rng = np.random.default_rng(seed)
    frame = (
        rng.normal(0.5, 0.15, size=(size, size)).clip(0.01, 1.0) * scale
    ).astype(np.uint16)
    return DatasetSnapshot(
        schema, frame[np.newaxis, np.newaxis, :, :], revision=revision
    )


def _composed_matches_full_draw(session) -> int:
    renderer = session._renderer
    composed = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
    renderer.draw()
    full = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
    return int(np.count_nonzero(np.any(composed != full, axis=-1)))


def test_composed_frame_is_full_draw_exact_across_a_tick_shrink() -> None:
    size = 128
    schema = _image_contract(size)
    session = PlotSession(
        _snapshot(schema, size, 40000.0, 1, seed=3),
        ImagePlot(
            AxisRef.data("camera_x"),
            AxisRef.data("camera_y"),
            labels=PlotLabels("compose", "x", "y", value="Counts"),
        ),
    )
    try:
        session.update_data(_snapshot(schema, size, 40000.0, 2, seed=4))
        assert _composed_matches_full_draw(session) == 0

        # The value range collapses 1000x: the distribution/colorbar tick
        # sets shrink, stranding stale tick instances at wide positions.
        session.update_data(_snapshot(schema, size, 35.0, 3, seed=5))
        assert _composed_matches_full_draw(session) == 0

        # Steady state composes over the cached chrome background.
        session.update_data(_snapshot(schema, size, 30.0, 4, seed=6))
        assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()


def test_color_limit_preview_composes_without_touching_chrome() -> None:
    size = 128
    schema = _image_contract(size)
    session = PlotSession(
        _snapshot(schema, size, 200.0, 1, seed=9),
        ImagePlot(
            AxisRef.data("camera_x"),
            AxisRef.data("camera_y"),
            labels=PlotLabels("preview", "x", "y", value="Counts"),
        ),
    )
    try:
        session.update_data(_snapshot(schema, size, 200.0, 2, seed=10))
        renderer = session._renderer
        before = renderer._background_signature
        with renderer.raster_transaction():
            renderer.preview_color_limits(20.0, 150.0)
        # The preview repainted pixels without invalidating the chrome
        # background: no colorbar label rewrite, no full recapture.
        assert not renderer._chrome_dirty_axes
        assert renderer._background_signature == before
        image = renderer._artists["image"]
        assert tuple(map(float, image.get_clim())) == (20.0, 150.0)
    finally:
        session.close()


@pytest.mark.parametrize("device_pixel_ratio", (1.0, 2.0))
def test_tight_chrome_reuses_one_exact_background_without_residue(
    monkeypatch,
    device_pixel_ratio: float,
) -> None:
    size = 128
    schema = _image_contract(size)
    spec = ImagePlot(
        AxisRef.data("camera_x"),
        AxisRef.data("camera_y"),
        labels=PlotLabels("tight", "x", "y", value="Counts"),
    )
    session = PlotSession(
        _snapshot(schema, size, 40000.0, 1, seed=12),
        spec,
        device_pixel_ratio=device_pixel_ratio,
    )
    try:
        session.update_data(_snapshot(schema, size, 35000.0, 2, seed=13))
        renderer = session._renderer
        background = renderer._background_region
        native_draw = MatplotlibRenderer._native_draw
        reference = PlotSession(
            _snapshot(schema, size, 35.0, 3, seed=14),
            spec,
            device_pixel_ratio=device_pixel_ratio,
        )
        try:
            reference._renderer.draw()
            small_full = np.array(
                reference._renderer.figure.canvas.buffer_rgba(),
                copy=True,
            )
        finally:
            reference.close()
        native_draws = 0

        def counted_draw(canvas) -> None:
            nonlocal native_draws
            native_draws += 1
            native_draw(canvas)

        monkeypatch.setattr(
            MatplotlibRenderer,
            "_native_draw",
            staticmethod(counted_draw),
        )
        session.update_data(_snapshot(schema, size, 35.0, 3, seed=14))
        small = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
        _cmap, limits, _label = renderer._artists["image:colorbar_state"]
        assert tuple(renderer._artists["image:colorbar_mappable"].get_clim()) == limits
        assert tuple(renderer._artists["image:colorbar"].get_ticks()) == limits
        np.testing.assert_array_equal(small, small_full)
        session.update_data(_snapshot(schema, size, 39000.0, 4, seed=15))

        assert native_draws == 0
        assert renderer._background_region is background
        composed = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
        renderer.draw()
        full = np.array(renderer.figure.canvas.buffer_rgba(), copy=True)
        np.testing.assert_array_equal(composed, full)
    finally:
        session.close()


@pytest.mark.parametrize("device_pixel_ratio", (1.0, 2.0))
def test_tight_fit_and_area_selector_remain_full_draw_exact(
    device_pixel_ratio: float,
) -> None:
    size = 128
    schema = _image_contract(size)
    coordinate = np.arange(size, dtype=float)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")

    def fitted_snapshot(revision: int, shift: float) -> DatasetSnapshot:
        values = 100.0 + 4000.0 * np.exp(
            -(
                (xx - (64.0 + shift)) ** 2
                + (yy - (64.0 - shift)) ** 2
            )
            / (2.0 * 7.0**2)
        )
        return DatasetSnapshot(
            schema,
            values.astype(np.uint16)[np.newaxis, np.newaxis, :, :],
            revision=revision,
        )

    session = PlotSession(
        fitted_snapshot(1, 0.0),
        ImagePlot(AxisRef.data("camera_x"), AxisRef.data("camera_y")),
        device_pixel_ratio=device_pixel_ratio,
    )
    try:
        session.set_area_selector(
            NumericRange(50.5, 76.5),
            NumericRange(50.5, 76.5),
            display=False,
        )
        session.fit("radial_gaussian_center", live=True)
        renderer = session._renderer
        background = renderer._background_region

        session.update_data(fitted_snapshot(2, 0.4))
        assert renderer._background_region is background
        assert _composed_matches_full_draw(session) == 0
    finally:
        session.close()
