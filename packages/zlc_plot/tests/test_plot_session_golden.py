from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    ImagePlot,
    PlotSession,
    curve,
    histogram,
    image,
)
from test_facet_live_fit import _facet_snapshot, _spec as facet_spec


@pytest.fixture
def snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns(
            {
                "x": np.arange(6, dtype=np.float64),
                "y": np.tile([0.0, 1.0], 3),
            }
        ),
        dtype=np.float64,
        generation="golden-session",
    )
    return DatasetSnapshot(schema, np.arange(6, dtype=np.float64).reshape(1, 6), revision=0)


def image_snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"sample": [0.0]}),
        data_axes=(
            Axis.create("x", values=[0.0, 1.0, 2.0]),
            Axis.create("y", values=[0.0, 1.0]),
        ),
        dtype=np.float64,
        generation="golden-image",
    )
    values = np.arange(6, dtype=np.float64).reshape(1, 1, 3, 2)
    return DatasetSnapshot(schema, values, revision=0)


def test_initial_fixed_color_mode_materializes_the_visible_limits() -> None:
    session = PlotSession(
        image_snapshot(),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        parameters={
            "relim_mode": "fixed",
            "color_min": None,
            "color_max": None,
        },
    )
    try:
        values = session.display_state.values
        assert values["relim_mode"] == "fixed"
        assert values["color_min"] is not None
        assert values["color_max"] is not None
        assert float(values["color_min"]) < float(values["color_max"])
    finally:
        session.close()


_GOLDEN_ROOT = Path(__file__).with_name("goldens")


@pytest.mark.parametrize("kind", ["curve", "histogram", "image"])
def test_headless_plot_kinds_have_stable_rgba_goldens(
    snapshot: DatasetSnapshot,
    kind: str,
    logical_shape,
) -> None:
    selected_snapshot = image_snapshot() if kind == "image" else snapshot
    factory = {
        "curve": lambda: curve(selected_snapshot, "x"),
        "histogram": lambda: histogram(selected_snapshot),
        "image": lambda: image(selected_snapshot, AxisRef.data("x"), AxisRef.data("y")),
    }[kind]
    session = factory()
    try:
        first = session.rgba()
        second = session.rgba()
        # Derived, not restated: the plan owns the size, this owns the
        # question of whether the raster matches its plan.
        assert first.shape == logical_shape()
        assert np.array_equal(first, second)
        expected = np.asarray(
            Image.open(_GOLDEN_ROOT / f"{kind}.png").convert("RGBA"),
            dtype=np.int16,
        )
        actual = first.astype(np.int16, copy=False)
        assert expected.shape == actual.shape
        delta = np.abs(actual - expected)
        assert int(delta.max()) <= 2
        # Keep both parts of the two-tier tolerance meaningful: a one-level
        # rasterization drift may occur at antialiased edges, but it must not
        # affect more than the small edge population.  The max assertion
        # above still rejects any larger single-pixel excursion.
        changed = np.max(delta, axis=2) > 1
        assert float(np.count_nonzero(changed)) / changed.size <= 0.005
    finally:
        session.close()


def test_facet_fit_overview_has_stable_rgba_golden() -> None:
    session = PlotSession(_facet_snapshot(noisy=True), facet_spec())
    try:
        result = session.fit("gaussian_offset", live=True)
        assert all(
            any(
                parameter.standard_error is not None
                and parameter.standard_error > 1.0e-6
                for parameter in overlay.parameter_display
            )
            for overlay in result.overlays
        )
        actual = session.rgba()
        expected = np.asarray(
            Image.open(_GOLDEN_ROOT / "facet_fit.png").convert("RGBA"),
            dtype=np.int16,
        )
        assert expected.shape == actual.shape
        delta = np.abs(actual.astype(np.int16, copy=False) - expected)
        assert int(delta.max()) <= 2
        changed = np.max(delta, axis=2) > 1
        assert float(np.count_nonzero(changed)) / changed.size <= 0.005
    finally:
        session.close()
