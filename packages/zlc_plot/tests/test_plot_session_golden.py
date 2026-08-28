from __future__ import annotations

import os

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


def test_one_cleared_end_of_a_fixed_pair_materializes_like_two() -> None:
    """Half-authored is the state an operator's backspace produces.

    Choosing Fixed with nothing authored was handled -- the visible limits
    were materialised into both ends.  Clearing ONE end of an already-fixed
    pair was not, and the same schema that materialises the first refuses to
    configure on the second: "fixed relim_mode requires color_min and
    color_max".  A host that will not configure has no display vocabulary,
    so the panel's whole Setting form collapsed from twenty-nine fields to
    eleven, around an operator who was deleting a colour maximum one
    character at a time.

    Both entrances now ask the same question of the same declaration.
    """

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
        authored = dict(session.display_state.values)
        low = float(authored["color_min"])

        # Clear the top end alone, the way a backspace does, through the
        # call the console actually makes: a complete target plus the
        # AUTHORED delta.  The delta carries no mode, so anything that asks
        # the delta what the mode is decides this pair is not fixed -- and
        # the half-authored pair then reaches the full-state validator,
        # which refuses to configure.
        cleared = dict(authored)
        cleared["color_max"] = None
        session.configure(
            parameters=cleared,
            parameter_updates={"color_max": None},
        )
        values = session.display_state.values
        assert values["relim_mode"] == "fixed"
        assert float(values["color_min"]) == low, "the authored end was kept"
        assert values["color_max"] is not None, (
            "a cleared end must materialise, not reach the store as None"
        )

        # And the bottom end alone.
        cleared = dict(session.display_state.values)
        cleared["color_min"] = None
        session.configure(
            parameters=cleared,
            parameter_updates={"color_min": None},
        )
        values = session.display_state.values
        assert values["color_min"] is not None
        assert float(values["color_min"]) < float(values["color_max"])
    finally:
        session.close()


def test_replacing_the_spec_keeps_the_promise_too() -> None:
    """Every DisplayStateStore in a session, not just the two on the edit path.

    A store refuses to hold a fixed pair with a missing end, and a store
    that will not build is a host that will not configure -- which takes
    the panel's whole display vocabulary with it.  There are exactly three
    ways a store gets built; the spec-replacement one never learned the
    rule, so a half-authored pair reaching it by any route (a saved board,
    a restored layout) would have wedged the panel where no keystroke can.
    """

    session = PlotSession(
        image_snapshot(),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        parameters={"relim_mode": "fixed", "color_min": None, "color_max": None},
    )
    try:
        session.rgba()
        # Half-author it directly in the store, the shape a stored board can
        # carry, then ask for the replacement every kind change performs.
        held = dict(session.display_state.values)
        held["color_max"] = None
        prepared = session._prepare_replacement(
            session._spec,
            held,
            session._size or session.surface_plan.preset,
        )
        store = prepared[2]
        assert store.state.values["color_max"] is not None, (
            "the replacement store took a gap the picture could have filled"
        )
        assert store.state.values["relim_mode"] == "fixed"
    finally:
        session.close()


def test_a_fixed_pair_takes_its_own_axis_limits_and_says_so() -> None:
    """Every pair names where it materialises from; none inherits.

    The walk this replaced asked "is it color_min?" and gave everything else
    the Y axis.  That was right while colour and y were the only pairs and
    silently wrong the moment a third existed: the histogram's value axis
    would have been fixed to its count axis's range.  An unknown pair raises
    rather than quietly taking somebody else's numbers.
    """

    from zlc_plot.specs import limit_pairs

    pairs = limit_pairs()
    assert ("relim_mode", "color_min", "color_max") in pairs
    assert ("x_relim_mode", "x_min", "x_max") in pairs

    session = PlotSession(
        image_snapshot(),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
    )
    try:
        assert session._current_limits_for("color_min")
        assert session._current_limits_for("y_min")
        assert session._current_limits_for("x_min")
        with pytest.raises(KeyError):
            session._current_limits_for("elevation_min")
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
        if os.environ.get("ZLC_WRITE_GOLDENS"):
            # Re-baselining is a deliberate act, so it is spelled out rather
            # than inferred from a failure: the layout moved on purpose.
            Image.fromarray(np.asarray(first)).save(_GOLDEN_ROOT / f"{kind}.png")
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
