"""Rolling x counts back from the newest shot, which sits at zero.

A rolling window shows the last N shots, so what a point MEANS is its
distance from now.  The absolute shot number is a fact about the run,
not about the picture: using it slid every tick label forward on every
single revision, so a full window never held still.
"""

from __future__ import annotations

import numpy as np

from zlc_data import PRIMARY_INDEX
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from zlc_plot import AxisRef, PlotSession, Reduction, RollingPlot
from zlc_plot.data_view import DataView
from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


def _snapshot(revision: int, repeats: int = 6) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({"x": np.arange(4.0)}),
        dtype=np.float64,
        generation="rolling-shot-axis",
    )
    values = np.arange(repeats * 4.0).reshape(repeats, 4) + float(revision)
    return DatasetSnapshot(schema, values, revision=revision)


def test_seeded_history_ends_at_zero_and_counts_back() -> None:
    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        series = session._payload.series[0]
        np.testing.assert_array_equal(
            np.asarray(series.x.canonical), np.arange(-5.0, 1.0)
        )
        assert series.x.label == "Shots from latest"
        assert float(np.asarray(series.x.canonical).max()) == 0.0
    finally:
        session.close()


def test_nonindexed_revisions_replace_instead_of_extending_the_shot_axis() -> None:
    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        session.update_data(_snapshot(1))
        session.update_data(_snapshot(2))
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-5.0, 1.0))
    finally:
        session.close()


def test_area_selector_display_coordinates_are_identity() -> None:
    """Display selector coordinates pass through unchanged on the shot axis.

    The dead negative-axis mapping used np.interp, which clamps out-of-domain
    input to the endpoints — a selector authored outside the current shot
    range collapsed to a degenerate (0, 0) span and raised.  The ordinal axis
    has no display conversion at all now.
    """

    from zlc_plot import NumericRange

    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        state = session.set_area_selector(
            NumericRange(1.0, 4.0), NumericRange(0.0, 30.0), display=True
        )
        assert state.value.x == NumericRange(1.0, 4.0)

        # Out-of-domain spans stay non-degenerate instead of being clamped.
        state = session.set_area_selector(
            NumericRange(-24.0, -8.0), NumericRange(0.0, 30.0), display=True
        )
        assert state.value.x == NumericRange(-24.0, -8.0)
    finally:
        session.close()


def test_a_full_window_holds_the_same_coordinates_as_it_slides() -> None:
    window = 100
    total = window + 8
    session = PlotSession(
        _snapshot(0, repeats=total),
        RollingPlot(),
        parameters={"window": window},
    )
    try:
        x = np.asarray(session._payload.series[0].x.canonical)
        assert x.size == min(window, total)
        # The window shows the most recent shots, the newest at zero -- the
        # same coordinates however far the run has got.
        np.testing.assert_array_equal(x, np.arange(1.0 - x.size, 1.0))
    finally:
        session.close()


def test_shot_axis_frames_the_full_window_from_the_first_revision() -> None:
    """The axis spans exactly ``window`` shots and then stands still.

    What you configure is what you see: the frame is always
    ``[-(window - 1), 0]``, the young trace grows rightward inside it to
    the newest shot at zero, and the frame never moves again.
    """

    session = PlotSession(
        _snapshot(0, repeats=30),
        RollingPlot(),
        parameters={"window": 20},
    )
    try:
        axes = session._renderer.primary_axes
        assert tuple(map(float, axes.get_xlim())) == (-19.0, 0.0)
    finally:
        session.close()


def test_window_selects_display_without_truncating_retention() -> None:
    """The window is a view, not a destructive cap on measured history.

    Shrinking must narrow the display immediately; committing new data under
    the small window must not discard older shots; enlarging must immediately
    reveal them again.
    """

    session = PlotSession(_snapshot(0, repeats=14), RollingPlot())
    try:
        session.set_parameter("window", 4)
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-3.0, 1.0))

        # Shrinking changes only the view over this Runtime-supplied Dataset.
        session.set_parameter("window", 100)
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-13.0, 1.0))
    finally:
        session.close()


def test_replace_spec_keeps_history_for_an_equivalent_rolling_spec() -> None:
    """A form submit that keeps group and reduction keeps the trace.

    Changing the reduction changes what one history point IS, so that
    replacement reseeds from the current snapshot instead.
    """

    session = PlotSession(_snapshot(0, repeats=10), RollingPlot())
    try:
        session.replace_spec(RollingPlot())
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-9.0, 1.0))

        session.replace_spec(RollingPlot(reduction=Reduction.MIN))
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-9.0, 1.0))
    finally:
        session.close()


def test_primary_index_history_keeps_source_order_holes_and_site_groups() -> None:
    source = [-2, -2, 0, 0]
    indexed_schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns(
            {"source index": source, "category": [0.0, 1.0, 0.0, 1.0]},
            ids={"source index": str(PRIMARY_INDEX_AXIS_ID)},
            roles={"source index": PRIMARY_INDEX},
        ),
        data_axes=(Axis.create("site", values=[0.0, 1.0, 2.0]),),
        dtype=np.float64,
        generation="rolling-indexed-vectorized",
    )
    indexed_values = np.arange(12.0).reshape(1, 4, 3)
    indexed_valid = np.ones(indexed_values.shape, dtype=np.bool_)
    indexed_valid[:, :2] = False
    snapshot = DatasetSnapshot(
        indexed_schema,
        indexed_values,
        revision=9,
        validity=indexed_valid,
    )
    history = DataView(snapshot).rolling_history(
        group=AxisRef.data("site"), aggregation=Reduction.MEAN
    )
    assert tuple(sample.source_index for sample in history) == (-2, 0)
    assert all(sample.revision == 9 for sample in history)
    assert all(
        sample.generation == snapshot.ref.stream_generation.value
        for sample in history
    )
    assert tuple(key[0].canonical for key in history[0].group_keys) == (
        0.0,
        1.0,
        2.0,
    )
    np.testing.assert_allclose(history[0].values, [np.nan] * 3, equal_nan=True)
    np.testing.assert_array_equal(history[0].valid, [False] * 3)
    np.testing.assert_array_equal(history[0].counts, [0] * 3)
    np.testing.assert_allclose(history[1].values, [7.5, 8.5, 9.5])
    np.testing.assert_array_equal(history[1].valid, [True] * 3)
    np.testing.assert_array_equal(history[1].counts, [2] * 3)
    np.testing.assert_allclose(history[1].sem, [1.5] * 3)

    repeat = DataView(_snapshot(0, repeats=3)).rolling_history()
    np.testing.assert_allclose(
        [sample.values[0] for sample in repeat], [1.5, 5.5, 9.5]
    )
    np.testing.assert_array_equal(
        [sample.counts[0] for sample in repeat], [4, 4, 4]
    )
    assert all(
        sample.source_index is None and sample.group_keys == ((),)
        for sample in repeat
    )
