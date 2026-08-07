"""Rolling x is the absolute shot index, never a negative countdown."""

from __future__ import annotations

import numpy as np

from zlc_plot import PlotSession, RollingPlot
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


def test_seeded_history_numbers_shots_from_zero() -> None:
    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        series = session._payload.series[0]
        np.testing.assert_array_equal(
            np.asarray(series.x.canonical), np.arange(6.0)
        )
        assert series.x.label == "Shot"
        assert float(np.asarray(series.x.canonical).min()) >= 0.0
    finally:
        session.close()


def test_live_revisions_extend_the_shot_axis_monotonically() -> None:
    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        session.update_data(_snapshot(1))
        session.update_data(_snapshot(2))
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(8.0))
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


def test_window_slides_forward_keeping_absolute_indices() -> None:
    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        window = int(session.display_state["window"])
        total = 6
        for revision in range(1, window + 3):
            session.update_data(_snapshot(revision))
            total += 1
        x = np.asarray(session._payload.series[0].x.canonical)
        assert x.size == min(window, total)
        # The window shows the most recent shots with their absolute indices.
        np.testing.assert_array_equal(x, np.arange(total - x.size, total))
    finally:
        session.close()
