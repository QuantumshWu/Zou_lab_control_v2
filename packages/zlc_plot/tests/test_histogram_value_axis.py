"""The histogram's value axis keeps the word its parameters give.

Three parameters describe it -- x_relim_mode, x_min, x_max -- and each
broke a different promise.  "fixed" held only until the first revision
that outgrew it, and every limit written after that was swallowed.  The
limits were written in the units the operator reads off the axis and
consumed as canonical.  And none of the three declared that it changes
the payload, so an edit waited for the next data revision, which never
comes once acquisition stops.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_plot import FacetGridPlot, HistogramPlot, PlotSession
from zlc_plot.specs import history_window_requirement
from zlc_plot.kinds import AxisRef
from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


def _snapshot(revision: int, high: float, unit: str | None = None) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"i": np.arange(20.0)}),
        dtype=np.float64,
        canonical_unit=unit,
        generation="histogram-value-axis",
    )
    return DatasetSnapshot(
        schema, np.linspace(0.0, high, 20).reshape(1, 20), revision=revision
    )


def _edges(session: PlotSession) -> np.ndarray:
    return np.asarray(session._payload.edges.display)


def test_a_fixed_value_axis_is_not_widened_by_the_data() -> None:
    session = PlotSession(_snapshot(1, 10.0), HistogramPlot())
    try:
        session.set_parameters(
            {"bin_count": 10, "x_relim_mode": "fixed", "x_min": 0.0, "x_max": 100.0}
        )
        assert (_edges(session)[0], _edges(session)[-1]) == (0.0, 100.0)
        session.update_data(_snapshot(2, 500.0))
        assert (_edges(session)[0], _edges(session)[-1]) == (0.0, 100.0)
    finally:
        session.close()


def test_a_fixed_value_axis_takes_a_new_limit_afterwards() -> None:
    """Retention is what NORMAL means; written as "not tight" it ate this."""

    session = PlotSession(_snapshot(1, 10.0), HistogramPlot())
    try:
        session.set_parameters(
            {"bin_count": 10, "x_relim_mode": "fixed", "x_min": 0.0, "x_max": 100.0}
        )
        session.update_data(_snapshot(2, 500.0))
        session.set_parameters(
            {"bin_count": 10, "x_relim_mode": "fixed", "x_min": 0.0, "x_max": 20.0}
        )
        assert (_edges(session)[0], _edges(session)[-1]) == (0.0, 20.0)
    finally:
        session.close()


def test_the_value_axis_answers_an_edit_without_waiting_for_data() -> None:
    """Once acquisition stops there is no next revision to wait for."""

    session = PlotSession(_snapshot(1, 10.0), HistogramPlot())
    try:
        session.set_parameters({"bin_count": 10, "x_relim_mode": "normal"})
        padded = (_edges(session)[0], _edges(session)[-1])
        session.set_parameters({"bin_count": 10, "x_relim_mode": "tight"})
        assert (_edges(session)[0], _edges(session)[-1]) == (0.0, 10.0)
        assert padded != (0.0, 10.0)
    finally:
        session.close()


def test_a_written_limit_means_the_number_on_the_axis() -> None:
    """Canonical seconds, shown in ms: 10 is ten ms, not ten thousand."""

    session = PlotSession(_snapshot(1, 0.01, unit="s"), HistogramPlot())
    try:
        session.set_parameters(
            {
                "bin_count": 10,
                "value_display_unit": "ms",
                "x_relim_mode": "fixed",
                "x_min": 0.0,
                "x_max": 10.0,
            }
        )
        assert (_edges(session)[0], _edges(session)[-1]) == (0.0, 10.0)
    finally:
        session.close()


def test_a_faceted_histogram_neither_collapses_axes_nor_leases_history() -> None:
    """It offers neither, because no facet build path does either.

    The grid unwraps to its cell for both questions, so the cell's
    `reduced` was accepted and ignored, and a `window` of 200 took a
    200-shot lease on Runtime history that the facet build never reads.
    """

    grid = FacetGridPlot(facet=AxisRef.point("i"), cell=HistogramPlot())
    assert history_window_requirement(grid, {"window": 200}) is None
    assert history_window_requirement(HistogramPlot(), {"window": 200}) == 200

    from zlc_plot.data_view import DataView, DataViewError

    snapshot = _snapshot(1, 10.0)
    view = DataView(snapshot)
    view.validate_facet(grid)
    with pytest.raises(DataViewError, match="cannot collapse axes"):
        view.validate_facet(
            FacetGridPlot(
                facet=AxisRef.point("i"),
                cell=HistogramPlot(reduced=(AxisRef.repeat(),)),
            )
        )


def test_naming_one_point_coordinate_collapses_that_coordinate() -> None:
    """A point coordinate is not a tensor axis.

    Every point column resolves to dimension 1 -- the shared point axis --
    so mapping a ref straight to a numpy axis collapsed the WHOLE point
    table for any of them, and the set() made two different columns
    literally the same reduction: on a detuning x power scan, reducing
    detuning and reducing power were two fate rows producing one
    byte-identical answer, neither of them the one the row named.
    """

    from zlc_plot.data_view import DataView

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns(
            {
                "detuning": np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
                "power": np.asarray([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
            }
        ),
        dtype=np.float64,
        generation="point-coordinate-reduce",
    )
    snapshot = DatasetSnapshot(schema, np.arange(6.0).reshape(1, 6), revision=1)
    view = DataView(snapshot)

    by_detuning = view.histogram(bins=4, reduce_axes=(AxisRef.point("detuning"),))
    by_power = view.histogram(bins=4, reduce_axes=(AxisRef.point("power"),))
    # One mean per remaining coordinate, not one for the whole scan.
    assert int(by_detuning.counts.sum()) == 3
    assert int(by_power.counts.sum()) == 2


def test_first_is_the_first_value_and_not_the_largest() -> None:
    """FIRST fell into the MIN/MAX branch and came back as MAX."""

    from zlc_plot.data_view import DataView
    from zlc_plot.specs import Reduction

    schema = DatasetSchema.create(
        Axis.create("repeat", size=3),
        PointTable.from_columns({"i": np.asarray([0.0])}),
        dtype=np.float64,
        generation="first-not-max",
    )
    values = np.asarray([[1.0], [10.0], [5.0]])
    view = DataView(DatasetSnapshot(schema, values, revision=1))
    edges = [0.0, 3.0, 8.0, 15.0]
    first = view.histogram(
        bins=edges, reduce_axes=(AxisRef.repeat(),), aggregation=Reduction.FIRST
    )
    largest = view.histogram(
        bins=edges, reduce_axes=(AxisRef.repeat(),), aggregation=Reduction.MAX
    )
    assert list(map(int, first.counts)) == [1, 0, 0]
    assert list(map(int, largest.counts)) == [0, 0, 1]
