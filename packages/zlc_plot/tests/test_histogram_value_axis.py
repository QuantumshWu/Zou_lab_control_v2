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


def test_a_faceted_histogram_leases_the_history_it_reads() -> None:
    """It reads one now, so it asks Runtime to keep one.

    The grid unwraps to its cell for this question, and for a while the
    facet build read no history at all -- so the lease was a hold on
    hundreds of shots for a picture drawn from the latest revision.  Both
    halves are true now: the cells pool the window, and the lease pays for
    what they pool.
    """

    grid = FacetGridPlot(facet=AxisRef.point("i"), cell=HistogramPlot())
    assert history_window_requirement(grid, {"window": 200}) == 200
    assert history_window_requirement(HistogramPlot(), {"window": 200}) == 200
    assert history_window_requirement(grid, {"window": 1}) is None


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


def test_a_reduced_histogram_is_binned_over_its_own_values() -> None:
    """The domain covers what is BINNED, and a reduce changes what that is.

    Site means over forty noisy shots span a few counts; the shots
    themselves span a hundred.  Binned into a domain taken from the shots,
    the six means landed in two bins out of twelve and the picture the
    operator asked for was a spike in the middle of an empty axis.  The
    rule was already written down for the history window -- "the domain has
    to cover what is actually being binned" -- and only half applied.
    """

    from zlc_plot import AxisRef
    from zlc_plot.specs import Reduction

    sites, repeats = 6, 40
    rng = np.random.default_rng(7)
    means = np.linspace(100.0, 105.0, sites)
    values = means[None, :] + rng.normal(0.0, 20.0, (repeats, sites))
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({"i": [0.0]}),
        data_axes=(Axis.create("site", size=sites),),
        dtype=np.float64,
        generation="reduced-domain",
    )
    snapshot = DatasetSnapshot(
        schema, values.reshape(repeats, 1, sites), revision=1
    )

    pooled = PlotSession(snapshot, HistogramPlot())
    reduced = PlotSession(
        snapshot,
        HistogramPlot(reduced=(AxisRef.repeat(),), reduction=Reduction.MEAN),
    )
    try:
        for session in (pooled, reduced):
            session.set_parameters({"bin_count": 12})
        raw_span = float(values.max()) - float(values.min())
        mean_span = float(values.mean(axis=0).max()) - float(values.mean(axis=0).min())

        pooled_edges = _edges(pooled)
        reduced_edges = _edges(reduced)
        assert pooled_edges[-1] - pooled_edges[0] >= raw_span
        # The reduced axis is sized for the means, not for the shots.
        assert reduced_edges[-1] - reduced_edges[0] < 0.5 * raw_span
        assert reduced_edges[-1] - reduced_edges[0] >= mean_span
        # Every mean is inside it, and they are spread across the bins.
        counts = np.asarray(reduced._payload.counts)
        assert int(counts.sum()) == sites
        assert int(np.count_nonzero(counts)) >= 4
    finally:
        pooled.close()
        reduced.close()
