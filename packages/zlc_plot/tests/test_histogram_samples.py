"""A histogram is the distribution of every acquired value."""

from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import HistogramPlot, PlotSession
from zlc_plot.data_view import DataView


def _snapshot(revision: int = 0) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns({"point": [0.0, 1.0]}),
        data_axes=(Axis.create("scan", values=[0.0, 1.0]),),
        dtype=np.float64,
        generation="histogram-pool",
    )
    values = np.arange(8, dtype=np.float64).reshape(schema.shape)
    return DatasetSnapshot(schema, values, revision=revision)


def test_histogram_pools_the_whole_box() -> None:
    """Repeat x points x data axes all land in the pool: 8 values, 8 counts."""

    histogram = DataView(_snapshot()).histogram(bins=4)
    assert int(np.asarray(histogram.counts).sum()) == 8


def test_histogram_spec_needs_no_axis_declaration() -> None:
    """No axis takes a ROLE here -- but every axis can still be pinned.

    A histogram pools whatever box it is given; which box that is remains the
    operator's to narrow, so the scope rows are present and the role rows are
    not.
    """

    spec = HistogramPlot()
    session = PlotSession(_snapshot(), spec)
    try:
        names = tuple(field.name for field in session.describe_semantics().fields)
        assert tuple(name for name in names if not name.startswith("scope:")) == (
            "kind",
        )
        assert all(name.startswith("scope:") for name in names[1:])
    finally:
        session.close()


def test_a_window_pools_past_shots_and_shrinking_it_is_not_destructive() -> None:
    """The window looks back over the SESSION's shots, not over one publication.

    Eight publications of four values each arrive one at a time.  A window of
    three pools the last three of them -- twelve values -- and narrowing it to
    one and widening it back finds those shots still there, because the
    retained history is bounded by memory policy and never by what is shown.
    """

    session = PlotSession(_snapshot(), HistogramPlot())
    try:
        assert int(np.asarray(session._payload.counts).sum()) == 8

        session.set_parameter("window", 3)
        for revision in range(1, 8):
            session.update_data(_snapshot(revision=revision))
        payload = session._payload
        assert int(np.asarray(payload.counts).sum()) == 3 * 8
        assert len(payload.source_revisions) == 3

        session.set_parameter("window", 1)
        assert int(np.asarray(session._payload.counts).sum()) == 8
        session.set_parameter("window", 3)
        assert int(np.asarray(session._payload.counts).sum()) == 3 * 8
    finally:
        session.close()


def test_representation_toggles_refit_the_count_axis() -> None:
    """Density/cumulative/bin edits re-fit the axis; they are not jitter.

    The expand/shrink hysteresis exists for shot-to-shot noise on live
    data.  A representation change alters what one count MEANS, so the
    axis snaps to the new scale — and a density peak far below one count
    must fill the axis instead of being pinned under a counts floor.
    """

    rng = np.random.default_rng(3)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=300),
        PointTable.from_columns({"x": np.arange(1.0)}),
        dtype=np.float64,
        generation="histogram-refit",
    )
    snapshot = DatasetSnapshot(schema, rng.normal(50, 8, (300, 1)), revision=0)
    session = PlotSession(snapshot, HistogramPlot())
    try:
        axes = session._renderer.primary_axes

        def ceiling() -> float:
            return float(axes.get_ylim()[1])

        counts_ceiling = ceiling()
        assert counts_ceiling > 10.0

        session.set_parameter("density", True)
        density_ceiling = ceiling()
        assert density_ceiling < 1.0  # fills the axis, no counts floor
        assert density_ceiling > 0.0

        session.set_parameter("density", False)
        assert ceiling() == counts_ceiling

        session.set_parameter("cumulative", True)
        assert abs(ceiling() - 300 * 1.08) < 1e-6

        session.set_parameter("cumulative", False)
        assert ceiling() == counts_ceiling

        session.set_parameter("bin_count", 15)
        assert ceiling() > counts_ceiling  # fewer bins, taller peaks, refit
    finally:
        session.close()
