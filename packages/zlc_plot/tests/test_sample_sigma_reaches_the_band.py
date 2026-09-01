"""A sample that knows its own error draws a band with one of itself.

The shape this is about: an image panel fits each shot and publishes the
fitted amplitude.  A curve or rolling panel plotting that amplitude pools
exactly ONE sample per point, and a scatter over one sample is NaN -- so
the band was empty everywhere, while a perfectly good error sat on the
sample and reached nothing.  Both ends of the sigma plane were declared;
no reduction ever read it.

The rule these pin, in both directions:

  * where the scatter cannot speak (a bucket of one), the sample's own
    sigma is the band;
  * where it can (two or more), the scatter IS the band and the sigma
    changes nothing -- a scatter already contains the measurement error,
    and adding it again would count it twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, PlotSession, RollingPlot
from zlc_plot.data_view import DataView
from zlc_plot.specs import Reduction

POINTS = 6
AMPLITUDES = np.array([3.0, 3.4, 3.9, 4.1, 3.7, 3.2])
ERRORS = np.array([0.11, 0.09, 0.14, 0.08, 0.12, 0.10])


def _schema(repeats: int = 1) -> DatasetSchema:
    return DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({"x": np.arange(POINTS, dtype=np.int64)}),
        data_axes=(),
        dtype=np.float64,
        generation="sample-sigma",
    )


def _one_fit_per_point(*, with_sigma: bool) -> DatasetSnapshot:
    """One fitted amplitude per x, each carrying its own error."""

    schema = _schema()
    return DatasetSnapshot(
        schema,
        AMPLITUDES.reshape(1, POINTS),
        revision=0,
        sigma=ERRORS.reshape(1, POINTS) if with_sigma else None,
    )


def _curve_sem(snapshot: DatasetSnapshot) -> np.ndarray:
    session = PlotSession(
        snapshot,
        CurvePlot(AxisRef.point("x")),
        parameters={"uncertainty": True},
    )
    try:
        session._renderer.draw()
        series = session._projection._payload.series[0]
        return np.asarray(series.sem, dtype=float)
    finally:
        session.close()


def test_one_fit_per_point_draws_its_own_error() -> None:
    """The band over a bucket of one is that sample's stated error."""

    sem = _curve_sem(_one_fit_per_point(with_sigma=True))
    assert np.allclose(sem, ERRORS, rtol=0, atol=0)


def test_without_a_stated_error_a_bucket_of_one_still_has_no_band() -> None:
    """The sigma is what produces it -- not some default that hides a NaN.

    Guards the test above against passing for the wrong reason: if a
    single-sample bucket had grown a band from somewhere else, the two
    tests could not both hold.
    """

    sem = _curve_sem(_one_fit_per_point(with_sigma=False))
    assert np.all(np.isnan(sem))


def test_a_scatter_of_several_ignores_the_stated_errors() -> None:
    """Two or more samples: the band is the scatter, sigma or no sigma.

    A scatter already contains the measurement error.  Adding the stated
    sigma on top -- as a sum, or as the larger of the two -- would count it
    twice and bias the bar high, so a plot of the same values must not move
    when the producer starts stating errors.
    """

    repeats = 5
    rng = np.random.default_rng(20260828)
    values = AMPLITUDES[None, :] + rng.normal(0.0, 0.3, size=(repeats, POINTS))
    schema = _schema(repeats)
    bare = DatasetSnapshot(schema, values, revision=0)
    stated = DatasetSnapshot(
        schema,
        values,
        revision=0,
        # Deliberately huge: ten times the scatter.  If the sigma leaked
        # into a bucket that has a scatter, this could not stay equal.
        sigma=np.full((repeats, POINTS), 3.0),
    )
    expected = values.std(axis=0, ddof=1) / np.sqrt(repeats)
    assert np.allclose(_curve_sem(bare), expected, rtol=1e-12)
    assert np.allclose(_curve_sem(stated), expected, rtol=1e-12)


def test_a_rolling_shot_of_one_fit_carries_its_error() -> None:
    """One fit per shot: each drawn point's band is that fit's error."""

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": np.arange(1, dtype=np.int64)}),
        data_axes=(),
        dtype=np.float64,
        generation="sample-sigma-rolling",
    )
    for amplitude, error in zip(AMPLITUDES, ERRORS):
        snapshot = DatasetSnapshot(
            schema,
            np.asarray([[amplitude]]),
            revision=0,
            sigma=np.asarray([[error]]),
        )
        history = DataView(snapshot).rolling_history(
            aggregation=Reduction.MEAN, uncertainty=True
        )
        assert len(history) == 1
        assert history[0].sem is not None
        assert float(history[0].sem[0]) == pytest.approx(error, rel=0, abs=0)


def test_a_grouped_rolling_shot_carries_the_error_of_each_group() -> None:
    """Per-site rolling: one sample per site per shot, each with its own."""

    sites = 4
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"site": np.arange(sites, dtype=np.int64)}),
        data_axes=(),
        dtype=np.float64,
        generation="sample-sigma-sites",
    )
    values = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    errors = np.asarray([[0.10, 0.20, 0.30, 0.40]])
    view = DataView(DatasetSnapshot(schema, values, revision=0, sigma=errors))
    history = view.rolling_history(
        group=AxisRef.point("site"),
        aggregation=Reduction.MEAN,
        uncertainty=True,
    )
    assert len(history) == 1
    assert np.allclose(np.asarray(history[0].sem, dtype=float), errors[0])


def test_the_whole_revision_pooled_uses_its_one_samples_error() -> None:
    """The ungrouped whole-revision reduction takes the same route."""

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": np.arange(1, dtype=np.int64)}),
        data_axes=(),
        dtype=np.float64,
        generation="sample-sigma-pooled",
    )
    view = DataView(
        DatasetSnapshot(
            schema,
            np.asarray([[7.5]]),
            revision=0,
            sigma=np.asarray([[0.25]]),
        )
    )
    sample = view.rolling_history(aggregation=Reduction.MEAN)[0]
    assert float(sample.sem[0]) == pytest.approx(0.25, rel=0, abs=0)
