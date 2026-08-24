"""The cumulative rolling trace: a rate converging shot by shot.

Each live revision appends one shot; cumulative replaces the per-shot
trace with the running mean of everything so far and carries the running
standard error as the band.  The display window never changes the numbers.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib.collections import PolyCollection

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, PlotSession, RollingPlot
from zlc_plot.specs import Reduction

import pytest


def _schema(sites: int) -> DatasetSchema:
    return DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"site": np.arange(sites, dtype=np.int64)}),
        data_axes=(),
        dtype=np.float64,
        generation="rolling-cumulative",
    )


def _shot(schema: DatasetSchema, occupied: np.ndarray, revision: int) -> DatasetSnapshot:
    return DatasetSnapshot(
        schema, occupied.reshape(1, -1).astype(np.float64), revision=revision
    )


def test_cumulative_trace_is_the_running_rate_with_binomial_sem() -> None:
    """Feed 0/1 occupancy shot by shot; the trace must equal the running
    mean over ALL samples pooled so far and its sem the sample standard
    error -- which for booleans IS the binomial error."""

    sites = 8
    schema = _schema(sites)
    rng = np.random.default_rng(21)
    shots = (rng.random((30, sites)) < 0.4).astype(np.float64)
    session = PlotSession(
        _shot(schema, shots[0], 0),
        RollingPlot(reduction=Reduction.MEAN, cumulative=True),
    )
    try:
        for revision in range(1, len(shots)):
            session.update_data(_shot(schema, shots[revision], revision))
        payload = session._projection._payload
        series = payload.series[0]
        pooled = shots.reshape(-1)  # every sample of every shot, in order
        k = np.arange(1, len(shots) + 1) * sites
        expected_mean = np.cumsum(pooled)[k - 1] / k
        y = np.asarray(series.y.canonical)
        np.testing.assert_allclose(y, expected_mean, rtol=1e-12)
        # sem at the last point == sample standard error over all samples
        final = pooled
        expected_sem = float(np.std(final, ddof=1) / np.sqrt(final.size))
        np.testing.assert_allclose(series.sem[-1], expected_sem, rtol=1e-12)
        # and it shrinks as shots accumulate
        assert series.sem[-1] < series.sem[1]
    finally:
        session.close()


def test_window_frames_the_view_without_changing_the_numbers() -> None:
    sites = 4
    schema = _schema(sites)
    rng = np.random.default_rng(5)
    shots = (rng.random((20, sites)) < 0.5).astype(np.float64)

    def last_value(window: int | None) -> tuple[float, float]:
        kwargs = {} if window is None else {"parameters": {"window": window}}
        session = PlotSession(
            _shot(schema, shots[0], 0),
            RollingPlot(reduction=Reduction.MEAN, cumulative=True),
            **kwargs,
        )
        try:
            for revision in range(1, len(shots)):
                session.update_data(_shot(schema, shots[revision], revision))
            series = session._projection._payload.series[0]
            return float(series.y.canonical[-1]), float(series.sem[-1])
        finally:
            session.close()

    assert last_value(5) == last_value(None)


def test_cumulative_band_renders(tmp_path) -> None:
    sites = 6
    schema = _schema(sites)
    rng = np.random.default_rng(3)
    session = PlotSession(
        _shot(schema, (rng.random(sites) < 0.5).astype(np.float64), 0),
        RollingPlot(reduction=Reduction.MEAN, cumulative=True),
    )
    try:
        for revision in range(1, 12):
            session.update_data(
                _shot(schema, (rng.random(sites) < 0.5).astype(np.float64), revision)
            )
        session._renderer.draw()
        bands = [
            artist
            for axes in session._renderer.figure.axes
            for artist in axes.collections
            if isinstance(artist, PolyCollection)
        ]
        assert bands, "cumulative rolling must draw its running-sem band"
    finally:
        session.close()


def test_cumulative_refuses_non_mean() -> None:
    with pytest.raises(ValueError, match="MEAN"):
        RollingPlot(reduction=Reduction.MEDIAN, cumulative=True)
