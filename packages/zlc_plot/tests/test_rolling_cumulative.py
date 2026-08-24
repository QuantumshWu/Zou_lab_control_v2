"""The cumulative rolling trace: a rate converging shot by shot.

Each live revision appends one shot; cumulative replaces the per-shot
trace with the running mean of everything so far and carries the running
standard error as the band.  The display window never changes the numbers.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib.collections import LineCollection

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
        RollingPlot(reduction=Reduction.MEAN),
        parameters={"cumulative": True, "uncertainty": True},
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
        values = {"cumulative": True, "uncertainty": True}
        if window is not None:
            values["window"] = window
        session = PlotSession(
            _shot(schema, shots[0], 0),
            RollingPlot(reduction=Reduction.MEAN),
            parameters=values,
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
        RollingPlot(reduction=Reduction.MEAN),
        parameters={"cumulative": True, "uncertainty": True},
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
            if isinstance(artist, LineCollection)
        ]
        assert bands, "cumulative rolling must draw its running-sem bars"
    finally:
        session.close()


def test_cumulative_is_inert_on_a_non_mean_reduction() -> None:
    """The switch is a display parameter; on a MIN trace the running
    standard error does not exist, so flipping it changes nothing."""

    sites = 4
    schema = _schema(sites)
    rng = np.random.default_rng(11)
    session = PlotSession(
        _shot(schema, (rng.random(sites) < 0.5).astype(np.float64), 0),
        RollingPlot(reduction=Reduction.MIN),
        parameters={"cumulative": True, "uncertainty": True},
    )
    try:
        for revision in range(1, 5):
            session.update_data(
                _shot(schema, (rng.random(sites) < 0.5).astype(np.float64), revision)
            )
        series = session._projection._payload.series[0]
        assert series.sem is None
    finally:
        session.close()


def test_plain_rolling_uncertainty_is_each_shot_pooled_error() -> None:
    """The survival-panel shape: uncertainty WITHOUT cumulative draws each
    shot's own pooled standard error."""

    sites = 30
    schema = _schema(sites)
    rng = np.random.default_rng(9)
    shots = (rng.random((6, sites)) < 0.5).astype(np.float64)
    session = PlotSession(
        _shot(schema, shots[0], 0),
        RollingPlot(reduction=Reduction.MEAN),
        parameters={"uncertainty": True},
    )
    try:
        for revision in range(1, len(shots)):
            session.update_data(_shot(schema, shots[revision], revision))
        series = session._projection._payload.series[0]
        assert series.sem is not None
        for index in range(len(shots)):
            expected = float(
                np.std(shots[index], ddof=1) / np.sqrt(sites)
            )
            np.testing.assert_allclose(series.sem[index], expected, rtol=1e-12)
        session._renderer.draw()
        bands = [
            artist
            for axes in session._renderer.figure.axes
            for artist in axes.collections
            if isinstance(artist, LineCollection)
        ]
        assert bands, "plain rolling with uncertainty must draw the bars"
    finally:
        session.close()


def test_structure_keeps_repeat_point_and_cell_brackets() -> None:
    """Three brackets: (repeat) x (points) x (data).

    Pair and site are both dimensions of one atomic cell payload, so they
    share the third bracket.  A READOUT_EVENT cell axis is instead a fact
    about WHEN within one point, so it joins the points bracket after the
    scan dimensions: (20) x (10x10x10x3) x (34).
    """

    from zlc_data import (
        COMPONENT,
        GridTopology,
        READOUT_EVENT,
        SITE,
        AxisId,
        AxisSpec,
        DatasetSchema as Schema,
        PointTable,
        REPEAT,
        SPATIAL_X,
        SPATIAL_Y,
        ValidityContract,
        ValueSchema,
    )
    from zlc_plot.semantics import schema_structure

    def _schema_for(axes):
        return Schema(
            AxisSpec(AxisId("cycle"), "cycle", REPEAT, 1),
            PointTable(1, ()),
            None,
            ValueSchema(
                axes,
                ValidityContract.components(axes[0].axis_id),
                np.dtype("<f8"),
                "1",
            ),
        )

    categorical = _schema_for(
        (
            AxisSpec(AxisId("fs.pair"), "pair", COMPONENT, 3),
            AxisSpec(AxisId("occ.site"), "site", SITE, 33),
        )
    )
    groups = schema_structure(categorical)
    assert tuple(tuple(name for name, _size in group) for group in groups) == (
        ("cycle",),
        ("pair", "site"),
    )

    scanned = Schema(
        AxisSpec(AxisId("cycle"), "cycle", REPEAT, 20),
        PointTable(8, ()),
        GridTopology(
            (AxisId("ax"), AxisId("ay"), AxisId("az")),
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            tuple((i % 2, (i // 2) % 2, i // 4) for i in range(8)),
        ),
        ValueSchema(
            (
                AxisSpec(AxisId("cm.frame"), "frame", READOUT_EVENT, 3),
                AxisSpec(AxisId("occ.site"), "site", SITE, 34),
            ),
            ValidityContract.components(AxisId("occ.site")),
            np.dtype("<f8"),
            "1",
        ),
    )
    groups = schema_structure(scanned)
    assert tuple(tuple(name for name, _size in group) for group in groups) == (
        ("cycle",),
        ("ax", "ay", "az", "frame"),
        ("site",),
    )

    picture = _schema_for(
        (
            AxisSpec(AxisId("cam.y"), "y", SPATIAL_Y, 4),
            AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 5),
        )
    )
    groups = schema_structure(picture)
    assert tuple(tuple(name for name, _size in group) for group in groups) == (
        ("cycle",),
        ("y", "x"),
    )


def test_labelled_axis_ticks_by_name() -> None:
    """A pair/model axis ticks by its declared names -- the same names the
    legend, hover and scope rows use -- never by bare indices."""

    from zlc_data import (
        COMPONENT,
        SITE,
        AxisId,
        AxisSpec,
        DatasetSchema as Schema,
        PointTable,
        REPEAT,
        ValidityContract,
        ValueSchema,
        owned_snapshot_from_arrays,
    )
    from zlc_plot import CurvePlot

    pair = AxisSpec(
        AxisId("fs.pair"), "pair", COMPONENT, 3,
        coordinate_labels=("0-1", "0-2", "1-2"),
    )
    site = AxisSpec(AxisId("occ.site"), "site", SITE, 5)
    schema = Schema(
        AxisSpec(AxisId("cycle"), "cycle", REPEAT, 8),
        PointTable(1, ()),
        None,
        ValueSchema(
            (pair, site),
            ValidityContract.components(pair.axis_id, site.axis_id),
            np.dtype("<f8"),
            "1",
        ),
    )
    rng = np.random.default_rng(0)
    snapshot = owned_snapshot_from_arrays(
        schema, (rng.random((8, 1, 3, 5)) < 0.5).astype("<f8"), 0
    )
    session = PlotSession(
        snapshot,
        CurvePlot(AxisRef.data("fs.pair")),
        parameters={"uncertainty": True},
    )
    try:
        session._renderer.draw()
        axes = session._renderer.figure.axes[0]
        labels = [tick.get_text() for tick in axes.get_xticklabels()]
        assert labels == ["0-1", "0-2", "1-2"]
        series = session._projection._payload.series[0]
        assert series.x_labels == ("0-1", "0-2", "1-2")
        assert series.sem is not None
    finally:
        session.close()
