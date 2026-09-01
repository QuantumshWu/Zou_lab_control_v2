"""The trailing rolling trace: the rate over the recent past.

Each live revision appends one shot.  ``trailing`` says how many shots
one drawn point averages -- 1 is the shot itself, N is the mean of the
last N, with the standard error of those same N as the band.  The
display window still never changes the numbers.
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


def _schema(sites: int, repeats: int = 1) -> DatasetSchema:
    return DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({"site": np.arange(sites, dtype=np.int64)}),
        data_axes=(),
        dtype=np.float64,
        generation="rolling-trailing",
    )


def _shot(schema: DatasetSchema, occupied: np.ndarray, revision: int) -> DatasetSnapshot:
    return DatasetSnapshot(
        schema, occupied.reshape(1, -1).astype(np.float64), revision=revision
    )


def _shots(occupied: np.ndarray, revision: int = 0) -> DatasetSnapshot:
    values = np.asarray(occupied, dtype=np.float64)
    schema = _schema(values.shape[1], values.shape[0])
    return DatasetSnapshot(schema, values, revision=revision)


def _trailing_mean(shots: np.ndarray, span: int) -> np.ndarray:
    """The mean of every sample in the last ``span`` shots, shot by shot."""

    return np.asarray(
        [
            shots[max(0, index - span + 1) : index + 1].reshape(-1).mean()
            for index in range(len(shots))
        ]
    )


def test_a_trailing_point_is_the_mean_of_the_last_n_shots() -> None:
    """Feed 0/1 occupancy shot by shot; each drawn point must equal the
    mean over every sample the last N shots pooled, and its sem the sample
    standard error of those -- which for booleans IS the binomial error."""

    sites = 8
    span = 10
    rng = np.random.default_rng(21)
    shots = (rng.random((30, sites)) < 0.4).astype(np.float64)
    session = PlotSession(
        _shots(shots),
        RollingPlot(reduction=Reduction.MEAN),
        parameters={"trailing": span, "uncertainty": True},
    )
    try:
        series = session._projection._payload.series[0]
        y = np.asarray(series.y.canonical)
        np.testing.assert_allclose(y, _trailing_mean(shots, span), rtol=1e-12)
        window = shots[-span:].reshape(-1)
        expected_sem = float(np.std(window, ddof=1) / np.sqrt(window.size))
        np.testing.assert_allclose(series.sem[-1], expected_sem, rtol=1e-12)
    finally:
        session.close()


def test_a_span_longer_than_the_run_is_the_running_mean() -> None:
    """The window fills from empty, so a trailing mean nobody has enough
    shots for yet IS the mean of everything so far -- the trace settles
    into its window with no discontinuity."""

    sites = 4
    rng = np.random.default_rng(7)
    shots = (rng.random((9, sites)) < 0.5).astype(np.float64)
    session = PlotSession(
        _shots(shots),
        RollingPlot(reduction=Reduction.MEAN),
        parameters={"trailing": 1000, "uncertainty": True},
    )
    try:
        series = session._projection._payload.series[0]
        pooled = shots.reshape(-1)
        k = np.arange(1, len(shots) + 1) * sites
        running = np.cumsum(pooled)[k - 1] / k
        np.testing.assert_allclose(
            np.asarray(series.y.canonical), running, rtol=1e-12
        )
    finally:
        session.close()


def test_the_default_span_of_one_is_each_shot_itself() -> None:
    """The default must not quietly redraw anybody's panel: one shot per
    point, exactly the trace drawn before this parameter existed."""

    sites = 5
    rng = np.random.default_rng(31)
    shots = (rng.random((7, sites)) < 0.5).astype(np.float64)
    session = PlotSession(
        _shots(shots), RollingPlot(reduction=Reduction.MEAN)
    )
    try:
        series = session._projection._payload.series[0]
        np.testing.assert_allclose(
            np.asarray(series.y.canonical), shots.mean(axis=1), rtol=1e-12
        )
    finally:
        session.close()


def test_window_frames_the_view_without_changing_the_numbers() -> None:
    sites = 4
    rng = np.random.default_rng(5)
    shots = (rng.random((20, sites)) < 0.5).astype(np.float64)

    def last_value(window: int | None) -> tuple[float, float]:
        values = {"trailing": 6, "uncertainty": True}
        if window is not None:
            values["window"] = window
        session = PlotSession(
            _shots(shots),
            RollingPlot(reduction=Reduction.MEAN),
            parameters=values,
        )
        try:
            series = session._projection._payload.series[0]
            return float(series.y.canonical[-1]), float(series.sem[-1])
        finally:
            session.close()

    assert last_value(5) == last_value(None)


def test_the_trailing_band_renders(tmp_path) -> None:
    sites = 6
    rng = np.random.default_rng(3)
    shots = (rng.random((12, sites)) < 0.5).astype(np.float64)
    session = PlotSession(
        _shots(shots),
        RollingPlot(reduction=Reduction.MEAN),
        parameters={"trailing": 5, "uncertainty": True},
    )
    try:
        session._renderer.draw()
        # A native prepared scene rasters the bars without artists; the
        # curve band tests set the precedent -- materialize first, then
        # assert on the public artists it builds back.
        session._renderer._materialize_prepared_curve()
        bands = [
            artist
            for axes in session._renderer.figure.axes
            for artist in axes.collections
            if isinstance(artist, LineCollection)
        ]
        assert bands, "a trailing rolling trace must draw its sem bars"
    finally:
        session.close()


def test_trailing_is_inert_on_a_non_mean_reduction() -> None:
    """It reads the pooled moments, weighting each shot by how many
    samples it pooled.  That is the mean's arithmetic and nobody else's --
    a count-weighted mean of minima is a number about nothing -- so on a
    MIN trace the parameter does nothing at all."""

    sites = 4
    schema = _schema(sites)
    rng = np.random.default_rng(11)
    session = PlotSession(
        _shot(schema, (rng.random(sites) < 0.5).astype(np.float64), 0),
        RollingPlot(reduction=Reduction.MIN),
        parameters={"trailing": 4, "uncertainty": True},
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
    """The survival-panel shape: uncertainty WITHOUT trailing draws each
    shot's own pooled standard error."""

    sites = 30
    rng = np.random.default_rng(9)
    shots = (rng.random((6, sites)) < 0.5).astype(np.float64)
    session = PlotSession(
        _shots(shots),
        RollingPlot(reduction=Reduction.MEAN),
        parameters={"uncertainty": True},
    )
    try:
        series = session._projection._payload.series[0]
        assert series.sem is not None
        for index in range(len(shots)):
            expected = float(
                np.std(shots[index], ddof=1) / np.sqrt(sites)
            )
            np.testing.assert_allclose(series.sem[index], expected, rtol=1e-12)
        session._renderer.draw()
        # A native prepared scene rasters the bars without artists; the
        # curve band tests set the precedent -- materialize first, then
        # assert on the public artists it builds back.
        session._renderer._materialize_prepared_curve()
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
