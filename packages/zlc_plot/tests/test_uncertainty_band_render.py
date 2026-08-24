"""The uncertainty band renders, follows live shrinkage, and stays absent
when not requested."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib.collections import LineCollection

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, PlotLabels, PlotSession


def _schema(repeats: int) -> DatasetSchema:
    return DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0, 3.0]}),
        data_axes=(),
        dtype=np.float64,
        generation="uncertainty-band",
    )


def _snapshot(repeats: int, revision: int, scale: float) -> DatasetSnapshot:
    rng = np.random.default_rng(revision)
    values = 0.5 + scale * rng.standard_normal((repeats, 4))
    return DatasetSnapshot(_schema(repeats), values, revision=revision)


def _bands(session: PlotSession) -> list[PolyCollection]:
    return [
        artist
        for axes in session._renderer.figure.axes
        for artist in axes.collections
        if isinstance(artist, LineCollection)
    ]


def test_uncertainty_curve_draws_a_band_and_covers_it_in_ylim() -> None:
    session = PlotSession(
        _snapshot(24, 1, 0.2),
        CurvePlot(AxisRef.point("x"), labels=PlotLabels("band", "x", "y")),
        parameters={"uncertainty": True},
    )
    try:
        session._renderer.draw()
        bands = _bands(session)
        assert bands, "uncertainty=True must draw error bars"
        axes = session._renderer.figure.axes[0]
        payload = session._projection._payload
        series = payload.series[0]
        low = float(np.nanmin(np.asarray(series.y.canonical) - series.sem))
        high = float(np.nanmax(np.asarray(series.y.canonical) + series.sem))
        y0, y1 = axes.get_ylim()
        assert y0 <= low and y1 >= high
    finally:
        session.close()


def test_band_shrinks_as_shots_accumulate() -> None:
    """More repeats, tighter sem: the live convergence the band exists for."""

    def band_height(repeats: int) -> float:
        session = PlotSession(
            _snapshot(repeats, 1, 0.2),
            CurvePlot(AxisRef.point("x")),
            parameters={"uncertainty": True},
        )
        try:
            session._renderer.draw()
            series = session._projection._payload.series[0]
            return float(np.nanmean(series.sem))
        finally:
            session.close()

    assert band_height(160) < band_height(10) / 2.5


def test_no_band_without_the_request() -> None:
    session = PlotSession(
        _snapshot(24, 1, 0.2),
        CurvePlot(AxisRef.point("x")),
    )
    try:
        session._renderer.draw()
        assert not _bands(session)
        assert session._projection._payload.series[0].sem is None
    finally:
        session.close()


def test_focus_dims_the_other_series_bars_with_their_lines() -> None:
    """The bars are part of the series: a locked focus dims the other
    series' bars to near-nothing and restores them on release."""

    from zlc_plot.config import DEFAULTS

    rng = np.random.default_rng(3)
    values = 0.5 + 0.2 * rng.standard_normal((24, 4, 2))
    from data_factory import Axis as FAxis, DatasetSchema as FSchema, PointTable as FTable
    schema = FSchema.create(
        FAxis.create("repeat", size=24),
        FTable.from_columns({"x": [0.0, 1.0, 2.0, 3.0]}),
        data_axes=(FAxis.create("site", values=[0.0, 1.0]),),
        dtype=np.float64,
        generation="focus-bars",
    )
    snapshot = DatasetSnapshot(schema, values, revision=1)
    session = PlotSession(
        snapshot,
        CurvePlot(AxisRef.point("x"), group=AxisRef.data("site")),
        parameters={"uncertainty": True},
    )
    try:
        renderer = session._renderer
        renderer.draw()
        axes = renderer.figure.axes[0]
        bars = renderer._series_bars[id(axes)]
        assert len(bars) == 2  # one group per series
        token = DEFAULTS.style.render.uncertainty_bar_alpha
        for artists in bars.values():
            assert all(a.get_alpha() == token for a in artists)

        entries = renderer._series_lines[id(axes)]
        first_line, first_identity, _ = entries[0]
        x = np.asarray(first_line.get_xdata(), dtype=float)
        y = np.asarray(first_line.get_ydata(), dtype=float)
        renderer._series_locked = (
            id(axes), first_identity, "s", float(x[0]), float(y[0])
        )
        renderer._apply_series_focus()
        for identity, artists in bars.items():
            expected = token if identity == first_identity else 0.06
            assert all(a.get_alpha() == expected for a in artists), identity

        renderer._series_locked = None
        renderer._apply_series_focus()
        for artists in bars.values():
            assert all(a.get_alpha() == token for a in artists)
    finally:
        session.close()
