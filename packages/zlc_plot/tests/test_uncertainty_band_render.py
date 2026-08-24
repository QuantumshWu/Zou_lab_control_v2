"""The uncertainty band renders, follows live shrinkage, and stays absent
when not requested."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib.collections import PolyCollection

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
        if isinstance(artist, PolyCollection)
    ]


def test_uncertainty_curve_draws_a_band_and_covers_it_in_ylim() -> None:
    session = PlotSession(
        _snapshot(24, 1, 0.2),
        CurvePlot(
            AxisRef.point("x"),
            uncertainty=True,
            labels=PlotLabels("band", "x", "y"),
        ),
    )
    try:
        session._renderer.draw()
        bands = _bands(session)
        assert bands, "uncertainty=True must draw a band"
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
            CurvePlot(AxisRef.point("x"), uncertainty=True),
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
