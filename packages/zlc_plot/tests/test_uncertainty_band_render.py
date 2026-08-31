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


def test_no_band_when_it_is_switched_off() -> None:
    """The band is on by default, so absence is now something ASKED for.

    A mean drawn without its spread reads as an exact number, and the
    operator had to know the switch existed to learn otherwise -- so the
    switch defaults on and this pins the other half: switched off, nothing
    is drawn and the statistic is not even computed.
    """

    session = PlotSession(
        _snapshot(24, 1, 0.2),
        CurvePlot(AxisRef.point("x")),
        parameters={"uncertainty": False},
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


def test_error_bars_live_in_the_dynamic_layer_with_their_lines() -> None:
    """Focus changes must land in the SAME composed frame as the line's.

    The compose path repaints only dynamic artists over a cached chrome
    background; a bar left out of the dynamic set keeps its pre-focus
    pixels baked into that background, so the line dimmed instantly while
    the bars answered one full draw later.
    """

    session = PlotSession(
        _snapshot(24, 1, 0.2),
        CurvePlot(AxisRef.point("x"), labels=PlotLabels("band", "x", "y")),
        parameters={"uncertainty": True},
    )
    try:
        renderer = session._renderer
        renderer.draw()
        dynamic = {id(artist) for _key, artist in renderer._dynamic_artists()}
        bars = [
            artist
            for groups in renderer._series_bars.values()
            for artists in groups.values()
            for artist in artists
        ]
        assert bars, "the uncertainty panel must have bar artists"
        missing = [a for a in bars if id(a) not in dynamic]
        assert not missing, (
            f"{len(missing)} bar artists sit in the cached background"
        )
    finally:
        session.close()


def test_hover_hit_tests_reuse_the_transformed_polyline() -> None:
    """One transform per view, not one per motion event.

    Transforming every point of every series on every mouse move was the
    hover lag; the pixel polyline only changes with the data, the view or
    the canvas, so it is cached on that signature and dropped the moment a
    series mutates.
    """

    session = PlotSession(
        _snapshot(24, 1, 0.2),
        CurvePlot(AxisRef.point("x"), labels=PlotLabels("band", "x", "y")),
        parameters={"uncertainty": True},
    )
    try:
        renderer = session._renderer
        renderer.draw()
        axes = renderer.figure.axes[0]
        assert not renderer._series_hit_cache
        first = renderer._series_hit(axes, 100.0, 100.0, 12.0)
        assert renderer._series_hit_cache, "the first hit test fills the cache"
        cached_before = {
            key: id(entry[1]) for key, entry in renderer._series_hit_cache.items()
        }
        second = renderer._series_hit(axes, 101.0, 101.0, 12.0)
        cached_after = {
            key: id(entry[1]) for key, entry in renderer._series_hit_cache.items()
        }
        assert cached_before == cached_after, "a nearby motion recomputed"
        del first, second

        # New data invalidates: the cache is of the OLD polyline, and the
        # series mutation clears it before the new lines draw.
        session.update_data(_snapshot(24, 2, 0.2))
        assert not renderer._series_hit_cache, (
            "a series mutation must drop every cached hover polyline"
        )
    finally:
        session.close()


def test_a_revision_moves_the_bars_it_does_not_rebuild_them() -> None:
    """The bars a revision draws are last revision's artists, with new data.

    Rebuilding them cost the errorbar constructor, a masked-array copy,
    three transform trees and a colour conversion per series per frame --
    on a 64-cell grid drawing 640 points in total, 129 ms a revision --
    and it handed the focus walk fresh object identities, which is why the
    memo written to skip that walk could never match.

    Two things have to hold together: the artists must be the SAME objects
    (or nothing was saved), and their segments must follow the data (or
    the picture is last revision's).  A frame is not compared against a
    freshly opened session's, because a session that has seen history is
    not required to frame its axes the same way -- limit retention is
    deliberate -- so re-submitting one revision is what proves the drawing
    itself does not drift.
    """

    session = PlotSession(
        _snapshot(6, 1, 1.0),
        CurvePlot(AxisRef.point("x"), labels=PlotLabels("band", "x", "y")),
        parameters={"uncertainty": True},
    )
    try:
        session.rgba()
        bars = _bands(session)
        assert bars, "the case must draw bars for this test to mean anything"
        identities = [id(artist) for artist in bars]
        segments = [np.array(bars[0].get_segments(), copy=True)]
        for revision in range(2, 6):
            session.update_data(_snapshot(6, revision, 1.0 + 0.3 * revision))
            session.rgba()
            current = _bands(session)
            assert [id(artist) for artist in current] == identities
            segments.append(np.array(current[0].get_segments(), copy=True))
        moved = sum(
            1
            for before, after in zip(segments, segments[1:])
            if before.shape != after.shape or not np.array_equal(before, after)
        )
        assert moved == len(segments) - 1, "reused bars must carry the new data"

        # The same numbers twice, under the two revisions the session
        # requires: one drawing must be the other, bit for bit.
        settled = 0.5 + np.random.default_rng(99).standard_normal((6, 4))
        session.update_data(DatasetSnapshot(_schema(6), settled, revision=6))
        once = np.array(session.rgba(), copy=True)
        session.update_data(DatasetSnapshot(_schema(6), settled, revision=7))
        twice = np.array(session.rgba(), copy=True)
        assert np.array_equal(once, twice)
    finally:
        session.close()


def test_a_band_of_no_width_draws_no_bar() -> None:
    """Zero is not an uncertainty; it is a tick mark that lies.

    Every repeat identical means the spread really is zero, which is a
    different statement from "unknown" (a single repeat, which reports NaN
    and was already excluded).  Neither one gets a bar.
    """

    identical = np.tile(np.asarray([1.0, 2.0, 3.0, 4.0]), (6, 1))
    session = PlotSession(
        DatasetSnapshot(_schema(6), identical, revision=1),
        CurvePlot(AxisRef.point("x"), labels=PlotLabels("band", "x", "y")),
        parameters={"uncertainty": True},
    )
    try:
        session.rgba()
        assert _bands(session) == []
    finally:
        session.close()
