"""A rolling panel's shot axis has ONE owner, and its chrome stands still.

The window frames the shot axis; the series painter frames ordinary curves
from their data.  When both wrote it, the axis moved twice per revision --
to the arrived shots and back to the window -- and each move marked the
chrome dirty, so a panel whose picture was completely static rebuilt and
re-captured its whole background on every single frame.
"""
from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable

from zlc_plot import AxisRef, PlotSession, RollingPlot
from zlc_plot.rendering import MatplotlibRenderer


def _session(window: int = 100) -> PlotSession:
    rng = np.random.default_rng(3)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=20),
        PointTable.from_columns({"site": np.arange(8.0)}),
        generation="rolling-owner",
    )
    session = PlotSession(
        DatasetSnapshot(schema, rng.normal(size=(20, 8)), revision=1),
        RollingPlot(),
        device_pixel_ratio=2.0,
    )
    session.set_parameters({"window": window})
    return session


#: The same numbers every revision.  These tests are about chrome that does
#: not change; new random values would move the y limits legitimately, and a
#: moving axis is a chrome change nobody should try to cache away.
_VALUES = np.random.default_rng(4).normal(size=(20, 8))


def _feed(session: PlotSession, revisions: int, *, start: list[int]) -> None:
    schema = session._projection.data.block.schema
    for _ in range(revisions):
        start[0] += 1
        session.update_data(
            DatasetSnapshot(schema, _VALUES, revision=start[0])
        )
        session.rgba()


def test_the_shot_axis_is_written_once_per_revision(monkeypatch) -> None:
    """One owner means one write; two owners meant two, and back again."""

    session = _session()
    try:
        session.rgba()
        clock = [1]
        _feed(session, 3, start=clock)
        writes = []
        original = MatplotlibRenderer._set_xlim

        def spy(self, axis, low, high):
            before = tuple(float(v) for v in axis.get_xlim())
            original(self, axis, low, high)
            after = tuple(float(v) for v in axis.get_xlim())
            if before != after:
                writes.append((str(axis.get_gid()), before, after))

        monkeypatch.setattr(MatplotlibRenderer, "_set_xlim", spy)
        _feed(session, 3, start=clock)
        assert writes == [], f"the shot axis moved: {writes}"
    finally:
        session.close()


def test_a_static_rolling_panel_reuses_its_chrome_background(monkeypatch) -> None:
    """The picture is not changing, so the background must not be rebuilt.

    A rebuild draws the whole scene with the dynamics hidden, copies the
    entire figure, restores it and paints the dynamics again -- worth it only
    when the copy survives to the next frame.
    """

    session = _session()
    try:
        session.rgba()
        clock = [1]
        _feed(session, 4, start=clock)
        draws = []
        original = MatplotlibRenderer._native_draw
        monkeypatch.setattr(
            MatplotlibRenderer,
            "_native_draw",
            staticmethod(lambda canvas: (draws.append(1), original(canvas))[1]),
        )
        _feed(session, 6, start=clock)
        assert draws == [], f"{len(draws)} full rebuilds for a static panel"
    finally:
        session.close()


def test_churn_counts_invalidation_not_a_missing_background() -> None:
    """The escape hatch must not be a one-way door.

    Dropping the background guarantees the next frame finds none; if that
    counted as churn, a panel that stopped capturing could never start again.
    """

    session = _session()
    try:
        session.rgba()
        clock = [1]
        _feed(session, 3, start=clock)
        renderer = session._renderer
        assert renderer._chrome_churn == 0
        renderer._background_region = None
        _feed(session, 1, start=clock)
        assert renderer._chrome_churn == 0
        assert renderer._background_region is not None
    finally:
        session.close()
