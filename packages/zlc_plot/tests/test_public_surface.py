from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    PlotSession,
    RollingPlot,
)
from zlc_plot._fit_projection import FitSelection
from zlc_plot.fit import FacetFitBatchResult


def _snapshot(*, revision: int = 0, repeats: int = 1) -> DatasetSnapshot:
    x = np.arange(6, dtype=np.float64)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({"x": x, "facet": np.repeat([0.0, 1.0], 3)}),
        dtype=np.float64,
        generation="public-surface",
    )
    values = np.tile(x, (repeats, 1))
    return DatasetSnapshot(schema, values, revision=revision)


def test_session_replace_spec_reuses_the_existing_surface() -> None:
    session = PlotSession(_snapshot(), CurvePlot(AxisRef.point("x")))
    figure = session._renderer.figure
    try:
        session.replace_spec(HistogramPlot(), parameters={"bin_count": 12})
        assert session.spec == HistogramPlot()
        assert session._renderer.figure is figure
        assert session.display_state["bin_count"] == 12
    finally:
        session.close()


def test_session_fit_selection_and_painted_payload_keep_ordered_source_revisions() -> None:
    session = PlotSession(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        selected = session.fit_selection("gaussian_offset")
        assert isinstance(selected, FitSelection)
        assert selected.source_revisions == (0,)
        assert session._payload.source_revisions == (0,)
    finally:
        session.close()

    rolling = PlotSession(_snapshot(repeats=2), RollingPlot())
    try:
        rolling.update_data(_snapshot(revision=1, repeats=2))
        # Revision 0 seeds one sample per repeat; revision 1 appends one.
        assert rolling._payload.source_revisions == (0, 0, 1)
        selected = rolling.fit_selection("gaussian_offset")
        assert selected.source_revisions == (0, 0, 1)
    finally:
        rolling.close()


def test_session_rolling_history_seeds_per_repeat_then_grows_one_sample_per_revision() -> None:
    rolling = PlotSession(_snapshot(repeats=3), RollingPlot())
    try:
        # A static snapshot is a complete shot record: the repeat axis seeds
        # the history so the initial render already shows every shot.
        payload = rolling._payload
        assert payload.source_revisions == (0, 0, 0)
        assert len(payload.series) == 1
        assert payload.series[0].x.canonical.size == 3

        # Later revisions keep the live contract: one sample per revision.
        rolling.update_data(_snapshot(revision=1, repeats=3))
        payload = rolling._payload
        assert payload.source_revisions == (0, 0, 0, 1)
        assert payload.series[0].x.canonical.size == 4
    finally:
        rolling.close()


def test_session_fit_all_facets_returns_one_result_per_painted_cell() -> None:
    spec = FacetGridPlot(
        AxisRef.point("facet"),
        CurvePlot(AxisRef.point("x")),
    )
    session = PlotSession(_snapshot(), spec)
    try:
        result = session.fit("gaussian_offset", fit_all_facets=True, live=False)
        assert isinstance(result, FacetFitBatchResult)
        assert len(result.results) == 2
        assert result.source_revision == 0
        assert result.facet == AxisRef.point("facet")
    finally:
        session.close()
