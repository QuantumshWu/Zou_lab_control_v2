from __future__ import annotations

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSession,
    RollingPlot,
)
from zlc_plot._fit_projection import FitSelection
from zlc_plot.fit import FacetFitBatchResult
from zlc_plot.primitives import ImageFrame, ImagePointOverlay, PointStatus


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


def _image_snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"sample": np.array([0.0])}),
        data_axes=(
            Axis.create("row", size=2),
            Axis.create("column", size=3),
        ),
        dtype=np.float64,
        generation="public-surface-image",
    )
    values = np.arange(6, dtype=np.float64).reshape(1, 1, 2, 3)
    return DatasetSnapshot(schema, values, revision=0)


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


def test_image_overlay_is_explicit_data_not_a_display_mode() -> None:
    snapshot = _image_snapshot()
    spec = ImagePlot(AxisRef.data("column"), AxisRef.data("row"))
    session = PlotSession(snapshot, spec)
    curve = PlotSession(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        described = session.describe_display()
        assert "site_overlay" not in described.parameter_schema
        assert "site_overlay" not in described.display_state.values
        assert described.display_state["show_point_labels"] is True

        curve_description = curve.describe_display()
        assert "site_overlay" not in curve_description.parameter_schema
        assert "site_overlay" not in curve_description.display_state.values
        with pytest.raises(KeyError, match="site_overlay"):
            curve.set_parameter("site_overlay", "centers")
    finally:
        curve.close()
        session.close()


def test_image_site_numbers_use_their_ring_status_style() -> None:
    """A small ordinal must remain visually attached to its status ring."""

    from matplotlib.colors import to_rgba

    snapshot = _image_snapshot()
    overlay = ImagePointOverlay(
        1,
        np.asarray(((0.5, 0.5), (1.5, 0.5), (2.5, 1.5))),
        point_ids=("trap-a", "trap-b", "trap-c"),
        labels=("1", "2", "3"),
        statuses=(PointStatus.EMPTY, PointStatus.OCCUPIED, PointStatus.INVALID),
    )
    session = PlotSession(
        ImageFrame(snapshot, overlay),
        ImagePlot(AxisRef.data("column"), AxisRef.data("row")),
        parameters={"show_point_labels": True},
    )
    try:
        artists = session._renderer._artists["image:point-labels"]
        tokens = (
            session._renderer.style.artists.point_empty,
            session._renderer.style.artists.point_occupied,
            session._renderer.style.artists.point_invalid,
        )
        assert tuple(label.get_text() for label in artists) == ("1", "2", "3")
        assert all(
            to_rgba(label.get_color(), label.get_alpha())
            == to_rgba(token.color, token.alpha)
            for label, token in zip(artists, tokens, strict=True)
        )
        assert all(
            label.get_fontsize() == session._renderer.style.fonts.fit_annotation_pt
            for label in artists
        )
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
