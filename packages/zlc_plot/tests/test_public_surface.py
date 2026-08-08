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


def test_image_site_overlay_is_plain_roundtrippable_display_state() -> None:
    snapshot = _image_snapshot()
    spec = ImagePlot(AxisRef.data("column"), AxisRef.data("row"))
    session = PlotSession(snapshot, spec)
    restored: PlotSession | None = None
    curve = PlotSession(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        described = session.describe_display()
        declaration = described.parameter_schema["site_overlay"]

        assert declaration.value_type is str
        assert declaration.default == "off"
        assert declaration.choices == ("off", "centers", "occupancy")
        assert described.display_state["site_overlay"] == "off"

        updated = session.set_parameter("site_overlay", "occupancy")
        assert updated["site_overlay"] == "occupancy"
        assert session.describe_display().display_state["site_overlay"] == "occupancy"

        # DisplayState values are the plain state read surface; the ordinary
        # constructor parameter mapping is the matching restore surface.
        restored = PlotSession(snapshot, spec, parameters=dict(updated.values))
        assert restored.describe_display().display_state["site_overlay"] == "occupancy"

        curve_description = curve.describe_display()
        assert "site_overlay" not in curve_description.parameter_schema
        assert "site_overlay" not in curve_description.display_state.values
        with pytest.raises(KeyError, match="site_overlay"):
            curve.set_parameter("site_overlay", "centers")
        with pytest.raises(ValueError, match="site_overlay"):
            session.set_parameter("site_overlay", "sites")
    finally:
        curve.close()
        if restored is not None:
            restored.close()
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
