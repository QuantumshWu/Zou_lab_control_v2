from __future__ import annotations

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import DEFAULTS, AxisRef, CurvePlot, HistogramPlot
from zlc_plot._fit_projection import FitProjection, FitScope, ProjectionContext
from zlc_plot.selectors import NumericRange, RectangleRange, SelectorKind, SelectorSnapshot, SelectorState
from zlc_plot.specs import parameter_schema_for
from zlc_plot.state import DisplayStateStore
from zlc_plot.fit import FitEngine


def _snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": np.arange(5, dtype=np.float64)}),
        dtype=np.float64,
        generation="fit-selection-test",
    )
    return DatasetSnapshot(schema, np.arange(5, dtype=np.float64).reshape(1, 5), revision=3)


def _projection(spec, *, selectors=(), viewport=None) -> FitProjection:
    snapshot = _snapshot()
    schema = parameter_schema_for(spec, style=DEFAULTS.style)
    display = DisplayStateStore(schema).state
    projection = FitProjection(
        data=snapshot,
            revision=snapshot.ref.revision.value,
        spec=spec,
        context=ProjectionContext(display, SelectorSnapshot(tuple(selectors)), viewport=viewport),
        unit_registry=None,
        defaults=DEFAULTS,
        histogram_projection=None,
    )
    projection._build_view_and_payload()
    return projection


def test_curve_fit_selection_prefers_area_then_x_range_then_viewport_then_all() -> None:
    spec = CurvePlot(AxisRef.point("x"))
    area = SelectorState(
        SelectorKind.AREA,
        RectangleRange(NumericRange(1, 4), NumericRange(2.5, 4.5)),
    )
    x_range = SelectorState(SelectorKind.X_RANGE, NumericRange(2, 4))
    viewport = RectangleRange(NumericRange(1, 3), NumericRange(-100, 100))
    model = FitEngine().registry.get("gaussian_offset")

    selected = _projection(spec, selectors=(area, x_range), viewport=viewport).fit_selection(model)
    assert selected.scope is FitScope.SELECTOR
    assert selected.selector_kind is SelectorKind.AREA
    assert selected.sample_count == 2

    selected = _projection(spec, selectors=(x_range,), viewport=viewport).fit_selection(model)
    assert selected.selector_kind is SelectorKind.X_RANGE
    assert selected.sample_count == 3

    selected = _projection(spec, viewport=viewport).fit_selection(model)
    assert selected.scope is FitScope.VIEWPORT
    assert selected.sample_count == 3

    selected = _projection(spec).fit_selection(model)
    assert selected.scope is FitScope.ALL
    assert selected.sample_count == 5


def test_histogram_fit_uses_painted_count_bins_only() -> None:
    spec = HistogramPlot()
    projection = _projection(spec)
    model = FitEngine().registry.get("histogram_gaussian")
    selection = projection.fit_selection(model)
    assert selection.coordinates[0].ndim == 1
    assert selection.observations.ndim == 1
    assert selection.selected_indices is not None
    assert selection.sample_count == selection.observations.size
    density_projection = projection._with_context(
        ProjectionContext(
            DisplayStateStore(
                parameter_schema_for(spec, style=DEFAULTS.style),
                {"density": True},
            ).state,
            SelectorSnapshot(()),
        )
    )
    density_projection._build_view_and_payload()
    with pytest.raises(ValueError, match="density=False"):
        density_projection.fit_selection(model)
