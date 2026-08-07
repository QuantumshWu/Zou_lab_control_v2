from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from test_facet_live_fit import _facet_snapshot, _spec
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, PlotSession
from zlc_plot.fit import FacetFitBatchResult


def test_facet_fit_exposes_numeric_columns_and_overlay_errors() -> None:
    session = PlotSession(_facet_snapshot(), _spec())
    try:
        result = session.fit("gaussian_offset", live=True)
        assert isinstance(result, FacetFitBatchResult)
        table = result.table
        assert table.parameter_names == result.parameter_names
        assert table.source_revision == result.source_revision == 0
        assert table.batch_revision == result.batch_revision == 1
        assert table.sample_axis_name == "facet"
        assert np.array_equal(table.sample_coordinates, [0.0, 1.0])
        assert table.sample_labels is None
        assert table.success.dtype == np.bool_
        for name in table.parameter_names:
            expected = np.asarray(
                [
                    parameter.standard_error
                    for overlay in result.overlays
                    for parameter in overlay.parameter_display
                    if parameter.name == name
                ],
                dtype=float,
            )
            assert np.allclose(table.parameter_errors[name], expected)
            assert np.array_equal(
                table.parameter_error_validity[name],
                table.success & np.isfinite(table.parameter_errors[name]),
            )
            assert table.parameter_values[name].dtype == np.float64
            assert table.parameter_values[name].flags.writeable is False
    finally:
        session.close()


def test_single_fit_has_the_same_numeric_table_shape() -> None:
    snapshot = _facet_snapshot()
    session = PlotSession(snapshot, CurvePlot(AxisRef.point("x")))
    try:
        result = session.fit("gaussian_offset", live=False)
        table = result.table
        assert table.success.shape == (1,)
        assert table.sample_axis_name == ""
        assert np.array_equal(table.sample_coordinates, [0.0])
        assert table.sample_unit == ""
        assert table.sample_labels is None
        assert set(table.parameter_units) == set(result.parameter_names)
        assert table.source_revision == 0
        assert table.batch_revision == 1
    finally:
        session.close()


def test_text_facet_coordinates_use_ordinals_and_labels() -> None:
    schema = DatasetSchema.create(
        Axis.create("repeat", values=["a", "b"]),
        PointTable.from_columns({"x": [0.0, 1.0]}),
        dtype=np.float64,
        generation="text-facet-table",
    )
    snapshot = DatasetSnapshot(schema, np.ones(schema.physical_shape), revision=0)
    spec = FacetGridPlot(AxisRef.repeat(), CurvePlot(AxisRef.point("x")))
    session = PlotSession(snapshot, spec)
    try:
        result = session.fit("gaussian_offset", live=True)
        assert isinstance(result, FacetFitBatchResult)
        assert np.array_equal(result.sample_coordinates, [0.0, 1.0])
        assert result.sample_unit == ""
        assert result.sample_labels == ("repeat=a", "repeat=b")
    finally:
        session.close()
