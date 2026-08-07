"""A histogram is the distribution of every acquired value."""

from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import HistogramPlot, PlotSession
from zlc_plot.data_view import DataView


def _snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns({"point": [0.0, 1.0]}),
        data_axes=(Axis.create("scan", values=[0.0, 1.0]),),
        dtype=np.float64,
        generation="histogram-pool",
    )
    values = np.arange(8, dtype=np.float64).reshape(schema.shape)
    return DatasetSnapshot(schema, values, revision=0)


def test_histogram_pools_the_whole_box() -> None:
    """Repeat x points x data axes all land in the pool: 8 values, 8 counts."""

    histogram = DataView(_snapshot()).histogram(bins=4)
    assert int(np.asarray(histogram.counts).sum()) == 8


def test_histogram_spec_needs_no_axis_declaration() -> None:
    spec = HistogramPlot()
    session = PlotSession(_snapshot(), spec)
    try:
        names = tuple(field.name for field in session.describe_semantics().fields)
        assert names == ("kind",)
    finally:
        session.close()
