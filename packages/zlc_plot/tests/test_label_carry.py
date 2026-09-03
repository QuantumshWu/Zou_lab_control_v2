"""Labels cross semantic edits by role, never by slot.

Regression for the semantic-UX audit finding: kind switches used to copy
``PlotLabels`` verbatim, so a histogram inherited a curve's x-axis label and
its x axis read "Time (mV)" while actually plotting signal values.
"""

from __future__ import annotations

import numpy as np

from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, PlotLabels, updated_spec
from data_factory import (
    axis,
    make_dataset_schema,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import DatasetSchema, REPEAT
from zlc_plot.kinds import PlotKind
from zlc_plot.session_policy import merge_labels


def _schema() -> DatasetSchema:
    return make_dataset_schema(
        repeat_domain(size=4),
        mapped_domain_from_columns(
            {
                "time": np.linspace(0.0, 1.0, 6),
                "sample": np.arange(6.0),
            }
        ),
        dtype=np.float64,
    )


_CURVE = CurvePlot(
    AxisRef.point("time"),
    labels=PlotLabels("Loading curve", "Time (ms)", "Signal (mV)"),
)


def test_kind_switch_moves_value_label_and_drops_axis_label() -> None:
    histogram = updated_spec(_schema(), _CURVE, "kind", PlotKind.HISTOGRAM)
    assert histogram.kind is PlotKind.HISTOGRAM
    # The curve's y described the plotted value; a histogram plots that value
    # on x, so the label moves slots.  "Time (ms)" described an axis no
    # histogram slot represents and must not survive anywhere.
    assert histogram.labels.x == "Signal (mV)"
    assert histogram.labels.y is None
    assert histogram.labels.title == "Loading curve"


def test_x_change_drops_only_the_x_label() -> None:
    edited = updated_spec(_schema(), _CURVE, "x", AxisRef.point("sample"))
    assert edited.labels.x is None
    assert edited.labels.y == "Signal (mV)"
    assert edited.labels.title == "Loading curve"


def test_reduction_change_keeps_every_label() -> None:
    from zlc_plot.specs import Reduction

    edited = updated_spec(_schema(), _CURVE, "reduction", Reduction.MIN)
    assert edited.labels == _CURVE.labels


def test_round_trip_through_rolling_keeps_value_label_only() -> None:
    schema = _schema()
    rolling = updated_spec(schema, _CURVE, "kind", PlotKind.ROLLING)
    # Rolling x is the shot counter, not the curve's time axis.
    assert rolling.labels.x is None
    assert rolling.labels.y == "Signal (mV)"
    back = updated_spec(schema, rolling, "kind", PlotKind.CURVE)
    assert back.labels.y == "Signal (mV)"
    assert back.labels.x is None
    assert back.labels.title == "Loading curve"


def test_facet_cell_edit_carries_by_cell_roles() -> None:
    facet = FacetGridPlot(
        AxisRef.repeat("repeat"),
        CurvePlot(AxisRef.point("time")),
        labels=PlotLabels("Grid", "Time (ms)", "Signal (mV)"),
    )
    edited = updated_spec(_schema(), facet, "x", AxisRef.point("sample"))
    assert edited.labels.title == "Grid"
    assert edited.labels.x is None
    assert edited.labels.y == "Signal (mV)"


def test_merge_labels_never_copies_by_slot() -> None:
    histogram_labels = merge_labels(
        _CURVE,
        updated_spec(_schema(), _CURVE, "kind", PlotKind.HISTOGRAM),
    )
    assert histogram_labels.x == "Signal (mV)"
    assert histogram_labels.title == "Loading curve"
    assert histogram_labels.y is None
