"""Labels cross semantic edits by role, never by slot.

Regression for the semantic-UX audit finding: kind switches used to copy
``PlotLabels`` verbatim, so a histogram inherited a curve's x-axis label and
its x axis read "Time (mV)" while actually plotting signal values.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zlc_plot import (
    AxisRef, CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, PlotLabels,
    PlotSession, RollingPlot, updated_spec,
)
from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
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
    assert edited.labels == PlotLabels(title="Grid")
    assert edited.cell.labels.x is None
    assert edited.cell.labels.y == "Signal (mV)"
    histogram = updated_spec(_schema(), edited, "kind", PlotKind.HISTOGRAM)
    assert histogram.labels == PlotLabels(title="Grid", x="Signal (mV)")


def test_merge_labels_never_copies_by_slot() -> None:
    histogram_labels = merge_labels(
        _CURVE,
        updated_spec(_schema(), _CURVE, "kind", PlotKind.HISTOGRAM),
    )
    assert histogram_labels.x == "Signal (mV)"
    assert histogram_labels.title == "Loading curve"
    assert histogram_labels.y is None


@pytest.mark.parametrize("spec", (
    CurvePlot(AxisRef.cell_data("x")),
    RollingPlot(),
    HistogramPlot(),
    ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
    FacetGridPlot(AxisRef.point("frame"), CurvePlot(AxisRef.cell_data("x"))),
))
def test_value_name_follows_the_quantity_and_explicit_labels_win(spec) -> None:
    schema = make_dataset_schema(
        repeat_domain(size=2), mapped_domain_from_columns({"frame": [0, 1]}),
        cell_axes=(axis("y", size=4), axis("x", size=5)), value_unit="V",
    )
    schema = replace(schema, value_schema=replace(schema.value_schema, name="Photon signal"))
    session = PlotSession(make_snapshot(schema, np.arange(80.0).reshape(2, 2, 4, 5), 1), spec)
    try:
        for expected in ("Photon signal (V)", "Photon signal (mV)", "Manual (mV)", "Display (mV)"):
            if expected == "Photon signal (mV)":
                session.set_value_unit("mV")
            elif expected == "Manual (mV)":
                session.replace_spec(replace(spec, labels=PlotLabels(value="Manual")))
            elif expected == "Display (mV)":
                session.set_parameter("value_label", "Display")
            renderer = session._renderer
            assert expected in renderer._effective_labels(session._payload, session.display_state)
            drawn = [text.get_text() for text in renderer.figure.texts]
            drawn += [label for axes in renderer.figure.axes for label in (axes.get_xlabel(), axes.get_ylabel())]
            assert expected in drawn
        if isinstance(spec, FacetGridPlot):
            session.focus_facet(1)
            assert session._renderer.primary_axes.get_ylabel() == "Display (mV)"
        elif isinstance(spec, ImagePlot):
            session.set_parameter("presentation", "height_bars")
            assert session._renderer._artists["image:colorbar"].ax.get_ylabel() == "Display (mV)"
        elif isinstance(spec, HistogramPlot):
            assert session._renderer.primary_axes.get_ylabel() == "Shots"
            session.set_parameter("density", True)
            assert session._renderer.primary_axes.get_ylabel() == "density"
    finally:
        session.close()
