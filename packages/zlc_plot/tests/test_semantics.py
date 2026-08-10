from __future__ import annotations

import numpy as np

from zlc_plot import (
    AxisRef,
    CurvePlot,
    DEFAULTS,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotKind,
    PlotSession,
    PulseTimelinePlot,
    Reduction,
    describe_semantics,
)
from zlc_plot.semantics import axis_size
from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot._kinds import HANDLERS
from zlc_plot.selectors import NumericRange, RectangleRange
from zlc_plot.specs import parameter_schema_for
from zlc_plot.session_policy import replace_spec_initial_state
from zlc_plot.ui import semantic_controls


def _snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns(
            {"x": np.arange(4.0), "row": np.array([0.0, 0.0, 1.0, 1.0])}
        ),
        dtype=np.float64,
        generation="semantic-tests",
    )
    return DatasetSnapshot(schema, np.arange(8.0).reshape(2, 4), revision=0)


def test_describe_semantics_is_registry_derived_and_marks_rebuild() -> None:
    snapshot = _snapshot()
    description = describe_semantics(
            snapshot.block.schema,
        CurvePlot(AxisRef.point("x")),
    )
    assert PlotKind.CURVE in description.kind_choices
    assert AxisRef.repeat() in description.axis_choices
    assert description.field("group").value is None
    assert all(field.rebuild for field in description.fields)
    controls = semantic_controls(description)
    assert tuple(control.name for control in controls) == (
        "kind",
        "x",
        "group",
        "reduction",
    )
    assert all(control.semantic and control.rebuild for control in controls)


def test_semantic_choices_are_labeled_once_and_kind_domain_is_registry_filtered() -> None:
    snapshot = _snapshot()
    description = describe_semantics(
            snapshot.block.schema,
        CurvePlot(AxisRef.point("x")),
    )
    for field in description.fields:
        values = tuple(choice[0] for choice in field.choices)
        labels = tuple(choice[1] for choice in field.choices)
        assert len(values) == len(set(values))
        assert all(isinstance(label, str) and label for label in labels)
        assert all("AxisRef(" not in label for label in labels)
    expected = tuple(
        handler.kind for handler in HANDLERS if handler.admits(snapshot.block.schema)
    )
    kind_field = description.field("kind")
    # Every offered kind is usable; here every admitted kind also has a
    # default, so the offered domain equals the admitted set.
    assert kind_field.choice_values == expected
    assert description.field("group").choices[0] == (None, "(none)")
    assert description.field("reduction").choices[-1][1] == "first"

def test_facet_domain_excludes_cell_axes() -> None:
    spec = FacetGridPlot(
        AxisRef.point("row"),
        CurvePlot(AxisRef.point("x")),
    )
    points = PointTable.from_columns(
        {
            "x": np.arange(4.0),
            "row": np.array([0.0, 0.0, 1.0, 1.0]),
            "row2": np.array([0.0, 1.0, 0.0, 1.0]),
        }
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1), points, dtype=np.float64
    )
    description = describe_semantics(schema, spec)
    facet_values = description.field("facet").choice_values
    assert AxisRef.point("x") not in facet_values
    assert AxisRef.point("row") in facet_values

def _camera_frame_schema():
    """One camera frame: R=1, one point row, dense y × x data axes."""

    return DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"frame": np.zeros(1)}),
        data_axes=(
            Axis.create("spatial-y", size=8),
            Axis.create("spatial-x", size=6),
        ),
        dtype=np.float64,
    )


def test_axis_size_measures_every_semantic_axis_domain() -> None:
    schema = _camera_frame_schema()
    assert axis_size(schema, AxisRef.repeat()) == 1
    assert axis_size(schema, AxisRef.point_rows()) == 1
    assert axis_size(schema, AxisRef.point("frame")) == 1
    assert axis_size(schema, AxisRef.data("spatial-y")) == 8
    assert axis_size(schema, AxisRef.data("spatial-x")) == 6
    scanned = _snapshot().block.schema
    assert axis_size(scanned, AxisRef.repeat()) == 2
    assert axis_size(scanned, AxisRef.point("x")) == 4


def test_series_x_and_group_choices_exclude_degenerate_axes() -> None:
    """A size-1 axis can never carry a multi-point series.

    On a camera frame the point domain has one row, so offering it as a curve
    x yields one invisible point.  The x/group domains of series-family kinds
    exclude every size-1 axis; the current value stays offered because it is
    the actual state.
    """

    schema = _camera_frame_schema()
    description = describe_semantics(schema, CurvePlot(AxisRef.point("frame")))
    x_values = description.field("x").choice_values
    assert AxisRef.point("frame") in x_values  # the current state stays
    assert AxisRef.point_rows() not in x_values
    assert AxisRef.repeat() not in x_values
    assert AxisRef.data("spatial-y") in x_values
    assert AxisRef.data("spatial-x") in x_values
    group_values = description.field("group").choice_values
    assert AxisRef.point("frame") not in group_values
    assert AxisRef.point_rows() not in group_values
    assert AxisRef.repeat() not in group_values
    # The axis vocabulary itself is unfiltered; only series roles are.
    assert AxisRef.point_rows() in description.axis_choices


def test_non_series_kinds_keep_degenerate_axes_where_legitimate() -> None:
    schema = _camera_frame_schema()
    description = describe_semantics(
        schema,
        ImagePlot(AxisRef.data("spatial-x"), AxisRef.data("spatial-y")),
    )
    # An image is not a series; its axis domains stay the schema's own.
    x_values = description.field("x").choice_values
    assert AxisRef.data("spatial-x") in x_values


def test_facet_grid_series_cell_filters_its_x_domain_too() -> None:
    points = PointTable.from_columns(
        {
            "x": np.arange(4.0),
            "row": np.array([0.0, 0.0, 1.0, 1.0]),
        }
    )
    schema = DatasetSchema.create(Axis.create("repeat", size=1), points)
    description = describe_semantics(
        schema,
        FacetGridPlot(AxisRef.point("row"), CurvePlot(AxisRef.point("x"))),
    )
    x_values = description.field("x").choice_values
    assert AxisRef.repeat() not in x_values  # size 1 cannot carry a series
    assert AxisRef.point("x") in x_values


def test_pulse_semantics_has_the_same_frontend_contract_without_dataset_axes() -> None:
    description = describe_semantics(None, PulseTimelinePlot())
    assert description.kind is PlotKind.PULSE_TIMELINE
    assert description.kind_choices == (PlotKind.PULSE_TIMELINE,)
    assert description.axis_choices == ()
    assert description.field("kind").rebuild


def test_replace_spec_policy_revalidates_and_retains_only_valid_state() -> None:
    old = CurvePlot(AxisRef.point("x"), reduction=Reduction.SUM)
    new = HistogramPlot()
    old_schema = parameter_schema_for(old, style=DEFAULTS.style)
    new_schema = parameter_schema_for(new, style=DEFAULTS.style)
    values = dict(old_schema.initial_values())
    values.update({"title": "kept", "x_display_unit": "mV"})
    viewport = RectangleRange(NumericRange(0.0, 1.0), NumericRange(0.0, 1.0))
    result = replace_spec_initial_state(
        old,
        new,
        values,
        new_schema,
        size="4x4",
        viewport=viewport,
    )
    assert result.parameters["title"] == "kept"
    assert "x_display_unit" not in result.parameters
    assert result.size == "4x4"
    assert result.viewport is None


def test_replace_spec_policy_keeps_viewport_for_reduction_only_change() -> None:
    snapshot = _snapshot()
    session = PlotSession(snapshot, CurvePlot(AxisRef.point("x")))
    try:
        viewport = RectangleRange(NumericRange(0.0, 1.0), NumericRange(0.0, 1.0))
        session.set_viewport(viewport.x, viewport.y)
        session.replace_spec(
            CurvePlot(AxisRef.point("x"), reduction=Reduction.MEDIAN)
        )
        assert session.viewport == viewport
        assert session.selectors == ()
    finally:
        session.close()
