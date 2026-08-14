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
from zlc_plot.semantics import axis_size, updated_spec
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
    assert description.fate(AxisRef.repeat()) == "reduce"
    assert all(field.rebuild for field in description.fields)
    controls = semantic_controls(description)
    # The editor IS the table: the kind, one row per axis, and how the axes
    # nobody drew along are collapsed.
    names = tuple(control.name for control in controls)
    assert names[0] == "kind" and names[-1] == "reduction"
    assert tuple(name for _axis, name in description.fate_rows) == names[1:-1]
    assert description.fate(AxisRef.point("x")) == "x"
    assert all(control.semantic and control.rebuild for control in controls)


def test_a_pinned_axis_narrows_everything_the_panel_shows() -> None:
    """"Row 1, please" -- and the fit sees row 1 as well.

    The snapshot is 2 repeats x 4 point rows of 0..7 whose "row" column reads
    [0, 0, 1, 1], so a panel pinned to row 1 holds exactly {2, 3, 6, 7}.
    Restricting at the single view-construction point is what makes that true
    of the payload, the fit and the selectors at once, not of the drawing
    alone.
    """

    snapshot = _snapshot()
    schema = snapshot.block.schema
    row = next(
        field
        for field in describe_semantics(schema, HistogramPlot()).fields
        if field.name == "fate:row"
    )
    assert (1.0, "= 1") in row.choices

    spec = updated_spec(schema, HistogramPlot(), row.name, 1.0)
    assert spec.scope == ((AxisRef.point("row"), 1.0),)

    session = PlotSession(snapshot, spec)
    try:
        seen = np.asarray(session._view._samples.value.canonical).reshape(-1)
        assert sorted(seen.tolist()) == [2.0, 3.0, 6.0, 7.0]
        assert int(np.asarray(session._payload.counts).sum()) == 4
    finally:
        session.close()

    unscoped = PlotSession(snapshot, HistogramPlot())
    try:
        assert int(np.asarray(unscoped._payload.counts).sum()) == 8
    finally:
        unscoped.close()


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
    # Every axis row offers the fate an unassigned axis already has, so
    # "none of the above" is a fate rather than a null.
    assert all(
        "reduce" in description.field(name).choice_values
        for _axis, name in description.fate_rows
    )
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
    facet_values = description.axes_offering("facet")
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
    x_values = description.axes_offering("x")
    assert AxisRef.point("frame") in x_values  # the current state stays
    assert AxisRef.point_rows() not in x_values
    assert AxisRef.repeat() not in x_values
    assert AxisRef.data("spatial-y") in x_values
    assert AxisRef.data("spatial-x") in x_values
    group_values = description.axes_offering("group")
    assert AxisRef.point("frame") not in group_values
    assert AxisRef.point_rows() not in group_values
    assert AxisRef.repeat() not in group_values
    # And the ordinal is not in the vocabulary at all: the frame column
    # names every row, so the rows ARE the frames.
    assert AxisRef.point_rows() not in description.axis_choices


def test_non_series_kinds_keep_degenerate_axes_where_legitimate() -> None:
    schema = _camera_frame_schema()
    description = describe_semantics(
        schema,
        ImagePlot(AxisRef.data("spatial-x"), AxisRef.data("spatial-y")),
    )
    # An image is not a series; its axis domains stay the schema's own.
    x_values = description.axes_offering("x")
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
    x_values = description.axes_offering("x")
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


def test_a_camera_cycle_names_its_rows_frames_and_nothing_else() -> None:
    """What an operator reads for the three shapes a bench produces."""

    from data_factory import PointTopology

    camera = describe_semantics(_camera_frame_schema(), CurvePlot(AxisRef.point("frame")))
    assert [name.removeprefix('fate:') for _ref, name in camera.fate_rows] == [
        "repeat",
        "frame",
        "spatial-y",
        "spatial-x",
    ]

    scan = DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns({"detuning": [0.0, 1.0, 2.0, 3.0]}),
        dtype=np.float64,
    )
    curve = describe_semantics(scan, CurvePlot(AxisRef.point("detuning")))
    assert [name.removeprefix("fate:") for _ref, name in curve.fate_rows][:2] == [
        "repeat",
        "detuning",
    ]


def test_a_two_dimensional_scan_reports_the_fate_its_panel_applies() -> None:
    """The table said "reduce" for axes the panel was giving cells to.

    It was built from the OFFERING alone, and a facet over the point rows is
    not in the offering of a dataset whose topology names them -- so the row
    was missing, `fate()` raised for the spec's own facet, and the two
    dimensions were reported as reduced while every (x, y) pair had its own
    cell.
    """

    import itertools

    from data_factory import PointTopology
    from zlc_plot import FacetGridPlot

    values = [0.0, 1.0, 2.0]
    rows = list(itertools.product(values, values))
    table = PointTable.from_columns(
        {"coil_x": [row[0] for row in rows], "coil_y": [row[1] for row in rows]}
    )
    topology = PointTopology.from_cartesian(
        (
            Axis.create("coil_x", values=values),
            Axis.create("coil_y", values=values),
        ),
        point_table=table,
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        table,
        point_topology=topology,
        dtype=np.float64,
    )
    spec = FacetGridPlot(
        AxisRef.point_rows(), CurvePlot(AxisRef.point_dimension("coil_y"))
    )
    description = describe_semantics(schema, spec)
    assert description.fate(AxisRef.point_rows()) == "facet"
    assert [name.removeprefix('fate:') for _ref, name in description.fate_rows][0] == "repeat"
    assert "fate:coil_x" in [name for _ref, name in description.fate_rows]


def test_the_semantic_description_says_what_the_panel_is_drawing() -> None:
    """One line beside the panel's name: structure, then what it hides.

    An axis drawn as x, y or a group is on the picture and says so.  The
    three ways data disappears without a mark -- faceted, pinned to one
    value, or collapsed into one -- are what an operator has to be told, and
    are exactly what this quotes.  It is derived from the same fate rows the
    semantic table is built from, so the strip and the table cannot disagree.
    """

    from zlc_plot.semantics import describe_semantics, schema_summary

    schema = _snapshot().block.schema

    def caption(spec) -> str:
        return describe_semantics(schema, spec).caption

    flat = CurvePlot(AxisRef.point("x"))
    assert caption(flat).startswith(schema_summary(schema))
    # The repeat axis has two values and nobody drew it: it is being reduced,
    # which is the whole reason to say so.  The drawn axis is not repeated.
    assert "repeat→reduce" in caption(flat) and "→x" not in caption(flat)
    assert "repeat→facet" in caption(
        FacetGridPlot(AxisRef.repeat(), CurvePlot(AxisRef.point("x")))
    )
    assert "repeat=1" in caption(
        CurvePlot(AxisRef.point("x"), scope=((AxisRef.repeat(), 1.0),))
    )
