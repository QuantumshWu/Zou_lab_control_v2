"""The console's default axis pick can always carry what the kind draws.

A camera frame's point table has one row, and the plot library's curve
default walks the point domain -- so "1D vector" on a camera signal opened as
one invisible point.  The workbench resolver re-points a degenerate series x
onto a dense axis through the library's own composition authority.
"""

from __future__ import annotations

import numpy as np

from zlc_plot import AxisRef, PlotKind
from zlc_plot.semantics import axis_size
from data_factory import Axis, DatasetSchema, PointTable

from zlc_workbench.panel_spec import fitting_panel_spec


def _camera_frame_schema():
    return DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"frame": np.zeros(1)}),
        data_axes=(
            Axis.create("spatial-y", size=8),
            Axis.create("spatial-x", size=6),
        ),
        dtype=np.float64,
    )


def _scanned_schema():
    return DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns(
            {"x": np.arange(4.0), "row": np.array([0.0, 0.0, 1.0, 1.0])}
        ),
        dtype=np.float64,
    )


def test_default_curve_on_a_camera_frame_lands_on_a_dense_axis() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "curve")
    assert spec is not None and spec.kind is PlotKind.CURVE
    assert axis_size(schema, spec.x) > 1


def test_default_curve_on_a_scan_keeps_the_scan_axis() -> None:
    schema = _scanned_schema()
    spec = fitting_panel_spec(schema, "curve")
    assert spec is not None
    assert spec.x == AxisRef.point("x")


def test_facet_grid_curve_cell_default_is_also_dense() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "facet_grid", "curve")
    if spec is None:
        # A camera frame may not admit a facet grid at all; that refusal is
        # the library's own and not this resolver's concern.
        return
    assert axis_size(schema, spec.cell.x) > 1


def test_image_specs_pass_through_untouched() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "image")
    assert spec is not None and spec.kind is PlotKind.IMAGE


def test_the_workbench_joins_the_parameters_that_move_together() -> None:
    """The view layer cannot see the declaration, so this seam answers.

    A limit pair is validated as a pair: moving (0, 10) to (12, 20) passes
    through (12, 10), which no owner accepts, so an editor has to send both
    ends as the operator currently sees them.  Which names form a pair is
    declared in zlc_plot, and zlc_plot is a forbidden import root for the
    view -- guarded mechanically in zlc_ui's own tests.

    The view therefore recovered the relationship from how the names were
    SPELLED, pairing any *_min with its *_max.  That is the same fact with a
    second and weaker owner: it agreed with the declaration only by luck,
    and would have paired the first *_min that was not a limit.  This seam
    turns a plot control into a frontend-neutral row and may ask; the join
    is made here, once, and travels on the row.
    """

    from zlc_plot.specs import limit_pairs, parameter_schema_for_kind
    from zlc_plot.style import build_plot_style
    from zlc_plot.ui import parameter_controls
    from zlc_workbench.panel_state import control_document

    style = build_plot_style()
    for kind, expected in (
        ("image", {"color_min": "color_max", "color_max": "color_min"}),
        (
            "histogram",
            {
                "y_min": "y_max",
                "y_max": "y_min",
                "x_min": "x_max",
                "x_max": "x_min",
            },
        ),
        ("curve", {"y_min": "y_max", "y_max": "y_min"}),
    ):
        schema = parameter_schema_for_kind(kind, style=style)
        values = {name: spec.default for name, spec in schema.items()}
        joined = {}
        for control in parameter_controls(schema, values):
            row = control_document(control)
            assert "co_edited_with" in row, row["key"]
            if row["co_edited_with"]:
                joined[row["key"]] = row["co_edited_with"]
        assert joined == expected, (kind, joined)

    # And the join is the DECLARATION's, not a guess that happens to agree:
    # every name it pairs is one of the declared pairs.
    declared = set()
    for _mode, low, high in limit_pairs():
        declared.add((low, high))
        declared.add((high, low))
    schema = parameter_schema_for_kind("histogram", style=style)
    values = {name: spec.default for name, spec in schema.items()}
    for control in parameter_controls(schema, values):
        row = control_document(control)
        if row["co_edited_with"]:
            assert (row["key"], row["co_edited_with"]) in declared
