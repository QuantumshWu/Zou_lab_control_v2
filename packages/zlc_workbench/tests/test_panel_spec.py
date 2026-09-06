"""The console fixes a panel's kind; every axis choice is the library's.

A camera frame's Point domain has one row, and "1D vector" on a camera
signal once opened as one invisible point -- so this resolver grew a rule
of its own that re-pointed a degenerate x, and the console's default and
the library's diverged on every shape the two rules read differently.
The one default table in ``zlc_plot._kinds.defaults`` carries the rule;
``fitting_panel_spec`` returns exactly what ``fitting_spec`` returns.
"""

from __future__ import annotations

import numpy as np

from zlc_data import (
    REPEAT,
    SCALAR_DOMAIN,
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
    ValueSchema,
)
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, PlotKind, fitting_spec
from zlc_plot.semantics import axis_size
from data_factory import (
    axis,
    make_dataset_schema,
    mapped_domain_from_columns,
    repeat_domain,
)

from zlc_workbench.panel_spec import fitting_panel_spec

def _camera_frame_schema():
    return make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"frame": np.zeros(1)}),
        cell_axes=(
            axis("spatial-y", size=8),
            axis("spatial-x", size=6),
        ),
        dtype=np.float64,
    )

def _scanned_schema():
    return make_dataset_schema(
        repeat_domain(size=2),
        mapped_domain_from_columns(
            {"row": np.array([0.0, 0.0, 1.0, 1.0]), "x": np.arange(4.0)}
        ),
        dtype=np.float64,
    )

def test_default_curve_on_a_camera_frame_lands_on_a_dense_axis() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "curve")
    assert spec == fitting_spec(schema, PlotKind.CURVE)
    assert spec is not None and spec.kind is PlotKind.CURVE
    assert axis_size(schema, spec.x) > 1

def test_default_curve_on_a_scan_keeps_the_scan_axis() -> None:
    schema = _scanned_schema()
    spec = fitting_panel_spec(schema, "curve")
    assert spec == fitting_spec(schema, PlotKind.CURVE)
    assert spec is not None
    assert spec.x == AxisRef.point("x")

def test_facet_grid_curve_cell_default_is_also_dense() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "facet_grid", "curve")
    assert spec == fitting_spec(schema, PlotKind.FACET_GRID, cell=PlotKind.CURVE)
    if spec is None:
        # A camera frame may not admit a facet grid at all; that refusal is
        # the library's own and not this resolver's concern.
        return
    assert axis_size(schema, spec.cell.x) > 1

def test_image_specs_pass_through_untouched() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "image")
    assert spec == fitting_spec(schema, PlotKind.IMAGE)
    assert spec is not None and spec.kind is PlotKind.IMAGE

def test_the_resolver_never_chooses_an_axis_of_its_own() -> None:
    """The shape on which two rules gave two answers.

    Two repeat axes, one of a single value and one of five, over an
    unnamed point row and a scalar cell: the library's table answered
    ``r1`` and this resolver re-pointed to ``r2``.  The table now moves a
    degenerate x to the next axis with variation itself, and this layer
    returns its answer for every kind it is asked for.
    """

    repeat = DomainSpec(
        (5,),
        (
            AxisSpec(AxisId("r1"), "r1", REPEAT, 1, (0,)),
            AxisSpec(AxisId("r2"), "r2", REPEAT, 5, (0, 1, 2, 3, 4)),
        ),
        ((0,) * 5, (0, 1, 2, 3, 4)),
    )
    schema = DatasetSchema(
        repeat,
        DomainSpec((1,), (), ()),
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("<f8")),
    )
    expected = CurvePlot(AxisRef.repeat("r2"))
    assert fitting_panel_spec(schema, "curve") == fitting_spec(schema, PlotKind.CURVE) == expected
    assert fitting_panel_spec(schema, "facet_grid", "curve") == fitting_spec(
        schema, PlotKind.FACET_GRID, cell=PlotKind.CURVE
    ) == FacetGridPlot(None, expected)
    assert fitting_panel_spec(schema) == fitting_spec(schema) == expected

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

def test_a_publisher_switch_never_prints_latex() -> None:
    r"""The Outputs switches are plain QLabels; nothing renders mathtext there.

    They took ``display_label`` verbatim, so the operator read the literal
    characters ``$\tau$`` and ``$\mathrm{FWHM}$ error`` beside a switch.
    The symbol is the same parameter written the way it is also typed into
    the Parameters box and printed in the formula above the plot.
    """

    from zlc_plot.fit import builtin_fit_models
    from zlc_workbench.panel_state import fit_output_fields

    models = builtin_fit_models()
    assert models
    for model in models:
        fields = fit_output_fields({"model": model.model_id}, models)
        assert fields, model.model_id
        published = {name for name, _label in fields}
        for name, label in fields:
            assert "$" not in label and "\\" not in label, (model.model_id, label)
        # The name half is the published signal id and the persisted toggle
        # key -- it is an identity and does not follow the label.  ONE per
        # parameter: the second switch published a separate "<name>_err"
        # signal that nothing related to its value, and a parameter now
        # carries its own uncertainty on the value plane instead.
        assert published == {
            str(parameter.name) for parameter in model.parameters
        }, (model.model_id, published)
