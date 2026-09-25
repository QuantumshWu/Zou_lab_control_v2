"""The rows the workbench builds from zlc_plot's parameter declarations.

A plot control reaches the view as a frontend-neutral row; what travels on
that row -- which parameters move together, the label a choice was given, the
value Auto resolves to, a published fit parameter's plain label -- is read
from the declaration here, once, and never reconstructed from spelling.
"""

from __future__ import annotations


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


def test_a_choice_row_shows_the_label_its_declaration_gave_it() -> None:
    """A unit symbol is shown as it is spelled; an enum member by its name.

    The document used to title-case every plain choice value, which turned
    ``mT`` into ``Mt`` and ``µT`` into ``Μt`` (a Greek capital mu): a symbol
    is not a word, and the only thing that knows what a choice is called is
    the declaration that offered it.
    """

    from zlc_data.units import DEFAULT_UNITS
    from zlc_plot.specs import parameter_schema_for_kind
    from zlc_plot.style import build_plot_style
    from zlc_plot.ui import parameter_controls
    from zlc_workbench.panel_state import control_document

    schema = parameter_schema_for_kind("curve", style=build_plot_style())
    values = {name: spec.default for name, spec in schema.items()}
    units = DEFAULT_UNITS.display_choices("uT")
    unit_key = next(name for name in schema if name.endswith("_display_unit"))
    rows = {
        row["key"]: row
        for row in (
            control_document(control)
            for control in parameter_controls(
                schema, values, choice_overrides={unit_key: units}
            )
        )
    }
    assert [label for label, _value in rows[unit_key]["choices"]] == list(units)
    assert [value for _label, value in rows[unit_key]["choices"]] == list(units)
    assert [label for label, _value in rows["relim_mode"]["choices"]] == [
        "Tight",
        "Normal",
        "Fixed",
    ]


def test_a_control_on_auto_is_built_holding_what_auto_resolves_to() -> None:
    """Switching a unit off Auto keeps the unit the axis was being read in.

    The control on Auto held nothing, so the moment Auto went off it took
    the first entry of its list -- tesla for a microtesla axis.  What Auto
    resolves to is the session's answer, travels on the row, and is what
    the control is built holding.
    """

    from zlc_data.units import DEFAULT_UNITS
    from zlc_plot.specs import parameter_schema_for_kind
    from zlc_plot.style import build_plot_style
    from zlc_plot.ui import parameter_controls
    from zlc_ui.console._panel_projection import parameter_form_spec, parameter_form_values
    from zlc_workbench.panel_state import control_document

    schema = parameter_schema_for_kind("curve", style=build_plot_style())
    values = {name: spec.default for name, spec in schema.items()}
    unit_key = next(name for name in schema if name.endswith("_display_unit"))
    units = DEFAULT_UNITS.display_choices("uT")
    rows = tuple(
        control_document(control)
        for control in parameter_controls(
            schema,
            values,
            choice_overrides={unit_key: units},
            automatic_values={unit_key: "µT"},
        )
    )
    row = next(entry for entry in rows if entry["key"] == unit_key)
    assert row["automatic"] and row["value"] is None and row["automatic_value"] == "µT"
    field = next(item for item in parameter_form_spec(rows).fields if item.key == unit_key)
    assert field.automatic and field.default == "µT"
    # The form's VALUE is still Auto: only the control's seed changed.
    assert parameter_form_values(rows)[unit_key] is None
