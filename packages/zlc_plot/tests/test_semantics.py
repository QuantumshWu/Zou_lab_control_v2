from __future__ import annotations

import numpy as np
import pytest

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
from zlc_plot.semantics import (
    typed_choice,
    SemanticVacancy,
    axis_size,
    composed_spec,
    fate_field_name,
    is_scope_fate,
    scope_fate,
    updated_spec,
)
from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot
from zlc_plot._kinds import HANDLERS
from zlc_plot.selectors import NumericRange, RectangleRange
from zlc_plot.specs import parameter_schema_for
from zlc_plot.session_policy import replace_spec_initial_state
from zlc_plot.ui import semantic_controls

def _snapshot() -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=2),
        mapped_domain_from_columns(
            {"x": np.arange(4.0), "row": np.array([0.0, 0.0, 1.0, 1.0])}
        ),
        dtype=np.float64,
    )
    return make_snapshot(schema, np.arange(8.0).reshape(2, 4), revision=0)

def test_describe_semantics_is_registry_derived_and_marks_rebuild() -> None:
    snapshot = _snapshot()
    description = describe_semantics(
            snapshot.block.schema,
        CurvePlot(AxisRef.point("x")),
    )
    assert PlotKind.CURVE in description.kind_choices
    assert AxisRef.repeat("repeat") in description.axis_choices
    assert description.fate(AxisRef.repeat("repeat")) == "reduce"
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
        if field.name == fate_field_name(AxisRef.point("row"))
    )
    assert row.cycle_choices is not None
    assert len(row.cycle_choices) == 2
    assert (scope_fate(1.0), "1") in tuple(row.cycle_choices)

    spec = updated_spec(schema, HistogramPlot(), row.name, scope_fate(1.0))
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

def test_a_scoped_axis_can_take_a_curve_role_again() -> None:
    snapshot = _snapshot()
    schema = snapshot.block.schema
    row = AxisRef.point("row")
    name = fate_field_name(row)
    base = CurvePlot(AxisRef.point("x"))
    scoped = updated_spec(schema, base, name, scope_fate(1.0))

    as_x = updated_spec(schema, scoped, name, "x")
    assert as_x.x == row and as_x.scope == ()

    as_group = updated_spec(schema, scoped, name, "group")
    assert as_group.group == row and as_group.scope == ()

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
        handler.kind
        for handler in HANDLERS
        if handler.admits(snapshot.block.schema)
        and (
            handler.kind is PlotKind.CURVE
            or handler.default_spec(snapshot.block.schema) is not None
        )
    )
    kind_field = description.field("kind")
    # Admission means an authored spec is valid; a kind switch additionally
    # needs an unambiguous default.  FacetGrid therefore stays out when the
    # only unused axis is repeat.
    assert kind_field.choice_values == expected
    # Every axis row offers the fate an unassigned axis already has, so
    # "none of the above" is a fate rather than a null.
    assert all(
        "reduce" in description.field(name).choice_values
        for _axis, name in description.fate_rows
    )
    assert description.field("reduction").choices[-1][1] == "first"

def test_a_choice_composes_typed_or_as_a_record_holds_it() -> None:
    """One vocabulary, two forms, and composition reads both.

    A choice travels as the typed member here and as the plain value a
    record holds everywhere else -- the panel state, the saved layout, the
    editor row a frontend hands back.  The Figure Viewer routes that row
    straight to apply_semantic, so a plain "sum" reached CurvePlot.reduction
    and the dataclass refused an edit made from this module's own list.
    """

    points = mapped_domain_from_columns(
        {"x": np.tile(np.arange(3.0), 3), "y": np.repeat(np.arange(3.0), 3)}
    )
    schema = make_dataset_schema(
        repeat_domain(size=2), points, dtype=np.float64
    )
    spec = CurvePlot(AxisRef.point("x"))
    assert updated_spec(schema, spec, "reduction", "sum").reduction is Reduction.SUM
    assert (
        updated_spec(schema, spec, "reduction", Reduction.SUM).reduction
        is Reduction.SUM
    )
    # The kind row is the same rule: a plain name reaches the kind branch
    # instead of being refused by an isinstance guard before it gets there.
    assert updated_spec(schema, spec, "kind", "curve") == spec
    assert updated_spec(schema, spec, "kind", PlotKind.CURVE) == spec
    assert typed_choice("kind", "image") is PlotKind.IMAGE
    # A value that names no choice is still refused, by the vocabulary.
    for name, bad in (("reduction", "nonsense"), ("kind", "bogus")):
        try:
            updated_spec(schema, spec, name, bad)
        except ValueError as error:
            assert bad in str(error)
        else:  # pragma: no cover
            raise AssertionError(f"{name}={bad!r} must be refused")

def test_the_facet_role_is_offered_by_the_same_rule_as_every_other_role() -> None:
    """A cell axis may take the facet, and taking it SWAPS.

    ``facet`` used to be the one role whose option list depended on where
    the other axes sat: an axis the cell already consumed was silently
    dropped from the list instead of swapping like every other role.  It
    left an axis the operator could not promote, and -- because an
    earlier swap can legitimately put ``facet`` on a cell axis -- a fate
    that its own row would refuse when the table was replayed.
    """

    spec = FacetGridPlot(
        AxisRef.point("row"),
        CurvePlot(AxisRef.point("x")),
    )
    points = mapped_domain_from_columns(
        {
            "x": np.arange(4.0),
            "row": np.array([0.0, 0.0, 1.0, 1.0]),
            "row2": np.array([0.0, 1.0, 0.0, 1.0]),
        }
    )
    schema = make_dataset_schema(
        repeat_domain(size=1), points, dtype=np.float64
    )
    description = describe_semantics(schema, spec)
    facet_values = description.axes_offering("facet")
    assert AxisRef.point("x") in facet_values
    assert AxisRef.point("row") in facet_values
    # and the swap is what makes it legal: the cell keeps an x axis.
    name = next(
        row for axis, row in description.fate_rows if axis == AxisRef.point("x")
    )
    composed = composed_spec(schema, spec, {name: "facet"})
    assert composed.facet == AxisRef.point("x")
    assert composed.cell.x == AxisRef.point("row")
    # a table stating that swap replays against the kind's own default
    replayed = describe_semantics(schema, composed)
    assert replayed.field(name).value == "facet"

def _camera_frame_schema():
    """One camera frame: R=1, one point row, dense y × x data axes."""

    return make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"frame": np.zeros(1)}),
        cell_axes=(
            axis("spatial-y", size=8),
            axis("spatial-x", size=6),
        ),
        dtype=np.float64,
    )

def test_axis_size_measures_every_semantic_axis_domain() -> None:
    schema = _camera_frame_schema()
    assert axis_size(schema, AxisRef.repeat("repeat")) == 1
    assert axis_size(schema, AxisRef.point("frame")) == 1
    assert axis_size(schema, AxisRef.cell_data("spatial-y")) == 8
    assert axis_size(schema, AxisRef.cell_data("spatial-x")) == 6
    scanned = _snapshot().block.schema
    assert axis_size(scanned, AxisRef.repeat("repeat")) == 2
    assert axis_size(scanned, AxisRef.point("x")) == 4

def test_equal_display_names_keep_distinct_stable_fate_keys() -> None:
    """A human label may repeat; the exact AxisRef remains the row identity."""

    from zlc_data import AxisId, AxisSpec, COMPONENT

    left = AxisRef.cell_data("channel.left")
    right = AxisRef.cell_data("channel.right")
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"sample": (0.0, 1.0)}),
        cell_axes=(
            AxisSpec(AxisId("channel.left"), "channel", COMPONENT, 2),
            AxisSpec(AxisId("channel.right"), "channel", COMPONENT, 3),
        ),
    )

    description = describe_semantics(
        schema,
        CurvePlot(AxisRef.point("sample")),
    )
    rows = dict(description.fate_rows)
    assert rows[left] == fate_field_name(left)
    assert rows[right] == fate_field_name(right)
    assert rows[left] != rows[right]
    assert description.field(rows[left]).label == "channel"
    assert description.field(rows[right]).label == "channel"
    left_cycles = description.field(rows[left]).cycle_choices
    right_cycles = description.field(rows[right]).cycle_choices
    assert left_cycles is not None and right_cycles is not None
    assert scope_fate(2.0) not in tuple(value for value, _label in left_cycles)
    assert scope_fate(2.0) in tuple(value for value, _label in right_cycles)

def test_exact_data_axis_id_wins_over_another_axis_name() -> None:
    """Declaration order cannot redirect an exact id through a name alias."""

    from zlc_data import AxisId, AxisSpec, COMPONENT
    from zlc_plot.data_view import DataView

    target = AxisRef.cell_data("axis.target")
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"sample": (0.0,)}),
        cell_axes=(
            AxisSpec(AxisId("axis.other"), "axis.target", COMPONENT, 2),
            AxisSpec(AxisId("axis.target"), "selected", COMPONENT, 3),
        ),
    )
    snapshot = make_snapshot(
        schema,
        np.arange(6.0).reshape(1, 1, 2, 3),
        revision=0,
    )

    assert axis_size(schema, target) == 3
    series = DataView(snapshot).curve(target).series[0]
    assert series.x.label == "selected"
    np.testing.assert_array_equal(series.x.canonical, np.arange(3.0))
    with pytest.raises(ValueError, match="no exact cell_data axis"):
        DataView(snapshot).curve(AxisRef.cell_data("selected"))
    session = PlotSession(
        snapshot,
        HistogramPlot(scope=((target, 1),)),
    )
    try:
        assert int(np.sum(session._payload.counts)) == 2
    finally:
        session.close()

def test_categorical_scope_values_do_not_collide_with_fate_tokens() -> None:
    """Text coordinates named x/reduce/latest remain coordinates, not verbs."""

    from zlc_data import LATEST_COORDINATE

    ref = AxisRef.point("mode")
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"mode": ("x", "reduce", "latest")}),
    )
    description = describe_semantics(schema, HistogramPlot())
    field = description.field(fate_field_name(ref))
    assert field.cycle_choices is not None
    scoped_choices = {
        label: value
        for value, label in field.cycle_choices
    }
    assert scoped_choices == {
        "x": scope_fate("x"),
        "reduce": scope_fate("reduce"),
        "latest": scope_fate("latest"),
    }
    assert scope_fate("latest") != scope_fate(LATEST_COORDINATE)

    for coordinate in ("x", "reduce", "latest"):
        selected = updated_spec(
            schema,
            HistogramPlot(),
            field.name,
            scope_fate(coordinate),
        )
        assert selected.scope == ((ref, coordinate),)
        assert describe_semantics(schema, selected).fate(ref) == scope_fate(
            coordinate
        )

def test_every_axis_may_take_every_role_its_kind_declares() -> None:
    """A size-one axis is still an axis the operator may draw along.

    Excluding degenerate axes from x and group left rows in the fate table
    that could not be edited at all -- a camera cycle's single point row,
    a single-shot repeat -- while the operator's reason for wanting them
    (provenance on the x axis, one frame split out) is legitimate and
    draws exactly one point or one group, which is what they asked for.
    """

    schema = _camera_frame_schema()
    description = describe_semantics(schema, CurvePlot(AxisRef.point("frame")))
    x_values = description.axes_offering("x")
    assert AxisRef.point("frame") in x_values
    assert AxisRef.repeat("repeat") in x_values
    assert AxisRef.cell_data("spatial-y") in x_values
    assert AxisRef.cell_data("spatial-x") in x_values
    group_values = description.axes_offering("group")
    assert AxisRef.repeat("repeat") in group_values
    assert description.axis_choices.count(AxisRef.point("frame")) == 1

def test_non_series_kinds_keep_degenerate_axes_where_legitimate() -> None:
    schema = _camera_frame_schema()
    description = describe_semantics(
        schema,
        ImagePlot(AxisRef.cell_data("spatial-x"), AxisRef.cell_data("spatial-y")),
    )
    # An image is not a series; its axis domains stay the schema's own.
    x_values = description.axes_offering("x")
    assert AxisRef.cell_data("spatial-x") in x_values

def test_facet_grid_series_cell_offers_every_axis_for_x() -> None:
    points = mapped_domain_from_columns(
        {
            "x": np.arange(4.0),
            "row": np.array([0.0, 0.0, 1.0, 1.0]),
        }
    )
    schema = make_dataset_schema(repeat_domain(size=1), points)
    description = describe_semantics(
        schema,
        FacetGridPlot(AxisRef.point("row"), CurvePlot(AxisRef.point("x"))),
    )
    x_values = description.axes_offering("x")
    assert AxisRef.repeat("repeat") in x_values  # a size-one axis is still offered
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
            CurvePlot(AxisRef.point("x"), reduction=Reduction.MIN)
        )
        assert session.viewport == viewport
        assert session.selectors == ()
    finally:
        session.close()

def test_a_camera_cycle_names_its_rows_frames_and_nothing_else() -> None:
    """What an operator reads for the three shapes a bench produces."""

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    from zlc_data import OwnedSnapshot

    camera = describe_semantics(_camera_frame_schema(), CurvePlot(AxisRef.point("frame")))
    assert [ref for ref, _name in camera.fate_rows] == [
        AxisRef.repeat("repeat"),
        AxisRef.point("frame"),
        AxisRef.cell_data("spatial-y"),
        AxisRef.cell_data("spatial-x"),
    ]

    scan = make_dataset_schema(
        repeat_domain(size=2),
        mapped_domain_from_columns({"detuning": [0.0, 1.0, 2.0, 3.0]}),
        dtype=np.float64,
    )
    curve = describe_semantics(scan, CurvePlot(AxisRef.point("detuning")))
    assert [ref for ref, _name in curve.fate_rows][:2] == [
        AxisRef.repeat("repeat"),
        AxisRef.point("detuning"),
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

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    from zlc_data import OwnedSnapshot
    from zlc_plot import FacetGridPlot

    values = [0.0, 1.0, 2.0]
    rows = list(itertools.product(values, values))
    table = mapped_domain_from_columns(
        {"coil_x": [row[0] for row in rows], "coil_y": [row[1] for row in rows]}
    )
    schema = make_dataset_schema(
        repeat_domain(size=1),
        table,
        dtype=np.float64,
    )
    spec = FacetGridPlot(
        AxisRef.point("coil_x"), CurvePlot(AxisRef.point("coil_y"))
    )
    description = describe_semantics(schema, spec)
    assert description.fate(AxisRef.point("coil_x")) == "facet"
    assert description.fate_rows[0][0] == AxisRef.repeat("repeat")
    assert fate_field_name(AxisRef.point("coil_x")) in [
        name for _ref, name in description.fate_rows
    ]
    swapped = composed_spec(
        schema,
        spec,
        {fate_field_name(AxisRef.point("coil_x")): "x"},
    )
    assert swapped.cell.x == AxisRef.point("coil_x")
    assert swapped.facet == AxisRef.point("coil_y")

def test_a_whole_fate_table_moves_dense_image_roles_to_scan_axes_atomically() -> None:
    """Settling the old dense axes cannot overwrite pending scan-axis roles."""

    import itertools

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    from zlc_data import OwnedSnapshot

    domain = np.arange(10.0)
    rows = tuple(itertools.product(domain, domain, domain))
    table = mapped_domain_from_columns({
        "field.x": [row[0] for row in rows],
        "field.y": [row[1] for row in rows],
        "field.z": [row[2] for row in rows],
    })
    schema = make_dataset_schema(
        repeat_domain(size=20),
        table,
        cell_axes=(axis("pair", size=3), axis("site", size=35)),
        dtype=np.float64,
    )
    initial = FacetGridPlot(
        AxisRef.point("field.x"),
        ImagePlot(AxisRef.cell_data("site"), AxisRef.cell_data("pair")),
    )
    fates = {
        fate_field_name(AxisRef.point("field.y")): "y",
        fate_field_name(AxisRef.point("field.z")): "x",
        fate_field_name(AxisRef.cell_data("pair")): "reduce",
        fate_field_name(AxisRef.cell_data("site")): "reduce",
    }

    selected = composed_spec(schema, initial, fates)
    assert selected == composed_spec(
        schema, initial, dict(reversed(tuple(fates.items())))
    ), "a fate table must not depend on row iteration order"
    assert selected.facet == AxisRef.point("field.x")
    assert selected.cell.x == AxisRef.point("field.z")
    assert selected.cell.y == AxisRef.point("field.y")
    description = describe_semantics(schema, selected)
    assert description.fate(AxisRef.cell_data("pair")) == "reduce"
    assert description.fate(AxisRef.cell_data("site")) == "reduce"

def test_a_scan_dimension_of_a_long_sweep_still_offers_its_scope() -> None:
    """A thousand rows, ten coordinates: the pinnable set is the DISTINCT one.

    A grid dimension arrives as one value per scan ROW, and the scope cap
    used to be judged on that raw list -- so exactly the axes a scope
    exists for (the scan dimensions of any real sweep) never offered one.
    """

    from zlc_data import (
        AxisId,
        AxisSpec,
        DatasetSchema as Schema,
        DomainSpec,
        REPEAT,
        SCAN_POINT,
        SITE,
        SPATIAL_Y,
        ValidityContract,
        ValueSchema,
    )

    cells = tuple((i % 10, (i // 10) % 10, i // 100) for i in range(1000))
    axes = tuple(
        AxisSpec(
            AxisId(name),
            name,
            SCAN_POINT,
            10,
            tuple(float(index) for index in range(10)),
        )
        for name in ("ax", "ay", "az")
    )
    schema = Schema(
        DomainSpec(
            (20,),
            (AxisSpec(AxisId("cycle"), "cycle", REPEAT, 20),),
            (tuple(range(20)),),
        ),
        DomainSpec(
            (1000,),
            axes,
            tuple(
                tuple(int(cell[position]) for cell in cells)
                for position in range(3)
            ),
        ),
        DomainSpec(
            (34, 1024),
            (
                AxisSpec(AxisId("occ.site"), "site", SITE, 34),
                AxisSpec(AxisId("camera.y"), "spatial-y", SPATIAL_Y, 1024),
            ),
        ),
        ValueSchema(
            ValidityContract.components(AxisId("occ.site")),
            np.dtype("<f8"),
            "1",
        ),
    )
    description = describe_semantics(schema, CurvePlot(AxisRef.point("ax")))
    for dimension in ("ay", "az"):
        field = next(
            item
            for item in description.fields
            if item.name
            == fate_field_name(AxisRef.point(dimension))
        )
        assert field.cycle_choices is not None
        pins = [label for _value, label in field.cycle_choices]
        assert len(pins) == 10, (
            f"{dimension} must offer its ten coordinates as scopes, got {pins}"
        )
    spatial = description.field(fate_field_name(AxisRef.cell_data("camera.y")))
    assert spatial.cycle_choices is not None
    assert len(spatial.cycle_choices) == 1024
    assert all(not is_scope_fate(value) for value in spatial.choice_values)

def test_taking_an_occupied_role_swaps_fates_never_repairs() -> None:
    """The operator's rule: a conflicting choice TRADES fates, that's it.

    Promoting a reduced axis onto an occupied role leaves the displaced
    axis reduced -- no axis the operator never chose is drafted into any
    role, in either direction.
    """

    snapshot = _snapshot()
    schema = snapshot.block.schema
    spec = CurvePlot(AxisRef.point("x"))
    promoted = updated_spec(
        schema, spec, fate_field_name(AxisRef.point("row")), "x"
    )
    assert promoted.x == AxisRef.point("row")
    description = describe_semantics(schema, promoted)
    # the displaced axis inherits the taker's former fate: reduced
    assert description.fate(AxisRef.point("x")) == "reduce"

def test_vacating_a_required_role_is_a_state_not_a_repair() -> None:
    """Demoting the x holder leaves x VACANT: nothing is drafted, the
    edit raises the vacancy for the caller to present, and the message
    names the role the operator must fill to draw again."""

    snapshot = _snapshot()
    schema = snapshot.block.schema
    spec = CurvePlot(AxisRef.point("x"))
    with pytest.raises(SemanticVacancy) as caught:
        updated_spec(
            schema, spec, fate_field_name(AxisRef.point("x")), "reduce"
        )
    assert caught.value.role == "x"
    assert "'x'" in str(caught.value)

def test_two_occupied_roles_trade_places_in_one_edit() -> None:
    snapshot = _snapshot()
    schema = snapshot.block.schema
    spec = ImagePlot(AxisRef.point("x"), AxisRef.point("row"))
    swapped = updated_spec(
        schema, spec, fate_field_name(AxisRef.point("row")), "x"
    )
    assert swapped.x == AxisRef.point("row")
    assert swapped.y == AxisRef.point("x")

def test_a_pin_the_run_no_longer_has_never_reaches_the_data() -> None:
    """A scope fate names a COORDINATE, and this schema decides.

    Composing is the one place a fate becomes a scope term, and it did
    not check.  A pin left from an earlier run -- another scan plan,
    another frame count -- composed cleanly and detonated later inside
    restrict_snapshot ("coordinate selection is empty on axis ..."),
    from the data layer, with a plot already on screen.  A pin with no
    referent says nothing about this schema: its row holds no fate, and
    the vacancy rules speak for what is left.
    """

    snapshot = _snapshot()
    schema = snapshot.block.schema
    name = fate_field_name(AxisRef.point("row"))

    kept = updated_spec(schema, HistogramPlot(), name, scope_fate(1.0))
    assert kept.scope == ((AxisRef.point("row"), 1.0),)

    dropped = updated_spec(schema, HistogramPlot(), name, scope_fate(7.5))
    assert dropped.scope == (), (
        "a coordinate this run does not have must not become a scope term"
    )
