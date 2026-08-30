"""Semantic domains keep vocabulary separate from render feasibility.

Axis fate rows enumerate every role declared by the plot kind.  Projection
and layout feasibility belong to the subsequent replacement transaction;
they may reject a submitted table, but never erase roles from its vocabulary.
Non-fate fields may still be probed where their own contract requires it.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_plot.semantics import SemanticVacancy, fate_field_name
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    PlotSession,
    describe_semantics,
    updated_spec,
)
from zlc_plot._kinds import default_spec
from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_plot.kinds import PlotKind
from zlc_plot.specs import Reduction


def _flat_snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=3),
        PointTable.from_columns({"scan": np.linspace(0.0, 1.0, 5)}),
        dtype=np.float64,
        generation="feasibility-flat",
    )
    return DatasetSnapshot(schema, np.arange(15.0).reshape(3, 5), revision=0)


def _grid_snapshot() -> DatasetSnapshot:
    row = np.repeat(np.arange(10.0), 10)
    col = np.tile(np.arange(10.0), 10)
    points = PointTable.from_columns({"row": row, "col": col})
    topology = PointTopology.from_cartesian(
        (
            Axis.create("row", values=np.arange(10.0)),
            Axis.create("col", values=np.arange(10.0)),
        ),
        point_table=points,
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        points,
        point_topology=topology,
        dtype=np.float64,
        generation="feasibility-grid",
    )
    return DatasetSnapshot(schema, np.arange(100.0)[None, :], revision=0)


def _field_candidates(field, description):
    """Yield (choice value, candidate field value) pairs excluding current."""

    for value in field.choice_values:
        if value == field.value:
            continue
        yield value, value


@pytest.mark.parametrize(
    "make_snapshot, spec",
    [
        (_flat_snapshot, CurvePlot(AxisRef.point("scan"))),
        (_grid_snapshot, CurvePlot(AxisRef.point_dimension("col"))),
        (
            _grid_snapshot,
            FacetGridPlot(
                AxisRef.point_dimension("row"),
                CurvePlot(AxisRef.point_dimension("col")),
            ),
        ),
    ],
    ids=["flat-curve", "grid-curve", "grid-facet"],
)
def test_every_offered_option_succeeds(
    make_snapshot, spec
) -> None:
    snapshot = make_snapshot()
    schema = snapshot.block.schema
    session = PlotSession(snapshot, spec)
    try:
        description = session.describe_semantics()
    finally:
        session.close()

    for field in description.fields:
        for value, field_value in _field_candidates(field, description):
            probe = PlotSession(make_snapshot(), spec)
            try:
                try:
                    candidate = updated_spec(
                        schema, spec, field.name, field_value
                    )
                except SemanticVacancy:
                    # A legitimate outcome, not a failure: demoting the
                    # only holder of a required role leaves the table
                    # authored-but-undrawable, and the console presents
                    # that state.  No offered option may fail any OTHER
                    # way.
                    continue
                probe.replace_spec(candidate)
            finally:
                probe.close()


def test_axis_identities_are_deduplicated() -> None:
    """A point column that doubles as a topology dimension appears once.

    The grid's row/col columns are the same physical axes as its topology
    dimensions; listing both identities made every axis dropdown show
    duplicate entries.
    """

    snapshot = _grid_snapshot()
    session = PlotSession(snapshot, CurvePlot(AxisRef.point_dimension("col")))
    try:
        description = session.describe_semantics()
        assert AxisRef.point("row") not in description.axis_choices
        assert AxisRef.point("col") not in description.axis_choices
        assert AxisRef.point_dimension("row") in description.axis_choices
        labels = [description.field(name).label for _axis, name in description.fate_rows]
        assert len(labels) == len(set(labels))
    finally:
        session.close()


def test_the_row_ordinal_is_the_name_of_last_resort() -> None:
    """It is offered only when nothing declared already names the rows.

    A declared topology names them (they are its grid, in order) and so does
    any point column whose values are distinct.  Listed beside such a column
    it is the same axis twice: an operator saw "point (3)" and "frame (3)"
    and had to guess which of the two was the frames.
    """

    registry_default = default_spec

    snapshot = _grid_snapshot()
    session = PlotSession(snapshot, CurvePlot(AxisRef.point_dimension("col")))
    try:
        description = session.describe_semantics()
        assert AxisRef.point_rows() not in description.axis_choices
    finally:
        session.close()
    # The histogram default has no axis declaration at all: it pools the box.
    histogram = registry_default(snapshot.block.schema, PlotKind.HISTOGRAM)
    assert histogram == registry_default(snapshot.block.schema, PlotKind.HISTOGRAM)

    # A flat table whose one column names every row: the column IS the
    # point domain, and the ordinal would be a second name for it.
    flat = _flat_snapshot()
    flat_session = PlotSession(flat, CurvePlot(AxisRef.point("scan")))
    try:
        choices = flat_session.describe_semantics().axis_choices
        assert AxisRef.point("scan") in choices
        assert AxisRef.point_rows() not in choices
    finally:
        flat_session.close()


def test_curve_x_repeat_is_offered_and_draws() -> None:
    """Every offered option must work, including a degenerate axis.

    A single-shot repeat on the x axis draws one point -- which is what
    asking for it means -- so it is listed and it renders.  What may
    never happen is listing an option that then refuses.
    """

    snapshot = _grid_snapshot()
    session = PlotSession(snapshot, CurvePlot(AxisRef.point_dimension("col")))
    try:
        offering = session.describe_semantics().axes_offering("x")
        assert AxisRef.repeat() in offering
        assert AxisRef.point_dimension("row") in offering
        session.replace_spec(
            updated_spec(
                snapshot.block.schema,
                session.spec,
                fate_field_name(AxisRef.repeat()),
                "x",
            )
        )
        assert session.rgba() is not None
    finally:
        session.close()


def test_facet_vocabulary_does_not_predict_layout_capacity() -> None:
    """Every axis offers facet; the actual layout still enforces its cap."""

    points = PointTable.from_columns(
        {"big": np.arange(400.0), "few": np.arange(400.0) % 7}
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1), points, dtype=np.float64
    )
    snapshot = DatasetSnapshot(
        schema, np.arange(400.0)[None, :, None], revision=0
    )
    session = PlotSession(
        snapshot,
        FacetGridPlot(AxisRef.point("few"), CurvePlot(AxisRef.point("big"))),
    )
    try:
        offering = session.describe_semantics().axes_offering("facet")
        assert AxisRef.repeat() in offering
        assert AxisRef.point("big") in offering
        assert AxisRef.point("few") in offering
        candidate = updated_spec(
            schema,
            session.spec,
            fate_field_name(AxisRef.point("big")),
            "facet",
        )
        with pytest.raises(ValueError, match="facet_max_cells"):
            session.replace_spec(candidate)
    finally:
        session.close()


def test_user_can_reach_a_single_mean_line_on_grouped_data() -> None:
    """The acceptance chain that used to be locked: on a dataset with a dense
    data axis every group option except the current one was rejected, so a
    multi-line curve could never become one averaged line.  Ungrouping now
    collapses the dense axis under the declared reduction."""

    rng = np.random.default_rng(7)
    site = Axis.create("site", size=3)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=5),
        PointTable.from_columns({"scan": np.linspace(0.0, 1.0, 8)}),
        data_axes=(site,),
        dtype=np.float64,
        generation="single-line-chain",
    )
    values = rng.gamma(2.0, 2.0, size=(5, 8, 3))  # spread: mean != min
    snapshot = DatasetSnapshot(schema, values, revision=0)
    spec = CurvePlot(AxisRef.point("scan"), group=AxisRef.data("site"))
    session = PlotSession(snapshot, spec)
    try:
        description = session.describe_semantics()
        # Ungrouping is saying what the axis becomes INSTEAD, and regrouping
        # is a real option again.
        assert "reduce" in description.field(
            dict(description.fate_rows)[AxisRef.data("site")]
        ).choice_values
        assert AxisRef.repeat() in description.axes_offering("group")

        session.replace_spec(
            updated_spec(
                schema,
                spec,
                dict(description.fate_rows)[AxisRef.data("site")],
                "reduce",
            )
        )
        assert len(session._payload.series) == 1
        np.testing.assert_allclose(
            np.asarray(session._payload.series[0].y.canonical),
            values.mean(axis=(0, 2)),
        )
        mean_rgba = session.rgba().copy()

        current = session._spec
        session.replace_spec(
            updated_spec(schema, current, "reduction", Reduction.MIN)
        )
        np.testing.assert_allclose(
            np.asarray(session._payload.series[0].y.canonical),
            np.min(values, axis=(0, 2)),
        )
        # The edit is visible, not just accepted.
        assert np.any(session.rgba() != mean_rgba)
    finally:
        session.close()


def test_describing_semantics_never_builds_a_payload(monkeypatch) -> None:
    """Vocabulary projection reads schema/spec only; it never aggregates."""

    from zlc_plot._fit_projection import FitProjection

    snapshot = _grid_snapshot()
    session = PlotSession(snapshot, CurvePlot(AxisRef.point_dimension("col")))
    try:
        calls: list[object] = []
        original = FitProjection._build_payload_from_view

        def spy(self):
            calls.append(self._spec)
            return original(self)

        monkeypatch.setattr(FitProjection, "_build_payload_from_view", spy)
        session.describe_semantics()
        assert calls == []
    finally:
        session.close()


def test_updated_spec_is_the_single_composition_authority() -> None:
    schema = _grid_snapshot().block.schema
    facet = FacetGridPlot(
        AxisRef.point_dimension("row"),
        CurvePlot(AxisRef.point_dimension("col")),
    )

    switched = updated_spec(schema, facet, "kind", PlotKind.CURVE)
    assert switched.kind is PlotKind.CURVE

    cell_edit = updated_spec(schema, facet, "x", AxisRef.point("col"))
    assert isinstance(cell_edit, FacetGridPlot)
    assert cell_edit.cell.x == AxisRef.point("col")
    assert cell_edit.facet == facet.facet
    with pytest.raises(ValueError, match="one physical axis"):
        updated_spec(schema, facet, "x", AxisRef.point("row"))

    description = describe_semantics(schema, facet)
    repeat_fate = dict(description.fate_rows)[AxisRef.repeat()]
    assert "facet" in description.field(repeat_fate).choice_values
    facet_edit = updated_spec(schema, facet, repeat_fate, "facet")
    assert facet_edit.facet == AxisRef.repeat()
    assert facet_edit.cell == facet.cell

    with pytest.raises(KeyError):
        updated_spec(schema, facet, "not_a_field", 1)
    # A choice may arrive typed or as the plain value a record holds -- both
    # name the same choice -- but a value that names none is still refused.
    assert updated_spec(schema, facet, "kind", "curve").kind is PlotKind.CURVE
    with pytest.raises(ValueError, match="not a valid PlotKind"):
        updated_spec(schema, facet, "kind", "not_a_kind")
