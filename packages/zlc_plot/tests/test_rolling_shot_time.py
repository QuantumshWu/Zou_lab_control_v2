"""A rolling plot may place its shots along the time they were taken: an x fate."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from data_factory import make_dataset_schema, make_snapshot, mapped_domain_from_columns, repeat_domain
from zlc_data import AxisId, AxisSpec, COMPONENT, DomainSpec, PRIMARY_INDEX, SHOT_TIME
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID, SHOT_TIME_AXIS_ID
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot, PlotKind, PlotSession, Reduction, RollingPlot
from zlc_plot.figure_artifact import decode_plot_recipe, encode_plot_recipe
from zlc_plot.semantics import describe_semantics, fate_field_name, projection_scope, updated_spec, scope_fate, composed_spec


def _stamped_history():
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns(
            {"source index": [-2, -1, 0], "shot time": [0.0, 0.1, 0.25]},
            ids={"source index": str(PRIMARY_INDEX_AXIS_ID), "shot time": str(SHOT_TIME_AXIS_ID)},
            roles={"source index": PRIMARY_INDEX, "shot time": SHOT_TIME},
            units={"shot time": "s"},
        ),
        dtype=np.float64,
    )
    domain = schema.point_domain
    schema = replace(schema, point_domain=replace(domain, axes=(
        domain.axes[0], replace(domain.axes[1], coordinate_of=PRIMARY_INDEX_AXIS_ID)
    )))
    return make_snapshot(schema, np.asarray([[10.0, 20.0, 30.0]]), revision=0)


def test_the_shot_time_axis_is_an_x_fate_of_a_rolling_plot() -> None:
    """The window's shots are placed along the seconds they were taken at.

    The rolling plot counted its shots back from the newest and nothing
    else could be its x, although a stamped history says when each shot
    happened.  That is an axis of the dataset, so it is chosen the way
    every axis is: by its fate. Time remains run-relative, while the index
    coordinate counts back from the newest shot.
    """

    snapshot = _stamped_history()
    schema = snapshot.block.schema
    time_ref = AxisRef.point(SHOT_TIME_AXIS_ID.value)
    index_ref = AxisRef.point(PRIMARY_INDEX_AXIS_ID.value)
    coordinate_field = f"coordinate:point:{PRIMARY_INDEX_AXIS_ID}"
    fate_field = fate_field_name(index_ref)
    description = describe_semantics(schema, RollingPlot())
    assert len(description.fate_rows) == 1  # Repeat only; the record X is fixed.
    assert not description.declares(fate_field)
    assert description.field(coordinate_field).label == "X coordinate"
    assert not description.axes_offering("x")
    along_time = updated_spec(schema, RollingPlot(), coordinate_field, time_ref.axis_id)
    assert along_time == RollingPlot(x=time_ref, coordinates=(time_ref,))
    for fate in ("reduce", "group", "x", scope_fate(0.1)):
        with pytest.raises(KeyError):
            updated_spec(schema, along_time, fate_field, fate)
    # The shot index may be named as x as well: that is the default, spelled out.
    along_index = updated_spec(schema, along_time, coordinate_field, index_ref.axis_id)
    assert along_index == RollingPlot(x=index_ref)
    # A saved figure keeps which axis the shots roll along.
    recipe = encode_plot_recipe(
        along_time, parameters={}, size="2x2", viewport=None, classifier_thresholds=()
    )
    assert decode_plot_recipe(recipe)["spec"] == along_time
    # Kind and Facet nesting do not erase a chosen coordinate when its fate
    # becomes Pool/Reduce and no x/group role carries that choice for it.
    histogram = updated_spec(schema, along_time, "kind", PlotKind.HISTOGRAM)
    assert histogram == HistogramPlot(coordinates=(time_ref,))
    grid = FacetGridPlot(None, HistogramPlot(), coordinates=histogram.coordinates)
    assert decode_plot_recipe(encode_plot_recipe(
        grid, parameters={}, size="2x2", viewport=None, classifier_thresholds=()
    ))["spec"] == grid
    # A curve walking the shot index keeps every shot: the twin axis is not
    # pinned to its latest coordinate behind the index's back.
    walking = CurvePlot(index_ref, reduction=Reduction.LAST)
    assert time_ref not in dict(projection_scope(schema, walking))
    # And the mirror: a curve walking the shot time keeps every shot too.
    along = CurvePlot(time_ref, reduction=Reduction.LAST)
    assert index_ref not in dict(projection_scope(schema, along))

    pinned = HistogramPlot(scope=((index_ref, -1),))
    chosen = updated_spec(schema, pinned, coordinate_field, time_ref.axis_id)
    assert chosen.scope == ((time_ref, 0.1),)
    # A saved full table's Scope is already in its selected coordinate.
    restored = composed_spec(schema, HistogramPlot(), {
        coordinate_field: time_ref.axis_id, fate_field: scope_fate(0.0),
    })
    assert restored.scope == ((time_ref, 0.0),)
    assert updated_spec(schema, restored, coordinate_field, index_ref.axis_id).scope == ((index_ref, -2),)

    # Coordinate selection changes only the coordinate, never the family's
    # fate. A complete replay must not build a conflicting intermediate x.
    component = AxisSpec(AxisId("component"), "component", COMPONENT, 2)
    component_ref = AxisRef.cell_data(component.axis_id.value)
    matrix_schema = replace(schema, cell_domain=DomainSpec((2,), (component,)))
    for original in (
        CurvePlot(component_ref, group=index_ref),
        CurvePlot(index_ref, group=component_ref),
        ImagePlot(index_ref, component_ref),
        HistogramPlot(reduced=(index_ref,)),
        FacetGridPlot(index_ref, HistogramPlot()),
    ):
        before = describe_semantics(matrix_schema, original).field(fate_field).value
        switched = updated_spec(matrix_schema, original, coordinate_field, time_ref.axis_id)
        description = describe_semantics(matrix_schema, switched)
        assert description.field(fate_field).value == before
        assert composed_spec(matrix_schema, switched, description.values) == switched
        assert updated_spec(matrix_schema, switched, coordinate_field, index_ref.axis_id) == original
    grouped = RollingPlot(group=time_ref, coordinates=(time_ref,))
    # An already-mounted invalid Group is repairable with the same complete
    # table used by ordinary configure, without an intermediate x/group clash.
    valid_table = describe_semantics(schema, along_time).values
    assert composed_spec(schema, grouped, valid_table) == along_time
    assert updated_spec(schema, grouped, coordinate_field, index_ref.axis_id) == along_index
    with pytest.raises(ValueError, match="record|history|group|Group"):
        PlotSession(snapshot, grouped, parameters={"window": 3})

    session = PlotSession(snapshot, along_time, parameters={"window": 3})
    try:
        (series,) = session._payload.series
        assert series.x.label == "shot time"
        assert series.x.canonical_unit.symbol == "s"
        assert np.asarray(series.x.canonical).tolist() == [0.0, 0.1, 0.25]
        assert np.asarray(series.y.canonical).tolist() == [10.0, 20.0, 30.0]
    finally:
        session.close()

    # A late window stays at its real time, not outside an old [-span, 0]
    # viewport. Both the renderer and units use the payload coordinates.
    domain = schema.point_domain
    shifted = replace(schema, point_domain=replace(domain, axes=(
        domain.axes[0], replace(domain.axes[1], coordinates=(10.0, 10.1, 10.25)),
    )))
    session = PlotSession(make_snapshot(shifted, np.asarray([[10.0, 20.0, 30.0]]), revision=1),
                          along_time, parameters={"window": 3, "x_display_unit": "ms"})
    try:
        limits = session.describe_display().limits.x
        assert 9900 < limits.low <= 10000 and 10250 <= limits.high < 10400
        assert session._renderer.primary_axes.get_xlabel() == 'shot time (ms)'
    finally:
        session.close()

    for spec in (RollingPlot(), along_index):
        session = PlotSession(snapshot, spec, parameters={"window": 3})
        try:
            (series,) = session._payload.series
            assert series.x.label == "Shots from latest"
            assert np.asarray(series.x.canonical).tolist() == [-2.0, -1.0, 0.0]
        finally:
            session.close()
