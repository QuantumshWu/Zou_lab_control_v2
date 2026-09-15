"""A rolling plot may place its shots along the time they were taken: an x fate."""

from __future__ import annotations

import numpy as np

from data_factory import make_dataset_schema, make_snapshot, mapped_domain_from_columns, repeat_domain
from zlc_data import PRIMARY_INDEX, SHOT_TIME
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID, SHOT_TIME_AXIS_ID
from zlc_plot import AxisRef, CurvePlot, PlotSession, Reduction, RollingPlot
from zlc_plot.figure_artifact import decode_plot_recipe, encode_plot_recipe
from zlc_plot.semantics import describe_semantics, fate_field_name, projection_scope, updated_spec


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
    return make_snapshot(schema, np.asarray([[10.0, 20.0, 30.0]]), revision=0)


def test_the_shot_time_axis_is_an_x_fate_of_a_rolling_plot() -> None:
    """The window's shots are placed along the seconds they were taken at.

    The rolling plot counted its shots back from the newest and nothing
    else could be its x, although a stamped history says when each shot
    happened.  That is an axis of the dataset, so it is chosen the way
    every axis is: by its fate.  Along it, 0 is still the newest shot and
    the others lie as many seconds back as they were.
    """

    snapshot = _stamped_history()
    schema = snapshot.block.schema
    time_ref = AxisRef.point(SHOT_TIME_AXIS_ID.value)
    row = describe_semantics(schema, RollingPlot()).field(fate_field_name(time_ref))
    labels = dict(row.choices)
    assert row.value == "reduce" and labels["reduce"] == "(shot axis)"
    assert "x" in labels
    along_time = updated_spec(schema, RollingPlot(), fate_field_name(time_ref), "x")
    assert along_time == RollingPlot(x=time_ref)
    assert updated_spec(schema, along_time, fate_field_name(time_ref), "reduce") == RollingPlot()
    # The shot index may be named as x as well: that is the default, spelled out.
    index_ref = AxisRef.point(PRIMARY_INDEX_AXIS_ID.value)
    along_index = updated_spec(schema, along_time, fate_field_name(index_ref), "x")
    assert along_index == RollingPlot(x=index_ref)
    # A saved figure keeps which axis the shots roll along.
    recipe = encode_plot_recipe(
        along_time, parameters={}, size="2x2", viewport=None, classifier_thresholds=()
    )
    assert decode_plot_recipe(recipe)["spec"] == along_time
    # A curve walking the shot index keeps every shot: the twin axis is not
    # pinned to its latest coordinate behind the index's back.
    walking = CurvePlot(index_ref, reduction=Reduction.LAST)
    assert time_ref not in dict(projection_scope(schema, walking))

    session = PlotSession(snapshot, along_time, parameters={"window": 3})
    try:
        (series,) = session._payload.series
        assert series.x.canonical_unit.symbol == "s"
        assert series.x.label == "Seconds from latest"
        assert np.asarray(series.x.canonical).tolist() == [-0.25, -0.15, 0.0]
        assert np.asarray(series.y.canonical).tolist() == [10.0, 20.0, 30.0]
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
