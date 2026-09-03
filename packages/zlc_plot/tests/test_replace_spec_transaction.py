"""A rejected ``replace_spec`` must leave the session exactly as it was.

Regression for the semantic-UX audit finding: layout-stage rejection (facet
cell cap) used to fire *after* the session state had been mutated, leaving a
half-committed session whose semantics reported the rejected spec and whose
rendering was frozen.  The transaction envelope now covers plan resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, curve
from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_plot.kinds import PlotKind

def _grid_session():
    row = np.repeat(np.arange(10.0), 10)
    col = np.tile(np.arange(10.0), 10)
    points = mapped_domain_from_columns({"row": row, "col": col, "x": np.arange(100.0)})
    schema = make_dataset_schema(
        repeat_domain(values=np.arange(1)),
        points,
        dtype=np.float64,
    )
    snapshot = make_snapshot(schema, np.arange(100.0)[None, :], revision=0)
    return curve(snapshot, AxisRef.point("x"))

def test_layout_rejected_replace_rolls_back_completely() -> None:
    session = _grid_session()
    try:
        baseline = session.rgba()
        # 100 point rows -> 100 cells against the 64-cell capacity.
        oversized = FacetGridPlot(
            AxisRef.point("x"),
            CurvePlot(AxisRef.point("col")),
        )
        with pytest.raises(ValueError, match="facet_max_cells"):
            session.replace_spec(oversized)

        description = session.describe_display()
        assert description.semantics.kind is PlotKind.CURVE
        assert description.semantics.x == AxisRef.point("x")
        assert description.semantics.facet is None

        # The session must stay fully operable: display edits, rendering and
        # a subsequent legal semantic replacement all work.
        session.set_labels(title="still alive")
        after = session.rgba()
        assert after.shape == baseline.shape

        accepted = session.replace_spec(
            FacetGridPlot(AxisRef.point("row"), CurvePlot(AxisRef.point("x")))
        )
        assert accepted.semantics.kind is PlotKind.FACET_GRID
    finally:
        session.close()

def test_projection_rejected_replace_is_untouched_precommit(logical_shape) -> None:
    session = _grid_session()
    try:
        # x names a coordinate the Point domain does not declare; the
        # projection layer rejects before any state is touched.  (x=repeat
        # is NOT a rejection any more: an axis this spec does not name pools
        # under the declared reduction, point axis included.)
        with pytest.raises(Exception):
            session.replace_spec(CurvePlot(AxisRef.point("undeclared")))
        description = session.describe_display()
        assert description.semantics.kind is PlotKind.CURVE
        assert description.semantics.x == AxisRef.point("x")
        session.set_labels(title="alive")
        # Derived from the plan, so a geometry correction is not a puzzle.
        assert session.rgba().shape == logical_shape()
    finally:
        session.close()
