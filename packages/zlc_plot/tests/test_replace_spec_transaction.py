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
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_plot.kinds import PlotKind


def _grid_session():
    row = np.repeat(np.arange(10.0), 10)
    col = np.tile(np.arange(10.0), 10)
    points = PointTable.from_columns({"row": row, "col": col, "x": np.arange(100.0)})
    topology = PointTopology.from_cartesian(
        (
            Axis.create("row", values=np.arange(10.0)),
            Axis.create("col", values=np.arange(10.0)),
        )
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", values=np.arange(1)),
        points,
        point_topology=topology,
        dtype=np.float64,
        generation="replace-transaction",
    )
    snapshot = DatasetSnapshot(schema, np.arange(100.0)[None, :], revision=0)
    return curve(snapshot, AxisRef.point("x"))


def test_layout_rejected_replace_rolls_back_completely() -> None:
    session = _grid_session()
    try:
        baseline = session.rgba()
        # 100 point rows -> 100 cells against the 64-cell capacity.
        oversized = FacetGridPlot(
            AxisRef.point_rows(),
            CurvePlot(AxisRef.point("x")),
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
            FacetGridPlot(AxisRef.point_dimension("row"), CurvePlot(AxisRef.point("x")))
        )
        assert accepted.semantics.kind is PlotKind.FACET_GRID
    finally:
        session.close()


def test_projection_rejected_replace_is_untouched_precommit(logical_shape) -> None:
    session = _grid_session()
    try:
        # x=repeat leaves the authored 100-row point domain unreferenced;
        # the projection layer rejects before any state is touched.
        with pytest.raises(Exception):
            session.replace_spec(CurvePlot(AxisRef.repeat()))
        description = session.describe_display()
        assert description.semantics.kind is PlotKind.CURVE
        assert description.semantics.x == AxisRef.point("x")
        session.set_labels(title="alive")
        # Derived from the plan, so a geometry correction is not a puzzle.
        assert session.rgba().shape == logical_shape()
    finally:
        session.close()
