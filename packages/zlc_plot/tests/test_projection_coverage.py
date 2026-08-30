from __future__ import annotations

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_data import PRIMARY_INDEX
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from zlc_plot.data_view import DataView, DataViewError
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import FacetGridPlot, ImagePlot, Reduction


def _snapshot(*, data_axes=(), points=None, values=None) -> DatasetSnapshot:
    points = {"x": [0.0, 1.0]} if points is None else points
    point_table = PointTable.from_columns(points)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2),
        point_table,
        data_axes=tuple(data_axes),
        dtype=np.float64,
        generation="coverage-test",
    )
    if values is None:
        values = np.arange(np.prod(schema.shape), dtype=np.float64).reshape(schema.shape)
    return DatasetSnapshot(schema, values, revision=0)


def test_curve_collapses_an_unassigned_data_axis_under_the_declared_reduction() -> None:
    """Every axis has one fate; an axis assigned to none of x/group/facet is
    pooled by the projection's explicit reduction, exactly like repeat."""

    scan = Axis.create("scan", values=[0.0, 1.0])
    snapshot = _snapshot(data_axes=(scan,))
    payload = DataView(snapshot).curve(AxisRef.point("x"))
    values = np.asarray(snapshot.block.values)  # (R=2, P=2, scan=2)
    series = payload.series[0]
    np.testing.assert_allclose(
        np.asarray(series.y.canonical),
        [values[:, 0, :].mean(), values[:, 1, :].mean()],
    )
    # The pool is repeat x scan per plotted x value.
    assert tuple(series.counts) == (4, 4)


def test_curve_min_pools_the_unassigned_axis_too() -> None:
    scan = Axis.create("scan", values=[0.0, 1.0])
    snapshot = _snapshot(data_axes=(scan,))
    payload = DataView(snapshot).curve(
        AxisRef.point("x"), aggregation=Reduction.MIN
    )
    values = np.asarray(snapshot.block.values)
    np.testing.assert_allclose(
        np.asarray(payload.series[0].y.canonical),
        [np.min(values[:, 0, :]), np.min(values[:, 1, :])],
    )


def test_image_rejects_two_ordinary_point_coordinates_without_topology() -> None:
    snapshot = _snapshot(points={"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]})
    with pytest.raises(DataViewError, match="GridTopology"):
        DataView(snapshot).image(AxisRef.point("x"), AxisRef.point("y"))


def test_image_collapses_an_unassigned_data_axis_under_the_declared_reduction() -> None:
    x = Axis.create("x_data", values=[0.0, 1.0])
    y = Axis.create("y_data", values=[0.0, 1.0])
    scan = Axis.create("scan", values=[0.0, 1.0])
    snapshot = _snapshot(
        data_axes=(x, y, scan),
        points={"point": [0.0]},
        values=np.arange(2 * 1 * 2 * 2 * 2, dtype=np.float64).reshape(2, 1, 2, 2, 2),
    )
    payload = DataView(snapshot).image(AxisRef.data("x_data"), AxisRef.data("y_data"))
    values = np.asarray(snapshot.block.values)  # (R, P=1, x, y, scan)
    expected = values.mean(axis=(0, 1, 4))  # pool repeat + scan per (x, y) cell
    grid = np.asarray(payload.z.canonical)
    # The projection's (x, y) grid is indexed [y, x] for rendering.
    np.testing.assert_allclose(grid, expected.T)


def test_curve_pools_the_point_domain_like_every_other_unassigned_axis() -> None:
    """Pooling is ONE rule with no exceptions.

    The point axis used to be privileged: a curve that named no point
    coordinate was refused outright while every other unassigned axis
    pooled under the declared reduction.  R, P and D now meet the same
    fate, and "you are averaging your whole scan into one number" is a hint
    the editor gives, not a construction-time error.
    """

    snapshot = _snapshot()  # (R=2, P=2)
    payload = DataView(snapshot).curve(AxisRef.repeat())
    values = np.asarray(snapshot.block.values)
    series = payload.series[0]
    np.testing.assert_allclose(
        np.asarray(series.y.canonical),
        [values[0, :].mean(), values[1, :].mean()],
    )
    # Each plotted repeat pooled the whole 2-row point domain.
    assert tuple(series.counts) == (2, 2)


def test_facet_curve_cell_pools_the_point_domain_too() -> None:
    """The same rule inside a facet cell: kinds do not each get a policy."""

    from zlc_plot.specs import CurvePlot, FacetGridPlot

    scan = Axis.create("scan", values=[0.0, 1.0])
    snapshot = _snapshot(data_axes=(scan,))  # (R=2, P=2, scan=2)
    spec = FacetGridPlot(AxisRef.repeat(), CurvePlot(AxisRef.data("scan")))
    payload = DataView(snapshot).facet(spec)
    values = np.asarray(snapshot.block.values)
    assert len(payload.cells) == 2
    for repeat, cell in enumerate(payload.cells):
        np.testing.assert_allclose(
            np.asarray(cell.payload.series[0].y.canonical),
            values[repeat].mean(axis=0),
        )


def test_image_facets_do_not_apply_histogram_window_to_history_axis() -> None:
    """A Facet Image has no window control, so every retained cell is data."""

    point_table = PointTable.from_columns(
        {"source index": [-4, -3, -2, -1, 0]},
        ids={"source index": str(PRIMARY_INDEX_AXIS_ID)},
        roles={"source index": PRIMARY_INDEX},
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        point_table,
        data_axes=(
            Axis.create("y", values=[0.0, 1.0]),
            Axis.create("x", values=[0.0, 1.0, 2.0]),
        ),
        dtype=np.float64,
        generation="indexed-image-facets",
    )
    snapshot = DatasetSnapshot(schema, np.ones(schema.shape), revision=0)
    spec = FacetGridPlot(
        AxisRef.point(str(PRIMARY_INDEX_AXIS_ID)),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
    )

    cells = DataView(snapshot).facet(spec).cells
    assert [cell.label for cell in cells] == [
        "source index=-4",
        "source index=-3",
        "source index=-2",
        "source index=-1",
        "source index=0",
    ]
    assert all(np.all(cell.payload.valid) for cell in cells)


def test_validate_curve_refuses_a_text_x_that_cannot_be_plotted() -> None:
    """Whatever passes validate_curve projects, and whatever projects passed.

    The real-numeric requirement used to live at the two BUILD sites only,
    so a text coordinate passed validation and then raised on the first draw.
    """

    snapshot = _snapshot(points={"label": ["a", "b"]})
    view = DataView(snapshot)
    with pytest.raises(DataViewError, match="real numeric"):
        view.validate_curve(AxisRef.point("label"))
    with pytest.raises(DataViewError, match="real numeric"):
        view.curve(AxisRef.point("label"))
