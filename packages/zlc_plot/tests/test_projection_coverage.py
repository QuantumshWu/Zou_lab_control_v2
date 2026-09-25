from __future__ import annotations

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)

from zlc_data import OwnedSnapshot
from zlc_data import PRIMARY_INDEX
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from zlc_plot import PlotSession
from zlc_plot.data_view import DataView
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import CurvePlot, FacetGridPlot, ImagePlot, Reduction

def _snapshot(*, cell_axes=(), points=None, values=None) -> OwnedSnapshot:
    points = {"x": [0.0, 1.0]} if points is None else points
    point_domain = mapped_domain_from_columns(points)
    schema = make_dataset_schema(
        repeat_domain(size=2),
        point_domain,
        cell_axes=tuple(cell_axes),
        dtype=np.float64,
    )
    if values is None:
        values = np.arange(np.prod(schema.physical_shape), dtype=np.float64).reshape(schema.physical_shape)
    return make_snapshot(schema, values, revision=0)

@pytest.mark.parametrize("reduction", (Reduction.MEAN, Reduction.MIN, Reduction.LAST))
@pytest.mark.parametrize("hole", (False, True))
def test_curve_collapses_an_unassigned_data_axis_under_the_declared_reduction(reduction, hole) -> None:
    """Every axis has one fate; an axis assigned to none of x/group/facet is
    pooled by the projection's explicit reduction, exactly like repeat.
    Last is the panel's Scope on those axes, so it is read off a panel."""

    scan = axis("scan", values=[1.0, 0.0])
    snapshot = _snapshot(cell_axes=(scan,))
    values = np.asarray(snapshot.block.values)  # (R=2, P=2, scan=2)
    valid = np.ones(values.shape, dtype=bool)
    valid[-1, 0, :] = not hole
    snapshot = make_snapshot(snapshot.block.schema, values, revision=0, validity=valid)
    session = PlotSession(snapshot, CurvePlot(AxisRef.point("x"), reduction=reduction))
    try:
        series = session._payload.series[0]
    finally:
        session.close()
    selected = np.where(valid, values, np.nan)
    expected = {
        Reduction.MEAN: lambda: np.nanmean(selected, axis=(0, 2)),
        Reduction.MIN: lambda: np.nanmin(selected, axis=(0, 2)),
        Reduction.LAST: lambda: selected[-1, :, -1],
    }[reduction]()
    np.testing.assert_allclose(
        np.where(series.valid, series.y.canonical, np.nan),
        expected,
    )
    expected_counts = valid[-1, :, -1].astype(int) if reduction is Reduction.LAST else valid.sum(axis=(0, 2))
    np.testing.assert_array_equal(series.valid, expected_counts > 0)

def test_two_point_axes_directly_define_an_image() -> None:
    snapshot = _snapshot(points={"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]})
    payload = DataView(snapshot).image(AxisRef.point("x"), AxisRef.point("y"))
    values = np.asarray(snapshot.block.values)[..., 0]
    expected = np.asarray(
        [
            [values[:, 0].mean(), values[:, 1].mean()],
            [values[:, 2].mean(), values[:, 3].mean()],
        ]
    )
    np.testing.assert_allclose(payload.z.canonical, expected)

@pytest.mark.parametrize("reduction", (Reduction.MEAN, Reduction.LAST))
def test_image_collapses_an_unassigned_data_axis_under_the_declared_reduction(reduction) -> None:
    x = axis("x_data", values=[0.0, 1.0])
    y = axis("y_data", values=[0.0, 1.0])
    scan = axis("scan", values=[1.0, 0.0])
    snapshot = _snapshot(
        cell_axes=(x, y, scan),
        points={"point": [0.0]},
        values=np.arange(2 * 1 * 2 * 2 * 2, dtype=np.float64).reshape(2, 1, 2, 2, 2),
    )
    session = PlotSession(snapshot, ImagePlot(
        AxisRef.cell_data("x_data"), AxisRef.cell_data("y_data"), reduction=reduction,
    ))
    try:
        payload = session._payload
    finally:
        session.close()
    values = np.asarray(snapshot.block.values)  # (R, P=1, x, y, scan)
    expected = values[-1, 0, :, :, -1] if reduction is Reduction.LAST else values.mean(axis=(0, 1, 4))
    grid = np.asarray(payload.z.canonical)
    # The projection's (x, y) grid is indexed [y, x] for rendering.
    np.testing.assert_allclose(grid, expected.T)

def test_facet_curve_cell_pools_the_point_domain_too() -> None:
    """The same rule inside a facet cell: kinds do not each get a policy."""

    from zlc_plot.specs import CurvePlot, FacetGridPlot

    scan = axis("scan", values=[0.0, 1.0])
    snapshot = _snapshot(cell_axes=(scan,))  # (R=2, P=2, scan=2)
    spec = FacetGridPlot(AxisRef.repeat("repeat"), CurvePlot(AxisRef.cell_data("scan")))
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

    point_domain = mapped_domain_from_columns(
        {"source index": [-4, -3, -2, -1, 0]},
        ids={"source index": str(PRIMARY_INDEX_AXIS_ID)},
        roles={"source index": PRIMARY_INDEX},
    )
    schema = make_dataset_schema(
        repeat_domain(size=1),
        point_domain,
        cell_axes=(
            axis("y", values=[0.0, 1.0]),
            axis("x", values=[0.0, 1.0, 2.0]),
        ),
        dtype=np.float64,
    )
    snapshot = make_snapshot(schema, np.ones(schema.physical_shape), revision=0)
    spec = FacetGridPlot(
        AxisRef.point(str(PRIMARY_INDEX_AXIS_ID)),
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
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
