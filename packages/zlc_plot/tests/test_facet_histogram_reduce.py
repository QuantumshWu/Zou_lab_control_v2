"""A faceted histogram honours the cell's own vocabulary.

``reduced`` is a legal fate on a histogram, and a grid of histogram cells is
still histograms: within each cell, collapse the named axes under the
reduction, then bin what is left.  The facet build never did it -- the
cell's ``reduced`` had no way out of the spec, since ``facet()`` took only
bins and uncertainty -- so the fate row was offered, accepted, written back,
and the picture did not move.

The domain goes with it.  A reduced pool is per-group statistics, whose
spread is narrower than the raw pool's by construction, so edges taken from
the raw values put every cell in a couple of bins out of a dozen.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    REPEAT,
    SCAN_POINT,
    SITE,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_plot import AxisRef, FacetGridPlot, HistogramPlot
from zlc_plot.data_view import DataView, DataViewError
from zlc_plot.specs import Reduction

_REPEATS, _ROWS, _SITES = 4, 2, 3


def _view(values: np.ndarray | None = None) -> DataView:
    schema = DatasetSchema(
        AxisSpec(AxisId("t.repeat"), "repeat", REPEAT, _REPEATS, tuple(range(_REPEATS))),
        PointTable(
            _ROWS,
            (
                PointColumn(
                    AxisId("p.frame"),
                    "frame",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    tuple(float(row) for row in range(_ROWS)),
                ),
            ),
        ),
        None,
        ValueSchema(
            (AxisSpec(AxisId("v.site"), "site", SITE, _SITES, tuple(range(_SITES))),),
            ValidityContract.value(),
            np.dtype("float64"),
            "count",
        ),
    )
    if values is None:
        values = np.arange(_REPEATS * _ROWS * _SITES, dtype=float).reshape(
            _REPEATS, _ROWS, _SITES
        )
    block = DataBlock(
        BlockId("d"),
        DatasetRevision(1),
        values,
        CellValidity(np.ones((_REPEATS, _ROWS), dtype=bool)),
        schema,
        None,
    )
    return DataView(OwnedSnapshot(block.ref(StreamGenerationId("g")), block))


def _grid(**cell: object) -> FacetGridPlot:
    return FacetGridPlot(facet=AxisRef.point("p.frame"), cell=HistogramPlot(**cell))


def _pools(view: DataView, grid: FacetGridPlot) -> tuple[np.ndarray, ...]:
    plan = view._facet_histogram_plan(grid, 1)
    return tuple(np.sort(pool) for pool in plan.pools)


def test_each_cell_bins_its_own_means_over_the_reduced_axis() -> None:
    """One value per surviving group, and the groups are the cell's."""

    values = np.arange(_REPEATS * _ROWS * _SITES, dtype=float).reshape(
        _REPEATS, _ROWS, _SITES
    )
    view = _view(values)
    grid = _grid(reduced=(AxisRef.repeat(),), reduction=Reduction.MEAN)

    expected = values.mean(axis=0)  # one mean per (row, site)
    for row, pool in enumerate(_pools(view, grid)):
        np.testing.assert_allclose(pool, np.sort(expected[row]))

    # And that is what the cells actually count.
    data = view.facet(grid, bins=np.linspace(8.0, 15.0, 8))
    assert [int(np.asarray(c.payload.counts).sum()) for c in data.cells] == [
        _SITES,
        _SITES,
    ]


def test_reducing_a_data_axis_leaves_one_value_per_shot_and_row() -> None:
    values = np.arange(_REPEATS * _ROWS * _SITES, dtype=float).reshape(
        _REPEATS, _ROWS, _SITES
    )
    view = _view(values)
    grid = _grid(reduced=(AxisRef.data("v.site"),), reduction=Reduction.MEAN)
    for row, pool in enumerate(_pools(view, grid)):
        np.testing.assert_allclose(pool, np.sort(values[:, row, :].mean(axis=1)))


def test_the_shared_domain_covers_what_the_cells_bin() -> None:
    """Edges from the raw pool put a reduced grid in two bins of twelve."""

    rng = np.random.default_rng(11)
    means = np.linspace(100.0, 105.0, _ROWS * _SITES).reshape(_ROWS, _SITES)
    values = means[None, :, :] + rng.normal(0.0, 20.0, (_REPEATS, _ROWS, _SITES))
    view = _view(values)
    grid = _grid(reduced=(AxisRef.repeat(),), reduction=Reduction.MEAN)

    pool, valid = view.facet_histogram_pool(grid)
    assert valid.all()
    assert pool.size == _ROWS * _SITES
    assert float(pool.min()) >= float(values.min())
    # The pool is the means, and they are far tighter than the raw samples.
    assert float(pool.max()) - float(pool.min()) < 0.5 * (
        float(values.max()) - float(values.min())
    )

    # Without a reduction the pool is simply every sample, which costs
    # nothing to say -- the cells partition them.
    plain, plain_valid = view.facet_histogram_pool(_grid())
    assert plain.size == values.size
    assert np.broadcast_to(plain_valid, values.shape).all()


def test_a_facet_axis_cannot_also_be_reduced() -> None:
    """The cells it names would be collapsed into one."""

    view = _view()
    with pytest.raises(DataViewError, match="cannot also be reduced"):
        view.validate_facet(
            FacetGridPlot(
                facet=AxisRef.repeat(),
                cell=HistogramPlot(
                    reduced=(AxisRef.repeat(),), reduction=Reduction.MEAN
                ),
            )
        )


def test_a_window_pools_the_last_shots_into_every_cell() -> None:
    """Ignored before, so a window of two drew the same picture as one."""

    values = np.arange(_REPEATS * _ROWS * _SITES, dtype=float).reshape(
        _REPEATS, _ROWS, _SITES
    )
    view = _view(values)
    grid = _grid()
    edges = np.linspace(-1.0, float(values.max()) + 1.0, 12)

    whole = view.facet(grid, bins=edges, window=_REPEATS)
    two = view.facet(grid, bins=edges, window=2)
    counts = lambda data: [int(np.asarray(c.payload.counts).sum()) for c in data.cells]
    assert counts(whole) == [_REPEATS * _SITES] * _ROWS
    assert counts(two) == [2 * _SITES] * _ROWS


def test_a_window_and_a_reduction_compose() -> None:
    """The last N shots, averaged per group, binned."""

    values = np.arange(_REPEATS * _ROWS * _SITES, dtype=float).reshape(
        _REPEATS, _ROWS, _SITES
    )
    view = _view(values)
    grid = _grid(reduced=(AxisRef.repeat(),), reduction=Reduction.MEAN)
    plan = view._facet_histogram_plan(grid, 2)
    expected = values[-2:].mean(axis=0)
    for row, pool in enumerate(plan.pools):
        np.testing.assert_allclose(np.sort(pool), np.sort(expected[row]))


def test_the_grid_and_the_single_panel_reduce_the_same_way() -> None:
    """A faceted cell is the standalone kind, one slice at a time."""

    rng = np.random.default_rng(5)
    values = rng.normal(50.0, 5.0, (_REPEATS, _ROWS, _SITES))
    view = _view(values)
    refs = (AxisRef.repeat(),)

    whole, whole_valid = view.histogram_pool(
        reduce_axes=refs, aggregation=Reduction.MEAN
    )
    single = np.sort(np.asarray(whole)[np.asarray(whole_valid)].reshape(-1))

    grid_pool, _ = view.facet_histogram_pool(
        _grid(reduced=refs, reduction=Reduction.MEAN)
    )
    np.testing.assert_allclose(np.sort(grid_pool), single)


def _two_column_view(values: np.ndarray) -> DataView:
    """A frame x detuning scan: facet by one coordinate, reduce the other."""

    rows = values.shape[1]
    schema = DatasetSchema(
        AxisSpec(AxisId("t.repeat"), "repeat", REPEAT, values.shape[0], tuple(range(values.shape[0]))),
        PointTable(
            rows,
            (
                PointColumn(
                    AxisId("p.frame"), "frame", SCAN_POINT, PointColumn.NUMERIC,
                    tuple(float(row % 2) for row in range(rows)),
                ),
                PointColumn(
                    AxisId("p.detuning"), "detuning", SCAN_POINT, PointColumn.NUMERIC,
                    tuple(float(row // 2) for row in range(rows)),
                ),
            ),
        ),
        None,
        ValueSchema(
            (AxisSpec(AxisId("v.site"), "site", SITE, values.shape[2], tuple(range(values.shape[2]))),),
            ValidityContract.value(),
            np.dtype("float64"),
            "count",
        ),
    )
    block = DataBlock(
        BlockId("d"),
        DatasetRevision(1),
        values,
        CellValidity(np.ones(values.shape[:2], dtype=bool)),
        schema,
        None,
    )
    return DataView(OwnedSnapshot(block.ref(StreamGenerationId("g")), block))


def test_a_reduced_point_coordinate_groups_the_rows_inside_each_cell() -> None:
    """The path a whole tensor axis does not take.

    Reducing "detuning" while facetting by "frame" cannot be a ufunc over an
    array axis: both coordinates live on the one point axis, so the surviving
    point axis is a REGROUPING of its rows and the identity has to be built
    per sample.  Whole-axis reductions take the cheaper route; this checks
    the other one still answers, against a hand computation.
    """

    repeats, rows, sites = 3, 6, 2  # frames {0,1} x detunings {0,1,2}
    values = np.arange(repeats * rows * sites, dtype=float).reshape(repeats, rows, sites)
    view = _two_column_view(values)
    grid = FacetGridPlot(
        facet=AxisRef.point("p.frame"),
        cell=HistogramPlot(
            reduced=(AxisRef.point("p.detuning"),), reduction=Reduction.MEAN
        ),
    )
    plan = view._facet_histogram_plan(grid, 1)
    assert len(plan.pools) == 2
    for frame in (0, 1):
        # Rows of this frame, averaged over the three detunings, per repeat
        # and per site -- so repeats x sites values survive in the cell.
        mine = values[:, frame::2, :].mean(axis=1)
        np.testing.assert_allclose(np.sort(plan.pools[frame]), np.sort(mine.reshape(-1)))


def test_both_reduction_routes_agree_on_a_reduction_they_share() -> None:
    """Reducing the repeat axis is expressible either way; they must match."""

    from zlc_plot.data_view import _aggregate_by_codes

    repeats, rows, sites = 4, 6, 2
    values = np.arange(repeats * rows * sites, dtype=float).reshape(repeats, rows, sites)
    view = _two_column_view(values)
    refs = (AxisRef.repeat(),)

    # The route the plan takes: a ufunc over the array axis.
    valid = np.ones(values.shape, dtype=bool)
    quick, present = view._collapse_axes(values, valid, refs, Reduction.MEAN)

    # The route a point-coordinate reduction is forced to take, on the same
    # reduction: one bucket identity per sample, scattered.
    dimensions, coordinates = view._reduction_plan(refs)
    buckets = view._reduction_buckets(dimensions, coordinates)
    scattered, counts = _aggregate_by_codes(
        values.reshape(-1),
        np.ones(values.size, dtype=bool),
        np.ascontiguousarray(buckets.codes).reshape(-1),
        buckets.count,
        Reduction.MEAN,
    )
    np.testing.assert_allclose(
        quick[present], scattered.reshape(buckets.shape)[counts.reshape(buckets.shape) > 0]
    )
