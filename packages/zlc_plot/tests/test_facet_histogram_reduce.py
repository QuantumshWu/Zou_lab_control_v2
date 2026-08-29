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
