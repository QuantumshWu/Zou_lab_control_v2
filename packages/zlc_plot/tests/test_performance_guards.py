from __future__ import annotations

import tracemalloc
from time import perf_counter

import numpy as np

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)

from zlc_data import OwnedSnapshot

import zlc_plot._fit_projection as fit_projection_module
import zlc_plot.data_view as data_view_module
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSession,
    Reduction,
    RollingPlot,
)
from zlc_plot.data_view import DataView

# These are intentionally named guards rather than hidden timing literals.
# They catch accidental full-tensor copies while leaving enough room for a
# shared CI worker's normal variance.
MAX_REPLACE_SPEC_SECONDS = 8.0
MAX_ROLLING_20_FRAME_SECONDS = 8.0

def _scan_snapshot(*, revision: int = 0, repeats: int = 5, points: int = 120, sites: int = 30) -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=repeats),
        mapped_domain_from_columns({"scan": np.linspace(0.0, 1.0, points)}),
        cell_axes=(axis("site", size=sites),),
        dtype=np.float64,
    )
    values = np.sin(np.linspace(0.0, 8.0, points))[None, :, None]
    values = np.broadcast_to(values, (repeats, points, sites)).copy()
    values += np.arange(repeats, dtype=float)[:, None, None] * 0.01
    return make_snapshot(schema, values, revision=revision)

def _large_dense_snapshot(
    height: int,
    width: int,
    *,
    points: int = 1,
    dtype=np.uint16,
    column_ramp: bool = False,
    invalid_first_point: bool = False,
    revision: int = 0,
) -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"batch": np.arange(points, dtype=float)}),
        cell_axes=(
            axis("row", values=np.arange(height, dtype=float)),
            axis("column", values=np.arange(width, dtype=float)),
        ),
        dtype=dtype,
    )
    if column_ramp:
        column = np.arange(width, dtype=dtype)
        values = np.broadcast_to(
            column.reshape(1, 1, 1, width),
            (1, points, height, width),
        ).copy()
    else:
        values = np.zeros((1, points, height, width), dtype=dtype)
    validity = None
    if invalid_first_point:
        validity = np.ones(values.shape, dtype=np.bool_)
        validity[:, 0] = False
    return make_snapshot(schema, values, revision=revision, validity=validity)

def test_replace_spec_and_rolling_projection_have_bounded_cost() -> None:
    snapshot = _scan_snapshot()
    session = PlotSession(
        snapshot,
        CurvePlot(AxisRef.point("scan"), group=AxisRef.cell_data("site")),
    )
    try:
        timings: dict[str, float] = {}
        start = perf_counter()
        session.replace_spec(
            CurvePlot(AxisRef.point("scan"), group=AxisRef.repeat("repeat"))
        )
        timings["group_to_repeat"] = perf_counter() - start

        start = perf_counter()
        session.replace_spec(
            CurvePlot(AxisRef.point("scan"), group=AxisRef.cell_data("site"))
        )
        timings["group_to_site"] = perf_counter() - start

        start = perf_counter()
        session.replace_spec(
            FacetGridPlot(
                AxisRef.cell_data("site"),
                CurvePlot(AxisRef.point("scan")),
            )
        )
        timings["kind_to_facet"] = perf_counter() - start
        assert all(value < MAX_REPLACE_SPEC_SECONDS for value in timings.values()), timings
    finally:
        session.close()

    rolling = PlotSession(
        snapshot,
        RollingPlot(group=AxisRef.cell_data("site")),
    )
    try:
        start = perf_counter()
        for revision in range(1, 21):
            rolling.update_data(_scan_snapshot(revision=revision))
        elapsed = perf_counter() - start
        assert elapsed < MAX_ROLLING_20_FRAME_SECONDS, elapsed
    finally:
        rolling.close()

def test_facet_cell_count_never_materializes_a_declared_domain(monkeypatch) -> None:
    """Counting a DECLARED facet domain reads axis-sized arrays only.

    ``facet_cell_count`` used to build ``np.arange`` over every ELEMENT
    (about 20 million for one 9x1200x1920 camera facet) plus full flat
    coordinate copies just to COUNT a declared domain -- measured as 2.63 s
    of a 3.1 s semantic sweep.  The declared paths must answer without one
    element pass; only the undeclared point-axis fallback may still
    walk elements.
    """

    bias = [float(v) for v in range(9)]
    table = mapped_domain_from_columns({"bias": bias})
    schema = make_dataset_schema(
        repeat_domain(size=3),
        table,
        cell_axes=(
            axis("sy", values=tuple(float(v) for v in range(120))),
            axis("sx", values=tuple(float(v) for v in range(160))),
        ),
        dtype=np.uint8,
    )
    values = np.zeros((3, 9, 120, 160), dtype=np.uint8)
    view = DataView(make_snapshot(schema, values, revision=0))
    cell = ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy"))

    element_passes: list[object] = []
    original = DataView._all_positions

    def spy(self):
        element_passes.append(True)
        return original(self)

    # DataView instances are slotted; spy at the class seam instead.
    monkeypatch.setattr(DataView, "_all_positions", spy)

    assert view.facet_cell_count(
        FacetGridPlot(AxisRef.point("bias"), cell)
    ) == 9
    assert view.facet_cell_count(FacetGridPlot(AxisRef.repeat("repeat"), cell)) == 3
    curve_cell = CurvePlot(AxisRef.point("bias"))
    assert (
        view.facet_cell_count(FacetGridPlot(AxisRef.cell_data("sy"), curve_cell))
        == 120
    )
    assert element_passes == []

def test_flat_planes_materialize_lazily_and_exactly_once() -> None:
    """Grouping flattens only broadcast axis codes, then reuses them.

    Axis resolution itself must NOT materialize the full-shape planes: the
    dense image path never groups, and eagerly copying full coordinate planes
    per resolved axis once cost ~150 ms per 2048^2 live frame.
    """

    snapshot = _scan_snapshot(points=80, sites=12)
    view = DataView(snapshot)
    ref = AxisRef.cell_data("site")
    view._resolve(ref)
    assert view._flat_cache == {}
    positions = np.arange(snapshot.block.values.size, dtype=np.int64)
    view._domain(ref, positions)
    indices = view._flat_cache[ref]
    assert indices.ndim == 1
    assert indices.flags.owndata
    for _ in range(12):
        view._domain(ref, positions)
    assert view._flat_cache[ref] is indices

def test_full_dense_image_keeps_native_values_and_boolean_validity(
    monkeypatch,
) -> None:
    """A singleton 2048² projection needs no full int64 count plane."""

    snapshot = _large_dense_snapshot(2048, 2048)
    view = DataView(snapshot)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("singleton dense image entered the reduction/count path")

    monkeypatch.setattr(data_view_module, "_masked_leading_reduce", forbidden)
    tracemalloc.start()
    tracemalloc.reset_peak()
    payload = view.image(
        AxisRef.cell_data("column"),
        AxisRef.cell_data("row"),
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert payload.z.canonical.dtype == np.dtype(np.uint16)
    assert np.shares_memory(payload.z.canonical, snapshot.block.values)
    assert payload.valid.dtype == np.dtype(np.bool_)
    assert not payload.valid.flags.owndata
    assert peak < 32 << 20

def test_large_contiguous_facet_cells_are_views_with_bounded_peak(
    monkeypatch,
) -> None:
    """Four 1200×1920 rows stay views; projection cannot amplify to hundreds of MiB."""

    snapshot = _large_dense_snapshot(1200, 1920, points=4, dtype=np.uint8)
    view = DataView(snapshot)
    spec = FacetGridPlot(
        AxisRef.point("batch"),
        ImagePlot(AxisRef.cell_data("column"), AxisRef.cell_data("row")),
    )
    shares: list[bool] = []
    histogram_inputs: list[tuple[bool, bool]] = []
    original = DataView._image_from_planes
    original_histogram = data_view_module._facet_kernel_counts

    def observed(
        self,
        x,
        y,
        x_domain,
        y_domain,
        values,
        counts,
        *,
        valid=None,
        used_y=None,
        used_x=None,
    ):
        shares.append(np.shares_memory(values, snapshot.block.values))
        return original(
            self,
            x,
            y,
            x_domain,
            y_domain,
            values,
            counts,
            valid=valid,
            used_y=used_y,
            used_x=used_x,
        )

    def observed_histogram(values, valid, *args, **kwargs):
        histogram_inputs.append(
            (np.shares_memory(values, snapshot.block.values), valid is not None)
        )
        return original_histogram(values, valid, *args, **kwargs)

    monkeypatch.setattr(DataView, "_image_from_planes", observed)
    monkeypatch.setattr(
        data_view_module, "_facet_kernel_counts", observed_histogram
    )
    tracemalloc.start()
    tracemalloc.reset_peak()
    payload = view.facet(spec)
    histogram = view.facet(
        FacetGridPlot(AxisRef.point("batch"), HistogramPlot()),
        bins=(-0.5, 0.5),
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(payload.cells) == 4
    assert len(histogram.cells) == 4
    assert shares == [True] * 4
    # Histogram cells are counted together in one tensor pass; the old guard
    # expected four calls to the retired per-cell owner and therefore failed
    # precisely when the batched path did its job.
    assert histogram_inputs == [(True, True)]
    assert all(not cell.payload.valid.flags.owndata for cell in payload.cells)
    assert peak < 64 << 20

def test_large_integer_histogram_uses_one_native_uniform_count(
    monkeypatch,
) -> None:
    """Aligned integer bins need neither positions nor NumPy's chunk sorter."""

    cases = (
        (2048, 2048, np.uint16, 32, 48 << 20),
        (1200, 1920, np.uint8, 16, 32 << 20),
    )
    expected: list[tuple[make_snapshot, np.ndarray, np.ndarray, int]] = []
    for height, width, dtype, step, peak_limit in cases:
        invalid_first = np.dtype(dtype) == np.dtype(np.uint8)
        snapshot = _large_dense_snapshot(
            height,
            width,
            points=2 if invalid_first else 1,
            dtype=dtype,
            column_ramp=True,
            invalid_first_point=invalid_first,
        )
        upper = min(width, np.iinfo(dtype).max + 1)
        edges = np.arange(-0.5, upper + 0.5, step, dtype=float)
        view = DataView(snapshot)
        values = np.asarray(snapshot.block.values).reshape(-1)
        valid = np.asarray(view.samples.valid_mask).reshape(-1)
        counts, checked_edges = np.histogram(values[valid], bins=edges)
        expected.append((snapshot, checked_edges, counts, peak_limit))

    def forbidden_positions(_self):
        raise AssertionError("full-box histogram allocated element positions")

    def forbidden_histogram(*_args, **_kwargs):
        raise AssertionError("aligned integer histogram entered the generic sorter")

    original_bincount = np.bincount
    bincount_calls = 0

    def observed_bincount(*args, **kwargs):
        nonlocal bincount_calls
        bincount_calls += 1
        return original_bincount(*args, **kwargs)

    monkeypatch.setattr(DataView, "_all_positions", forbidden_positions)
    monkeypatch.setattr(data_view_module.np, "histogram", forbidden_histogram)
    monkeypatch.setattr(data_view_module.np, "bincount", observed_bincount)
    for index, (snapshot, edges, counts, peak_limit) in enumerate(expected, start=1):
        tracemalloc.start()
        tracemalloc.reset_peak()
        payload = DataView(snapshot).histogram(bins=edges)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        np.testing.assert_array_equal(payload.edges.canonical, edges)
        np.testing.assert_array_equal(payload.counts, counts)
        assert bincount_calls == index
        assert peak < peak_limit

def test_large_integer_histogram_domain_uses_native_statistics(
    monkeypatch,
) -> None:
    """Domain discovery cannot copy a full detector frame into float64."""

    initial = _large_dense_snapshot(
        2048,
        2048,
        dtype=np.uint16,
        column_ramp=True,
    )
    session = PlotSession(initial, HistogramPlot())
    observed: list[tuple[np.dtype, int]] = []
    original = fit_projection_module.aligned_histogram_edges

    def observed_edges(values, *args, **kwargs):
        array = np.asarray(values)
        observed.append((array.dtype, int(array.size)))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(
        fit_projection_module,
        "aligned_histogram_edges",
        observed_edges,
    )
    try:
        tracemalloc.start()
        tracemalloc.reset_peak()
        session.update_data(
            _large_dense_snapshot(
                2048,
                2048,
                dtype=np.uint16,
                column_ramp=True,
                revision=1,
            )
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert observed == [(np.dtype(np.uint16), 0)]
        assert peak < 48 << 20
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        session.close()

def test_extreme_uint64_histogram_falls_back_without_overflow() -> None:
    template = _large_dense_snapshot(1, 1, dtype=np.uint64)
    snapshot = make_snapshot(
        template.block.schema,
        np.asarray([[[[2**63 + 7]]]], dtype=np.uint64),
        revision=0,
    )

    payload = DataView(snapshot).histogram(bins=(-0.5, 0.5))
    np.testing.assert_array_equal(payload.edges.canonical, (-0.5, 0.5))
    np.testing.assert_array_equal(payload.counts, (0,))

def test_large_ungrouped_rolling_reuses_its_exact_valid_pool(
    monkeypatch,
) -> None:
    """One validity extraction feeds both the scalar and retained history pool."""

    snapshot = _large_dense_snapshot(
        1200,
        1920,
        dtype=np.uint8,
        column_ramp=True,
    )
    expected_pool = np.asarray(snapshot.block.values).reshape(-1)
    reducers = {
        Reduction.MEAN: np.mean,
        Reduction.SUM: np.sum,
        Reduction.MIN: np.min,
        Reduction.MAX: np.max,
        Reduction.FIRST: lambda values: values[0],
    }

    def forbidden_positions(_self):
        raise AssertionError("ungrouped rolling allocated element positions")

    monkeypatch.setattr(DataView, "_all_positions", forbidden_positions)
    for reduction, reducer in reducers.items():
        view = DataView(snapshot)
        tracemalloc.start()
        tracemalloc.reset_peak()
        sample = view.rolling_history(
            group=None, aggregation=reduction
        )[0]
        pooled = view.pooled_values()
        again = view.pooled_values()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert pooled is again
        np.testing.assert_array_equal(pooled, expected_pool)
        assert bool(sample.valid[0])
        np.testing.assert_allclose(sample.values[0], reducer(expected_pool))
        assert peak < 32 << 20
