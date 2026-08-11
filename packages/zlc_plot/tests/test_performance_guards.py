from __future__ import annotations

from time import perf_counter

import numpy as np

from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, ImagePlot, PlotSession, RollingPlot
from zlc_plot.data_view import DataView


# These are intentionally named guards rather than hidden timing literals.
# They catch accidental full-tensor copies while leaving enough room for a
# shared CI worker's normal variance.
MAX_REPLACE_SPEC_SECONDS = 8.0
MAX_ROLLING_20_FRAME_SECONDS = 8.0


def _scan_snapshot(*, revision: int = 0, repeats: int = 5, points: int = 120, sites: int = 30) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({"scan": np.linspace(0.0, 1.0, points)}),
        data_axes=(Axis.create("site", size=sites),),
        dtype=np.float64,
        generation="projection-performance-guard",
    )
    values = np.sin(np.linspace(0.0, 8.0, points))[None, :, None]
    values = np.broadcast_to(values, (repeats, points, sites)).copy()
    values += np.arange(repeats, dtype=float)[:, None, None] * 0.01
    return DatasetSnapshot(schema, values, revision=revision)


def test_replace_spec_and_rolling_projection_have_bounded_cost() -> None:
    snapshot = _scan_snapshot()
    session = PlotSession(
        snapshot,
        CurvePlot(AxisRef.point("scan"), group=AxisRef.data("site")),
    )
    try:
        timings: dict[str, float] = {}
        start = perf_counter()
        session.replace_spec(
            CurvePlot(AxisRef.point("scan"), group=AxisRef.repeat())
        )
        timings["group_to_repeat"] = perf_counter() - start

        start = perf_counter()
        session.replace_spec(
            CurvePlot(AxisRef.point("scan"), group=AxisRef.data("site"))
        )
        timings["group_to_site"] = perf_counter() - start

        start = perf_counter()
        session.replace_spec(
            FacetGridPlot(
                AxisRef.data("site"),
                CurvePlot(AxisRef.point("scan")),
            )
        )
        timings["kind_to_facet"] = perf_counter() - start
        assert all(value < MAX_REPLACE_SPEC_SECONDS for value in timings.values()), timings
    finally:
        session.close()

    rolling = PlotSession(
        snapshot,
        RollingPlot(group=AxisRef.data("site")),
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
    element pass; only the undeclared point-coordinate fallback may still
    walk elements.
    """

    bias = [float(v) for v in range(9)]
    table = PointTable.from_columns({"bias": bias})
    topology = PointTopology.from_cartesian(
        (Axis.create("bias", values=bias),), point_table=table
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=3),
        table,
        data_axes=(
            Axis.create("sy", values=tuple(float(v) for v in range(120))),
            Axis.create("sx", values=tuple(float(v) for v in range(160))),
        ),
        point_topology=topology,
        dtype=np.uint8,
        generation="facet-count-guard",
    )
    values = np.zeros((3, 9, 120, 160), dtype=np.uint8)
    view = DataView(DatasetSnapshot(schema, values, revision=0))
    cell = ImagePlot(AxisRef.data("sx"), AxisRef.data("sy"))

    element_passes: list[object] = []
    original = DataView._all_positions

    def spy(self):
        element_passes.append(True)
        return original(self)

    # DataView instances are slotted; spy at the class seam instead.
    monkeypatch.setattr(DataView, "_all_positions", spy)

    assert view.facet_cell_count(
        FacetGridPlot(AxisRef.point_dimension("bias"), cell)
    ) == 9
    assert view.facet_cell_count(FacetGridPlot(AxisRef.repeat(), cell)) == 3
    curve_cell = CurvePlot(AxisRef.point_dimension("bias"))
    assert (
        view.facet_cell_count(FacetGridPlot(AxisRef.data("sy"), curve_cell))
        == 120
    )
    assert element_passes == []

    # The undeclared point-coordinate fallback keeps the exact generic
    # count -- and is the only path allowed to walk elements.
    assert view.facet_cell_count(FacetGridPlot(AxisRef.point("bias"), cell)) == 9
    assert element_passes


def test_flat_planes_materialize_lazily_and_exactly_once() -> None:
    """Grouping flattens a broadcast coordinate on first use, then reuses it.

    Axis resolution itself must NOT materialize the full-shape planes: the
    dense image path never groups, and eagerly copying two full-size planes
    per resolved axis once cost ~150 ms per 2048^2 live frame.
    """

    snapshot = _scan_snapshot(points=80, sites=12)
    view = DataView(snapshot)
    ref = AxisRef.data("site")
    view._resolve(ref)
    assert view._flat_cache == {}
    positions = np.arange(snapshot.block.values.size, dtype=np.int64)
    view._domain(ref, positions)
    canonical, indices = view._flat_cache[ref]
    assert canonical.ndim == 1
    assert indices.ndim == 1
    assert canonical.flags.owndata
    for _ in range(12):
        view._domain(ref, positions)
    cached_canonical, cached_indices = view._flat_cache[ref]
    assert cached_canonical is canonical
    assert cached_indices is indices
