from __future__ import annotations

from time import perf_counter

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, PlotSession, RollingPlot
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


def test_resolved_axis_materializes_one_flat_plane_for_domains() -> None:
    """A broadcast coordinate is flattened once during resolution, not per group."""

    snapshot = _scan_snapshot(points=80, sites=12)
    view = DataView(snapshot)
    ref = AxisRef.data("site")
    resolved = view._resolve(ref)
    assert resolved.canonical_flat.ndim == 1
    assert resolved.indices_flat.ndim == 1
    assert resolved.canonical_flat.flags.owndata
    canonical_id = id(resolved.canonical_flat)
    indices_id = id(resolved.indices_flat)
    positions = np.arange(snapshot.block.values.size, dtype=np.int64)
    for _ in range(12):
        view._domain(ref, positions)
    assert id(view._resolve(ref).canonical_flat) == canonical_id
    assert id(view._resolve(ref).indices_flat) == indices_id
