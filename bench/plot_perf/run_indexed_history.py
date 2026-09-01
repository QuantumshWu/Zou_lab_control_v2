"""Long-indexed-history costs, isolated per layer.

The bench that was missing when a 35-site occupancy-agreement panel with
a 5000-deep history window made the whole four-panel console visibly
slower: nothing under bench/ exercised an indexed history past window=40.

Two layers, measured separately so a regression names its owner:

* plane  -- one indexed processor signal on a real SignalDataPlane with a
  history lease of ``--window``; per shot, one ``current_dataset`` view
  (exactly what the console's presentation thread pays per presented
  publication).  Ramp (window filling) and steady state are reported
  apart, because their costs differ structurally.
* session -- a RollingPlot(window=..., group by site) PlotSession fed the
  plane-materialized snapshots; per shot, one ``update_data`` plus one
  composed ``rgba`` (projection + payload + render, no Qt).

Run: python -m bench.plot_perf.run_indexed_history --sites 35 --window 5000
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .common import stats, write_result


def _build_plane(sites: int):
    import zou_lab_control  # noqa: F401 - current checkout owns every package
    from zlc_data import (
        AxisId,
        AxisSpec,
        BlockId,
        CellValidity,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        PointColumn,
        PointTable,
        REPEAT,
        SITE,
        StreamGenerationId,
        ValueSchema,
    )
    from zlc_runtime.dataset import MonitorCoverage
    from zlc_runtime.dataset_output import (
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )
    from zlc_runtime.plane import SignalDataPlane

    source_declaration = DatasetOutputDeclaration("frame", "bench.frame")
    counts_declaration = DatasetOutputDeclaration(
        "counts",
        "bench.counts",
        index_by_source=True,
    )

    class Source:
        instance_id = "bench-source"
        dataset_output_declarations = (source_declaration,)

        @staticmethod
        def signal_key(name: str) -> str:
            return f"bench-source/{name}"

    class Derived:
        instance_id = "bench-occupancy"
        dataset_output_declarations = (counts_declaration,)

        @staticmethod
        def signal_key(name: str) -> str:
            return f"bench-occupancy/{name}"

        @staticmethod
        def validate_processor_source(_source) -> None:
            return None

        @staticmethod
        def evaluate_processor(_source, _publication):
            raise AssertionError("paused processor must not evaluate")

        @staticmethod
        def accept_processor_result(_source, _publication, _result) -> None:
            return None

        @staticmethod
        def accept_processor_failure(error: Exception) -> None:
            raise error

        @staticmethod
        def accept_processor_cancelled() -> None:
            return None

        @staticmethod
        def request_processor_owner_wake() -> None:
            return None

    repeat = AxisSpec(AxisId("bench.repeat"), "repeat", REPEAT, 1, (0,))
    site_column = PointColumn(
        AxisId("bench.site"),
        "site",
        SITE,
        PointColumn.NUMERIC,
        tuple(range(sites)),
        coordinate_labels=tuple(f"site-{index:02d}" for index in range(sites)),
    )
    event_schema = DatasetSchema(
        repeat,
        PointTable(sites, (site_column,)),
        None,
        ValueSchema.scalar(np.dtype(np.float64), "count"),
    )

    def source_event(revision: int):
        source_point = PointColumn(
            AxisId("bench.frame-point"),
            "point",
            SITE,
            PointColumn.NUMERIC,
            (0,),
        )
        schema = DatasetSchema(
            repeat,
            PointTable(1, (source_point,)),
            None,
            ValueSchema.scalar(np.dtype(np.float64), "count"),
        )
        block = DataBlock(
            BlockId(f"frame-{revision}"),
            DatasetRevision(revision),
            np.asarray([[[float(revision)]]], dtype=np.float64),
            CellValidity(np.ones((1, 1), dtype=np.bool_)),
            schema,
        )
        from zlc_data import OwnedSnapshot

        return OwnedSnapshot(
            block.ref(StreamGenerationId("bench-generation")), block
        )

    rng = np.random.default_rng(7)

    def counts_event(revision: int):
        from zlc_data import OwnedSnapshot

        values = rng.uniform(0.0, 40.0, size=(1, sites, 1)).astype(np.float64)
        block = DataBlock(
            BlockId(f"counts-{revision}"),
            DatasetRevision(revision),
            values,
            CellValidity(np.ones((1, sites), dtype=np.bool_)),
            event_schema,
        )
        return OwnedSnapshot(
            block.ref(StreamGenerationId("bench-generation")), block
        )

    plane = SignalDataPlane()
    source = Source()
    derived = Derived()
    plane.begin_generation(source)

    def commit_source(revision: int):
        plane.commit_live(
            source,
            {
                "frame": LiveDatasetOutput(
                    source_declaration,
                    source_event(revision),
                    MonitorCoverage(1, 1),
                )
            },
        )
        return plane.latest_publication("bench-source/frame")

    def commit_counts(revision: int, publication) -> None:
        plane.commit_processor(
            derived,
            {
                "counts": LiveDatasetOutput(
                    counts_declaration,
                    counts_event(revision),
                    MonitorCoverage(sites, sites),
                )
            },
            source_publication=publication,
        )

    first = commit_source(1)
    plane.attach_latest_only_processor(
        derived,
        source_name="bench-source/frame",
        initial_publication=first,
        paused=True,
    )
    commit_counts(1, first)

    def commit(revision: int) -> None:
        commit_counts(revision, commit_source(revision))

    return plane, commit


def run_plane_layer(*, sites: int, window: int, steady: int) -> dict:
    plane, commit = _build_plane(sites)
    signal = "bench-occupancy/counts"
    lease = plane.acquire_indexed_history(signal, window)
    del lease  # held for the plane's life; the bench never shrinks it

    ramp_ms: list[float] = []
    steady_ms: list[float] = []
    revision = 1
    try:
        for phase, count, sink in (
            ("ramp", window, ramp_ms),
            ("steady", steady, steady_ms),
        ):
            for _ in range(count):
                revision += 1
                commit(revision)
                begin = time.perf_counter()
                snapshot = plane.current_dataset(signal)
                sink.append(time.perf_counter() - begin)
            del snapshot
        last = plane.current_dataset(signal)
        rows = last.block.schema.point_table.row_count
        assert rows == window * sites, (rows, window, sites)
    finally:
        plane.close()
    return {
        "ramp_view_ms": stats(ramp_ms),
        "ramp_last10_ms": stats(ramp_ms[-10:]),
        "steady_view_ms": stats(steady_ms),
    }


def run_session_layer(*, sites: int, window: int, steady: int) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import zou_lab_control  # noqa: F401
    from zlc_plot import AxisRef, PlotLabels, PlotSession, RollingPlot

    plane, commit = _build_plane(sites)
    signal = "bench-occupancy/counts"
    plane.acquire_indexed_history(signal, window)
    revision = 1
    for _ in range(window + 1):
        revision += 1
        commit(revision)

    def materialized():
        return plane.current_dataset(signal)

    session = PlotSession(
        materialized(),
        RollingPlot(
            group=AxisRef.point("bench.site"),
            labels=PlotLabels("occupancy history", "shots", "counts"),
        ),
        parameters={"window": window},
    )
    update_ms: list[float] = []
    render_ms: list[float] = []
    try:
        session.set_size("2x2")
        session.rgba()
        for _ in range(steady):
            revision += 1
            commit(revision)
            snapshot = materialized()
            begin = time.perf_counter()
            session.update_data(snapshot)
            update_ms.append(time.perf_counter() - begin)
            begin = time.perf_counter()
            session.rgba()
            render_ms.append(time.perf_counter() - begin)
    finally:
        session.close()
        plane.close()
    return {
        "update_ms": stats(update_ms),
        "render_ms": stats(render_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=35)
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--steady", type=int, default=40)
    parser.add_argument(
        "--layer", choices=("plane", "session", "both"), default="both"
    )
    arguments = parser.parse_args()
    label = f"indexed_history_s{arguments.sites}_w{arguments.window}"
    payload: dict = {
        "sites": arguments.sites,
        "window": arguments.window,
        "steady": arguments.steady,
    }
    if arguments.layer in ("plane", "both"):
        payload["plane"] = run_plane_layer(
            sites=arguments.sites,
            window=arguments.window,
            steady=arguments.steady,
        )
        print("plane:", payload["plane"])
    if arguments.layer in ("session", "both"):
        payload["session"] = run_session_layer(
            sites=arguments.sites,
            window=arguments.window,
            steady=arguments.steady,
        )
        print("session:", payload["session"])
    print("wrote", write_result(payload, label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
