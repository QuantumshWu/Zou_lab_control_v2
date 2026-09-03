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


def _build_plane(sites: int, *, bimodal: bool = False, site_axis: str = "cell"):
    import zou_lab_control  # noqa: F401 - current checkout owns every package
    from zlc_data import (
        AxisId,
        AxisSpec,
        BlockId,
        CellValidity,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        DomainSpec,
        READOUT_EVENT,
        REPEAT,
        SCALAR_DOMAIN,
        SITE,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
        owned_snapshot_from_arrays,
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
    repeat_domain = DomainSpec((1,), (repeat,), ((0,),))
    site_labels = tuple(f"site-{index:02d}" for index in range(sites))
    if site_axis == "point":
        # A site per point row: the shape of a scan's per-site table, and
        # the shape whose history rows multiply by the site count.
        site_spec = AxisSpec(
            AxisId("bench.site"),
            "site",
            SITE,
            sites,
            tuple(range(sites)),
            coordinate_labels=site_labels,
        )
        event_schema = DatasetSchema(
            repeat_domain,
            DomainSpec(
                (sites,),
                (site_spec,),
                (tuple(range(sites)),),
            ),
            SCALAR_DOMAIN,
            ValueSchema.scalar(np.dtype(np.float64), "count"),
        )
        counts_shape = (1, sites, 1)
        validity_shape = (1, sites, 1)
    elif site_axis == "cell":
        # The occupancy processor's own geometry: the camera cycle's frame
        # row stays in the Point domain and the sites are the Cell payload.
        frame_spec = AxisSpec(
            AxisId("bench.frame"), "frame", READOUT_EVENT, 1, (0,)
        )
        site_spec = AxisSpec(
            AxisId("bench.site"),
            "site",
            SITE,
            sites,
            tuple(range(sites)),
            coordinate_labels=site_labels,
        )
        event_schema = DatasetSchema(
            repeat_domain,
            DomainSpec((1,), (frame_spec,), ((0,),)),
            DomainSpec((sites,), (site_spec,)),
            ValueSchema(
                ValidityContract.components(AxisId("bench.site")),
                np.dtype(np.float64),
                "count",
            ),
        )
        counts_shape = (1, 1, sites)
        validity_shape = (1, 1, sites)
    else:
        raise ValueError("site_axis must be 'point' or 'cell'")
    event_rows = event_schema.point_domain.size

    def source_event(revision: int):
        source_point = AxisSpec(
            AxisId("bench.frame-point"),
            "point",
            SITE,
            1,
            (0,),
        )
        schema = DatasetSchema(
            repeat_domain,
            DomainSpec((1,), (source_point,), ((0,),)),
            SCALAR_DOMAIN,
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
        if bimodal:
            # What a readout histogram actually looks like: a dark and a
            # bright population, so a bimodal fit converges as it would on
            # the bench instead of thrashing on uniform noise.
            bright = rng.random(size=counts_shape) < 0.5
            values = np.where(
                bright,
                rng.normal(40.0, 6.0, size=counts_shape),
                rng.normal(8.0, 3.0, size=counts_shape),
            )
            values = np.clip(values, 0.0, None).astype(np.float64)
        else:
            values = rng.uniform(0.0, 40.0, size=counts_shape).astype(np.float64)
        return owned_snapshot_from_arrays(
            event_schema,
            values,
            revision,
            validity=np.ones(validity_shape, dtype=np.bool_),
            stream_generation=StreamGenerationId("bench-generation"),
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
                    MonitorCoverage(event_rows, event_rows),
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


def run_plane_layer(
    *, sites: int, window: int, steady: int, site_axis: str = "cell"
) -> dict:
    plane, commit = _build_plane(sites, site_axis=site_axis)
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
        rows = last.block.schema.point_domain.size
        event_rows = sites if site_axis == "point" else 1
        assert rows == window * event_rows, (rows, window, event_rows)
    finally:
        plane.close()
    return {
        "ramp_view_ms": stats(ramp_ms),
        "ramp_last10_ms": stats(ramp_ms[-10:]),
        "steady_view_ms": stats(steady_ms),
    }


def run_session_layer(
    *,
    sites: int,
    window: int,
    steady: int,
    kind: str = "rolling",
    fit: str = "",
    size: str = "2x2",
    site_axis: str = "cell",
) -> dict:
    """One PlotSession fed the plane-materialized history, per shot.

    ``rolling`` is the trace that first exposed the window cost; ``facet_histogram``
    is the occupancy-counts grid an operator opens on a qCMOS run -- one
    histogram cell per site over the whole window, with an optional live
    bimodal fit that re-solves every cell on every shot, exactly as the
    console's live fit does.  ``site_axis`` says where the sites live:
    ``cell`` is the occupancy processor's geometry, ``point`` a per-site
    Point domain whose history rows multiply by the site count.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import zou_lab_control  # noqa: F401
    from zlc_plot import (
        AxisRef,
        FacetGridPlot,
        HistogramPlot,
        PlotLabels,
        PlotSession,
        RollingPlot,
    )

    plane, commit = _build_plane(
        sites, bimodal=kind == "facet_histogram", site_axis=site_axis
    )
    signal = "bench-occupancy/counts"
    plane.acquire_indexed_history(signal, window)
    revision = 1
    for _ in range(window + 1):
        revision += 1
        commit(revision)

    def materialized():
        return plane.current_dataset(signal)

    site_ref = (
        AxisRef.point("bench.site") if site_axis == "point" else AxisRef.cell_data("bench.site")
    )
    if kind == "rolling":
        spec = RollingPlot(
            group=site_ref,
            labels=PlotLabels("occupancy history", "shots", "counts"),
        )
    elif kind == "facet_histogram":
        spec = FacetGridPlot(
            site_ref,
            HistogramPlot(),
            labels=PlotLabels("occupancy counts", "counts", "shots"),
        )
    else:
        raise ValueError("kind must be rolling or facet_histogram")
    session = PlotSession(materialized(), spec, parameters={"window": window})
    update_ms: list[float] = []
    render_ms: list[float] = []
    fit_ms: list[float] = []
    fit_cells: list[int] = []
    first_fit_ms = None
    try:
        session.set_size(size)
        session.rgba()
        if fit:
            # The console's live fit: solved once now, then again on every
            # data revision.  The batch seam is timed in place so the fit's
            # share of a shot is read off the same call the product makes.
            batch = session._fit_facet_batch

            def timed_batch(*args, **kwargs):
                begin = time.perf_counter()
                try:
                    return batch(*args, **kwargs)
                finally:
                    fit_ms.append(time.perf_counter() - begin)

            session._fit_facet_batch = timed_batch
            begin = time.perf_counter()
            result = session.fit(fit)
            first_fit_ms = round((time.perf_counter() - begin) * 1000.0, 2)
            fit_ms.clear()
            fit_cells.append(len(tuple(getattr(result, "results", ()))))
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
            if fit:
                last = session.last_fit
                fit_cells.append(len(tuple(getattr(last, "results", ()))))
    finally:
        session.close()
        plane.close()
    payload = {
        "kind": kind,
        "update_ms": stats(update_ms),
        "render_ms": stats(render_ms),
    }
    if fit:
        payload["fit"] = fit
        payload["first_fit_ms"] = first_fit_ms
        payload["live_fit_batch_ms"] = stats(fit_ms) if fit_ms else None
        payload["live_fit_batches"] = len(fit_ms)
        payload["fit_cells"] = sorted(set(fit_cells))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=35)
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--steady", type=int, default=40)
    parser.add_argument(
        "--layer", choices=("plane", "session", "both"), default="both"
    )
    parser.add_argument(
        "--kind", choices=("rolling", "facet_histogram"), default="rolling"
    )
    parser.add_argument(
        "--fit",
        default="",
        help="live fit model armed on the session, e.g. bimodal_gaussian",
    )
    parser.add_argument("--size", default="2x2")
    parser.add_argument(
        "--site-axis",
        choices=("cell", "point"),
        default="cell",
        help="where the sites live: the cell payload (occupancy) or point rows",
    )
    arguments = parser.parse_args()
    label = (
        f"indexed_history_{arguments.kind}_{arguments.site_axis}"
        f"_s{arguments.sites}_w{arguments.window}"
        + (f"_{arguments.fit}" if arguments.fit else "")
    )
    payload: dict = {
        "sites": arguments.sites,
        "site_axis": arguments.site_axis,
        "window": arguments.window,
        "steady": arguments.steady,
        "kind": arguments.kind,
        "fit": arguments.fit,
        "size": arguments.size,
    }
    if arguments.layer in ("plane", "both"):
        payload["plane"] = run_plane_layer(
            sites=arguments.sites,
            window=arguments.window,
            steady=arguments.steady,
            site_axis=arguments.site_axis,
        )
        print("plane:", payload["plane"])
    if arguments.layer in ("session", "both"):
        payload["session"] = run_session_layer(
            sites=arguments.sites,
            window=arguments.window,
            steady=arguments.steady,
            kind=arguments.kind,
            fit=arguments.fit,
            size=arguments.size,
            site_axis=arguments.site_axis,
        )
        print("session:", payload["session"])
    print("wrote", write_result(payload, label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
