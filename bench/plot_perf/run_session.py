"""Compute-layer benchmark: PlotSession.update_data (projection + Agg render).

The same case catalog as run_host, without Qt or the raster worker: this is
the pure per-revision compute floor the pipeline builds on.

Run:  python -m bench.plot_perf.run_session [--only substring] [--label name]
"""
from __future__ import annotations

import argparse
import time
import traceback

import matplotlib

matplotlib.use("Agg", force=True)

from .cases import catalog  # noqa: E402
from .common import stats, write_result  # noqa: E402


def run_case(case) -> dict:
    from zlc_plot import PlotSession

    feed = case.feed()
    report: dict = {"case": case.name, "points": feed.size}
    t0 = time.perf_counter()
    session = PlotSession(feed.next(), case.spec())
    session.set_size("4x4")
    if case.parameters:
        session.set_parameters(dict(case.parameters))
    session.rgba()
    report["first_render_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    try:
        samples = []
        for index in range(10):
            fresh = feed.next()
            start = time.perf_counter()
            session.update_data(fresh)
            elapsed = time.perf_counter() - start
            if index >= 2:
                samples.append(elapsed)
        report["update_data"] = stats(samples)
        if case.fit:
            fit_times = []
            model = case.fit["model"]
            is_facet = "facet" in case.name
            for index in range(5):
                session.update_data(feed.next())
                start = time.perf_counter()
                session.fit(
                    model,
                    live=not is_facet,
                    fit_all_facets=is_facet,
                )
                if index >= 1:
                    fit_times.append(time.perf_counter() - start)
            report["fit_solve"] = stats(fit_times)
    finally:
        session.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--label", default="session")
    arguments = parser.parse_args()
    results = []
    for case in catalog():
        if arguments.only and arguments.only not in case.name:
            continue
        print(f"=== {case.name} ===", flush=True)
        try:
            report = run_case(case)
        except Exception:
            report = {"case": case.name, "fatal": traceback.format_exc()}
        results.append(report)
        print(report, flush=True)
    path = write_result({"results": results}, arguments.label)
    print("written:", path)


if __name__ == "__main__":
    main()
