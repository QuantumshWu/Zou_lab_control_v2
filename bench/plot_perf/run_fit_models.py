"""Every registered fit model, timed on the data shape it is FOR.

One row per (model, target): solve wall over repeated product-path fits
(``session.fit``), plus the solver's own effort counters where the result
carries them.  The sweep exists to catch the model whose seeds, bounds or
Jacobian make it an outlier on ordinary data -- a per-model regression a
whole-panel benchmark averages away.

Run: python -m bench.plot_perf.run_fit_models [--rounds 8]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .common import bootstrap, stats, write_result

bootstrap()


def _curve_feed():
    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable

    x = np.linspace(-6.0, 6.0, 2000)
    rng = np.random.default_rng(7)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=8),
        PointTable.from_columns({"x": x}),
        data_axes=(),
        dtype=np.float64,
        generation="fit-sweep-curve",
    )

    def snapshot(revision: int):
        base = (
            5.0 * np.exp(-0.5 * ((x - 0.4) / 1.3) ** 2)
            + 0.8
            + 0.35 * np.sin(3.0 * x) * np.exp(-0.2 * np.abs(x))
        )
        values = base + rng.normal(0.0, 0.12, (8, x.size))
        return DatasetSnapshot(schema, values, revision=revision)

    return snapshot


def _histogram_feed():
    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable

    rng = np.random.default_rng(11)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=64),
        PointTable.from_columns({"sample": np.arange(512, dtype=np.int64)}),
        data_axes=(),
        dtype=np.float64,
        generation="fit-sweep-histogram",
    )

    def snapshot(revision: int):
        low = rng.normal(8.0, 2.0, (64, 256))
        high = rng.normal(30.0, 4.0, (64, 256))
        return DatasetSnapshot(
            schema,
            np.concatenate((low, high), axis=1),
            revision=revision,
        )

    return snapshot


def _image_feed():
    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable

    rng = np.random.default_rng(23)
    yy, xx = np.mgrid[0:96, 0:128]
    spot = 40.0 * np.exp(
        -0.5 * (((xx - 70.0) / 9.0) ** 2 + ((yy - 44.0) / 6.0) ** 2)
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"point": [0.0]}),
        data_axes=(Axis.create("y", size=96), Axis.create("x", size=128)),
        dtype=np.float64,
        generation="fit-sweep-image",
    )

    def snapshot(revision: int):
        frame = spot + 3.0 + rng.normal(0.0, 0.6, spot.shape)
        return DatasetSnapshot(schema, frame[None, None], revision=revision)

    return snapshot


def _session(kind: str):
    from zlc_plot import (
        AxisRef,
        CurvePlot,
        HistogramPlot,
        ImagePlot,
        PlotLabels,
        PlotSession,
    )

    if kind == "curve":
        feed = _curve_feed()
        spec = CurvePlot(AxisRef.point("x"), labels=PlotLabels("c", "x", "y"))
    elif kind == "histogram":
        feed = _histogram_feed()
        spec = HistogramPlot(labels=PlotLabels("h", "value", "count"))
    else:
        feed = _image_feed()
        spec = ImagePlot(
            AxisRef.data("x"),
            AxisRef.data("y"),
            labels=PlotLabels("i", "x", "y"),
        )
    session = PlotSession(feed(1), spec)
    session.set_size("2x2")
    return session, feed


def run(rounds: int) -> dict:
    payload: dict = {}
    for kind in ("curve", "histogram", "image"):
        session, feed = _session(kind)
        revision = 1
        try:
            session.rgba()
            for model in session.fit_models:
                identifier = str(model.model_id)
                # Warm twice: the first solve pays cache deserialization,
                # the second settles the warm-start table.
                try:
                    for _ in range(2):
                        session.fit(model=identifier)
                except Exception as error:
                    payload[f"{kind}:{identifier}"] = {
                        "error": str(error) or type(error).__name__
                    }
                    continue
                times = []
                effort: dict[str, int] = {}
                failed = None
                for _round in range(rounds):
                    revision += 1
                    session.update_data(feed(revision))
                    begin = time.perf_counter()
                    result = session.fit(model=identifier)
                    times.append(time.perf_counter() - begin)
                    fit = getattr(result, "fit", result)
                    if not getattr(fit, "success", True):
                        failed = getattr(fit, "message", "failed")
                    for name in ("nfev", "njev", "iterations", "seed_count"):
                        value = getattr(fit, name, None)
                        if isinstance(value, (int, np.integer)):
                            effort[name] = max(
                                effort.get(name, 0), int(value)
                            )
                row: dict = {"solve_ms": stats(times)}
                if effort:
                    row["max_effort"] = effort
                if failed is not None:
                    row["last_failure"] = str(failed)
                payload[f"{kind}:{identifier}"] = row
                print(
                    f"{kind}:{identifier}: "
                    f"median {row['solve_ms']['median_ms']} ms"
                    + (f"  effort {effort}" if effort else "")
                    + (f"  FAILED {failed}" if failed else "")
                )
        finally:
            session.close()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    arguments = parser.parse_args()
    payload = run(arguments.rounds)
    print("wrote", write_result(payload, "fit-model-sweep"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
