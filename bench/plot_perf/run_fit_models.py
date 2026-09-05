"""Every registered fit model, timed on the data shape it is FOR.

One row per (model, target): solve wall over repeated product-path fits
(``session.fit``), plus the solver's own effort counters where the result
carries them.  The sweep exists to catch the model whose seeds, bounds or
Jacobian make it an outlier on ordinary data -- a per-model regression a
whole-panel benchmark averages away.

Run: python -m bench.plot_perf.run_fit_models [--rounds 8]
     python -m bench.plot_perf.run_fit_models --model release_recapture --rounds 20
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .common import bootstrap, stats, write_result

bootstrap()

def _curve_feed():
    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    x = np.linspace(-6.0, 6.0, 2000)
    rng = np.random.default_rng(7)
    schema = make_dataset_schema(
        repeat_domain(size=8),
        mapped_domain_from_columns({"x": x}),
        cell_axes=(),
        dtype=np.float64,
    )

    def snapshot(revision: int):
        base = (
            5.0 * np.exp(-0.5 * ((x - 0.4) / 1.3) ** 2)
            + 0.8
            + 0.35 * np.sin(3.0 * x) * np.exp(-0.2 * np.abs(x))
        )
        values = base + rng.normal(0.0, 0.12, (8, x.size))
        return make_snapshot(schema, values, revision=revision)

    return snapshot

def _histogram_feed():
    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    rng = np.random.default_rng(11)
    schema = make_dataset_schema(
        repeat_domain(size=64),
        mapped_domain_from_columns({"sample": np.arange(512, dtype=np.int64)}),
        cell_axes=(),
        dtype=np.float64,
    )

    def snapshot(revision: int):
        low = rng.normal(8.0, 2.0, (64, 256))
        high = rng.normal(30.0, 4.0, (64, 256))
        return make_snapshot(
            schema,
            np.concatenate((low, high), axis=1),
            revision=revision,
        )

    return snapshot

def _image_feed():
    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    rng = np.random.default_rng(23)
    yy, xx = np.mgrid[0:96, 0:128]
    spot = 40.0 * np.exp(
        -0.5 * (((xx - 70.0) / 9.0) ** 2 + ((yy - 44.0) / 6.0) ** 2)
    )
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"point": [0.0]}),
        cell_axes=(axis("y", size=96), axis("x", size=128)),
        dtype=np.float64,
    )

    def snapshot(revision: int):
        frame = spot + 3.0 + rng.normal(0.0, 0.6, spot.shape)
        return make_snapshot(schema, frame[None, None], revision=revision)

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
            AxisRef.cell_data("x"),
            AxisRef.cell_data("y"),
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


def run_release_recapture(rounds: int) -> dict:
    """Simple physical-time curves: complete FitEngine, excluding rendering.

    SciPy's Lambert W generates the observations independently of the model
    under test. Both solvers get identical data, weights, bounds and cold seeds.
    """

    from dataclasses import replace
    import platform
    from scipy.special import lambertw
    from numba import get_num_threads, set_num_threads
    from zlc_plot.fit import FitEngine

    if rounds < 1:
        raise ValueError("rounds must be positive")
    previous_threads = get_num_threads()
    set_num_threads(min(4, previous_threads))
    engine = FitEngine()
    model = engine.registry.get("release_recapture")
    scipy_model = replace(model, compiled_descriptor=None)
    truth = np.array((0.8, 0.05, 6.0, 40_000.0))
    rng = np.random.default_rng(20260905)
    payload = {
        "model": model.model_id,
        "python": platform.python_version(),
        "numba_threads": get_num_threads(),
        "rounds": rounds,
        "time_range_seconds": [0.0, 200e-6],
        "truth": dict(zip(model.parameter_names, truth.tolist())),
        "observation_sigma": 0.003,
        "objective": "weighted least squares",
        "includes": "FitEngine preparation + cold multi-start solver + result/covariance",
        "excludes": "render, Qt, camera acquisition",
        "cases": {},
    }

    def timed(work):
        work()
        durations = []
        for _ in range(rounds):
            began = time.perf_counter()
            result = work()
            durations.append(time.perf_counter() - began)
        return stats(durations), result

    try:
        for points in (32, 128, 512):
            t = np.linspace(0.0, 200e-6, points)
            q = np.exp(-lambertw((2 * np.pi * truth[3] * t) ** 2).real)
            clean = truth[0] * -np.expm1(-truth[2] * q) / -np.expm1(-truth[2]) + truth[1]
            observed = clean + rng.normal(0.0, 0.003, points)
            sigma = np.full(points, 0.003)
            begin = time.perf_counter()
            engine.fit(model, (t,), observed, observation_sigma=sigma)
            first_ms = 1000.0 * (time.perf_counter() - begin)
            native, result = timed(lambda: engine.fit(model, (t,), observed, observation_sigma=sigma))
            scipy, reference = timed(lambda: engine.fit(scipy_model, (t,), observed, observation_sigma=sigma))
            value_jac, _ = timed(lambda: model.evaluate_jacobian((t,), truth))
            error = np.max(np.abs(result.fitted_values - reference.fitted_values))
            if not result.success or not reference.success or error > 1e-6:
                raise RuntimeError(f"release-recapture solver comparison failed at {points} points")
            payload["cases"][str(points)] = {
                "first_fit_ms": first_ms,
                "compiled": native,
                "scipy": scipy,
                "value_and_jacobian": value_jac,
                "fitted_parameters": dict(result.parameters),
                "standard_errors": result.standard_errors.tolist(),
                "eta_frequency_correlation": float(result.covariance[2, 3] / np.sqrt(result.covariance[2, 2] * result.covariance[3, 3])),
                "max_curve_difference": float(error),
            }
            if points == 128:
                batch_values = [clean + rng.normal(0.0, 0.003, points) for _ in range(64)]
                batch_time, batch = timed(lambda: engine.fit_batch(
                    model, [(t,)] * 64, batch_values,
                    observation_sigmas=[sigma] * 64,
                ))
                results, failures = batch
                if any(failures) or any(item is None or not item.success for item in results):
                    raise RuntimeError("release-recapture batch did not fit every cell")
                payload["batch_64x128"] = batch_time
    finally:
        set_num_threads(previous_threads)
    return payload

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--model", choices=("release_recapture",))
    arguments = parser.parse_args()
    if arguments.model == "release_recapture":
        import json
        payload = run_release_recapture(arguments.rounds)
        print(json.dumps(payload, indent=2))
    else:
        payload = run(arguments.rounds)
    print("wrote", write_result(payload, arguments.model or "fit-model-sweep"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
