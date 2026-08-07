"""Internal stage profiler for a latest-only camera image, without a GUI.

It exercises the same ``ImagePlot -> LivePlotController -> RasterPlotHost``
path used by a GUI and records the separately observable preparation, render,
raster handoff and front-promotion stages. Protected overrides below are timing
instrumentation, not application integration APIs.

Run one resolution per process so the RSS baseline is meaningful::

    python examples/camera_live_profile.py --resolution 1024
    python examples/camera_live_profile.py --resolution 2048
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import sys
from threading import Event, Lock, Thread
from time import monotonic, perf_counter, sleep
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

try:
    from examples._role_data import (
        Axis, DatasetSchema, DatasetSnapshot, PointTable, PointTopology,
    )
except ModuleNotFoundError:
    from _role_data import (
        Axis, DatasetSchema, DatasetSnapshot, PointTable, PointTopology,
    )
from zlc_plot import (
    AxisRef,
    ImagePlot,
    PlotLabels,
    PlotSession,
    RasterPlotHost,
)

try:
    import psutil
except (ImportError, ModuleNotFoundError):  # optional profiling aid
    psutil = None


@dataclass(slots=True)
class _Recorder:
    _samples: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _lock: Lock = field(default_factory=Lock)

    def add(self, name: str, elapsed_seconds: float) -> None:
        with self._lock:
            self._samples[name].append(float(elapsed_seconds) * 1_000.0)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def snapshot(self) -> dict[str, tuple[float, ...]]:
        with self._lock:
            return {
                name: tuple(values)
                for name, values in self._samples.items()
            }


class _TimedSession(PlotSession):
    """Instrument lifecycle boundaries while retaining normal session logic."""

    def __init__(self, *args: Any, recorder: _Recorder, **kwargs: Any) -> None:
        self._profile_recorder = recorder
        super().__init__(*args, **kwargs)

    def prepare_live_frame(self, *args: Any, **kwargs: Any) -> Future[Any]:
        started = perf_counter()
        future = super().prepare_live_frame(*args, **kwargs)
        future.add_done_callback(
            lambda _completed: self._profile_recorder.add(
                "projection_prepare", perf_counter() - started
            )
        )
        return future

    def commit_live_frame(self, prepared: Any) -> Any:
        started = perf_counter()
        try:
            return super().commit_live_frame(prepared)
        finally:
            self._profile_recorder.add(
                "artist_render_commit", perf_counter() - started
            )

    def _raster_capture_rgba(
        self,
        *,
        redraw: bool = False,
    ) -> np.ndarray:
        started = perf_counter()
        try:
            return super()._raster_capture_rgba(
                redraw=redraw,
            )
        finally:
            self._profile_recorder.add(
                "rgba_handoff", perf_counter() - started
            )


class _TimedRasterPlotHost(RasterPlotHost):
    def __init__(self, session_factory: Any, recorder: _Recorder) -> None:
        self._profile_recorder = recorder
        super().__init__(session_factory)

    def _capture_front(self) -> Any:
        started = perf_counter()
        try:
            return super()._capture_front()
        finally:
            self._profile_recorder.add(
                "front_capture_and_pack", perf_counter() - started
            )

    def _promote(self, front: Any) -> bool:
        started = perf_counter()
        try:
            return bool(super()._promote(front))
        finally:
            self._profile_recorder.add(
                "front_promotion_and_callbacks", perf_counter() - started
            )


class _RssSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self._interval_seconds = float(interval_seconds)
        self._stop = Event()
        self._thread: Thread | None = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self.baseline_bytes: int | None = None
        self.peak_bytes: int | None = None

    def start(self) -> None:
        if self._process is None:
            return
        initial = int(self._process.memory_info().rss)
        self.baseline_bytes = initial
        self.peak_bytes = initial

        def sample() -> None:
            while not self._stop.wait(self._interval_seconds):
                current = int(self._process.memory_info().rss)
                assert self.peak_bytes is not None
                self.peak_bytes = max(self.peak_bytes, current)

        self._thread = Thread(target=sample, name="zlc-profile-rss", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._process is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        current = int(self._process.memory_info().rss)
        assert self.peak_bytes is not None
        self.peak_bytes = max(self.peak_bytes, current)


def _camera_contract(
    resolution: int,
    geometry: str,
) -> tuple[DatasetSchema, np.ndarray[Any, Any], ImagePlot]:
    repeat = Axis.create("repeat", values=np.array([0], dtype=np.int64))
    y_axis = Axis.create(
        "camera_y",
        values=np.arange(resolution, dtype=np.int32),
        label="Camera Y",
    )
    x_axis = Axis.create(
        "camera_x",
        values=np.arange(resolution, dtype=np.int32),
        label="Camera X",
    )
    if geometry == "data-dim":
        points = PointTable.from_columns(
            {"sample": np.array([0], dtype=np.int64)},
            labels={"sample": "Camera sample"},
        )
        data_axes = (y_axis, x_axis)
        topology = None
        x_ref = AxisRef.data("camera_x")
        y_ref = AxisRef.data("camera_y")
    elif geometry == "point-topology":
        points = PointTable.from_columns(
            {
                "camera_y": np.repeat(y_axis.coordinates, resolution),
                "camera_x": np.tile(x_axis.coordinates, resolution),
            },
            labels={"camera_y": "Camera Y", "camera_x": "Camera X"},
        )
        data_axes = ()
        topology = PointTopology.from_cartesian(
            (y_axis, x_axis),
            point_table=points,
        )
        x_ref = AxisRef.point_dimension("camera_x")
        y_ref = AxisRef.point_dimension("camera_y")
    else:
        raise ValueError("geometry must be 'data-dim' or 'point-topology'")
    schema = DatasetSchema.create(
        repeat,
        points,
        data_axes=data_axes,
        point_topology=topology,
        value_label="Camera counts",
        canonical_unit="1",
        display_unit="1",
        dtype=np.uint16,
        generation=f"camera-profile-{resolution}",
        metadata={"source": "synthetic-monitor-camera"},
    )
    coordinate = np.arange(resolution, dtype=np.uint32)
    base = (
        coordinate[:, np.newaxis] * np.uint32(17)
        + coordinate[np.newaxis, :] * np.uint32(31)
    )
    base = np.asarray(base & np.uint32(0x0FFF), dtype=np.uint16)
    spec = ImagePlot(
        x_ref,
        y_ref,
        labels=PlotLabels(
            title=f"Live camera {resolution} x {resolution}",
            x="Camera X",
            y="Camera Y",
            value="Counts",
        ),
    )
    return schema, base, spec


def _snapshot(
    schema: DatasetSchema,
    base: np.ndarray[Any, Any],
    revision: int,
) -> DatasetSnapshot:
    frame = np.array(base, copy=True)
    height, width = frame.shape
    half = max(4, min(height, width) // 64)
    center_x = (revision * 37) % width
    center_y = (revision * 23) % height
    x0, x1 = max(0, center_x - half), min(width, center_x + half)
    y0, y1 = max(0, center_y - half), min(height, center_y + half)
    frame[y0:y1, x0:x1] = np.uint16(0xFFFF)
    values = (
        frame[np.newaxis, np.newaxis, :, :]
        if schema.data_axes
        else frame.reshape(1, -1)
    )
    return DatasetSnapshot(
        schema,
        values,
        revision=revision,
        metadata={"camera_frame": revision},
    )


def _distribution(values: tuple[float, ...] | list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": int(array.size),
        "p50_ms": float(np.percentile(array, 50.0)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "max_ms": float(np.max(array)),
    }


def _front_has_painted_data(front: Any) -> bool:
    pixels = np.frombuffer(front.buffer.pixels, dtype=np.uint8)
    expected = front.buffer.width * front.buffer.height * 4
    if pixels.size != expected or pixels.size == 0:
        return False
    rgba = pixels.reshape(front.buffer.height, front.buffer.width, 4)
    rgb = rgba[:, :, :3]
    return bool(np.any(rgb != rgb[0, 0]) and np.any(rgba[:, :, 3] != 0))


def profile_camera(
    *,
    resolution: int,
    frames: int,
    producer_hz: float,
    refresh_interval_ms: int,
    size: str,
    device_pixel_ratio: float,
    timeout_seconds: float,
    geometry: str = "data-dim",
) -> dict[str, Any]:
    if resolution <= 0 or frames < 2:
        raise ValueError("resolution must be positive and frames must be at least two")
    if producer_hz <= 0.0 or timeout_seconds <= 0.0:
        raise ValueError("producer_hz and timeout_seconds must be positive")

    cleanup = ExitStack()
    cleanup_status = {"live_closed": False, "host_closed": False}
    try:
        rss = _RssSampler()
        rss.start()
        cleanup.callback(rss.stop)
        process = psutil.Process(os.getpid()) if psutil is not None else None
        cpu_before = process.cpu_times() if process is not None else None
        case_started = perf_counter()
        recorder = _Recorder()
        schema, base, spec = _camera_contract(resolution, geometry)
        initial = _snapshot(schema, base, 0)
        host = _TimedRasterPlotHost(
            lambda: _TimedSession(
                initial,
                spec,
                size=size,
                parameters={"interpolation": "nearest"},
                device_pixel_ratio=device_pixel_ratio,
                recorder=recorder,
            ),
            recorder,
        )
        cleanup.callback(
            lambda: cleanup_status.__setitem__(
                "host_closed",
                host.close(timeout=min(timeout_seconds, 60.0)),
            )
        )
        initial_front = host.wait_for_front(timeout=timeout_seconds)
        initial_painted = _front_has_painted_data(initial_front)
        # Exclude one-time Figure/session construction from steady-state live stages.
        recorder.clear()

        publish_times: dict[int, float] = {}
        presented_revisions: list[int] = []
        presented_at: list[float] = []
        publish_to_front_ms: list[float] = []
        callback_lock = Lock()

        def on_front(front: Any) -> None:
            revision = int(front.identity.data_revision)
            now = perf_counter()
            with callback_lock:
                presented_revisions.append(revision)
                presented_at.append(now)
                published_at = publish_times.get(revision)
                if published_at is not None:
                    publish_to_front_ms.append((now - published_at) * 1_000.0)

        unsubscribe = host.subscribe_front(on_front)
        cleanup.callback(unsubscribe)
        live = host.live_controller(
            initial,
            refresh_interval_ms=refresh_interval_ms,
        )
        cleanup.callback(
            lambda: cleanup_status.__setitem__(
                "live_closed",
                live.close(timeout=min(timeout_seconds, 60.0)),
            )
        )
        snapshot_freeze_ms: list[float] = []
        ingress_publish_ms: list[float] = []

        first_started = perf_counter()
        first = _snapshot(schema, base, 1)
        snapshot_freeze_ms.append((perf_counter() - first_started) * 1_000.0)
        publish_times[1] = perf_counter()
        published_started = perf_counter()
        live.publish(first)
        ingress_publish_ms.append((perf_counter() - published_started) * 1_000.0)
        live.start()

        producer_started = perf_counter()
        producer_period = 1.0 / producer_hz
        next_publish = producer_started + producer_period
        for revision in range(2, frames + 1):
            remaining = next_publish - perf_counter()
            if remaining > 0.0:
                sleep(remaining)
            snapshot_started = perf_counter()
            frame = _snapshot(schema, base, revision)
            snapshot_freeze_ms.append(
                (perf_counter() - snapshot_started) * 1_000.0
            )
            publish_times[revision] = perf_counter()
            published_started = perf_counter()
            live.publish(frame)
            ingress_publish_ms.append((perf_counter() - published_started) * 1_000.0)
            next_publish += producer_period
        producer_elapsed = perf_counter() - producer_started

        deadline = monotonic() + timeout_seconds
        reached_latest = False
        while monotonic() < deadline:
            metrics = live.metrics()
            if metrics.last_presented_revision == frames:
                reached_latest = True
                break
            if metrics.failed_updates:
                break
            sleep(0.05)

        metrics = live.metrics()
        last_error = live.last_error
        errors = () if last_error is None else (str(last_error.error),)
        final_front = host.front
        final_painted = (
            final_front is not None
            and final_front.identity.data_revision == frames
            and _front_has_painted_data(final_front)
        )
    finally:
        cleanup.close()
    live_closed = cleanup_status["live_closed"]
    host_closed = cleanup_status["host_closed"]
    wall_seconds = perf_counter() - case_started
    cpu_after = process.cpu_times() if process is not None else None

    recorded = recorder.snapshot()
    stage_statistics = {
        name: _distribution(values)
        for name, values in sorted(recorded.items())
    }
    stage_statistics["publish_to_promoted_front"] = _distribution(
        publish_to_front_ms
    )
    render_totals = []
    stage_names = (
        "projection_prepare",
        "artist_render_commit",
        "front_capture_and_pack",
    )
    if all(recorded.get(name) for name in stage_names):
        render_totals = [
            sum(parts)
            for parts in zip(
                *(recorded[name] for name in stage_names),
                strict=False,
            )
        ]
    stage_statistics["complete_render_pipeline"] = _distribution(render_totals)

    peak_delta = None
    if rss.baseline_bytes is not None and rss.peak_bytes is not None:
        peak_delta = rss.peak_bytes - rss.baseline_bytes
    cpu_seconds = None
    equivalent_cores = None
    if cpu_before is not None and cpu_after is not None:
        cpu_seconds = (
            cpu_after.user
            + cpu_after.system
            - cpu_before.user
            - cpu_before.system
        )
        equivalent_cores = cpu_seconds / wall_seconds if wall_seconds else None

    complete_p95 = stage_statistics["complete_render_pipeline"]["p95_ms"]
    cadence_budget_met = (
        complete_p95 is not None
        and float(complete_p95) <= float(refresh_interval_ms)
    )
    with callback_lock:
        accepted_revisions = tuple(presented_revisions)
        accepted_times = tuple(presented_at)
    presentation_intervals_ms = [
        (current - previous) * 1_000.0
        for previous, current in zip(
            accepted_times,
            accepted_times[1:],
            strict=False,
        )
    ]
    presentation_hz_observed = (
        (len(accepted_times) - 1) / (accepted_times[-1] - accepted_times[0])
        if len(accepted_times) > 1 and accepted_times[-1] > accepted_times[0]
        else None
    )

    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "logical_cpu_count": os.cpu_count(),
        },
        "configuration": {
            "camera_geometry": geometry,
            "input_shape": list(schema.shape),
            "input_dtype": "uint16",
            "frames_published": frames,
            "producer_hz_requested": producer_hz,
            "refresh_interval_ms": refresh_interval_ms,
            "size_preset": size,
            "device_pixel_ratio": device_pixel_ratio,
            "raster_size": [
                initial_front.buffer.width,
                initial_front.buffer.height,
            ],
            "front_bytes": len(initial_front.buffer.pixels),
        },
        "producer": {
            "elapsed_seconds": producer_elapsed,
            "achieved_hz": (frames - 1) / producer_elapsed,
            "snapshot_freeze": _distribution(snapshot_freeze_ms),
            "ingress_publish": _distribution(ingress_publish_ms),
        },
        "live_metrics": {
            "published": metrics.published,
            "consumed": metrics.consumed,
            "coalesced_updates": metrics.coalesced_updates,
            "update_attempts": metrics.update_attempts,
            "successful_updates": metrics.successful_updates,
            "failed_updates": metrics.failed_updates,
            "cancelled_updates": metrics.cancelled_updates,
            "last_published_revision": metrics.last_published_revision,
            "last_presented_revision": metrics.last_presented_revision,
            "presented_revisions": list(accepted_revisions),
            "presentation_hz_over_case": metrics.successful_updates / wall_seconds,
            "presentation_hz_observed": presentation_hz_observed,
            "presentation_intervals": _distribution(presentation_intervals_ms),
        },
        "stage_timings": stage_statistics,
        "resources": {
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            "equivalent_cpu_cores": equivalent_cores,
            "rss_baseline_bytes": rss.baseline_bytes,
            "rss_peak_bytes": rss.peak_bytes,
            "rss_peak_delta_bytes": peak_delta,
        },
        "validation": {
            "latest_revision_reached": reached_latest,
            "initial_front_painted": initial_painted,
            "final_front_painted": final_painted,
            "cadence_budget_met": cadence_budget_met,
            "live_closed": live_closed,
            "host_closed": host_closed,
            "deadlock_or_timeout": not reached_latest,
            "errors": list(errors),
        },
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument(
        "--geometry",
        choices=("data-dim", "point-topology"),
        default="data-dim",
        help="data-dim is the real camera contract; point-topology is comparison only",
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--producer-hz", type=float, default=20.0)
    parser.add_argument(
        "--refresh-ms",
        type=int,
        choices=(100, 200, 400, 800),
        default=100,
    )
    parser.add_argument("--size", default="8x8")
    parser.add_argument("--dpr", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = profile_camera(
        resolution=args.resolution,
        frames=args.frames,
        producer_hz=args.producer_hz,
        refresh_interval_ms=args.refresh_ms,
        size=args.size,
        device_pixel_ratio=args.dpr,
        timeout_seconds=args.timeout,
        geometry=args.geometry,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    validated = report["validation"]
    return 0 if (
        validated["latest_revision_reached"]
        and validated["initial_front_painted"]
        and validated["final_front_painted"]
        and validated["live_closed"]
        and validated["host_closed"]
        and not validated["errors"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
