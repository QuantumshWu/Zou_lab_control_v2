"""Show a continuously updating Rolling plot in a PyQt5 window.

Run from the repository root::

    python examples/live_simulation.py

The producer publishes immutable revisions at 10 Hz.  The visible surface uses
the configured latest-only display cadence and keeps running until the window
is closed.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys

# This checkout must win over any installed zlc_* distribution.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import zou_lab_control_v2  # noqa: F401

import numpy as np

try:
    from examples._role_data import DatasetSnapshot
except ModuleNotFoundError:
    from _role_data import DatasetSnapshot
from zlc_plot import (
    BackendUnavailableError,
    DEFAULTS,
    AxisRef,
    PlotLabels,
    Qt5PlotWidget,
    RasterPlotHost,
    RollingPlot,
    ensure_qt5_application,
)

try:
    from examples.static_and_fit import make_demo_snapshot
except ModuleNotFoundError:
    from static_and_fit import make_demo_snapshot


_LIVE_TEMPLATE = make_demo_snapshot(revision=0)


def make_live_snapshot(revision: int) -> DatasetSnapshot:
    """Return a revision with an unmistakable peak moving across repeat."""

    template = _LIVE_TEMPLATE
    values = np.array(template.block.values, dtype=np.float64, copy=True)
    repeat_count = template.block.schema.repeat_axis.size
    repeat_index = np.arange(repeat_count, dtype=np.float64)
    center = float((2 * revision) % repeat_count)
    distance = np.abs(repeat_index - center)
    distance = np.minimum(distance, repeat_count - distance)
    moving_peak = 1.4e-3 * np.exp(-0.5 * (distance / 3.5) ** 2)
    moving_ripple = 0.18e-3 * np.sin(
        2.0 * np.pi * repeat_index / repeat_count + 0.22 * revision
    )
    values += (moving_peak + moving_ripple)[:, np.newaxis, np.newaxis]
    return DatasetSnapshot(template.block.schema, values, revision=revision)


def run() -> int:
    try:
        from PyQt5 import QtCore, QtWidgets
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailableError(
            "live_simulation requires `pip install -e .[qt]`"
        ) from error

    with ExitStack() as cleanup:
        app = ensure_qt5_application(sys.argv)
        timeout = DEFAULTS.runtime.shutdown_timeout_ms / 1000.0
        initial = make_live_snapshot(0)
        plot_host = RasterPlotHost.from_plot(
            initial,
            RollingPlot(
                labels=PlotLabels(
                    title="Live rolling signal",
                    x="Shots ago",
                    y="Signal",
                ),
            ),
            size="2x4",
            parameters={"window": 64, "side_distribution": True},
        )
        cleanup.callback(plot_host.close, timeout=timeout)
        window = QtWidgets.QWidget()
        window.setWindowTitle("zlc_plot continuous live example")
        layout = QtWidgets.QVBoxLayout(window)
        status = QtWidgets.QLabel("Starting producer …", window)
        layout.addWidget(status)
        host = Qt5PlotWidget(
            plot_host,
            window,
        )
        cleanup.callback(host.close_adapter)
        layout.addWidget(host, 1)

        live = plot_host.live_controller(
            initial,
            refresh_interval_ms=DEFAULTS.live.default_refresh_interval_ms,
        ).start()
        cleanup.callback(live.close, timeout=timeout)
        revision = 0

        producer = QtCore.QTimer(window)
        cleanup.callback(producer.stop)

        def publish() -> None:
            nonlocal revision
            revision += 1
            live.publish(make_live_snapshot(revision))

        producer.timeout.connect(publish)
        producer.start(DEFAULTS.live.refresh_intervals_ms[0])

        status_timer = QtCore.QTimer(window)
        cleanup.callback(status_timer.stop)

        def show_status() -> None:
            metrics = live.metrics()
            status.setText(
                f"published={metrics.last_published_revision}  "
                f"displayed={metrics.last_presented_revision}  "
                f"coalesced={metrics.coalesced_updates}"
            )

        status_timer.timeout.connect(show_status)
        status_timer.start(DEFAULTS.live.default_refresh_interval_ms)
        host.errorOccurred.connect(lambda message: status.setText(str(message)))

        def fit_named_surface() -> None:
            window.adjustSize()
            window.setFixedSize(window.sizeHint())

        host.surfaceChanged.connect(
            lambda _plan: QtCore.QTimer.singleShot(0, fit_named_surface)
        )
        window.show()
        QtCore.QTimer.singleShot(0, fit_named_surface)
        result = int(app.exec_())
        producer.stop()
        status_timer.stop()
        live_closed = live.close(timeout=timeout)
        host.close_adapter()
        host_closed = plot_host.close(timeout=timeout)
        if not live_closed or not host_closed:
            raise RuntimeError("plot workers did not stop before GUI shutdown")
        return result


def _main() -> int:
    try:
        return run()
    except BackendUnavailableError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
