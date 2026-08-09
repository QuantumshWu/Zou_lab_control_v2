from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from test_facet_live_fit import _facet_snapshot, _spec as facet_spec
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    Qt5PlotWidget,
    ensure_qt5_application,
)
from zlc_plot.raster import RasterPlotHost


@pytest.mark.gui
def test_qt_widget_receives_front_and_commits_area_drag() -> None:
    try:
        app = ensure_qt5_application([])
        from PyQt5.QtCore import QPoint, Qt
        from PyQt5.QtTest import QTest
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": np.linspace(0.0, 1.0, 20)}),
        dtype=np.float64,
        generation="qt-widget-test",
    )
    snapshot = DatasetSnapshot(schema, np.linspace(0.0, 1.0, 20).reshape(1, -1), 0)
    host = RasterPlotHost.from_plot(snapshot, CurvePlot(AxisRef.point("x")))
    widget = None
    try:
        def forbidden_wait(*_args, **_kwargs):
            raise AssertionError("a Qt widget must not wait for the worker's first front")

        host.wait_for_front = forbidden_wait
        widget = Qt5PlotWidget(host)
        widget.show()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and widget.presented_front is None:
            app.processEvents()
            time.sleep(0.005)
        assert widget.presented_front is not None
        width, height = widget.width(), widget.height()
        start = QPoint(max(2, width // 3), max(2, height // 3))
        end = QPoint(max(3, width * 2 // 3), max(3, height * 2 // 3))
        QTest.mousePress(widget, Qt.LeftButton, pos=start)
        QTest.mouseMove(widget, end, delay=10)
        QTest.mouseRelease(widget, Qt.LeftButton, pos=end)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            current = widget.presented_front
            if current is not None and any(item.kind.value == "area" for item in current.interaction.selectors):
                break
            time.sleep(0.01)
        current = widget.presented_front
        assert current is not None
        assert any(item.kind.value == "area" for item in current.interaction.selectors)
    finally:
        if widget is not None:
            widget.close_adapter()
        host.close(timeout=10)


def test_qt_raster_host_accepts_facet_grid_spec() -> None:
    spec = facet_spec()
    assert isinstance(spec, FacetGridPlot)
    host = RasterPlotHost.from_plot(_facet_snapshot(), spec)
    try:
        front = host.wait_for_front(timeout=10)
        assert front.identity.kind == "facet_grid"
        assert len(front.interaction.axes) >= 2
    finally:
        host.close(timeout=10)
