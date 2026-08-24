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
    ImageFrame,
    ImagePlot,
    ImagePointOverlay,
    PointStatus,
    Qt5PlotWidget,
    ensure_qt5_application,
    review_image_points,
)
from zlc_plot.raster import RasterPlotHost


@pytest.mark.gui
def test_point_review_returns_the_final_excluded_identities() -> None:
    try:
        app = ensure_qt5_application([])
        from PyQt5 import QtCore, QtWidgets
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"sample": (0.0,)}),
        data_axes=(Axis.create("y", size=8), Axis.create("x", size=10)),
        dtype=np.float64,
        generation="point-review-test",
    )
    snapshot = DatasetSnapshot(
        schema, np.arange(80, dtype=float).reshape(1, 1, 8, 10), 0
    )
    overlay = ImagePointOverlay(
        1,
        np.asarray(((2.0, 2.0), (7.0, 5.0))),
        point_ids=("site-a", "site-b"),
        labels=("1", "2"),
        static_statuses=(PointStatus.UNKNOWN, PointStatus.UNKNOWN),
    )
    host = RasterPlotHost.from_plot(
        ImageFrame(snapshot, overlay),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        size="2x2",
    )
    try:
        def answer() -> None:
            dialog = next(
                widget
                for widget in app.topLevelWidgets()
                if isinstance(widget, QtWidgets.QDialog)
                and widget.windowTitle() == "Review sites"
            )
            points = dialog.findChild(QtWidgets.QListWidget)
            assert points is not None
            points.item(1).setCheckState(QtCore.Qt.Unchecked)
            button = next(
                item
                for item in dialog.findChildren(QtWidgets.QPushButton)
                if item.text() == "Continue"
            )
            button.click()

        QtCore.QTimer.singleShot(100, answer)
        assert review_image_points(
            host,
            overlay,
            title="Review sites",
            confirm_label="Continue",
        ) == ("site-b",)
    finally:
        host.close(timeout=10)


@pytest.mark.gui
def test_bound_plot_controls_never_wait_for_the_raster_worker() -> None:
    from concurrent.futures import Future
    from threading import Thread, get_ident
    from types import SimpleNamespace

    try:
        app = ensure_qt5_application([])
        from PyQt5 import QtCore
        from zlc_plot.qt_controls import _qt5_bound_controls_class
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": (0.0, 1.0)}),
        dtype=np.float64,
        generation="qt-controls-test",
    )
    host = RasterPlotHost.from_plot(
        DatasetSnapshot(schema, np.asarray([[0.0, 1.0]]), 0),
        CurvePlot(AxisRef.point("x")),
    )

    owner_thread = get_ident()

    class GuardedFuture(Future):
        def result(self, *args, **kwargs):
            assert get_ident() != owner_thread, (
                "Qt owner resolved a plot operation instead of receiving plain data"
            )
            return super().result(*args, **kwargs)

    pending = GuardedFuture()
    proxy = SimpleNamespace(
        describe_display=lambda: pending,
        set_parameter=lambda _name, _value: GuardedFuture(),
        apply_semantic=lambda _name, _value: GuardedFuture(),
    )
    controls = None
    try:
        description = host.describe_display().result(timeout=10).value
        controls = _qt5_bound_controls_class()(proxy)
        owner_turned: list[bool] = []
        QtCore.QTimer.singleShot(0, lambda: owner_turned.append(True))
        app.processEvents()
        assert owner_turned
        assert controls.panel is None

        resolver = Thread(
            target=lambda: pending.set_result(SimpleNamespace(value=description))
        )
        resolver.start()
        resolver.join()
        deadline = time.monotonic() + 2.0
        while controls.panel is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert controls.panel is not None
    finally:
        if controls is not None:
            controls.close()
        host.close(timeout=10)


@pytest.mark.gui
def test_qt_widget_receives_front_and_commits_area_drag() -> None:
    try:
        app = ensure_qt5_application([])
        from PyQt5.QtCore import QEvent, QPoint, Qt
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

        class WheelEvent:
            accepted = False
            ignored = False

            @staticmethod
            def angleDelta():
                return QPoint(0, 120)

            def accept(self):
                self.accepted = True

            def ignore(self):
                self.ignored = True

        widget.set_interaction_enabled(False)
        wheel = WheelEvent()
        widget.wheelEvent(wheel)
        assert wheel.ignored and not wheel.accepted
        widget.set_interaction_enabled(True)

        width, height = widget.width(), widget.height()
        axis = widget.presented_front.interaction.axes[0]
        nx, ny = axis.display_to_normalized(0.5, 0.5)
        hover = QPoint(round(nx * width), round(ny * height))
        sequence = widget.presented_front.identity.sequence
        QTest.mouseMove(widget, hover, delay=10)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            if widget.presented_front.identity.sequence > sequence:
                break
            time.sleep(0.01)
        assert widget.presented_front.identity.sequence > sequence

        sequence = widget.presented_front.identity.sequence
        app.sendEvent(widget, QEvent(QEvent.Leave))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            if widget.presented_front.identity.sequence > sequence:
                break
            time.sleep(0.01)
        assert widget.presented_front.identity.sequence > sequence

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


@pytest.mark.gui
def test_staged_widget_accepts_its_exact_current_front_idempotently() -> None:
    try:
        ensure_qt5_application([])
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": (0.0, 1.0, 2.0)}),
        dtype=np.float64,
        generation="qt-idempotent-present",
    )
    snapshot = DatasetSnapshot(schema, np.asarray([[1.0, 2.0, 3.0]]), 0)
    host = RasterPlotHost.from_plot(snapshot, CurvePlot(AxisRef.point("x")))
    widget = None
    try:
        host.wait_for_front(timeout=10)
        widget = Qt5PlotWidget(host, auto_present=False)
        current = widget.presented_front
        assert current is not None

        assert widget.present_front(current) is True
        assert widget.presented_front is current

        newer = host.set_parameter("title", "newer").result(timeout=10).front
        assert newer.identity.sequence > current.identity.sequence
        assert widget.present_front(newer) is True
        assert widget.present_front(current) is False
        assert widget.presented_front is newer
    finally:
        if widget is not None:
            widget.close_adapter()
        host.close(timeout=10)


@pytest.mark.gui
def test_qt_double_click_focus_repaints_a_static_facet_host() -> None:
    """The focus-rendered front supersedes the in-flight gesture's surface.

    ``_install_front`` used to drop the focused front because the layout
    change replaced the axes the double-click press had captured
    (``replacement_axes != gesture_axes``) -- and on STATIC data no later
    front ever arrived, so the session focused while the widget kept the
    overview pixels forever.  A front whose facet focus differs from the
    gesture front's is the focus transition itself and must repaint.
    """

    try:
        app = ensure_qt5_application([])
        from PyQt5.QtCore import QPoint, Qt
        from PyQt5.QtTest import QTest
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")

    from zlc_data import SPATIAL_X, SPATIAL_Y
    from data_factory import PointTopology
    from zlc_plot import ImagePlot

    table = PointTable.from_columns({"bias": [-1.0, 0.0, 1.0]})
    topology = PointTopology.from_cartesian(
        (Axis.create("bias", values=[-1.0, 0.0, 1.0]),), point_table=table
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        table,
        data_axes=(
            Axis.create(
                "sy", values=tuple(float(v) for v in range(40)), role=SPATIAL_Y
            ),
            Axis.create(
                "sx", values=tuple(float(v) for v in range(60)), role=SPATIAL_X
            ),
        ),
        point_topology=topology,
        dtype=np.float64,
        generation="qt-focus-test",
    )
    frame = np.add.outer(np.arange(40.0), np.arange(60.0))
    values = np.stack([frame + 50.0 * index for index in range(3)])[None]
    host = RasterPlotHost.from_plot(
        DatasetSnapshot(schema, values, 1),
        FacetGridPlot(
            AxisRef.point_dimension("bias"),
            ImagePlot(AxisRef.data("sx"), AxisRef.data("sy")),
        ),
        size="4x4",
    )
    widget = None
    try:
        widget = Qt5PlotWidget(host)
        widget.show()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and widget.presented_front is None:
            app.processEvents()
            time.sleep(0.005)
        overview = widget.presented_front
        assert overview is not None
        assert overview.interaction.facet_focus_index is None
        overview_pixels = overview.buffer.pixels

        cell = next(
            item
            for item in overview.interaction.axes
            if item.role == "facet_cell" and item.cell_index == 1
        )
        left, top, right, bottom = cell.bounds
        target = QPoint(
            int((left + right) / 2.0 * widget.width()),
            int((top + bottom) / 2.0 * widget.height()),
        )
        QTest.mouseDClick(widget, Qt.LeftButton, pos=target)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            app.processEvents()
            current = widget.presented_front
            if (
                current is not None
                and current.interaction.facet_focus_index == 1
            ):
                break
            time.sleep(0.01)
        current = widget.presented_front
        assert current is not None
        assert current.interaction.facet_focus_index == 1
        assert current.buffer.pixels != overview_pixels
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
