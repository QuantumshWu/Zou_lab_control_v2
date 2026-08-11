"""The Selectors switch gates selector CREATION, never navigation.

These tests mount REAL zlc_plot raster surfaces in the console's PanelCardView
and drive them through Qt's own pointer routes, because the regression they
guard was exactly a widget-level one: the card closed the plot's whole input
transport when Selectors was off, so the double-click that is the only way to
focus a facet cell -- and pan, and zoom -- were silently dropped.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO_ROOT = ROOT.parents[1]
PLOT_TESTS = REPO_ROOT / "packages" / "zlc_plot" / "tests"


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), str(SRC), str(PLOT_TESTS))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment.setdefault("MPLBACKEND", "Agg")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


_PROLOGUE = """
import time

import zou_lab_control_v2
print(zou_lab_control_v2.__file__)
import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    Qt5PlotWidget,
    ensure_qt5_application,
)
from zlc_plot.raster import RasterPlotHost
from zlc_ui.console import panel_card_view as tested_module
print(tested_module.__file__)
PanelCardView = tested_module.PanelCardView

app = ensure_qt5_application([])


def wait_until(condition, timeout=15.0, message="condition did not settle"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError(message)


def mounted_card(host):
    widget = Qt5PlotWidget(host)
    card = PanelCardView('panel-1', 'Plot')
    card.set_size_choices(('2x2',), '2x2')
    card.set_surface(widget)
    card.show()
    wait_until(
        lambda: widget.presented_front is not None,
        message='the plot never presented its first front',
    )
    return card, widget


def axes_center(widget, axes):
    left, top, right, bottom = axes.bounds
    return QtCore.QPoint(
        int((left + right) / 2.0 * widget.width()),
        int((top + bottom) / 2.0 * widget.height()),
    )


def committed_selectors(host):
    return tuple(host.selectors().result(timeout=10).value)


def viewport(host):
    return host.describe_display().result(timeout=10).value.viewport


def send_wheel(widget, steps):
    center = widget.rect().center()
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(center),
        QtCore.QPointF(widget.mapToGlobal(center)),
        QtCore.QPoint(),
        QtCore.QPoint(0, steps),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(widget, wheel)
"""


def test_double_click_focuses_a_facet_cell_with_selectors_off() -> None:
    """The headline regression: Selectors OFF must not drop facet focus."""

    _run_qt(
        _PROLOGUE
        + """
x = np.linspace(-3.0, 3.0, 20)
facets = np.repeat([0.0, 1.0], 10)
schema = DatasetSchema.create(
    Axis.create('repeat', size=1),
    PointTable.from_columns({'x': x, 'facet': facets}),
    dtype=np.float64,
    generation='card-facet-focus',
)
values = (2.0 * np.exp(-0.5 * ((x - 0.2) / 1.0) ** 2) + 0.2)[None, :]
snapshot = DatasetSnapshot(schema, values, 0)
spec = FacetGridPlot(AxisRef.point('facet'), CurvePlot(AxisRef.point('x')))
host = RasterPlotHost.from_plot(snapshot, spec)
card = widget = None
try:
    card, widget = mounted_card(host)
    assert widget.interaction_enabled, (
        'mounting must keep the plot input transport open')
    # Selectors is OFF by default -- exactly the state the console opens in.
    cell = next(
        axes
        for axes in widget.presented_front.interaction.axes
        if axes.role == 'facet_cell' and axes.cell_index == 0
    )
    point = axes_center(widget, cell)
    # The REAL operator stream: single press/release (gated), then the
    # double-click Qt synthesizes, then its release.
    QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton, pos=point)
    QtTest.QTest.mouseDClick(widget, QtCore.Qt.LeftButton, pos=point)
    # The observable at this seam is the plot session itself: the double-click
    # must ROUTE and focus.  (Whether the focused front is re-presented while
    # the click gesture is still armed is zlc_plot staging behaviour, the same
    # with Selectors ON.)
    wait_until(
        lambda: (
            host.front is not None
            and host.front.interaction.facet_focus_index == 0
        ),
        message='double-click with Selectors OFF did not focus the facet cell',
    )
    assert committed_selectors(host) == (), (
        'facet focus must not leave a selector behind')
finally:
    if card is not None:
        card.set_surface(None)
    if widget is not None:
        widget.close_adapter()
    host.close(timeout=10)
"""
    )


def test_selectors_off_gates_area_drag_while_pan_and_zoom_still_work() -> None:
    """OFF: an area drag creates NO selector; wheel zoom and pan stay live."""

    _run_qt(
        _PROLOGUE
        + """
x = np.linspace(0.0, 1.0, 20)
schema = DatasetSchema.create(
    Axis.create('repeat', size=1),
    PointTable.from_columns({'x': x}),
    dtype=np.float64,
    generation='card-selector-gate',
)
snapshot = DatasetSnapshot(schema, x[None, :], 0)
host = RasterPlotHost.from_plot(snapshot, CurvePlot(AxisRef.point('x')))
card = widget = None
try:
    card, widget = mounted_card(host)
    main_axes = next(
        (
            axes
            for axes in widget.presented_front.interaction.axes
            if axes.role == 'main'
        ),
        widget.presented_front.interaction.axes[0],
    )
    left, top, right, bottom = main_axes.bounds
    start = QtCore.QPoint(
        int((left + (right - left) / 3.0) * widget.width()),
        int((top + (bottom - top) / 3.0) * widget.height()),
    )
    end = QtCore.QPoint(
        int((left + (right - left) * 2.0 / 3.0) * widget.width()),
        int((top + (bottom - top) * 2.0 / 3.0) * widget.height()),
    )

    # Selectors OFF (the default): a left drag must create NO selector.
    QtTest.QTest.mousePress(widget, QtCore.Qt.LeftButton, pos=start)
    QtTest.QTest.mouseMove(widget, end, delay=10)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.LeftButton, pos=end)
    app.processEvents()
    assert committed_selectors(host) == (), (
        'Selectors OFF must gate selector creation')

    # ... while wheel zoom still works.
    before = viewport(host)
    send_wheel(widget, -120)
    wait_until(
        lambda: viewport(host) != before,
        message='wheel zoom was dropped with Selectors OFF',
    )

    # ... and middle-button pan still works.
    before = viewport(host)
    QtTest.QTest.mousePress(widget, QtCore.Qt.MiddleButton, pos=end)
    QtTest.QTest.mouseMove(widget, start, delay=10)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.MiddleButton, pos=start)
    wait_until(
        lambda: viewport(host) != before,
        message='middle-button pan was dropped with Selectors OFF',
    )
    assert committed_selectors(host) == ()

    # ON: the exact same drag now starts and commits an area selector.
    card.set_selectors_enabled(True)
    QtTest.QTest.mousePress(widget, QtCore.Qt.LeftButton, pos=start)
    QtTest.QTest.mouseMove(widget, end, delay=10)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.LeftButton, pos=end)
    wait_until(
        lambda: any(
            state.kind.value == 'area' for state in committed_selectors(host)
        ),
        message='Selectors ON no longer creates an area selector',
    )
finally:
    if card is not None:
        card.set_surface(None)
    if widget is not None:
        widget.close_adapter()
    host.close(timeout=10)
"""
    )


def test_a_mounted_surfaces_error_report_reaches_the_handle() -> None:
    """errorOccurred on a mounted panel widget lands on the console seam."""

    _run_qt(
        """
import zou_lab_control_v2
print(zou_lab_control_v2.__file__)
from PyQt5 import QtCore, QtWidgets
from zlc_ui.console import TaskConsoleHandle, TaskConsoleView, panel_card_view
print(panel_card_view.__file__)
from zlc_ui.qt import ensure_qt_app

app = ensure_qt_app(['panel-plot-error'])


class _PlotSurface(QtWidgets.QWidget):
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.interaction = None

    def set_interaction_enabled(self, enabled):
        self.interaction = bool(enabled)


class _Host:
    def __init__(self):
        self.widget = _PlotSurface()

    def qt_widget(self):
        return self.widget


view = TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_sizes(('2x2',), '2x2')
handle.add_panel('panel-1', 'Camera')
events = []
handle.panel_plot_error.connect(
    lambda panel_id, message: events.append((panel_id, message))
)
first = _Host()
handle.show_panel('panel-1', first)
assert first.widget.interaction is True, (
    'mounting must open the plot input transport')
message = 'the painted pointer front is no longer layout-compatible'
first.widget.errorOccurred.emit(message)
assert events == [('panel-1', message)], events

# Replacing the surface moves the relay with it: the old widget goes quiet,
# the new one reports.
second = _Host()
handle.show_panel('panel-1', second)
first.widget.errorOccurred.emit('stale surface noise')
assert events == [('panel-1', message)], events
second.widget.errorOccurred.emit('second refusal')
assert events == [
    ('panel-1', message),
    ('panel-1', 'second refusal'),
], events
"""
    )
