"""The TaskConsole Selectors switch owns all plot pointer interaction.

These tests mount REAL zlc_plot raster surfaces in the console's PanelCardView
and drive them through Qt's own pointer routes.  Off leaves the surrounding
page in charge; On gives the plot its selector and navigation gestures.
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


def test_selectors_off_blocks_double_click_facet_focus() -> None:
    """Off means the plot cannot focus a facet cell."""

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
    card.set_selectors_enabled(False)
    assert not widget.interaction_enabled
    cell = next(
        axes
        for axes in widget.presented_front.interaction.axes
        if axes.role == 'facet_cell' and axes.cell_index == 0
    )
    point = axes_center(widget, cell)
    QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton, pos=point)
    QtTest.QTest.mouseDClick(widget, QtCore.Qt.LeftButton, pos=point)
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert host.front.interaction.facet_focus_index is None
    assert committed_selectors(host) == ()
finally:
    if card is not None:
        card.set_surface(None)
    if widget is not None:
        widget.close_adapter()
    host.close(timeout=10)
"""
    )


def test_selectors_off_blocks_area_pan_and_zoom_until_enabled() -> None:
    """Off blocks every plot gesture; On enables the same real Qt stream."""

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
    card.set_selectors_enabled(False)
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

    # Selectors OFF: no selector, zoom, or pan reaches the plot.
    QtTest.QTest.mousePress(widget, QtCore.Qt.LeftButton, pos=start)
    QtTest.QTest.mouseMove(widget, end, delay=10)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.LeftButton, pos=end)
    app.processEvents()
    assert committed_selectors(host) == (), (
        'Selectors OFF must gate selector creation')

    before = viewport(host)
    send_wheel(widget, -120)
    app.processEvents()
    assert viewport(host) == before

    before = viewport(host)
    QtTest.QTest.mousePress(widget, QtCore.Qt.MiddleButton, pos=end)
    QtTest.QTest.mouseMove(widget, start, delay=10)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.MiddleButton, pos=start)
    app.processEvents()
    assert viewport(host) == before
    assert committed_selectors(host) == ()

    # ON: the exact same drag now starts and commits an area selector.
    card.set_selectors_enabled(True)
    assert widget.interaction_enabled
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


def test_a_cards_size_is_its_picture_plus_its_chrome_whatever_it_says() -> None:
    """The strip may not decide how big a card is.

    A card is its title strip plus the picture mounted in it plus its own
    padding, and the board packs it by asking.  Two things follow, and both
    are what broke when the strip was put in the Qt title: the picture keeps
    every pixel the plot planned for it, and a long signal name changes
    nothing about the rectangle -- otherwise renaming a signal reflows the
    whole board, and a card clamped to a smaller rectangle than its layout
    demands clips the figure it was opened to show.
    """

    _run_qt(
        _PROLOGUE
        + """
schema = DatasetSchema.create(
    Axis.create('repeat', size=1),
    PointTable.from_columns({'x': [0.0, 1.0, 2.0, 3.0]}),
    dtype=np.float64,
)
snapshot = DatasetSnapshot(schema, np.arange(4.0).reshape(1, 4), revision=0)
host = RasterPlotHost.from_plot(snapshot, CurvePlot(AxisRef.point('x')), size='2x2')
card, widget = mounted_card(host)
projection = {
    'signal': '@logic/cm/frames', 'kind': 'curve', 'cell_kind': '', 'size': '2x2',
    'interval_ms': 100, 'title': '@logic/cm/frames', 'semantic': {}, 'display': {},
    'fit': {}, 'overlay_signal': '',
}
surface = {
    'data_structure': ((('repeat', 1),), (('x', 4),)),
    'data_scope': (('x', 1.0),),
    'semantic': (), 'display': (), 'fit': (),
}
card.set_interval_choices((100,), 100)
card.set_panel_projection(dict(projection), dict(surface))
app.processEvents()

band = card._title_band
reserved = card.size()
app.processEvents()

# The picture keeps every pixel the plotting package planned for it: the
# card is the strip plus that picture plus its own padding, exactly.
planned = widget.presented_front.logical_size
assert (widget.width(), widget.height()) == planned, (widget.size(), planned)
body_pad = tested_module.scaled_px(2) + tested_module.CARD_PAD
assert card.height() - band.height() - widget.height() == body_pad
assert card.width() - widget.width() == 2 * tested_module.CARD_PAD

# Two lines, and what they say does not move the rectangle.
assert '<br>' in card._title_label.text()
assert '(1)x(4)' in card._title_label.toolTip()
assert 'x=1' in card._title_label.toolTip()
long_name = '@logic/' + 'a-very-long-node-name/' * 6 + 'frames'
card.set_panel_projection({**projection, 'title': long_name}, dict(surface))
app.processEvents()
assert card.size() == reserved, (card.size(), reserved)
assert card.minimumSizeHint().width() <= reserved.width()

# The command sits on the strip, right-padded the same as top and bottom.
right_pad = band.width() - (card.settings_button.x() + card.settings_button.width())
top_pad = card.settings_button.y()
assert right_pad == tested_module.CARD_TITLE_PAD, (right_pad, top_pad)
assert abs((band.height() - card.settings_button.height()) // 2 - top_pad) <= 1

card.set_surface(None)
widget.close_adapter()
host.close(timeout=10)
"""
    )


def test_a_card_takes_the_new_presets_room_the_moment_it_is_picked() -> None:
    """A preset is a size the plotting package already knows.

    It plans the canvas, and its answer does not depend on the kind or the
    cell count -- so the card asks, and is right at once, instead of waiting
    for a picture to be drawn at the new size and being wrong until then.
    The card's own half -- its strip and its margins -- it measures.

    Measured before this: the card kept the old rectangle, so growing clipped
    241 px of plot until a drag re-packed the board, and shrinking left 263 px
    of blank under the strip.
    """

    _run_qt(
        _PROLOGUE
        + """
from zlc_ui.board import panel_display_size

schema = DatasetSchema.create(
    Axis.create('repeat', size=1),
    PointTable.from_columns({'x': [0.0, 1.0, 2.0, 3.0]}),
    dtype=np.float64,
)
snapshot = DatasetSnapshot(schema, np.arange(4.0).reshape(1, 4), revision=0)
host = RasterPlotHost.from_plot(snapshot, CurvePlot(AxisRef.point('x')), size='2x2')
card, widget = mounted_card(host)
card.set_interval_choices((100,), 100)
card.set_size_choices(('1x2', '2x2', '4x4'), '2x2')
band = card._title_band
pad = tested_module.scaled_px(2) + tested_module.CARD_PAD
packed = []
card.geometry_changed.connect(lambda: packed.append(card.size()))


def framed(size):
    # Whatever this context can say a preset's picture measures: with the
    # Workbench present that IS the plotting package's plan, and in this
    # package alone it is zlc_ui's own cell.  Either way the card asks and
    # is right at once, and the picture corrects it if the two differ.
    planned = panel_display_size(size)
    return (
        card.width() == planned[0] + 2 * tested_module.CARD_PAD
        and card.height() == planned[1] + band.height() + pad
    )


for size in ('4x4', '2x2'):
    packed.clear()
    card.set_panel_size(size)
    app.processEvents()
    # At once -- not one front later.
    assert framed(size), (size, card.size(), panel_display_size(size))
    assert packed, 'the card resized without telling the board'
    # And when the picture arrives it fits the room exactly.
    host.set_size(size).result(timeout=20)
    wait_until(
        lambda size=size: (
            widget.presented_front is not None
            and widget.presented_front.identity.preset == size
        ),
        message=f'the plot never redrew at {size}',
    )
    app.processEvents()
    # The picture landed, and the card frames THAT: a preset's planned size
    # and a drawn picture's size are the same number wherever both are known.
    assert card.height() - band.height() - widget.height() == pad, size
    assert card.width() - widget.width() == 2 * tested_module.CARD_PAD, size

card.set_surface(None)
widget.close_adapter()
host.close(timeout=10)
"""
    )
