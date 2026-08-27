from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import numpy as np

import zlc_plot.backends as backends


class _Signal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback


class _Timer:
    created = []

    def __init__(self, parent) -> None:
        self.parent = parent
        self.interval = None
        self.timeout = _Signal()
        self.started = False
        self.__class__.created.append(self)

    def setInterval(self, interval) -> None:
        self.interval = interval

    def start(self) -> None:
        self.started = True


def test_ipykernel_wake_timer_uses_dedicated_qt_loop(monkeypatch) -> None:
    """The notebook liveness timer only quits ipykernel's private loop."""

    _Timer.created.clear()
    loop = SimpleNamespace(quit=lambda: setattr(loop, "quit_count", loop.quit_count + 1), quit_count=0)
    shell = SimpleNamespace(
        kernel=SimpleNamespace(app=SimpleNamespace(qt_event_loop=loop))
    )
    modules = SimpleNamespace(
        QtCore=SimpleNamespace(QTimer=_Timer),
        QtWidgets=SimpleNamespace(QApplication=SimpleNamespace(instance=lambda: object())),
    )
    monkeypatch.setattr(backends, "_IPYKERNEL_WAKE_TIMER", None)
    monkeypatch.setattr(backends, "_load_qt5_modules", lambda: modules)

    backends._install_ipykernel_wake_timer(shell)

    assert len(_Timer.created) == 1
    timer = _Timer.created[0]
    assert timer.interval == 50
    assert timer.started is True
    assert timer.timeout.callback is not None
    timer.timeout.callback()
    assert loop.quit_count == 1

    backends._install_ipykernel_wake_timer(shell)
    assert len(_Timer.created) == 1


def test_ipykernel_wake_timer_ignores_shell_without_private_loop(monkeypatch) -> None:
    monkeypatch.setattr(backends, "_IPYKERNEL_WAKE_TIMER", None)
    shell = SimpleNamespace(kernel=SimpleNamespace(app=SimpleNamespace()))
    monkeypatch.setattr(
        backends,
        "_load_qt5_modules",
        lambda: (_ for _ in ()).throw(AssertionError("Qt must not load")),
    )
    backends._install_ipykernel_wake_timer(shell)
    assert backends._IPYKERNEL_WAKE_TIMER is None


def test_a_widget_outliving_its_host_refuses_input_instead_of_raising() -> None:
    """An exception out of a Qt handler kills the application, silently.

    The console retires a plot host whenever a panel retargets, and the
    widget it made can still be mounted and still receive mouse events.  The
    first thing a move asks for is the selector handle radius, which comes
    out of ``RasterPlotHost.defaults`` -- and a closed host answers that by
    raising.  Inside ``mouseMoveEvent`` PyQt does not propagate it: it aborts
    the process, with no traceback, which is what "the panel just stops" and
    "the whole thing disappeared" both look like from the outside.

    So the widget asks whether the host can still serve, and refuses quietly
    when it cannot.  This drives real Qt events at a widget whose host has
    been closed and requires that nothing escapes.
    """

    from PyQt5 import QtCore, QtGui, QtWidgets

    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
    from zlc_plot import (
        AxisRef,
        ImagePlot,
        RasterPlotHost,
        ensure_qt5_application,
    )

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": np.asarray([0.0])}),
        data_axes=(
            Axis.create("y", values=[float(i) for i in range(24)]),
            Axis.create("x", values=[float(i) for i in range(32)]),
        ),
        dtype=np.uint8,
        generation="widget-outlives-host",
    )
    values = np.arange(24 * 32, dtype=np.uint8).reshape(1, 1, 24, 32)

    ensure_qt5_application([])
    host = RasterPlotHost.from_plot(
        DatasetSnapshot(schema, values, revision=1),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
    )
    try:
        host.wait_for_front(10.0)
        widget = host.qt_widget()
        widget.resize(240, 180)
        widget.set_interaction_enabled(True)
    finally:
        # From ANOTHER thread, so the host cannot close the widget for us:
        # that is the arrangement the console produces, and the one where a
        # live widget is left pointing at a dead host.
        closed = threading.Thread(target=lambda: host.close(timeout=5.0))
        closed.start()
        closed.join(10.0)
    assert host.closing
    assert not widget._closed, (
        "the widget closed itself; this test needs one that did not, or it "
        "proves nothing about a widget outliving its host"
    )

    escaped: list[BaseException] = []
    original = sys.excepthook
    sys.excepthook = lambda kind, value, tb: escaped.append(value)
    try:
        centre = QtCore.QPointF(120.0, 90.0)
        for kind, button, buttons in (
            (QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton,
             QtCore.Qt.LeftButton),
            (QtCore.QEvent.MouseMove, QtCore.Qt.NoButton,
             QtCore.Qt.LeftButton),
            (QtCore.QEvent.MouseButtonRelease, QtCore.Qt.LeftButton,
             QtCore.Qt.NoButton),
        ):
            QtWidgets.QApplication.sendEvent(
                widget,
                QtGui.QMouseEvent(
                    kind, centre, button, buttons, QtCore.Qt.NoModifier
                ),
            )
        QtWidgets.QApplication.sendEvent(
            widget,
            QtGui.QWheelEvent(
                centre,
                centre,
                QtCore.QPoint(),
                QtCore.QPoint(0, -120),
                QtCore.Qt.NoButton,
                QtCore.Qt.NoModifier,
                QtCore.Qt.NoScrollPhase,
                False,
            ),
        )
    finally:
        sys.excepthook = original
    assert not escaped, "an exception escaped a Qt handler: %r" % (escaped[0],)
