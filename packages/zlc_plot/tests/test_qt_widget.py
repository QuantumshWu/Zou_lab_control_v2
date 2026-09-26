from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)

from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    Qt5PlotWidget,
)
from zlc_plot.raster import RasterPlotHost

def test_qt_widget_receives_front_and_commits_area_drag(qt_app, pump_until) -> None:
    from PyQt5.QtCore import QEvent, QPoint, Qt
    from PyQt5.QtTest import QTest

    # TWO series on purpose.  The hover below is here to prove a pointer
    # event round-trips to the worker and comes back as a new front, and
    # choosing a series is what a hover DOES -- but a lone line is not a
    # choice, so on one series the hover is now correctly inert and would
    # prove nothing.
    columns = np.linspace(0.0, 1.0, 20)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns(
            {
                "x": np.tile(columns, 2),
                "series": np.repeat((0.0, 1.0), columns.size),
            }
        ),
        dtype=np.float64,
    )
    values = np.concatenate((columns, columns + 0.35)).reshape(1, -1)
    snapshot = make_snapshot(schema, values, 0)
    host = RasterPlotHost.from_plot(
        snapshot,
        CurvePlot(AxisRef.point("x"), group=AxisRef.point("series")),
    )
    widget = None
    try:
        def forbidden_wait(*_args, **_kwargs):
            raise AssertionError("a Qt widget must not wait for the worker's first front")

        host.wait_for_front = forbidden_wait
        widget = Qt5PlotWidget(host)
        widget.show()
        pump_until(lambda: widget.presented_front is not None, 10.0)
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
        pump_until(lambda: widget.presented_front.identity.sequence > sequence, 3.0)
        assert widget.presented_front.identity.sequence > sequence

        sequence = widget.presented_front.identity.sequence
        qt_app.sendEvent(widget, QEvent(QEvent.Leave))
        pump_until(lambda: widget.presented_front.identity.sequence > sequence, 3.0)
        assert widget.presented_front.identity.sequence > sequence

        start = QPoint(max(2, width // 3), max(2, height // 3))
        end = QPoint(max(3, width * 2 // 3), max(3, height * 2 // 3))
        QTest.mousePress(widget, Qt.LeftButton, pos=start)
        QTest.mouseMove(widget, end, delay=10)
        QTest.mouseRelease(widget, Qt.LeftButton, pos=end)
        pump_until(
            lambda: widget.presented_front is not None and any(
                item.kind.value == "area"
                for item in widget.presented_front.interaction.selectors
            ),
            3.0,
        )
        current = widget.presented_front
        assert current is not None
        assert any(item.kind.value == "area" for item in current.interaction.selectors)
    finally:
        if widget is not None:
            widget.close_adapter()
        host.close(timeout=10)

def test_staged_widget_accepts_its_exact_current_front_idempotently(qt_app) -> None:
    from PyQt5 import QtCore, QtGui, QtTest

    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": (0.0, 1.0, 2.0)}),
        dtype=np.float64,
    )
    snapshot = make_snapshot(schema, np.asarray([[1.0, 2.0, 3.0]]), 0)
    host = RasterPlotHost.from_plot(snapshot, CurvePlot(AxisRef.point("x")))
    widget = None
    try:
        host.wait_for_front(timeout=10)
        widget = host.qt_widget(auto_present=False)
        assert host.qt_widget() is widget
        assert host.qt_widget(auto_present=False) is widget
        with pytest.raises(ValueError, match="presentation policy"):
            host.qt_widget(auto_present=True)
        assert widget.presented_front is None
        current = host.front
        assert current is not None

        assert widget.present_front(current) is True
        assert widget.presented_front is current

        newer = host.set_parameter("title", "newer").result(timeout=10).front
        assert newer.identity.sequence > current.identity.sequence
        assert widget.present_front(newer) is True
        assert widget.present_front(current) is False
        assert widget.presented_front is newer

        widget.show(); qt_app.processEvents()
        nx, ny = newer.interaction.axes[0].display_to_normalized(1.0, 2.0)
        start = QtCore.QPointF(nx * widget.width(), ny * widget.height())
        end = start + QtCore.QPointF(12.0, 8.0)
        for kind, point, button in (
            (QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.MiddleButton),
            (QtCore.QEvent.MouseMove, end, QtCore.Qt.NoButton),
        ):
            qt_app.sendEvent(widget, QtGui.QMouseEvent(
                kind, point, button, QtCore.Qt.MiddleButton, QtCore.Qt.NoModifier))
        host.describe_display().result(timeout=10); QtTest.QTest.qWait(20)
        panned = widget.presented_front
        assert panned.identity.sequence > newer.identity.sequence
        assert panned.interaction.axes != newer.interaction.axes

        # A real active pan can preview this shot, but cannot accept the next.
        data_front = host.update_data(make_snapshot(
            schema, np.asarray([[4.0, 5.0, 6.0]]), 1,
            generation=newer.identity.data_generation,
        )).result(timeout=10).front
        QtTest.QTest.qWait(20)
        assert widget.presented_front.identity.data_revision == 0
        qt_app.sendEvent(widget, QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease, end, QtCore.Qt.MiddleButton,
            QtCore.Qt.NoButton, QtCore.Qt.NoModifier))
        qt_app.sendEvent(widget, QtGui.QWheelEvent(
            end, QtCore.QPointF(widget.mapToGlobal(end.toPoint())),
            QtCore.QPoint(), QtCore.QPoint(0, 120), QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False))
        host.describe_display().result(timeout=10); QtTest.QTest.qWait(20)
        assert widget.presented_front.identity.data_revision == 0
        assert data_front.identity.data_revision == 1
        assert widget.present_front(host.front) is True
        assert widget.presented_front.identity.data_revision == 1
    finally:
        if widget is not None:
            widget.close_adapter()
        host.close(timeout=10)

def test_qt_double_click_focus_repaints_a_static_facet_host(qt_app, pump_until) -> None:
    """The focus-rendered front supersedes the in-flight gesture's surface.

    ``_install_front`` used to drop the focused front because the layout
    change replaced the axes the double-click press had captured
    (``replacement_axes != gesture_axes``) -- and on STATIC data no later
    front ever arrived, so the session focused while the widget kept the
    overview pixels forever.  A front whose facet focus differs from the
    gesture front's is the focus transition itself and must repaint.
    """

    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    from zlc_data import SPATIAL_X, SPATIAL_Y
    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )
    from zlc_plot import ImagePlot

    table = mapped_domain_from_columns({"bias": [-1.0, 0.0, 1.0]})
    schema = make_dataset_schema(
        repeat_domain(size=1),
        table,
        cell_axes=(
            axis(
                "sy", values=tuple(float(v) for v in range(40)), role=SPATIAL_Y
            ),
            axis(
                "sx", values=tuple(float(v) for v in range(60)), role=SPATIAL_X
            ),
        ),
        dtype=np.float64,
    )
    frame = np.add.outer(np.arange(40.0), np.arange(60.0))
    values = np.stack([frame + 50.0 * index for index in range(3)])[None]
    host = RasterPlotHost.from_plot(
        make_snapshot(schema, values, 1),
        FacetGridPlot(
            AxisRef.point("bias"),
            ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy")),
        ),
        size="4x4",
    )
    widget = None
    try:
        widget = Qt5PlotWidget(host)
        widget.show()
        pump_until(lambda: widget.presented_front is not None, 10.0)
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
        pump_until(
            lambda: widget.presented_front is not None
            and widget.presented_front.interaction.facet_focus_index == 1,
            10.0,
        )
        current = widget.presented_front
        assert current is not None
        assert current.interaction.facet_focus_index == 1
        assert current.buffer.pixels != overview_pixels
    finally:
        if widget is not None:
            widget.close_adapter()
        host.close(timeout=10)

def test_the_widget_asks_the_screen_before_it_subscribes(qt_app, pump_until) -> None:
    """The correction starts at construction, not one render later.

    Nothing tells a host what screen it is for, so its opening frame is
    rendered at density 1 and painted stretched into the widget.  The
    pixel-ratio observer used to be created only when the FIRST front
    ARRIVED, so the request for the real density went out one whole render
    after that -- the soft frame every rebuilt surface showed, and Panel
    Edit's Refresh replaces its host, so it showed it every time.  Asking
    during construction puts the corrected render in flight while the
    opening frame is still being handed over.

    Refusing to PAINT the density-1 front is NOT the answer: a staged
    console panel is presented exactly once, and a refusal there is read as
    a stale race -- the port closes the host and the panel never appears.
    """

    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": (0.0, 1.0, 2.0)}),
        dtype=np.float64,
    )

    def build_host():
        return RasterPlotHost.from_plot(
            make_snapshot(schema, np.asarray([[0.0, 1.0, 2.0]]), 0),
            CurvePlot(AxisRef.point("x")),
            size="2x2",
        )

    # The observer class is private to the lazily built Qt module; one
    # ordinary widget hands it over.
    probe_host = build_host()
    probe_host.wait_for_front(timeout=30)
    probe = Qt5PlotWidget(probe_host)
    observer = type(probe._pixel_ratio_observer)
    probe.close_adapter()
    probe_host.close(timeout=30)

    host = build_host()
    original = observer.current_ratio
    observer.current_ratio = property(lambda self: 2.0)
    calls: list[str] = []
    try:
        host.wait_for_front(timeout=30)
        assert float(host.front.device_pixel_ratio) == 1.0
        host_dpr = host.set_device_pixel_ratio
        host_subscribe = host.subscribe_front

        def spy_dpr(ratio):
            calls.append(f"dpr:{float(ratio)}")
            return host_dpr(ratio)

        def spy_subscribe(callback):
            calls.append("subscribe")
            return host_subscribe(callback)

        host.set_device_pixel_ratio = spy_dpr
        host.subscribe_front = spy_subscribe
        widget = Qt5PlotWidget(host)
        try:
            assert calls[:2] == ["dpr:2.0", "subscribe"], calls
            # and the widget still SHOWS the opening frame it was handed,
            # whatever density it was rendered at
            assert widget.presented_front is not None
            pump_until(
                lambda: float(widget.presented_front.device_pixel_ratio) == 2.0, 30.0
            )
            assert float(widget.presented_front.device_pixel_ratio) == 2.0
        finally:
            widget.close_adapter()
    finally:
        observer.current_ratio = original
        host.close(timeout=30)

def test_a_bare_hover_is_not_a_hand_but_every_part_of_a_drag_is() -> None:
    """The arbiter's bargain is the hand's pixels for the camera's.

    A drag repaints the thing being dragged on every move, so trading the
    camera's frames for the gesture's is the trade the operator asked for.  A
    bare hover publishes nothing on an image surface, so the same trade gives
    up the camera and buys nothing -- and because every raw move arms a hold
    of at least 40 ms while a pointer reports at 60-125 Hz, the hold is
    renewed three to five times faster than it can expire.  Measured on a live
    console with four panels: still 9.22 fps, hovering 0.11 fps on EVERY panel
    (the arbiter is process-wide), dragging 85 fps.  With the hover excluded,
    hovering measures 9.11 and dragging still measures 90.

    A scroll carries no button and is still a hand: the wheel repaints and the
    operator is waiting for it.
    """

    from zlc_plot.raster import _is_a_hand

    # The whole vocabulary pointer_event accepts, so a new action cannot be
    # added without deciding this question for it.
    assert _is_a_hand("move", False) is False
    assert _is_a_hand("leave", False) is False

    assert _is_a_hand("move", True) is True
    assert _is_a_hand("press", False) is True
    assert _is_a_hand("release", False) is True
    assert _is_a_hand("scroll", False) is True
    assert _is_a_hand("key", False) is True
    assert _is_a_hand("cancel", False) is True

def test_a_drag_stays_a_hand_from_press_to_release(qt_app, pump_until) -> None:
    """Every move of a drag must reach the host carrying its button.

    The hand is decided from the button the widget reports, so anything that
    cleared ``_pointer_button`` in the middle of a gesture would silently
    demote the rest of the drag to hovers and hand the machine back to the
    cameras half way through.  This walks a real press-move-release over a
    real widget and asserts the classification for every event the host saw.
    """

    from PyQt5.QtCore import QPointF, Qt
    from PyQt5.QtGui import QMouseEvent

    from zlc_plot.raster import _is_a_hand

    # Explicit events, not QTest: QTest.mouseMove goes through the platform's
    # cursor, and offscreen it stops delivering once other windows have come
    # and gone -- this test passed alone and saw no moves at all in a full
    # file run.  A QMouseEvent sent to the widget is the same event the
    # platform would deliver and depends on nothing outside it.
    def send(kind, position, button, buttons):
        qt_app.sendEvent(
            widget,
            QMouseEvent(kind, QPointF(*position), button, buttons, Qt.NoModifier),
        )
        qt_app.processEvents()

    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": np.linspace(0.0, 1.0, 20)}),
        dtype=np.float64,
    )
    snapshot = make_snapshot(schema, np.linspace(0.0, 1.0, 20).reshape(1, -1), 0)
    host = RasterPlotHost.from_plot(snapshot, CurvePlot(AxisRef.point("x")))
    widget = None
    seen: list[tuple[str, object]] = []
    original = type(host).pointer_event

    def watched(self, action, x, y, *, button=None, held=False, **kwargs):
        seen.append((str(action), button, bool(held)))
        return original(self, action, x, y, button=button, held=held, **kwargs)

    type(host).pointer_event = watched
    try:
        widget = Qt5PlotWidget(host)
        widget.show()
        pump_until(lambda: widget.presented_front is not None, 10.0)
        assert widget.presented_front is not None

        width, height = widget.width(), widget.height()
        start = (max(2, width // 3), max(2, height // 3))

        # HOVER: moves with no button held.
        seen.clear()
        for step in range(4):
            send(
                QMouseEvent.MouseMove,
                (start[0] + 3 * step, start[1] + 2 * step),
                Qt.NoButton,
                Qt.NoButton,
            )
        hovers = [row for row in seen if row[0] == "move"]
        assert hovers, "the hover did not reach the host at all"
        assert all(not held for _action, _button, held in hovers)
        assert not any(_is_a_hand(action, held) for action, _button, held in hovers)

        # DRAG: press, several moves, release.
        seen.clear()
        send(QMouseEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton)
        for step in range(1, 6):
            send(
                QMouseEvent.MouseMove,
                (start[0] + 12 * step, start[1] + 9 * step),
                Qt.NoButton,
                Qt.LeftButton,
            )
        end = (start[0] + 72, start[1] + 54)
        send(QMouseEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton)

        actions = [action for action, _button, _held in seen]
        assert "press" in actions and "release" in actions
        dragged = [row for row in seen if row[0] == "move"]
        assert dragged, "the drag sent no moves"
        # The widget's own _pointer_button is cleared the moment the press
        # resolves nothing to grab -- which an area rubber-band press does --
        # so it cannot be what the hand is read from.
        assert all(held for _action, _button, held in dragged), (
            "a drag move reached the host with no button held: %s" % (seen,)
        )
        assert all(_is_a_hand(action, held) for action, _button, held in seen), (
            "part of a drag was not classified as a hand: %s" % (seen,)
        )
    finally:
        type(host).pointer_event = original
        if widget is not None:
            widget.close_adapter()
        host.close(timeout=10)
