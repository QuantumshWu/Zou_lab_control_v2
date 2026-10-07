from __future__ import annotations

import sys
import threading

import numpy as np


def test_a_widget_outliving_its_host_refuses_input_instead_of_raising(qt_app) -> None:
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

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )
    from zlc_data import REPEAT, SPATIAL_X, SPATIAL_Y
    from zlc_plot import AxisRef, ImagePlot, RasterPlotHost

    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"shot": np.asarray([0.0])}),
        cell_axes=(
            axis("y", values=[float(i) for i in range(24)], role=SPATIAL_Y),
            axis("x", values=[float(i) for i in range(32)], role=SPATIAL_X),
        ),
        dtype=np.uint8,
    )
    values = np.arange(24 * 32, dtype=np.uint8).reshape(1, 1, 24, 32)

    host = RasterPlotHost.from_plot(
        make_snapshot(schema, values, revision=1),
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
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


def test_a_gesture_is_not_cancelled_by_the_fronts_it_causes(qt_app) -> None:
    """Turning a scene must not stop it turning.

    A camera drag WRITES display state, so every frame it causes carries a
    new display revision.  The widget read that as "the surface changed
    under me" and cancelled the gesture on its own first front: the scene
    turned twice -- for the moves already in flight -- and then answered
    nothing until the button came up, which is what "it moves a little and
    then stops" looks like.  A live panel does the same thing to a selector
    drag, because a tight colour limit re-derives from every shot.

    Kind, preset, layout, size and device pixel ratio are what a gesture's
    transform depends on.  This drives eighteen real middle-button moves
    and requires the camera to answer every one of them.
    """

    from PyQt5 import QtCore, QtGui, QtWidgets

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )
    from zlc_data import REPEAT, SPATIAL_X, SPATIAL_Y
    from zlc_plot import AxisRef, ImagePlot, RasterPlotHost

    rows, columns = 40, 60
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"shot": np.asarray([0.0])}),
        cell_axes=(
            axis("y", values=[float(i) for i in range(rows)], unit="pixel", role=SPATIAL_Y),
            axis("x", values=[float(i) for i in range(columns)], unit="pixel", role=SPATIAL_X),
        ),
        dtype=np.uint8,
    )

    def snapshot(revision: int):
        rng = np.random.default_rng(revision)
        return make_snapshot(
            schema,
            rng.integers(0, 255, size=(rows, columns), dtype=np.uint8)[None, None],
            revision=revision,
        )

    # The screen's OWN ratio: a widget that observes a scale different from
    # its host's rightly cancels the gesture, since the surface really did
    # change under it.  That is a different question from this one.
    screen = qt_app.primaryScreen()
    host = RasterPlotHost.from_plot(
        snapshot(1),
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
        parameters={"presentation": "height_bars"},
        device_pixel_ratio=(
            1.0 if screen is None else float(screen.devicePixelRatio())
        ),
    )
    try:
        host.wait_for_front(20.0)
        widget = host.qt_widget()
        widget.resize(480, 360)
        widget.show()
        for _ in range(40):
            qt_app.processEvents()
        widget.set_interaction_enabled(True)

        def azimuth() -> float:
            values = host.describe_display().result(
                timeout=10
            ).value.display_state.values
            return round(float(values["camera_azimuth"]), 3)

        def at(fx: float, fy: float) -> QtCore.QPointF:
            return QtCore.QPointF(widget.width() * fx, widget.height() * fy)

        def send(kind, position, button, buttons) -> None:
            QtWidgets.QApplication.sendEvent(
                widget,
                QtGui.QMouseEvent(
                    kind, position, button, buttons, QtCore.Qt.NoModifier
                ),
            )
            for _ in range(10):
                qt_app.processEvents()

        send(QtCore.QEvent.MouseButtonPress, at(0.5, 0.5),
             QtCore.Qt.MiddleButton, QtCore.Qt.MiddleButton)
        seen = []
        for step in range(1, 19):
            host.update_data(snapshot(1 + step))
            send(QtCore.QEvent.MouseMove, at(0.5 + 0.02 * step, 0.5),
                 QtCore.Qt.NoButton, QtCore.Qt.MiddleButton)
            seen.append(azimuth())
        send(QtCore.QEvent.MouseButtonRelease, at(0.86, 0.5),
             QtCore.Qt.MiddleButton, QtCore.Qt.NoButton)
    finally:
        host.close(timeout=10.0)
    assert len(set(seen)) == len(seen), (
        "the scene stopped following the hand after %d of %d moves: %s"
        % (len(set(seen)), len(seen), seen)
    )
