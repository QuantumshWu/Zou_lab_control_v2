import zou_lab_control

import os
import time

import pytest

import zlc_workbench.console as tested_module


print(tested_module.__file__)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_workbench.apps.task_console import build_console
from zlc_workbench.logic import stable_signal_key
from zlc_workbench.session import ExperimentSession
from zlc_ui.qt import ensure_qt_app

from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


def _wait_until(predicate, presenter, *, timeout: float = 10.0) -> None:
    from PyQt5 import QtWidgets

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        presenter.beat()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("TaskConsole interaction did not settle before timeout")


def _commit_area(host) -> None:
    """Commit one real pointer gesture through the raster interaction seam."""

    front = host.wait_for_front(5.0)
    axes = front.interaction.axes[0]
    left, bottom, right, top = axes.bounds

    def point(x_fraction: float, y_fraction: float) -> tuple[float, float]:
        return (
            left + (right - left) * x_fraction,
            bottom + (top - bottom) * y_fraction,
        )

    start, end = point(0.25, 0.25), point(0.75, 0.75)
    for action, location in (("press", start), ("move", end), ("release", end)):
        host._pointer_event(
            action,
            location[0],
            location[1],
            button=1,
            identity=front.identity,
            axes=axes,
            interaction=front.interaction,
        ).result()


def _wheel_frozen_snapshot(editor, host, *, delta: int) -> None:
    """Send a wheel event to the QWidget actually hit inside Panel Edit."""

    from PyQt5 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance()
    assert app is not None
    surface = editor._surface
    assert surface is host.qt_widget()
    scroll = editor.findChild(QtWidgets.QScrollArea)
    assert scroll is not None
    scroll.ensureWidgetVisible(surface, 0, 0)
    app.processEvents()
    global_position = surface.mapToGlobal(surface.rect().center())
    target = editor.childAt(editor.mapFromGlobal(global_position))
    assert target is surface, (
        "the frozen plot is not the QWidget hit at its visible centre: "
        f"{target!r}"
    )
    local_position = target.mapFromGlobal(global_position)
    before = host.describe_display().result().value.viewport
    event = QtGui.QWheelEvent(
        QtCore.QPointF(local_position),
        QtCore.QPointF(global_position),
        QtCore.QPoint(),
        QtCore.QPoint(0, int(delta)),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(target, event)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        app.processEvents()
        if host.describe_display().result().value.viewport != before:
            return
        time.sleep(0.005)
    raise AssertionError("wheel on the embedded frozen plot did not change its viewport")


def _visible_logic_roi(editor) -> tuple[int, int, int, int]:
    values = editor.form.read_all()
    return tuple(
        int(values[name])
        for name in ("roi_x", "roi_y", "roi_width", "roi_height")
    )


def test_guard_b_task_console_selector_updates_shared_draft_and_logic_restart_restarts(
    tmp_path,
) -> None:
    plot = pytest.importorskip("zlc_plot")
    app = ensure_qt_app(["guard-b-task-console-interaction"])
    session = ExperimentSession.open(tmp_path, template="virtual")
    write_ordinary_pulse(tmp_path)
    view, presenter = build_console(session)
    try:
        session.load_pulse(PULSE_NAME)
        node_id = presenter.add_logic("camera_measurement")
        assert node_id in view._logic_editors

        authored_roi = (4, 3, 32, 24)
        view.logic_draft_changed.emit(
            node_id,
            {
                "device_keys": {"camera": "camera"},
                "values": {
                    "exposure_seconds": 0.013,
                    "roi_x": authored_roi[0],
                    "roi_y": authored_roi[1],
                    "roi_width": authored_roi[2],
                    "roi_height": authored_roi[3],
                    "repeat": 0,
                    "frames_per_cycle": CAMERA_WINDOWS,
                },
            },
        )
        assert presenter.start_logic(node_id) is True
        old_host = presenter.logic[node_id].host
        assert old_host is not None and old_host.running
        signal_key = stable_signal_key(node_id, "frames")
        old_generation = old_host.generation

        _wait_until(lambda: session.camera.capture_state(), presenter)
        session.fire(shots=1)
        _wait_until(
            lambda: session.signal_plane.freeze().publication(signal_key) is not None,
            presenter,
        )
        panel = presenter.add_selected_panel("image")
        assert panel is not None and panel.kind == "image"
        view.panel_state_changed.emit(
            panel.panel_id, {"signal": signal_key}
        )
        _wait_until(lambda: panel.host is not None, presenter)
        assert panel.signal == signal_key and panel.host is not None

        # Selector Off is the default.  On the formal mounted TaskConsole card
        # its wheel belongs to the outer board, not the plot viewport.
        from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

        window = view._view
        window.tabs.setCurrentIndex(0)
        top = view._window if view._window is not None else window
        top.resize(620, 280)
        top.show()
        app.processEvents()
        card = view._cards[panel.panel_id]
        surface = card.surface
        assert getattr(surface, "host", None) is panel.host
        assert not surface.interaction_enabled
        board_scroll = window.scroll.verticalScrollBar()
        window.board.setMinimumHeight(window.scroll.viewport().height() + 500)
        app.processEvents()
        assert board_scroll.maximum() > 0
        board_scroll.setValue(0)
        before_viewport = panel.host.describe_display().result().value.viewport
        centre = surface.rect().center()
        wheel = QtGui.QWheelEvent(
            QtCore.QPointF(centre),
            QtCore.QPointF(surface.mapToGlobal(centre)),
            QtCore.QPoint(),
            QtCore.QPoint(0, -120),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
            QtCore.Qt.NoScrollPhase,
            False,
        )
        QtWidgets.QApplication.sendEvent(surface, wheel)
        deadline = time.monotonic() + 2.0
        while board_scroll.value() == 0 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert board_scroll.value() > 0
        assert panel.host.describe_display().result().value.viewport == before_viewport

        assert presenter.edit_panel(panel.panel_id) is True
        _wait_until(lambda: panel.editor_host is not None, presenter)
        assert panel.editor_host is not None and panel.editor_host is not panel.host
        _wait_until(
            lambda: panel.editor_selections is not None
            and panel.frozen_data is not None
            and panel.frozen_data.description is not None,
            presenter,
        )
        panel_editor = view._panel_editors[panel.panel_id]
        assert panel_editor._surface is panel.editor_host.qt_widget()
        assert not panel_editor._surface.interaction_enabled
        panel_editor.open_producer_button.click()
        app.processEvents()
        logic_editor = view._logic_editors[node_id]
        assert window.tabs.currentWidget() is logic_editor
        assert view.focus_panel_editor(panel.panel_id)
        app.processEvents()

        # The real header switch hands all pointer gestures to both the live
        # card and its already-open frozen Edit surface.
        QtTest.QTest.mouseClick(
            window.selectors_switch,
            QtCore.Qt.LeftButton,
        )
        _wait_until(
            lambda: (
                presenter._deriving
                and surface.interaction_enabled
                and panel_editor._surface.interaction_enabled
            ),
            presenter,
        )

        _commit_area(panel.editor_host)
        _wait_until(
            lambda: _visible_logic_roi(logic_editor) != authored_roi,
            presenter,
        )
        draft = presenter.logic[node_id].draft.values
        selected_roi = tuple(
            int(draft[name])
            for name in ("roi_x", "roi_y", "roi_width", "roi_height")
        )
        assert selected_roi != authored_roi, (
            "committing the Image Area selector left the producer row ROI draft unchanged"
        )
        assert _visible_logic_roi(logic_editor) == selected_roi

        _wheel_frozen_snapshot(panel_editor, panel.editor_host, delta=120)
        _wait_until(lambda: panel.interaction_viewport is not None, presenter)
        assert _visible_logic_roi(logic_editor) == selected_roi
        visible = panel.editor_host.describe_display().result().value.viewport
        assert visible is not None
        assert visible.x.span > authored_roi[2]
        assert visible.y.span > authored_roi[3]

        # Taking the region away leaves the form saying what the camera is
        # actually set to -- the ROI the running node was started with -- so
        # restarting from here changes nothing.  It used to write the zoomed
        # VIEWPORT into the producer instead, which re-pointed the camera at
        # whatever happened to be on screen.
        panel.editor_host.remove_selector(plot.SelectorKind.AREA).result()
        _wait_until(lambda: panel.state.selector == {}, presenter)
        _wait_until(
            lambda: _visible_logic_roi(logic_editor) == authored_roi,
            presenter,
        )
        assert _visible_logic_roi(logic_editor) == authored_roi
        logic_values = view._logic_editors[node_id].form.read_all()
        assert tuple(
            int(logic_values[name])
            for name in ("roi_x", "roi_y", "roi_width", "roi_height")
        ) == authored_roi

        assert presenter.start_logic(node_id) is True
        _wait_until(lambda: presenter.logic[node_id].host is not old_host, presenter)
        replacement = presenter.logic[node_id].host
        assert replacement is not None and replacement.running
        assert replacement.signal_key("frames") == signal_key
        assert replacement.generation != old_generation
        actual = session.camera.working_point()
        assert actual is not None
        assert (
            int(actual.roi_origin_yx[1]),
            int(actual.roi_origin_yx[0]),
            int(actual.roi_shape_yx[1]),
            int(actual.roi_shape_yx[0]),
        ) == authored_roi
    finally:
        presenter.close()
        _wait_until(presenter.close, presenter)
        view.set_close_guard(lambda: True)
        view.close()
        app.processEvents()
        session.close()
