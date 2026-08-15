from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_atom.devices.slm import canonical_phase
from zlc_atom.install import DeviceSpec, create_installation
from zlc_ui import ensure_qt_app
from zlc_workbench.device_use import (
    DeviceClaim,
    DeviceUseBusy,
    DeviceUseCoordinator,
)
from zlc_workbench.session import Workspace


def test_slm_control_factory_is_plugin_owned_and_lazy() -> None:
    script = r"""
import zou_lab_control_v2
import sys
import zlc_atom.install as tested
print(zou_lab_control_v2.__file__)
print(tested.__file__)
assert "PyQt5" not in sys.modules
assert "zlc_plot" not in sys.modules
assert "zlc_ui" not in sys.modules
assert "zlc_workbench" not in sys.modules
items = {item.type_id: item for item in tested.discover_device_catalog().available}
factory = items["slm.virtual"].control_factory
assert factory is not None
assert factory.__module__ == "zlc_atom.devices.slm"
assert "PyQt5" not in sys.modules
assert "zlc_plot" not in sys.modules
assert "zlc_ui" not in sys.modules
assert "zlc_workbench" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _session(tmp_path: Path):
    installation = create_installation((DeviceSpec("slm", "slm.virtual"),))
    return SimpleNamespace(
        installation=installation,
        workspace=Workspace(tmp_path).prepare(),
        device_use=DeviceUseCoordinator(),
    )


def _pump(app, predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("Qt-owned SLM operation did not finish")


def _dispose(control, app) -> None:
    hosts = [control._target_host, control._phase_host]
    if hasattr(control, "_wavefront_host"):
        hosts.append(control._wavefront_host)
    widgets = tuple(host.qt_widget() for host in hosts)
    body = control._body
    control._finish_close()
    deadline = time.monotonic() + 5.0
    while not control._cleaned and time.monotonic() < deadline:
        app.processEvents()
        control._finish_close()
        time.sleep(0.002)
    assert control._cleaned
    body.close()
    body.deleteLater()
    for widget in widgets:
        widget.close()
        widget.deleteLater()
    control.deleteLater()
    app.sendPostedEvents(None, 0)
    app.processEvents()


def test_editor_keeps_only_latest_solve_and_clear_does_not_drive_hardware(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    device = session.installation.device("slm")
    incoming = device.last_commanded_phase
    first_started = threading.Event()
    calls: list[np.ndarray] = []

    def controlled_solve(target, *, stop_requested, **_kwargs):
        calls.append(np.array(target, copy=True))
        if len(calls) == 1:
            first_started.set()
            while not stop_requested():
                time.sleep(0.001)
            raise InterruptedError("superseded")
        phase = canonical_phase(np.full(target.shape, 0.37), target.shape)
        return phase, {"method": "test", "iterations": 1}

    monkeypatch.setattr(editor, "solve_phase", controlled_solve)
    control = editor.SlmEditorControl(session, "slm")
    try:
        assert first_started.wait(2.0)
        middle = np.full(control.shape, 0.5, dtype=np.float32)
        latest = np.full(control.shape, 0.8, dtype=np.float32)
        control.set_target(middle)
        control.set_target(latest)
        _pump(app, lambda: control.solver_idle)
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[-1], latest)
        np.testing.assert_array_equal(
            control._phase,
            canonical_phase(np.full(control.shape, 0.37), control.shape),
        )
        np.testing.assert_array_equal(device.last_commanded_phase, incoming)

        count = len(calls)
        control.set_target(np.zeros(control.shape, dtype=np.float32))
        app.processEvents()
        assert len(calls) == count
        assert control.status_text == "Empty target; hardware unchanged"
        np.testing.assert_array_equal(device.last_commanded_phase, incoming)
    finally:
        monkeypatch.setattr(editor, "solve_phase", lambda *_args, **_kwargs: None)
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()
        del control, session
        app.processEvents()


def test_send_refuses_a_phase_stale_against_the_latest_target(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    device = session.installation.device("slm")
    incoming = np.array(device.last_commanded_phase, copy=True)
    started, release = threading.Event(), threading.Event()
    calls = 0
    latest_phase = canonical_phase(
        np.full(device.shape_yx, 0.83), device.shape_yx
    )

    def controlled_solve(target, *, stop_requested, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            release.wait(2.0)
            if stop_requested():
                raise InterruptedError("superseded")
        return latest_phase, {"method": "test", "iterations": 1}

    monkeypatch.setattr(editor, "solve_phase", controlled_solve)
    control = editor.SlmEditorControl(session, "slm")
    try:
        assert started.wait(2.0)
        control.set_target(np.full(control.shape, 0.75, dtype=np.float32))
        assert control.send() is False
        assert "latest target" in control.status_text
        np.testing.assert_array_equal(device.last_commanded_phase, incoming)

        release.set()
        _pump(app, lambda: control.solver_idle)
        assert control.send() is True
        _pump(app, lambda: not control.command_active)
        np.testing.assert_array_equal(device.last_commanded_phase, latest_phase)
    finally:
        release.set()
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_toggle_is_binary_while_brush_uses_authored_intensity(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    objectives: list[object] = []

    def solved(target, **kwargs):
        objectives.append(kwargs.get("objective_kind"))
        return (
            canonical_phase(np.zeros(target.shape), target.shape),
            {"method": "test", "iterations": 1},
        )

    monkeypatch.setattr(editor, "solve_phase", solved)
    control = editor.SlmEditorControl(session, "slm")
    try:
        control._body.resize(1100, 650)
        control._body.show()
        control._layer_tabs.setCurrentIndex(control._editor_tab_index)
        _pump(app, lambda: control._target_widget.presented_front is not None)
        row, column = control.shape[0] // 2, control.shape[1] // 2
        axes = next(
            item
            for item in control._target_widget.presented_front.interaction.axes
            if item.role == "image"
        )
        nx, ny = axes.display_to_normalized(column, row)
        point = QtCore.QPoint(
            round(nx * control._target_widget.width()),
            round(ny * control._target_widget.height()),
        )

        _pump(app, lambda: control.solver_idle)
        control.set_target(np.zeros(control.shape, dtype=np.float32))
        control._intensity.setValue(7.5)
        control._mode.setCurrentText("Toggle")
        assert control._paint(point)
        _pump(app, lambda: control.solver_idle)
        assert objectives[-1] == "spots"
        np.testing.assert_array_equal(
            np.unique(control._target[control._target > 0.0]),
            np.array([1.0], dtype=np.float32),
        )
        assert np.count_nonzero(control._target) == 1
        assert control._paint(point)
        assert not np.any(control._target)

        control._mode.setCurrentText("Brush")
        assert control._paint(point)
        _pump(app, lambda: control.solver_idle)
        assert objectives[-1] == "spots"
        assert control._paint(point)
        assert np.any(control._target == 7.5)
        assert np.max(control._target) == 7.5
    finally:
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_selectors_switch_changes_real_target_events_from_paint_to_plot(
    tmp_path: Path, monkeypatch,
) -> None:
    """One visible switch makes zoom/select available without editing target."""

    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    monkeypatch.setattr(
        editor,
        "solve_phase",
        lambda target, **_kwargs: (
            canonical_phase(np.zeros(target.shape), target.shape),
            {"method": "test", "iterations": 1},
        ),
    )
    control = editor.SlmEditorControl(session, "slm")
    try:
        control._body.resize(1100, 650)
        control._body.show()
        control._layer_tabs.setCurrentIndex(control._editor_tab_index)
        _pump(app, lambda: control._target_widget.presented_front is not None)
        assert control._selectors.text() == "Selectors"
        assert not control._selectors.isChecked()
        assert not control._target_host.interaction_enabled
        assert not control._phase_host.interaction_enabled

        axis = next(
            item
            for item in control._target_widget.presented_front.interaction.axes
            if item.role == "image"
        )
        left, top, right, bottom = axis.bounds
        start = QtCore.QPoint(
            round((left + (right - left) / 3.0) * control._target_widget.width()),
            round((top + (bottom - top) / 3.0) * control._target_widget.height()),
        )
        end = QtCore.QPoint(
            round((left + 2.0 * (right - left) / 3.0) * control._target_widget.width()),
            round((top + 2.0 * (bottom - top) / 3.0) * control._target_widget.height()),
        )

        control.set_target(np.zeros(control.shape, dtype=np.float32))
        QtTest.QTest.mouseClick(
            control._target_widget, QtCore.Qt.LeftButton, pos=start
        )
        assert np.any(control._target > 0.0), "Selectors off must retain target paint"

        control._selectors.setChecked(True)
        app.processEvents()
        assert control._target_host.interaction_enabled
        assert control._phase_host.interaction_enabled
        assert not control._mode.isEnabled() and not control._intensity.isEnabled()
        target_before_selector = np.array(control._target, copy=True)
        QtTest.QTest.mousePress(
            control._target_widget, QtCore.Qt.LeftButton, pos=start
        )
        QtTest.QTest.mouseMove(control._target_widget, end, delay=10)
        QtTest.QTest.mouseRelease(
            control._target_widget, QtCore.Qt.LeftButton, pos=end
        )
        _pump(
            app,
            lambda: any(
                state.kind.value == "area"
                for state in control._target_host.selectors().result(timeout=2).value
            ),
        )
        np.testing.assert_array_equal(control._target, target_before_selector)

        before_viewport = control._target_host.describe_display().result(
            timeout=2
        ).value.viewport
        center = QtCore.QPoint(
            round((left + right) * control._target_widget.width() / 2.0),
            round((top + bottom) * control._target_widget.height() / 2.0),
        )
        wheel = QtGui.QWheelEvent(
            QtCore.QPointF(center),
            QtCore.QPointF(control._target_widget.mapToGlobal(center)),
            QtCore.QPoint(),
            QtCore.QPoint(0, -120),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
            QtCore.Qt.NoScrollPhase,
            False,
        )
        QtWidgets.QApplication.sendEvent(control._target_widget, wheel)
        _pump(
            app,
            lambda: control._target_host.describe_display().result(
                timeout=2
            ).value.viewport != before_viewport,
        )

        control._selectors.setChecked(False)
        app.processEvents()
        assert not control._target_host.interaction_enabled
        assert not control._phase_host.interaction_enabled
        assert control._mode.isEnabled() and control._intensity.isEnabled()
    finally:
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_editor_files_send_busy_and_close_have_exact_phase_lifecycle(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    device = session.installation.device("slm")
    incoming = device.last_commanded_phase
    phase = canonical_phase(np.full(device.shape_yx, 1.25), device.shape_yx)

    monkeypatch.setattr(
        editor,
        "solve_phase",
        lambda target, **_kwargs: (
            canonical_phase(np.full(target.shape, 0.2), target.shape),
            {"method": "test", "iterations": 1},
        ),
    )
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        control.set_phase(phase, {})
        target_path = tmp_path / "target.json"
        phase_path = tmp_path / "phase.npz"
        control.save_target(target_path)
        control.save_phase(phase_path)
        assert target_path.is_file() and phase_path.is_file()
        assert editor.load_phase(phase_path)[1] == {}
        np.testing.assert_array_equal(device.last_commanded_phase, incoming)

        blocker = object()
        lease = session.device_use.acquire_command(
            blocker, "feedback task",
            (DeviceClaim("slm", "slm", device, "exclusive"),),
        )
        try:
            assert control.send() is True
            _pump(app, lambda: not control.command_active)
            assert "feedback task" in control.status_text
            np.testing.assert_array_equal(device.last_commanded_phase, incoming)
        finally:
            lease.release()

        assert control.send() is True
        _pump(app, lambda: not control.command_active)
        np.testing.assert_array_equal(device.last_commanded_phase, phase)
        session.device_use.assert_idle()

        original_apply = device.apply_phase
        monkeypatch.setattr(
            device,
            "apply_phase",
            lambda radians: canonical_phase(
                np.zeros(device.shape_yx), device.shape_yx
            ),
        )
        assert control.send() is True
        _pump(app, lambda: not control.command_active)
        assert "did not confirm" in control.status_text
        session.device_use.assert_idle()

        def fail_apply(_radians):
            raise RuntimeError("adapter write failed")

        monkeypatch.setattr(device, "apply_phase", fail_apply)
        assert control.send() is True
        _pump(app, lambda: not control.command_active)
        assert control.status_text == "adapter write failed"
        session.device_use.assert_idle()
        monkeypatch.setattr(device, "apply_phase", original_apply)

        commanded = device.last_commanded_phase
        control._finish_close()
        np.testing.assert_array_equal(device.last_commanded_phase, commanded)
        session.device_use.assert_idle()
    finally:
        monkeypatch.setattr(editor, "solve_phase", lambda *_args, **_kwargs: None)
        _dispose(control, app)
        session.installation.close()
        del control, session
        app.processEvents()


def test_mask_wavefront_crop_compose_and_science_phase_roundtrip(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    monkeypatch.setattr(
        editor,
        "solve_phase",
        lambda target, **_kwargs: (
            canonical_phase(np.zeros(target.shape), target.shape),
            {"method": "test", "iterations": 1},
        ),
    )
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        height, width = control.shape
        yy, xx = np.ogrid[:height, :width]
        full_y, full_x = np.ogrid[
            -1.0:1.0:height * 1j, -1.0:1.0:width * 1j,
        ]
        mask = canonical_phase(
            0.2 + 0.7 * (full_x + 1.0) + 0.3 * (full_y + 1.0),
            control.shape,
        )
        control.set_mask_phase(mask, {"source": "authored mask"})
        control._steering_enabled.setChecked(True)
        control._zernike_enabled.setChecked(True)
        control._carrier_x.setValue(1.25)
        control._carrier_y.setValue(-0.5)
        control._zernike["defocus"].setValue(0.125)
        control._crop_x.setValue(2)
        control._crop_y.setValue(3)
        control._crop_width.setValue(width - 5)
        control._crop_height.setValue(height - 7)
        control._crop_enabled.setChecked(True)

        center_x, center_y = control._pupil_center_xy
        radius_x, radius_y = control._pupil_radius_xy
        zx = (xx - center_x) / radius_x
        zy = (yy - center_y) / radius_y
        r2 = zx * zx + zy * zy
        pupil = r2 <= 1.0
        expected_wavefront = np.pi * (1.25 * full_x - 0.5 * full_y)
        expected_wavefront[pupil] += (
            2.0 * np.pi * 0.125 * np.sqrt(3.0) * (2.0 * r2[pupil] - 1.0)
        )
        np.testing.assert_allclose(
            control._wavefront_phase,
            canonical_phase(expected_wavefront, control.shape),
            rtol=0.0,
            atol=5e-6,
        )
        expected = np.array(expected_wavefront, copy=True)
        expected[3:height - 4, 2:width - 3] += mask[3:height - 4, 2:width - 3]
        expected = canonical_phase(expected, control.shape)
        np.testing.assert_allclose(control._phase, expected, rtol=0.0, atol=5e-6)
        assert control._phase_request_revision == control._request_revision

        mask_path = tmp_path / "mask.npz"
        final_path = tmp_path / "final.npz"
        control.save_mask(mask_path)
        control.save_phase(final_path)
        loaded_mask, mask_metadata = editor.load_phase(mask_path)
        loaded_final, final_metadata = editor.load_phase(final_path)
        np.testing.assert_array_equal(loaded_mask, mask)
        assert mask_metadata == {"source": "authored mask"}
        np.testing.assert_array_equal(loaded_final, expected)
        assert final_metadata["source"] == "composite"
        assert final_metadata["hardware_correction"] == "excluded"

        control.set_phase(loaded_final, {"loaded": "final"})
        np.testing.assert_array_equal(control._mask_phase, loaded_final)
        np.testing.assert_array_equal(control._phase, loaded_final)
        assert control._carrier_x.value() == 0.0
        assert control._carrier_y.value() == 0.0
        assert all(spin.value() == 0.0 for spin in control._zernike.values())
        assert not control._crop_enabled.isChecked()
        assert not control._steering_enabled.isChecked()
        assert not control._zernike_enabled.isChecked()
        assert control._phase_metadata == {"loaded": "final"}

        control.set_mask_phase(np.zeros(control.shape), {})
        control._steering_enabled.setChecked(True)
        control._carrier_x.setValue(1.0)
        pixel_step = np.remainder(
            float(control._phase[0, 1]) - float(control._phase[0, 0]),
            2.0 * np.pi,
        )
        np.testing.assert_allclose(pixel_step, 2.0 * np.pi / (width - 1), atol=1e-6)
        control._carrier_x.setValue(0.0)
        control._steering_enabled.setChecked(False)
        control._zernike_enabled.setChecked(True)
        control._zernike["defocus"].setValue(0.25)
        np.testing.assert_array_equal(control._phase[~pupil], 0.0)

        assert editor._ZERNIKE[:2] == (
            ("tilt_x", "Noll Z2 · X tilt", "2x"),
            ("tilt_y", "Noll Z3 · Y tilt", "2y"),
        )
        control._zernike["defocus"].setValue(0.0)
        terms = {
            "tilt_x": 2.0 * zx,
            "tilt_y": 2.0 * zy,
            "defocus": np.sqrt(3.0) * (2.0 * r2 - 1.0),
            "astig_oblique": 2.0 * np.sqrt(6.0) * zx * zy,
            "astig_vertical": np.sqrt(6.0) * (zx * zx - zy * zy),
            "coma_y": np.sqrt(8.0) * zy * (3.0 * r2 - 2.0),
            "coma_x": np.sqrt(8.0) * zx * (3.0 * r2 - 2.0),
            "trefoil_y": np.sqrt(8.0) * zy * (3.0 * zx * zx - zy * zy),
            "trefoil_x": np.sqrt(8.0) * zx * (zx * zx - 3.0 * zy * zy),
            "spherical": np.sqrt(5.0) * (
                6.0 * r2 * r2 - 6.0 * r2 + 1.0
            ),
        }
        for key, values in terms.items():
            control._zernike[key].setValue(0.125)
            expected_mode = np.zeros(control.shape, dtype=np.float64)
            expected_mode[pupil] = (
                2.0 * np.pi * 0.125
                * np.broadcast_to(values, control.shape)[pupil]
            )
            np.testing.assert_allclose(
                control._phase,
                canonical_phase(expected_mode, control.shape),
                rtol=0.0,
                atol=5e-6,
            )
            control._zernike[key].setValue(0.0)
    finally:
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_editor_keeps_the_original_plot_size_and_resizes_both_scrollable_surfaces(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor
    from zlc_ui.fluent import FluentScrollArea, FluentTabWidget

    app = ensure_qt_app()
    session = _session(tmp_path)
    monkeypatch.setattr(
        editor,
        "solve_phase",
        lambda target, **_kwargs: (
            canonical_phase(np.zeros(target.shape), target.shape),
            {"method": "test", "iterations": 1},
        ),
    )
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        tabs = control._layer_tabs
        assert isinstance(tabs, FluentTabWidget)
        assert [tabs.tabText(index) for index in range(tabs.count())] == [
            "Mask", "Wavefront",
        ]
        assert tabs.currentIndex() == control._editor_tab_index
        assert not any(
            text in {"Mask ROI", "Advanced"}
            for text in (
                tabs.tabText(index) for index in range(tabs.count())
            )
        )

        control._body.resize(1024, 577)
        control._body.show()
        _pump(
            app,
            lambda: (
                control._target_host.logical_size == (490, 357)
                and control._phase_host.logical_size == (490, 357)
                and control._wavefront_host.logical_size == (490, 357)
                and control._target_widget.size() == QtCore.QSize(490, 357)
                and control._phase_widget.size() == QtCore.QSize(490, 357)
                and not control._target_widget.geometry().intersects(
                    control._phase_widget.geometry()
                )
            ),
        )
        assert control._plot_panel.height() >= 350
        assert isinstance(control._plot_scroll, FluentScrollArea)
        assert not control._plot_scroll.widgetResizable()
        assert control._plot_scroll.widget() is control._plot_panel
        assert control._plot_size.currentText() == "2x2"
        assert tuple(
            control._plot_size.itemText(index)
            for index in range(control._plot_size.count())
        ) == editor.PANEL_SIZE_NAMES
        assert control._target_widget.size() == QtCore.QSize(490, 357)
        assert control._phase_widget.size() == QtCore.QSize(490, 357)
        assert control._target_widget is not control._phase_widget
        assert control._target_host is not control._phase_host
        assert control._wavefront_host is not control._target_host
        assert control._wavefront_host is not control._phase_host
        assert not control._target_widget.geometry().intersects(
            control._phase_widget.geometry()
        )
        assert control._plot_panel.layout().stretch(0) == 1
        assert control._plot_panel.layout().stretch(1) == 1

        tabs.setCurrentIndex(control._wavefront_tab_index)
        _pump(
            app,
            lambda: control._wavefront_widget.size() == QtCore.QSize(490, 357),
        )
        assert isinstance(control._wavefront_parameter_scroll, FluentScrollArea)
        assert isinstance(control._wavefront_plot_scroll, FluentScrollArea)
        assert control._wavefront_widget is not control._target_widget
        assert control._wavefront_widget is not control._phase_widget
        assert not control._wavefront_parameter_scroll.geometry().intersects(
            control._wavefront_plot_scroll.geometry()
        )
        assert all(key in control._zernike for key, *_ in editor._ZERNIKE)
        tabs.setCurrentIndex(control._editor_tab_index)

        large_index = control._plot_size.findText("4x4")
        control._plot_size.setCurrentIndex(large_index)
        control._plot_size.activated[int].emit(large_index)
        _pump(
            app,
            lambda: (
                control._target_host.logical_size == (826, 609)
                and control._phase_host.logical_size == (826, 609)
                and control._wavefront_host.logical_size == (826, 609)
                and control._target_widget.size() == QtCore.QSize(826, 609)
                and control._phase_widget.size() == QtCore.QSize(826, 609)
                and not control._target_widget.geometry().intersects(
                    control._phase_widget.geometry()
                )
                and control._plot_scroll.horizontalScrollBar().maximum() > 0
                and control._plot_scroll.verticalScrollBar().maximum() > 0
            ),
        )
        assert control._target_widget.size() == QtCore.QSize(826, 609)
        assert control._phase_widget.size() == QtCore.QSize(826, 609)
        assert control._wavefront_widget.size() == QtCore.QSize(826, 609)
        assert not control._target_widget.geometry().intersects(
            control._phase_widget.geometry()
        )
        assert control.status_text == "Plot size 4x4; hardware unchanged"

        default_index = control._plot_size.findText("2x2")
        control._plot_size.setCurrentIndex(default_index)
        control._plot_size.activated[int].emit(default_index)
        _pump(
            app,
            lambda: (
                control._target_host.logical_size == (490, 357)
                and control._phase_host.logical_size == (490, 357)
                and control._wavefront_host.logical_size == (490, 357)
                and control._target_widget.size() == QtCore.QSize(490, 357)
                and control._phase_widget.size() == QtCore.QSize(490, 357)
                and not control._target_widget.geometry().intersects(
                    control._phase_widget.geometry()
                )
            ),
        )

        control._zernike["coma_x"].setValue(0.25)
        control._carrier_x.setValue(3.0)
        control._reset_wavefront()
        assert control._carrier_x.value() == 0.0
        assert all(spin.value() == 0.0 for spin in control._zernike.values())
    finally:
        control._body.hide()
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_preset_popup_materializes_each_authored_target_only_on_apply(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    calls: list[tuple[np.ndarray, object]] = []

    def controlled_solve(target, **kwargs):
        calls.append((np.array(target, copy=True), kwargs.get("objective_kind")))
        return canonical_phase(np.zeros(target.shape), target.shape), {
            "method": "test", "iterations": 1,
        }

    monkeypatch.setattr(editor, "solve_phase", controlled_solve)
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        before = len(calls)
        control._preset_type.setCurrentText("Grid")
        for key, value in {
            "rows": 3, "columns": 4, "spacing_y": 12,
            "spacing_x": 14, "intensity": 0.7,
        }.items():
            control._preset_fields[key].setValue(value)
        assert len(calls) == before
        control._apply_preset_popup()
        _pump(app, lambda: control.solver_idle)
        np.testing.assert_array_equal(
            control._target,
            editor.preset_grid(
                control.shape, (3, 4), spacing_yx=(12, 14), intensity=0.7,
            ),
        )
        assert calls[-1][1] == "spots"

        cases = (
            (
                "Checkerboard",
                {"rows": 3, "columns": 4, "spacing_y": 12,
                 "spacing_x": 14, "intensity_a": 1.0, "intensity_b": 0.25},
                lambda: editor.preset_checkerboard(
                    control.shape, (3, 4), spacing_yx=(12, 14),
                    intensity_a=1.0, intensity_b=0.25,
                ),
                "spots",
            ),
            (
                "Rectangle", {"height": 20, "width": 28, "edge": 3,
                              "intensity": 0.8},
                lambda: editor.preset_rectangle(
                    control.shape, (20, 28), edge=3, intensity=0.8,
                ),
                "image",
            ),
            (
                "Ellipse", {"radius_y": 10, "radius_x": 16, "edge": 2,
                            "intensity": 0.6},
                lambda: editor.preset_ellipse(
                    control.shape, (10, 16), edge=2, intensity=0.6,
                ),
                "image",
            ),
            (
                "Ring", {"radius": 14, "ring_width": 4, "edge": 2,
                         "intensity": 0.9},
                lambda: editor.preset_ring(
                    control.shape, radius=14, width=4, edge=2, intensity=0.9,
                ),
                "image",
            ),
        )
        for kind, values, expected, objective in cases:
            control._preset_type.setCurrentText(kind)
            for key, value in values.items():
                control._preset_fields[key].setValue(value)
            control._apply_preset_popup()
            _pump(app, lambda: control.solver_idle)
            np.testing.assert_array_equal(control._target, expected())
            assert calls[-1][1] == objective

        control._preset_type.setCurrentText("Grid")
        control._preset_fields["rows"].setValue(control.shape[0])
        control._preset_fields["spacing_y"].setValue(control.shape[0])
        target_before = np.array(control._target, copy=True)
        revision_before, calls_before = control._request_revision, len(calls)
        control._preset_popup.show()
        app.processEvents()
        assert control._preset_popup.isVisible()
        control._apply_preset_popup()
        np.testing.assert_array_equal(control._target, target_before)
        assert control._request_revision == revision_before
        assert len(calls) == calls_before
        assert control._preset_popup.isVisible()
        assert "does not fit" in control.status_text
    finally:
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_input_pupil_is_draft_until_apply_and_reaches_the_solver_as_amplitude(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    pupils: list[np.ndarray] = []

    def controlled_solve(target, **kwargs):
        pupils.append(np.array(kwargs["pupil_amplitude"], copy=True))
        return canonical_phase(np.zeros(target.shape), target.shape), {
            "method": "test", "iterations": 1,
        }

    monkeypatch.setattr(editor, "solve_phase", controlled_solve)
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        assert not hasattr(control, "_pupil_enabled")
        assert [control._pupil_type.itemText(i) for i in range(control._pupil_type.count())] == [
            "Assumed ellipse", "Measured",
        ]
        before = len(pupils)
        control._pupil_center_x.setValue(61.0)
        control._pupil_center_y.setValue(63.0)
        control._pupil_radius_x.setValue(42.0)
        control._pupil_radius_y.setValue(38.0)
        assert len(pupils) == before
        control._apply_pupil()
        _pump(app, lambda: control.solver_idle)
        yy, xx = np.ogrid[:control.shape[0], :control.shape[1]]
        expected = (
            ((xx - 61.0) / 42.0) ** 2
            + ((yy - 63.0) / 38.0) ** 2
            <= 1.0
        ).astype(np.float32)
        np.testing.assert_array_equal(pupils[-1], expected)
        assert control._pupil_center_xy == (61.0, 63.0)
        assert control._pupil_radius_xy == (42.0, 38.0)
        assert "assumed ellipse" in control._pupil_status.text().lower()
        applied_description = control._pupil_applied_description

        intensity = np.linspace(
            0.0, 1.0, num=np.prod(control.shape), dtype=np.float32,
        ).reshape(control.shape)
        path = tmp_path / "measured_pupil.npy"
        np.save(path, intensity)
        control._load_pupil_path(path)
        assert "measured_pupil.npy" in control._pupil_status.text()
        control._carrier_x.setValue(0.25)
        assert control._phase_metadata["input_pupil"] == applied_description
        control._apply_pupil()
        _pump(app, lambda: control.solver_idle)
        support = expected.astype(bool)
        np.testing.assert_allclose(
            pupils[-1], np.sqrt(intensity) * support, rtol=0.0, atol=1e-7,
        )
        assert "measured_pupil.npy" in control._pupil_applied_description
    finally:
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_measured_pupil_and_zernike_share_the_configured_unit_ellipse(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    pupils: list[np.ndarray] = []

    def controlled_solve(target, **kwargs):
        pupils.append(np.array(kwargs["pupil_amplitude"], copy=True))
        return canonical_phase(np.zeros(target.shape), target.shape), {
            "method": "test", "iterations": 1,
        }

    monkeypatch.setattr(editor, "solve_phase", controlled_solve)
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        control._pupil_center_x.setValue(61.0)
        control._pupil_center_y.setValue(43.0)
        control._pupil_radius_x.setValue(27.0)
        control._pupil_radius_y.setValue(19.0)
        intensity = np.full(control.shape, 4.0, dtype=np.float32)
        path = tmp_path / "beam.npy"
        np.save(path, intensity)
        control._load_pupil_path(path)
        control._apply_pupil()
        _pump(app, lambda: control.solver_idle)

        yy, xx = np.ogrid[:control.shape[0], :control.shape[1]]
        support = ((xx - 61.0) / 27.0) ** 2 + ((yy - 43.0) / 19.0) ** 2 <= 1.0
        np.testing.assert_array_equal(pupils[-1] > 0.0, support)
        np.testing.assert_array_equal(pupils[-1][support], 2.0)
        np.testing.assert_array_equal(pupils[-1][~support], 0.0)

        control._steering_enabled.setChecked(False)
        control._zernike_enabled.setChecked(True)
        control._zernike["defocus"].setValue(0.25)
        assert np.any(control._wavefront_phase[support] > 0.0)
        np.testing.assert_array_equal(control._wavefront_phase[~support], 0.0)
        assert control._phase_metadata["input_pupil"] == (
            "measured beam.npy · configured unit ellipse"
        )
    finally:
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_editor_exposes_x15213_correction_load_and_enable_without_sending(
    tmp_path: Path, monkeypatch,
) -> None:
    from PIL import Image
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    device = session.installation.device("slm")
    state = {"path": "", "enabled": False}

    monkeypatch.setattr(
        type(device), "correction_path",
        property(lambda _self: state["path"]), raising=False,
    )
    monkeypatch.setattr(
        type(device), "correction_name",
        property(lambda _self: Path(state["path"]).name), raising=False,
    )
    monkeypatch.setattr(
        type(device), "correction_enabled",
        property(lambda _self: state["enabled"]), raising=False,
    )

    def load_correction(_self, path) -> None:
        state["path"] = str(Path(path))
        state["enabled"] = True

    def set_correction_enabled(_self, enabled: bool) -> None:
        if enabled and not state["path"]:
            raise RuntimeError("no correction loaded")
        state["enabled"] = bool(enabled)

    monkeypatch.setattr(
        type(device), "load_correction", load_correction, raising=False,
    )
    monkeypatch.setattr(
        type(device), "set_correction_enabled", set_correction_enabled,
        raising=False,
    )
    monkeypatch.setattr(
        editor,
        "solve_phase",
        lambda target, **_kwargs: (
            canonical_phase(np.zeros(target.shape), target.shape),
            {"method": "test", "iterations": 1},
        ),
    )
    correction = tmp_path / "correction_Pattern.bmp"
    Image.fromarray(
        np.arange(np.prod(device.shape_yx), dtype=np.uint8).reshape(
            device.shape_yx
        ),
        mode="L",
    ).save(correction)
    monkeypatch.setattr(editor, "fluent_open_path", lambda *_args: str(correction))
    incoming = np.array(device.last_commanded_phase, copy=True)
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        control._body.show()
        app.processEvents()
        assert control._correction_load.isVisible()
        assert control._correction_load.isEnabled()
        assert not control._correction_enabled.isChecked()
        assert "off" in control._correction_status.text().lower()

        QtTest.QTest.mouseClick(control._correction_load, QtCore.Qt.LeftButton)
        assert state == {"path": str(correction), "enabled": True}
        assert control._correction_enabled.isChecked()
        assert correction.name in control._correction_status.text()
        np.testing.assert_array_equal(device.last_commanded_phase, incoming)

        control._correction_enabled.setChecked(False)
        assert state["enabled"] is False
        assert "off" in control._correction_status.text().lower()
        assert correction.name in control._correction_status.text()
        np.testing.assert_array_equal(device.last_commanded_phase, incoming)
    finally:
        control._body.hide()
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_target_solve_updates_mask_then_recomposes_without_weakening_stale_send(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    started, release = threading.Event(), threading.Event()
    calls = 0
    solved_mask = canonical_phase(
        np.full(session.installation.device("slm").shape_yx, 0.4),
        session.installation.device("slm").shape_yx,
    )

    def controlled_solve(target, *, stop_requested, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return canonical_phase(np.zeros(target.shape), target.shape), {
                "method": "test", "iterations": 1,
            }
        started.set()
        release.wait(2.0)
        if stop_requested():
            raise InterruptedError("superseded")
        return solved_mask, {"method": "test", "iterations": 1}

    monkeypatch.setattr(editor, "solve_phase", controlled_solve)
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        control._carrier_x.setValue(0.75)
        control.set_target(np.full(control.shape, 0.8, dtype=np.float32))
        assert started.wait(2.0)
        control._carrier_y.setValue(0.25)
        assert control.status_text == "Solving latest target…"
        assert control.send() is False
        assert "latest target" in control.status_text
        release.set()
        _pump(app, lambda: control.solver_idle)

        xx = np.linspace(-1.0, 1.0, control.shape[1])[None, :]
        yy = np.linspace(-1.0, 1.0, control.shape[0])[:, None]
        expected = canonical_phase(
            solved_mask + np.pi * (0.75 * xx + 0.25 * yy),
            control.shape,
        )
        np.testing.assert_array_equal(control._mask_phase, solved_mask)
        np.testing.assert_allclose(control._phase, expected, rtol=0.0, atol=5e-6)
        assert control._phase_request_revision == control._request_revision
        assert control.send() is True
        _pump(app, lambda: not control.command_active)
        np.testing.assert_array_equal(
            session.installation.device("slm").last_commanded_phase,
            control._phase,
        )
    finally:
        release.set()
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_import_8bit_mask_image_is_raw_phase_not_target_or_correction(
    tmp_path: Path, monkeypatch,
) -> None:
    from PIL import Image
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    monkeypatch.setattr(
        editor,
        "solve_phase",
        lambda target, **_kwargs: (
            canonical_phase(np.zeros(target.shape), target.shape),
            {"method": "test", "iterations": 1},
        ),
    )
    control = editor.SlmEditorControl(session, "slm")
    try:
        _pump(app, lambda: control.solver_idle)
        target_before = np.array(control._target, copy=True)
        control._crop_x.setValue(4)
        control._crop_y.setValue(5)
        control._crop_width.setValue(6)
        control._crop_height.setValue(7)
        control._crop_enabled.setChecked(True)
        gray = np.arange(42, dtype=np.uint8).reshape(7, 6)
        path = tmp_path / "phase_mask.bmp"
        Image.fromarray(gray, mode="L").save(path)
        control.import_mask_image(path)

        expected_mask = np.zeros(control.shape, dtype=np.float64)
        expected_mask[5:12, 4:10] = gray * (2.0 * np.pi / 256.0)
        expected_mask = canonical_phase(expected_mask, control.shape)
        np.testing.assert_array_equal(control._mask_phase, expected_mask)
        np.testing.assert_array_equal(control._phase, expected_mask)
        np.testing.assert_array_equal(control._target, target_before)
        assert "not a correction" in control._import_mask.toolTip().lower()

        bad = tmp_path / "rgb.png"
        Image.fromarray(np.zeros((7, 6, 3), dtype=np.uint8), mode="RGB").save(bad)
        with np.testing.assert_raises_regex(ValueError, "8-bit grayscale"):
            control.import_mask_image(bad)
    finally:
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_editor_close_guard_never_waits_for_a_running_solver(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    started, release = threading.Event(), threading.Event()

    def slow_solve(target, **_kwargs):
        started.set()
        release.wait(2.0)
        raise InterruptedError("closing")

    monkeypatch.setattr(editor, "solve_phase", slow_solve)
    control = editor.SlmEditorControl(session, "slm")
    timer = threading.Timer(0.2, release.set)
    try:
        assert started.wait(2.0)
        timer.start()
        began = time.monotonic()
        assert control._finish_close() is False
        assert time.monotonic() - began < 0.05
        assert not control._body.isEnabled()
        _pump(app, lambda: control.solver_idle)
        deadline = time.monotonic() + 2.0
        while not control._finish_close() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.002)
        assert control._cleaned
    finally:
        release.set()
        timer.cancel()
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()


def test_send_command_keeps_qt_responsive_holds_lease_and_close_retries(
    tmp_path: Path, monkeypatch,
) -> None:
    import zlc_atom.devices.slm.editor as editor

    app = ensure_qt_app()
    session = _session(tmp_path)
    device = session.installation.device("slm")
    clicked_phase = canonical_phase(
        np.linspace(0.0, 8.0, np.prod(device.shape_yx)).reshape(device.shape_yx),
        device.shape_yx,
    )
    later_phase = canonical_phase(
        np.full(device.shape_yx, 2.75), device.shape_yx
    )
    solve_started, solve_release = threading.Event(), threading.Event()
    solve_threads: list[int] = []

    def slow_solve(_target, **_kwargs):
        solve_threads.append(threading.get_ident())
        solve_started.set()
        solve_release.wait(2.0)
        raise InterruptedError("superseded")

    monkeypatch.setattr(editor, "solve_phase", slow_solve)
    control = editor.SlmEditorControl(session, "slm")
    original_apply = device.apply_phase
    apply_started, apply_release = threading.Event(), threading.Event()
    apply_finished = threading.Event()
    apply_threads: list[int] = []

    def slow_apply(radians):
        apply_threads.append(threading.get_ident())
        began = time.monotonic()
        apply_started.set()
        apply_release.wait(2.0)
        time.sleep(max(0.0, began + 0.08 - time.monotonic()))
        try:
            return original_apply(radians)
        finally:
            apply_finished.set()

    monkeypatch.setattr(device, "apply_phase", slow_apply)
    heartbeat: list[float] = []
    timer = QtCore.QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeat.append(time.monotonic()))
    close_attempts: list[float] = []
    control._window = SimpleNamespace(
        close=lambda: (
            close_attempts.append(time.monotonic()), control._finish_close()
        )[-1]
    )
    try:
        assert solve_started.wait(1.0)
        control.set_phase(clicked_phase, {})
        timer.start()
        owner_thread = threading.get_ident()
        began = time.monotonic()
        assert control.send() is True
        assert time.monotonic() - began < 0.04
        assert not control._send.isEnabled()
        assert apply_started.wait(1.0)
        assert not solve_release.is_set()
        solve_release.set()
        _pump(app, lambda: control.solver_idle)

        control.set_phase(later_phase, {})
        assert control.send() is False
        assert "already in progress" in control.status_text
        with np.testing.assert_raises(DeviceUseBusy):
            session.device_use.acquire_command(
                object(), "other command",
                (DeviceClaim("slm", "slm", device, "exclusive"),),
            )

        _pump(app, lambda: len(heartbeat) >= 3)
        assert not apply_finished.is_set()
        began = time.monotonic()
        assert control._finish_close() is False
        assert time.monotonic() - began < 0.04
        assert not control._body.isEnabled()

        apply_release.set()
        _pump(app, lambda: control._cleaned)
        assert close_attempts
        assert len(solve_threads) == 1
        assert len(apply_threads) == 1
        assert solve_threads[0] != owner_thread
        assert apply_threads[0] != owner_thread
        assert apply_threads[0] != solve_threads[0]
        np.testing.assert_array_equal(device.last_commanded_phase, clicked_phase)
        session.device_use.assert_idle()
    finally:
        solve_release.set()
        apply_release.set()
        timer.stop()
        monkeypatch.setattr(device, "apply_phase", original_apply)
        _dispose(control, app)
        session.device_use.assert_idle()
        session.installation.close()
