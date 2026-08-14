"""The phase-only SLM plugin's target/phase control window."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import threading

import numpy as np
from PyQt5 import QtCore, QtWidgets
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, ImagePlot, PlotLabels, RasterPlotHost
from zlc_ui.fluent import (
    FluentButton, FluentComboBox, FluentDoubleSpinBox, FluentFrame,
    fluent_open_path, fluent_save_path, open_fluent_window,
)

from zlc_atom.data import snapshot_from_array
from .device import SlmAdapter, canonical_phase
from .solver import (
    imported_target, load_phase, load_target, preset_checkerboard,
    preset_ellipse, preset_grid, preset_rectangle, preset_ring, save_phase,
    save_target, solve_phase, validate_target,
)


def _snapshot(values: np.ndarray, signal: str, revision: int):
    return snapshot_from_array(
        values[None], producer="slm_editor", signal=signal,
        roles=(SPATIAL_Y, SPATIAL_X), generation="control", revision=revision,
    )


def _host(values: np.ndarray, signal: str, title: str) -> RasterPlotHost:
    prefix = f"slm_editor.{signal}"
    return RasterPlotHost.from_plot(
        _snapshot(values, signal, 0),
        ImagePlot(
            AxisRef.data(f"{prefix}.1.spatial-x"),
            AxisRef.data(f"{prefix}.0.spatial-y"),
            labels=PlotLabels(title=title, x="x", y="y"),
        ),
        size="2x2",
    )


class SlmEditorControl(QtCore.QObject):
    """Plugin-local handle, brush controller, and latest-only solve owner."""

    closed = QtCore.pyqtSignal()
    _solve_ready = QtCore.pyqtSignal(object)
    _command_ready = QtCore.pyqtSignal(object)

    def __init__(self, session: object, device_key: str) -> None:
        super().__init__()
        self.session, self.device_key = session, str(device_key)
        self.device = session.installation.device(self.device_key)
        if not isinstance(self.device, SlmAdapter):
            raise TypeError("SLM Editor requires a canonical SlmAdapter")
        self.shape = tuple(self.device.shape_yx)
        self._target = preset_grid(self.shape, (5, 7))
        self._phase = canonical_phase(self.device.last_commanded_phase, self.shape)
        self._phase_metadata: dict[str, object] = {"source": "device"}
        self._request_revision = self._target_revision = self._phase_revision = 0
        self._pending: tuple[int, np.ndarray] | None = None
        self._running = self._painting = self._closed = self._cleaned = False
        self._command_active = False
        self._window = None
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slm-solve")
        self._command_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="slm-command"
        )
        self._target_host = _host(self._target, "target", "Target intensity")
        self._phase_host = _host(self._phase, "phase", "Canonical phase (rad)")
        self._target_host.set_interaction_enabled(False)
        self._phase_host.set_interaction_enabled(False)
        self._body = self._build_body()
        self._solve_ready.connect(self._finish_solve)
        self._command_ready.connect(
            self._finish_send, QtCore.Qt.QueuedConnection
        )
        self._queue_solve()

    def _build_body(self) -> QtWidgets.QWidget:
        body = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(body)
        root.setContentsMargins(16, 12, 16, 12)
        edit, controls = FluentFrame(body), QtWidgets.QHBoxLayout()
        edit.setLayout(controls)
        controls.addWidget(QtWidgets.QLabel("Paint"))
        self._mode = FluentComboBox(edit)
        self._mode.addItems(("Toggle", "Brush", "Erase"))
        controls.addWidget(self._mode)
        controls.addWidget(QtWidgets.QLabel("Intensity"))
        self._intensity = FluentDoubleSpinBox(5, parent=edit)
        self._intensity.setRange(0.0, 1000.0)
        self._intensity.setDecimals(3)
        self._intensity.setValue(1.0)
        controls.addWidget(self._intensity)
        self._preset = FluentComboBox(edit)
        self._preset.addItems((
            "5 x 7 grid", "5 x 7 checkerboard", "Flat-top rectangle",
            "Flat-top ellipse", "Ring",
        ))
        controls.addWidget(self._preset)
        for label, action in (
            ("Apply preset", self.apply_preset),
            ("Clear", lambda: self.set_target(np.zeros(self.shape, dtype=np.float32))),
        ):
            button = FluentButton(label, edit)
            button.clicked.connect(action)
            controls.addWidget(button)
        controls.addStretch(1)
        root.addWidget(edit)

        panes = QtWidgets.QHBoxLayout()
        self._target_widget = self._target_host.qt_widget()
        self._target_widget.installEventFilter(self)
        panes.addWidget(self._target_widget, 1, QtCore.Qt.AlignCenter)
        panes.addWidget(self._phase_host.qt_widget(), 1, QtCore.Qt.AlignCenter)
        root.addLayout(panes, 1)

        files, buttons = FluentFrame(body), QtWidgets.QHBoxLayout()
        files.setLayout(buttons)
        for label, action in (
            ("Load target", "load_target"), ("Import array/image", "import"),
            ("Save target", "save_target"), ("Load phase", "load_phase"),
            ("Save phase", "save_phase"),
        ):
            button = FluentButton(label, files)
            button.clicked.connect(partial(self._choose, action))
            buttons.addWidget(button)
        buttons.addStretch(1)
        self._send = FluentButton("Send to SLM", files)
        self._send.clicked.connect(self.send)
        buttons.addWidget(self._send)
        root.addWidget(files)
        self._status = QtWidgets.QLabel("Solving latest target…", body)
        root.addWidget(self._status)
        return body

    def set_target(self, values: object) -> None:
        target = validate_target(values)
        if target.shape != self.shape:
            raise ValueError(f"target shape must be {self.shape!r}")
        self._target, self._request_revision = target, self._request_revision + 1
        self._target_revision += 1
        self._target_host.update_data(_snapshot(target, "target", self._target_revision))
        if np.any(target > 0.0):
            self._queue_solve()
        else:
            self._pending = None
            self._status.setText("Empty target; hardware unchanged")

    def set_phase(self, values: object, metadata: object = None) -> None:
        self._request_revision += 1
        self._pending = None
        self._phase = canonical_phase(values, self.shape)
        self._phase_metadata = dict(
            {"source": "loaded"} if metadata is None else metadata
        )
        self._show_phase()
        self._status.setText("Phase loaded; hardware unchanged")

    @property
    def solver_idle(self) -> bool:
        return not self._running and self._pending is None

    @property
    def status_text(self) -> str:
        return self._status.text()

    @property
    def command_active(self) -> bool:
        return self._command_active

    def _queue_solve(self) -> None:
        self._pending = (self._request_revision, np.array(self._target, copy=True))
        self._status.setText("Solving latest target…")
        if not self._running:
            self._start_pending()

    def _start_pending(self) -> None:
        if self._closed or self._pending is None:
            return
        revision, target = self._pending
        self._pending, self._running = None, True
        future = self._executor.submit(
            solve_phase, target, initial_phase=np.array(self._phase, copy=True),
            stop_requested=lambda: (
                self._stop.is_set() or revision != self._request_revision
            ),
        )
        future.add_done_callback(lambda done: self._solve_ready.emit((revision, done)))

    @QtCore.pyqtSlot(object)
    def _finish_solve(self, payload: object) -> None:
        revision, future = payload
        self._running = False
        try:
            phase, metadata = future.result()
        except InterruptedError:
            pass
        except Exception as error:
            if revision == self._request_revision and not self._closed:
                self._status.setText(str(error))
        else:
            if revision == self._request_revision and not self._closed:
                self._phase, self._phase_metadata = phase, dict(metadata)
                self._show_phase()
                self._status.setText(f"Solved with {metadata['method']}; hardware unchanged")
        self._start_pending()

    def _show_phase(self) -> None:
        self._phase_revision += 1
        self._phase_host.update_data(_snapshot(self._phase, "phase", self._phase_revision))

    def apply_preset(self) -> None:
        height, width = self.shape
        makers = {
            "5 x 7 grid": lambda: preset_grid(self.shape, (5, 7)),
            "5 x 7 checkerboard": lambda: preset_checkerboard(self.shape, (5, 7), intensity_b=0.25),
            "Flat-top rectangle": lambda: preset_rectangle(self.shape, (height // 3, width // 2), edge=max(2, min(self.shape) // 32)),
            "Flat-top ellipse": lambda: preset_ellipse(self.shape, (height // 6, width // 4), edge=max(2, min(self.shape) // 32)),
            "Ring": lambda: preset_ring(self.shape, radius=min(self.shape) // 4, width=max(2, min(self.shape) // 12), edge=2),
        }
        self.set_target(makers[self._preset.currentText()]())

    def eventFilter(self, watched: object, event: object) -> bool:  # noqa: N802
        if watched is self._target_widget:
            kind = event.type()
            if kind == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self._painting = self._mode.currentText() != "Toggle"
                return self._paint(event.pos())
            if kind == QtCore.QEvent.MouseMove and self._painting and event.buttons() & QtCore.Qt.LeftButton:
                return self._paint(event.pos())
            if kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                self._painting = False
                return True
        return super().eventFilter(watched, event)

    def _paint(self, position: object) -> bool:
        front = self._target_widget.presented_front
        if front is None:
            return False
        axis = next(item for item in front.interaction.axes if item.role == "image")
        point = axis.canonical_from_normalized(
            position.x() / max(1, self._target_widget.width()),
            position.y() / max(1, self._target_widget.height()),
        )
        column, row = int(np.floor(point.x + 0.5)), int(np.floor(point.y + 0.5))
        if not (0 <= row < self.shape[0] and 0 <= column < self.shape[1]):
            return False
        target = np.array(self._target, copy=True)
        yy, xx = np.ogrid[:self.shape[0], :self.shape[1]]
        mask = (yy - row) ** 2 + (xx - column) ** 2 <= 4
        mode = self._mode.currentText()
        if mode == "Toggle":
            target[row, column] = 0.0 if target[row, column] > 0.0 else 1.0
        else:
            target[mask] = 0.0 if mode == "Erase" else self._intensity.value()
        self.set_target(target)
        return True

    def import_target(self, path: object) -> None:
        source = Path(path)
        if source.suffix.lower() == ".npy":
            values = np.load(source, allow_pickle=False)
        else:
            from PIL import Image
            with Image.open(source) as image:
                values = np.asarray(image.convert("F"), dtype=np.float32)
        self.set_target(imported_target(values))

    def save_target(self, path: object) -> Path:
        return save_target(path, self._target)

    def save_phase(self, path: object) -> Path:
        return save_phase(path, self._phase, self._phase_metadata)

    def send(self) -> bool:
        """Queue the clicked canonical phase; return whether it was accepted."""

        if self._closed:
            self._status.setText("SLM Editor is closing")
            return False
        if self._command_active:
            self._status.setText("SLM command already in progress")
            return False
        expected = canonical_phase(self._phase, self.shape)
        self._command_active = True
        self._send.setEnabled(False)
        self._status.setText("Sending phase to SLM…")
        try:
            future = self._command_executor.submit(self._send_phase, expected)
        except Exception as error:
            self._command_active = False
            self._send.setEnabled(True)
            self._status.setText(str(error))
            return False
        future.add_done_callback(lambda done: self._command_ready.emit(done))
        return True

    def _send_phase(self, expected: np.ndarray) -> None:
        from zlc_workbench.device_use import DeviceClaim

        lease = self.session.device_use.acquire_command(
            self, f"{self.device_key} SLM Editor",
            (DeviceClaim(self.device_key, self.device_key, self.device, "exclusive"),),
        )
        try:
            applied = self.device.apply_phase(expected)
            if not np.array_equal(applied, expected) or not np.array_equal(
                self.device.last_commanded_phase, expected
            ):
                raise RuntimeError("SLM did not confirm the commanded canonical phase")
        finally:
            lease.release()

    @QtCore.pyqtSlot(object)
    def _finish_send(self, future: object) -> None:
        self._command_active = False
        self._send.setEnabled(not self._closed)
        try:
            future.result()
        except Exception as error:
            self._status.setText(str(error))
        else:
            self._status.setText("Phase sent to SLM")

    def _choose(self, action: str) -> None:
        actions = {
            "load_target": (False, "target.json", "SLM target (*.json)", lambda path: self.set_target(load_target(path))),
            "save_target": (True, "target.json", "SLM target (*.json)", lambda path: save_target(path, self._target)),
            "load_phase": (False, "phase.npz", "SLM phase (*.npz)", lambda path: self.set_phase(*load_phase(path))),
            "save_phase": (True, "phase.npz", "SLM phase (*.npz)", lambda path: save_phase(path, self._phase, self._phase_metadata)),
            "import": (False, "", "Array or image (*.npy *.png *.tif *.tiff *.bmp)", self.import_target),
        }
        saving, name, filters, callback = actions[action]
        start = str(Path(self.session.workspace.data) / name)
        picker = fluent_save_path if saving else fluent_open_path
        path = picker(self._body, action.replace("_", " ").title(), start, filters)
        if path:
            try:
                callback(path)
            except Exception as error:
                self._status.setText(str(error))

    def _finish_close(self) -> bool:
        if self._cleaned:
            return True
        if not self._closed:
            self._closed, self._pending = True, None
            self._stop.set()
            self._body.setEnabled(False)
            self._status.setText("Stopping SLM Editor…")
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._command_executor.shutdown(wait=False, cancel_futures=False)
        for host in (self._target_host, self._phase_host):
            host.qt_widget().close_adapter()
        hosts_stopped = tuple(
            host.close(timeout=0.0)
            for host in (self._target_host, self._phase_host)
        )
        if self._running or self._command_active or not all(hosts_stopped):
            if self._window is not None:
                QtCore.QTimer.singleShot(20, self._window.close)
            return False
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._command_executor.shutdown(wait=True, cancel_futures=False)
        self._cleaned = True
        return True

    def close(self) -> None:
        self._window.close()

    def restore(self) -> None:
        self._window.showNormal()
        self._window.raise_()
        self._window.activateWindow()

    show = restore

    def is_visible(self) -> bool:
        return bool(self._window.isVisible())


def open_slm_control(session: object, device_key: str, window_ratio=None) -> SlmEditorControl:
    """Open one Editor against the named SLM of an existing session."""
    held: dict[str, SlmEditorControl] = {}

    def body() -> QtWidgets.QWidget:
        held["control"] = SlmEditorControl(session, device_key)
        return held["control"]._body

    window = open_fluent_window(
        body, title=f"{device_key} SLM Editor",
        window_ratio=0.8 if window_ratio is None else float(window_ratio),
    )
    control = held["control"]
    control._window = window
    window.set_close_guard(control._finish_close)
    window.closed.connect(control.closed)
    return control


__all__ = ["SlmEditorControl", "open_slm_control"]
