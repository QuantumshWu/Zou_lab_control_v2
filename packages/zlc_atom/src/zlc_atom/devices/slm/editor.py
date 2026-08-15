"""The phase-only SLM plugin's target/phase control window."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import threading

import numpy as np
from PyQt5 import QtCore, QtWidgets
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_plot import (
    PANEL_SIZE_NAMES, AxisRef, ImagePlot, PlotLabels, RasterPlotHost,
)
from zlc_ui.fluent import (
    FluentButton, FluentComboBox, FluentDoubleSpinBox, FluentFrame,
    FluentScrollArea, FluentSpinBox, FluentSwitch, FluentTabWidget,
    fluent_open_path, fluent_save_path, open_fluent_window,
)

from zlc_atom.data import snapshot_from_array
from .device import SlmAdapter, canonical_phase
from .solver import (
    imported_target, load_phase, load_target, preset_checkerboard,
    preset_ellipse, preset_grid, preset_rectangle, preset_ring, save_phase,
    save_target, solve_phase, validate_target,
)


_ZERNIKE = (
    ("tilt_x", "Noll Z2 · X tilt", "2x"),
    ("tilt_y", "Noll Z3 · Y tilt", "2y"),
    ("defocus", "Noll Z4 · Defocus", "√3(2r²−1)"),
    ("astig_oblique", "Noll Z5 · Oblique astig", "2√6xy"),
    ("astig_vertical", "Noll Z6 · Vertical astig", "√6(x²−y²)"),
    ("coma_y", "Noll Z7 · Y coma", "√8y(3r²−2)"),
    ("coma_x", "Noll Z8 · X coma", "√8x(3r²−2)"),
    ("trefoil_y", "Noll Z9 · Y trefoil", "√8y(3x²−y²)"),
    ("trefoil_x", "Noll Z10 · X trefoil", "√8x(x²−3y²)"),
    ("spherical", "Noll Z11 · Spherical", "√5(6r⁴−6r²+1)"),
)
_DEFAULT_PLOT_SIZE = "2x2"


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
        # One plot occupies half of the Editor row.  ``2x2`` is the standing
        # 490 x 357 logical viewport users had before phase-layer controls;
        # settings live on separate tabs instead of shrinking this surface.
        size=_DEFAULT_PLOT_SIZE,
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
        self._mask_phase = self._phase
        self._mask_metadata: dict[str, object] = {"source": "device"}
        self._phase_metadata: dict[str, object] = {"source": "device"}
        self._request_revision = self._target_revision = self._phase_revision = 0
        self._phase_request_revision: int | None = None
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
        self._phase_host = _host(self._phase, "phase", "Final composite (rad)")
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
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._build_layer_tabs(body))
        return body

    def _build_editor_page(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(parent)
        root = QtWidgets.QVBoxLayout(page)
        root.setContentsMargins(16, 8, 16, 8)
        root.setSpacing(6)
        edit, controls = FluentFrame(page), QtWidgets.QHBoxLayout()
        edit.setLayout(controls)
        controls.addWidget(QtWidgets.QLabel("Paint"))
        self._mode = FluentComboBox(edit)
        self._mode.addItems(("Toggle", "Brush", "Erase"))
        self._mode.setMinimumWidth(104)
        controls.addWidget(self._mode)
        controls.addWidget(QtWidgets.QLabel("Intensity"))
        self._intensity = FluentDoubleSpinBox(5, parent=edit)
        self._intensity.setRange(0.0, 1000.0)
        self._intensity.setDecimals(3)
        self._intensity.setValue(1.0)
        controls.addWidget(self._intensity)
        self._selectors = FluentSwitch("Selectors", edit)
        self._selectors.setToolTip(
            "Enable plot selectors, wheel zoom, and pan; disable to paint the target"
        )
        self._selectors.toggled.connect(self._set_selectors_enabled)
        controls.addWidget(self._selectors)
        controls.addWidget(QtWidgets.QLabel("Size"))
        self._plot_size = FluentComboBox(edit)
        self._plot_size.addItems(PANEL_SIZE_NAMES)
        self._plot_size.setCurrentText(_DEFAULT_PLOT_SIZE)
        self._plot_size.setMinimumWidth(76)
        self._plot_size.activated[int].connect(self._plot_size_picked)
        controls.addWidget(self._plot_size)
        self._preset = FluentComboBox(edit)
        self._preset.addItems((
            "5 x 7 grid", "5 x 7 checkerboard", "Flat-top rectangle",
            "Flat-top ellipse", "Ring",
        ))
        self._preset.setMinimumWidth(190)
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

        self._plot_panel = QtWidgets.QWidget(page)
        panes = QtWidgets.QHBoxLayout(self._plot_panel)
        panes.setContentsMargins(0, 0, 0, 0)
        panes.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self._target_widget = self._target_host.qt_widget()
        self._target_widget.installEventFilter(self)
        panes.addWidget(self._target_widget, 1)
        self._phase_widget = self._phase_host.qt_widget()
        panes.addWidget(self._phase_widget, 1)
        self._plot_scroll = FluentScrollArea(page)
        self._plot_scroll.setWidgetResizable(False)
        self._plot_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._plot_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._plot_scroll.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self._plot_scroll.setWidget(self._plot_panel)
        root.addWidget(self._plot_scroll, 1)

        files, buttons = FluentFrame(page), QtWidgets.QHBoxLayout()
        files.setLayout(buttons)
        for label, action in (
            ("Load target", "load_target"),
            ("Import target", "import"),
            ("Save target", "save_target"),
            ("Load mask", "load_mask"),
            ("Save mask", "save_mask"),
            ("Load final", "load_phase"),
            ("Save final", "save_phase"),
        ):
            button = FluentButton(label, files)
            button.clicked.connect(partial(self._choose, action))
            buttons.addWidget(button)
        self._import_mask = FluentButton("Import mask", files)
        self._import_mask.setToolTip(
            "Import an 8-bit grayscale phase mask: gray × 2π/256. This is not "
            "a correction pattern and does not apply a vendor LUT."
        )
        self._import_mask.clicked.connect(partial(self._choose, "import_mask"))
        buttons.addWidget(self._import_mask)
        buttons.addStretch(1)
        self._send = FluentButton("Send to SLM", files)
        self._send.clicked.connect(self.send)
        self._sync_send_enabled()
        buttons.addWidget(self._send)
        root.addWidget(files)
        self._status = QtWidgets.QLabel("Solving latest target…", page)
        root.addWidget(self._status)
        return page

    def _plot_size_picked(self, _index: int) -> None:
        """Resize both independent plot surfaces through zlc_plot's presets."""

        selected = self._plot_size.currentText()
        self._target_host.set_size(selected)
        self._phase_host.set_size(selected)
        self._status.setText(f"Plot size {selected}; hardware unchanged")

    @staticmethod
    def _settings_page(
        parent: QtWidgets.QWidget,
    ) -> tuple[FluentScrollArea, QtWidgets.QWidget]:
        scroll = FluentScrollArea(parent)
        page = QtWidgets.QWidget(scroll)
        scroll.set_width_bounded_widget(page)
        return scroll, page

    def _build_layer_tabs(self, parent: QtWidgets.QWidget) -> FluentTabWidget:
        tabs = FluentTabWidget(parent)
        self._layer_tabs = tabs
        self._editor_tab_index = tabs.add_permanent_tab(
            self._build_editor_page(tabs), "Mask"
        )

        mask_scroll, mask_page = self._settings_page(tabs)
        mask_layout = QtWidgets.QHBoxLayout()
        mask_layout.setContentsMargins(16, 16, 16, 16)
        mask_layout.setAlignment(QtCore.Qt.AlignTop)
        mask_page.setLayout(mask_layout)
        self._roi_enabled = FluentSwitch("Limit mask to ROI", mask_page)
        self._roi_enabled.setToolTip(
            "Only the Mask layer is limited; carrier and Zernike remain full-raster."
        )
        mask_layout.addWidget(self._roi_enabled)
        height, width = self.shape
        side = min(self.shape)
        roi_values = (
            ("_roi_x", "x", 0, width - 1, (width - side) // 2),
            ("_roi_y", "y", 0, height - 1, (height - side) // 2),
            ("_roi_width", "width", 1, width, side),
            ("_roi_height", "height", 1, height, side),
        )
        for name, label, minimum, maximum, value in roi_values:
            mask_layout.addWidget(QtWidgets.QLabel(label))
            spin = FluentSpinBox(mask_page)
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            setattr(self, name, spin)
            mask_layout.addWidget(spin)
        mask_layout.addStretch(1)
        tabs.add_permanent_tab(mask_scroll, "Mask ROI")

        wave_scroll, wave_page = self._settings_page(tabs)
        wave_layout = QtWidgets.QGridLayout()
        wave_layout.setContentsMargins(16, 16, 16, 16)
        wave_layout.setAlignment(QtCore.Qt.AlignTop)
        wave_page.setLayout(wave_layout)

        def add_wave_spin(
            page: QtWidgets.QWidget,
            layout: QtWidgets.QGridLayout,
            row: int,
            column: int,
            label: str,
            tooltip: str,
        ) -> FluentDoubleSpinBox:
            title = QtWidgets.QLabel(label, page)
            title.setToolTip(tooltip)
            spin = FluentDoubleSpinBox(5, parent=page)
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(5)
            spin.setSingleStep(0.01)
            spin.setValue(0.0)
            spin.setToolTip(tooltip)
            layout.addWidget(title, row, 2 * column)
            layout.addWidget(spin, row, 2 * column + 1)
            return spin

        self._carrier_x = add_wave_spin(
            wave_page, wave_layout, 0, 0, "Steering X",
            "Full-raster carrier, in edge-to-edge waves across the device width.",
        )
        self._carrier_y = add_wave_spin(
            wave_page, wave_layout, 0, 1, "Steering Y",
            "Full-raster carrier, in edge-to-edge waves across the device height.",
        )
        reset = FluentButton("Reset wavefront", wave_page)
        reset.clicked.connect(self._reset_wavefront)
        wave_layout.addWidget(reset, 0, 4, 1, 2)

        self._zernike: dict[str, FluentDoubleSpinBox] = {}
        common = (
            ("defocus", "Z4 Defocus"),
            ("astig_oblique", "Z5 Astig 45°"),
            ("astig_vertical", "Z6 Astig 0°"),
        )
        definitions = {key: (label, polynomial) for key, label, polynomial in _ZERNIKE}
        for column, (key, short_label) in enumerate(common):
            full_label, polynomial = definitions[key]
            self._zernike[key] = add_wave_spin(
                wave_page, wave_layout, 1, column, short_label,
                f"{full_label}: {polynomial}; Noll-normalized waves RMS on the unit disk.",
            )
        wave_layout.setRowStretch(2, 1)
        tabs.add_permanent_tab(wave_scroll, "Wavefront")

        advanced_scroll, advanced_page = self._settings_page(tabs)
        advanced_layout = QtWidgets.QGridLayout(advanced_page)
        advanced_layout.setContentsMargins(16, 16, 16, 16)
        advanced_layout.setAlignment(QtCore.Qt.AlignTop)
        advanced = (
            ("tilt_x", "Z2 Pupil tilt X"),
            ("tilt_y", "Z3 Pupil tilt Y"),
            ("coma_y", "Z7 Coma Y"),
            ("coma_x", "Z8 Coma X"),
            ("trefoil_y", "Z9 Trefoil Y"),
            ("trefoil_x", "Z10 Trefoil X"),
            ("spherical", "Z11 Spherical"),
        )
        self._advanced_zernike_keys = tuple(key for key, _label in advanced)
        for index, (key, short_label) in enumerate(advanced):
            full_label, polynomial = definitions[key]
            self._zernike[key] = add_wave_spin(
                advanced_page, advanced_layout, index // 3, index % 3,
                short_label,
                f"{full_label}: {polynomial}; Noll-normalized waves RMS on the unit disk. "
                "Pupil tilt adds to full-raster Steering X/Y and is normally left at zero.",
            )
        advanced_layout.setRowStretch(3, 1)
        self._advanced_tab_index = tabs.add_permanent_tab(
            advanced_scroll, "Advanced"
        )
        tabs.setCurrentIndex(self._editor_tab_index)

        self._roi_enabled.toggled.connect(self._layer_changed)
        for spin in (
            self._roi_x, self._roi_y, self._roi_width, self._roi_height,
        ):
            spin.valueChanged.connect(self._roi_changed)
        for spin in (self._carrier_x, self._carrier_y, *self._zernike.values()):
            spin.valueChanged.connect(self._layer_changed)
        return tabs

    def _reset_wavefront(self) -> None:
        widgets = (self._carrier_x, self._carrier_y, *self._zernike.values())
        blocked = tuple(widget.blockSignals(True) for widget in widgets)
        try:
            for widget in widgets:
                widget.setValue(0.0)
        finally:
            for widget, previous in zip(widgets, blocked):
                widget.blockSignals(previous)
        self._layer_changed()

    def _update_advanced_tab(self) -> None:
        active = sum(bool(self._zernike[key].value()) for key in self._advanced_zernike_keys)
        label = "Advanced" if active == 0 else f"Advanced ({active} active)"
        self._layer_tabs.setTabText(self._advanced_tab_index, label)

    def _roi_changed(self) -> None:
        self._roi_width.setMaximum(self.shape[1] - self._roi_x.value())
        self._roi_height.setMaximum(self.shape[0] - self._roi_y.value())
        self._layer_changed()

    def _layer_changed(self) -> None:
        self._update_advanced_tab()
        self._compose_phase()
        self._sync_send_enabled()
        status = getattr(self, "_status", None)
        if status is not None:
            status.setText("Final composite updated; hardware unchanged")

    def _compose_phase(self) -> None:
        height, width = self.shape
        yy, xx = np.ogrid[-1.0:1.0:height * 1j, -1.0:1.0:width * 1j]
        carrier_x = self._carrier_x.value()
        carrier_y = self._carrier_y.value()
        if carrier_x or carrier_y:
            phase = np.pi * (carrier_x * xx + carrier_y * yy)
        else:
            phase = np.zeros(self.shape, dtype=np.float64)
        coefficients = {key: self._zernike[key].value() for key, *_ in _ZERNIKE}
        if any(coefficients.values()):
            r2 = xx * xx + yy * yy
            pupil = r2 <= 1.0
            for key, coefficient in coefficients.items():
                if not coefficient:
                    continue
                if key == "tilt_x":
                    values = 2.0 * xx
                elif key == "tilt_y":
                    values = 2.0 * yy
                elif key == "defocus":
                    values = np.sqrt(3.0) * (2.0 * r2 - 1.0)
                elif key == "astig_oblique":
                    values = 2.0 * np.sqrt(6.0) * xx * yy
                elif key == "astig_vertical":
                    values = np.sqrt(6.0) * (xx * xx - yy * yy)
                elif key == "coma_y":
                    values = np.sqrt(8.0) * yy * (3.0 * r2 - 2.0)
                elif key == "coma_x":
                    values = np.sqrt(8.0) * xx * (3.0 * r2 - 2.0)
                elif key == "trefoil_y":
                    values = np.sqrt(8.0) * yy * (3.0 * xx * xx - yy * yy)
                elif key == "trefoil_x":
                    values = np.sqrt(8.0) * xx * (xx * xx - 3.0 * yy * yy)
                else:
                    values = np.sqrt(5.0) * (
                        6.0 * r2 * r2 - 6.0 * r2 + 1.0
                    )
                phase[pupil] += (
                    2.0 * np.pi * coefficient
                    * np.broadcast_to(values, self.shape)[pupil]
                )
        if self._roi_enabled.isChecked():
            x, y = self._roi_x.value(), self._roi_y.value()
            width, height = self._roi_width.value(), self._roi_height.value()
            phase[y:y + height, x:x + width] += self._mask_phase[
                y:y + height, x:x + width
            ]
        else:
            phase += self._mask_phase
        self._phase = canonical_phase(phase, self.shape)
        self._phase_metadata = {
            "source": "composite",
            "mask": dict(self._mask_metadata),
            "carrier_waves_xy": [
                self._carrier_x.value(), self._carrier_y.value(),
            ],
            "zernike_noll_waves_rms": coefficients,
            "mask_roi_xywh": [
                self._roi_x.value(), self._roi_y.value(),
                self._roi_width.value(), self._roi_height.value(),
            ] if self._roi_enabled.isChecked() else None,
        }
        self._show_phase()

    def _set_selectors_enabled(self, enabled: bool) -> None:
        """Choose plot gestures or target painting without rebuilding either host."""

        active = bool(enabled)
        self._painting = False
        self._mode.setEnabled(not active)
        self._intensity.setEnabled(not active)
        self._target_host.set_interaction_enabled(active)
        self._phase_host.set_interaction_enabled(active)

    def set_target(self, values: object) -> None:
        target = validate_target(values)
        if target.shape != self.shape:
            raise ValueError(f"target shape must be {self.shape!r}")
        self._target, self._request_revision = target, self._request_revision + 1
        self._sync_send_enabled()
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
        self._mask_phase = self._phase
        self._phase_request_revision = self._request_revision
        self._phase_metadata = dict(
            {"source": "loaded"} if metadata is None else metadata
        )
        self._mask_metadata = dict(self._phase_metadata)
        widgets = (
            self._carrier_x, self._carrier_y, *self._zernike.values(),
            self._roi_enabled,
        )
        blocked = tuple(widget.blockSignals(True) for widget in widgets)
        try:
            self._carrier_x.setValue(0.0)
            self._carrier_y.setValue(0.0)
            for spin in self._zernike.values():
                spin.setValue(0.0)
            self._roi_enabled.setChecked(False)
        finally:
            for widget, previous in zip(widgets, blocked):
                widget.blockSignals(previous)
        self._update_advanced_tab()
        self._show_phase()
        self._sync_send_enabled()
        self._status.setText("Final phase loaded; layers reset; hardware unchanged")

    def set_mask_phase(self, values: object, metadata: object = None) -> None:
        self._request_revision += 1
        self._pending = None
        self._mask_phase = canonical_phase(values, self.shape)
        self._mask_metadata = dict(
            {"source": "loaded mask"} if metadata is None else metadata
        )
        self._phase_request_revision = self._request_revision
        self._compose_phase()
        self._sync_send_enabled()
        self._status.setText("Mask loaded; final recomposed; hardware unchanged")

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
            solve_phase, target, initial_phase=np.array(self._mask_phase, copy=True),
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
                self._mask_phase = canonical_phase(phase, self.shape)
                self._mask_metadata = dict(metadata)
                self._phase_request_revision = revision
                self._compose_phase()
                self._status.setText(f"Solved with {metadata['method']}; hardware unchanged")
        self._sync_send_enabled()
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
        if watched is self._target_widget and not self._selectors.isChecked():
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

    def save_mask(self, path: object) -> Path:
        return save_phase(path, self._mask_phase, self._mask_metadata)

    def import_mask_image(self, path: object) -> None:
        from PIL import Image

        source = Path(path)
        with Image.open(source) as image:
            if image.mode != "L":
                raise ValueError("phase mask image must be two-dimensional 8-bit grayscale")
            gray = np.asarray(image)
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("phase mask image must be two-dimensional 8-bit grayscale")
        if gray.shape == self.shape:
            values = gray.astype(np.float64) * (2.0 * np.pi / 256.0)
        elif self._roi_enabled.isChecked() and gray.shape == (
            self._roi_height.value(), self._roi_width.value(),
        ):
            values = np.zeros(self.shape, dtype=np.float64)
            x, y = self._roi_x.value(), self._roi_y.value()
            values[
                y:y + gray.shape[0], x:x + gray.shape[1]
            ] = gray.astype(np.float64) * (2.0 * np.pi / 256.0)
        else:
            raise ValueError(
                "phase mask image shape must match the device or the enabled Mask ROI"
            )
        self.set_mask_phase(values, {
            "source": "8-bit phase mask image",
            "mapping": "gray * 2*pi/256 (no correction or vendor LUT)",
            "file": source.name,
        })

    def send(self) -> bool:
        """Queue the clicked canonical phase; return whether it was accepted."""

        if self._closed:
            self._status.setText("SLM Editor is closing")
            return False
        if self._command_active:
            self._status.setText("SLM command already in progress")
            return False
        if self._phase_request_revision != self._request_revision:
            self._status.setText("Wait for the latest target solve before Send")
            self._sync_send_enabled()
            return False
        expected = canonical_phase(self._phase, self.shape)
        self._command_active = True
        self._sync_send_enabled()
        self._status.setText("Sending phase to SLM…")
        try:
            future = self._command_executor.submit(self._send_phase, expected)
        except Exception as error:
            self._command_active = False
            self._sync_send_enabled()
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
        self._sync_send_enabled()
        try:
            future.result()
        except Exception as error:
            self._status.setText(str(error))
        else:
            self._status.setText("Phase sent to SLM")

    def _sync_send_enabled(self) -> None:
        send = getattr(self, "_send", None)
        if send is not None:
            send.setEnabled(
                not self._closed
                and not self._command_active
                and self._phase_request_revision == self._request_revision
            )

    def _choose(self, action: str) -> None:
        actions = {
            "load_target": (False, "target.json", "SLM target (*.json)", lambda path: self.set_target(load_target(path))),
            "save_target": (True, "target.json", "SLM target (*.json)", lambda path: save_target(path, self._target)),
            "load_mask": (False, "mask.npz", "SLM mask phase (*.npz)", lambda path: self.set_mask_phase(*load_phase(path))),
            "save_mask": (True, "mask.npz", "SLM mask phase (*.npz)", lambda path: self.save_mask(path)),
            "load_phase": (False, "final.npz", "SLM final phase (*.npz)", lambda path: self.set_phase(*load_phase(path))),
            "save_phase": (True, "final.npz", "SLM final phase (*.npz)", lambda path: self.save_phase(path)),
            "import": (False, "", "Array or image (*.npy *.png *.tif *.tiff *.bmp)", self.import_target),
            "import_mask": (False, "", "8-bit mask image (*.bmp *.png *.tif *.tiff)", self.import_mask_image),
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
