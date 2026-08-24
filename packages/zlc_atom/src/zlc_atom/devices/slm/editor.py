"""The phase-only SLM plugin's target/phase control window."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
import threading
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_plot import (
    PANEL_SIZE_NAMES, AxisRef, ImagePlot, PlotLabels, RasterPlotHost,
)
from zlc_ui.fluent import (
    FluentButton, FluentComboBox, FluentDoubleSpinBox, FluentFrame,
    FluentLineEdit, FluentPopup, FluentScrollArea, FluentSettingsPopupAnchor,
    FluentSpinBox, FluentSwitch, FluentTabWidget, fluent_open_path, fluent_save_path,
    open_fluent_window,
)

from zlc_atom.data import snapshot_from_array
from .device import SlmAdapter, canonical_phase
from .solver import (
    compose_science_phase, freeze_pattern_phase,
    imported_target, load_science_context, load_target, preset_checkerboard,
    preset_flat_top, preset_gaussian, preset_grid, preset_text,
    save_science_context, save_target, science_operator_wavefront,
    science_pupil_fields, solve_phase, validate_target,
)


_ZERNIKE = (
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


def _number_spin(parent: QtWidgets.QWidget, value: float, minimum: float,
                 maximum: float, *, integral: bool = False, decimals: int = 3):
    spin = FluentSpinBox(parent) if integral else FluentDoubleSpinBox(5, parent=parent)
    spin.setRange(int(minimum), int(maximum)) if integral else spin.setRange(
        float(minimum), float(maximum)
    )
    if not integral:
        spin.setDecimals(decimals)
    spin.setValue(int(value) if integral else float(value))
    return spin


def _read_imported_target(path: object) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        values = np.load(source, allow_pickle=False)
    else:
        from PIL import Image
        with Image.open(source) as image:
            values = np.asarray(image.convert("F"), dtype=np.float32)
    return imported_target(values)


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
        self._phase = canonical_phase(np.zeros(self.shape), self.shape)
        self._pattern_phase = self._phase
        self._pattern_metadata: dict[str, object] = {"source": "deterministic-seed"}
        self._phase_metadata: dict[str, object] = {"source": "authoring-draft"}
        self._system_correction: dict[str, object] | None = None
        self._context_command_receipt = dict(self.device.last_command_receipt)
        self._draft_command_revision = int(self.device.command_revision)
        self._draft_mapping_revision = int(self.device.mapping_revision)
        self._device_diverged = False
        self._objective_kind = "spots"
        height, width = self.shape
        self._pupil_center_xy = (0.5 * (width - 1), 0.5 * (height - 1))
        default_diameter = 0.70 * height
        self._pupil_diameter_xy = (default_diameter, default_diameter)
        initial_pupil = {
            "enabled": True,
            "center_xy": list(self._pupil_center_xy),
            "diameter_xy": list(self._pupil_diameter_xy),
        }
        self._pupil_support, self._pupil_amplitude = science_pupil_fields(
            self.shape, initial_pupil
        )
        self._pupil_applied_description = self._pupil_description(True)
        self._wavefront_phase = science_operator_wavefront(
            self.shape,
            initial_pupil,
            {
                "enabled": False,
                "carrier_waves_xy": [0.0, 0.0],
                "zernike_noll_waves_rms": {},
            },
        )
        self._request_revision = self._target_revision = self._phase_revision = 0
        self._phase_request_revision: int | None = None
        self._spot_optimizer_state: dict[str, object] | None = None
        self._pending: tuple[
            int, np.ndarray, str, np.ndarray, dict[str, object] | None
        ] | None = None
        self._running = self._painting = self._closed = self._cleaned = False
        self._command_active = False
        self._close_deadline: float | None = None
        self._window = None
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slm-solve")
        self._command_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="slm-command"
        )
        self._target_host = _host(self._target, "target", "Target intensity")
        self._phase_host = _host(
            self._phase, "phase", "Science phase (pre-correction)"
        )
        self._wavefront_host = _host(
            self._wavefront_phase, "wavefront", "Wavefront phase (rad)"
        )
        self._target_host.set_interaction_enabled(False)
        self._phase_host.set_interaction_enabled(False)
        self._wavefront_host.set_interaction_enabled(False)
        self._body = self._build_body()
        self._solve_ready.connect(self._finish_solve)
        self._command_ready.connect(
            self._finish_command, QtCore.Qt.QueuedConnection
        )
        self._device_poll = QtCore.QTimer(self)
        self._device_poll.setInterval(100)
        self._device_poll.timeout.connect(self._sync_device_state)
        self._device_poll.start()
        self._sync_device_state()
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
        self._preset_button = FluentButton("Preset…", edit)
        controls.addWidget(self._preset_button)
        self._build_preset_popup(page)
        self._preset_button.clicked.connect(
            lambda: self._preset_anchor.toggle(
                self._preset_body,
                minimum_width=470,
                minimum_height=360,
            )
        )
        clear = FluentButton("Clear", edit)
        clear.clicked.connect(
            lambda: self.set_target(
                np.zeros(self.shape, dtype=np.float32), objective_kind="spots"
            )
        )
        controls.addWidget(clear)
        controls.addStretch(1)
        root.addWidget(edit)

        layers = FluentFrame(page)
        layer_layout = QtWidgets.QGridLayout(layers)
        layer_layout.setContentsMargins(8, 5, 8, 5)
        layer_layout.setHorizontalSpacing(8)
        layer_layout.setVerticalSpacing(4)

        self._pupil_enabled = FluentSwitch("Input pupil", layers)
        self._pupil_enabled.setChecked(True)
        self._pupil_enabled.setToolTip(
            "On uses the configured Gaussian for solver illumination. Off uses "
            "uniform full-raster illumination; center and diameter still define "
            "the Zernike unit disk."
        )
        self._pupil_status = QtWidgets.QLabel(
            self._pupil_applied_description, layers
        )
        self._pupil_edit = FluentButton("Edit…", layers)
        self._build_pupil_popup(page)
        self._pupil_edit.clicked.connect(
            lambda: self._pupil_anchor.toggle(
                self._pupil_body, minimum_width=430, minimum_height=270,
            )
        )
        self._pupil_enabled.toggled.connect(self._pupil_toggled)

        self._zernike_enabled = FluentSwitch("Zernike", layers)
        self._zernike_enabled.setChecked(False)
        self._zernike_status = QtWidgets.QLabel("Off", layers)
        zernike_edit = FluentButton("Edit…", layers)
        zernike_edit.clicked.connect(
            lambda: self._layer_tabs.setCurrentIndex(self._wavefront_tab_index)
        )
        for row, widgets in (
            (0, (self._pupil_enabled, self._pupil_status, self._pupil_edit)),
            (1, (self._zernike_enabled, self._zernike_status, zernike_edit)),
        ):
            layer_layout.addWidget(widgets[0], row, 0)
            layer_layout.addWidget(widgets[1], row, 1, 1, 4)
            layer_layout.addWidget(widgets[2], row, 5)

        self._correction_enabled = FluentSwitch("Vendor correction", layers)
        self._correction_status = QtWidgets.QLabel("Unavailable", layers)
        self._correction_load = FluentButton("Load correction", layers)
        correction_hint = (
            "Changes the device mapping used by the next explicit Send; "
            "science phase and hardware remain unchanged now."
        )
        self._correction_enabled.setToolTip(correction_hint)
        self._correction_load.setToolTip(correction_hint)
        layer_layout.addWidget(self._correction_enabled, 2, 0)
        layer_layout.addWidget(self._correction_status, 2, 1, 1, 4)
        layer_layout.addWidget(self._correction_load, 2, 5)
        self._correction_load.clicked.connect(partial(self._choose, "correction"))
        self._correction_enabled.toggled.connect(self._set_correction_enabled)
        self._sync_correction_controls()
        layer_layout.setColumnStretch(1, 1)
        root.addWidget(layers)

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
            ("Load science context", "load_context"),
            ("Save science context", "save_context"),
        ):
            button = FluentButton(label, files)
            button.clicked.connect(partial(self._choose, action))
            buttons.addWidget(button)
        buttons.addStretch(1)
        self._adopt = FluentButton("Adopt device command", files)
        self._adopt.clicked.connect(self._adopt_device_command)
        buttons.addWidget(self._adopt)
        self._send = FluentButton("Send to SLM", files)
        self._send.clicked.connect(self.send)
        self._sync_send_enabled()
        buttons.addWidget(self._send)
        root.addWidget(files)
        self._status = QtWidgets.QLabel("Solving latest target…", page)
        root.addWidget(self._status)
        self._device_status = QtWidgets.QLabel("Device command: checking…", page)
        root.addWidget(self._device_status)
        return page

    def _plot_size_picked(self, _index: int) -> None:
        """Resize the three independent plot surfaces through zlc_plot presets."""

        selected = self._plot_size.currentText()
        self._target_host.set_size(selected)
        self._phase_host.set_size(selected)
        self._wavefront_host.set_size(selected)
        if self._running or self._pending is not None:
            self._status.setText("Solving latest target…")
        else:
            self._status.setText(f"Plot size {selected}; hardware unchanged")

    @staticmethod
    def _popup_form(parent: QtWidgets.QWidget, anchor: QtWidgets.QWidget):
        popup = FluentPopup(parent)
        outer = QtWidgets.QVBoxLayout(popup)
        body = QtWidgets.QWidget(popup)
        form = QtWidgets.QGridLayout(body)
        form.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(body)
        return popup, body, form, FluentSettingsPopupAnchor(popup, anchor)

    def _build_preset_popup(self, parent: QtWidgets.QWidget) -> None:
        popup, body, form, anchor = self._popup_form(parent, self._preset_button)
        form.addWidget(QtWidgets.QLabel("Pattern", body), 0, 0)
        self._preset_type = FluentComboBox(body)
        self._preset_type.addItems(
            ("Grid", "Checkerboard", "Gaussian", "Flat Top", "Text")
        )
        form.addWidget(self._preset_type, 0, 1)
        specs = (
            ("rows", "N rows", True, 5, 1, self.shape[0]),
            ("columns", "M columns", True, 7, 1, self.shape[1]),
            ("spacing_y", "Spacing Y", True, max(1, self.shape[0] // 6), 1, self.shape[0]),
            ("spacing_x", "Spacing X", True, max(1, self.shape[1] // 8), 1, self.shape[1]),
            ("intensity", "Intensity", False, 1.0, 0.0, 1000.0),
            ("edge", "Edge", False, max(2, min(self.shape) // 32), 0.0, min(self.shape)),
            ("radius_y", "Radius Y", False, max(1, self.shape[0] // 8), 1, self.shape[0]),
            ("radius_x", "Radius X", False, max(1, self.shape[1] // 8), 1, self.shape[1]),
            ("text_spacing", "Minimum atom spacing (target px)", True, 4, 1, min(self.shape)),
            ("atom_budget", "Atom budget", True, 256, 1, min(4096, int(np.prod(self.shape)))),
        )
        self._preset_fields: dict[str, QtWidgets.QAbstractSpinBox] = {}
        self._preset_field_rows: dict[str, tuple[QtWidgets.QWidget, QtWidgets.QWidget]] = {}
        for row, (key, label, integral, value, minimum, maximum) in enumerate(
            specs, start=1
        ):
            title = QtWidgets.QLabel(label, body)
            spin = _number_spin(
                body, value, minimum, maximum, integral=integral
            )
            form.addWidget(title, row, 0)
            form.addWidget(spin, row, 1)
            self._preset_fields[key] = spin
            self._preset_field_rows[key] = (title, spin)
        text_row = len(specs) + 1
        self._preset_text_title = QtWidgets.QLabel("Text", body)
        self._preset_text = FluentLineEdit("USTC", body)
        self._preset_text.setPlaceholderText("Chinese and English text")
        form.addWidget(self._preset_text_title, text_row, 0)
        form.addWidget(self._preset_text, text_row, 1)
        apply_button = FluentButton("Apply", body)
        apply_button.clicked.connect(self._apply_preset_popup)
        form.addWidget(apply_button, text_row + 1, 0, 1, 2)
        self._preset_popup, self._preset_body = popup, body
        self._preset_anchor = anchor
        self._preset_type.currentTextChanged.connect(self._sync_preset_popup)
        self._sync_preset_popup()

    def _sync_preset_popup(self) -> None:
        kind = self._preset_type.currentText()
        visible = {
            "Grid": {"rows", "columns", "spacing_y", "spacing_x", "intensity"},
            "Checkerboard": {
                "rows", "columns", "spacing_y", "spacing_x", "intensity",
            },
            "Gaussian": {"radius_y", "radius_x", "intensity"},
            "Flat Top": {"radius_y", "radius_x", "edge", "intensity"},
            "Text": {"text_spacing", "atom_budget", "intensity"},
        }[kind]
        for key, widgets in self._preset_field_rows.items():
            for widget in widgets:
                widget.setVisible(key in visible)
        self._preset_text_title.setVisible(kind == "Text")
        self._preset_text.setVisible(kind == "Text")
        self._preset_field_rows["columns"][0].setText(
            "Sites in long row" if kind == "Checkerboard" else "M columns"
        )
        spacing_x_title, spacing_x = self._preset_field_rows["spacing_x"]
        spacing_x_title.setText(
            "Checker cell spacing X" if kind == "Checkerboard" else "Spacing X"
        )
        checker_hint = (
            "Base checker-cell spacing; long-row sites are 2× this distance."
            if kind == "Checkerboard" else ""
        )
        spacing_x_title.setToolTip(checker_hint)
        spacing_x.setToolTip(checker_hint)
        radius_prefix = "1/e² " if kind == "Gaussian" else ""
        self._preset_field_rows["radius_y"][0].setText(
            f"{radius_prefix}radius Y"
        )
        self._preset_field_rows["radius_x"][0].setText(
            f"{radius_prefix}radius X"
        )
        self._preset_popup.adjustSize()

    def _apply_preset_popup(self) -> None:
        value = lambda key: self._preset_fields[key].value()
        kind = self._preset_type.currentText()
        grid = (value("rows"), value("columns"))
        spacing = (value("spacing_y"), value("spacing_x"))
        try:
            if kind == "Grid":
                objective = "spots"
                target = preset_grid(
                    self.shape, grid, spacing_yx=spacing,
                    intensity=value("intensity"),
                )
            elif kind == "Checkerboard":
                objective = "spots"
                target = preset_checkerboard(
                    self.shape, grid, spacing_yx=spacing,
                    intensity=value("intensity"),
                )
            elif kind == "Gaussian":
                objective = "image"
                target = preset_gaussian(
                    self.shape, (value("radius_y"), value("radius_x")),
                    intensity=value("intensity"),
                )
            elif kind == "Flat Top":
                objective = "image"
                target = preset_flat_top(
                    self.shape, (value("radius_y"), value("radius_x")),
                    edge=value("edge"), intensity=value("intensity"),
                )
            else:
                objective = "spots"
                target = preset_text(
                    self.shape, self._preset_text.text(),
                    spacing=value("text_spacing"),
                    atom_budget=value("atom_budget"),
                    intensity=value("intensity"),
                )
        except (FileNotFoundError, TypeError, ValueError) as error:
            self._status.setText(str(error))
            return
        self._preset_popup.hide()
        self.set_target(target, objective_kind=objective)

    def _build_pupil_popup(self, parent: QtWidgets.QWidget) -> None:
        popup, body, form, anchor = self._popup_form(parent, self._pupil_edit)
        cx, cy = self._pupil_center_xy
        dx, dy = self._pupil_diameter_xy
        fields = (
            ("_pupil_center_x", "Center X", cx, 0.0, self.shape[1] - 1),
            ("_pupil_center_y", "Center Y", cy, 0.0, self.shape[0] - 1),
            ("_pupil_diameter_x", "1/e² intensity diameter X", dx, 1.0, 2.0 * self.shape[1]),
            ("_pupil_diameter_y", "1/e² intensity diameter Y", dy, 1.0, 2.0 * self.shape[0]),
        )
        for row, (name, label, value, minimum, maximum) in enumerate(fields):
            spin = _number_spin(body, value, minimum, maximum, decimals=2)
            setattr(self, name, spin)
            form.addWidget(QtWidgets.QLabel(label, body), row, 0)
            form.addWidget(spin, row, 1)
        apply_button = FluentButton("Apply pupil", body)
        apply_button.clicked.connect(self._apply_pupil)
        form.addWidget(apply_button, len(fields), 0, 1, 2)
        self._pupil_popup, self._pupil_body = popup, body
        self._pupil_anchor = anchor

    def _pupil_settings(self) -> dict[str, object]:
        return {
            "enabled": self._pupil_enabled.isChecked(),
            "center_xy": list(self._pupil_center_xy),
            "diameter_xy": list(self._pupil_diameter_xy),
        }

    def _operator_settings(self) -> dict[str, object]:
        return {
            "enabled": self._zernike_enabled.isChecked(),
            "carrier_waves_xy": [
                self._carrier_x.value(), self._carrier_y.value(),
            ],
            "zernike_noll_waves_rms": {
                key: spin.value() for key, spin in self._zernike.items()
            },
        }

    def _pupil_description(self, enabled: bool) -> str:
        if not enabled:
            return "Off · uniform full raster"
        dx, dy = self._pupil_diameter_xy
        return f"On · Gaussian · 1/e² intensity diameter {dx:g} × {dy:g} px"

    def _apply_pupil(self, *_args: object) -> None:
        self._pupil_center_xy = (
            self._pupil_center_x.value(), self._pupil_center_y.value(),
        )
        self._pupil_diameter_xy = (
            self._pupil_diameter_x.value(), self._pupil_diameter_y.value(),
        )
        self._resolve_pupil(self._pupil_enabled.isChecked())
        self._pupil_popup.hide()

    def _pupil_toggled(self, enabled: bool) -> None:
        self._resolve_pupil(bool(enabled))

    def _resolve_pupil(self, enabled: bool) -> None:
        self._pupil_support, self._pupil_amplitude = science_pupil_fields(
            self.shape,
            {
                "enabled": bool(enabled),
                "center_xy": list(self._pupil_center_xy),
                "diameter_xy": list(self._pupil_diameter_xy),
            },
        )
        self._spot_optimizer_state = None
        self._pupil_applied_description = self._pupil_description(enabled)
        self._pupil_status.setText(self._pupil_applied_description)
        self._request_revision += 1
        self._phase_request_revision = None
        self._compose_phase()
        self._sync_send_enabled()
        if np.any(self._target > 0.0):
            self._queue_solve()
        else:
            self._pending = None
            self._status.setText("Empty target; hardware unchanged")

    def _sync_correction_controls(self) -> None:
        available = all(callable(getattr(self.device, name, None)) for name in (
            "load_correction", "set_correction_enabled",
        ))
        self._correction_load.setEnabled(available)
        self._correction_enabled.setEnabled(available)
        if not available:
            self._correction_status.setText("Unavailable")
            return
        enabled = bool(getattr(self.device, "correction_enabled", False))
        name = str(getattr(self.device, "correction_name", ""))
        previous = self._correction_enabled.blockSignals(True)
        self._correction_enabled.setChecked(enabled)
        self._correction_enabled.blockSignals(previous)
        state = "On" if enabled else "Off"
        details = [state]
        if name:
            details.append(name)
        wavelength = getattr(self.device, "wavelength_nm", None)
        two_pi_gray = getattr(self.device, "two_pi_gray", None)
        if wavelength is not None and two_pi_gray is not None:
            details.extend((f"{float(wavelength):g} nm", f"2π gray {float(two_pi_gray):g}"))
        self._correction_status.setText(" · ".join(details))

    def _set_correction_enabled(self, enabled: bool) -> None:
        if not self._queue_work(
            "Vendor correction updated",
            partial(self.device.set_correction_enabled, bool(enabled)),
            claim_device=True,
        ):
            self._sync_correction_controls()

    def _build_layer_tabs(self, parent: QtWidgets.QWidget) -> FluentTabWidget:
        tabs = FluentTabWidget(parent)
        self._layer_tabs = tabs
        self._editor_tab_index = tabs.add_permanent_tab(
            self._build_editor_page(tabs), "Pattern"
        )

        wave_page = QtWidgets.QWidget(tabs)
        wave_root = QtWidgets.QHBoxLayout(wave_page)
        wave_root.setContentsMargins(8, 8, 8, 8)
        wave_root.setSpacing(8)
        self._wavefront_parameter_scroll = FluentScrollArea(wave_page)
        parameter_page = QtWidgets.QWidget(self._wavefront_parameter_scroll)
        self._wavefront_parameter_scroll.set_width_bounded_widget(parameter_page)
        parameter_page.setMinimumWidth(440)
        wave_layout = QtWidgets.QGridLayout(parameter_page)
        wave_layout.setContentsMargins(12, 12, 12, 12)
        wave_layout.setHorizontalSpacing(8)
        wave_layout.setVerticalSpacing(6)
        wave_layout.setAlignment(QtCore.Qt.AlignTop)
        note = QtWidgets.QLabel(
            "Steering X/Y adds a full-raster carrier. A physical "
            "Fourier-plane stop/aperture is still required to reject zero order. "
            "Coefficients are Noll-normalized waves RMS on the configured unit disk.",
            parameter_page,
        )
        note.setWordWrap(True)
        wave_layout.addWidget(note, 0, 0, 1, 4)

        self._zernike: dict[str, FluentDoubleSpinBox] = {}
        carrier_specs = (
            ("_carrier_x", "Steering X", "Full-raster edge-to-edge waves across device width."),
            ("_carrier_y", "Steering Y", "Full-raster edge-to-edge waves across device height."),
        )
        zernike_specs = tuple(
            (key, label, f"{label}: {polynomial}; Noll-normalized waves RMS on the configured unit disk.")
            for key, label, polynomial in _ZERNIKE
        )
        for index, (key, label, tooltip) in enumerate(carrier_specs + zernike_specs):
            row, column = (1, index) if index < 2 else (2 + (index - 2) // 2, (index - 2) % 2)
            title = QtWidgets.QLabel(label, parameter_page)
            spin = _number_spin(parameter_page, 0.0, -1000.0, 1000.0, decimals=5)
            title.setToolTip(tooltip); spin.setToolTip(tooltip); spin.setSingleStep(0.01)
            wave_layout.addWidget(title, row, 2 * column)
            wave_layout.addWidget(spin, row, 2 * column + 1)
            if key.startswith("_"):
                setattr(self, key, spin)
            else:
                self._zernike[key] = spin
        reset = FluentButton("Reset wavefront", parameter_page)
        reset.clicked.connect(self._reset_wavefront)
        wave_layout.addWidget(reset, 7, 0, 1, 4)
        wave_root.addWidget(self._wavefront_parameter_scroll, 0)

        self._wavefront_plot_panel = QtWidgets.QWidget(wave_page)
        plot_layout = QtWidgets.QVBoxLayout(self._wavefront_plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self._wavefront_widget = self._wavefront_host.qt_widget()
        plot_layout.addWidget(self._wavefront_widget)
        self._wavefront_plot_scroll = FluentScrollArea(wave_page)
        self._wavefront_plot_scroll.setWidget(self._wavefront_plot_panel)
        wave_root.addWidget(self._wavefront_plot_scroll, 1)
        self._wavefront_tab_index = tabs.add_permanent_tab(wave_page, "Wavefront")
        tabs.setCurrentIndex(self._editor_tab_index)

        self._zernike_enabled.toggled.connect(self._layer_changed)
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

    def _layer_changed(self, *_args: object) -> None:
        active = sum(bool(spin.value()) for spin in (
            self._carrier_x, self._carrier_y, *self._zernike.values()
        ))
        self._zernike_status.setText(
            f"On · {active} active"
            if self._zernike_enabled.isChecked()
            else "Off"
        )
        self._compose_phase()
        self._sync_send_enabled()
        status = getattr(self, "_status", None)
        if status is not None:
            status.setText(
                "Solving latest target…"
                if self._running or self._pending is not None
                else "Science phase updated; hardware unchanged"
            )

    def _compose_phase(self) -> None:
        operator = self._operator_settings()
        self._wavefront_phase = science_operator_wavefront(
            self.shape, self._pupil_settings(), operator
        )
        self._phase = compose_science_phase(
            self._pattern_phase, self._wavefront_phase
        )
        self._phase_metadata = {
            "source": "composite",
            "hardware_correction": "excluded",
            "pattern": dict(self._pattern_metadata),
            "input_pupil": self._pupil_applied_description,
            "carrier_waves_xy": operator["carrier_waves_xy"],
            "zernike_enabled": operator["enabled"],
            "zernike_noll_waves_rms": operator["zernike_noll_waves_rms"],
        }
        self._show_phase()
        self._show_wavefront()

    def _set_selectors_enabled(self, enabled: bool) -> None:
        """Choose plot gestures or target painting without rebuilding either host."""

        active = bool(enabled)
        self._painting = False
        self._mode.setEnabled(not active)
        self._intensity.setEnabled(not active)
        self._target_host.set_interaction_enabled(active)
        self._phase_host.set_interaction_enabled(active)

    def set_target(self, values: object, *, objective_kind: str = "auto") -> None:
        target = validate_target(values)
        if target.shape != self.shape:
            raise ValueError(f"target shape must be {self.shape!r}")
        if objective_kind not in {"auto", "spots", "image"}:
            raise ValueError("objective_kind must be 'auto', 'spots', or 'image'")
        keep_optimizer = (
            self._objective_kind == objective_kind == "spots"
            and np.any(target > 0.0)
            and np.array_equal(target > 0.0, self._target > 0.0)
        )
        if not keep_optimizer:
            self._spot_optimizer_state = None
        self._objective_kind = objective_kind
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
        self._phase = freeze_pattern_phase(values, self.shape)
        self._spot_optimizer_state = None
        self._pattern_phase = self._phase
        self._phase_request_revision = self._request_revision
        self._phase_metadata = dict(
            {"source": "loaded"} if metadata is None else metadata
        )
        self._pattern_metadata = dict(self._phase_metadata)
        widgets = (
            self._carrier_x, self._carrier_y, *self._zernike.values(),
            self._zernike_enabled,
        )
        blocked = tuple(widget.blockSignals(True) for widget in widgets)
        try:
            self._carrier_x.setValue(0.0)
            self._carrier_y.setValue(0.0)
            for spin in self._zernike.values():
                spin.setValue(0.0)
            self._zernike_enabled.setChecked(False)
        finally:
            for widget, previous in zip(widgets, blocked):
                widget.blockSignals(previous)
        self._wavefront_phase = canonical_phase(
            np.zeros(self.shape, dtype=np.float32), self.shape
        )
        self._system_correction = None
        self._context_command_receipt = dict(self.device.last_command_receipt)
        self._draft_command_revision = int(self.device.command_revision)
        self._draft_mapping_revision = int(self.device.mapping_revision)
        self._zernike_status.setText("Off")
        self._show_phase()
        self._show_wavefront()
        self._sync_send_enabled()
        self._status.setText(
            "Science phase loaded; wavefront reset; hardware unchanged"
        )
        self._sync_device_state()

    @property
    def solver_idle(self) -> bool:
        return not self._running and self._pending is None

    @property
    def status_text(self) -> str:
        return self._status.text()

    @property
    def command_active(self) -> bool:
        return self._command_active

    def _sync_device_state(self) -> None:
        if self._closed:
            return
        command_revision = int(self.device.command_revision)
        mapping_revision = int(self.device.mapping_revision)
        phase = self.device.last_commanded_phase
        receipt = dict(self.device.last_command_receipt)
        self._device_diverged = (
            command_revision != self._draft_command_revision or mapping_revision != self._draft_mapping_revision
        )
        prefix = f"Device {receipt['transport']} · {receipt['outcome']} · command r{command_revision} · mapping r{mapping_revision}"
        if self._device_diverged:
            text = f"{prefix} · changed externally; Adopt or Load before Send"
        elif phase is None:
            text = f"{prefix} · command is unknown; draft is unsent"
        elif np.array_equal(phase, self._phase):
            text = f"{prefix} · draft matches device"
        else:
            text = f"{prefix} · authoring draft differs from device"
        if self._phase_metadata.get("source") == "science-context":
            text += f" · context provenance {self._context_command_receipt['identity']}"
        self._device_status.setText(text)
        self._adopt.setEnabled(not self._command_active and phase is not None)
        self._sync_send_enabled()

    def _adopt_device_command(self) -> None:
        try:
            lease = self.session.acquire_device_command(
                self, f"{self.device_key} SLM Editor adopt",
                self.device_key, self.device,
            )
        except Exception as error:
            self._status.setText(str(error))
            return
        try:
            phase = self.device.last_commanded_phase
            receipt = dict(self.device.last_command_receipt)
            revisions = (
                int(self.device.command_revision), int(self.device.mapping_revision)
            )
        finally:
            lease.release()
        if phase is None:
            self._status.setText("Device command outcome is unknown; nothing can be adopted")
            return
        self.set_phase(phase, {"source": "explicit-device-adopt"})
        self._context_command_receipt = receipt
        self._draft_command_revision, self._draft_mapping_revision = revisions
        self._sync_device_state()
        self._status.setText(
            "Device changed after Adopt snapshot; retry"
            if self._device_diverged
            else "Device command adopted as an exact authoring draft"
        )

    def _queue_solve(self) -> None:
        optimizer_state = (
            dict(self._spot_optimizer_state or {})
            if self._objective_kind == "spots" else None
        )
        self._pending = (
            self._request_revision,
            np.array(self._target, copy=True),
            self._objective_kind,
            np.array(self._pupil_amplitude, copy=True),
            optimizer_state,
        )
        self._status.setText("Solving latest target…")
        if not self._running:
            self._start_pending()

    def _start_pending(self) -> None:
        if self._closed or self._pending is None:
            return
        revision, target, objective_kind, pupil_amplitude, optimizer_state = (
            self._pending
        )
        self._pending, self._running = None, True
        future = self._executor.submit(
            solve_phase, target, initial_phase=np.array(self._pattern_phase, copy=True),
            objective_kind=objective_kind,
            pupil_amplitude=pupil_amplitude,
            spot_optimizer_state=optimizer_state,
            stop_requested=lambda: (
                self._stop.is_set() or revision != self._request_revision
            ),
        )
        future.add_done_callback(
            lambda done: self._solve_ready.emit((revision, optimizer_state, done))
        )

    @QtCore.pyqtSlot(object)
    def _finish_solve(self, payload: object) -> None:
        revision, optimizer_state, future = payload
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
                self._pattern_phase = freeze_pattern_phase(phase, self.shape)
                self._pattern_metadata = dict(metadata)
                self._spot_optimizer_state = (
                    None if optimizer_state is None else dict(optimizer_state)
                )
                self._phase_request_revision = revision
                self._compose_phase()
                self._status.setText(f"Solved with {metadata['method']}; hardware unchanged")
        self._sync_send_enabled()
        self._start_pending()
        if self._closed and self._window is not None:
            QtCore.QTimer.singleShot(0, self._window.close)

    def _show_phase(self) -> None:
        self._phase_revision += 1
        self._phase_host.update_data(_snapshot(self._phase, "phase", self._phase_revision))

    def _show_wavefront(self) -> None:
        self._wavefront_host.update_data(
            _snapshot(self._wavefront_phase, "wavefront", self._phase_revision)
        )

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
        brush = (yy - row) ** 2 + (xx - column) ** 2 <= 4
        mode = self._mode.currentText()
        if mode == "Toggle":
            target[row, column] = 0.0 if target[row, column] > 0.0 else 1.0
        else:
            target[brush] = 0.0 if mode == "Erase" else self._intensity.value()
        objective_kind = self._objective_kind if mode == "Erase" else "spots"
        self.set_target(target, objective_kind=objective_kind)
        return True

    def _save_target_operation(self, path: object):
        return partial(
            save_target, path, np.array(self._target, copy=True),
            objective_kind=self._objective_kind,
        )

    def _set_context(self, context: dict[str, object]) -> None:
        pupil, operator = context["pupil"], context["operator_metadata"]
        frozen_target = validate_target(context["target_intensity"])
        carrier = operator.get("carrier_waves_xy", [0.0, 0.0])
        coefficients = operator.get("zernike_noll_waves_rms", {})
        arrays = (
            context["phase"], context["pattern_phase"],
            context["operator_wavefront"], context["pupil_amplitude"],
            context["pupil_support"],
        )
        if any(np.asarray(array).shape != self.shape for array in arrays):
            raise ValueError(f"Science Context shape must be {self.shape!r}")
        if frozen_target.shape != self.shape:
            raise ValueError(f"Science Context Target shape must be {self.shape!r}")
        if (
            type(operator.get("enabled", False)) is not bool
            or not isinstance(carrier, list) or len(carrier) != 2
            or any(type(value) not in (int, float) or not np.isfinite(value)
                   or abs(value) > 1000.0 for value in carrier)
            or not isinstance(coefficients, dict)
            or set(coefficients) - {key for key, *_ in _ZERNIKE}
            or any(type(value) not in (int, float) or not np.isfinite(value)
                   or abs(value) > 1000.0 for value in coefficients.values())
        ):
            raise ValueError("Science Context operator metadata is invalid")
        center, diameter = pupil["center_xy"], pupil["diameter_xy"]
        if not (
            0.0 <= center[0] <= self.shape[1] - 1
            and 0.0 <= center[1] <= self.shape[0] - 1
            and diameter[0] <= 2.0 * self.shape[1]
            and diameter[1] <= 2.0 * self.shape[0]
        ):
            raise ValueError("Science Context pupil lies outside Editor limits")
        receipt = context["command_receipt"]
        if (
            type(receipt["command_revision"]) is not int
            or type(receipt["mapping_revision"]) is not int
            or receipt["outcome"] not in {"known-old", "known-new", "unknown"}
        ):
            raise ValueError("Science Context command receipt is invalid")
        lease = self.session.acquire_device_command(
            self, f"{self.device_key} SLM Editor load",
            self.device_key, self.device,
        )
        try:
            revisions = (
                int(self.device.command_revision), int(self.device.mapping_revision)
            )
        finally:
            lease.release()
        widgets = (
            self._pupil_enabled, self._pupil_center_x, self._pupil_center_y,
            self._pupil_diameter_x, self._pupil_diameter_y,
            self._zernike_enabled, self._carrier_x, self._carrier_y,
            *self._zernike.values(),
        )
        blocked = tuple(widget.blockSignals(True) for widget in widgets)
        try:
            self._pupil_enabled.setChecked(bool(pupil["enabled"]))
            self._pupil_center_x.setValue(float(pupil["center_xy"][0]))
            self._pupil_center_y.setValue(float(pupil["center_xy"][1]))
            self._pupil_diameter_x.setValue(float(pupil["diameter_xy"][0]))
            self._pupil_diameter_y.setValue(float(pupil["diameter_xy"][1]))
            self._zernike_enabled.setChecked(bool(operator.get("enabled", False)))
            self._carrier_x.setValue(float(carrier[0]))
            self._carrier_y.setValue(float(carrier[1]))
            for key, spin in self._zernike.items():
                spin.setValue(float(coefficients.get(key, 0.0)))
        finally:
            for widget, previous in zip(widgets, blocked):
                widget.blockSignals(previous)
        self._request_revision += 1
        self._pending, self._spot_optimizer_state = None, None
        self._target = frozen_target
        self._target_revision += 1
        self._target_host.update_data(
            _snapshot(frozen_target, "target", self._target_revision)
        )
        self._phase = context["phase"]
        self._pattern_phase = context["pattern_phase"]
        self._wavefront_phase = context["operator_wavefront"]
        self._pupil_amplitude = context["pupil_amplitude"]
        self._pupil_support = context["pupil_support"]
        self._pupil_center_xy = tuple(float(value) for value in pupil["center_xy"])
        self._pupil_diameter_xy = tuple(float(value) for value in pupil["diameter_xy"])
        self._pupil_applied_description = self._pupil_description(bool(pupil["enabled"]))
        self._pupil_status.setText(self._pupil_applied_description)
        self._zernike_status.setText("On · frozen context" if operator.get("enabled") else "Off")
        self._objective_kind = str(context["objective_kind"])
        self._pattern_metadata = dict(context["pattern_metadata"])
        self._phase_metadata = {"source": "science-context"}
        self._system_correction = None if context["system_correction"] is None else dict(context["system_correction"])
        self._context_command_receipt = dict(context["command_receipt"])
        self._draft_command_revision, self._draft_mapping_revision = revisions
        self._phase_request_revision = self._request_revision
        self._show_phase(); self._show_wavefront(); self._sync_device_state()
        self._status.setText("Frozen Science Context loaded; hardware unchanged")

    def _save_context_operation(self, path: object):
        if self._phase_request_revision != self._request_revision:
            raise RuntimeError("wait for the latest target solve before saving context")
        return partial(
            save_science_context, path,
            np.array(self._pattern_phase, copy=True),
            target_intensity=np.array(self._target, copy=True),
            objective_kind=self._objective_kind,
            pupil=self._pupil_settings(),
            system_correction=deepcopy(self._system_correction),
            command_receipt=deepcopy(self._context_command_receipt),
            pattern_metadata=deepcopy(self._pattern_metadata),
            operator_metadata=self._operator_settings(),
        )

    def send(self) -> bool:
        """Queue the clicked canonical phase; return whether it was accepted."""

        if self._phase_request_revision != self._request_revision:
            self._status.setText("Wait for the latest target solve before Send")
            self._sync_send_enabled()
            return False
        self._sync_device_state()
        if self._device_diverged:
            self._status.setText(
                "Device command changed externally; Adopt or Load before Send"
            )
            return False
        expected = canonical_phase(self._phase, self.shape)
        return self._queue_work(
            "Phase sent to SLM", partial(self._apply_phase, expected),
            claim_device=True,
        )

    def _queue_work(
        self, label: str, operation: object, *, claim_device: bool = False,
        completion: object = None,
    ) -> bool:
        if self._closed or self._command_active:
            self._status.setText(
                "SLM Editor is closing" if self._closed
                else "SLM command already in progress"
            )
            return False
        command_revision = self._draft_command_revision
        mapping_revision = self._draft_mapping_revision
        self._command_active = True
        self._sync_send_enabled()
        self._status.setText("Working…")
        try:
            future = self._command_executor.submit(
                self._run_device_command, label, operation,
                command_revision, mapping_revision,
            ) if claim_device else self._command_executor.submit(
                operation,
            )
        except Exception as error:
            self._command_active = False
            self._sync_send_enabled()
            self._status.setText(str(error))
            return False
        future.add_done_callback(
            lambda done: self._command_ready.emit(
                (label, claim_device, completion, done)
            )
        )
        return True

    def _run_device_command(
        self, label: str, operation: object,
        command_revision: int, mapping_revision: int,
    ) -> object:
        lease = self.session.acquire_device_command(
            self, f"{self.device_key} SLM Editor {label}",
            self.device_key, self.device,
        )
        try:
            if (
                self.device.command_revision != command_revision
                or self.device.mapping_revision != mapping_revision
            ):
                raise RuntimeError(
                    "Device command changed before ownership was acquired; reconcile first"
                )
            result = operation()
            return (
                result, int(self.device.command_revision),
                int(self.device.mapping_revision), dict(self.device.last_command_receipt),
            )
        finally:
            lease.release()

    def _apply_phase(self, expected: np.ndarray) -> None:
        applied = self.device.apply_phase(expected)
        receipt = dict(self.device.last_command_receipt)
        if not np.array_equal(applied, expected) or not np.array_equal(
            self.device.last_commanded_phase, expected
        ) or (
            receipt.get("outcome") != "known-new"
            or receipt.get("identity") != self.device.identity
            or receipt.get("command_revision") != self.device.command_revision
            or receipt.get("mapping_revision") != self.device.mapping_revision
        ):
            raise RuntimeError("SLM did not confirm a known-new command receipt")

    @QtCore.pyqtSlot(object)
    def _finish_command(self, payload: object) -> None:
        label, claimed, completion, future = payload
        self._command_active = False
        try:
            result = future.result()
            if claimed:
                _value, command_revision, mapping_revision, receipt = result
            elif completion is not None:
                completion(result)
        except Exception as error:
            self._status.setText(str(error))
        else:
            if claimed:
                self._draft_mapping_revision = mapping_revision
                if label.startswith("Phase"):
                    self._draft_command_revision = command_revision
                    self._context_command_receipt = receipt
            if completion is None:
                self._status.setText(label)
        self._sync_correction_controls()
        self._sync_device_state()
        if self._closed and self._window is not None:
            QtCore.QTimer.singleShot(0, self._window.close)

    def _sync_send_enabled(self) -> None:
        send = getattr(self, "_send", None)
        if send is not None:
            send.setEnabled(
                not self._closed
                and not self._command_active
                and not self._device_diverged
                and self._phase_request_revision == self._request_revision
            )

    def _choose(self, action: str) -> None:
        actions = {
            "load_target": (False, "target.json", "SLM target (*.json)"),
            "save_target": (True, "target.json", "SLM target (*.json)"),
            "load_context": (False, "science-context.npz", "SLM Science Context (*.npz)"),
            "save_context": (True, "science-context.npz", "SLM Science Context (*.npz)"),
            "import": (False, "", "Array or image (*.npy *.png *.tif *.tiff *.bmp)"),
            "correction": (False, "correction_Pattern.bmp", "8-bit correction BMP (*.bmp)"),
        }
        saving, name, filters = actions[action]
        start = str(Path(self.session.workspace.data) / name)
        picker = fluent_save_path if saving else fluent_open_path
        path = picker(self._body, action.replace("_", " ").title(), start, filters)
        if path:
            try:
                if action == "load_target":
                    operation, completion, label = (
                        partial(load_target, path),
                        lambda result: self.set_target(
                            result[0], objective_kind=result[1]
                        ), "Target loaded",
                    )
                elif action == "load_context":
                    operation, completion, label = (
                        partial(load_science_context, path), self._set_context,
                        "Science Context loaded",
                    )
                elif action == "import":
                    operation, completion, label = (
                        partial(_read_imported_target, path),
                        lambda target: self.set_target(target), "Target imported",
                    )
                elif action == "save_target":
                    operation, completion, label = (
                        self._save_target_operation(path), None, "Target saved",
                    )
                elif action == "save_context":
                    operation, completion, label = (
                        self._save_context_operation(path), None,
                        "Science Context saved",
                    )
                else:
                    self._queue_work(
                        "Vendor correction loaded",
                        partial(self.device.load_correction, path),
                        claim_device=True,
                    )
                    return
                self._queue_work(label, operation, completion=completion)
            except Exception as error:
                self._status.setText(str(error))

    def _finish_close(self) -> bool:
        if self._cleaned:
            return True
        if not self._closed:
            self._closed, self._pending = True, None
            self._close_deadline = time.monotonic() + 2.0
            self._device_poll.stop()
            self._spot_optimizer_state = None
            self._stop.set()
            self._body.setEnabled(False)
            self._status.setText("Stopping SLM Editor…")
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._command_executor.shutdown(wait=False, cancel_futures=False)
        for host in (self._target_host, self._phase_host, self._wavefront_host):
            host.qt_widget().close_adapter()
        hosts_stopped = tuple(
            host.close(timeout=0.0)
            for host in (self._target_host, self._phase_host, self._wavefront_host)
        )
        if self._running or self._command_active or not all(hosts_stopped):
            if time.monotonic() >= self._close_deadline:
                self._status.setText(
                    "SLM Editor close timed out; waiting for active work to finish"
                )
            elif self._window is not None:
                QtCore.QTimer.singleShot(50, self._window.close)
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
