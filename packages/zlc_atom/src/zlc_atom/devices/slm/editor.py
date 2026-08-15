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
    FluentPopup, FluentScrollArea, FluentSettingsPopupAnchor, FluentSpinBox,
    FluentSwitch, FluentTabWidget, fluent_open_path, fluent_save_path,
    open_fluent_window,
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
        self._objective_kind = "spots"
        height, width = self.shape
        self._pupil_center_xy = (0.5 * (width - 1), 0.5 * (height - 1))
        self._pupil_radius_xy = (0.45 * (width - 1), 0.45 * (height - 1))
        self._pupil_measured: np.ndarray | None = None
        self._pupil_measured_name = ""
        self._pupil_support = self._assumed_pupil(
            self._pupil_center_xy, self._pupil_radius_xy
        ).astype(bool)
        self._pupil_amplitude = self._pupil_support.astype(np.float32)
        self._pupil_applied_description = "assumed ellipse · configured unit ellipse"
        self._wavefront_phase = canonical_phase(
            np.zeros(self.shape, dtype=np.float32), self.shape
        )
        self._request_revision = self._target_revision = self._phase_revision = 0
        self._phase_request_revision: int | None = None
        self._pending: tuple[int, np.ndarray, str, np.ndarray] | None = None
        self._running = self._painting = self._closed = self._cleaned = False
        self._command_active = False
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

        self._pupil_title = QtWidgets.QLabel("Input pupil", layers)
        self._pupil_status = QtWidgets.QLabel("Assumed ellipse", layers)
        self._pupil_edit = FluentButton("Edit…", layers)
        self._build_pupil_popup(page)
        self._pupil_edit.clicked.connect(
            lambda: self._pupil_anchor.toggle(
                self._pupil_body, minimum_width=430, minimum_height=330,
            )
        )

        self._crop_enabled = FluentSwitch("CGH crop", layers)
        self._crop_status = QtWidgets.QLabel("Off", layers)
        self._crop_edit = FluentButton("Edit…", layers)
        self._build_crop_popup(page)
        self._crop_edit.clicked.connect(
            lambda: self._crop_anchor.toggle(
                self._crop_body, minimum_width=390, minimum_height=260,
            )
        )

        self._steering_enabled = FluentSwitch("Zero-order steering", layers)
        self._steering_enabled.setChecked(True)
        self._steering_enabled.setToolTip(
            "Moves the modulated order; a physical Fourier-plane stop/aperture "
            "is still required to reject zero order."
        )
        self._steering_status = QtWidgets.QLabel("On · 0, 0 waves", layers)
        steering_edit = FluentButton("Edit…", layers)
        steering_edit.clicked.connect(
            lambda: self._layer_tabs.setCurrentIndex(self._wavefront_tab_index)
        )

        self._zernike_enabled = FluentSwitch("Zernike / system wavefront", layers)
        self._zernike_enabled.setChecked(True)
        self._zernike_status = QtWidgets.QLabel("On · 0 active", layers)
        zernike_edit = FluentButton("Edit…", layers)
        zernike_edit.clicked.connect(
            lambda: self._layer_tabs.setCurrentIndex(self._wavefront_tab_index)
        )
        for row, column, widgets in (
            (0, 0, (self._pupil_title, self._pupil_status, self._pupil_edit)),
            (0, 3, (self._crop_enabled, self._crop_status, self._crop_edit)),
            (1, 0, (self._steering_enabled, self._steering_status, steering_edit)),
            (1, 3, (self._zernike_enabled, self._zernike_status, zernike_edit)),
        ):
            for offset, widget in enumerate(widgets):
                layer_layout.addWidget(widget, row, column + offset)

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
        for column in (1, 4):
            layer_layout.setColumnStretch(column, 1)
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
            ("Load mask", "load_mask"),
            ("Save mask", "save_mask"),
            ("Load science phase", "load_phase"),
            ("Save science phase", "save_phase"),
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
            ("Grid", "Checkerboard", "Rectangle", "Ellipse", "Ring")
        )
        form.addWidget(self._preset_type, 0, 1)
        specs = (
            ("rows", "N rows", True, 5, 1, self.shape[0]),
            ("columns", "M columns", True, 7, 1, self.shape[1]),
            ("spacing_y", "Spacing Y", True, max(1, self.shape[0] // 6), 1, self.shape[0]),
            ("spacing_x", "Spacing X", True, max(1, self.shape[1] // 8), 1, self.shape[1]),
            ("intensity", "Intensity", False, 1.0, 0.0, 1000.0),
            ("intensity_a", "Intensity A", False, 1.0, 0.0, 1000.0),
            ("intensity_b", "Intensity B", False, 0.25, 0.0, 1000.0),
            ("height", "Height", True, max(2, self.shape[0] // 3), 1, self.shape[0]),
            ("width", "Width", True, max(2, self.shape[1] // 2), 1, self.shape[1]),
            ("edge", "Edge", False, max(2, min(self.shape) // 32), 0.0, min(self.shape)),
            ("radius_y", "Radius Y", True, max(1, self.shape[0] // 6), 1, self.shape[0] // 2),
            ("radius_x", "Radius X", True, max(1, self.shape[1] // 4), 1, self.shape[1] // 2),
            ("radius", "Radius", True, max(1, min(self.shape) // 4), 1, min(self.shape) // 2),
            ("ring_width", "Ring width", True, max(1, min(self.shape) // 12), 1, min(self.shape)),
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
        apply_button = FluentButton("Apply", body)
        apply_button.clicked.connect(self._apply_preset_popup)
        form.addWidget(apply_button, len(specs) + 1, 0, 1, 2)
        self._preset_popup, self._preset_body = popup, body
        self._preset_anchor = anchor
        self._preset_type.currentTextChanged.connect(self._sync_preset_popup)
        self._sync_preset_popup()

    def _sync_preset_popup(self) -> None:
        visible = {
            "Grid": {"rows", "columns", "spacing_y", "spacing_x", "intensity"},
            "Checkerboard": {
                "rows", "columns", "spacing_y", "spacing_x",
                "intensity_a", "intensity_b",
            },
            "Rectangle": {"height", "width", "edge", "intensity"},
            "Ellipse": {"radius_y", "radius_x", "edge", "intensity"},
            "Ring": {"radius", "ring_width", "edge", "intensity"},
        }[self._preset_type.currentText()]
        for key, widgets in self._preset_field_rows.items():
            for widget in widgets:
                widget.setVisible(key in visible)
        self._preset_popup.adjustSize()

    def _apply_preset_popup(self) -> None:
        value = lambda key: self._preset_fields[key].value()
        kind = self._preset_type.currentText()
        grid = (value("rows"), value("columns"))
        spacing = (value("spacing_y"), value("spacing_x"))
        makers = {
            "Grid": ("spots", preset_grid, (grid,), {"spacing_yx": spacing, "intensity": value("intensity")}),
            "Checkerboard": ("spots", preset_checkerboard, (grid,), {"spacing_yx": spacing, "intensity_a": value("intensity_a"), "intensity_b": value("intensity_b")}),
            "Rectangle": ("image", preset_rectangle, ((value("height"), value("width")),), {"edge": value("edge"), "intensity": value("intensity")}),
            "Ellipse": ("image", preset_ellipse, ((value("radius_y"), value("radius_x")),), {"edge": value("edge"), "intensity": value("intensity")}),
            "Ring": ("image", preset_ring, (), {"radius": value("radius"), "width": value("ring_width"), "edge": value("edge"), "intensity": value("intensity")}),
        }
        objective, make, args, kwargs = makers[kind]
        try:
            target = make(self.shape, *args, **kwargs)
        except (TypeError, ValueError) as error:
            self._status.setText(str(error))
            return
        self._preset_popup.hide()
        self.set_target(target, objective_kind=objective)

    def _build_pupil_popup(self, parent: QtWidgets.QWidget) -> None:
        popup, body, form, anchor = self._popup_form(parent, self._pupil_edit)
        self._pupil_type = FluentComboBox(body)
        self._pupil_type.addItems(("Assumed ellipse", "Measured"))
        form.addWidget(QtWidgets.QLabel("Source", body), 0, 0)
        form.addWidget(self._pupil_type, 0, 1)
        cx, cy = self._pupil_center_xy
        rx, ry = self._pupil_radius_xy
        fields = (
            ("_pupil_center_x", "Center X", cx, 0.0, self.shape[1] - 1),
            ("_pupil_center_y", "Center Y", cy, 0.0, self.shape[0] - 1),
            ("_pupil_radius_x", "Radius / half-width X", rx, 1.0, self.shape[1]),
            ("_pupil_radius_y", "Radius / half-height Y", ry, 1.0, self.shape[0]),
        )
        for row, (name, label, value, minimum, maximum) in enumerate(fields, start=1):
            spin = _number_spin(body, value, minimum, maximum, decimals=2)
            setattr(self, name, spin)
            form.addWidget(QtWidgets.QLabel(label, body), row, 0)
            form.addWidget(spin, row, 1)
        load = FluentButton("Load measured intensity…", body)
        load.clicked.connect(partial(self._choose, "load_pupil"))
        apply_button = FluentButton("Apply pupil", body)
        apply_button.clicked.connect(self._apply_pupil)
        form.addWidget(load, 5, 0, 1, 2)
        form.addWidget(apply_button, 6, 0, 1, 2)
        self._pupil_popup, self._pupil_body = popup, body
        self._pupil_anchor = anchor

    def _build_crop_popup(self, parent: QtWidgets.QWidget) -> None:
        popup, body, form, anchor = self._popup_form(parent, self._crop_edit)
        side, height, width = min(self.shape), self.shape[0], self.shape[1]
        values = (
            ("_crop_x", "X", 0, width - 1, (width - side) // 2),
            ("_crop_y", "Y", 0, height - 1, (height - side) // 2),
            ("_crop_width", "Width", 1, width, side),
            ("_crop_height", "Height", 1, height, side),
        )
        for row, (name, label, minimum, maximum, value) in enumerate(values):
            spin = _number_spin(
                body, value, minimum, maximum, integral=True
            )
            setattr(self, name, spin)
            spin.valueChanged.connect(self._crop_changed)
            form.addWidget(QtWidgets.QLabel(label, body), row, 0)
            form.addWidget(spin, row, 1)
        self._crop_popup, self._crop_body = popup, body
        self._crop_anchor = anchor

    def _assumed_pupil(self, center_xy: tuple[float, float],
                       radius_xy: tuple[float, float]) -> np.ndarray:
        yy, xx = np.ogrid[:self.shape[0], :self.shape[1]]
        cx, cy = center_xy
        rx, ry = radius_xy
        pupil = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        return np.asarray(pupil, dtype=np.float32)

    def _load_pupil_path(self, path: object) -> None:
        source = Path(path)
        if source.suffix.lower() == ".npy":
            intensity = np.load(source, allow_pickle=False)
        else:
            from PIL import Image
            with Image.open(source) as image:
                intensity = np.asarray(image.convert("F"), dtype=np.float32)
        intensity = np.asarray(intensity, dtype=np.float32)
        if intensity.shape != self.shape:
            raise ValueError(f"measured pupil intensity shape must be {self.shape!r}")
        if not np.all(np.isfinite(intensity)) or np.any(intensity < 0.0):
            raise ValueError("measured pupil intensity must be finite and non-negative")
        if not np.any(intensity > 0.0):
            raise ValueError("measured pupil intensity must contain positive values")
        self._pupil_measured = np.sqrt(intensity).astype(np.float32)
        self._pupil_measured_name = source.name
        self._pupil_type.setCurrentText("Measured")
        self._pupil_status.setText(f"Draft · measured {source.name} · Apply")

    def _apply_pupil(self, *_args: object) -> None:
        self._pupil_center_xy = (
            self._pupil_center_x.value(), self._pupil_center_y.value(),
        )
        self._pupil_radius_xy = (
            self._pupil_radius_x.value(), self._pupil_radius_y.value(),
        )
        kind = self._pupil_type.currentText()
        support = self._assumed_pupil(
            self._pupil_center_xy, self._pupil_radius_xy
        )
        if kind == "Measured" and self._pupil_measured is not None:
            amplitude = self._pupil_measured * support
            description = f"measured {self._pupil_measured_name} · configured unit ellipse"
        else:
            if kind == "Measured":
                kind = "Assumed ellipse"
                self._pupil_type.setCurrentText(kind)
            amplitude = support
            description = "assumed ellipse · configured unit ellipse"
        self._pupil_support = support.astype(bool)
        self._pupil_amplitude = amplitude
        self._pupil_applied_description = description
        self._pupil_status.setText(description)
        self._pupil_popup.hide()
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
        self._correction_status.setText(f"{state} · {name}" if name else state)

    def _set_correction_enabled(self, enabled: bool) -> None:
        try:
            self.device.set_correction_enabled(bool(enabled))
        except Exception as error:
            self._status.setText(str(error))
        self._sync_correction_controls()

    def _build_layer_tabs(self, parent: QtWidgets.QWidget) -> FluentTabWidget:
        tabs = FluentTabWidget(parent)
        self._layer_tabs = tabs
        self._editor_tab_index = tabs.add_permanent_tab(
            self._build_editor_page(tabs), "Mask"
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
            "Zero-order steering moves the modulated order. A physical "
            "Fourier-plane stop/aperture is still required to reject zero order. "
            "Coefficients are Noll-normalized waves RMS on the applied pupil.",
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
            (key, label, f"{label}: {polynomial}; Noll-normalized waves RMS on the pupil.")
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

        self._crop_enabled.toggled.connect(self._crop_changed)
        self._steering_enabled.toggled.connect(self._layer_changed)
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

    def _crop_changed(self, *_args: object) -> None:
        self._crop_width.setMaximum(self.shape[1] - self._crop_x.value())
        self._crop_height.setMaximum(self.shape[0] - self._crop_y.value())
        if self._crop_enabled.isChecked():
            self._crop_status.setText(
                f"On · {self._crop_x.value()},{self._crop_y.value()} · "
                f"{self._crop_width.value()}×{self._crop_height.value()}"
            )
        else:
            self._crop_status.setText("Off")
        self._layer_changed()

    def _layer_changed(self, *_args: object) -> None:
        if self._steering_enabled.isChecked():
            self._steering_status.setText(
                f"On · {self._carrier_x.value():g}, "
                f"{self._carrier_y.value():g} waves"
            )
        else:
            self._steering_status.setText("Off")
        active = sum(bool(spin.value()) for spin in self._zernike.values())
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
        height, width = self.shape
        full_y, full_x = np.ogrid[
            -1.0:1.0:height * 1j, -1.0:1.0:width * 1j
        ]
        phase = np.zeros(self.shape, dtype=np.float64)
        if self._steering_enabled.isChecked():
            phase += np.pi * (
                self._carrier_x.value() * full_x
                + self._carrier_y.value() * full_y
            )
        coefficients = {key: self._zernike[key].value() for key, *_ in _ZERNIKE}
        if self._zernike_enabled.isChecked() and any(coefficients.values()):
            yy, xx = np.ogrid[:height, :width]
            center_x, center_y = self._pupil_center_xy
            radius_x, radius_y = self._pupil_radius_xy
            zx = (xx - center_x) / radius_x
            zy = (yy - center_y) / radius_y
            r2 = zx * zx + zy * zy
            pupil = self._pupil_support
            modes = {
                "tilt_x": lambda: 2.0 * zx,
                "tilt_y": lambda: 2.0 * zy,
                "defocus": lambda: np.sqrt(3.0) * (2.0 * r2 - 1.0),
                "astig_oblique": lambda: 2.0 * np.sqrt(6.0) * zx * zy,
                "astig_vertical": lambda: np.sqrt(6.0) * (zx * zx - zy * zy),
                "coma_y": lambda: np.sqrt(8.0) * zy * (3.0 * r2 - 2.0),
                "coma_x": lambda: np.sqrt(8.0) * zx * (3.0 * r2 - 2.0),
                "trefoil_y": lambda: np.sqrt(8.0) * zy * (3.0 * zx * zx - zy * zy),
                "trefoil_x": lambda: np.sqrt(8.0) * zx * (zx * zx - 3.0 * zy * zy),
                "spherical": lambda: np.sqrt(5.0) * (6.0 * r2 * r2 - 6.0 * r2 + 1.0),
            }
            for key, coefficient in coefficients.items():
                if not coefficient:
                    continue
                values = modes[key]()
                phase[pupil] += (
                    2.0 * np.pi * coefficient
                    * np.broadcast_to(values, self.shape)[pupil]
                )
        self._wavefront_phase = canonical_phase(phase, self.shape)
        if self._crop_enabled.isChecked():
            x, y = self._crop_x.value(), self._crop_y.value()
            width, height = self._crop_width.value(), self._crop_height.value()
            phase[y:y + height, x:x + width] += self._mask_phase[
                y:y + height, x:x + width
            ]
        else:
            phase += self._mask_phase
        self._phase = canonical_phase(phase, self.shape)
        self._phase_metadata = {
            "source": "composite",
            "hardware_correction": "excluded",
            "mask": dict(self._mask_metadata),
            "input_pupil": self._pupil_applied_description,
            "steering_enabled": self._steering_enabled.isChecked(),
            "carrier_waves_xy": [
                self._carrier_x.value(), self._carrier_y.value(),
            ],
            "zernike_enabled": self._zernike_enabled.isChecked(),
            "zernike_noll_waves_rms": coefficients,
            "cgh_crop_xywh": [
                self._crop_x.value(), self._crop_y.value(),
                self._crop_width.value(), self._crop_height.value(),
            ] if self._crop_enabled.isChecked() else None,
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
        self._phase = canonical_phase(values, self.shape)
        self._mask_phase = self._phase
        self._phase_request_revision = self._request_revision
        self._phase_metadata = dict(
            {"source": "loaded"} if metadata is None else metadata
        )
        self._mask_metadata = dict(self._phase_metadata)
        widgets = (
            self._carrier_x, self._carrier_y, *self._zernike.values(),
            self._crop_enabled, self._steering_enabled, self._zernike_enabled,
        )
        blocked = tuple(widget.blockSignals(True) for widget in widgets)
        try:
            self._carrier_x.setValue(0.0)
            self._carrier_y.setValue(0.0)
            for spin in self._zernike.values():
                spin.setValue(0.0)
            self._crop_enabled.setChecked(False)
            self._steering_enabled.setChecked(False)
            self._zernike_enabled.setChecked(False)
        finally:
            for widget, previous in zip(widgets, blocked):
                widget.blockSignals(previous)
        self._wavefront_phase = canonical_phase(
            np.zeros(self.shape, dtype=np.float32), self.shape
        )
        self._crop_status.setText("Off")
        self._steering_status.setText("Off")
        self._zernike_status.setText("Off")
        self._show_phase()
        self._show_wavefront()
        self._sync_send_enabled()
        self._status.setText(
            "Science phase loaded; composition layers reset; hardware unchanged"
        )

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
        self._pending = (
            self._request_revision,
            np.array(self._target, copy=True),
            self._objective_kind,
            np.array(self._pupil_amplitude, copy=True),
        )
        self._status.setText("Solving latest target…")
        if not self._running:
            self._start_pending()

    def _start_pending(self) -> None:
        if self._closed or self._pending is None:
            return
        revision, target, objective_kind, pupil_amplitude = self._pending
        self._pending, self._running = None, True
        future = self._executor.submit(
            solve_phase, target, initial_phase=np.array(self._mask_phase, copy=True),
            objective_kind=objective_kind,
            pupil_amplitude=pupil_amplitude,
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
        mask = (yy - row) ** 2 + (xx - column) ** 2 <= 4
        mode = self._mode.currentText()
        if mode == "Toggle":
            target[row, column] = 0.0 if target[row, column] > 0.0 else 1.0
        else:
            target[mask] = 0.0 if mode == "Erase" else self._intensity.value()
        objective_kind = self._objective_kind if mode == "Erase" else "spots"
        self.set_target(target, objective_kind=objective_kind)
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
        elif self._crop_enabled.isChecked() and gray.shape == (
            self._crop_height.value(), self._crop_width.value(),
        ):
            values = np.zeros(self.shape, dtype=np.float64)
            x, y = self._crop_x.value(), self._crop_y.value()
            values[
                y:y + gray.shape[0], x:x + gray.shape[1]
            ] = gray.astype(np.float64) * (2.0 * np.pi / 256.0)
        else:
            raise ValueError(
                "phase mask image shape must match the device or enabled CGH crop"
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
            "load_pupil": (False, "", "Array or image (*.npy *.png *.tif *.tiff *.bmp)", self._load_pupil_path),
            "correction": (False, "correction_Pattern.bmp", "8-bit correction BMP (*.bmp)", lambda path: (self.device.load_correction(path), self._sync_correction_controls())),
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
        for host in (self._target_host, self._phase_host, self._wavefront_host):
            host.qt_widget().close_adapter()
        hosts_stopped = tuple(
            host.close(timeout=0.0)
            for host in (self._target_host, self._phase_host, self._wavefront_host)
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
