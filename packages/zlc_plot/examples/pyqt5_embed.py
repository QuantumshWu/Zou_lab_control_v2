"""Schema-driven PyQt5 embedding example for every public plot kind.

The window is intentionally a thin application shell.  Plot parameters come
from ``DisplayDescription`` and are rendered by ``Qt5ParameterPanel``; all
accepted edits still pass through ``RasterPlotHost`` and the core validators.
The acquisition timer publishes immutable revisions independently of rendering.

Run from the repository root::

    python examples/pyqt5_embed.py

``create_window()`` is the non-blocking entry point for a larger PyQt5
application or a notebook whose Qt event loop is already running.
"""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any, Callable

# This checkout must win over any installed zlc_* distribution.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import zou_lab_control_v2  # noqa: F401

import numpy as np

try:
    from examples._role_data import (
        Axis, DatasetSchema, DatasetSnapshot, OwnedSnapshot, PointTable, PointTopology,
    )
except ModuleNotFoundError:  # direct ``python examples/pyqt5_embed.py``
    from _role_data import (
        Axis, DatasetSchema, DatasetSnapshot, OwnedSnapshot, PointTable, PointTopology,
    )
from zlc_plot import (
    AxisRef,
    BackendUnavailableError,
    CurvePlot,
    DEFAULTS,
    schema_summary,
    updated_spec,
    FacetGridPlot,
    FitEvent,
    HistogramPlot,
    ImageFrame,
    ImagePointOverlay,
    ImagePlot,
    LiveDataRevision,
    PlotKind,
    PlotLabels,
    PlotSpec,
    PointStatus,
    PulseAnalogTrace,
    PulseBlock,
    PulseChannel,
    PulseDacScanSegment,
    PulseRepeatMarker,
    PulseScanRegion,
    PulseTimelineData,
    PulseTimelinePlot,
    Qt5ParameterPanel,
    Qt5PlotWidget,
    RasterPlotHost,
    RollingPlot,
    SelectionChange,
    SelectorData,
    SelectorKind,
    PulseTimelineSelectionData,
    ensure_qt5_application,
)
_DEMO_GENERATION = "zlc-plot-pyqt5-demo"


@dataclass(frozen=True, slots=True)
class _DatasetSource:
    schema: DatasetSchema
    time_rows: np.ndarray
    field_rows: np.ndarray


@dataclass(frozen=True, slots=True)
class _DemoDefinition:
    """One data source page.

    A page's identity is its dataset, never a plot kind: the semantic panel
    is the only place a kind is chosen, and every kind a page's data admits
    is reachable from there.  Two kind selectors — one silently swapping the
    example data — was exactly the confusion this replaces.
    """

    page_id: str
    label: str
    initial: OwnedSnapshot | ImageFrame | PulseTimelineData
    spec: PlotSpec
    make_update: Callable[[int], OwnedSnapshot | ImageFrame | LiveDataRevision]

    @property
    def live_contract(self) -> object:
        if isinstance(self.initial, (OwnedSnapshot, ImageFrame)):
            return self.initial
        return LiveDataRevision(0, self.initial)


@dataclass(slots=True)
class _ActivePlot:
    definition: _DemoDefinition
    host: RasterPlotHost
    widget: Qt5PlotWidget
    panel: Qt5ParameterPanel
    live: object
    revision: int
    semantics: object


def _dataset_source() -> _DatasetSource:
    time_axis = Axis.create(
        "time",
        values=np.linspace(-2.5e-3, 2.5e-3, 61),
        label="Time",
        canonical_unit="s",
        display_unit="ms",
    )
    field_axis = Axis.create(
        "field",
        values=np.linspace(-3.0e-3, 3.0e-3, 21),
        label="Field",
        canonical_unit="V",
        display_unit="mV",
    )
    field_rows = np.repeat(field_axis.coordinates, time_axis.size)
    time_rows = np.tile(time_axis.coordinates, field_axis.size)
    points = PointTable.from_columns(
        {"field": field_rows, "time": time_rows},
        units={"field": "V", "time": "s"},
        display_units={"field": "mV", "time": "ms"},
        labels={"field": "Field", "time": "Time"},
    )
    topology = PointTopology.from_cartesian(
        (field_axis, time_axis),
        point_table=points,
    )
    repeat = Axis.create("repeat", values=np.arange(24), label="Repeat")
    site = Axis.create(
        "site", values=np.arange(4), label="Site", canonical_unit="1"
    )
    schema = DatasetSchema.create(
        repeat,
        points,
        data_axes=(site,),
        point_topology=topology,
        value_label="Signal",
        canonical_unit="V",
        display_unit="mV",
        dtype=np.float64,
        generation=_DEMO_GENERATION,
    )
    return _DatasetSource(schema, time_rows, field_rows)


_DATASET_SOURCE = _dataset_source()


def _image_schema() -> DatasetSchema:
    repeat = Axis.create("repeat", values=np.array([0], dtype=np.int64))
    points = PointTable.from_columns(
        {"sample": np.array([0], dtype=np.int64)},
        labels={"sample": "Camera sample"},
    )
    y_axis = Axis.create(
        "image_y",
        values=np.linspace(-3.0e-6, 3.0e-6, 81),
        label="Position Y",
        canonical_unit="m",
        display_unit="um",
    )
    x_axis = Axis.create(
        "image_x",
        values=np.linspace(-4.0e-6, 4.0e-6, 101),
        label="Position X",
        canonical_unit="m",
        display_unit="um",
    )
    return DatasetSchema.create(
        repeat,
        points,
        data_axes=(y_axis, x_axis),
        value_label="Signal",
        canonical_unit="V",
        display_unit="mV",
        dtype=np.float64,
        generation=f"{_DEMO_GENERATION}-image",
    )


_IMAGE_SCHEMA = _image_schema()


def _image_frame(revision: int) -> OwnedSnapshot:
    y = np.asarray(_IMAGE_SCHEMA.data_axes[0].coordinates, dtype=float)[:, None]
    x = np.asarray(_IMAGE_SCHEMA.data_axes[1].coordinates, dtype=float)[None, :]
    phase = 0.11 * float(revision)
    center_x = 0.65e-6 * np.sin(phase)
    center_y = 0.55e-6 * np.cos(0.8 * phase)
    radius = 1.15e-6
    image = 0.25e-3 + 2.4e-3 * np.exp(
        -((x - center_x) ** 2 + (y - center_y) ** 2) / radius**2
    )
    return DatasetSnapshot(
        _IMAGE_SCHEMA,
        image[np.newaxis, np.newaxis, :, :],
        revision=revision,
    )


def _dataset_frame(revision: int) -> OwnedSnapshot:
    source = _DATASET_SOURCE
    phase = 0.13 * float(revision)
    center_x = 0.55e-3 * np.sin(phase)
    center_y = 0.8e-3 * np.cos(0.7 * phase)
    image = 0.32e-3 + 2.2e-3 * np.exp(
        -0.5 * ((source.time_rows - center_x) / 0.72e-3) ** 2
        -0.5 * ((source.field_rows - center_y) / 1.15e-3) ** 2
    )
    repeat = np.arange(source.schema.R, dtype=np.float64)
    repeat_signal = 0.18e-3 * np.sin(
        2.0 * np.pi * repeat / source.schema.R + phase
    )
    repeat_scale = np.where((repeat % 2.0) == 0.0, 1.0, 1.3)
    repeat_offset = 0.06e-3 * ((repeat % 4.0) - 1.5)
    sites = np.arange(source.schema.data_axes[0].size, dtype=np.float64)
    values = (
        image[np.newaxis, :, np.newaxis]
        * repeat_scale[:, np.newaxis, np.newaxis]
        + repeat_signal[:, np.newaxis, np.newaxis]
        + repeat_offset[:, np.newaxis, np.newaxis]
        + 0.035e-3 * sites[np.newaxis, np.newaxis, :]
    )
    return DatasetSnapshot(source.schema, values, revision=revision)


_IMAGE_POINT_COORDINATES = np.asarray(
    ((-1.4e-6, 1.2e-6), (0.0, -0.9e-6), (1.6e-6, 0.8e-6)),
    dtype=np.float64,
)


def _image_point_overlay(revision: int) -> ImagePointOverlay:
    phase = 0.08 * float(revision)
    offset = np.asarray(
        (0.16e-6 * np.sin(phase), 0.12e-6 * np.cos(phase)),
        dtype=np.float64,
    )
    return ImagePointOverlay(
        revision=revision,
        coordinates=_IMAGE_POINT_COORDINATES + offset,
        point_ids=("point-01", "point-02", "point-03"),
        labels=("A", "B", "C"),
        statuses=(PointStatus.EMPTY, PointStatus.OCCUPIED, PointStatus.INVALID),
    )


def _pulse_frame(revision: int) -> PulseTimelineData:
    scanned_value = -0.42 + 0.16 * np.sin(0.12 * float(revision))
    return PulseTimelineData(
        channels=(
            PulseChannel("laser", "Laser"),
            PulseChannel("microwave", "Microwave"),
            PulseChannel("detect", "Detect"),
        ),
        blocks=(
            PulseBlock("laser", 0.0, 1.4e-6, label="Init"),
            PulseBlock("microwave", 2.0e-6, 4.0e-6, label="Drive"),
            PulseBlock("laser", 7.1e-6, 8.3e-6, label="Read"),
            PulseBlock("detect", 8.3e-6, 9.5e-6, label="Count"),
        ),
        analog_traces=(
            PulseAnalogTrace(
                "bias",
                "Bias",
                -1.0,
                1.0,
                starts=(0.0, 2.0e-6, 4.2e-6, 6.8e-6, 10.0e-6),
                values=(0.0, 0.25, scanned_value, 0.55),
            ),
        ),
        scan_regions=(PulseScanRegion(4.2e-6, 6.8e-6, 1),),
        scan_dac_segments=(
            PulseDacScanSegment(
                "bias",
                4.2e-6,
                6.8e-6,
                value=scanned_value,
                number=2,
            ),
        ),
        repeat_markers=(PulseRepeatMarker(1.8e-6, 9.6e-6, "×4"),),
        time_unit="s",
        total_duration=10.0e-6,
    )


def _definitions() -> tuple[_DemoDefinition, ...]:
    initial_data = _dataset_frame(0)
    repeat_values = np.asarray(initial_data.block.values)[:, :, 0]
    assert not np.allclose(repeat_values[0], repeat_values[1])
    assert not np.allclose(
        np.mean(repeat_values, axis=0), np.median(repeat_values, axis=0)
    )
    initial_image = ImageFrame(_image_frame(0), _image_point_overlay(0))
    initial_pulse = _pulse_frame(0)
    time = AxisRef.point_dimension("time")
    site = AxisRef.data("site")

    def dataset_update(revision: int) -> OwnedSnapshot:
        return _dataset_frame(revision)

    def image_update(revision: int) -> ImageFrame:
        return ImageFrame(_image_frame(revision), _image_point_overlay(revision))

    def pulse_update(revision: int) -> LiveDataRevision:
        return LiveDataRevision(revision, _pulse_frame(revision))

    # Titles name the data source, so role-based label carry keeps them
    # truthful across every semantic kind switch.
    return (
        _DemoDefinition(
            "scan",
            "Scan dataset",
            initial_data,
            CurvePlot(
                time,
                group=site,
                labels=PlotLabels("Scan dataset", "Time", "Signal"),
            ),
            dataset_update,
        ),
        _DemoDefinition(
            "image",
            "Camera image",
            initial_image,
            ImagePlot(
                AxisRef.data("image_x"),
                AxisRef.data("image_y"),
                labels=PlotLabels(
                    "Camera image", "Position X", "Position Y", "Signal"
                ),
            ),
            image_update,
        ),
        _DemoDefinition(
            "pulse",
            "Pulse timeline",
            initial_pulse,
            PulseTimelinePlot(PlotLabels("Pulse timeline", "Time")),
            pulse_update,
        ),
    )


class PyQt5DemoHandle:
    """Own the example window, active plot host, and live producer."""

    def __init__(self, application: object, qt_core: object, qt_widgets: object) -> None:
        self.application = application
        self._QtCore = qt_core
        self._QtWidgets = qt_widgets
        self._definitions = _definitions()
        self._active: _ActivePlot | None = None
        self._closing = False
        self._closed = False
        self._allow_window_close = False
        self._switching = False
        self._pending_semantic: tuple[str, object] | None = None
        self._unsubscribers: tuple[Callable[[], Future[Any]], ...] = ()

        owner = self

        class DemoWindow(qt_widgets.QWidget):
            def closeEvent(window_self, event: object) -> None:
                if owner._allow_window_close:
                    super(DemoWindow, window_self).closeEvent(event)
                    return
                event.ignore()
                owner.close()

        self.window = DemoWindow()
        self.window.setAttribute(qt_core.Qt.WA_DeleteOnClose, True)
        self.window.setWindowTitle("zlc_plot · PyQt5 API example")
        root = qt_widgets.QVBoxLayout(self.window)

        header = qt_widgets.QHBoxLayout()
        root.addLayout(header)
        self.page = qt_widgets.QComboBox(self.window)
        for definition in self._definitions:
            self.page.addItem(definition.label, definition.page_id)
        self.size = qt_widgets.QComboBox(self.window)
        self.size.addItems(DEFAULTS.layout.size_names)
        preferred_size = self.size.findText("2x2")
        self.size.setCurrentIndex(max(0, preferred_size))
        self.refresh = qt_widgets.QComboBox(self.window)
        for interval in DEFAULTS.live.refresh_intervals_ms:
            self.refresh.addItem(f"{interval} ms", interval)
        default_refresh = self.refresh.findData(DEFAULTS.live.default_refresh_interval_ms)
        self.refresh.setCurrentIndex(max(0, default_refresh))
        self.live_toggle = qt_widgets.QCheckBox("Live", self.window)
        self.live_toggle.setChecked(True)
        self.save_button = qt_widgets.QPushButton("Save…", self.window)
        for label, widget in (
            ("Data source", self.page),
            ("Fixed size", self.size),
            ("Refresh", self.refresh),
        ):
            header.addWidget(qt_widgets.QLabel(label, self.window))
            header.addWidget(widget)
        header.addWidget(self.live_toggle)
        header.addWidget(self.save_button)
        header.addStretch(1)
        # The dataset structure line comes verbatim from the library's single
        # structure-description authority so the user always sees what shape
        # the semantic editors are operating on.
        self.schema_info = qt_widgets.QLabel("", self.window)
        self.schema_info.setStyleSheet("color: #555555;")
        root.addWidget(self.schema_info)

        body = qt_widgets.QHBoxLayout()
        root.addLayout(body, 1)
        controls_column = qt_widgets.QVBoxLayout()
        body.addLayout(controls_column)
        self.parameter_scroll = qt_widgets.QScrollArea(self.window)
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setMinimumWidth(285)
        self.parameter_scroll.setHorizontalScrollBarPolicy(qt_core.Qt.ScrollBarAlwaysOff)
        controls_column.addWidget(qt_widgets.QLabel("Display parameters", self.window))
        controls_column.addWidget(self.parameter_scroll, 1)

        view_group = qt_widgets.QGroupBox("Viewport (display units)", self.window)
        view_form = qt_widgets.QGridLayout(view_group)
        self.x_low = self._limit_spin(view_group)
        self.x_high = self._limit_spin(view_group)
        self.y_low = self._limit_spin(view_group)
        self.y_high = self._limit_spin(view_group)
        for row, (label, low, high) in enumerate(
            (("X", self.x_low, self.x_high), ("Y", self.y_low, self.y_high))
        ):
            view_form.addWidget(qt_widgets.QLabel(label, view_group), row, 0)
            view_form.addWidget(low, row, 1)
            view_form.addWidget(high, row, 2)
        self.apply_view = qt_widgets.QPushButton("Apply", view_group)
        self.reset_view = qt_widgets.QPushButton("Reset", view_group)
        view_form.addWidget(self.apply_view, 2, 1)
        view_form.addWidget(self.reset_view, 2, 2)
        controls_column.addWidget(view_group)

        analysis_group = qt_widgets.QGroupBox("Selection and fit", self.window)
        analysis_layout = qt_widgets.QGridLayout(analysis_group)
        self.selector_kind = qt_widgets.QComboBox(analysis_group)
        self.fit_model = qt_widgets.QComboBox(analysis_group)
        self.fit_button = qt_widgets.QPushButton("Fit live", analysis_group)
        self.clear_fit_button = qt_widgets.QPushButton("Clear fit", analysis_group)
        self.read_selection_button = qt_widgets.QPushButton(
            "Read selection data", analysis_group
        )
        self.clear_selection_button = qt_widgets.QPushButton(
            "Clear selection", analysis_group
        )
        analysis_layout.addWidget(
            qt_widgets.QLabel("Selector kind", analysis_group), 0, 0
        )
        analysis_layout.addWidget(self.selector_kind, 0, 1)
        analysis_layout.addWidget(self.fit_model, 1, 0, 1, 2)
        analysis_layout.addWidget(self.fit_button, 2, 0)
        analysis_layout.addWidget(self.clear_fit_button, 2, 1)
        analysis_layout.addWidget(self.read_selection_button, 3, 0)
        analysis_layout.addWidget(self.clear_selection_button, 3, 1)
        controls_column.addWidget(analysis_group)

        self.plot_container = qt_widgets.QWidget(self.window)
        self.plot_layout = qt_widgets.QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.plot_container, 1)

        self.help = qt_widgets.QLabel(
            "Left drag: selector · middle drag: pan · wheel: zoom · "
            "FacetGrid: double-click a cell",
            self.window,
        )
        self.status = qt_widgets.QLabel("Starting…", self.window)
        self.live_status = qt_widgets.QLabel(self.window)
        root.addWidget(self.help)
        root.addWidget(self.status)
        root.addWidget(self.live_status)

        self._producer = qt_core.QTimer(self.window)
        self._producer.timeout.connect(self._publish)
        self._producer.start(DEFAULTS.live.refresh_intervals_ms[0])
        self._status_timer = qt_core.QTimer(self.window)
        self._status_timer.timeout.connect(self._show_live_status)
        self._status_timer.start(DEFAULTS.live.default_refresh_interval_ms)

        # ``activated`` fires on every user pick, including re-selecting the
        # already-current entry.  After a semantic kind switch the combo still
        # points at the authored page the user edited away from; re-picking
        # that same entry must restore the pristine demo, and
        # ``currentIndexChanged`` would swallow exactly that click.
        self.page.activated.connect(self._page_changed)
        self.size.currentTextChanged.connect(self._size_changed)
        self.refresh.currentIndexChanged.connect(self._refresh_changed)
        self.live_toggle.toggled.connect(self._live_toggled)
        self.save_button.clicked.connect(self._save)
        self.apply_view.clicked.connect(self._apply_view_limits)
        self.reset_view.clicked.connect(self._reset_viewport)
        self.fit_button.clicked.connect(self._fit)
        self.clear_fit_button.clicked.connect(self._clear_fit)
        self.read_selection_button.clicked.connect(self._read_selection)
        self.clear_selection_button.clicked.connect(self._clear_selection)

        try:
            self._build_active(self._definitions[0])
        except Exception:
            self._producer.stop()
            self._status_timer.stop()
            self._allow_window_close = True
            self.window.close()
            raise
        self.window.show()
        qt_core.QTimer.singleShot(0, self.window.adjustSize)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def plot_host(self) -> RasterPlotHost:
        if self._active is None:
            raise RuntimeError("plot kind is changing or the window is closed")
        return self._active.host

    @property
    def plot_widget(self) -> Qt5PlotWidget:
        if self._active is None:
            raise RuntimeError("plot kind is changing or the window is closed")
        return self._active.widget

    @property
    def parameter_panel(self) -> Qt5ParameterPanel:
        if self._active is None:
            raise RuntimeError("plot kind is changing or the window is closed")
        return self._active.panel

    def _limit_spin(self, parent: object) -> object:
        spin = self._QtWidgets.QDoubleSpinBox(parent)
        spin.setDecimals(9)
        spin.setRange(-1.0e100, 1.0e100)
        spin.setKeyboardTracking(False)
        return spin

    def _current_definition(self) -> _DemoDefinition:
        selected = self.page.currentData()
        return next(item for item in self._definitions if item.page_id == selected)

    def _build_active(self, definition: _DemoDefinition) -> None:
        if self._closing:
            return
        preset = self.size.currentText()
        timeout = DEFAULTS.runtime.shutdown_timeout_ms / 1000.0
        initial = definition.initial
        schema = (
            initial.snapshot.block.schema
            if isinstance(initial, ImageFrame)
            else initial.block.schema
            if isinstance(initial, OwnedSnapshot)
            else None
        )
        self.schema_info.setText(
            f"{definition.label}: "
            + (schema_summary(schema) if schema is not None else "pulse timeline")
        )
        with ExitStack() as cleanup:
            host = RasterPlotHost.from_plot(
                definition.initial,
                definition.spec,
                size=preset,
            )
            cleanup.callback(host.close, timeout=timeout)
            widget = Qt5PlotWidget(host, self.plot_container)
            cleanup.callback(widget.close_adapter)
            description = host.describe_display().result(timeout=timeout).value
            panel = Qt5ParameterPanel(description, self.parameter_scroll)
            panel.parameterEdited.connect(self._parameter_edited)
            panel.semanticEdited.connect(self._semantic_edited)
            self.parameter_scroll.setWidget(panel)
            cleanup.callback(panel.deleteLater)
            cleanup.callback(self.parameter_scroll.takeWidget)
            self.plot_layout.addWidget(widget, 0, self._QtCore.Qt.AlignCenter)
            cleanup.callback(widget.deleteLater)
            cleanup.callback(self.plot_layout.removeWidget, widget)
            live = host.live_controller(
                definition.live_contract,
                refresh_interval_ms=int(self.refresh.currentData()),
            )
            cleanup.callback(live.close, timeout=timeout)
            if self.live_toggle.isChecked():
                live.start()
                self._producer.start(DEFAULTS.live.refresh_intervals_ms[0])
            active = _ActivePlot(
                definition,
                host,
                widget,
                panel,
                live,
                0,
                description.semantics,
            )
            widget.errorOccurred.connect(self._show_error)
            widget.surfaceChanged.connect(
                lambda _identity: self._QtCore.QTimer.singleShot(
                    0, self.window.adjustSize
                )
            )
            unsubscribers = (
                host.subscribe_display(
                    lambda state, selected=active: self._display_changed(selected, state)
                ).result(timeout=timeout).value,
                host.subscribe_fit(
                    lambda event, selected=active: self._fit_accepted(selected, event)
                ).result(timeout=timeout).value,
                host.subscribe_selection(
                    lambda event, selected=active: self._selection_changed(selected, event)
                ).result(timeout=timeout).value,
            )
            self._set_limit_values(description)
            self._populate_fit_models(active)
            self._active = active
            self._unsubscribers = unsubscribers
            cleanup.pop_all()
        self.selector_kind.clear()
        self.read_selection_button.setEnabled(False)
        self.clear_selection_button.setEnabled(False)
        self.status.setText(f"{definition.label} ready")
        self._switching = False
        self.page.setEnabled(True)
        self.live_toggle.setEnabled(True)

    def _populate_fit_models(self, active: _ActivePlot) -> None:
        self.fit_model.clear()
        timeout = DEFAULTS.runtime.shutdown_timeout_ms / 1000.0
        models = active.host.fit_models().result(timeout=timeout).value
        for model in models:
            self.fit_model.addItem(model.display_name, model.model_id)
        enabled = self.fit_model.count() > 0
        self.fit_model.setEnabled(enabled)
        self.fit_button.setEnabled(enabled)
        self.clear_fit_button.setEnabled(enabled)

    def _set_limit_values(self, description: object) -> None:
        for spin, value in (
            (self.x_low, description.limits.x.low),
            (self.x_high, description.limits.x.high),
            (self.y_low, description.limits.y.low),
            (self.y_high, description.limits.y.high),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def _dispatch(self, active: _ActivePlot, callback: Callable[[], None]) -> None:
        if self._active is active and not self._closed:
            def guarded() -> None:
                try:
                    callback()
                except Exception as exc:
                    # dispatch futures are discarded; without this hook a
                    # callback bug disappears silently and the window looks
                    # dead with no diagnostic at all.
                    self._show_error(f"internal UI callback error: {exc!r}")
                    raise

            active.widget.dispatch(guarded)

    def _track(
        self,
        active: _ActivePlot,
        future: Future[Any],
        action: str,
        accepted: Callable[[object], None] | None = None,
        *,
        restore_display_on_error: bool = False,
        rejected: Callable[[Exception], None] | None = None,
    ) -> None:
        def finished(done: Future[Any]) -> None:
            try:
                operation = done.result()
            except Exception as error:
                # Python unbinds the ``except ... as error`` name when the
                # block exits, and this closure runs later on the Qt thread.
                # Referencing ``error`` directly raised NameError inside a
                # discarded dispatch future and deadlocked every guarded edit.
                err = error

                def report_error() -> None:
                    self._show_error(f"{action}: {err}")
                    active.panel.set_error(f"{action}: {err}")
                    if rejected is not None:
                        rejected(err)

                self._dispatch(active, report_error)
                if restore_display_on_error and self._active is active:
                    self._track(
                        active,
                        active.host.describe_display(),
                        "restore display controls",
                        lambda description: active.panel.set_description(description),
                    )
                return

            def report_success() -> None:
                active.panel.set_error(None)
                if accepted is not None:
                    accepted(operation.value)

            self._dispatch(active, report_success)

        future.add_done_callback(finished)

    def _definition_after_semantic(
        self,
        active: _ActivePlot,
        candidate: PlotSpec,
    ) -> _DemoDefinition:
        """Keep the acquisition route valid when an ImageFrame changes kind."""

        definition = replace(active.definition, spec=candidate)
        if isinstance(definition.initial, ImageFrame) and not isinstance(
            candidate, ImagePlot
        ):
            image_update = definition.make_update

            def snapshot_update(revision: int) -> OwnedSnapshot:
                payload = image_update(revision)
                if not isinstance(payload, ImageFrame):
                    raise TypeError("image demo update must return ImageFrame")
                return payload.snapshot

            definition = replace(
                definition,
                initial=definition.initial.snapshot,
                make_update=snapshot_update,
            )
        return definition

    def _rebind_live_contract(
        self,
        active: _ActivePlot,
        definition: _DemoDefinition,
    ) -> None:
        old_type = active.live.payload_type
        contract = definition.live_contract
        if isinstance(contract, LiveDataRevision):
            new_type = type(contract.payload)
        else:
            new_type = type(contract)
        if old_type is new_type:
            active.definition = definition
            return
        was_running = active.live.running
        active.live.close(timeout=DEFAULTS.runtime.shutdown_timeout_ms / 1000.0)
        active.definition = definition
        active.live = active.host.live_controller(
            definition.live_contract,
            refresh_interval_ms=int(self.refresh.currentData()),
        )
        if was_running and self.live_toggle.isChecked():
            active.live.start()

    def _parameter_edited(self, name: str, value: object) -> None:
        active = self._active
        if active is None:
            return
        future = active.host.set_parameter(name, value)
        self._track(
            active,
            future,
            f"set {name}",
            restore_display_on_error=True,
        )

    def _semantic_edited(self, name: str, value: object) -> None:
        """Route every semantic editor through one host ``replace_spec`` call.

        Edits arriving while a rebuild is in flight are never dropped: the
        latest one is parked and submitted the moment the current rebuild
        settles.  Silently discarding busy-window clicks left the combo
        showing a value the plot never took — the "nothing happened" feel.
        """

        active = self._active
        if active is None:
            return
        if self._switching:
            self._pending_semantic = (name, value)
            self.status.setText(f"rebuilding… queued {name} change")
            return
        self._switching = True
        current = active.definition.spec
        if name == "kind" and (not isinstance(value, PlotKind) or value is current.kind):
            self._finish_semantic()
            return
        initial = active.definition.initial
        schema = (
            initial.snapshot.block.schema
            if isinstance(initial, ImageFrame)
            else initial.block.schema
            if isinstance(initial, OwnedSnapshot)
            else None
        )
        try:
            # Candidate composition (including role-bound label carry-over)
            # is a library concern; the embedder only forwards (name, value)
            # and submits the result.
            candidate = updated_spec(schema, current, name, value)
        except Exception as error:
            active.panel.set_error(str(error))
            self._show_error(f"semantic {name}: {error}")
            self._finish_semantic()
            return
        self.status.setText(f"rebuilding {name}…")

        def accepted(description: object) -> None:
            definition = self._definition_after_semantic(active, candidate)
            self._rebind_live_contract(active, definition)
            active.semantics = description.semantics
            active.panel.set_description(description)
            active.panel.set_error(None)
            self._populate_fit_models(active)
            self.status.setText(f"{name} applied")
            self._finish_semantic()

        def rejected(_error: Exception) -> None:
            active.panel.set_error(str(_error))
            self._finish_semantic()

        self._track(
            active,
            active.host.replace_spec(candidate),
            f"replace semantic {name}",
            accepted,
            restore_display_on_error=True,
            rejected=rejected,
        )

    def _finish_semantic(self) -> None:
        """Release the rebuild latch and immediately run the parked edit."""

        self._switching = False
        pending = self._pending_semantic
        self._pending_semantic = None
        if pending is not None and not self._closing:
            self._semantic_edited(*pending)

    def _display_changed(self, active: _ActivePlot, state: object) -> None:
        self._dispatch(active, lambda: active.panel.set_values(state.values))

    def _selection_changed(self, active: _ActivePlot, event: object) -> None:
        def apply() -> None:
            kind = event.selector.kind
            index = self.selector_kind.findData(kind)
            if event.change is SelectionChange.REMOVED:
                if index >= 0:
                    self.selector_kind.removeItem(index)
                self.status.setText("selection cleared")
            else:
                if index < 0:
                    self.selector_kind.addItem(kind.value, kind)
                    index = self.selector_kind.count() - 1
                self.selector_kind.setCurrentIndex(index)
                self.status.setText(
                    f"selector={kind.value}; revision={event.data_revision}"
                )
            available = self.selector_kind.count() > 0
            self.read_selection_button.setEnabled(available)
            self.clear_selection_button.setEnabled(available)

        self._dispatch(active, apply)

    def _fit_accepted(self, active: _ActivePlot, event: FitEvent) -> None:
        parameters = ", ".join(
            f"{item.name}={item.value:.4g}"
            + (
                " ± n/a"
                if item.standard_error is None
                else f" ± {item.standard_error:.2g}"
            )
            + (f" {item.unit}" if item.unit else "")
            for item in event.display_parameters
        )
        self._dispatch(
            active,
            lambda: self.status.setText(
                f"fit {event.selection.scope.value}: {parameters}"
            ),
        )

    def _page_changed(self, _index: int) -> None:
        if self._closing or self._switching:
            return
        definition = self._current_definition()
        active = self._active
        # Compare identity, not kind: after a semantic edit the active
        # definition is a modified copy, and re-picking the authored demo
        # page must rebuild the pristine demo instead of dead-clicking.
        if active is not None and active.definition is definition:
            return
        self._switching = True
        self.page.setEnabled(False)
        self.live_toggle.setEnabled(False)
        self.status.setText(f"Opening {definition.label}…")

        def rebuild() -> None:
            try:
                self._build_active(definition)
            except Exception as error:
                self._switching = False
                self.page.setEnabled(True)
                self.live_toggle.setEnabled(True)
                self._show_error(error)

        self._dispose_active(rebuild)

    def _size_changed(self, preset: str) -> None:
        active = self._active
        if active is not None and not self._switching:
            self._track(active, active.host.set_size(str(preset)), "set size")

    def _refresh_changed(self, _index: int) -> None:
        active = self._active
        if active is None:
            return
        try:
            active.live.set_refresh_interval(int(self.refresh.currentData()))
        except Exception as error:
            self._show_error(error)

    def _live_toggled(self, enabled: bool) -> None:
        active = self._active
        if active is None or self._switching or self._closing:
            return
        if enabled:
            try:
                active.live.start()
            except Exception as error:
                self._show_error(error)
            else:
                self._producer.start(DEFAULTS.live.refresh_intervals_ms[0])
            return
        self.live_toggle.setEnabled(False)
        self._producer.stop()
        active.live.stop(wait=False)

        def poll() -> None:
            if self._active is not active or self._closing:
                return
            if active.live.worker_alive:
                self._QtCore.QTimer.singleShot(10, poll)
                return
            self.live_toggle.setEnabled(True)
            self.status.setText("live paused")

        self._QtCore.QTimer.singleShot(0, poll)

    def _publish(self) -> None:
        active = self._active
        if (
            active is None
            or self._switching
            or not self.live_toggle.isChecked()
            or not active.live.running
        ):
            return
        active.revision += 1
        try:
            active.live.publish(active.definition.make_update(active.revision))
        except Exception as error:
            self._show_error(error)

    def _show_live_status(self) -> None:
        active = self._active
        if active is None:
            self.live_status.clear()
            return
        metrics = active.live.metrics()
        self.live_status.setText(
            f"published={metrics.last_published_revision}  "
            f"displayed={metrics.last_presented_revision}  "
            f"coalesced={metrics.coalesced_updates}"
        )

    def _apply_view_limits(self, _checked: bool = False) -> None:
        active = self._active
        if active is None:
            return
        future = active.host.set_view_limits(
            x=(self.x_low.value(), self.x_high.value()),
            y=(self.y_low.value(), self.y_high.value()),
        )
        self._track(active, future, "set viewport")

    def _reset_viewport(self, _checked: bool = False) -> None:
        active = self._active
        if active is None:
            return

        def refresh(_value: object) -> None:
            description_future = active.host.describe_display()
            self._track(
                active,
                description_future,
                "read limits",
                self._set_limit_values,
            )

        self._track(active, active.host.reset_viewport(), "reset viewport", refresh)

    def _fit(self, _checked: bool = False) -> None:
        active = self._active
        if active is None:
            return
        model_id = self.fit_model.currentData()
        if not isinstance(model_id, str):
            return
        self.status.setText(f"fitting {self.fit_model.currentText()}…")
        self._track(active, active.host.fit(model_id, live=True), "fit")

    def _clear_fit(self, _checked: bool = False) -> None:
        active = self._active
        if active is not None:
            self._track(active, active.host.clear_fit(), "clear fit")

    def _read_selection(self, _checked: bool = False) -> None:
        active = self._active
        selector_kind = self.selector_kind.currentData()
        if active is None or not isinstance(selector_kind, SelectorKind):
            self.status.setText("drag a selector first")
            return

        def show(selected: object) -> None:
            if isinstance(selected, SelectorData):
                count = int(selected.canonical_values.size)
            elif isinstance(selected, PulseTimelineSelectionData):
                count = sum(
                    len(records)
                    for records in (
                        selected.blocks,
                        selected.analog_traces,
                        selected.scan_regions,
                        selected.scan_dac_segments,
                        selected.repeat_markers,
                    )
                )
            else:
                raise TypeError("unsupported selector result")
            self.status.setText(
                f"selection materialized on call: {count} item(s), "
                f"revision={selected.data_revision}"
            )

        self._track(
            active,
            active.host.selector_data(selector_kind),
            "read selection",
            show,
        )

    def _clear_selection(self, _checked: bool = False) -> None:
        active = self._active
        selector_kind = self.selector_kind.currentData()
        if active is not None and isinstance(selector_kind, SelectorKind):
            self._track(
                active,
                active.host.remove_selector(selector_kind),
                "clear selection",
            )

    def _save(self, _checked: bool = False) -> None:
        active = self._active
        if active is None:
            return
        suggested = str(Path.cwd() / f"{active.definition.spec.kind.value}.png")
        path, _filter = self._QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save plot",
            suggested,
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        self._track(
            active,
            active.host.save(Path(path)),
            "save",
            lambda _value: self.status.setText(f"saved {path}"),
        )

    def _show_error(self, error: object) -> None:
        if not self._closed:
            self.status.setText(str(error))

    def _dispose_active(self, finished: Callable[[], None]) -> None:
        active, self._active = self._active, None
        unsubscribers, self._unsubscribers = self._unsubscribers, ()
        self._producer.stop()
        if active is None:
            finished()
            return
        active.live.close(timeout=0)

        def wait_live() -> None:
            if not active.live.close(timeout=0):
                self._QtCore.QTimer.singleShot(10, wait_live)
                return
            release_futures = tuple(unsubscribe() for unsubscribe in unsubscribers)
            wait_unsubscribe(release_futures)

        def wait_unsubscribe(futures: tuple[Future[Any], ...]) -> None:
            if any(not future.done() for future in futures):
                self._QtCore.QTimer.singleShot(10, lambda: wait_unsubscribe(futures))
                return
            active.widget.close_adapter()
            self.plot_layout.removeWidget(active.widget)
            active.widget.deleteLater()
            old_panel = self.parameter_scroll.takeWidget()
            if old_panel is not None:
                old_panel.deleteLater()
            wait_host()

        def wait_host() -> None:
            if not active.host.close(timeout=0):
                self._QtCore.QTimer.singleShot(10, wait_host)
                return
            finished()

        self._QtCore.QTimer.singleShot(0, wait_live)

    def close(self, *, wait: bool = False) -> bool:
        if self._closed:
            return True
        if wait:
            return self._close_synchronously()
        if self._closing:
            return False
        self._closing = True
        self.page.setEnabled(False)
        self.live_toggle.setEnabled(False)
        self._producer.stop()
        self._status_timer.stop()

        def finish() -> None:
            self._closed = True
            self._allow_window_close = True
            self.window.close()

        self._dispose_active(finish)
        return False

    def _close_synchronously(self) -> bool:
        self._closing = True
        self._producer.stop()
        self._status_timer.stop()
        active = self._active
        timeout = DEFAULTS.runtime.shutdown_timeout_ms / 1000.0
        if active is not None:
            live_closed = False
            host_closed = False
            try:
                live_closed = active.live.close(timeout=timeout)
                for unsubscribe in self._unsubscribers:
                    unsubscribe().result(timeout=timeout)
            finally:
                self._unsubscribers = ()
                try:
                    active.widget.close_adapter()
                finally:
                    host_closed = active.host.close(timeout=timeout)
            if not live_closed or not host_closed:
                return False
            self._active = None
        self._closed = True
        self._allow_window_close = True
        self.window.close()
        return True


def create_window() -> PyQt5DemoHandle:
    """Show the reusable PyQt5 example and return without starting an event loop."""

    try:
        from PyQt5 import QtCore, QtWidgets
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailableError(
            "pyqt5_embed requires `pip install -e .[qt]`"
        ) from error
    application = ensure_qt5_application(sys.argv)
    return PyQt5DemoHandle(application, QtCore, QtWidgets)


def run() -> int:
    handle = create_window()
    try:
        return int(handle.application.exec_())
    finally:
        if not handle.closed and not handle.close(wait=True):
            raise RuntimeError("PyQt5 example workers did not stop during shutdown")


def _main() -> int:
    try:
        return run()
    except BackendUnavailableError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
