"""Pure Scan page: projections in, immediate text intents out."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT, GREEN, GREY, ORANGE, RED, YELLOW, FluentButton, FluentCodeEdit,
    FluentDoubleSpinBox, FluentFrame, FluentGroupBox, FluentLabel, FluentLineEdit,
    signals_blocked,
)

from ._layout import px, row_height
from .models import BindingRecord, ScanPageRecord  # noqa: F401


class PulseScanView(QtWidgets.QWidget):
    repeats_committed = QtCore.pyqtSignal(int)
    hold_requested = QtCore.pyqtSignal()
    step_requested = QtCore.pyqtSignal(int)
    load_program_requested = QtCore.pyqtSignal()
    template_requested = QtCore.pyqtSignal(str)
    source_edited = QtCore.pyqtSignal(str)
    run_requested = QtCore.pyqtSignal()
    save_array_requested = QtCore.pyqtSignal()
    progress_refresh_requested = QtCore.pyqtSignal()
    #: ``(old id, new id)`` -- the NAME a plan and a saved value set use.
    binding_renamed = QtCore.pyqtSignal(str, str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(self)
        margin = px(8, minimum=5)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(px(8, minimum=5))

        info = FluentFrame()
        info.setMinimumHeight(px(64, minimum=52))
        info_layout = QtWidgets.QVBoxLayout(info)
        info_layout.setContentsMargins(px(12), px(8), px(12), px(8))
        self.scan_slots_label = FluentLabel("")
        self.scan_slots_label.setWordWrap(True)
        info_layout.addWidget(self.scan_slots_label)
        # The ids themselves, editable.  A binding gets its name minted from
        # the period it sits in -- ``dac_load_da_bias_x`` -- which is fine
        # until a saved set of values has to name the same slot in another
        # pulse.  The field does not move; only what everything else calls it.
        self.bindings_grid = QtWidgets.QGridLayout()
        self.bindings_grid.setContentsMargins(0, px(4, minimum=2), 0, 0)
        info_layout.addLayout(self.bindings_grid)
        self._binding_edits: dict[str, FluentLineEdit] = {}

        run_row = QtWidgets.QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(px(6, minimum=4))
        run_row.addWidget(FluentLabel("Scan repeats (0 = ∞)"))
        # Qt's integer spin stops at signed 31-bit.  The zero-decimal Fluent
        # double spin represents every uint32 integer exactly, so the visible
        # control and the hardware count share the full domain.
        self.scan_repeats_spin = FluentDoubleSpinBox()
        self.scan_repeats_spin._step_btn.hide()
        self.scan_repeats_spin.setDecimals(0)
        self.scan_repeats_spin.setSingleStep(1.0)
        self.scan_repeats_spin.setMaximum(float((1 << 32) - 1))
        self._minimum_repeats = 0
        self._committed_repeats = 0
        self.set_repeats_range(0, 0)
        self.scan_repeats_spin.setFixedHeight(row_height())
        self.scan_repeats_spin.editingFinished.connect(self._commit_repeats)
        run_row.addWidget(self.scan_repeats_spin)

        self.scan_hold_button = FluentButton("Hold current point", color=RED)
        self.scan_hold_button.setFixedHeight(row_height())
        self.scan_hold_button.clicked.connect(self.hold_requested)
        run_row.addWidget(self.scan_hold_button)
        self.scan_step_back_button = FluentButton("◀ step", color=ORANGE)
        self.scan_step_forward_button = FluentButton("step ▶", color=ORANGE)
        for button, delta in ((self.scan_step_back_button, -1), (self.scan_step_forward_button, 1)):
            button.setFixedHeight(row_height())
            button.clicked.connect(lambda _checked=False, value=delta: self.step_requested.emit(value))
            run_row.addWidget(button)
        self.scan_progress_label = FluentLabel("")
        run_row.addWidget(self.scan_progress_label, 1)
        info_layout.addLayout(run_row)
        layout.addWidget(info)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(px(8, minimum=5))
        editor_box = FluentGroupBox("Generate the scan table (Python)")
        editor_layout = QtWidgets.QVBoxLayout(editor_box)
        editor_layout.setContentsMargins(px(8), px(28, minimum=24), px(8), px(8))
        editor_layout.setSpacing(px(6, minimum=4))
        self.scan_code = FluentCodeEdit()

        template_buttons = QtWidgets.QHBoxLayout()
        template_buttons.setSpacing(px(6, minimum=4))
        self.scan_load_program_button = FluentButton("Load Program", color=ACCENT)
        self.scan_column_template_button = FluentButton("Template: column_stack", color=GREY)
        self.scan_grid_template_button = FluentButton("Template: grid", color=GREY)
        self.scan_load_program_button.clicked.connect(self.load_program_requested)
        self.scan_column_template_button.clicked.connect(lambda: self.template_requested.emit("column_stack"))
        self.scan_grid_template_button.clicked.connect(lambda: self.template_requested.emit("grid"))
        for button in (self.scan_load_program_button, self.scan_column_template_button, self.scan_grid_template_button):
            button.setFixedHeight(row_height())
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            template_buttons.addWidget(button, 1)
        editor_layout.addLayout(template_buttons)
        editor_layout.addWidget(self.scan_code, 1)

        code_buttons = QtWidgets.QHBoxLayout()
        code_buttons.setSpacing(px(6, minimum=4))
        self.scan_run_button = FluentButton("Run", color=GREEN)
        self.scan_save_array_button = FluentButton("Save Array", color=YELLOW)
        self.scan_run_button.setFixedHeight(row_height())
        self.scan_save_array_button.setFixedHeight(row_height())
        self.scan_run_button.clicked.connect(lambda: self.run_requested.emit())
        self.scan_save_array_button.clicked.connect(self.save_array_requested)
        self.scan_code.textChanged.connect(self._on_code_changed)
        for button in (self.scan_run_button, self.scan_save_array_button):
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            code_buttons.addWidget(button, 1)
        editor_layout.addLayout(code_buttons)
        body.addWidget(editor_box, 3)

        preview_box = FluentGroupBox("Scan table")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(px(8), px(28, minimum=24), px(8), px(8))
        self.scan_table_view = FluentCodeEdit(read_only=True)
        preview_layout.addWidget(self.scan_table_view, 1)
        preview_layout.addWidget(QtWidgets.QWidget(), 0)
        body.addWidget(preview_box, 2)
        layout.addLayout(body, 1)

        self._scan_progress_timer = QtCore.QTimer(self)
        self._scan_progress_timer.setInterval(200)
        self._scan_progress_timer.timeout.connect(self._request_visible_progress)

    def _on_code_changed(self) -> None:
        self.source_edited.emit(self.scan_code.toPlainText())

    def _request_visible_progress(self) -> None:
        if self.isVisible():
            self.progress_refresh_requested.emit()

    def set_progress_polling(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self._scan_progress_timer.isActive():
            self._scan_progress_timer.start()
        elif not enabled and self._scan_progress_timer.isActive():
            self._scan_progress_timer.stop()

    def set_repeats_range(self, minimum: int, default: int) -> None:
        minimum = int(minimum)
        default = int(default)
        if minimum < 0 or default < minimum:
            raise ValueError("repeat range is invalid")
        self._minimum_repeats = minimum
        self.scan_repeats_spin.setMinimum(minimum)
        self.set_repeats(default)

    def set_repeats(self, repeats: int) -> None:
        value = max(self._minimum_repeats, int(repeats))
        with signals_blocked(self.scan_repeats_spin):
            self.scan_repeats_spin.setValue(float(value))
        self._committed_repeats = value

    def _commit_repeats(self) -> None:
        value = int(self.scan_repeats_spin.value())
        if value == self._committed_repeats:
            return
        self._committed_repeats = value
        self.repeats_committed.emit(value)

    def set_page(self, record: ScanPageRecord) -> None:
        """Show the whole page from its one presenter-owned state."""

        if not isinstance(record, ScanPageRecord):
            raise TypeError("record must be ScanPageRecord")
        self.set_slots_text(record.slots_text)
        self.set_bindings(record.bindings)
        self.set_scan_table_text(record.table_text)
        self.set_repeats(record.repeats)
        self.set_progress_text(record.progress_text)
        self.set_workspace_busy(record.busy)
        self.set_progress_polling(record.progress_polling)
        if record.source_text != self.scan_code.toPlainText():
            with signals_blocked(self.scan_code):
                self.scan_code.setPlainText(str(record.source_text))
        self.set_run_dirty(record.source_dirty)

    def set_slots_text(self, text: str) -> None:
        self.scan_slots_label.setText(str(text))

    def set_bindings(self, bindings) -> None:
        while self.bindings_grid.count():
            item = self.bindings_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._binding_edits = {}
        for row, record in enumerate(tuple(bindings)):
            edit = FluentLineEdit(str(record.binding_id))
            edit.setToolTip(
                "The name a scan plan, a saved value set and a run record use"
            )
            edit.editingFinished.connect(
                lambda edit=edit, was=str(record.binding_id): (
                    self.binding_renamed.emit(was, edit.text().strip())
                    if edit.text().strip() and edit.text().strip() != was
                    else None
                )
            )
            self.bindings_grid.addWidget(FluentLabel(str(record.label)), row, 0)
            self.bindings_grid.addWidget(edit, row, 1)
            self.bindings_grid.addWidget(FluentLabel(str(record.kind)), row, 2)
            self._binding_edits[str(record.binding_id)] = edit

    def set_progress_text(self, text: str) -> None:
        self.scan_progress_label.setText(str(text))

    def set_scan_table_text(self, text: str) -> None:
        self.scan_table_view.setPlainText(str(text))

    def set_run_dirty(self, dirty: bool) -> None:
        self.scan_run_button.set_dirty(bool(dirty))

    def set_workspace_busy(self, busy: bool) -> None:
        enabled = not bool(busy)
        for button in (self.scan_load_program_button, self.scan_column_template_button,
                       self.scan_grid_template_button, self.scan_run_button,
                       self.scan_save_array_button):
            button.setEnabled(enabled)


__all__ = ["PulseScanView"]
