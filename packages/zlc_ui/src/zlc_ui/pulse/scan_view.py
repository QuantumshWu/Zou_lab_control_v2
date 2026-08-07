"""Pure Scan page.

The presenter supplies already formatted text and owns execution, file IO and
scan-table validation.  The three draft methods intentionally remain
separate: they are the small protocol that prevents controller updates from
clobbering text a person is currently editing.
"""

from __future__ import annotations

import sys

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT, GREEN, GREY, ORANGE, RED, YELLOW, FluentButton, FluentCodeEdit,
    FluentDoubleSpinBox, FluentFrame, FluentGroupBox, FluentLabel, signals_blocked,
)

from ._layout import px, row_height
from .models import ScanPageRecord


class PulseScanView(QtWidgets.QWidget):
    repeats_committed = QtCore.pyqtSignal(int)
    hold_requested = QtCore.pyqtSignal()
    step_requested = QtCore.pyqtSignal(int)
    load_program_requested = QtCore.pyqtSignal()
    template_requested = QtCore.pyqtSignal(str)
    run_requested = QtCore.pyqtSignal(str)
    save_array_requested = QtCore.pyqtSignal()
    progress_refresh_requested = QtCore.pyqtSignal()

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

        run_row = QtWidgets.QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(px(6, minimum=4))
        run_row.addWidget(FluentLabel("Scan repeats (0 = ∞)"))
        self.scan_repeats_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
        self.scan_repeats_spin.setDecimals(0)
        self.scan_repeats_spin.setMaximum(sys.float_info.max)
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
        self._code_dirty = False
        self._source_revision = -1

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
        self.scan_run_button.clicked.connect(lambda: self.run_requested.emit(self.scan_code.toPlainText()))
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
        self.set_run_dirty(True)

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
            self.scan_repeats_spin.setValue(value)
        self._committed_repeats = value

    def _commit_repeats(self) -> None:
        value = int(self.scan_repeats_spin.value())
        if value == self._committed_repeats:
            return
        self._committed_repeats = value
        self.repeats_committed.emit(value)

    def set_page(self, record: ScanPageRecord) -> None:
        """Show the whole page, including the program it is showing.

        The source text was the one field this dropped, so the editor opened
        blank -- the presenter generates a starter template for exactly this
        moment -- and then went stale while the revision it carries advanced.

        A half-typed program is never overwritten: an operator mid-edit owns
        the box, and the page catching up must not take what they were writing.
        """

        if not isinstance(record, ScanPageRecord):
            raise TypeError("record must be ScanPageRecord")
        self.set_slots_text(record.slots_text)
        self.set_scan_table_text(record.table_text)
        self.set_repeats(record.repeats)
        self.set_progress_text(record.progress_text)
        self.set_workspace_busy(record.busy)
        self.set_progress_polling(record.progress_polling)
        if not self._code_dirty and record.source_text != self.scan_code.toPlainText():
            with signals_blocked(self.scan_code):
                self.scan_code.setPlainText(str(record.source_text))
        self.set_run_dirty(record.source_dirty)
        self._source_revision = int(record.source_revision)

    def set_slots_text(self, text: str) -> None:
        self.scan_slots_label.setText(str(text))

    def set_progress_text(self, text: str) -> None:
        self.scan_progress_label.setText(str(text))

    @property
    def code_dirty(self) -> bool:
        return self._code_dirty

    @property
    def source_revision(self) -> int:
        return self._source_revision

    def set_scan_code(self, source: str, *, dirty: bool = False, source_revision: int) -> None:
        with signals_blocked(self.scan_code):
            self.scan_code.setPlainText(str(source))
        self._source_revision = int(source_revision)
        self.set_run_dirty(dirty)

    def replace_scan_draft(self, source: str) -> None:
        with signals_blocked(self.scan_code):
            self.scan_code.setPlainText(str(source))
        self.set_run_dirty(True)

    def acknowledge_scan_draft(self, *, dirty: bool, source_revision: int) -> None:
        self._source_revision = int(source_revision)
        self.set_run_dirty(dirty)

    def set_scan_table_text(self, text: str) -> None:
        self.scan_table_view.setPlainText(str(text))

    def set_run_dirty(self, dirty: bool) -> None:
        self._code_dirty = bool(dirty)
        self.scan_run_button.set_dirty(self._code_dirty)

    def set_workspace_busy(self, busy: bool) -> None:
        enabled = not bool(busy)
        for button in (self.scan_load_program_button, self.scan_column_template_button,
                       self.scan_grid_template_button, self.scan_run_button,
                       self.scan_save_array_button):
            button.setEnabled(enabled)


__all__ = ["PulseScanView"]
