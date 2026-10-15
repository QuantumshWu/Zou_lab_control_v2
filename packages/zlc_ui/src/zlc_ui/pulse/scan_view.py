"""Pure Scan page: projections in, immediate text intents out."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    FONT, MUTED_LABEL_STYLE, ElidedLabel, FluentButton, FluentCodeEdit, FluentGroupBox, FluentLabel,
    FluentSectionLabel, centred_line_top, fluent_count_box, fluent_font_size, retire_widget, signals_blocked,
    set_status_ink,
)
from zlc_ui.form import being_edited

from ._layout import count_box_width, quiet, row_height
from zlc_ui.fluent import card_pad, row_gap, section_gap
from .models import BindingRecord, ScanPageRecord
from .scan_line_edit import _SOURCES


class PulseScanView(QtWidgets.QWidget):
    repeats_committed = QtCore.pyqtSignal(int)
    hold_requested = QtCore.pyqtSignal()
    step_requested = QtCore.pyqtSignal(int)
    load_program_requested = QtCore.pyqtSignal()
    template_requested = QtCore.pyqtSignal(str)
    source_edited = QtCore.pyqtSignal(str)
    run_requested = QtCore.pyqtSignal()
    save_array_requested = QtCore.pyqtSignal()
    load_array_requested = QtCore.pyqtSignal()
    clear_array_requested = QtCore.pyqtSignal()
    progress_refresh_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        # The page's edges and gaps are every tab's; inside each card its
        # text, title and controls start on the card pad, the title's x.
        layout = QtWidgets.QVBoxLayout(self)
        page = section_gap()
        layout.setContentsMargins(page, page, page, page)
        layout.setSpacing(section_gap())
        pad = card_pad()

        # Titled like every card on the page: the list's intro is the
        # card's first row, under the title, not standing in for it.
        info = FluentGroupBox("Bound fields")
        info_layout = QtWidgets.QVBoxLayout(info)
        info_layout.setContentsMargins(pad, pad, pad, pad)
        info_layout.setSpacing(row_gap())
        # The list's intro is its first line: as tall as the lines under it,
        # its text on the line they centre theirs on, so the intro and the
        # list share one pitch.  A long one (a Config error) wraps below.
        self.scan_slots_label = FluentLabel("")
        self.scan_slots_label.setWordWrap(True)
        self.scan_slots_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.scan_slots_label.ensurePolished()
        self.scan_slots_label.setContentsMargins(
            0, centred_line_top(0, row_height(), self.scan_slots_label.font()), 0, 0)
        self.scan_slots_label.setMinimumHeight(row_height())
        # The bound fields, under a section title per group when there are
        # groups; rebuilt only when the list changes.  One line per field,
        # each as tall as a section title, so the list has one pitch.
        self.bindings_box = QtWidgets.QWidget()
        self.bindings_box.setStyleSheet("background: transparent;")
        self._bindings_layout = QtWidgets.QVBoxLayout(self.bindings_box)
        self._bindings_layout.setContentsMargins(0, 0, 0, 0)
        self._bindings_layout.setSpacing(0)
        self._bindings_shown: tuple[BindingRecord, ...] | None = None
        listing = QtWidgets.QVBoxLayout()
        listing.setContentsMargins(0, 0, 0, 0)
        listing.setSpacing(0)
        listing.addWidget(self.scan_slots_label)
        listing.addWidget(self.bindings_box)
        info_layout.addLayout(listing)
        layout.addWidget(info)

        # How the scan plays -- how many times the table is walked, holding
        # a point, stepping -- is its own card: under "Bound fields" an
        # operator looking for Hold found it among a list of fields.
        playback = FluentGroupBox("Playback")
        run_row = QtWidgets.QHBoxLayout(playback)
        run_row.setContentsMargins(pad, pad, pad, pad)
        run_row.setSpacing(row_gap())
        run_row.addWidget(FluentLabel("Scan repeats"))
        self.scan_repeats_spin = fluent_count_box()
        # As wide as every repeat box, wider only while it shows or takes a
        # longer count.
        self._size_repeats()
        self.scan_repeats_spin.lineEdit().textChanged.connect(self._size_repeats)
        self._minimum_repeats = 0
        self._committed_repeats = 0
        self.set_repeats_range(0, 0)
        self.scan_repeats_spin.setFixedHeight(row_height())
        self.scan_repeats_spin.editingFinished.connect(self._commit_repeats)
        run_row.addWidget(self.scan_repeats_spin)

        self.scan_hold_button = FluentButton("Hold current point")
        self.scan_hold_button.setFixedHeight(row_height())
        self.scan_hold_button.clicked.connect(self.hold_requested)
        run_row.addWidget(self.scan_hold_button)
        self.scan_step_back_button = FluentButton("◀ Step")
        self.scan_step_forward_button = FluentButton("Step ▶")
        for button, delta in ((self.scan_step_back_button, -1), (self.scan_step_forward_button, 1)):
            button.setFixedHeight(row_height())
            button.clicked.connect(lambda _checked=False, value=delta: self.step_requested.emit(value))
            run_row.addWidget(button)
        self.scan_progress_label = FluentLabel("")
        run_row.addWidget(self.scan_progress_label, 1)
        layout.addWidget(playback)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(row_gap())
        editor_box = FluentGroupBox("Generate the scan table (Python)")
        editor_layout = QtWidgets.QVBoxLayout(editor_box)
        editor_layout.setContentsMargins(pad, pad, pad, pad)
        editor_layout.setSpacing(row_gap())
        self.scan_code = FluentCodeEdit()

        template_buttons = QtWidgets.QHBoxLayout()
        template_buttons.setSpacing(row_gap())
        self.scan_load_program_button = FluentButton("Open program…")
        self.scan_column_template_button = FluentButton("Template: columns")
        self.scan_grid_template_button = FluentButton("Template: grid")
        self.scan_load_program_button.clicked.connect(self.load_program_requested)
        self.scan_column_template_button.clicked.connect(lambda: self.template_requested.emit("columns"))
        self.scan_grid_template_button.clicked.connect(lambda: self.template_requested.emit("grid"))
        for button in (self.scan_load_program_button, self.scan_column_template_button, self.scan_grid_template_button):
            button.setFixedHeight(row_height())
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            template_buttons.addWidget(button, 1)
        editor_layout.addLayout(template_buttons)
        editor_layout.addWidget(self.scan_code, 1)

        code_buttons = QtWidgets.QHBoxLayout()
        code_buttons.setSpacing(row_gap())
        self.scan_run_button = FluentButton("Run", role="primary")
        self.scan_save_array_button = FluentButton("Save array")
        self.scan_load_array_button = FluentButton("Open array…")
        self.scan_clear_array_button = FluentButton("Clear array", role="destructive")
        self.scan_clear_array_button.setToolTip("Remove the current scan table; keep the Python draft")
        self.scan_load_array_button.clicked.connect(self.load_array_requested)
        self.scan_clear_array_button.clicked.connect(self.clear_array_requested)
        self.scan_run_button.setFixedHeight(row_height())
        self.scan_save_array_button.setFixedHeight(row_height())
        self.scan_run_button.clicked.connect(lambda: self.run_requested.emit())
        self.scan_save_array_button.clicked.connect(self.save_array_requested)
        self.scan_code.textChanged.connect(self._on_code_changed)
        for button in (self.scan_run_button, self.scan_load_array_button,
                       self.scan_save_array_button, self.scan_clear_array_button):
            button.setFixedHeight(row_height())
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            code_buttons.addWidget(button, 1)
        editor_layout.addLayout(code_buttons)
        body.addWidget(editor_box, 3)

        preview_box = FluentGroupBox("Scan table")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(pad, pad, pad, pad)
        self.scan_table_view = FluentCodeEdit(read_only=True)
        # Where there is no table, the box says why and what to do: muted
        # ink, where the table's first line would start.  A label of its
        # own over the viewport -- the read-only edit does not paint a
        # placeholder.
        self.table_note_label = quiet(FluentLabel(""))
        self.table_note_label.setParent(self.scan_table_view.viewport())
        self.table_note_label.setWordWrap(True)
        self.table_note_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.table_note_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.table_note_label.hide()
        self.scan_table_view.viewport().installEventFilter(self)
        preview_layout.addWidget(self.scan_table_view, 1)
        body.addWidget(preview_box, 2)
        layout.addLayout(body, 1)

        self._scan_progress_timer = QtCore.QTimer(self)
        self._scan_progress_timer.setInterval(200)
        self._scan_progress_timer.timeout.connect(self._request_visible_progress)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt name
        if watched is self.scan_table_view.viewport() and event.type() == QtCore.QEvent.Resize:
            self.table_note_label.setGeometry(watched.rect())
        return super().eventFilter(watched, event)

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
        # A count of 0 walks the table forever, and the box says so where
        # it is typed: it shows ∞.  A range that starts above 0 has no
        # forever in it.
        self.scan_repeats_spin.setSpecialValueText("∞" if minimum == 0 else "")
        self.set_repeats(default)

    def _size_repeats(self, text: str = "") -> None:
        self.scan_repeats_spin.setFixedWidth(count_box_width(self.scan_repeats_spin, text))

    def set_repeats(self, repeats: int) -> None:
        value = max(self._minimum_repeats, int(repeats))
        self._committed_repeats = value
        # The box the operator is typing in is theirs; the committed count
        # is what a projection says, and it lands when they are done.
        if being_edited(self.scan_repeats_spin):
            return
        # Wide enough first: a count wider than the box is refused by it.
        self._size_repeats(str(value))
        with signals_blocked(self.scan_repeats_spin):
            self.scan_repeats_spin.setValue(float(value))

    def _commit_repeats(self) -> None:
        value = int(self.scan_repeats_spin.value())
        if value == self._committed_repeats:
            return
        self.repeats_committed.emit(value)

    def set_page(self, record: ScanPageRecord) -> None:
        """Show the whole page from its one presenter-owned state."""

        if not isinstance(record, ScanPageRecord):
            raise TypeError("record must be ScanPageRecord")
        self.set_slots_text(record.slots_text)
        self.set_bindings(record.bindings)
        self.set_scan_table_text(record.table_text)
        self.table_note_label.setText(str(record.table_note))
        self.table_note_label.setVisible(not record.table_text and bool(record.table_note))
        self.set_repeats(record.repeats)
        self.set_progress_text(record.progress_text, record.progress_word)
        self.set_workspace_busy(record.busy)
        self.set_progress_polling(record.progress_polling)
        if record.source_text != self.scan_code.toPlainText():
            with signals_blocked(self.scan_code):
                self.scan_code.setPlainText(str(record.source_text))
        self.set_run_dirty(record.source_dirty)

    def set_slots_text(self, text: str) -> None:
        self.scan_slots_label.setText(str(text))

    def set_bindings(self, bindings) -> None:
        records = tuple(bindings)
        if records == self._bindings_shown:
            return
        self._bindings_shown = records
        while self._bindings_layout.count():
            retire_widget(self._bindings_layout.takeAt(0).widget())
        groups = {}
        for record in records:
            groups.setdefault(record.group_id, (record.group_label, []))[1].append(record)
        show_groups = any(groups)
        sources = dict(_SOURCES)
        for group_id in sorted(groups, key=lambda group_id: bool(group_id)):
            label, members = groups[group_id]
            if show_groups:
                self._bindings_layout.addWidget(FluentSectionLabel(label or "Pulse"))
            for record in members:
                line = ElidedLabel(f"{record.label} · " + " + ".join(
                    (["Scan"] if record.scan else [])
                    + ([sources[record.source]] if record.source != "default" else []))
                    + (f" — {record.note}" if record.note else ""))
                if record.note:
                    line.setStyleSheet(f'QLabel {{ {MUTED_LABEL_STYLE} font: {fluent_font_size()}pt "{FONT}"; }}')
                line.setFixedHeight(row_height())
                self._bindings_layout.addWidget(line)
        self.bindings_box.setVisible(bool(records))

    def set_progress_text(self, text: str, word: str) -> None:
        """The Scan page's line, in the ink of status ``word``."""

        self.scan_progress_label.setText(str(text))
        set_status_ink(self.scan_progress_label, word)

    def set_scan_table_text(self, text: str) -> None:
        text = str(text)
        if text != self.scan_table_view.toPlainText():
            self.scan_table_view.setPlainText(text)

    def set_run_dirty(self, dirty: bool) -> None:
        self.scan_run_button.set_dirty(bool(dirty))

    def set_workspace_busy(self, busy: bool) -> None:
        enabled = not bool(busy)
        for button in (self.scan_load_program_button, self.scan_column_template_button,
                       self.scan_grid_template_button, self.scan_run_button,
                       self.scan_save_array_button, self.scan_load_array_button,
                       self.scan_clear_array_button):
            button.setEnabled(enabled)


__all__ = ["PulseScanView"]
