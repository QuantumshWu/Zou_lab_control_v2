"""Subpulse editing with the same Period/Bracket surface as a full Pulse."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ElidedLabel, FluentButton, FluentComboBox,
    FluentFrame, FluentGroupBox, FluentLabel, FluentLineEdit, FluentScrollArea,
    fluent_scrollbar_thickness, read_editable_combo, retire_widget,
    setting_label_width, signals_blocked,
)

from ._layout import (
    bound_field_width, config_name_width, px, quiet, row_height,
)
from zlc_ui.fluent import card_pad, row_gap, section_gap
from .models import ScheduleVM
from .schedule_view import PulseScheduleView


class PulseComponentView(QtWidgets.QWidget):
    """Only editor intents and view records cross this page's boundary."""

    action_requested = QtCore.pyqtSignal(str, object)
    edit_requested = QtCore.pyqtSignal(str, object)
    feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._context_id = ""
        self._selected_id = ""
        self._accepted_name = ""
        self._binding_rows: dict[str, tuple] = {}
        self._sources: tuple[tuple[str, str, str, str, str], ...] = ()
        self._contexts: tuple[tuple[str, str], ...] = ()
        self.schedule_view: PulseScheduleView | None = None
        # The choice of Subpulse and the schedule's commands stand still at
        # the top; under them the page scrolls as one, with one vertical
        # bar: the schedule as tall as its rows, then the references.  A
        # schedule scrolling inside the scrolling page would stand two bars
        # side by side, the wheel moving whichever the pointer is over.
        # The gap under the commands is above the scrolling body, so
        # scrolled content clips one gap below them.  The cards end on every
        # tab's right edge: the bar's gutter, kept at all times, stands
        # inside the right inset.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        page = section_gap()
        self._pinned = QtWidgets.QVBoxLayout()
        self._pinned.setContentsMargins(page, page, page, page)
        self._pinned.setSpacing(section_gap())
        outer.addLayout(self._pinned)
        scroll = FluentScrollArea()
        page_body = QtWidgets.QWidget()
        scroll.set_width_bounded_widget(page_body)
        outer.addWidget(scroll, 1)
        self._page = QtWidgets.QVBoxLayout(page_body)
        self._page.setContentsMargins(page, 0, max(0, page - fluent_scrollbar_thickness()), page)
        self._page.setSpacing(section_gap())
        pad = card_pad()
        header = FluentFrame()
        head = QtWidgets.QVBoxLayout(header)
        head.setContentsMargins(pad, pad, pad, pad)
        head.setSpacing(row_gap())
        # The two rows are one form: one label column, so the Subpulse
        # choice and the Config file start on one x, and both rows end on
        # the card's right edge, Browse under the last command.
        label_width = setting_label_width(("Subpulse", "Config file"), minimum=0)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(row_gap())
        subpulse_label = FluentLabel("Subpulse")
        subpulse_label.setFixedWidth(label_width)
        top.addWidget(subpulse_label)
        self.context_combo = FluentComboBox()
        self.context_combo.setMinimumWidth(px(150))
        self.context_combo.currentIndexChanged.connect(self._context_changed)
        top.addWidget(self.context_combo, 1)
        top.addWidget(FluentLabel("Name"))
        self.name_edit = FluentLineEdit("")
        self.name_edit.setPlaceholderText("Subpulse name")
        self.name_edit.set_placeholder_prompt()
        self.name_edit.editingFinished.connect(self._name_committed)
        top.addWidget(self.name_edit, 1)
        self._buttons = {}
        for action, text in (
            ("new", "New"), ("open", "Open"), ("save", "Save"),
            ("save_as", "Save as"), ("insert", "Insert into Pulse"),
        ):
            button = FluentButton(text)
            button.clicked.connect(lambda _checked=False, a=action: self.action_requested.emit(a, self._context_id))
            self._buttons[action] = button
            top.addWidget(button)
        head.addLayout(top)
        self.config_file_row = QtWidgets.QWidget()
        self.config_file_row.setObjectName("subpulseConfigFileRow")
        self.config_file_row.setStyleSheet("QWidget#subpulseConfigFileRow { background: transparent; }")
        file_row = QtWidgets.QHBoxLayout(self.config_file_row)
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(row_gap())
        file_label = FluentLabel("Config file")
        file_label.setFixedWidth(label_width)
        file_row.addWidget(file_label)
        self.config_path_edit = FluentLineEdit("")
        self.config_path_edit.setPlaceholderText("Choose a Subpulse file")
        self.config_path_edit.editingFinished.connect(self._commit_source)
        file_row.addWidget(self.config_path_edit, 1)
        self.browse_source_button = FluentButton("Browse…")
        self.browse_source_button.clicked.connect(
            lambda _checked=False: self.action_requested.emit("browse_source", self._selected_id)
        )
        file_row.addWidget(self.browse_source_button)
        head.addWidget(self.config_file_row)
        self.config_file_row.hide()
        self.path_label = ElidedLabel("")
        head.addWidget(self.path_label)
        # What Save does here from the card's left edge, and the source's
        # state at its right -- not squeezed in after Browse, where it ended
        # the file row short of the commands above it.
        hint = QtWidgets.QHBoxLayout()
        hint.setSpacing(row_gap())
        self.scope_label = ElidedLabel("")
        hint.addWidget(self.scope_label, 1)
        self.source_status = quiet(FluentLabel(""))
        hint.addWidget(self.source_status)
        self.source_status.hide()
        head.addLayout(hint)
        self._pinned.addWidget(header)
        self.placeholder = FluentLabel("Choose a Subpulse above, or create or open a Subpulse file.")
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._page.addWidget(self.placeholder, 1)
        # The references read like Value Config's bindings: the field's
        # label in a column as wide as the widest, the Config name beside it
        # in a box of the same width as there.
        self.bindings_box = FluentGroupBox("Numeric Config references")
        self._binding_layout = QtWidgets.QGridLayout(self.bindings_box)
        self._binding_layout.setContentsMargins(pad, pad, pad, pad)
        self._binding_layout.setHorizontalSpacing(row_gap())
        self._binding_layout.setVerticalSpacing(row_gap())
        self._binding_layout.setColumnStretch(2, 1)
        self._binding_layout.setAlignment(QtCore.Qt.AlignTop)
        self._page.addWidget(self.bindings_box)
        self.bindings_box.hide()
        # The slack under a short schedule, so it and the references stay
        # at the top; the placeholder's stretch takes it while it shows.
        self._page.addStretch(0)

    def set_sources(
        self, rows: tuple[tuple[str, str, str, str, str], ...], *,
        enabled: bool, busy: bool = False, error: str = "",
    ) -> None:
        """Update choices and the selected reference; global mode never changes selection."""
        del enabled, error
        self._sources = tuple(rows)
        self._set_context_choices()
        self._project_source(busy=busy)

    def _set_context_choices(self) -> None:
        contexts = tuple((key, name) for key, name, _source, _path, _status in self._sources) or self._contexts
        choices = (("", "Subpulse file"),) + tuple((key, label) for key, label in contexts if key)
        with signals_blocked(self.context_combo):
            existing = tuple((self.context_combo.itemData(index), self.context_combo.itemText(index))
                             for index in range(self.context_combo.count()))
            if existing != choices:
                self.context_combo.clear()
                for key, label in choices:
                    self.context_combo.addItem(label, key)
            self.context_combo.setCurrentIndex(self.context_combo.findData(self._selected_id))

    def _project_source(self, *, busy: bool = False) -> None:
        record = next((row for row in self._sources if row[0] == self._selected_id), None)
        self.config_file_row.setVisible(record is not None)
        # The source is said once on the line: editing a Default Subpulse in
        # the Pulse, the scope already names it, and the status would only
        # say "Default" again at the far end.
        self.source_status.setVisible(
            record is not None and bool(record[4])
            and not (self._context_id and record[2] != "config")
        )
        if record is None:
            return
        key, _name, source, path, status = record
        with signals_blocked(self.config_path_edit):
            draft = (self.config_path_edit.property("component_id") == key
                     and self.config_path_edit.hasFocus() and self.config_path_edit.isModified())
            if not draft:
                self.config_path_edit.setText(path)
            self.config_path_edit.setProperty("component_id", key)
            self.config_path_edit.setProperty("accepted_source", source)
            self.config_path_edit.setProperty("accepted_path", path)
        self.config_path_edit.setEnabled(not busy)
        self.browse_source_button.setEnabled(not busy)
        self.config_path_edit.setToolTip(path or "Choose a Subpulse file")
        self.source_status.setText(status)
        self.source_status.setToolTip(status)

    def _commit_source(self) -> None:
        key = self.config_path_edit.property("component_id")
        if not key or key != self._selected_id:
            return
        path = self.config_path_edit.text().strip()
        source = self.config_path_edit.property("accepted_source")
        self.config_path_edit.setModified(False)
        if path != (self.config_path_edit.property("accepted_path") or ""):
            self.action_requested.emit("source", (key, source, path))
        with signals_blocked(self.config_path_edit):
            self.config_path_edit.setText(self.config_path_edit.property("accepted_path") or "")

    def _context_changed(self) -> None:
        key = self.context_combo.currentData()
        if key is not None and key != self._selected_id:
            self.action_requested.emit("context", str(key))
            with signals_blocked(self.context_combo):
                self.context_combo.setCurrentIndex(self.context_combo.findData(self._selected_id))

    def _name_committed(self) -> None:
        name = self.name_edit.text().strip()
        if name != self._accepted_name:
            self.action_requested.emit("rename", (self._context_id, name))

    def _make_schedule(self) -> PulseScheduleView:
        schedule = PulseScheduleView(embedded=True)
        schedule.setObjectName("componentSchedule")
        for action, signal in (
            ("period_name", "period_name_committed"), ("duration", "duration_committed"),
            ("digital", "digital_committed"), ("analog", "analog_committed"),
            ("binding", "binding_committed"), ("insert_period", "insert_period_requested"),
            ("insert_spacer", "insert_spacer_requested"), ("reorder_items", "reorder_items_requested"),
            ("remove_items", "remove_items_requested"), ("bracket", "bracket_committed"),
            ("bracket_add", "bracket_add_requested"),
            ("visible_ports", "visible_ports_committed"),
            ("config_binding", "config_binding_committed"),
        ):
            getattr(schedule, signal).connect(lambda *args, a=action: self.edit_requested.emit(a, args))
        schedule.feedback_requested.connect(self.feedback_requested)
        self._pinned.addWidget(schedule.toolbar)
        self._page.insertWidget(1, schedule)
        return schedule

    def set_document(
        self,
        schedule: ScheduleVM | None,
        *,
        contexts: tuple[tuple[str, str], ...] = (),
        context_id: str = "",
        selected_id: str | None = None,
        path: str = "",
        dirty: bool = False,
        bindings: tuple[tuple[str, str, str], ...] = (),
        config_names: tuple[str, ...] = (),
        busy: bool = False,
    ) -> None:
        self._context_id = str(context_id)
        self._selected_id = str(context_id if selected_id is None else selected_id)
        self._contexts = tuple(contexts)
        self._set_context_choices()
        self._project_source(busy=busy)
        self._accepted_name = schedule.document_name if schedule else ""
        self.name_edit.setText(self._accepted_name)
        self.name_edit.setEnabled(schedule is not None and not busy)
        instance = bool(self._context_id)
        self._buttons["save"].setText("Save Pulse" if instance else "Save")
        self._buttons["save_as"].setText("Export file" if instance else "Save as")
        self._buttons["insert"].setText("Duplicate in Pulse" if instance else "Insert into Pulse")
        for action, button in self._buttons.items():
            button.setEnabled(not busy and (schedule is not None or action in ("new", "open")))
        # Primary with its star only while there is something to save.
        self._buttons["save"].set_dirty(dirty)
        self.path_label.setText(path or ("Embedded in the current Pulse" if instance else "Unsaved Subpulse"))
        self.path_label.setToolTip(path)
        self.scope_label.setText(
            "Default in Pulse — Save Pulse keeps these edits."
            if instance else "Config file — Save writes this Subpulse file."
            if self._selected_id else "Subpulse file — Save writes this independent file."
        )
        self.path_label.setVisible(schedule is not None and not self._selected_id)
        self.scope_label.setVisible(schedule is not None)
        self.placeholder.setVisible(schedule is None)
        if schedule is not None:
            if self.schedule_view is None:
                self.schedule_view = self._make_schedule()
            self.schedule_view.set_schedule(schedule)
            self.schedule_view.setEnabled(not busy)
            self.schedule_view.show()
            self.schedule_view.toolbar.setEnabled(not busy)
            self.schedule_view.toolbar.show()
        elif self.schedule_view is not None:
            self.schedule_view.hide()
            self.schedule_view.toolbar.hide()
        # Under the commands, the schedule's own row gap; under the header
        # alone, the gap between two sections.
        margins = self._pinned.contentsMargins()
        self._pinned.setContentsMargins(margins.left(), margins.top(), margins.right(),
                                        row_gap() if schedule is not None else section_gap())
        wanted = {field_id for field_id, _label, _key in bindings}
        for field_id in tuple(self._binding_rows):
            if field_id not in wanted:
                for widget in self._binding_rows.pop(field_id):
                    self._binding_layout.removeWidget(widget)
                    retire_widget(widget)
        for row, (field_id, label, key) in enumerate(bindings):
            widgets = self._binding_rows.get(field_id)
            if widgets is None:
                label_widget = ElidedLabel(label)
                label_widget.setFixedHeight(row_height())
                combo = FluentComboBox()
                combo.setFixedSize(config_name_width(), row_height())
                combo.setEditable(True)
                combo.lineEdit().setPlaceholderText("Config name")
                combo.lineEdit().editingFinished.connect(lambda fid=field_id: self._commit_binding(fid))
                combo.activated.connect(lambda _index, fid=field_id: self._commit_binding(fid))
                widgets = self._binding_rows[field_id] = (label_widget, combo)
            label_widget, combo = widgets
            label_widget.setText(label)
            with signals_blocked(combo, combo.lineEdit()):
                values = tuple(dict.fromkeys(config_names))
                if tuple(combo.itemData(i) for i in range(combo.count())) != values:
                    combo.clear()
                    for value in values:
                        combo.addItem(value, value)
                combo.setCurrentIndex(combo.findData(key))
                combo.setEditText(key)
                combo.setProperty("accepted_key", key)
                combo.setEnabled(not busy)
            self._binding_layout.addWidget(label_widget, row, 0)
            self._binding_layout.addWidget(combo, row, 1)
        width = bound_field_width([widgets[0].text() for widgets in self._binding_rows.values()])
        for label_widget, _combo in self._binding_rows.values():
            label_widget.setFixedWidth(width)
        self.bindings_box.setVisible(bool(bindings))

    def _commit_binding(self, field_id: str) -> None:
        combo = self._binding_rows[field_id][1]
        key = read_editable_combo(combo)
        if key != (combo.property("accepted_key") or ""):
            self.edit_requested.emit("config_binding", (field_id, key))


__all__ = ["PulseComponentView"]
