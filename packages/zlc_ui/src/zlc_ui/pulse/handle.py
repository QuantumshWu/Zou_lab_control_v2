"""The whole of what a pulse editor offers the outside: one object, no widgets.

A window used to be handed over as its own widget tree, so a host could reach
`view.schedule_view.channel_panel` and push a value straight into one panel.
Nothing stopped it, so it happened, and "which rows exist" quietly became two
facts kept in two places -- the delay column went on showing rows that Hide Off
had already taken out of the cards.  Whatever the outside can hold, the outside
will assemble, and then there are two owners of the same picture.

So the outside holds this instead: signals to hear and methods to call, and no
way to reach a QWidget at all.  The port is flat on purpose.  Grouping it back
into pages would have kept the sub-view shape under a new name, and flattening
it is what surfaced two collisions that had been hiding behind that shape --
the schedule and the scan page both called their trigger `run_requested`, and
the schedule and the preview both called their button `save_requested`.

A drawn panel arrives as its HOST, never as a widget: this package may not
import zlc_plot (a canvas would drag matplotlib into the GUI layer), and the
host is asked for its widget here, where widgets are allowed.
"""

from __future__ import annotations

from typing import Any

from PyQt5 import QtCore

from .editor_view import PulseEditorView
from .models import ConnectionVM


class PulseEditorHandle(QtCore.QObject):
    """One pulse editor, as the outside sees it."""

    # -- the window ------------------------------------------------------
    close_requested = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    # -- the document ----------------------------------------------------
    document_name_committed = QtCore.pyqtSignal(str)
    clear_all_requested = QtCore.pyqtSignal()
    save_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    values_save_requested = QtCore.pyqtSignal()
    values_load_requested = QtCore.pyqtSignal()

    # -- the schedule ----------------------------------------------------
    port_label_committed = QtCore.pyqtSignal(str, str)
    period_name_committed = QtCore.pyqtSignal(str, str)
    duration_committed = QtCore.pyqtSignal(str, object, str)
    digital_committed = QtCore.pyqtSignal(str, str, bool)
    analog_committed = QtCore.pyqtSignal(str, str, str, object)
    #: The page the operator turned to.  What a page costs is only worth
    #: paying while it is being looked at.
    page_changed = QtCore.pyqtSignal(str)
    delay_committed = QtCore.pyqtSignal(str, object, str)
    binding_cycle_requested = QtCore.pyqtSignal(str, object, object)
    insert_period_requested = QtCore.pyqtSignal(object)
    move_period_requested = QtCore.pyqtSignal(str, object)
    remove_period_requested = QtCore.pyqtSignal(str)
    repeat_committed = QtCore.pyqtSignal(object, object, int)
    visible_ports_committed = QtCore.pyqtSignal(object)
    clear_port_requested = QtCore.pyqtSignal(str)
    feedback_requested = QtCore.pyqtSignal(str)

    # -- the board -------------------------------------------------------
    connection_requested = QtCore.pyqtSignal(str, str)
    fire_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    sync_requested = QtCore.pyqtSignal()

    # -- the scan --------------------------------------------------------
    scan_array_load_requested = QtCore.pyqtSignal()
    scan_source_edited = QtCore.pyqtSignal(str)
    scan_repeats_committed = QtCore.pyqtSignal(int)
    scan_hold_requested = QtCore.pyqtSignal()
    scan_step_requested = QtCore.pyqtSignal(int)
    scan_program_load_requested = QtCore.pyqtSignal()
    scan_template_requested = QtCore.pyqtSignal(str)
    scan_run_requested = QtCore.pyqtSignal()
    scan_array_save_requested = QtCore.pyqtSignal()
    scan_progress_refresh_requested = QtCore.pyqtSignal()

    # -- the preview -----------------------------------------------------
    preview_include_off_toggled = QtCore.pyqtSignal(bool)
    preview_size_committed = QtCore.pyqtSignal(str)
    preview_selectors_toggled = QtCore.pyqtSignal(bool)
    preview_save_requested = QtCore.pyqtSignal()

    # -- the target ------------------------------------------------------
    target_apply_requested = QtCore.pyqtSignal(tuple)
    target_feedback_requested = QtCore.pyqtSignal(str)

    def __init__(self, window: Any, view: PulseEditorView) -> None:
        super().__init__()
        self._window = window
        self._view = view
        schedule = view.schedule_view
        scan = view.scan_view
        preview = view.preview_view
        target = view.target_view

        view.clear_all_requested.connect(self.clear_all_requested)
        view.close_requested.connect(self.close_requested)
        view.page_changed.connect(self.page_changed)
        for name in (
            "document_name_committed", "port_label_committed",
            "period_name_committed", "duration_committed", "digital_committed",
            "analog_committed", "delay_committed", "binding_cycle_requested",
            "insert_period_requested", "move_period_requested",
            "remove_period_requested", "repeat_committed",
            "visible_ports_committed", "clear_port_requested",
            "feedback_requested", "connection_requested", "stop_requested",
            "sync_requested", "save_requested", "load_requested",
            "values_save_requested", "values_load_requested",
            "scan_array_load_requested",
        ):
            getattr(schedule, name).connect(getattr(self, name))
        schedule.run_requested.connect(self.fire_requested)
        scan.repeats_committed.connect(self.scan_repeats_committed)
        scan.hold_requested.connect(self.scan_hold_requested)
        scan.step_requested.connect(self.scan_step_requested)
        scan.load_program_requested.connect(self.scan_program_load_requested)
        scan.template_requested.connect(self.scan_template_requested)
        scan.source_edited.connect(self.scan_source_edited)
        scan.run_requested.connect(self.scan_run_requested)
        scan.save_array_requested.connect(self.scan_array_save_requested)
        scan.progress_refresh_requested.connect(self.scan_progress_refresh_requested)
        preview.include_off_toggled.connect(self.preview_include_off_toggled)
        preview.size_committed.connect(self.preview_size_committed)
        preview.selectors_toggled.connect(self.preview_selectors_toggled)
        preview.save_requested.connect(self.preview_save_requested)
        target.apply_requested.connect(self.target_apply_requested)
        target.feedback_requested.connect(self.target_feedback_requested)
        if window is not None and hasattr(window, "closed"):
            window.closed.connect(self.closed)

    # ------------------------------------------------------------ the window

    def close(self) -> None:
        """Ask the owning window to close through its lifecycle guard."""

        if self._window is None:
            self._view.finish_close()
            return
        self._window.close()

    def set_close_guard(self, guard) -> None:
        """Require plugin cleanup to finish before the window disappears."""

        if self._window is not None:
            self._window.set_close_guard(guard)

    def finish_close(self) -> None:
        self._view.finish_close()

    def is_visible(self) -> bool:
        target = self._window if self._window is not None else self._view
        return bool(target.isVisible())

    def restore(self) -> None:
        """Raise this existing editor; never construct a second window."""

        target = self._window if self._window is not None else self._view
        if hasattr(target, "showNormal"):
            target.showNormal()
        else:
            target.show()
        if hasattr(target, "raise_"):
            target.raise_()
        if hasattr(target, "activateWindow"):
            target.activateWindow()

    def window_size(self) -> tuple[int, int]:
        """Width and height of the window, for acceptance to check the rule."""

        target = self._window if self._window is not None else self._view
        size = target.size()
        return int(size.width()), int(size.height())

    def window_title(self) -> str:
        target = self._window if self._window is not None else self._view
        return str(target.windowTitle())

    # ---------------------------------------------------------- the document

    def set_title(self, text: str) -> None:
        self._view.set_title(text)

    def set_summary(self, text: str) -> None:
        self._view.set_summary(text)

    def set_status_color(self, token: str) -> None:
        self._view.set_status_color(token)

    def set_capabilities(self, can_sync: bool, can_hold: bool, can_step: bool) -> None:
        self._view.set_capabilities(can_sync, can_hold, can_step)

    def show_done(self, text: str) -> None:
        self._view.show_done(text)

    def show_warning(self, text: str) -> None:
        self._view.show_warning(text)

    @property
    def current_page(self) -> str:
        """Which page is on screen, for a host wired after the window was built."""

        return self._view.current_page

    def ask_open_path(self, caption: str, start: str, filter: str) -> str:
        return self._view.ask_open_path(caption, start, filter)

    def ask_save_path(self, caption: str, suggested: str, filter: str) -> str:
        return self._view.ask_save_path(caption, suggested, filter)

    # ---------------------------------------------------------- the schedule

    def set_schedule(self, schedule: Any) -> None:
        self._view.schedule_view.set_schedule(schedule)

    def set_period(self, period: Any) -> None:
        self._view.schedule_view.set_period(period)

    def set_delay_row(self, row: Any) -> None:
        self._view.schedule_view.set_delay_row(row)

    def set_port_label(self, key: str, label: str) -> None:
        self._view.schedule_view.set_port_label(key, label)

    def set_schedule_summary(
        self,
        total_text: str,
        total_tooltip: str,
        period_count: int,
        visible_text: str,
        summary_text: str,
        scan_summary_text: str,
    ) -> None:
        self._view.schedule_view.set_summary(
            total_text, total_tooltip, period_count, visible_text,
            summary_text, scan_summary_text,
        )

    def set_visible_ports(self, ports: tuple[str, ...]) -> None:
        self._view.schedule_view.set_visible_ports(ports)

    def set_control_state(
        self,
        running: bool,
        synchronized: bool,
        file_dirty: bool,
        *,
        can_run: bool,
        can_stop: bool,
    ) -> None:
        self._view.schedule_view.set_control_state(
            running, synchronized, file_dirty, can_run=can_run, can_stop=can_stop
        )

    def set_connection(self, connection: ConnectionVM) -> None:
        self._view.schedule_view.set_connection(connection)

    def set_scan_busy(self, busy: bool) -> None:
        self._view.schedule_view.set_scan_busy(busy)

    # -------------------------------------------------------------- the scan

    def set_scan_page(self, record: Any) -> None:
        self._view.scan_view.set_page(record)

    def set_scan_progress_text(self, text: str) -> None:
        self._view.scan_view.set_progress_text(text)

    def set_scan_repeats_range(self, low: int, high: int) -> None:
        self._view.scan_view.set_repeats_range(low, high)

    # ----------------------------------------------------------- the preview

    @property
    def preview_include_off_rows(self) -> bool:
        return bool(self._view.preview_view.include_off_rows)

    @property
    def preview_size(self) -> str:
        return str(self._view.preview_view.preview_size)

    @property
    def preview_size_pinned(self) -> bool:
        return bool(self._view.preview_view.preview_size_pinned)

    def set_preview_size(self, size: str, *, pinned: bool | None = None) -> None:
        self._view.preview_view.set_preview_size(size, pinned=pinned)

    def set_preview_size_names(self, names: tuple[str, ...]) -> None:
        self._view.preview_view.set_size_names(names)

    def reset_preview_size_pin(self) -> None:
        self._view.preview_view.reset_preview_size_pin()

    def set_preview_status(self, text: str) -> None:
        self._view.preview_view.set_status(text)

    def show_preview_placeholder(self, text: str) -> None:
        self._view.preview_view.show_placeholder(text)

    def show_preview(self, host: Any) -> None:
        """Put a drawn panel on the preview page, given the thing that owns it.

        The host is asked for its widget HERE.  zlc_ui may not import the
        package that draws -- a canvas would carry matplotlib into the GUI
        layer -- and the outside may not hold a widget, so the only seam that
        breaks neither rule is one where the drawing package makes the widget
        and this package is the only one that ever touches it.
        """

        widget = host.qt_widget()
        self._view.preview_view.mount_content(
            widget,
            logical_size=getattr(host, "logical_size", None),
            wheel_target=host.wheel_target() if hasattr(host, "wheel_target") else None,
        )

    # ------------------------------------------------------------ the target

    def set_target_ports(self, records: tuple, editable: bool, status_text: str) -> None:
        self._view.target_view.set_ports(records, editable, status_text)

    def set_target_width_rules(self, digital: Any, dac: Any) -> None:
        self._view.target_view.set_width_rules(digital, dac)

    def set_target_feedback(self, text: str) -> None:
        self._view.target_view.set_feedback(text)


__all__ = ["PulseEditorHandle"]
