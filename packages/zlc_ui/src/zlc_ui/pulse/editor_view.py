"""Pure editor shell that composes the four pulse pages."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    fluent_confirm,
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    FluentButton,
    fluent_message,
    FluentFrame,
    FluentLabel,
    FluentLineEdit,
    FluentStatusDot,
    FluentTabWidget,
    fluent_widget_stylesheet,
    scaled_px,
    window_pad,
)

from .preview_view import PulsePreviewView
from .scan_view import PulseScanView
from .schedule_view import PulseScheduleView
from .target_view import PulseTargetView


class PulseEditorView(QtWidgets.QWidget):
    clear_all_requested = QtCore.pyqtSignal()
    close_requested = QtCore.pyqtSignal()
    warning_requested = QtCore.pyqtSignal(str)
    #: Something the operator asked for, and got.  A window that only ever
    #: speaks up to refuse leaves every success looking like silence -- and
    #: routing a confirmation through the warning channel says it in the
    #: colour of a problem.
    done_requested = QtCore.pyqtSignal(str)
    #: Which page the operator is looking at, by its title.  A page nobody has
    #: turned to is not a page that should cost anything: the Preview builds a
    #: render worker and a drawing session, and building that at open drew a
    #: timeline into a tab the window had not shown yet.
    page_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PulseGUI@Zou lab")
        self.setStyleSheet(fluent_widget_stylesheet())
        self._closing = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(window_pad(1), window_pad(1), window_pad(1), window_pad(1))
        outer.setSpacing(window_pad(0.5))

        header = FluentFrame(bordered=False)
        header.setFixedHeight(scaled_px(48, minimum=38))
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))
        header_layout.setSpacing(scaled_px(8, minimum=5))
        self.status_dot = FluentStatusDot(size=16)
        self.title_label = FluentLabel("PulseGUI - Untitled*")
        self.title_label.setMinimumWidth(scaled_px(260, minimum=180))
        self.title_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.summary_label = FluentLineEdit("")
        self.summary_label.setEnabled(False)
        self.clear_button = FluentButton("Clear All", color=ORANGE)
        self.clear_button.clicked.connect(self.clear_all_requested)
        header_layout.addWidget(self.status_dot)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.summary_label, 1)
        header_layout.addWidget(self.clear_button)
        outer.addWidget(header)

        self.tabs = FluentTabWidget()
        self.schedule_view = PulseScheduleView()
        self.preview_view = PulsePreviewView()
        self.scan_view = PulseScanView()
        self.target_view = PulseTargetView()
        self.tabs.add_permanent_tab(self.schedule_view, "Edit")
        self.tabs.add_permanent_tab(self.preview_view, "Preview")
        self.tabs.add_permanent_tab(self.scan_view, "Scan")
        self.tabs.add_permanent_tab(self.target_view, "Target")
        outer.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(
            lambda index: self.page_changed.emit(self.tabs.tabText(index))
        )
        # What just happened is SAID, in the one modal this project already
        # owns -- the same dialog v1 used for every refusal and every
        # confirmation.  A strip along the bottom was invented here instead,
        # and inventing it brought its own severity vocabulary, which this
        # window then got wrong: a Save or a Sync that worked raised
        # ValueError out of a Qt slot and took the process with it.
        self.done_requested.connect(
            lambda text: fluent_message(self, "Pulse", str(text), kind="info")
        )
        self.warning_requested.connect(
            lambda text: fluent_message(self, "Pulse", str(text), kind="warning")
        )

    @property
    def current_page(self) -> str:
        """The page on screen right now, for a host that has just been wired.

        A signal only reports CHANGES, so a host connected after the window was
        built would not know which page it opened on.
        """

        return str(self.tabs.tabText(self.tabs.currentIndex()))

    def set_title(self, text: str) -> None:
        self.title_label.setText(str(text))

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(str(text))

    def set_status_color(self, token: str) -> None:
        colors = {
            "running-synced": GREEN,
            "running-stale": ORANGE,
            "failed": RED,
            "dirty-ready": ACCENT,
            "idle": GREY,
        }
        self.status_dot.set_color(colors.get(str(token), GREY))

    def set_capabilities(self, can_sync: bool, can_hold: bool, can_step: bool) -> None:
        self.schedule_view.set_capabilities(can_sync, can_hold, can_step)
        self.scan_view.scan_hold_button.setEnabled(bool(can_hold))
        self.scan_view.scan_step_back_button.setEnabled(bool(can_step))
        self.scan_view.scan_step_forward_button.setEnabled(bool(can_step))

    def ask_open_path(self, caption: str, start_dir: str, filter: str) -> str:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(self, str(caption), str(start_dir), str(filter))
        return str(path)

    def ask_save_path(self, caption: str, start_dir: str, filter: str) -> str:
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(self, str(caption), str(start_dir), str(filter))
        return str(path)

    def confirm(self, title: str, text: str, ok: str = "OK", cancel: str = "Cancel") -> bool:
        """Ask a yes-or-no, in the same dialog every other message uses.

        It was a bare QMessageBox: system-styled, differently sized, and
        arriving in the middle of a window that had spent a lot of effort
        looking like one thing.  Nothing here is worth a second dialog family.
        """

        return fluent_confirm(
            self, str(title), str(text),
            confirm_text=str(ok), cancel_text=str(cancel),
        )

    def show_done(self, text: str) -> None:
        """Say that something asked for has happened."""

        self.done_requested.emit(str(text))

    def show_warning(self, text: str) -> None:
        """Say what was refused, where it stays until something else happens.

        It used to also overwrite the summary -- a derived line about the pulse
        -- so the message survived only until the next redraw, and the summary
        then lied about the pulse until something recomputed it.
        """

        self.warning_requested.emit(str(text))

    def finish_close(self) -> None:
        self._closing = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._closing:
            event.ignore()
            self.close_requested.emit()
            return
        super().closeEvent(event)


__all__ = ["PulseEditorView"]
