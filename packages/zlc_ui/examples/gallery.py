"""Interactive gallery for the domain-independent zlc_ui controls.

The gallery deliberately uses only fake strings, numbers, and graph records.
It consumes the installed `zlc_ui` layer and needs no domain or device package.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The product bootstrap, first, exactly as every launcher does it: it is
# what puts THIS checkout's layers on the path.  Without it a demo run
# from a source tree imports whichever copy of each layer an editable
# install happens to point at, and demonstrates that one instead.
import zou_lab_control  # noqa: F401

from PyQt5 import QtCore, QtWidgets  # noqa: E402

import zlc_ui.console as _console  # noqa: E402
import zlc_ui.fluent as _fluent  # noqa: E402
import zlc_ui.form as _form  # noqa: E402
import zlc_ui.pulse as _pulse  # noqa: E402
from zlc_ui import (  # noqa: E402
    BoardMetrics,
    FormChoice,
    FormFieldProps,
    FormRuntimeContext,
    FormSpec,
    ensure_qt_app,
)

ConsoleBoardView = _console.ConsoleBoardView
PanelCardView = _console.PanelCardView
ACCENT = _fluent.ACCENT
FluentButton = _fluent.FluentButton
FluentCheckBox = _fluent.FluentCheckBox
FluentCodeEdit = _fluent.FluentCodeEdit
FluentComboBox = _fluent.FluentComboBox
FluentDoubleSpinBox = _fluent.FluentDoubleSpinBox
FluentFrame = _fluent.FluentFrame
FluentGroupBox = _fluent.FluentGroupBox
FluentLabel = _fluent.FluentLabel
FluentLineEdit = _fluent.FluentLineEdit
FluentPathEdit = _fluent.FluentPathEdit
FluentReadoutEdit = _fluent.FluentReadoutEdit
FluentReadoutMultiline = _fluent.FluentReadoutMultiline
FluentScrollArea = _fluent.FluentScrollArea
FluentSectionLabel = _fluent.FluentSectionLabel
FluentSettingRow = _fluent.FluentSettingRow
FluentSpinBox = _fluent.FluentSpinBox
FluentStatusDot = _fluent.FluentStatusDot
FluentStatusStrip = _fluent.FluentStatusStrip
FluentSwitch = _fluent.FluentSwitch
FluentTabWidget = _fluent.FluentTabWidget
FluentTreeComboBox = _fluent.FluentTreeComboBox
GREEN = _fluent.GREEN
GREY = _fluent.GREY
ORANGE = _fluent.ORANGE
PublishedItemsLegend = _fluent.PublishedItemsLegend
fill_grouped_choice_combo = _fluent.fill_grouped_choice_combo
ElidedLabel = _fluent.ElidedLabel
FluentWindow = _fluent.FluentWindow
InfoPane = _fluent.InfoPane
launch_qt_window = _fluent.launch_qt_window
muted_note_label = _fluent.muted_note_label
retire_widget = _fluent.retire_widget
scaled_px = _fluent.scaled_px
WINDOW_SCREEN_FRACTION = _fluent.WINDOW_SCREEN_FRACTION
window_pad = _fluent.window_pad
FluentParameterForm = _form.FluentParameterForm
FluentScanLineEdit = _pulse.FluentScanLineEdit


class _InteractiveBindingField(QtWidgets.QWidget):
    """The production binding popup with injected Scan/source state."""

    binding_changed = QtCore.pyqtSignal(object, object)

    def __init__(
        self,
        *,
        text: str,
        scan: bool,
        source: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.scan = bool(scan)
        self.source = source

        self.field = FluentScanLineEdit(
            str(text),
            tooltip="Select Scan and value source",
        )
        self.field.setFixedHeight(scaled_px(32, minimum=26))
        self.state_label = muted_note_label("")
        self.state_label.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(window_pad(0.25))
        layout.addWidget(self.field)
        layout.addWidget(self.state_label)
        self.field.binding_committed.connect(self.binding_changed)
        self.field.set_field_state(editable=True, scan=scan, source=source)
        self.state_label.setText(f"Scan: {'on' if scan else 'off'} · Source: {source}")


class _GalleryBody(QtWidgets.QWidget):
    """The scrollable body hosted by the shared frameless window."""

    def __init__(self) -> None:
        super().__init__()

        self.binding_examples: dict[str, _InteractiveBindingField] = {}
        self._board_metrics = BoardMetrics(12)
        self.scroll = FluentScrollArea()
        self.scroll.set_width_bounded_widget(self._build_body())
        shell = QtWidgets.QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self.scroll)

    def _build_body(self) -> QtWidgets.QWidget:
        body = QtWidgets.QWidget()
        body.setObjectName("GalleryBody")
        body.setStyleSheet("QWidget#GalleryBody { background: #F3F3F3; }")
        layout = QtWidgets.QVBoxLayout(body)
        pad = window_pad()
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(window_pad(0.75))

        title = FluentLabel("zlc_ui — pure PyQt5 controls")
        title.setStyleSheet("font-size: 22pt; font-weight: bold; color: #323130;")
        layout.addWidget(title)
        intro = muted_note_label(
            "Every item below is driven by plain fake data. Click buttons, edit fields, "
            "open combos, switch tabs, and resize the window to inspect scaling."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(self._heading("1", "基础组件", "可单独复用的 Fluent 控件；每个控件旁边标出组件名。"))
        layout.addWidget(self._build_typography_section())
        layout.addWidget(self._build_input_section())
        layout.addWidget(self._heading("2", "组合件", "由基础控件组成的可复用 view；先检查局部结构，再检查完整窗口。"))
        layout.addWidget(self._build_status_form_section())
        layout.addWidget(self._build_form_section())
        layout.addWidget(self._build_pulse_binding_section())
        layout.addWidget(self._build_board_section())
        layout.addWidget(self._build_tabs_section())
        layout.addWidget(self._build_info_section())
        layout.addWidget(self._heading("3", "完整 GUI 示例", "下面四个页面各自用本示例的假数据填充正式的 view；独立窗口使用同一套 create_window()。"))
        layout.addWidget(self._build_gui_examples_section())
        layout.addStretch(1)
        return body

    @staticmethod
    def _heading(number: str, title: str, description: str) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setObjectName(f"GalleryHeading{number}")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, window_pad(0.5), 0, 0)
        layout.setSpacing(window_pad(0.25))
        label = FluentSectionLabel(f"{number}. {title}")
        label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #323130; background: transparent; border: none;")
        layout.addWidget(label)
        note = muted_note_label(description)
        note.setWordWrap(True)
        layout.addWidget(note)
        return widget

    @staticmethod
    def _section(title: str) -> tuple[FluentGroupBox, QtWidgets.QVBoxLayout]:
        card = FluentGroupBox(title)
        inner = QtWidgets.QVBoxLayout(card)
        inner.setContentsMargins(window_pad(), window_pad(1.5), window_pad(), window_pad())
        inner.setSpacing(window_pad(0.5))
        return card, inner

    @staticmethod
    def _named(name: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(window_pad(0.25))
        label = muted_note_label(str(name))
        label.setAlignment(QtCore.Qt.AlignLeft)
        layout.addWidget(label)
        layout.addWidget(widget)
        return holder

    def _build_typography_section(self) -> QtWidgets.QWidget:
        card, inner = self._section("基础：Typography / labels / status dot")
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(window_pad())
        row.addWidget(self._named("FluentStatusDot", FluentStatusDot(size=18, color="#7FC2AD")))
        row.addWidget(self._named("FluentSectionLabel", FluentSectionLabel("Section label")))
        row.addWidget(self._named("ElidedLabel", ElidedLabel("An eliding label with a longer tooltip")), 1)
        row.addWidget(self._named("FluentLabel", FluentLabel("Plain FluentLabel")))
        row.addStretch(1)
        inner.addLayout(row)
        return card

    def _build_input_section(self) -> QtWidgets.QWidget:
        card, inner = self._section("基础：buttons / text / choices / numeric controls")

        buttons = QtWidgets.QHBoxLayout()
        for text, color in (("Primary", "#77AADD"), ("Green", "#7FC2AD"), ("Warning", "#D69A6E")):
            button = FluentButton(text, color=color)
            button.clicked.connect(lambda checked=False, value=text: self._log(value))
            buttons.addWidget(self._named(f"FluentButton · {text}", button))
        check = FluentCheckBox("Fake option")
        check.setChecked(True)
        buttons.addWidget(self._named("FluentCheckBox", check))
        switch = FluentSwitch("Selectors")
        switch.setChecked(True)
        buttons.addWidget(self._named("FluentSwitch", switch))
        buttons.addStretch(1)
        inner.addLayout(buttons)

        edits = QtWidgets.QGridLayout()
        edits.setHorizontalSpacing(window_pad())
        edits.addWidget(self._named("FluentLineEdit", FluentLineEdit("editable text")), 0, 0)
        edits.addWidget(self._named("FluentReadoutEdit", FluentReadoutEdit("single-line readout")), 0, 1)
        edits.addWidget(self._named("FluentReadoutMultiline", FluentReadoutMultiline("multi-line\nreadout")), 1, 0)
        edits.addWidget(self._named("FluentCodeEdit", FluentCodeEdit("fake_value = 42", read_only=False)), 1, 1)
        edits.addWidget(self._named("FluentPathEdit", FluentPathEdit("fake/example.txt", base_dir=str(ROOT))), 2, 0, 1, 2)
        inner.addLayout(edits)

        controls = QtWidgets.QHBoxLayout()
        combo = FluentComboBox()
        combo.addItems(["first choice", "second choice", "third choice"])
        combo.setCurrentIndex(1)
        tree = FluentTreeComboBox()
        fill_grouped_choice_combo(
            tree,
            names=("temperature", "count", "waiting"),
            sources={"temperature": ("sensor",), "count": ("counter",), "waiting": ()},
            metadata={"temperature": "float", "count": "int"},
            current="temperature",
            none_label="(none)",
            labels={"temperature": "Temperature", "count": "Count"},
            state_labels=("ready", "waiting", "unassigned"),
            empty_source_label="(no source)",
        )
        controls.addWidget(self._named("FluentComboBox", combo))
        controls.addWidget(self._named("FluentTreeComboBox", tree))
        controls.addWidget(self._named("FluentSpinBox", FluentSpinBox()))
        decimal_box = FluentDoubleSpinBox()
        decimal_box.setSingleStep(0.1)
        decimal_box.setValue(-1.5)
        controls.addWidget(self._named("FluentDoubleSpinBox", decimal_box))
        controls.addStretch(1)
        inner.addLayout(controls)
        return card

    def _build_status_form_section(self) -> QtWidgets.QWidget:
        card, inner = self._section("组合：FluentStatusStrip")
        status = FluentStatusStrip(action_text="Review")
        status.show_message("task: fake acquisition is ready", severity="task")
        status.set_action_visible(True)
        inner.addWidget(self._named("FluentStatusStrip", status))

        return card

    def _build_pulse_binding_section(self) -> QtWidgets.QWidget:
        card, inner = self._section("FluentScanLineEdit — Scan / API / Config")
        note = muted_note_label(
            "Production Pulse fields: select Scan and a value source. All fields keep an editable default."
        )
        note.setWordWrap(True)
        inner.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(window_pad())
        examples = (
            ("duration", "Duration", "0", False, "default"),
            ("scan_duration", "Scan · duration", "1000", True, "default"),
            ("api_duration", "Scan + API · duration", "1000", True, "api"),
            ("dac", "Scan + Config · da_bias_y", "0", True, "config"),
            ("delay", "Delay", "0", False, "default"),
        )
        echo = muted_note_label("last binding click: —")
        echo.setWordWrap(True)
        inner.addWidget(echo)
        for key, name, text, scan, source in examples:
            demo = _InteractiveBindingField(
                text=text,
                scan=scan,
                source=source,
            )
            demo.field.setObjectName(f"GalleryBinding_{key}")
            demo.field.setFixedWidth(scaled_px(150, minimum=120))
            demo.binding_changed.connect(
                lambda next_scan, next_source, key=key: self._echo_binding(
                    echo, key, next_scan, next_source
                )
            )
            self.binding_examples[key] = demo
            row.addWidget(self._named(f"FluentScanLineEdit · {name}", demo), 1)
        inner.addLayout(row)
        return card

    @staticmethod
    def _echo_binding(label: FluentLabel, key: str, scan: bool, source: str) -> None:
        label.setText(f"last binding intent: {key} → Scan={scan}, source={source}")
        print(label.text(), flush=True)

    def _build_board_section(self) -> QtWidgets.QWidget:
        card, inner = self._section("ConsoleBoardView — drag cards to reorder")
        board = ConsoleBoardView(metrics=self._board_metrics)
        board.setMinimumHeight(285)

        cards = []
        for index, (title, text, color) in enumerate(
            (
                ("Fake board card 1", "drag me — blue surface", "#EAF4FF"),
                ("Fake board card 2", "drag me — green surface", "#ECF8F0"),
            ),
            start=1,
        ):
            panel = PanelCardView(f"gallery-panel-{index}", title)
            panel.set_surface(self._board_surface(text, color))
            panel.set_status("fake data · ready", error=False)
            cards.append(panel)
        board.set_cards(tuple(cards))

        echo = FluentLabel("order_committed: (drag a card to see the new tuple)")
        echo.setWordWrap(True)
        board.order_committed.connect(
            lambda order: self._echo_board_order(echo, tuple(order))
        )
        inner.addWidget(self._named("ConsoleBoardView", board))
        inner.addWidget(self._named("order_committed output", echo))
        return card

    @staticmethod
    def _board_surface(text: str, color: str) -> QtWidgets.QWidget:
        surface = FluentLabel(text)
        surface.setAlignment(QtCore.Qt.AlignCenter)
        surface.setMinimumHeight(105)
        surface.setStyleSheet(
            f"background: {color}; color: #323130; border: 1px solid #E1DFDD;"
        )
        return surface

    @staticmethod
    def _echo_board_order(label: FluentLabel, order: tuple[str, ...]) -> None:
        text = f"order_committed: {order}"
        label.setText(text)
        print(text, flush=True)

    def _build_form_section(self) -> QtWidgets.QWidget:
        card, inner = self._section("Headless FormSpec → Qt projection")
        spec = FormSpec(
            (
                FormFieldProps("label", "text", "Label", default="fake panel", required=True),
                FormFieldProps("count", "int", "Count", default=3, minimum=0, maximum=99),
                FormFieldProps(
                    "mode",
                    "choice",
                    "Mode",
                    default="preview",
                    choices=(
                        FormChoice("Preview", "preview"),
                        FormChoice("Live", "live"),
                    ),
                ),
            )
        )
        form = FluentParameterForm(
            spec,
            spec.default_values(),
            runtime=FormRuntimeContext(choice_names=lambda key: (f"{key}: fake",)),
        )
        form.setObjectName("GalleryForm")
        inner.addWidget(self._named("FluentParameterForm", form))
        return card

    def _build_gui_examples_section(self) -> QtWidgets.QWidget:
        if __package__:
            from .demo_console import populate as populate_console_demo
        else:
            from demo_console import populate as populate_console_demo

        def build_console_demo():
            # A complete GUI is a window behind a handle now; the gallery is
            # inside this package and may still compose a body for its tab.
            view = _console.TaskConsoleView()
            handle = _console.TaskConsoleHandle(None, view)
            # The handle owns the named intent relays.  A normal window returns
            # and retains it; the embedded gallery must do the same instead of
            # dropping the QObject while its relays are still connected.
            view._gallery_handle = handle
            # This page is a visual gallery, not a second interactive console
            # host.  Its real rows/cards remain complete; intent wiring is
            # exercised by the standalone demo and would have two owners here.
            populate_console_demo(handle, wire_intents=False)
            return view
        if __package__:
            from .demo_device_manager import populate as populate_device_demo
        else:
            from demo_device_manager import populate as populate_device_demo

        def build_device_demo():
            # A complete GUI is a window behind a handle now; the gallery is
            # inside this package and may still compose a body for its tab.
            import zlc_ui.device_manager as _dm

            view = _dm.DeviceManagerView()
            handle = _dm.DeviceManagerHandle(None, view)
            view._gallery_handle = handle
            populate_device_demo(handle)
            return view
        if __package__:
            from .demo_figure_viewer import populate as populate_figure_demo
        else:
            from demo_figure_viewer import populate as populate_figure_demo

        def build_figure_demo():
            # A complete GUI is a window behind a handle now; the gallery is
            # inside this package and may still compose a body for its tab.
            import zlc_ui.figure_viewer as _viewer

            view = _viewer.FigureViewerView()
            handle = _viewer.FigureViewerHandle(None, view)
            view._gallery_handle = handle
            populate_figure_demo(handle)
            return view
        if __package__:
            from .demo_pulse_editor import populate as populate_pulse_demo
        else:
            from demo_pulse_editor import populate as populate_pulse_demo

        def build_pulse_demo():
            # A complete GUI is a WINDOW now, reached through one handle, so
            # there is no body to hand out any more -- see zlc_ui/windows.py.
            # The gallery is inside this package and may still compose one for
            # a tab; the fake data stays in the demo, which is its one source.
            view = _pulse.PulseEditorView()
            # ``populate`` schedules zero-delay emits on the handle itself.
            # Keep that QObject alive for as long as its embedded page.
            handle = _pulse.PulseEditorHandle(None, view)
            view._gallery_handle = handle
            populate_pulse_demo(handle)
            return view

        card, inner = self._section("完整 GUI：TaskConsole / PulseEditor / FigureViewer / DeviceManager")
        tabs = FluentTabWidget()
        tabs.setMinimumHeight(scaled_px(620, minimum=500))
        for title, factory in (
            ("TaskConsole", build_console_demo),
            ("PulseEditor", build_pulse_demo),
            ("FigureViewer", build_figure_demo),
            ("DeviceManager", build_device_demo),
        ):
            page = factory()
            page.setObjectName(f"Gallery{title}")
            tabs.add_permanent_tab(page, title)
        inner.addWidget(self._named("FluentTabWidget · complete GUI navigation", tabs))
        return card

    def _build_tabs_section(self) -> QtWidgets.QWidget:
        card, inner = self._section("Permanent and closable tabs")
        tabs = FluentTabWidget()
        for name, text in (("Monitor", "A permanent tab"), ("Logic", "Another permanent tab")):
            tabs.add_permanent_tab(FluentLabel(text), name)
        edit = FluentLabel("A closable fake Edit page")
        tabs.add_closable_tab(edit, "Edit — fake panel")
        # The X emits; SOMETHING has to act on it, or the gallery demonstrates a
        # control that does nothing.  Removing the page and retiring it is
        # what a host does.
        def close_page(widget: QtWidgets.QWidget) -> None:
            tabs.removeTab(tabs.indexOf(widget))
            retire_widget(widget)

        tabs.tab_close_requested.connect(close_page)
        tabs.setMinimumHeight(130)
        inner.addWidget(self._named("FluentTabWidget", tabs))
        return card

    def _build_info_section(self) -> QtWidgets.QWidget:
        card, inner = self._section("InfoPane and published-signal legend")
        row = QtWidgets.QHBoxLayout()
        info = InfoPane(
            label_names=("Name", "Description", "Value"),
            tabs=(
                ("Summary", (("Name", "Fake figure"), ("Value", "42"))),
                ("State", (("State", "ready"),)),
                ("Owner", (("Owner", "fake device"),)),
                ("Raw", (("Payload", "{\\n  'fake': True\\n}"),)),
            ),
            path_label="Artifact",
            path_caption="Choose a fake artifact",
            initial_status="Fake data only.",
        )
        info.setMinimumHeight(310)
        row.addWidget(self._named("InfoPane", info), 1)
        legend = PublishedItemsLegend()
        legend.set_rows(
            (
                ("temperature", "1", "fake temperature"),
                ("smoothed", "1", "fake processed value"),
            )
        )
        row.addWidget(self._named("PublishedItemsLegend", legend), 1)
        inner.addLayout(row)
        return card

    @staticmethod
    def _log(value: str) -> None:
        print(f"gallery button: {value}", flush=True)


class GalleryWindow(FluentWindow):
    """The public gallery hosted by the reference frameless Fluent window."""

    def __init__(self) -> None:
        body = _GalleryBody()
        super().__init__(
            widget=body,
            title="zlc_ui control gallery — fake data only",
        )
        self.body = body
        self.scroll = body.scroll


def create_window(
    argv: list[str] | None = None,
    *,
    window_ratio: float = WINDOW_SCREEN_FRACTION,
) -> GalleryWindow:
    """Create and show the non-blocking gallery window for scripts or Notebook cells."""

    ensure_qt_app(["zlc-ui-gallery", *(argv or [])])
    return launch_qt_window(GalleryWindow, window_ratio=float(window_ratio))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="construct the gallery and exit")
    args = parser.parse_args(argv)
    window = create_window(argv)
    app = ensure_qt_app()
    platform = app.platformName().strip().lower()
    if args.once or os.environ.get("ZLC_UI_GALLERY_ONESHOT") == "1" or platform == "offscreen":
        app.processEvents()
        window.close()
        window.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()
        app.quit()
        return 0
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
