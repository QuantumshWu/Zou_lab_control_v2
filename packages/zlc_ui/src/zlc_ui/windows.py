"""Opening a GUI: one call per window, and a handle rather than a widget tree.

What the outside used to receive was the window's own widgets, so it could
reach a sub-view and push a value into one panel of one page.  Nothing stopped
it, so it happened, and the picture on screen ended up with two owners: the
delay column kept showing rows that Hide Off had already taken out of the cards
beside it.  A composition root that can hold a widget will sooner or later
assemble a UI, which is the one job it does not have.

So each of these returns a handle: signals to hear, methods to call, and no way
through to a QWidget.  The window's lifecycle belongs here too -- the launcher,
the shared screen-fit size, the centring, the retention and the close handshake
are one sequence, and every window in this project goes through it.
"""

from __future__ import annotations

from typing import Any

from .fluent import WINDOW_SCREEN_FRACTION, open_fluent_window


def open_device_control(
    *,
    title: str,
    spec: Any,
    projection: Any,
    window_ratio: float | None = None,
) -> Any:
    """Open one projection-driven generic control without exposing Qt.

    A plain top-level like every launcher window (the window law lives on
    :func:`zlc_ui.fluent.fluent.launch_fluent_window`), opened at half the
    standard width and its content's own height, still resizable.
    """

    from .device_manager.handle import DeviceControlHandle
    from .device_manager.view import DeviceControlView

    held: dict[str, Any] = {}

    def _body() -> DeviceControlView:
        view = DeviceControlView(spec, projection)
        held["view"] = view
        return view

    window = open_fluent_window(
        _body,
        title=str(title),
        window_ratio=(
            WINDOW_SCREEN_FRACTION if window_ratio is None else float(window_ratio)
        ),
    )
    view = held["view"]
    # Snug fit: a control's field set is fixed for the window's life, so the
    # window takes the content's own height (plus the chrome) and half the
    # standard width, instead of a big screen fraction full of blank space.
    # It stays resizable -- this is the OPENING size, not a fixed one.
    chrome = max(0, window.height() - view.height())
    hint = view.sizeHint()
    width = max(window.width() // 2, hint.width())
    height = hint.height() + chrome
    screen = window.screen()
    available = None if screen is None else screen.availableGeometry()
    if available is not None:
        height = min(height, int(available.height() * 0.9))
        width = min(width, available.width())
    window.resize(width, height)
    if available is not None:
        frame = window.frameGeometry()
        frame.moveCenter(available.center())
        window.move(frame.topLeft())
    return DeviceControlHandle(window, view)


def open_pulse_editor(
    *,
    title: str = "PulseGUI@Zou lab",
    window_ratio: float | None = None,
) -> Any:
    """Open the pulse editor and return the handle that drives it."""

    from .pulse.editor_view import PulseEditorView
    from .pulse.handle import PulseEditorHandle

    held: dict[str, Any] = {}

    def _body() -> PulseEditorView:
        # Built inside the launcher so the QApplication, High-DPI flags and
        # font are settled before any widget constructor runs -- the ordering
        # the launcher exists to own.
        view = PulseEditorView()
        held["view"] = view
        return view

    window = open_fluent_window(
        _body,
        title=str(title),
        window_ratio=(
            WINDOW_SCREEN_FRACTION if window_ratio is None else float(window_ratio)
        ),
    )
    return PulseEditorHandle(window, held["view"])


def open_figure_viewer(
    *,
    title: str = "FigureViewer@Zou lab",
    window_ratio: float | None = None,
) -> Any:
    """Open the figure viewer and return the handle that drives it."""

    from .figure_viewer.handle import FigureViewerHandle
    from .figure_viewer.view import FigureViewerView

    held: dict[str, Any] = {}

    def _body() -> FigureViewerView:
        view = FigureViewerView()
        held["view"] = view
        return view

    window = open_fluent_window(
        _body,
        title=str(title),
        window_ratio=(
            WINDOW_SCREEN_FRACTION if window_ratio is None else float(window_ratio)
        ),
    )
    return FigureViewerHandle(window, held["view"])


def open_device_manager(
    *,
    title: str = "DeviceManager@Zou lab",
    window_ratio: float | None = None,
) -> Any:
    """Open the device manager and return the handle that drives it."""

    from .device_manager.handle import DeviceManagerHandle
    from .device_manager.view import DeviceManagerView

    held: dict[str, Any] = {}

    def _body() -> DeviceManagerView:
        view = DeviceManagerView()
        held["view"] = view
        return view

    window = open_fluent_window(
        _body,
        title=str(title),
        window_ratio=(
            WINDOW_SCREEN_FRACTION if window_ratio is None else float(window_ratio)
        ),
    )
    return DeviceManagerHandle(window, held["view"])


def open_task_console(
    *,
    title: str = "TaskConsole@Zou lab",
    window_ratio: float | None = None,
    plot_surface: Any | None = None,
) -> Any:
    """Open the task console and return the handle that drives it.

    ``plot_surface`` is the composition root's panel-widget policy: called
    with a plotting host, it returns the QWidget that shows it (the console
    passes a staging widget so the board can present same-shot groups
    atomically).  ``None`` keeps the host's own default widget.
    """

    from .console.handle import TaskConsoleHandle
    from .console.task_console_view import TaskConsoleView

    held: dict[str, Any] = {}

    def _body() -> TaskConsoleView:
        view = TaskConsoleView()
        held["view"] = view
        return view

    window = open_fluent_window(
        _body,
        title=str(title),
        window_ratio=(
            WINDOW_SCREEN_FRACTION if window_ratio is None else float(window_ratio)
        ),
    )
    return TaskConsoleHandle(window, held["view"], plot_surface=plot_surface)


__all__ = [
    "open_device_control",
    "open_device_manager",
    "open_figure_viewer",
    "open_pulse_editor",
    "open_task_console",
]
