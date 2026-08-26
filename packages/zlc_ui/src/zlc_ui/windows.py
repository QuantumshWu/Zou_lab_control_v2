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
    owner=None,
) -> Any:
    """Open one projection-driven generic control without exposing Qt.

    ``owner`` is for a frame that has no life of its own: a console opens
    this to show the settings of a card it is already showing, so the two
    belong together and the desktop must not stack anything between them.
    An APPLICATION window is not that.  The pulse editor, the figure
    viewer, the device manager and the console itself each open on their
    own, outlive whatever happened to launch them and are moved, raised
    and closed on their own -- tying one to another would mean minimising
    a console takes an editor down with it, which is a relationship
    neither of them has.
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
        owner=owner,
        title=str(title),
        window_ratio=(
            WINDOW_SCREEN_FRACTION if window_ratio is None else float(window_ratio)
        ),
    )
    return DeviceControlHandle(window, held["view"])


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
