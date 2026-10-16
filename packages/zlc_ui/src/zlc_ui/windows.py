"""Opening a GUI: one call per window, and a handle rather than a widget tree.

Each of these returns a handle: signals to hear, methods to call, and no way
through to a QWidget, so no picture on screen has two owners.  The window's
lifecycle belongs here too -- the launcher, the shared screen-fit size, the
centring, the retention and the close handshake are one sequence, and every
window in this project goes through it.
"""

from __future__ import annotations

import math
import weakref
from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets, sip

from .fluent import (
    WINDOW_SCREEN_FRACTION,
    launch_fluent_window,
    open_fluent_window,
    screen_fit_window_size,
)
from .qt import ensure_qt_app


def _handle_window(handle: Any) -> QtWidgets.QWidget | None:
    window = handle if isinstance(handle, QtWidgets.QWidget) else getattr(handle, "_window", None)
    return window if isinstance(window, QtWidgets.QWidget) and not sip.isdeleted(window) else None


def associate_window(parent_handle: Any, child_handle: Any) -> None:
    """Declare a launched companion's UI parent without changing native ownership."""
    parent, child = _handle_window(parent_handle), _handle_window(child_handle)
    if parent is None or child is None:
        raise ValueError("window association needs two live GUI handles")
    parent = parent.window()
    while (reference := getattr(parent, "_zlc_shared_parent", None)) is not None:
        ancestor = reference()
        if ancestor is None or sip.isdeleted(ancestor):
            break
        parent = ancestor.window()
    child._zlc_shared_parent = weakref.ref(parent)


class SharedWindows(QtCore.QObject):
    """Mirror only one experiment's Qt trees; its existing owners still handle input."""

    def __init__(self, roots: Any) -> None:
        app = ensure_qt_app()
        super().__init__(app)
        self._app, self._roots = app, roots
        self._ids: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._next_id = 0
        self._windows: dict[str, QtWidgets.QWidget] = {}
        self._owners: dict[QtWidgets.QWidget, QtWidgets.QWidget] = {}
        self._dirty: set[QtWidgets.QWidget] = set()
        self._frames: dict[str, dict[str, Any]] = {}
        self._capturing = False
        self._closed = False
        self._controller: str | None = None
        self._control_epoch = 0
        self._pressed: QtWidgets.QWidget | None = None
        self._press_source: tuple[Any, ...] | None = None
        self._buttons = 0
        self._keys: dict[int, QtWidgets.QWidget] = {}
        self._pointer = QtCore.QPoint()
        self._hover: QtWidgets.QWidget | None = None
        self._native_dialogs = app.testAttribute(QtCore.Qt.AA_DontUseNativeDialogs)
        app.setAttribute(QtCore.Qt.AA_DontUseNativeDialogs, True)
        app.installEventFilter(self)

    def _check_thread(self) -> None:
        if QtCore.QThread.currentThread() != self._app.thread():
            raise RuntimeError("shared windows must be accessed on the Qt thread")
        if self._closed:
            raise RuntimeError("window sharing is closed")

    def theme(self) -> dict[str, str]:
        """Project the native Fluent tokens into the browser's small control strip."""
        from .fluent import fluent as f

        self._check_thread()
        return {
            "font-family": f'"{f.FONT}", sans-serif', "font-size": f"{f.fluent_font_size()}pt",
            "heading-weight": str(f.HEADING_CSS_WEIGHT), "control-height": f"{f.control_height()}px",
            "radius": f"{f._radius()}px", "row-gap": f"{f.row_gap()}px",
            "card-pad": f"{f.card_pad()}px", "window-pad": f"{f.window_pad()}px",
            "button-pad": f"{f.scaled_px(f.BUTTON_PADDING_H, minimum=6)}px",
            "field-pad": f"{f.field_text_inset()}px", "text": f.TEXT, "bg": f.BG,
            "surface": f.SURFACE, "muted": f.MUTED_INK, "button": f.BUTTON_TINT,
            "button-ink": f.HOVER, "hover": f._mixed(f.BUTTON_TINT, f.ACCENT, .25).name(),
            "pressed": f.SELECTED, "primary": f.ACCENT_STRONG, "primary-ink": f.SURFACE,
            "border": f.DIVIDER, "field-border": f.PLACEHOLDER, "focus": f.ACCENT,
            "danger": f.RED_DARK, "disabled-bg": f.DIVIDER, "disabled-ink": f.DISABLED_INK,
        }

    @staticmethod
    def _parent(window: QtWidgets.QWidget) -> QtWidgets.QWidget | None:
        for name in ("_zlc_shared_parent", "_toggle_anchor_ref"):
            reference = getattr(window, name, None)
            parent = None if reference is None else reference()
            if parent is not None and not sip.isdeleted(parent):
                return parent.window()
        anchor = getattr(window, "_dialog_anchor", None)
        if anchor is not None and not sip.isdeleted(anchor):
            return anchor.window()
        parent = window.parentWidget()
        return None if parent is None else parent.window()

    def _refresh(self) -> None:
        roots = {window for value in self._roots() if (window := _handle_window(value)) is not None}
        retained = set(getattr(self._app, "_zlc_retained_windows", ())) | roots
        candidates = set(self._app.topLevelWidgets()) | retained
        owners: dict[QtWidgets.QWidget, QtWidgets.QWidget] = {}
        for window in candidates:
            if (sip.isdeleted(window) or window.windowType() == QtCore.Qt.ToolTip
                    or getattr(window, "_zlc_close_committed", False)
                    or window not in retained and not window.isVisible()):
                continue
            current, visited = window, set()
            while current is not None and current not in visited:
                visited.add(current)
                if current in roots:
                    owners[window] = current
                    break
                current = self._parent(current)
        for window in owners:
            if window not in self._ids:
                self._next_id += 1
                self._ids[window] = f"w{self._next_id}"
                self._dirty.add(window)
        self._owners = owners
        self._windows = {self._ids[w]: w for w in owners if w.windowType() != QtCore.Qt.Popup}
        self._frames = {key: value for key, value in self._frames.items() if key in self._windows}
        self._dirty.intersection_update(owners)

    def catalog(self) -> dict[str, Any]:
        self._check_thread()
        self._refresh()
        modal = self._app.activeModalWidget()
        records = []
        for identity, window in self._windows.items():
            parent = self._parent(window)
            while parent is not None and parent.windowType() == QtCore.Qt.Popup:
                parent = self._parent(parent)
            records.append(dict(
                id=identity, title=window.windowTitle(), width=window.width(),
                height=window.height(), visible=window.isVisible() and not window.isMinimized(),
                modal=window.isModal(), parent=None if parent is None else self._ids.get(parent),
                blocked=modal is not None and modal is not window and not modal.isAncestorOf(window),
            ))
        return {"windows": sorted(records, key=lambda item: int(item["id"][1:])), "controller": self._controller}

    def _surface(self, identity: str) -> tuple[Any, Any, Any]:
        self._refresh()
        window = self._windows.get(str(identity))
        if window is None or not window.isVisible() or window.isMinimized():
            raise ValueError("the experiment window is no longer visible")
        origin = window.mapToGlobal(QtCore.QPoint())
        bounds = window.rect()
        popups = []
        for candidate in self._owners:
            if candidate.windowType() != QtCore.Qt.Popup or not candidate.isVisible():
                continue
            parent = self._parent(candidate)
            while parent is not None and parent.windowType() == QtCore.Qt.Popup:
                parent = self._parent(parent)
            if parent is window:
                position = candidate.mapToGlobal(QtCore.QPoint()) - origin
                rect = QtCore.QRect(position, candidate.size())
                bounds = bounds.united(rect)
                popups.append((candidate, rect))
        active = self._app.activePopupWidget()
        popups.sort(key=lambda item: (item[0] is active, int(self._ids[item[0]][1:])))
        return window, bounds, popups

    @staticmethod
    def _geometry(window: Any, bounds: Any, popups: Any) -> tuple[Any, ...]:
        return (window.width(), window.height(), bounds.getRect(), tuple(
            (id(popup), rect.getRect()) for popup, rect in popups
        ))

    def capture(self, identity: str, max_width: int = 1600, force: bool = False) -> dict[str, Any] | None:
        self._check_thread()
        window, bounds, popups = self._surface(identity)
        geometry = self._geometry(window, bounds, popups)
        previous = self._frames.get(identity, {})
        changed = previous.get("geometry") != geometry
        if (not force and not changed and previous.get("max_width") == max_width
                and not any(w in self._dirty for w in (window, *(p[0] for p in popups)))):
            return None
        scale = min(1.0, max(1, int(max_width)) / bounds.width())
        width, height = math.ceil(bounds.width() * scale), math.ceil(bounds.height() * scale)
        frame = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)
        frame.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(frame)
        painter.scale(scale, scale)
        self._capturing = True
        try:
            window.render(painter, -bounds.topLeft())
            for popup, rect in popups:
                popup.render(painter, rect.topLeft() - bounds.topLeft())
        finally:
            painter.end()
            self._capturing = False
        self._dirty.difference_update((window, *(p[0] for p in popups)))
        record = dict(
            id=identity, epoch=previous.get("epoch", 0) + int(changed),
            sequence=previous.get("sequence", 0) + 1, width=bounds.width(), height=bounds.height(),
            origin=[bounds.x(), bounds.y()], pixel_width=width, pixel_height=height,
        )
        self._frames[identity] = {**record, "geometry": geometry, "max_width": max_width}
        record["rgba"] = frame.constBits().asstring(frame.sizeInBytes())
        return record

    def take_control(self, client_id: str) -> dict[str, Any]:
        self._check_thread()
        if self._controller not in (None, client_id):
            raise ValueError("another browser controls the experiment")
        if self._controller != client_id:
            self._control_epoch += 1
        self._controller = client_id
        return {"controller": self._controller}

    def release_control(self, client_id: str) -> dict[str, Any]:
        self._check_thread()
        if self._controller == client_id:
            self._release_inputs()
            self._controller = None
        return {"controller": self._controller}

    def show(self, client_id: str, identity: str) -> dict[str, Any]:
        self._check_thread()
        if client_id != self._controller:
            raise ValueError("this browser does not control the experiment")
        self._refresh()
        window = self._windows.get(str(identity))
        if window is None:
            raise ValueError("the experiment window is no longer available")
        window.showNormal()
        return {"accepted": True}

    def _pressed_target(self) -> Any:
        target = self._pressed
        if target is not None and sip.isdeleted(target):
            self._pressed, self._buttons, target = None, 0, None
        if target is not None:
            grabber = QtWidgets.QWidget.mouseGrabber()
            if grabber is not None and grabber.window() is target.window():
                target = self._pressed = grabber
        return target

    def _release_inputs(self, *, cancel: bool = False) -> None:
        target = self._pressed_target()
        if target is not None:
            if isinstance(target, QtWidgets.QAbstractButton):
                target.setDown(False)
            if cancel:
                self._post(target, QtCore.QEvent(QtCore.QEvent.UngrabMouse), guard=False)
            for button in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
                if self._buttons & int(button):
                    self._buttons &= ~int(button)
                    self._mouse(target, QtCore.QEvent.MouseButtonRelease, self._pointer, button, QtCore.Qt.NoModifier, guard=False)
        for key, target in self._keys.items():
            if not sip.isdeleted(target):
                self._post(target, QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, key, QtCore.Qt.NoModifier), guard=False)
        self._pressed, self._press_source, self._buttons, self._keys = None, None, 0, {}

    @staticmethod
    def _modifiers(message: dict[str, Any]) -> QtCore.Qt.KeyboardModifiers:
        result = 0
        for name in message.get("modifiers", ()):
            result |= int({"Shift": QtCore.Qt.ShiftModifier, "Control": QtCore.Qt.ControlModifier,
                           "Alt": QtCore.Qt.AltModifier, "Meta": QtCore.Qt.MetaModifier}.get(name, QtCore.Qt.NoModifier))
        return QtCore.Qt.KeyboardModifiers(result)

    @staticmethod
    def _require_remote_input(target: Any) -> None:
        while target is not None:
            if target.property("zlc_local_only"):
                raise ValueError("Sharing permissions can only be changed on the experiment computer")
            target = target.parent()

    def _post(self, target: Any, event: Any, *, guard: bool = True) -> None:
        if guard:
            event._zlc_shared_control = (self._controller, self._control_epoch)
        self._app.postEvent(target, event)

    def _mouse(self, target: Any, kind: Any, position: Any, button: Any, modifiers: Any, *, guard: bool = True) -> None:
        local = QtCore.QPointF(target.mapFromGlobal(position))
        event = QtGui.QMouseEvent(kind, local, QtCore.QPointF(target.window().mapFromGlobal(position)),
                                 QtCore.QPointF(position), button, QtCore.Qt.MouseButtons(self._buttons), modifiers)
        self._post(target, event, guard=guard)

    def input(self, client_id: str, message: dict[str, Any]) -> dict[str, Any]:
        self._check_thread()
        if client_id != self._controller:
            raise ValueError("this browser does not control the experiment")
        if message["type"] in ("release", "cancel"):
            self._release_inputs(cancel=message["type"] == "cancel")
            return {"accepted": True}
        identity = str(message["window"])
        window, bounds, popups = self._surface(identity)
        frame = self._frames.get(identity)
        releasing = (message["type"] == "up" and self._pressed is not None
                     and self._press_source is not None and self._press_source[:2] == (identity, message.get("epoch")))
        if not releasing and (frame is None or message.get("epoch") != frame["epoch"]
                or frame["geometry"] != self._geometry(window, bounds, popups)):
            raise ValueError("window geometry changed; wait for its current image")
        modal = self._app.activeModalWidget()
        if modal is not None and modal is not window and not modal.isAncestorOf(window):
            raise ValueError("a modal dialog must be answered first")
        kind, modifiers = str(message["type"]), self._modifiers(message)
        if kind in ("key_down", "key_up", "text"):
            target = self._app.focusWidget()
            surfaces = {window, *(popup for popup, _rect in popups)}
            if target is None or target.window() not in surfaces:
                target = (popups[-1][0] if popups else window).focusWidget()
            if target is None or not target.isVisible() or not target.isEnabled():
                raise ValueError("no experimental input field has keyboard focus")
            self._require_remote_input(target)
            if kind == "text":
                event = QtGui.QInputMethodEvent()
                event.setCommitString(str(message.get("text", "")))
            else:
                key = str(message.get("key", ""))
                names = {"Enter": "Return", "Escape": "Escape", "Backspace": "Backspace",
                         "Tab": "Tab", "Delete": "Delete", "Insert": "Insert", "Home": "Home",
                         "End": "End", "PageUp": "PageUp", "PageDown": "PageDown",
                         "ArrowLeft": "Left", "ArrowRight": "Right", "ArrowUp": "Up", "ArrowDown": "Down",
                         "Shift": "Shift", "Control": "Control", "Alt": "Alt", "Meta": "Meta", " ": "Space"}
                code = getattr(QtCore.Qt, "Key_" + names.get(key, key), 0)
                if not code and len(key) == 1:
                    code = ord(key.upper()) if len(key.upper()) == 1 else 0
                if kind == "key_down":
                    self._keys[code] = target
                else:
                    target = self._keys.pop(code, target)
                    if sip.isdeleted(target):
                        return {"accepted": False, "reason": "the focused control was closed"}
                text = key if len(key) == 1 and not modifiers & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier) else ""
                event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress if kind == "key_down" else QtCore.QEvent.KeyRelease,
                                       code, modifiers, str(message.get("text", text)), bool(message.get("repeat", False)))
            self._post(target, event)
            return {"accepted": True}
        x, y = float(message["x"]), float(message["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("pointer coordinates must be finite")
        canvas_origin = self._press_source[2] if releasing else bounds.topLeft()
        point = QtCore.QPoint(round(x), round(y)) + canvas_origin
        position = window.mapToGlobal(point)
        self._pointer = position
        target = self._pressed_target()
        if target is None:
            surface = window
            for popup, rect in reversed(popups):
                if rect.contains(point):
                    surface = popup
                    break
            if popups and surface is window:
                if kind in ("down", "double"):
                    self._post(popups[-1][0], QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, modifiers))
                return {"accepted": False, "reason": "pointer is outside the open popup"}
            local = surface.mapFromGlobal(position)
            if not surface.rect().contains(local):
                return {"accepted": False, "reason": "pointer is outside the window"}
            target = surface.childAt(local) or surface
        if not target.isEnabled():
            return {"accepted": False, "reason": "the control is disabled"}
        self._require_remote_input(target)
        if kind == "move":
            if target is not self._hover:
                if self._hover is not None and not sip.isdeleted(self._hover):
                    self._post(self._hover, QtCore.QEvent(QtCore.QEvent.Leave))
                self._post(target, QtCore.QEvent(QtCore.QEvent.Enter))
                self._hover = target
            self._mouse(target, QtCore.QEvent.MouseMove, position, QtCore.Qt.NoButton, modifiers)
        elif kind in ("down", "up", "double"):
            button = {0: QtCore.Qt.LeftButton, 1: QtCore.Qt.MiddleButton, 2: QtCore.Qt.RightButton}.get(message.get("button", 0))
            if button is None:
                raise ValueError("unsupported mouse button")
            if kind == "up":
                self._buttons &= ~int(button)
            else:
                self._pressed = target
                self._press_source = (identity, message["epoch"], bounds.topLeft())
                self._buttons |= int(button)
                if target.focusPolicy() & QtCore.Qt.ClickFocus:
                    target.setFocus(QtCore.Qt.MouseFocusReason)
            event_type = {"down": QtCore.QEvent.MouseButtonPress, "up": QtCore.QEvent.MouseButtonRelease,
                          "double": QtCore.QEvent.MouseButtonDblClick}[kind]
            self._mouse(target, event_type, position, button, modifiers)
            if kind == "up" and button == QtCore.Qt.RightButton:
                self._post(target, QtGui.QContextMenuEvent(
                    QtGui.QContextMenuEvent.Mouse, target.mapFromGlobal(position), position, modifiers,
                ))
            if not self._buttons:
                self._pressed = None
                self._press_source = None
        elif kind in ("wheel", "pinch"):
            delta = QtCore.QPoint(round(-float(message.get("delta_x", 0))), round(-float(message.get("delta_y", 0))))
            pixel_delta = delta if message.get("deltaMode", 0) == 0 else QtCore.QPoint()
            scale = float(message.get("scale", 1.0))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("pinch scale must be positive and finite")
            while target is not None:
                local = QtCore.QPointF(target.mapFromGlobal(position))
                if kind == "pinch":
                    event = QtGui.QNativeGestureEvent(QtCore.Qt.ZoomNativeGesture, local,
                        QtCore.QPointF(target.window().mapFromGlobal(position)), QtCore.QPointF(position), scale - 1.0, 0, 0)
                else:
                    event = QtGui.QWheelEvent(local, QtCore.QPointF(position),
                                             pixel_delta, delta, QtCore.Qt.MouseButtons(self._buttons), modifiers,
                                             QtCore.Qt.ScrollUpdate if not pixel_delta.isNull() else QtCore.Qt.NoScrollPhase, False)
                event._zlc_shared_control = (self._controller, self._control_epoch)
                event.ignore()
                self._app.sendEvent(target, event)
                if event.isAccepted() or target.isWindow():
                    break
                target = target.parentWidget()
        else:
            raise ValueError("unsupported input event")
        return {"accepted": True}

    def eventFilter(self, watched: Any, event: Any) -> bool:
        control = getattr(event, "_zlc_shared_control", None)
        if control is not None and control != (self._controller, self._control_epoch):
            return True
        if self._closed or self._capturing or not isinstance(watched, QtWidgets.QWidget):
            return False
        kind = event.type()
        if kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.KeyPress, QtCore.QEvent.Wheel) and event.spontaneous():
            if self._controller is not None:
                self._refresh()
                if watched.window() in self._owners:
                    self._release_inputs()
                    self._controller = None
        if kind in (QtCore.QEvent.Paint, QtCore.QEvent.UpdateRequest, QtCore.QEvent.Resize,
                    QtCore.QEvent.Show, QtCore.QEvent.Hide, QtCore.QEvent.Move):
            window = watched.window()
            if window in self._owners:
                self._dirty.add(window)
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._check_thread()
        self._release_inputs()
        self._controller = None
        self._app.setAttribute(QtCore.Qt.AA_DontUseNativeDialogs, self._native_dialogs)
        self._windows.clear()
        self._owners.clear()
        self._frames.clear()
        self._dirty.clear()
        self._ids.clear()
        self._hover = None
        self._roots = lambda: ()
        self._closed = True
        self.deleteLater()


def _window_screenshot(window: Any, *, include_dialog: bool = False) -> Any:
    """Capture once on the Qt owner, through the same path for file and bytes."""
    if include_dialog:
        from PyQt5 import QtWidgets

        dialog = QtWidgets.QApplication.activeModalWidget()
        if dialog is not None and window.isAncestorOf(dialog):
            window = dialog
    pixmap = window.grab()
    if pixmap.isNull():
        raise OSError("the window has no image to capture")
    return pixmap


def save_window_screenshot(window: Any, path: str) -> str:
    """Save the complete visible GUI through the shared window boundary."""
    from pathlib import Path

    pixmap = _window_screenshot(window)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(path), "PNG"):
        raise OSError(f"could not save screenshot to {path!r}")
    return str(path)


def window_screenshot_png(window: Any, *, include_dialog: bool = False) -> bytes:
    """Return current GUI pixels, optionally showing its active modal instead.

    This does not resize, switch tabs, advance events or save an image. The
    caller schedules it on the Qt owner like every other window operation.
    """
    from PyQt5 import QtCore

    pixmap = _window_screenshot(window, include_dialog=include_dialog)
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.WriteOnly)
    if not pixmap.save(buffer, "PNG"):
        raise OSError("could not encode the window screenshot")
    return bytes(buffer.data())


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

    fraction = WINDOW_SCREEN_FRACTION if window_ratio is None else float(window_ratio)
    app = ensure_qt_app()
    view = DeviceControlView(spec, projection)

    def snug(window: Any) -> tuple[int, int]:
        # Snug fit: a control's field set is fixed for the window's life, so
        # the window takes the content's own height plus the chrome and half
        # the standard width, instead of a screen fraction of blank space.
        # Decided BEFORE the window is shown, so it never jumps.  It stays
        # resizable: this is the OPENING size, not a fixed one.
        standard = screen_fit_window_size(fraction)
        # The columns are aligned before the hint is read: a control's
        # width is the width of its agreed columns, not of its widgets'
        # first guesses.
        view.align_columns()
        hint = view.sizeHint()
        chrome = max(0, window.sizeHint().height() - hint.height())
        width = max(standard.width() // 2, hint.width())
        height = hint.height() + chrome
        screen = window.screen() or app.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            height = min(height, int(available.height() * 0.9))
            width = min(width, available.width())
        return width, height

    window = launch_fluent_window(view, title=str(title), size=snug, hide_on_close=True)
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
    path_base_dir: str = "",
    plot_surface: Any | None = None,
) -> Any:
    """Open the figure viewer and return the handle that drives it.

    ``plot_surface`` is the composition root's panel-widget policy, the
    same one ``open_task_console`` takes: called with a plotting host, it
    returns the QWidget that shows it.  The viewer's board is a console
    board, so it is handed the same staging widget and the board presents
    every panel front.  ``None`` keeps the host's own default widget.
    """

    from .figure_viewer.handle import FigureViewerHandle
    from .figure_viewer.view import FigureViewerView

    held: dict[str, Any] = {}

    def _body() -> FigureViewerView:
        view = FigureViewerView(path_base_dir=path_base_dir)
        held["view"] = view
        return view

    window = open_fluent_window(
        _body,
        title=str(title),
        window_ratio=(
            WINDOW_SCREEN_FRACTION if window_ratio is None else float(window_ratio)
        ),
    )
    return FigureViewerHandle(window, held["view"], plot_surface=plot_surface)


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
    "SharedWindows",
    "associate_window",
    "save_window_screenshot",
    "window_screenshot_png",
    "open_device_control",
    "open_device_manager",
    "open_figure_viewer",
    "open_pulse_editor",
    "open_task_console",
]
