"""Opening a GUI: one call per window, and a handle rather than a widget tree.

Each of these returns a handle: signals to hear, methods to call, and no way
through to a QWidget, so no picture on screen has two owners.  The window's
lifecycle belongs here too -- the launcher, the shared screen-fit size, the
centring, the retention and the close handshake are one sequence, and every
window in this project goes through it.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from collections import deque
import logging
import math
import secrets
import struct
import sys
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
    """Mirror one experiment's windows and their native dialogs."""

    input_failed = QtCore.pyqtSignal(str, str)
    owner_turn_requested = QtCore.pyqtSignal()

    def __init__(self, roots: Any) -> None:
        app = ensure_qt_app()
        super().__init__(app)
        self._app, self._roots = app, roots
        self._ids: dict[QtWidgets.QWidget | int, str] = {}
        self._next_id = 0
        self._windows: dict[str, QtWidgets.QWidget | int] = {}
        self._owners: dict[QtWidgets.QWidget | int, QtWidgets.QWidget | int] = {}
        self._dirty: set[QtWidgets.QWidget | int] = set()
        self._frames: dict[str, dict[str, Any]] = {}
        self._capture_image = QtGui.QImage()
        self._capturing = False
        self._closed = False
        self._controller: str | None = None
        self._control_epoch = 0
        self._pressed: QtWidgets.QWidget | int | None = None
        self._press_source: tuple[Any, ...] | None = None
        self._buttons = 0
        self._keys: set[int] = set()
        self._input_window: str | None = None
        self.control_enabled = sys.platform == "win32" and app.platformName() == "windows"
        self._hooks: list[Any] = []
        self._native_windows: set[int] = set()
        self._input_origins = deque(maxlen=512)
        self._input_event: Any = None
        self._input_origin: Any = (False, None)
        self._touch_press: Any = None
        if self.control_enabled:
            self._init_native_input()
        app.installEventFilter(self)

    def _init_native_input(self) -> None:
        self._win = ctypes.WinDLL("user32", use_last_error=True)
        pointer = ctypes.c_void_p
        signatures = {
            "SendInput": (wintypes.UINT, (wintypes.UINT, pointer, ctypes.c_int)),
            "GetMessageExtraInfo": (ctypes.c_size_t, ()),
            "GetForegroundWindow": (wintypes.HWND, ()),
            "SetForegroundWindow": (wintypes.BOOL, (wintypes.HWND,)),
            "WindowFromPoint": (wintypes.HWND, (wintypes.POINT,)),
            "ClientToScreen": (wintypes.BOOL, (wintypes.HWND, ctypes.POINTER(wintypes.POINT))),
            "ScreenToClient": (wintypes.BOOL, (wintypes.HWND, ctypes.POINTER(wintypes.POINT))),
            "GetClientRect": (wintypes.BOOL, (wintypes.HWND, ctypes.POINTER(wintypes.RECT))),
            "GetWindowRect": (wintypes.BOOL, (wintypes.HWND, ctypes.POINTER(wintypes.RECT))),
            "GetWindow": (wintypes.HWND, (wintypes.HWND, wintypes.UINT)),
            "GetAncestor": (wintypes.HWND, (wintypes.HWND, wintypes.UINT)),
            "GetWindowThreadProcessId": (wintypes.DWORD, (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))),
            "GetWindowTextW": (ctypes.c_int, (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)),
            "GetClassNameW": (ctypes.c_int, (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)),
            "GetWindowLongPtrW": (ctypes.c_ssize_t, (wintypes.HWND, ctypes.c_int)),
            "GetDpiForWindow": (wintypes.UINT, (wintypes.HWND,)),
            "GetGUIThreadInfo": (wintypes.BOOL, (wintypes.DWORD, pointer)),
            "GetComboBoxInfo": (wintypes.BOOL, (wintypes.HWND, pointer)),
            "IsWindowVisible": (wintypes.BOOL, (wintypes.HWND,)),
            "IsWindowEnabled": (wintypes.BOOL, (wintypes.HWND,)),
            "IsIconic": (wintypes.BOOL, (wintypes.HWND,)),
            "ShowWindow": (wintypes.BOOL, (wintypes.HWND, ctypes.c_int)),
            "EnumWindows": (wintypes.BOOL, (pointer, ctypes.c_ssize_t)),
            "EnumChildWindows": (wintypes.BOOL, (wintypes.HWND, pointer, ctypes.c_ssize_t)),
            "PrintWindow": (wintypes.BOOL, (wintypes.HWND, pointer, wintypes.UINT)),
            "GetSystemMetrics": (ctypes.c_int, (ctypes.c_int,)),
            "RegisterWindowMessageW": (wintypes.UINT, (wintypes.LPCWSTR,)),
            "PostMessageW": (wintypes.BOOL, (wintypes.HWND, wintypes.UINT, ctypes.c_size_t, ctypes.c_ssize_t)),
            "SendNotifyMessageW": (wintypes.BOOL, (wintypes.HWND, wintypes.UINT, ctypes.c_size_t, ctypes.c_ssize_t)),
            "SetWindowsHookExW": (pointer, (ctypes.c_int, pointer, pointer, wintypes.DWORD)),
            "UnhookWindowsHookEx": (wintypes.BOOL, (pointer,)),
            "CallNextHookEx": (ctypes.c_ssize_t, (pointer, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t)),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(self._win, name)
            function.restype, function.argtypes = restype, argtypes
        self._gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        for name, restype, argtypes in (
            ("CreateCompatibleDC", pointer, (pointer,)),
            ("CreateDIBSection", pointer, (pointer, pointer, wintypes.UINT, ctypes.POINTER(pointer), pointer, wintypes.DWORD)),
            ("SelectObject", pointer, (pointer, pointer)),
            ("DeleteObject", wintypes.BOOL, (pointer,)),
            ("DeleteDC", wintypes.BOOL, (pointer,)),
        ):
            function = getattr(self._gdi, name)
            function.restype, function.argtypes = restype, argtypes
        self._wide_input = ctypes.sizeof(pointer) == 8
        # Windows can retain only the low DWORD; keep private tags below its pointer signatures.
        self._input_mask = 0xFFFF0000
        self._input_tag = (secrets.randbelow(0xFF50) + 1) << 16
        root = next(window for value in self._roots() if (window := _handle_window(value)) is not None)
        self._dispatch_hwnd = int(root.winId())
        self._wake_message = self._win.RegisterWindowMessageW("ZLC SharedWindows owner turn")
        callback = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t)
        for kind in (13, 14):  # WH_KEYBOARD_LL and WH_MOUSE_LL run on this Qt thread.
            handler = callback(lambda code, message, data, keyboard=kind == 13:
                               self._native_hook(code, message, data, keyboard))
            handle = self._win.SetWindowsHookExW(kind, handler, None, 0)
            if not handle:
                for previous, _handler in self._hooks:
                    self._win.UnhookWindowsHookEx(previous)
                self._hooks.clear()
                raise ctypes.WinError(ctypes.get_last_error())
            self._hooks.append((handle, handler))
        self._controls = ctypes.WinDLL("comctl32", use_last_error=True)
        self._controls.SetWindowSubclass.argtypes = (wintypes.HWND, pointer, ctypes.c_size_t, ctypes.c_size_t)
        self._controls.SetWindowSubclass.restype = wintypes.BOOL
        self._controls.RemoveWindowSubclass.argtypes = (wintypes.HWND, pointer, ctypes.c_size_t)
        self._controls.RemoveWindowSubclass.restype = wintypes.BOOL
        self._controls.DefSubclassProc.argtypes = (wintypes.HWND, wintypes.UINT, ctypes.c_size_t, ctypes.c_ssize_t)
        self._controls.DefSubclassProc.restype = ctypes.c_ssize_t
        procedure = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                     ctypes.c_size_t, ctypes.c_ssize_t, ctypes.c_size_t, ctypes.c_size_t)
        self._window_proc = procedure(self._native_message)
        if not self._controls.SetWindowSubclass(self._dispatch_hwnd, self._window_proc, id(self), 0):
            for handle, _handler in self._hooks:
                self._win.UnhookWindowsHookEx(handle)
            self._hooks.clear()
            raise OSError("Windows could not attach the experiment input dispatcher")
        self._native_windows.add(self._dispatch_hwnd)

    def wake_owner(self) -> None:
        """Wake the same UI queue even inside Windows' native OLE drag loop."""
        send = (self._win.PostMessageW if QtCore.QThread.currentThread() == self._app.thread()
                else self._win.SendNotifyMessageW)
        if not send(self._dispatch_hwnd, self._wake_message, 0, 0):
            raise ctypes.WinError(ctypes.get_last_error())

    def _native_marker(self, *, cleanup: bool = False, touch_click: bool | None = None) -> int:
        return (self._input_tag | (1 if cleanup else ((self._control_epoch + 1) & 0x1FFF) << 3)
                | (0 if touch_click is None else 2 | (4 if touch_click else 0)))

    def _send_native(self, records: list[tuple[int, tuple[int, ...]]], *, cleanup: bool = False,
                     touch_click: bool | None = None) -> None:
        packets = []
        for kind, values in records:
            marker = self._native_marker(cleanup=cleanup,
                touch_click=touch_click if kind == 0 and values[3] & 0x2 else None)
            fmt = ("<llIII4xQ" if self._wide_input else "<llIIII") if kind == 0 else (
                "<HHII4xQ" if self._wide_input else "<HHIII")
            payload = struct.pack(fmt, *values, 0, marker)
            packets.append(struct.pack("<I4x" if self._wide_input else "<I", kind)
                           + payload.ljust(32 if self._wide_input else 24, b"\0"))
        if packets:
            buffer = ctypes.create_string_buffer(b"".join(packets))
            sent = self._win.SendInput(len(packets), buffer, 40 if self._wide_input else 28)
            if sent != len(packets):
                raise OSError("Windows could not deliver shared input; check the experiment window's privilege level")

    def _allowed_native_target(self, hwnd: int, point: Any = None) -> Any:
        selected = self._windows.get(self._input_window)
        if isinstance(selected, int):
            current = self._win.GetAncestor(hwnd, 2) if hwnd else None
            while current and current != selected:
                current = self._owners.get(current) if isinstance(self._owners.get(current), int) else self._win.GetWindow(current, 4)
            if current != selected or not self._win.IsWindowVisible(selected):
                raise ValueError("input is outside the selected experiment window")
            return hwnd
        target = QtWidgets.QWidget.find(hwnd) if hwnd else None
        if target is None or selected is None or sip.isdeleted(target) or not selected.isVisible():
            raise ValueError("input is outside the selected experiment window")
        current = target.window()
        while current is not selected and current is not None and current.windowType() == QtCore.Qt.Popup:
            current = self._parent(current)
        if current is not selected:
            raise ValueError("input is outside the selected experiment window")
        if point is not None:
            local, rect = wintypes.POINT(point.x, point.y), wintypes.RECT()
            self._win.ScreenToClient(hwnd, ctypes.byref(local))
            self._win.GetClientRect(hwnd, ctypes.byref(rect))
            position = QtCore.QPoint(round(local.x * target.width() / max(1, rect.right)),
                                    round(local.y * target.height() / max(1, rect.bottom)))
            target = target.childAt(position) or target
            self._require_remote_input(target)
        return target

    def _native_input_error(self, error: Exception) -> None:
        identity, self._controller = self._controller, None
        try:
            self._release_inputs(cancel=True)
        except Exception:
            logging.getLogger(__name__).exception("Could not release native shared input")
        if identity is not None:
            self.input_failed.emit(identity, str(error))

    def _native_hook(self, code: int, message: int, data: int, keyboard: bool) -> int:
        if code >= 0:
            try:
                extra_offset = 16 if keyboard else (24 if self._wide_input else 20)
                extra = ctypes.c_size_t.from_address(data + extra_offset).value & 0xFFFFFFFF
                own = extra & self._input_mask == self._input_tag
                if own and not extra & 1:
                    if self._controller is None or (extra & ~6) != self._native_marker():
                        return 1
                    if keyboard:
                        self._allowed_native_target(self._win.GetForegroundWindow())
                    else:
                        point = wintypes.POINT.from_address(data)
                        self._allowed_native_target(self._win.WindowFromPoint(point), point)
                elif not own and self._controller is not None:
                    # Do not release a key/button that the local operator has just taken over.
                    if keyboard:
                        key = wintypes.DWORD.from_address(data).value
                        self._keys.discard(key)
                    else:
                        self._buttons &= ~{0x201: 1, 0x202: 1, 0x204: 2, 0x205: 2, 0x207: 4, 0x208: 4}.get(message, 0)
                    self._controller = None
                    self._release_inputs(cancel=True)
            except Exception as error:
                self._native_input_error(error)
                return 1
        return self._win.CallNextHookEx(None, code, message, data)

    def _native_message(self, hwnd: int, message: int, wparam: int, lparam: int, identity: int, reference: int) -> int:
        if message == self._wake_message:
            if not self._closed:
                self.owner_turn_requested.emit()
            return 0
        if not self._capturing and hwnd in self._owners and message in (0xF, 0x5, 0x18, 0x47):
            self._dirty.add(hwnd)
        if message in (0x100, 0x101, 0x104, 0x105, 0x201, 0x203):
            extra = self._win.GetMessageExtraInfo() & 0xFFFFFFFF
            origin = None
            if extra & self._input_mask == self._input_tag:
                origin = ((self._controller, self._control_epoch)
                          if (extra & ~6) in (self._native_marker(), self._native_marker(cleanup=True)) else (None, -1))
            if message in (0x201, 0x203):
                click = bool(extra & 4) if origin is not None and extra & 2 else None
                self._input_origins.append((hwnd, int(QtCore.Qt.LeftButton), QtCore.QEvent.MouseButtonPress, origin, click))
            else:
                kind = QtCore.QEvent.KeyRelease if message in (0x101, 0x105) else QtCore.QEvent.KeyPress
                self._input_origins.append((hwnd, int(wparam), kind, origin, None))
        elif message == 0x82:  # WM_NCDESTROY
            self._native_windows.discard(hwnd)
            self._input_origins = deque((item for item in self._input_origins if item[0] != hwnd), maxlen=512)
        return self._controls.DefSubclassProc(hwnd, message, wparam, lparam)

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
        owners: dict[QtWidgets.QWidget | int, QtWidgets.QWidget] = {}
        native_dialog = False
        for window in candidates:
            if (sip.isdeleted(window) or window.windowType() == QtCore.Qt.ToolTip
                    or getattr(window, "_zlc_close_committed", False)
                    or window not in retained and not window.isVisible()):
                continue
            current, visited = window, set()
            while current is not None and current not in visited:
                visited.add(current)
                if current in roots:
                    if window.testAttribute(QtCore.Qt.WA_DontShowOnScreen):
                        native_dialog = True
                    else:
                        owners[window] = current
                    break
                current = self._parent(current)
        if self.control_enabled and native_dialog:
            handles = {int(window.winId()): root for window, root in owners.items()}
            process = wintypes.DWORD()
            self._win.GetWindowThreadProcessId(self._dispatch_hwnd, ctypes.byref(process))
            def native(hwnd, _context):
                pid = wintypes.DWORD()
                self._win.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if (pid.value == process.value and hwnd not in handles and self._win.IsWindowVisible(hwnd)
                        and self._win.GetWindowLongPtrW(hwnd, -16) & 0xC00000 == 0xC00000):
                    parent = self._win.GetWindow(hwnd, 4)
                    while parent and parent not in handles:
                        parent = self._win.GetWindow(parent, 4)
                    if parent in handles:
                        owners[int(hwnd)] = handles[parent]
                return True
            callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, ctypes.c_ssize_t)(native)
            self._win.EnumWindows(callback, 0)
            for window in tuple(owners):
                if not isinstance(window, int):
                    continue
                def combo_popup(hwnd, _context):
                    name = ctypes.create_unicode_buffer(64)
                    self._win.GetClassNameW(hwnd, name, len(name))
                    if name.value == "ComboBox":
                        info = ctypes.create_string_buffer(64 if self._wide_input else 52)
                        struct.pack_into("<I", info, 0, len(info))
                        if self._win.GetComboBoxInfo(hwnd, info):
                            popup = struct.unpack_from("<Q" if self._wide_input else "<I", info, 56 if self._wide_input else 48)[0]
                            if popup and self._win.IsWindowVisible(popup):
                                owners[popup] = window
                    return True
                callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, ctypes.c_ssize_t)(combo_popup)
                self._win.EnumChildWindows(window, callback, 0)
        self._ids = {window: identity for window, identity in self._ids.items() if window in owners}
        for window in owners:
            if window not in self._ids:
                self._next_id += 1
                self._ids[window] = f"w{self._next_id}"
                self._dirty.add(window)
        self._owners = owners
        self._windows = {self._ids[w]: w for w in owners
                         if (not isinstance(owners[w], int) if isinstance(w, int) else w.windowType() != QtCore.Qt.Popup)}
        if self.control_enabled:
            handles = {window if isinstance(window, int) else int(window.winId()) for window in owners}
            for hwnd in handles - self._native_windows:
                if not self._controls.SetWindowSubclass(hwnd, self._window_proc, id(self), 0):
                    raise OSError("Windows could not attach an experiment input window")
                self._native_windows.add(hwnd)
            removed = self._native_windows - handles
            for hwnd in removed:
                self._controls.RemoveWindowSubclass(hwnd, self._window_proc, id(self))
                self._native_windows.discard(hwnd)
            if removed:
                self._input_origins = deque((item for item in self._input_origins if item[0] in handles), maxlen=512)
        self._frames = {key: value for key, value in self._frames.items() if key in self._windows}
        self._dirty.intersection_update(owners)

    def catalog(self) -> dict[str, Any]:
        self._check_thread()
        self._refresh()
        modal = self._app.activeModalWidget()
        records = []
        for identity, window in self._windows.items():
            if isinstance(window, int):
                bounds, _dpr = self._native_geometry(window)
                title = ctypes.create_unicode_buffer(1024)
                self._win.GetWindowTextW(window, title, len(title))
                parent = self._win.GetWindow(window, 4)
                records.append(dict(id=identity, title=title.value, width=bounds.width(), height=bounds.height(),
                    visible=True, modal=True, parent=self._ids.get(parent) or self._ids[self._owners[window]],
                    blocked=not self._win.IsWindowEnabled(window)))
                continue
            parent = self._parent(window)
            while parent is not None and parent.windowType() == QtCore.Qt.Popup:
                parent = self._parent(parent)
            records.append(dict(
                id=identity, title=window.windowTitle(), width=window.width(),
                height=window.height(), visible=window.isVisible() and not window.isMinimized(),
                modal=window.isModal(), parent=None if parent is None else self._ids.get(parent),
                blocked=(modal is not None and modal is not window and not modal.isAncestorOf(window))
                        or (self.control_enabled and not self._win.IsWindowEnabled(int(window.winId()))),
            ))
        hints = self._app.styleHints()
        return {"windows": sorted(records, key=lambda item: int(item["id"][1:])), "controller": self._controller,
                "hold_ms": hints.mousePressAndHoldInterval(), "drag_distance": hints.startDragDistance()}

    def _surface(self, identity: str) -> tuple[Any, Any, Any]:
        self._refresh()
        window = self._windows.get(str(identity))
        if isinstance(window, int):
            bounds, _dpr = self._native_geometry(window)
            origin = wintypes.POINT()
            self._win.ClientToScreen(window, ctypes.byref(origin))
            popups = []
            for popup, parent in self._owners.items():
                if parent == window:
                    rect = wintypes.RECT()
                    self._win.GetWindowRect(popup, ctypes.byref(rect))
                    placed = QtCore.QRectF((rect.left-origin.x)/_dpr, (rect.top-origin.y)/_dpr,
                                          (rect.right-rect.left)/_dpr, (rect.bottom-rect.top)/_dpr)
                    bounds = bounds.united(placed)
                    popups.append((popup, placed))
            return window, bounds, popups
        if window is None or not window.isVisible() or window.isMinimized():
            raise ValueError("the experiment window is no longer visible")
        origin = window.mapToGlobal(QtCore.QPoint())
        bounds = window.rect()
        popups = []
        for candidate in self._owners:
            if isinstance(candidate, int) or candidate.windowType() != QtCore.Qt.Popup or not candidate.isVisible():
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
        if isinstance(window, int):
            return (window, bounds.getRect(), tuple((popup, rect.getRect()) for popup, rect in popups))
        return (window.width(), window.height(), bounds.getRect(), tuple(
            (id(popup), rect.getRect()) for popup, rect in popups
        ))

    def _native_geometry(self, hwnd: int) -> tuple[Any, float]:
        rect = wintypes.RECT()
        if (not self._win.IsWindowVisible(hwnd) or self._win.IsIconic(hwnd)
                or not self._win.GetClientRect(hwnd, ctypes.byref(rect)) or rect.right <= 0 or rect.bottom <= 0):
            raise ValueError("the experiment window is no longer visible")
        dpr = self._win.GetDpiForWindow(hwnd) / 96.
        if not dpr:
            raise ValueError("the experiment window is no longer visible")
        return QtCore.QRectF(0., 0., rect.right / dpr, rect.bottom / dpr), dpr

    def _native_image(self, hwnd: int, *, client: bool = True) -> Any:
        rect = wintypes.RECT()
        (self._win.GetClientRect if client else self._win.GetWindowRect)(hwnd, ctypes.byref(rect))
        width, height = rect.right - rect.left, rect.bottom - rect.top
        info = ctypes.create_string_buffer(struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0, 0, 0, 0, 0, 0))
        pixels = ctypes.c_void_p()
        dc = self._gdi.CreateCompatibleDC(None)
        bitmap = self._gdi.CreateDIBSection(dc, info, 0, ctypes.byref(pixels), None, 0)
        original = None
        self._capturing = True
        try:
            if not dc or not bitmap:
                raise ctypes.WinError(ctypes.get_last_error())
            original = self._gdi.SelectObject(dc, bitmap)
            if not self._win.PrintWindow(hwnd, dc, 3 if client else 2):
                raise OSError("Windows could not capture the selected system dialog")
            data = ctypes.string_at(pixels, width * height * 4)
            return QtGui.QImage(data, width, height, width * 4, QtGui.QImage.Format_RGB32).convertToFormat(QtGui.QImage.Format_RGBA8888)
        finally:
            self._capturing = False
            if original:
                self._gdi.SelectObject(dc, original)
            if bitmap:
                self._gdi.DeleteObject(bitmap)
            if dc:
                self._gdi.DeleteDC(dc)

    def _focused_input(self, window: Any, popups: Any) -> Any:
        target = self._app.focusWidget()
        if target is None or target.window() not in {window, *(popup for popup, _rect in popups)}:
            target = (popups[-1][0] if popups else window).focusWidget()
        return target

    def _input_targets(self, window: Any, bounds: Any, popups: Any) -> list[dict[str, Any]]:
        """Expose visible native text-input geometry, never field contents."""
        if isinstance(window, int):
            origin = wintypes.POINT()
            self._win.ClientToScreen(window, ctypes.byref(origin))
            dpr = self._win.GetDpiForWindow(window)/96.
            thread = self._win.GetWindowThreadProcessId(window, None)
            info = ctypes.create_string_buffer(72 if self._wide_input else 48)
            struct.pack_into("<I", info, 0, len(info))
            focused = 0
            if self._win.GetGUIThreadInfo(thread, info):
                focused = struct.unpack_from("<Q" if self._wide_input else "<I", info, 16 if self._wide_input else 12)[0]
            targets = []
            def visit(hwnd, _context):
                name = ctypes.create_unicode_buffer(64)
                self._win.GetClassNameW(hwnd, name, len(name))
                if name.value == "Edit" and self._win.IsWindowVisible(hwnd) and self._win.IsWindowEnabled(hwnd):
                    rect = wintypes.RECT()
                    self._win.GetWindowRect(hwnd, ctypes.byref(rect))
                    style = self._win.GetWindowLongPtrW(hwnd, -16)
                    targets.append(dict(id=f"native-{hwnd}",
                        rect=[(rect.left-origin.x)/dpr-bounds.x(), (rect.top-origin.y)/dpr-bounds.y(), (rect.right-rect.left)/dpr, (rect.bottom-rect.top)/dpr],
                        inputmode="numeric" if style & 0x2000 else "text", multiline=bool(style & 4),
                        readonly=bool(style & 0x800), password=bool(style & 0x20), focused=hwnd == focused))
                return True
            callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, ctypes.c_ssize_t)(visit)
            self._win.EnumChildWindows(window, callback, 0)
            return targets
        origin = window.mapToGlobal(bounds.topLeft())
        focused = self._focused_input(window, popups)
        targets = []
        for surface in (window, *(popup for popup, _rect in popups)):
            for kind in (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit):
                for widget in surface.findChildren(kind):
                    if widget.window() is not surface or not widget.isVisible() or not widget.isEnabled():
                        continue
                    content = widget if isinstance(widget, QtWidgets.QLineEdit) else widget.viewport()
                    rect = content.visibleRegion().boundingRect()
                    if rect.isEmpty():
                        continue
                    try:
                        self._require_remote_input(widget)
                    except ValueError:
                        continue
                    position = content.mapToGlobal(rect.topLeft()) - origin
                    context = widget.parentWidget() if isinstance(widget.parentWidget(), QtWidgets.QAbstractSpinBox) else widget
                    hints = context.inputMethodHints()
                    mode = "text"
                    for flag, value in ((QtCore.Qt.ImhDigitsOnly, "numeric"), (QtCore.Qt.ImhFormattedNumbersOnly, "decimal"),
                                        (QtCore.Qt.ImhDialableCharactersOnly, "tel"), (QtCore.Qt.ImhEmailCharactersOnly, "email"),
                                        (QtCore.Qt.ImhUrlCharactersOnly, "url")):
                        if hints & flag:
                            mode = value
                            break
                    targets.append(dict(id=str(id(widget)), rect=[position.x(), position.y(), rect.width(), rect.height()],
                        inputmode=mode, multiline=not isinstance(widget, QtWidgets.QLineEdit), readonly=widget.isReadOnly() or context.isReadOnly(),
                        password=isinstance(widget, QtWidgets.QLineEdit) and widget.echoMode() != QtWidgets.QLineEdit.Normal,
                        focused=(widget is focused or context is focused or widget.isAncestorOf(focused)) if focused is not None else False))
        return targets

    def describe(self, identity: str) -> dict[str, Any]:
        """Describe one logical input surface; native handles and popup pixels stay server-side."""
        from .console.panel_card_view import PanelCardView
        from .fluent import FluentOverlayFrame

        self._check_thread()
        window, bounds, popups = self._surface(identity)
        geometry = self._geometry(window, bounds, popups)
        previous = self._frames.get(identity, {})
        cached = previous.get("popup_images", {})
        images, overlays = {}, []
        popup_redrawn = False
        for popup, rect in popups:
            key = self._ids[popup]
            if isinstance(popup, int):
                dpr = self._win.GetDpiForWindow(popup)/96.
                physical = wintypes.RECT()
                self._win.GetWindowRect(popup, ctypes.byref(physical))
                width, height = physical.right-physical.left, physical.bottom-physical.top
                pixel_geometry = (width, height, dpr)
                image = cached.get(key)
                if image is None or image[0] != pixel_geometry or popup in self._dirty:
                    pixels = self._native_image(popup, client=False)
                    image = (pixel_geometry, width, height, pixels.constBits().asstring(pixels.sizeInBytes()))
                    self._dirty.discard(popup)
                    popup_redrawn = True
                images[key] = image
                overlays.append(dict(id=key, rect=list(rect.translated(-bounds.topLeft()).getRect()), native_dpr=dpr,
                                     pixel_width=width, pixel_height=height, rgba=image[3]))
                continue
            dpr = float(popup.devicePixelRatioF())
            pixel_geometry = (popup.width(), popup.height(), dpr, popup.logicalDpiX(), popup.logicalDpiY())
            image = cached.get(key)
            if image is None or image[0] != pixel_geometry or popup in self._dirty:
                width, height = math.ceil(popup.width() * dpr), math.ceil(popup.height() * dpr)
                pixels = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)
                pixels.setDevicePixelRatio(dpr)
                pixels.setDotsPerMeterX(round(popup.logicalDpiX() / .0254))
                pixels.setDotsPerMeterY(round(popup.logicalDpiY() / .0254))
                pixels.fill(QtCore.Qt.transparent)
                painter = QtGui.QPainter(pixels)
                self._capturing = True
                try:
                    popup.render(painter)
                finally:
                    painter.end()
                    self._capturing = False
                image = (pixel_geometry, width, height, pixels.constBits().asstring(pixels.sizeInBytes()))
                self._dirty.discard(popup)
                popup_redrawn = True
            images[key] = image
            placed = rect.translated(-bounds.topLeft())
            overlays.append(dict(id=key, rect=list(placed.getRect()), native_dpr=dpr,
                                 pixel_width=image[1], pixel_height=image[2], rgba=image[3]))
        covered = QtGui.QRegion()
        native = isinstance(window, int)
        for overlay in (() if native else window.findChildren(FluentOverlayFrame)):
            if overlay.isVisible() and overlay.window() is window:
                covered |= overlay.visibleRegion().translated(overlay.mapTo(window, QtCore.QPoint()))
        panels = []
        for card in (() if native else window.findChildren(PanelCardView)):
            surface = card.surface
            if surface is None or sip.isdeleted(surface) or surface.window() is not window or not surface.isVisible():
                continue
            position = surface.mapTo(window, QtCore.QPoint())
            visible = surface.visibleRegion().subtracted(covered.translated(-position))
            if visible.isEmpty():
                continue
            rect = QtCore.QRect(position, surface.size())
            available = QtGui.QRegion(surface.rect()).subtracted(visible).isEmpty() and window.rect().contains(rect)
            panels.append(dict(id=card.panel_id, rect=list(rect.translated(-bounds.topLeft()).getRect()),
                               hit_rects=[list(part.translated(position - bounds.topLeft()).getRect())
                                          for part in visible.rects()],
                               available=available, reason="" if available else
                               "Show the whole panel in the experiment window before opening its full-screen view."))
        root_bounds, native_dpr = self._native_geometry(window) if native else (window.rect(), float(window.devicePixelRatioF()))
        record = dict(
            id=identity, epoch=previous.get("epoch", 0) + int(previous.get("geometry") != geometry),
            width=bounds.width(), height=bounds.height(), origin=[bounds.x(), bounds.y()],
            inputs=self._input_targets(window, bounds, popups), hwnd=window if native else int(window.winId()),
            root_rect=[-bounds.x(), -bounds.y(), root_bounds.width(), root_bounds.height()],
            native_dpr=native_dpr,
            popups=tuple(overlays), panels=panels,
        )
        self._frames[identity] = {**previous, **record, "geometry": geometry, "popup_images": images}
        if popup_redrawn:
            self._frames[identity].pop("capture", None)
        return record

    def capture(self, identity: str, max_width: int | None = None, force: bool = False,
                *, root_only: bool = False) -> dict[str, Any] | None:
        """Capture at native density, optionally capped by a physical-pixel width."""
        description = self.describe(identity)
        if root_only:
            description = {**description, "width": description["root_rect"][2],
                           "height": description["root_rect"][3], "origin": [0, 0], "popups": ()}
        window = self._windows[identity]
        previous = self._frames[identity]
        scale = description["native_dpr"]
        if max_width is not None:
            scale = min(scale, max(1, int(max_width)) / description["width"])
        unchanged = previous.get("capture") == (description["epoch"], scale, root_only)
        if not isinstance(window, int) and not force and unchanged and window not in self._dirty:
            return None
        width, height = math.ceil(description["width"] * scale), math.ceil(description["height"] * scale)
        frame = self._capture_image
        if frame.width() != width or frame.height() != height:
            frame = self._capture_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)
        frame.setDevicePixelRatio(scale)
        if not isinstance(window, int):
            frame.setDotsPerMeterX(round(window.logicalDpiX() / .0254))
            frame.setDotsPerMeterY(round(window.logicalDpiY() / .0254))
        frame.fill(QtCore.Qt.white)
        root = self._native_image(window) if isinstance(window, int) else None
        painter = QtGui.QPainter(frame)
        self._capturing = True
        try:
            if root is not None:
                root.setDevicePixelRatio(description["native_dpr"])
                painter.drawImage(-QtCore.QPointF(*description["origin"]), root)
            else:
                window.render(painter, -QtCore.QPoint(*description["origin"]))
            for popup in description["popups"]:
                image = QtGui.QImage(popup["rgba"], popup["pixel_width"], popup["pixel_height"],
                                     popup["pixel_width"] * 4, QtGui.QImage.Format_RGBA8888)
                image.setDevicePixelRatio(popup["native_dpr"])
                painter.drawImage(QtCore.QRectF(*popup["rect"]), image)
        finally:
            painter.end()
            self._capturing = False
        self._dirty.discard(window)
        record = {key: description[key] for key in ("id", "epoch", "width", "height", "origin", "inputs")}
        record.update(sequence=previous.get("sequence", 0) + 1, pixel_width=width, pixel_height=height)
        previous.update(sequence=record["sequence"], capture=(description["epoch"], scale, root_only))
        record["rgba"] = frame.constBits().asstring(frame.sizeInBytes())
        return record

    def _focus_window(self, window: Any, popups: Any) -> None:
        hwnd = window if isinstance(window, int) else int(window.winId())
        foregrounds = {hwnd, *(popup if isinstance(popup, int) else int(popup.winId()) for popup, _ in popups)}
        if self._win.GetForegroundWindow() in foregrounds or not self._win.IsWindowEnabled(hwnd):
            return
        if not isinstance(window, int):
            window.raise_()
            window.activateWindow()
        self._win.SetForegroundWindow(hwnd)
        if self._win.GetForegroundWindow() not in foregrounds:
            raise ValueError("Windows could not focus the experiment window; activate it on the experiment computer")

    def take_control(self, client_id: str, identity: str, *, activate: bool = True) -> dict[str, Any]:
        self._check_thread()
        if not self.control_enabled:
            raise ValueError("Native shared input requires Windows; this host is view-only")
        if self._controller not in (None, client_id):
            raise ValueError("another browser controls the experiment")
        window, _bounds, popups = self._surface(identity)
        if self._controller != client_id or self._input_window != identity:
            self._release_inputs(cancel=True)
            self._control_epoch += 1
        if activate:
            self._focus_window(window, popups)
        self._input_window = identity
        self._controller = client_id
        return {"controller": self._controller}

    def release_control(self, client_id: str) -> dict[str, Any]:
        self._check_thread()
        if self._controller == client_id:
            self._controller = None
            self._release_inputs(cancel=True)
        return {"controller": self._controller}

    def show(self, client_id: str, identity: str) -> dict[str, Any]:
        self._check_thread()
        if client_id != self._controller:
            raise ValueError("this browser does not control the experiment")
        self._refresh()
        window = self._windows.get(str(identity))
        if window is None:
            raise ValueError("the experiment window is no longer available")
        if isinstance(window, int):
            self._win.ShowWindow(window, 9)
        else:
            window.showNormal()
        return {"accepted": True}

    def _release_inputs(self, *, cancel: bool = True) -> None:
        self._touch_press = None
        if not self.control_enabled:
            return
        target = self._pressed
        buttons, keys = self._buttons, self._keys
        self._pressed, self._press_source, self._buttons, self._keys = None, None, 0, set()
        if cancel and isinstance(target, QtWidgets.QWidget):
            QtGui.QDrag.cancel()
        if isinstance(target, QtWidgets.QWidget) and not sip.isdeleted(target):
            if isinstance(target, QtWidgets.QAbstractButton):
                target.setDown(False)
            if cancel:
                self._app.sendEvent(QtWidgets.QWidget.mouseGrabber() or target, QtCore.QEvent(QtCore.QEvent.UngrabMouse))
        records = [(0, (0, 0, 0, flags)) for button, flags in ((1, 0x4), (2, 0x10), (4, 0x40)) if buttons & button]
        records.extend((1, (key, 0, 0x2 | self._extended_key(key))) for key in keys)
        self._send_native(records, cleanup=True)

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

    @staticmethod
    def _extended_key(key: int) -> int:
        return int(key in (0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5C))

    @staticmethod
    def _virtual_key(key: str) -> int:
        names = {"Enter": 0x0D, "Escape": 0x1B, "Backspace": 0x08, "Tab": 0x09,
                 "Delete": 0x2E, "Insert": 0x2D, "Home": 0x24, "End": 0x23,
                 "PageUp": 0x21, "PageDown": 0x22, "ArrowLeft": 0x25, "ArrowRight": 0x27,
                 "ArrowUp": 0x26, "ArrowDown": 0x28, "Shift": 0xA0, "Control": 0xA2,
                 "Alt": 0xA4, " ": 0x20}
        if key in names:
            return names[key]
        if key.startswith("F") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
            return 0x70 + int(key[1:]) - 1
        return ord(key.upper()) if len(key) == 1 and key.isascii() and key.isalnum() else 0

    def _sync_modifiers(self, modifiers: Any) -> list[Any]:
        records = []
        for flag, key in ((QtCore.Qt.ShiftModifier, 0xA0), (QtCore.Qt.ControlModifier, 0xA2),
                          (QtCore.Qt.AltModifier, 0xA4)):
            held, desired = key in self._keys, bool(modifiers & flag)
            if held != desired:
                (self._keys.add if desired else self._keys.discard)(key)
                records.append((1, (key, 0, 0 if desired else 0x2)))
        return records

    def input(self, client_id: str, message: dict[str, Any]) -> dict[str, Any]:
        self._check_thread()
        if self._controller is None:
            return {"accepted": False, "reason": "remote control was released"}
        if client_id != self._controller:
            raise ValueError("this browser does not control the experiment")
        if message["type"] in ("release", "cancel"):
            self._release_inputs(cancel=True)
            return {"accepted": True}
        if message["type"] == "key_up":
            key = self._virtual_key(str(message.get("key", "")))
            if key in self._keys:
                self._keys.remove(key)
                self._send_native([(1, (key, 0, 0x2 | self._extended_key(key)))], cleanup=True)
            return {"accepted": True}
        identity = str(message["window"])
        if identity != self._input_window:
            raise ValueError("select the experiment window before sending input")
        try:
            window, bounds, popups = self._surface(identity)
        except ValueError:
            current = self._windows.get(identity)
            if current is not None and (self._win.IsWindowVisible(current) if isinstance(current, int)
                                        else current.isVisible() and not current.isMinimized()):
                raise
            self._release_inputs(cancel=True)
            return {"accepted": False, "reason": "the selected window is no longer visible"}
        frame = self._frames.get(identity)
        releasing = (message["type"] == "up" and self._pressed is not None
                     and self._press_source is not None and self._press_source[:2] == (identity, message.get("epoch")))
        if not releasing and (frame is None or message.get("epoch") != frame["epoch"]
                or frame["geometry"] != self._geometry(window, bounds, popups)):
            self._release_inputs(cancel=True)
            return {"accepted": False, "reason": "window geometry changed; wait for its current image"}
        native = isinstance(window, int)
        modal = self._app.activeModalWidget()
        hwnd = window if native else int(window.winId())
        if not self._win.IsWindowEnabled(hwnd) or (
                not native and modal is not None and modal is not window and not modal.isAncestorOf(window)):
            self._release_inputs(cancel=True)
            return {"accepted": False, "reason": "a modal dialog must be answered first"}
        kind, modifiers = str(message["type"]), self._modifiers(message)
        if kind in ("key_down", "text"):
            if modifiers & QtCore.Qt.MetaModifier:
                raise ValueError("Desktop-switching keys are not part of experiment-window sharing")
            target = None
            if kind == "text" and "preedit" in message and not native:
                target = self._focused_input(window, popups)
                if target is None or not target.isVisible() or not target.isEnabled():
                    raise ValueError("no experimental input field has keyboard focus")
                self._require_remote_input(target)
            if kind == "key_down":
                key = str(message.get("key", ""))
                if key == "Meta" or (modifiers & QtCore.Qt.AltModifier and key in ("Tab", "Escape")) or (
                        modifiers & QtCore.Qt.ControlModifier and key == "Escape"):
                    raise ValueError("Desktop-switching keys are not part of experiment-window sharing")
                code = self._virtual_key(key)
                if not code and len(key) != 1:
                    raise ValueError("unsupported keyboard key")
            self._focus_window(window, popups)
            self._allowed_native_target(self._win.GetForegroundWindow())
            if kind == "text":
                if target is not None:
                    event = QtGui.QInputMethodEvent(str(message["preedit"]), [])
                    event._zlc_shared_control = (self._controller, self._control_epoch)
                    self._app.sendEvent(target, event)
                records = self._sync_modifiers(modifiers)
                text = str(message.get("text", "")).encode("utf-16-le")
                for (unit,) in struct.iter_unpack("<H", text):
                    records.extend(((1, (0, unit, 0x4)), (1, (0, unit, 0x6))))
            else:
                records = self._sync_modifiers(modifiers)
                if not code:
                    for (unit,) in struct.iter_unpack("<H", key.encode("utf-16-le")):
                        records.extend(((1, (0, unit, 0x4)), (1, (0, unit, 0x6))))
                elif code not in (0xA0, 0xA2, 0xA4):
                    self._keys.add(code)
                    records.append((1, (code, 0, self._extended_key(code))))
            self._send_native(records)
            return {"accepted": True}
        x, y = float(message["x"]), float(message["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("pointer coordinates must be finite")
        canvas_origin = self._press_source[2] if releasing else bounds.topLeft()
        point = QtCore.QPoint(round(x), round(y)) + canvas_origin
        position = point if native else window.mapToGlobal(point)
        surface = window
        native_popup_rect = None
        for popup, rect in reversed(popups):
            if rect.contains(point):
                surface = popup
                native_popup_rect = rect if native else None
                break
        local = (point if native_popup_rect is None else point-native_popup_rect.topLeft()) if native else surface.mapFromGlobal(position)
        size = (self._native_geometry(window)[0] if native_popup_rect is None else
                QtCore.QRectF(0., 0., native_popup_rect.width(), native_popup_rect.height())) if native else surface.rect()
        if not size.contains(local):
            self._release_inputs(cancel=True)
            return {"accepted": False, "reason": "pointer left the experiment window; gesture cancelled"}
        target = surface if native else surface.childAt(local) or surface
        if not native and kind == "wheel" and message.get("touch") and self._pressed is not None and not self._buttons:
            if sip.isdeleted(self._pressed):
                self._release_inputs()
                return {"accepted": False, "reason": "the scrolling control was closed"}
            target = self._pressed
        if not native and not target.isEnabled() and kind not in ("move", "up"):
            return {"accepted": False, "reason": "the control is disabled"}
        if not native:
            self._require_remote_input(target)
        if kind in ("move", "down", "up") or native and kind == "wheel":
            touch_click = None
            touch = not native and message.get("pointerType") == "touch"
            hints = self._app.styleHints()
            last = self._touch_press
            if last is not None and (not touch or max(abs(position.x() - last[1].x()),
                    abs(position.y() - last[1].y())) > hints.touchDoubleTapDistance()):
                self._touch_press = last = None
            if kind == "down" and touch and message.get("button", 0) == 0:
                timestamp = float(message["timestamp"])
                if not math.isfinite(timestamp):
                    raise ValueError("touch timestamp must be finite")
                touch_click = bool(last is not None and last[0]() is target
                                   and 0 < timestamp - last[2] < hints.mouseDoubleClickInterval())
                self._touch_press = None if touch_click else (weakref.ref(target), position, timestamp)
            rect, hwnd = wintypes.RECT(), surface if native else int(surface.winId())
            if native_popup_rect is not None:
                self._win.GetWindowRect(hwnd, ctypes.byref(rect))
                physical = wintypes.POINT(rect.left+round(local.x()*(rect.right-rect.left)/size.width()),
                                         rect.top+round(local.y()*(rect.bottom-rect.top)/size.height()))
            else:
                self._win.GetClientRect(hwnd, ctypes.byref(rect))
                physical = wintypes.POINT(round(local.x() * rect.right / size.width()),
                                         round(local.y() * rect.bottom / size.height()))
                self._win.ClientToScreen(hwnd, ctypes.byref(physical))
            self._focus_window(window, popups)
            self._allowed_native_target(self._win.WindowFromPoint(physical), physical)
            get = self._win.GetSystemMetrics
            px = round((physical.x - get(76)) * 65535 / max(1, get(78) - 1))
            py = round((physical.y - get(77)) * 65535 / max(1, get(79) - 1))
            records = self._sync_modifiers(modifiers)
            records.append((0, (px, py, 0, 0xC001)))  # absolute virtual-desktop move
            if kind == "wheel":
                for key, flag, sign in (("delta_y", 0x800, -1), ("delta_x", 0x1000, 1)):
                    amount = float(message.get(key, 0))
                    if not math.isfinite(amount):
                        raise ValueError("wheel displacement must be finite")
                    if amount:
                        records.append((0, (0, 0, round(sign * amount) & 0xFFFFFFFF, flag)))
            elif kind != "move":
                button, down, up = {0: (1, 0x2, 0x4), 1: (4, 0x20, 0x40), 2: (2, 0x8, 0x10)}[message.get("button", 0)]
                if kind == "down":
                    self._pressed = target
                    self._press_source = (identity, message["epoch"], bounds.topLeft())
                    self._buttons |= button
                    records.append((0, (0, 0, 0, down)))
                elif self._buttons & button:
                    self._buttons &= ~button
                    records.append((0, (0, 0, 0, up)))
                    if not self._buttons:
                        self._pressed = self._press_source = None
            self._send_native(records, touch_click=touch_click)
        elif kind in ("wheel", "pinch"):
            if native:
                return {"accepted": False, "reason": "system dialogs have no plot zoom gesture"}
            delta = QtCore.QPoint(round(-float(message.get("delta_x", 0))), round(-float(message.get("delta_y", 0))))
            pixel_delta = delta if message.get("deltaMode", 0) == 0 else QtCore.QPoint()
            scale = float(message.get("scale", 1.0))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("pinch scale must be positive and finite")
            phase = {"begin": QtCore.Qt.ScrollBegin, "update": QtCore.Qt.ScrollUpdate,
                     "end": QtCore.Qt.ScrollEnd}.get(message.get("phase"), QtCore.Qt.NoScrollPhase)
            self._focus_window(window, popups)
            while target is not None:
                local = QtCore.QPointF(target.mapFromGlobal(position))
                if kind == "pinch":
                    event = QtGui.QNativeGestureEvent(QtCore.Qt.ZoomNativeGesture, local,
                        QtCore.QPointF(target.window().mapFromGlobal(position)), QtCore.QPointF(position), scale - 1.0, 0, 0)
                else:
                    event = QtGui.QWheelEvent(local, QtCore.QPointF(position),
                                             pixel_delta, delta, QtCore.Qt.MouseButtons(self._buttons), modifiers,
                                             phase, False)
                event._zlc_shared_control = (self._controller, self._control_epoch)
                event.ignore()
                self._app.sendEvent(target, event)
                if event.isAccepted() or target.isWindow():
                    break
                target = target.parentWidget()
            if kind == "wheel" and message.get("touch"):
                if phase == QtCore.Qt.ScrollEnd:
                    self._pressed = self._press_source = None
                elif event.isAccepted() and self._pressed is None:
                    self._pressed = target
                    self._press_source = (identity, message["epoch"], bounds.topLeft())
        else:
            raise ValueError("unsupported input event")
        return {"accepted": True}

    def _input_metadata(self, watched: Any, event: Any, code: int, kind: Any) -> tuple[Any, Any]:
        # Qt can reuse the same Python event wrapper for a later native input.
        identity = (event, event.type(), event.timestamp())
        if identity != self._input_event:
            lookup = (int(watched.window().winId()), code, kind)
            metadata = (False, None)
            for index, item in enumerate(self._input_origins):
                if item[:3] == lookup:
                    metadata = item[3:]
                    if event.type() != QtCore.QEvent.ShortcutOverride:
                        del self._input_origins[index]
                    break
            self._input_event, self._input_origin = identity, metadata
        return self._input_origin

    def eventFilter(self, watched: Any, event: Any) -> bool:
        control = getattr(event, "_zlc_shared_control", None)
        if control is not None and control != (self._controller, self._control_epoch):
            return True
        if self._closed or not isinstance(watched, QtWidgets.QWidget):
            return False
        kind = event.type()
        if self.control_enabled and event.spontaneous() and kind in (
            QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease, QtCore.QEvent.ShortcutOverride,
        ):
            # Qt drops native extra-info and retimestamps input; provenance crosses that boundary once.
            origin, _click = self._input_metadata(watched, event, int(event.nativeVirtualKey()),
                QtCore.QEvent.KeyPress if kind == QtCore.QEvent.ShortcutOverride else kind)
            if origin is not None and origin is not False and (
                    origin[0] is None or origin != (self._controller, self._control_epoch)):
                event.accept()
                return True
            if origin is not None and kind != QtCore.QEvent.KeyRelease:
                try:
                    self._require_remote_input(watched)
                except ValueError as error:
                    event.accept()
                    self._native_input_error(error)
                    return True
            return False
        if self.control_enabled and event.spontaneous() and kind in (
            QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease, QtCore.QEvent.MouseButtonDblClick,
            QtCore.QEvent.MouseMove, QtCore.QEvent.ContextMenu,
        ):
            if kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonDblClick):
                origin, touch_click = self._input_metadata(watched, event, int(event.button()), QtCore.QEvent.MouseButtonPress)
                if origin is not None and origin is not False and (
                        origin[0] is None or origin != (self._controller, self._control_epoch)):
                    return True
                if touch_click is not None:
                    expected = QtCore.QEvent.MouseButtonDblClick if touch_click else QtCore.QEvent.MouseButtonPress
                    if kind != expected:
                        replacement = QtGui.QMouseEvent(expected, event.localPos(), event.windowPos(), event.screenPos(),
                            event.button(), event.buttons(), event.modifiers(), QtCore.Qt.MouseEventSynthesizedByApplication)
                        replacement._zlc_shared_control = origin
                        self._app.sendEvent(watched, replacement)
                        return True
            extra = self._win.GetMessageExtraInfo() & 0xFFFFFFFF
            if extra & self._input_mask == self._input_tag and not extra & 1:
                if self._controller is None or (extra & ~6) != self._native_marker():
                    return True
            if self._controller is not None:
                try:
                    self._require_remote_input(watched)
                except ValueError as error:
                    self._native_input_error(error)
                    return True
        if control is not None:
            try:
                self._require_remote_input(watched)
            except ValueError as error:
                self._native_input_error(error)
                return True
        if not self._capturing and kind in (QtCore.QEvent.Paint, QtCore.QEvent.UpdateRequest, QtCore.QEvent.Resize,
                    QtCore.QEvent.Show, QtCore.QEvent.Hide, QtCore.QEvent.Move):
            window = watched.window()
            if window in self._owners:
                self._dirty.add(window)
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._check_thread()
        self._controller = None
        self._release_inputs(cancel=True)
        if self.control_enabled:
            for hwnd in self._native_windows:
                self._controls.RemoveWindowSubclass(hwnd, self._window_proc, id(self))
            self._native_windows.clear()
            QtCore.QTimer.singleShot(0, lambda: setattr(self, "_window_proc", None))
        self._input_origins.clear()
        self._input_event = None
        for handle, _handler in self._hooks:
            self._win.UnhookWindowsHookEx(handle)
        # A native callback may be executing this close; retain its ctypes trampoline until it returns.
        QtCore.QTimer.singleShot(0, self._hooks.clear)
        self._windows.clear()
        self._owners.clear()
        self._frames.clear()
        self._capture_image = QtGui.QImage()
        self._dirty.clear()
        self._ids.clear()
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
