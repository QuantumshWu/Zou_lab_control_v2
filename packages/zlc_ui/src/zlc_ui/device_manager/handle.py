"""The whole of what a device manager offers the outside: one object, no widgets.

Same rule as every window here.  This one is worth stating twice, because its
page is a list of cards a host could otherwise be tempted to build itself: the
cards, their forms and their status line are all this side of the wall, and
what crosses is which devices exist, what each is called, and the spec its
settings follow.
"""

from __future__ import annotations

from typing import Any

from PyQt5 import QtCore

from .view import DeviceManagerView


class DeviceManagerHandle(QtCore.QObject):
    """One device manager, as the outside sees it."""

    # -- the window ------------------------------------------------------
    close_requested = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    # -- what the operator asks for --------------------------------------
    device_add_requested = QtCore.pyqtSignal(str)
    device_remove_requested = QtCore.pyqtSignal(str)
    role_committed = QtCore.pyqtSignal(str, str)
    type_picked = QtCore.pyqtSignal(str, str)
    parameter_committed = QtCore.pyqtSignal(str, str)
    save_requested = QtCore.pyqtSignal()
    test_requested = QtCore.pyqtSignal()

    def __init__(self, window: Any, view: DeviceManagerView) -> None:
        super().__init__()
        self._window = window
        self._view = view
        for name in (
            "device_add_requested", "device_remove_requested", "role_committed",
            "type_picked", "parameter_committed", "save_requested",
            "test_requested",
        ):
            getattr(view, name).connect(getattr(self, name))
        if hasattr(view, "close_requested"):
            view.close_requested.connect(self.close_requested)
        if window is not None and hasattr(window, "closed"):
            window.closed.connect(self.closed)

    # ------------------------------------------------------------ the window

    def close(self) -> None:
        if hasattr(self._view, "finish_close"):
            self._view.finish_close()
        if self._window is not None:
            self._window.close()

    def is_visible(self) -> bool:
        target = self._window if self._window is not None else self._view
        return bool(target.isVisible())

    def window_size(self) -> tuple[int, int]:
        target = self._window if self._window is not None else self._view
        size = target.size()
        return int(size.width()), int(size.height())

    def window_title(self) -> str:
        target = self._window if self._window is not None else self._view
        return str(target.windowTitle())

    # ------------------------------------------------------------- the page

    def set_apparatus(self, name: str, dirty: bool, saved: bool) -> None:
        self._view.set_apparatus(name, dirty, saved)

    def set_device_choices(
        self,
        choices: tuple[tuple[str, str], ...],
        unavailable: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._view.set_device_choices(choices, unavailable)

    def set_devices(self, devices: tuple[tuple[str, str, str], ...]) -> None:
        self._view.set_devices(devices)

    def set_form_spec(
        self,
        instance_id: str,
        spec: Any,
        values: tuple[tuple[str, object], ...],
    ) -> None:
        self._view.set_form_spec(instance_id, spec, values)

    def read_values(self, instance_id: str) -> tuple[tuple[str, object], ...]:
        return self._view.read_values(instance_id)

    def show_status(self, text: str, severity: str) -> None:
        self._view.show_status(text, severity)


__all__ = ["DeviceManagerHandle"]
