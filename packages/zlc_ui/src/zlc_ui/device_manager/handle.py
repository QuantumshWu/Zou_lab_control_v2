"""The whole of what a device manager offers the outside: one object, no widgets.

Same rule as every window here.  This one is worth stating twice, because its
page is a list of cards a host could otherwise be tempted to build itself: the
cards, their forms and their status line are all this side of the wall, and
what crosses is which devices exist, what each is called, the spec its settings
follow, and the narrow document/session intents the operator submits.
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
    template_selected = QtCore.pyqtSignal(str)
    discovery_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    save_requested = QtCore.pyqtSignal()
    save_as_requested = QtCore.pyqtSignal()
    cancel_requested = QtCore.pyqtSignal()
    lifecycle_requested = QtCore.pyqtSignal()

    _FORWARDED = (
        "device_add_requested",
        "device_remove_requested",
        "role_committed",
        "type_picked",
        "parameter_committed",
        "template_selected",
        "discovery_requested",
        "load_requested",
        "save_requested",
        "save_as_requested",
        "cancel_requested",
        "lifecycle_requested",
    )

    def __init__(self, window: Any, view: DeviceManagerView) -> None:
        super().__init__()
        self._window = window
        self._view = view
        for name in self._FORWARDED:
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

    def hide(self) -> None:
        """Hide this top-level window without retiring its shared session."""

        target = self._window if self._window is not None else self._view
        target.hide()

    def restore(self) -> None:
        """Restore the existing top-level window; do not create another one."""

        target = self._window if self._window is not None else self._view
        if hasattr(target, "showNormal"):
            target.showNormal()
        else:
            target.show()
        if hasattr(target, "raise_"):
            target.raise_()
        if hasattr(target, "activateWindow"):
            target.activateWindow()

    def show(self) -> None:
        """Compatibility spelling for hosts that do not reserve ``show``."""

        self.restore()

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
        choices: tuple[tuple[str, str, str], ...],
        unavailable: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._view.set_device_choices(choices, unavailable)

    def set_devices(
        self,
        devices: tuple[tuple[str, str, str, str], ...],
    ) -> None:
        self._view.set_devices(devices)

    def set_templates(self, templates: tuple[tuple[str, str], ...]) -> None:
        self._view.set_templates(templates)

    def set_installation_source(self, source: str | None, detail: str) -> None:
        self._view.set_installation_source(source, detail)

    def set_loaded_devices(
        self,
        devices: tuple[tuple[str, str, str], ...],
    ) -> None:
        self._view.set_loaded_devices(devices)

    def set_discovery_enabled(
        self,
        enabled: bool,
        reason: str = "No installed device type declares discovery",
    ) -> None:
        self._view.set_discovery_enabled(enabled, reason)

    def set_lifecycle(
        self,
        text: str,
        *,
        enabled: bool,
        active: bool,
        busy: bool = False,
    ) -> None:
        self._view.set_lifecycle(
            text,
            enabled=enabled,
            active=active,
            busy=busy,
        )

    def set_form_spec(
        self,
        instance_id: str,
        spec: Any,
        values: tuple[tuple[str, object], ...],
    ) -> None:
        self._view.set_form_spec(instance_id, spec, values)

    def read_values(self, instance_id: str) -> tuple[tuple[str, object], ...]:
        return self._view.read_values(instance_id)

    def ask_open_path(self, caption: str, start_dir: str, filter: str) -> str:
        from PyQt5 import QtWidgets

        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self._view,
            str(caption),
            str(start_dir),
            str(filter),
        )
        return str(path)

    def ask_save_path(self, caption: str, start_dir: str, filter: str) -> str:
        from PyQt5 import QtWidgets

        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self._view,
            str(caption),
            str(start_dir),
            str(filter),
        )
        return str(path)

    def show_status(self, text: str, severity: str) -> None:
        self._view.show_status(text, severity)


__all__ = ["DeviceManagerHandle"]
