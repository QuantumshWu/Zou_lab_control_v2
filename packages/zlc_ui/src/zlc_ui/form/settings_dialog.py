"""A modal form over one declared spec: show these settings, return what was set.

Generic on purpose.  It knows a ``FormSpec`` and nothing about what the settings
mean, which is what lets the same dialog serve a logic node, a device, or
anything else that declares its parameters -- and what keeps the window that
opens it from having to build a dialog by hand.
"""

from __future__ import annotations

from collections.abc import Mapping

from PyQt5 import QtWidgets

from ..fluent import fluent_widget_stylesheet, window_pad
from .form import FormSpec
from .qt_form import FluentParameterForm


__all__ = ["SettingsDialog", "edit_values"]


class SettingsDialog(QtWidgets.QDialog):
    """One form, OK and Cancel."""

    def __init__(
        self,
        spec: FormSpec,
        values: Mapping[str, object] | None = None,
        parent=None,
        *,
        title: str = "Settings",
    ) -> None:
        super().__init__(parent)
        if not isinstance(spec, FormSpec):
            raise TypeError("a settings dialog needs a FormSpec")
        self.setWindowTitle(str(title))
        self.setStyleSheet(fluent_widget_stylesheet())
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        margin = window_pad(1)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(window_pad(0.5))
        self.form = FluentParameterForm(spec, dict(values or {}))
        layout.addWidget(self.form)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def values(self) -> dict[str, object]:
        return self.form.read_all()


def edit_values(
    spec: FormSpec,
    values: Mapping[str, object] | None = None,
    parent=None,
    *,
    title: str = "Settings",
) -> dict[str, object] | None:
    """Ask for these settings; None when the operator declines.

    None rather than the unchanged values: "cancelled" and "changed nothing"
    look identical to a caller that only gets a dict back, and they are not the
    same -- one of them should leave a running thing alone.
    """

    dialog = SettingsDialog(spec, values, parent, title=title)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.values()
