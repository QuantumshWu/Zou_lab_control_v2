"""PyQt5 editors projected from the frontend-neutral display schema.

Importing :mod:`zlc_plot` does not import Qt.  Applications that need a Qt
parameter panel opt into this module, whose only plot-side inputs are
``DisplayDescription`` and ``ParameterControl`` values.  Validation remains in
the plot session: this layer merely emits the Python value represented by a
widget.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Callable

from .backends import _load_qt5_modules, ensure_qt5_application
from .session import DisplayDescription
from .ui import ControlKind, ParameterControl, parameter_controls, semantic_controls


_PANEL_CLASS: type[Any] | None = None


def _qt5_parameter_panel_class() -> type[Any]:
    """Return the lazily-created QWidget class used by ``Qt5ParameterPanel``."""

    global _PANEL_CLASS
    if _PANEL_CLASS is not None:
        return _PANEL_CLASS
    modules = _load_qt5_modules()
    QtCore, QtWidgets = modules.QtCore, modules.QtWidgets

    @contextmanager
    def _signals_blocked(*widgets: object) -> Iterator[None]:
        blockers = tuple(QtCore.QSignalBlocker(widget) for widget in widgets)
        try:
            yield
        finally:
            for blocker in blockers:
                blocker.unblock()

    def _choice_entry(choice: object, *, semantic: bool) -> tuple[object, str]:
        """Resolve one control choice without inventing semantic labels."""

        if semantic:
            if (
                not isinstance(choice, tuple)
                or len(choice) != 2
                or not isinstance(choice[1], str)
            ):
                raise TypeError(
                    "semantic controls must provide (value, label) choices"
                )
            return choice[0], choice[1]
        return choice, "(none)" if choice is None else str(choice)

    def _find_choice_index(editor: object, value: object) -> int:
        """Compare Python values directly; Qt QVariant identity is not stable."""

        for index in range(editor.count()):
            if editor.itemData(index) == value:
                return index
        return -1

    class _OptionalTextEditor(QtWidgets.QWidget):
        valueChanged = QtCore.pyqtSignal(object)

        def __init__(self, parent: object | None = None) -> None:
            super().__init__(parent)
            layout = QtWidgets.QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            self.text = QtWidgets.QLineEdit(self)
            self.automatic = QtWidgets.QCheckBox("Auto", self)
            layout.addWidget(self.text, 1)
            layout.addWidget(self.automatic)
            self.text.editingFinished.connect(self._text_edited)
            self.automatic.toggled.connect(self._automatic_toggled)

        def _text_edited(self) -> None:
            if not self.automatic.isChecked():
                self.valueChanged.emit(self.text.text())

        def _automatic_toggled(self, automatic: bool) -> None:
            self.text.setEnabled(not automatic)
            self.valueChanged.emit(None if automatic else self.text.text())

        def set_value(self, value: object) -> None:
            automatic = value is None
            with _signals_blocked(self.text, self.automatic):
                self.automatic.setChecked(automatic)
                self.text.setEnabled(not automatic)
                if value is not None:
                    self.text.setText(str(value))

    class _OptionalNumberEditor(QtWidgets.QWidget):
        valueChanged = QtCore.pyqtSignal(object)

        def __init__(
            self,
            spin_box: object,
            parent: object | None = None,
        ) -> None:
            super().__init__(parent)
            layout = QtWidgets.QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            self.spin_box = spin_box
            self.spin_box.setParent(self)
            self.automatic = QtWidgets.QCheckBox("Auto", self)
            layout.addWidget(self.spin_box, 1)
            layout.addWidget(self.automatic)
            self.spin_box.valueChanged.connect(self._number_edited)
            self.automatic.toggled.connect(self._automatic_toggled)

        def _number_edited(self, value: object) -> None:
            if not self.automatic.isChecked():
                self.valueChanged.emit(value)

        def _automatic_toggled(self, automatic: bool) -> None:
            self.spin_box.setEnabled(not automatic)
            value = None if automatic else self.spin_box.value()
            self.valueChanged.emit(value)

        def set_value(self, value: object) -> None:
            automatic = value is None
            with _signals_blocked(self.spin_box, self.automatic):
                self.automatic.setChecked(automatic)
                self.spin_box.setEnabled(not automatic)
                if value is not None:
                    self.spin_box.setValue(value)

    class _Qt5ParameterPanel(QtWidgets.QWidget):
        """A schema-driven PyQt5 form with no plot validation of its own.

        ``parameterEdited`` emits ``(name, value)``.  Connect it to
        ``RasterPlotHost.set_parameter`` (or any equivalent application
        command).  Accepted display callbacks can be fed back through
        :meth:`set_values` to keep the form synchronized.
        """

        parameterEdited = QtCore.pyqtSignal(str, object)
        semanticEdited = QtCore.pyqtSignal(str, object)

        def __init__(
            self,
            description: DisplayDescription,
            parent: object | None = None,
        ) -> None:
            if not isinstance(description, DisplayDescription):
                raise TypeError("description must be DisplayDescription")
            ensure_qt5_application()
            super().__init__(parent)
            root_layout = QtWidgets.QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            self._display_group = QtWidgets.QGroupBox("Display", self)
            self._layout = QtWidgets.QFormLayout(self._display_group)
            self._layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            self._semantic_group = QtWidgets.QGroupBox("Semantics", self)
            self._semantic_layout = QtWidgets.QFormLayout(self._semantic_group)
            self._semantic_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            self._error_label = QtWidgets.QLabel(self)
            self._error_label.setWordWrap(True)
            self._error_label.setStyleSheet("color: #b00020;")
            self._error_label.hide()
            root_layout.addWidget(self._display_group)
            root_layout.addWidget(self._semantic_group)
            root_layout.addWidget(self._error_label)
            self._editors: dict[str, object] = {}
            self._setters: dict[str, Callable[[object], None]] = {}
            self._semantic_editors: dict[str, object] = {}
            self._semantic_setters: dict[str, Callable[[object], None]] = {}
            self._signature: tuple[object, ...] = ()
            self._semantic_signature: tuple[object, ...] = ()
            self.set_description(description)

        @property
        def parameter_names(self) -> tuple[str, ...]:
            return tuple(self._editors)

        def editor(self, name: str) -> object:
            if not isinstance(name, str):
                raise TypeError("parameter name must be a string")
            try:
                return self._editors[name]
            except KeyError as error:
                raise KeyError(f"unknown parameter editor: {name!r}") from error

        @property
        def semantic_names(self) -> tuple[str, ...]:
            return tuple(self._semantic_editors)

        def semantic_editor(self, name: str) -> object:
            if not isinstance(name, str):
                raise TypeError("semantic field name must be a string")
            try:
                return self._semantic_editors[name]
            except KeyError as error:
                raise KeyError(f"unknown semantic editor: {name!r}") from error

        def set_description(
            self,
            description: DisplayDescription,
        ) -> None:
            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError("set_description must run on the Qt owner thread")
            if not isinstance(description, DisplayDescription):
                raise TypeError("description must be DisplayDescription")
            controls = parameter_controls(
                description.parameter_schema,
                description.display_state.values,
                choice_overrides=description.parameter_choices,
            )
            semantic = semantic_controls(description.semantics)
            signature = tuple(
                (
                    control.name,
                    control.label,
                    control.kind,
                    control.allow_none,
                    control.choices,
                    control.minimum,
                    control.maximum,
                    control.step,
                )
                for control in controls
            )
            if signature != self._signature:
                self._rebuild(
                    controls,
                    self._layout,
                    self._editors,
                    self._setters,
                    self.parameterEdited,
                )
                self._signature = signature
            else:
                self._sync_controls(controls, self._setters)

            semantic_signature = tuple(
                (
                    control.name,
                    control.label,
                    control.kind,
                    control.allow_none,
                    control.choices,
                    control.value,
                )
                for control in semantic
            )
            if semantic_signature != self._semantic_signature:
                self._rebuild(
                    semantic,
                    self._semantic_layout,
                    self._semantic_editors,
                    self._semantic_setters,
                    self.semanticEdited,
                )
                self._semantic_signature = semantic_signature
            else:
                self._sync_controls(semantic, self._semantic_setters)

        def set_values(self, values: Mapping[str, object]) -> None:
            """Synchronize accepted values without emitting edit signals."""

            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError("set_values must run on the Qt owner thread")
            if not isinstance(values, Mapping):
                raise TypeError("values must be a mapping")
            for name, setter in self._setters.items():
                if name not in values:
                    raise KeyError(f"display state is missing parameter {name!r}")
                setter(values[name])

        def set_semantic_values(self, values: Mapping[str, object]) -> None:
            """Synchronize semantic editors without emitting rebuild signals."""

            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError("set_semantic_values must run on the Qt owner thread")
            if not isinstance(values, Mapping):
                raise TypeError("values must be a mapping")
            for name, setter in self._semantic_setters.items():
                if name not in values:
                    raise KeyError(f"semantic state is missing field {name!r}")
                setter(values[name])

        def set_error(self, message: str | None) -> None:
            """Show a semantic command failure beside the controls."""

            if QtCore.QThread.currentThread() != self.thread():
                raise RuntimeError("set_error must run on the Qt owner thread")
            if message is not None and not isinstance(message, str):
                raise TypeError("panel error must be text or None")
            text = "" if message is None else message.strip()
            self._error_label.setText(text)
            self._error_label.setVisible(bool(text))

        def _rebuild(
            self,
            controls: tuple[ParameterControl, ...],
            form_layout: object,
            editors: dict[str, object],
            setters: dict[str, Callable[[object], None]],
            signal: object,
        ) -> None:
            self.setUpdatesEnabled(False)
            try:
                while form_layout.rowCount():
                    form_layout.removeRow(0)
                editors.clear()
                setters.clear()
                for control in controls:
                    editor, setter = self._make_editor(control, signal)
                    editors[control.name] = editor
                    setters[control.name] = setter
                    form_layout.addRow(control.label, editor)
                    setter(control.value)
            finally:
                self.setUpdatesEnabled(True)

        def _sync_controls(
            self,
            controls: tuple[ParameterControl, ...],
            setters: Mapping[str, Callable[[object], None]],
        ) -> None:
            for control in controls:
                setters[control.name](control.value)

        def _make_editor(
            self,
            control: ParameterControl,
            signal: object,
        ) -> tuple[object, Callable[[object], None]]:
            if control.kind is ControlKind.BOOLEAN:
                editor = QtWidgets.QCheckBox(self)
                editor.toggled.connect(
                    lambda value, name=control.name: signal.emit(
                        name, bool(value)
                    )
                )
                return editor, lambda value: self._set_checked(editor, value)

            if control.kind is ControlKind.CHOICE:
                editor = QtWidgets.QComboBox(self)
                if control.allow_none:
                    editor.addItem("(none)", None)
                for choice in control.choices:
                    value, label = _choice_entry(
                        choice,
                        semantic=control.semantic,
                    )
                    if value is None and control.allow_none:
                        continue
                    editor.addItem(label, value)
                if control.value is not None and _find_choice_index(editor, control.value) < 0:
                    if control.semantic:
                        raise RuntimeError(
                            f"semantic value for {control.name!r} is absent from choices"
                        )
                    value, label = _choice_entry(
                        control.value,
                        semantic=control.semantic,
                    )
                    editor.addItem(label, value)
                editor.currentIndexChanged.connect(
                    lambda _index, widget=editor, name=control.name: (
                        signal.emit(name, widget.currentData())
                    )
                )
                return editor, lambda value: self._set_choice(
                    editor, value, semantic=control.semantic
                )

            if control.kind is ControlKind.INTEGER:
                spin = QtWidgets.QSpinBox(self)
                spin.setKeyboardTracking(False)
                minimum = (
                    -2_147_483_648
                    if control.minimum is None
                    else int(control.minimum)
                )
                maximum = (
                    2_147_483_647
                    if control.maximum is None
                    else int(control.maximum)
                )
                spin.setRange(minimum, maximum)
                spin.setSingleStep(
                    1 if control.step is None else max(1, int(control.step))
                )
                if control.allow_none:
                    editor = _OptionalNumberEditor(spin, self)
                    editor.valueChanged.connect(
                        lambda value, name=control.name: (
                            signal.emit(name, value)
                        )
                    )
                    return editor, editor.set_value
                spin.valueChanged.connect(
                    lambda value, name=control.name: (
                        signal.emit(name, int(value))
                    )
                )
                return spin, lambda value: self._set_number(spin, value)

            if control.kind is ControlKind.NUMBER:
                spin = QtWidgets.QDoubleSpinBox(self)
                spin.setKeyboardTracking(False)
                spin.setDecimals(12)
                spin.setRange(
                    -1.0e100 if control.minimum is None else control.minimum,
                    1.0e100 if control.maximum is None else control.maximum,
                )
                spin.setSingleStep(0.1 if control.step is None else control.step)
                if control.allow_none:
                    editor = _OptionalNumberEditor(spin, self)
                    editor.valueChanged.connect(
                        lambda value, name=control.name: (
                            signal.emit(name, value)
                        )
                    )
                    return editor, editor.set_value
                spin.valueChanged.connect(
                    lambda value, name=control.name: signal.emit(
                        name, float(value)
                    )
                )
                return spin, lambda value: self._set_number(spin, value)

            if control.kind is ControlKind.TEXT:
                if control.allow_none:
                    editor = _OptionalTextEditor(self)
                    editor.valueChanged.connect(
                        lambda value, name=control.name: (
                            signal.emit(name, value)
                        )
                    )
                    return editor, editor.set_value
                editor = QtWidgets.QLineEdit(self)
                editor.editingFinished.connect(
                    lambda widget=editor, name=control.name: (
                        signal.emit(name, widget.text())
                    )
                )
                return editor, lambda value: self._set_text(editor, value)
            raise TypeError(f"unsupported control kind: {control.kind!r}")

        @staticmethod
        def _set_checked(editor: object, value: object) -> None:
            with _signals_blocked(editor):
                editor.setChecked(bool(value))

        @staticmethod
        def _set_choice(
            editor: object,
            value: object,
            *,
            semantic: bool = False,
        ) -> None:
            with _signals_blocked(editor):
                index = _find_choice_index(editor, value)
                if index < 0:
                    if semantic:
                        raise RuntimeError(
                            "semantic value is absent from its canonical choices"
                        )
                    editor.addItem("(none)" if value is None else str(value), value)
                    index = editor.count() - 1
                editor.setCurrentIndex(index)

        @staticmethod
        def _set_number(editor: object, value: object) -> None:
            with _signals_blocked(editor):
                editor.setValue(value)

        @staticmethod
        def _set_text(editor: object, value: object) -> None:
            with _signals_blocked(editor):
                editor.setText(str(value))

    _Qt5ParameterPanel.__name__ = "Qt5ParameterPanel"
    _Qt5ParameterPanel.__qualname__ = "Qt5ParameterPanel"
    _Qt5ParameterPanel.__module__ = __name__
    _PANEL_CLASS = _Qt5ParameterPanel
    return _PANEL_CLASS




_BOUND_CLASS: type[Any] | None = None


def _qt5_bound_controls_class() -> type[Any]:
    """Return the lazily-created QWidget class used by ``Qt5PlotControls``."""

    global _BOUND_CLASS
    if _BOUND_CLASS is not None:
        return _BOUND_CLASS
    modules = _load_qt5_modules()
    QtCore, QtWidgets = modules.QtCore, modules.QtWidgets
    panel_class = _qt5_parameter_panel_class()

    class _Qt5PlotControls(QtWidgets.QWidget):
        """The parameter panel, wired to the host it describes.

        Every embedder used to wire this itself: read the description, route
        ``parameterEdited`` to ``set_parameter``, compose a candidate spec for
        ``semanticEdited``, submit it, and put the accepted description back on
        the panel.  That is four copies of one rule, and each copy had to keep
        a shadow of the spec and the schema to compose against -- state the
        session owns and can change underneath them.

        Here it is once.  What a semantic edit produces is composed by the
        session (``apply_semantic``); this only forwards what the operator did
        and shows what came back, including the refusal.
        """

        operationFinished = QtCore.pyqtSignal(object)

        def __init__(self, host: object, parent: object = None) -> None:
            super().__init__(parent)
            for name in ("describe_display", "set_parameter", "apply_semantic"):
                if not callable(getattr(host, name, None)):
                    raise TypeError(f"plot controls need a host offering {name}()")
            self._host = host
            self._request_serial = 0
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self._layout = layout
            self.panel = None
            self._placeholder = QtWidgets.QLabel("Loading plot controls…", self)
            layout.addWidget(self._placeholder)
            self.operationFinished.connect(
                self._operation_finished,
                type=QtCore.Qt.QueuedConnection,
            )
            self._submit(host.describe_display())

        def _submit(self, pending: object) -> None:
            """Resolve a worker operation later on the Qt owner thread.

            A refusal that only reaches a log leaves the editor showing a value
            the plot never took, which is the "nothing happened" feel.
            """

            add_done = getattr(pending, "add_done_callback", None)
            if not callable(add_done):
                raise TypeError("plot control operations must return Future-like values")
            self._request_serial += 1
            serial = self._request_serial

            def completed(future: object) -> None:
                try:
                    description = future.result().value
                    outcome = (serial, description, None)
                except Exception as error:  # noqa: BLE001 - delivered to Qt
                    outcome = (serial, None, error)
                try:
                    self.operationFinished.emit(outcome)
                except RuntimeError:
                    # The dialog closed before its worker operation finished.
                    return

            add_done(completed)

        @QtCore.pyqtSlot(object)
        def _operation_finished(self, resolved: object) -> None:
            serial, description, error = resolved
            if serial != self._request_serial:
                return
            if error is not None:
                if self.panel is None:
                    self._placeholder.setText(str(error))
                else:
                    self.panel.set_error(str(error))
                return
            if self.panel is None:
                self.panel = panel_class(description, self)
                self.panel.parameterEdited.connect(self._parameter_edited)
                self.panel.semanticEdited.connect(self._semantic_edited)
                self._layout.replaceWidget(self._placeholder, self.panel)
                self._placeholder.hide()
                self._placeholder.deleteLater()
            else:
                self.panel.set_description(description)
                self.panel.set_error(None)

        def _parameter_edited(self, name: str, value: object) -> None:
            self._submit(self._host.set_parameter(name, value))

        def _semantic_edited(self, name: str, value: object) -> None:
            self._submit(self._host.apply_semantic(name, value))

    _Qt5PlotControls.__name__ = "Qt5PlotControls"
    _Qt5PlotControls.__qualname__ = "Qt5PlotControls"
    _Qt5PlotControls.__module__ = __name__
    _BOUND_CLASS = _Qt5PlotControls
    return _BOUND_CLASS


def edit_plot_display(host: object, parent: object = None, *, title: str = "Display") -> object:
    """Open one plot's controls in a modal window, and return the controls.

    A window rather than a docked pane, because the caller is usually a board of
    many plots and only one of them is being adjusted.  It is here, not in the
    application, because a dialog that titles and sizes itself is the sort of
    chrome every window would otherwise reinvent slightly differently.
    """

    modules = _load_qt5_modules()
    QtWidgets = modules.QtWidgets
    controls_class = _qt5_bound_controls_class()

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(str(title))
    layout = QtWidgets.QVBoxLayout(dialog)
    controls = controls_class(host, dialog)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(controls)
    layout.addWidget(scroll)
    dialog.resize(480, 520)
    dialog.exec_()
    return controls


def __getattr__(name: str) -> object:
    """Resolve the optional Qt classes as their real types on explicit access."""

    if name == "Qt5ParameterPanel":
        resolved = _qt5_parameter_panel_class()
    elif name == "Qt5PlotControls":
        resolved = _qt5_bound_controls_class()
    else:
        raise AttributeError(name)
    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"Qt5ParameterPanel", "Qt5PlotControls"})


__all__ = ["Qt5ParameterPanel", "Qt5PlotControls", "edit_plot_display"]
