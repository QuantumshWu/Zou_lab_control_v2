"""The v1-shaped console panel card.

The board owns placement; this widget owns only the titled Fluent surface and
the small ``Setting`` affordance from v1.  Signal/size/update controls remain
available through the settings popup for the lightweight presenter API, but
they do not add an invented toolbar to the card face.
"""

from __future__ import annotations

from collections.abc import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_ui.board import DEFAULT_PANEL_SIZE, PANEL_SIZES
from zlc_ui.fluent import (
    ACCENT,
    CARD_PAD,
    CARD_TITLE_PX,
    FluentButton,
    FluentComboBox,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPopup,
    FluentScrollArea,
    FluentSettingsPopupAnchor,
    FluentStatusDot,
    GREY,
    ORANGE,
    RED,
    scaled_px,
    signals_blocked,
)
from zlc_ui.form.form import FormChoice, FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm

from ._panel_projection import (
    decode_parameter_value,
    interval_form_field,
    panel_state_document,
    parameter_fields,
    parameter_form_spec,
    parameter_form_values,
    signal_form_runtime,
)


def _set_interaction(surface: object | None, enabled: bool) -> None:
    """Let the operator drag on a surface, when the surface has dragging.

    Duck-typed on purpose: a plot widget owns an interaction gate, and a label
    standing in for one in a demo does not.  Asking is how one switch can drive
    both without either knowing about the other.
    """

    gate = getattr(surface, "set_interaction_enabled", None)
    if callable(gate):
        gate(bool(enabled))


class PanelCardView(FluentGroupBox):
    """A v1 titled card with a replaceable QWidget surface."""

    signal_picked = QtCore.pyqtSignal(str)
    size_picked = QtCore.pyqtSignal(str)
    title_committed = QtCore.pyqtSignal(str)
    remove_requested = QtCore.pyqtSignal()
    edit_requested = QtCore.pyqtSignal()
    state_changed = QtCore.pyqtSignal(object)
    dropped = QtCore.pyqtSignal(tuple)
    drag_started = QtCore.pyqtSignal(tuple)
    drag_moved = QtCore.pyqtSignal(tuple)
    geometry_changed = QtCore.pyqtSignal()

    def __init__(self, panel_id: str, title: str = "Panel", parent=None) -> None:
        super().__init__(str(title), parent)
        self.panel_id = str(panel_id)
        self._base_title = str(title)
        self._surface: QtWidgets.QWidget | None = None
        self._settings_popup: FluentPopup | None = None
        self._settings_anchor: FluentSettingsPopupAnchor | None = None
        self._settings_drag_handle: FluentLabel | None = None
        self._settings_scroll: FluentScrollArea | None = None
        self._settings_body: QtWidgets.QWidget | None = None
        self._settings_form: FluentParameterForm | None = None
        self._groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
        self._overlay_groups: tuple[
            tuple[str, tuple[tuple[str, str], ...]], ...
        ] = ()
        self._signal_runtime = signal_form_runtime(self._choice_groups)
        # A replace-only projection of the Workbench state.  It is view state,
        # never a second saved panel configuration or an authority of its own.
        self._state_projection = panel_state_document(
            {
                "signal": "",
                "kind": "",
                "size": DEFAULT_PANEL_SIZE,
                "interval_ms": 100,
                "title": str(title),
            }
        )
        self._parameter_surface: Mapping[str, object] = {}
        self._interval_choices: tuple[int, ...] = ()
        self._drag_offset: QtCore.QPoint | None = None
        self._settings_drag_offset: QtCore.QPoint | None = None

        # This is copied from v1 PanelCard: the title strip is supplied by
        # FluentGroupBox; the body is exactly CARD_PAD / 2 px / CARD_PAD with
        # no invented header, footer, status row, or stretch spacer.
        holder = QtWidgets.QVBoxLayout(self)
        holder.setContentsMargins(
            CARD_PAD,
            scaled_px(2),
            CARD_PAD,
            CARD_PAD,
        )
        holder.setSpacing(0)
        self._surface_layout = holder
        self._placeholder = FluentLabel("Pick a signal in Setting")
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        holder.addWidget(self._placeholder)

        # The v1 card has one visible command on its title strip.  It is a
        # child overlay rather than a layout item, so it never changes the
        # panel data rectangle or the packer's card size.
        #: Whether what this card shows will deliver again.  A live panel
        #: redraws on a beat and can be taken off the board; a saved figure
        #: does neither, and must not offer controls for both.
        self._live = True
        self._editing_enabled = True
        self.settings_button = FluentButton("Setting", color=GREY)
        self.settings_button.setParent(self)
        self.settings_button.setFixedSize(
            scaled_px(74, minimum=64),
            scaled_px(26, minimum=22),
        )
        self.settings_button.clicked.connect(self._open_settings)

        self.status_dot = FluentStatusDot(size=12, parent=self)
        self.status_dot.hide()
        self.status_label = FluentLabel("", parent=self)
        self.status_label.hide()
        # FigureViewer also embeds this card and exposes these lightweight
        # handles through its own port. TaskConsole state never reads them.
        self.title_edit = FluentLineEdit(str(title), parent=self)
        self.title_edit.hide()
        self.title_edit.editingFinished.connect(self._commit_title)
        self.signal_combo = FluentComboBox(parent=self)
        self.signal_combo.hide()
        self.signal_combo.currentIndexChanged[int].connect(self._signal_changed)
        self.size_combo = FluentComboBox(parent=self)
        for value in PANEL_SIZES:
            self.size_combo.addItem(value, value)
        self.size_combo.setCurrentIndex(self.size_combo.findData(DEFAULT_PANEL_SIZE))
        self.size_combo.hide()
        self.size_combo.currentIndexChanged[int].connect(self._size_changed)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self._selectors_on = False
        self._apply_card_size(DEFAULT_PANEL_SIZE)
        self.set_status("", error=False)
        self.set_selectors_enabled(False)

    @staticmethod
    def card_size(size: str) -> tuple[int, int]:
        """Return the v1 outer card size for a named panel preset."""

        from zlc_ui.board import panel_display_size

        panel_width, panel_height = panel_display_size(size)
        return (
            panel_width + 2 * CARD_PAD,
            scaled_px(CARD_TITLE_PX) + scaled_px(2) + panel_height + CARD_PAD,
        )

    def _apply_card_size(self, size: str) -> None:
        width, height = self.card_size(str(size))
        # The real v1 card is fixed by its board geometry.  Keep the standalone
        # size as the initial hint while allowing an injected BoardMetrics in
        # tests/embedded hosts to place an arbitrary rectangle.
        self.setMinimumSize(0, 0)
        self.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
        self.resize(width, height)
        self._place_settings_button()

    def set_live(self, live: bool) -> None:
        """Whether what this card shows will ever deliver again.

        A saved figure will not.  Its redraw interval means nothing, and there
        is no board to remove it from -- offering both anyway is two controls
        that cannot do what they say, which is the failure this project keeps
        being audited for.
        """

        self._live = bool(live)
        remove = getattr(self, "remove_button", None)
        if remove is not None:
            remove.setVisible(self._live)

    def set_interval_choices(self, intervals: object) -> None:
        """Receive the scheduler's one finite refresh policy."""

        values = tuple(int(value) for value in tuple(intervals or ()))
        # The shared helper validates both the domain and the current state.
        interval_form_field(values, values[0] if values else 0)
        self._interval_choices = values
        self._rebuild_settings_form()

    def set_panel_state(self, state: object) -> None:
        """Project the one Workbench-owned state into this Setting view."""

        self._apply_panel_state(self._validated_panel_state(state))

    def _validated_panel_state(self, state: object) -> dict[str, object]:
        incoming = panel_state_document(state)
        if incoming["size"] not in PANEL_SIZES:
            raise ValueError(
                f"unknown panel size {incoming['size']!r}; choose from "
                f"{', '.join(PANEL_SIZES)}"
            )
        if (
            self._interval_choices
            and int(incoming["interval_ms"]) not in self._interval_choices
        ):
            raise ValueError(
                f"display interval {incoming['interval_ms']} is not in "
                f"{self._interval_choices}"
            )
        return incoming

    def _apply_panel_state(self, incoming: Mapping[str, object]) -> None:
        previous_size = str(self._state_projection.get("size") or "")
        self._state_projection = dict(incoming)
        self._base_title = incoming["title"] or "Panel"
        self.setTitle(str(self._base_title))
        with signals_blocked(self.title_edit, self.signal_combo, self.size_combo):
            self.title_edit.setText(self._base_title)
            signal_index = self.signal_combo.findData(incoming["signal"])
            if signal_index >= 0:
                self.signal_combo.setCurrentIndex(signal_index)
            size_index = self.size_combo.findData(incoming["size"])
            if size_index >= 0:
                self.size_combo.setCurrentIndex(size_index)
        self._apply_card_size(str(incoming["size"]))
        if incoming["size"] != previous_size:
            self.geometry_changed.emit()
        self._rebuild_settings_form()

    @property
    def panel_size(self) -> str:
        return str(self._state_projection["size"])

    def set_title(self, title: str) -> None:
        incoming = dict(self._state_projection)
        incoming["title"] = str(title)
        self.set_panel_state(incoming)

    def set_panel_size(self, size: str) -> None:
        incoming = dict(self._state_projection)
        incoming["size"] = str(size)
        self.set_panel_state(incoming)

    def set_panel_projection(self, state: object, surface: object) -> None:
        """Replace state and plot metadata before reconciling the form once."""

        incoming = self._validated_panel_state(state)
        self._parameter_surface = dict(surface) if isinstance(surface, Mapping) else {}
        self._apply_panel_state(incoming)

    def _place_settings_button(self) -> None:
        button = getattr(self, "settings_button", None)
        if button is None:
            return
        button.move(
            max(0, self.width() - button.width() - CARD_PAD),
            max(0, scaled_px(2)),
        )
        button.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._place_settings_button()

    def set_surface(self, widget: QtWidgets.QWidget | None) -> None:
        """Mount or replace an arbitrary view surface."""

        if widget is not None and not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("surface must be QWidget or None")
        if widget is not None:
            # The switch may have been thrown while this card was empty.
            _set_interaction(widget, self._selectors_on)
        if self._surface is not None:
            self._surface.removeEventFilter(self)
            self._surface_layout.removeWidget(self._surface)
            self._surface.hide()
            self._surface.setParent(None)
        self._surface = widget
        if widget is None:
            self._placeholder.show()
            return
        self._placeholder.hide()
        widget.setParent(self)
        widget.installEventFilter(self)
        self._surface_layout.addWidget(widget)
        widget.show()
        self._place_settings_button()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        popup = self._settings_popup
        if watched is self._settings_drag_handle and popup is not None:
            if (
                event.type() == QtCore.QEvent.MouseButtonPress
                and event.button() == QtCore.Qt.LeftButton
            ):
                self._settings_drag_offset = (
                    event.globalPos() - popup.frameGeometry().topLeft()
                )
                return True
            if (
                event.type() == QtCore.QEvent.MouseMove
                and self._settings_drag_offset is not None
            ):
                popup.move(event.globalPos() - self._settings_drag_offset)
                return True
            if (
                event.type() == QtCore.QEvent.MouseButtonRelease
                and event.button() == QtCore.Qt.LeftButton
            ):
                self._settings_drag_offset = None
                return True
            if event.type() == QtCore.QEvent.Hide:
                self._settings_drag_offset = None
        if (
            watched is self._surface
            and event.type() == QtCore.QEvent.Wheel
            and not self._selectors_on
        ):
            ancestor = self.parentWidget()
            while ancestor is not None:
                if isinstance(ancestor, QtWidgets.QAbstractScrollArea):
                    QtWidgets.QApplication.sendEvent(ancestor.viewport(), event)
                    return True
                ancestor = ancestor.parentWidget()
        return super().eventFilter(watched, event)

    def set_signal_choices(
        self,
        groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
        *,
        current: str = "",
        overlay_groups: tuple[
            tuple[str, tuple[tuple[str, str], ...]], ...
        ] = (),
        overlay_current: str = "",
    ) -> None:
        """Replace the signal domain used by the Setting form."""

        self._groups = tuple(
            (str(producer), tuple((str(display), str(key)) for display, key in leaves))
            for producer, leaves in groups
        )
        self._overlay_groups = tuple(
            (str(producer), tuple((str(display), str(key)) for display, key in leaves))
            for producer, leaves in overlay_groups
        )
        current = str(current or self._state_projection.get("signal") or "")
        if current:
            self._state_projection["signal"] = current
        self._state_projection["overlay_signal"] = str(
            overlay_current or self._state_projection.get("overlay_signal") or ""
        )
        with signals_blocked(self.signal_combo):
            self.signal_combo.clear()
            for producer, leaves in self._groups:
                self.signal_combo.addItem(producer, None)
                header = self.signal_combo.model().item(self.signal_combo.count() - 1)
                if header is not None:
                    header.setEnabled(False)
                for display, key in leaves:
                    self.signal_combo.addItem(f"    {display}", key)
            index = self.signal_combo.findData(current)
            if index >= 0:
                self.signal_combo.setCurrentIndex(index)
        self._rebuild_settings_form()

    def set_status(self, text: str, *, error: bool) -> None:
        value = str(text)
        self.status_label.setText(value)
        self.status_dot.set_color(RED if error else GREY)
        # v1 does not reserve a transient status row in the card body.  Keep
        # the status available to the presenter without changing geometry.
        self.status_label.setToolTip(value)

    def set_selectors_enabled(self, enabled: bool) -> None:
        """Allow or suspend dragging on this panel's plot.

        On the PLOT.  This disabled the card's own dropdowns and settings
        button instead -- a different question that happens to have a similar
        name -- so the switch appeared to work, greyed out some controls, and
        never reached the one thing it is named after.
        """

        self._selectors_on = bool(enabled)
        _set_interaction(self._surface, self._selectors_on)

    def set_editing_enabled(self, enabled: bool) -> None:
        """Gate persisted panel edits without disabling plot interaction."""

        self._editing_enabled = bool(enabled)
        self.settings_button.setEnabled(self._editing_enabled)
        if not self._editing_enabled and self._settings_popup is not None:
            self._settings_popup.hide()

    def _commit_title(self) -> None:
        value = self.title_edit.text().strip()
        self.title_committed.emit(value)

    def _signal_changed(self, index: int) -> None:
        value = self.signal_combo.itemData(index)
        if isinstance(value, str):
            self.signal_picked.emit(value)

    def _size_changed(self, index: int) -> None:
        value = self.size_combo.itemData(index)
        if isinstance(value, str):
            self.size_picked.emit(value)

    def _form_spec(self) -> FormSpec:
        state = self._state_projection
        current_signal = str(state.get("signal") or "")
        fields: list[FormFieldProps] = [
            FormFieldProps(
                "kind",
                "text",
                "Plot kind",
                default=str(state.get("kind") or "automatic"),
            ),
        ]
        if state.get("cell_kind"):
            fields.append(
                FormFieldProps(
                    "cell_kind",
                    "text",
                    "Cell kind",
                    default=str(state["cell_kind"]),
                )
            )
        fields.extend([
            FormFieldProps("title", "text", "Panel name", default=state["title"]),
            FormFieldProps(
                "signal",
                "keyed_choice",
                "Signal",
                default=current_signal,
                required=True,
            ),
        ])
        if state.get("kind") == "image":
            fields.append(FormFieldProps(
                "overlay_signal",
                "keyed_choice",
                "Overlay",
                default=str(state.get("overlay_signal") or ""),
            ))
        fields.append(
            FormFieldProps(
                "size",
                "choice",
                "Size",
                default=state["size"],
                choices=tuple(FormChoice(value, value) for value in PANEL_SIZES),
            )
        )
        if self._live and self._interval_choices:
            fields.append(
                interval_form_field(
                    self._interval_choices,
                    state["interval_ms"],
                )
            )
        section_labels = {
            "semantic": "Semantic parameters",
            "display": "Display / interaction parameters",
            "fit": "Fit parameters / result",
        }
        for section in ("semantic", "display", "fit"):
            declared_fields = tuple(
                parameter_fields(self._parameter_surface, section)
            )
            for declared in declared_fields:
                field = parameter_form_spec(
                    (declared,),
                ).fields[0]
                fields.append(
                    FormFieldProps(
                        key=f"{section}__{str(declared['key'])}",
                        kind=field.kind,
                        label=field.label,
                        default=field.default,
                        required=field.required,
                        minimum=field.minimum,
                        maximum=field.maximum,
                        choices=field.choices,
                        allow_blank=field.allow_blank,
                        unavailable_reason=field.unavailable_reason,
                        automatic=field.automatic,
                    )
                )
            unavailable = str(
                self._parameter_surface.get(f"{section}_unavailable") or ""
            )
            if unavailable:
                fields.append(
                    FormFieldProps(
                        f"{section}_unavailable",
                        "text",
                        section_labels[section],
                        default=unavailable,
                    )
                )
        return FormSpec(tuple(fields))

    def _form_values(self) -> dict[str, object]:
        signal = str(self._state_projection.get("signal") or "") or None
        values: dict[str, object] = {
            "kind": str(self._state_projection.get("kind") or "automatic"),
            "title": str(self._state_projection.get("title") or "Panel"),
            "signal": signal,
            "size": str(self._state_projection.get("size") or DEFAULT_PANEL_SIZE),
        }
        if self._state_projection.get("kind") == "image":
            values["overlay_signal"] = str(
                self._state_projection.get("overlay_signal") or ""
            )
        if self._state_projection.get("cell_kind"):
            values["cell_kind"] = str(self._state_projection["cell_kind"])
        if self._live and self._interval_choices:
            values["interval_ms"] = int(self._state_projection["interval_ms"])
        form_keys = self._form_spec().keys
        for section in ("semantic", "display", "fit"):
            declared_fields = tuple(
                parameter_fields(self._parameter_surface, section)
            )
            declared_values = parameter_form_values(
                declared_fields,
            )
            for key, value in declared_values.items():
                values[f"{section}__{key}"] = value
            unavailable_key = f"{section}_unavailable"
            if unavailable_key in form_keys:
                values[unavailable_key] = str(
                    self._parameter_surface.get(unavailable_key) or ""
                )
        return values

    def _rebuild_settings_form(self) -> None:
        if self._settings_form is not None:
            self._settings_form.refresh()
            self._settings_form.reconcile(self._form_spec(), self._form_values())
            self._settings_form.refresh()
            self._settings_form.widget_for("signal").setEnabled(bool(self._groups))
            self._settings_form.widget_for("kind").setEnabled(False)
            if "cell_kind" in self._settings_form.spec.keys:
                self._settings_form.widget_for("cell_kind").setEnabled(False)
            for section in ("semantic", "display", "fit"):
                key = f"{section}_unavailable"
                if key in self._settings_form.spec.keys:
                    self._settings_form.widget_for(key).setEnabled(False)
            self._sync_settings_body_height()

    def _sync_settings_body_height(self) -> None:
        body = self._settings_body
        if body is None or body.layout() is None:
            return
        body.setMinimumHeight(0)
        body.layout().activate()
        body.setMinimumHeight(body.layout().sizeHint().height())

    def _open_settings(self) -> None:
        if self._settings_popup is None:
            popup = FluentPopup(self)
            layout = QtWidgets.QVBoxLayout(popup)
            pad = max(1, scaled_px(10))
            layout.setContentsMargins(pad, pad, 0, pad)
            layout.setSpacing(0)
            drag_handle = FluentLabel("Setting", popup)
            drag_handle.setCursor(QtCore.Qt.SizeAllCursor)
            drag_handle.installEventFilter(self)
            layout.addWidget(drag_handle)
            scroll = FluentScrollArea(popup)
            body = QtWidgets.QWidget()
            body.setStyleSheet("background: transparent;")
            body_layout = QtWidgets.QVBoxLayout(body)
            body_layout.setContentsMargins(0, 0, pad, 0)
            body_layout.setSpacing(max(1, scaled_px(5)))
            self._settings_form = FluentParameterForm(
                self._form_spec(),
                self._form_values(),
                runtime=self._signal_runtime,
                parent=body,
            )
            self._settings_form.widget_for("signal").setEnabled(bool(self._groups))
            self._settings_form.changed.connect(self._setting_changed)
            self._settings_form.widget_for("kind").setEnabled(False)
            if "cell_kind" in self._settings_form.spec.keys:
                self._settings_form.widget_for("cell_kind").setEnabled(False)
            for section in ("semantic", "display", "fit"):
                key = f"{section}_unavailable"
                if key in self._settings_form.spec.keys:
                    self._settings_form.widget_for(key).setEnabled(False)
            body_layout.addWidget(self._settings_form)
            buttons = QtWidgets.QHBoxLayout()
            buttons.setContentsMargins(0, 0, 0, 0)
            buttons.setSpacing(max(1, scaled_px(5)))
            # Here rather than on the card face, which is where every other
            # per-panel decision already lives.  They existed as hidden widgets
            # nothing ever showed, so a panel opened in the real window could
            # not be removed at all and its Edit signal could not be raised.
            self.edit_button = FluentButton("Edit", body, color=ACCENT)
            self.edit_button.clicked.connect(self._request_edit)
            self.remove_button = FluentButton("Remove", body, color=ORANGE)
            self.remove_button.clicked.connect(self._request_remove)
            self.remove_button.setVisible(self._live)
            buttons.addWidget(self.edit_button)
            buttons.addStretch(1)
            buttons.addWidget(self.remove_button)
            body_layout.addLayout(buttons)
            scroll.set_width_bounded_widget(body)
            layout.addWidget(scroll)
            self._settings_popup = popup
            self._settings_anchor = FluentSettingsPopupAnchor(
                popup, self.settings_button
            )
            self._settings_drag_handle = drag_handle
            self._settings_scroll = scroll
            self._settings_body = body
            self._sync_settings_body_height()
            popup.hide()
        anchor = self._settings_anchor
        scroll = self._settings_scroll
        anchor.toggle(
            scroll,
            prepare=self._rebuild_settings_form,
            minimum_width=280,
            minimum_height=320,
            maximum_height=max(1, self.height()),
        )

    def _request_edit(self) -> None:
        self._settings_popup.hide()
        self.edit_requested.emit()

    def _request_remove(self) -> None:
        self._settings_popup.hide()
        self.remove_requested.emit()

    def _setting_changed(self, key: str) -> None:
        if self._settings_form is None:
            return
        name = str(key)
        if name in {
            "kind",
            "cell_kind",
            "semantic_unavailable",
            "display_unavailable",
            "fit_unavailable",
        }:
            return
        try:
            value = self._settings_form.read_value(name)
            section, separator, parameter_key = name.partition("__")
            if separator and section in {"semantic", "display", "fit"}:
                declared = {
                    str(field["key"]): field
                    for field in parameter_fields(self._parameter_surface, section)
                }
                patch: dict[str, object] = {
                    section: {
                        parameter_key: decode_parameter_value(
                            declared[parameter_key], value
                        )
                    }
                }
            elif name == "interval_ms":
                patch = {name: int(value)}
            elif name in {"title", "signal", "overlay_signal", "size"}:
                patch = {name: str(value or "")}
            else:
                return
        except (KeyError, TypeError, ValueError) as error:
            self.set_status(str(error), error=True)
            return
        self.state_changed.emit(patch)

    def _choice_groups(self, field: str):
        if str(field) == "signal":
            return self._groups
        if str(field) == "overlay_signal":
            return self._overlay_groups
        return ()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.raise_()
            self.grabMouse()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag_offset is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.move(self.mapToParent(event.pos() - self._drag_offset))
            point = (int(event.pos().x()), int(event.pos().y()))
            self.drag_started.emit(point)
            self.drag_moved.emit(point)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == QtCore.Qt.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            self.releaseMouse()
            self.dropped.emit((int(event.pos().x()), int(event.pos().y())))
        super().mouseReleaseEvent(event)


__all__ = ["PanelCardView"]
