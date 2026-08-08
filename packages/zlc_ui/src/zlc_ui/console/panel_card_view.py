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
    FluentSettingsPopupAnchor,
    FluentSpinBox,
    FluentStatusDot,
    GREY,
    ORANGE,
    RED,
    scaled_px,
    signals_blocked,
)
from zlc_ui.form.form import FormChoice, FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FormRuntimeContext, FluentParameterForm

from ._panel_projection import (
    decode_parameter_value,
    decode_mapping_value,
    mapping_form_spec,
    mapping_form_values,
    panel_state_document,
    parameter_fields,
    parameter_form_spec,
    parameter_form_values,
    signal_form_choices,
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
    update_ms_picked = QtCore.pyqtSignal(int)
    title_committed = QtCore.pyqtSignal(str)
    remove_requested = QtCore.pyqtSignal()
    edit_requested = QtCore.pyqtSignal()
    state_changed = QtCore.pyqtSignal(object)
    dropped = QtCore.pyqtSignal(tuple)
    drag_started = QtCore.pyqtSignal(tuple)
    drag_moved = QtCore.pyqtSignal(tuple)

    def __init__(self, panel_id: str, title: str = "Panel", parent=None) -> None:
        super().__init__(str(title), parent)
        self.panel_id = str(panel_id)
        self._base_title = str(title)
        self._surface: QtWidgets.QWidget | None = None
        self._settings_popup: FluentPopup | None = None
        self._settings_form: FluentParameterForm | None = None
        self._settings_anchor: FluentSettingsPopupAnchor | None = None
        self._groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
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
        self._drag_offset: QtCore.QPoint | None = None

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
        self._placeholder = FluentLabel("No surface mounted")
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        holder.addWidget(self._placeholder)

        # The v1 card has one visible command on its title strip.  It is a
        # child overlay rather than a layout item, so it never changes the
        # panel data rectangle or the packer's card size.
        #: Whether what this card shows will deliver again.  A live panel
        #: redraws on a beat and can be taken off the board; a saved figure
        #: does neither, and must not offer controls for both.
        self._live = True
        self.settings_button = FluentButton("Setting", color=GREY)
        self.settings_button.setParent(self)
        self.settings_button.setFixedSize(
            scaled_px(74, minimum=64),
            scaled_px(26, minimum=22),
        )
        self.settings_button.clicked.connect(self._open_settings)

        # Compatibility handles for the presenter-facing lightweight API.
        # They are deliberately not placed on the card face: v1 exposes these
        # edits from Setting, not as a second invented toolbar.
        self.status_dot = FluentStatusDot(size=12, parent=self)
        self.status_dot.hide()
        self.status_label = FluentLabel("", parent=self)
        self.status_label.hide()
        self.title_edit = FluentLineEdit(str(title), parent=self)
        self.title_edit.setPlaceholderText("panel title…")
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
        self.update_spin = FluentSpinBox(parent=self)
        self.update_spin.setRange(1, 60_000)
        self.update_spin.hide()
        self.update_spin.valueChanged.connect(
            lambda value: self.update_ms_picked.emit(int(value))
        )

        self.setCursor(QtCore.Qt.OpenHandCursor)
        self._selectors_on = True
        self._apply_card_size(DEFAULT_PANEL_SIZE)
        self.set_status("", error=False)
        self.set_selectors_enabled(True)

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
        self.update_spin.setVisible(self._live)
        remove = getattr(self, "remove_button", None)
        if remove is not None:
            remove.setVisible(self._live)

    def set_update_ms(self, interval_ms: int) -> None:
        """Show how often this panel actually redraws.

        Told, never assumed.  This box used to open on a literal 100 while the
        panel behind it redrew on the board's own default, so the one number an
        operator reads to answer "how fast is this going" was the one number
        nothing had asked.
        """

        value = int(interval_ms)
        if value <= 0:
            raise ValueError("a redraw interval must be positive")
        self._state_projection["interval_ms"] = value
        with signals_blocked(self.update_spin):
            self.update_spin.setValue(value)

    def set_panel_state(self, state: object) -> None:
        """Project the one Workbench-owned state into this Setting view."""

        incoming = panel_state_document(state)
        self._state_projection = incoming
        self._base_title = incoming["title"] or "Panel"
        self.setTitle(self._base_title)
        with signals_blocked(
            self.title_edit,
            self.signal_combo,
            self.size_combo,
            self.update_spin,
        ):
            self.title_edit.setText(self._base_title)
            signal_index = self.signal_combo.findData(incoming["signal"])
            if signal_index >= 0:
                self.signal_combo.setCurrentIndex(signal_index)
            size_index = self.size_combo.findData(incoming["size"])
            if size_index >= 0:
                self.size_combo.setCurrentIndex(size_index)
            self.update_spin.setValue(int(incoming["interval_ms"]))
        self._apply_card_size(incoming["size"])
        self._rebuild_settings_form()

    def set_parameter_surface(self, surface: object) -> None:
        """Project host-owned editor metadata; authored values stay in state."""

        self._parameter_surface = dict(surface) if isinstance(surface, Mapping) else {}
        self._rebuild_settings_form()

    def set_title(self, title: str) -> None:
        """Project an accepted title without raising another edit intent."""

        incoming = dict(self._state_projection)
        incoming["title"] = str(title)
        self.set_panel_state(incoming)

    def set_panel_size(self, size: str) -> None:
        """Project one v1 panel-size preset without exposing the hidden combo."""

        key = str(size).strip().lower().replace(" ", "")
        index = self.size_combo.findData(key)
        if index < 0:
            raise ValueError(
                f"unknown panel size {size!r}; choose from {', '.join(PANEL_SIZES)}"
            )
        with signals_blocked(self.size_combo):
            self.size_combo.setCurrentIndex(index)
        self._state_projection["size"] = key
        self._apply_card_size(key)

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
            self._surface_layout.removeWidget(self._surface)
            self._surface.hide()
            self._surface.setParent(None)
        self._surface = widget
        if widget is None:
            self._placeholder.show()
            return
        self._placeholder.hide()
        widget.setParent(self)
        self._surface_layout.addWidget(widget)
        widget.show()
        self._place_settings_button()

    def set_signal_choices(
        self,
        groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
        *,
        current: str = "",
    ) -> None:
        """Replace opaque signal choices grouped by a producer label.

        ``current`` names the one to show as chosen.  A card built for a signal
        has to be told which it is: an empty combo has nothing to preserve, so
        without this the card that IS showing the camera opens claiming to show
        nothing.  Left empty, whatever is selected stays selected, which is what
        a refresh of the offer wants.
        """

        self._groups = tuple(
            (str(producer), tuple((str(display), str(key)) for display, key in leaves))
            for producer, leaves in groups
        )
        current = str(
            current
            or self._state_projection.get("signal")
            or self.signal_combo.currentData()
            or ""
        )
        if current:
            self._state_projection["signal"] = current
        with signals_blocked(self.signal_combo):
            self.signal_combo.clear()
            for producer, leaves in self._groups:
                self.signal_combo.addItem(producer, None)
                header = self.signal_combo.model().item(self.signal_combo.count() - 1)
                if header is not None:
                    header.setEnabled(False)
                    font = header.font()
                    font.setBold(True)
                    header.setFont(font)
                for display, key in leaves:
                    self.signal_combo.addItem(f"    {display}", key)
            index = self.signal_combo.findData(current)
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

    def _commit_title(self) -> None:
        value = self.title_edit.text().strip()
        self._base_title = value or "Panel"
        self.setTitle(self._base_title)
        self.title_committed.emit(value)

    def _signal_changed(self, index: int) -> None:
        value = self.signal_combo.itemData(index)
        if isinstance(value, str):
            self.signal_picked.emit(value)

    def _size_changed(self, index: int) -> None:
        value = self.size_combo.itemData(index)
        if isinstance(value, str):
            self._apply_card_size(value)
            self.size_picked.emit(value)

    def _form_spec(self) -> FormSpec:
        state = self._state_projection
        choices = signal_form_choices(self._groups, str(state.get("signal") or ""))
        current_signal = str(state.get("signal") or "")
        signal_default = (
            current_signal
            if any(choice.value == current_signal for choice in choices)
            else (choices[0].value if choices else None)
        )
        fields: list[FormFieldProps] = [
            FormFieldProps(
                "kind",
                "text",
                "Plot kind",
                default=str(state.get("kind") or "automatic"),
            ),
            FormFieldProps("title", "text", "Title", default=self.title_edit.text()),
            FormFieldProps(
                "signal",
                "choice",
                "Signal",
                default=signal_default,
                choices=choices,
                required=not bool(choices),
                unavailable_reason="no choices" if not choices else "",
            ),
            FormFieldProps(
                "size",
                "choice",
                "Size",
                default=self.size_combo.currentData() or DEFAULT_PANEL_SIZE,
                choices=tuple(FormChoice(value, value) for value in PANEL_SIZES),
            ),
        ]
        if self._live:
            fields.append(
                FormFieldProps(
                    "interval_ms",
                    "int",
                    "Update interval",
                    default=int(self.update_spin.value()),
                    unit="ms",
                    minimum=1,
                    maximum=60_000,
                )
            )
        overlay = dict(self._parameter_surface.get("site_overlay") or {})
        if overlay:
            overlay["value"] = str(state.get("site_overlay") or "off")
            fields.append(parameter_form_spec((overlay,)).fields[0])
        declared_display = tuple(
            field
            for field in parameter_fields(self._parameter_surface, "display")
            if bool(field.get("quick"))
        )
        if declared_display:
            for declared in declared_display:
                field = parameter_form_spec((declared,)).fields[0]
                fields.append(
                    FormFieldProps(
                        key=f"display__{str(declared['key'])}",
                        kind=field.kind,
                        label=field.label,
                        default=field.default,
                        required=field.required,
                        minimum=field.minimum,
                        maximum=field.maximum,
                        choices=field.choices,
                        allow_blank=field.allow_blank,
                    )
                )
        else:
            for key, value in dict(state.get("display") or {}).items():
                field = mapping_form_spec({str(key): value}).fields[0]
                fields.append(
                    FormFieldProps(
                        key=f"display__{str(key)}",
                        kind=field.kind,
                        label=field.label,
                        default=field.default,
                        allow_blank=field.allow_blank,
                    )
                )
        return FormSpec(tuple(fields))

    def _form_values(self) -> dict[str, object]:
        signal = self.signal_combo.currentData()
        if not isinstance(signal, str):
            signal = str(self._state_projection.get("signal") or "") or None
        values: dict[str, object] = {
            "kind": str(self._state_projection.get("kind") or "automatic"),
            "title": self.title_edit.text(),
            "signal": signal,
            "size": self.size_combo.currentData() or DEFAULT_PANEL_SIZE,
        }
        if self._live:
            values["interval_ms"] = int(self.update_spin.value())
        if "site_overlay" in self._form_spec().keys:
            values["site_overlay"] = str(
                self._state_projection.get("site_overlay") or "off"
            )
        declared_display = tuple(
            field
            for field in parameter_fields(self._parameter_surface, "display")
            if bool(field.get("quick"))
        )
        if declared_display:
            declared_values = parameter_form_values(declared_display)
            for key, value in declared_values.items():
                values[f"display__{key}"] = value
        else:
            for key, value in dict(self._state_projection.get("display") or {}).items():
                values[f"display__{str(key)}"] = mapping_form_values(
                    {str(key): value}
                )[str(key)]
        return values

    def _rebuild_settings_form(self) -> None:
        if self._settings_form is not None:
            self._settings_form.reconcile(self._form_spec(), self._form_values())
            self._settings_form.widget_for("kind").setEnabled(False)

    def _open_settings(self) -> None:
        if self._settings_popup is None:
            popup = FluentPopup(self)
            layout = QtWidgets.QVBoxLayout(popup)
            pad = max(1, scaled_px(10))
            layout.setContentsMargins(pad, pad, pad, pad)
            layout.setSpacing(max(1, scaled_px(5)))
            self._settings_form = FluentParameterForm(
                self._form_spec(),
                self._form_values(),
                runtime=FormRuntimeContext(),
                parent=popup,
            )
            self._settings_form.widget_for("kind").setEnabled(False)
            layout.addWidget(self._settings_form)
            buttons = QtWidgets.QHBoxLayout()
            buttons.setContentsMargins(0, 0, 0, 0)
            buttons.setSpacing(max(1, scaled_px(5)))
            apply_button = FluentButton("Apply")
            apply_button.clicked.connect(self._apply_settings)
            # Here rather than on the card face, which is where every other
            # per-panel decision already lives.  They existed as hidden widgets
            # nothing ever showed, so a panel opened in the real window could
            # not be removed at all and its Edit signal could not be raised.
            self.edit_button = FluentButton("Edit", color=ACCENT)
            self.edit_button.clicked.connect(self._request_edit)
            self.remove_button = FluentButton("Remove", color=ORANGE)
            self.remove_button.clicked.connect(self._request_remove)
            self.remove_button.setVisible(self._live)
            buttons.addWidget(apply_button)
            buttons.addWidget(self.edit_button)
            buttons.addStretch(1)
            buttons.addWidget(self.remove_button)
            layout.addLayout(buttons)
            self._settings_popup = popup
            self._settings_anchor = FluentSettingsPopupAnchor(popup, self.settings_button)
        self._settings_anchor.toggle(
            self._settings_form,
            prepare=self._rebuild_settings_form,
        )

    def _request_edit(self) -> None:
        self._settings_popup.hide()
        self.edit_requested.emit()

    def _request_remove(self) -> None:
        self._settings_popup.hide()
        self.remove_requested.emit()

    def _apply_settings(self) -> None:
        if self._settings_form is None:
            return
        values = self._settings_form.read_all()
        display: dict[str, object] = {}
        originals = dict(self._state_projection.get("display") or {})
        declared = {
            str(field["key"]): field
            for field in parameter_fields(self._parameter_surface, "display")
            if bool(field.get("quick"))
        }
        for form_key, value in values.items():
            if not str(form_key).startswith("display__"):
                continue
            key = str(form_key).split("__", 1)[1]
            try:
                display[key] = (
                    decode_parameter_value(declared[key], value)
                    if key in declared
                    else decode_mapping_value(originals[key], value)
                )
            except (KeyError, TypeError, ValueError) as error:
                self.set_status(str(error), error=True)
                return
        patch: dict[str, object] = {
            "title": str(values["title"]),
            "signal": str(values.get("signal") or ""),
            "size": str(values["size"]),
        }
        if "interval_ms" in values:
            patch["interval_ms"] = int(values["interval_ms"])
        if "site_overlay" in values:
            patch["site_overlay"] = str(values["site_overlay"])
        if display:
            patch["display"] = display
        self.state_changed.emit(patch)
        self._settings_popup.hide()

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
