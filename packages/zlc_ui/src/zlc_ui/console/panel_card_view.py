"""The v1-shaped console panel card.

The board owns placement; this widget owns only the titled Fluent surface and
the small ``Setting`` affordance from v1.  Signal/size/update controls remain
available through the settings popup for the lightweight presenter API, but
they do not add an invented toolbar to the card face.
"""

from __future__ import annotations

from collections.abc import Mapping

from PyQt5 import QtCore, QtWidgets

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
    popup_gap,
    RED,
    scaled_px,
    show_fluent_popup_for_anchor,
    signals_blocked,
)
from zlc_ui.form.form import FormChoice, FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm

from ._panel_projection import (
    decode_parameter_value,
    draws_image_surfaces,
    interval_form_field,
    panel_state_document,
    parameter_fields,
    parameter_form_spec,
    parameter_form_values,
    signal_form_runtime,
)


def _set_interaction(surface: object | None, enabled: bool) -> None:
    """Keep a plot surface's input transport alive, when it has one.

    Duck-typed on purpose: a plot widget owns an interaction gate, and a label
    standing in for one in a demo does not.  The card always enables the
    transport at mount -- navigation (double-click focus, pan, zoom, scroll)
    must survive the Selectors switch, so the switch gates only the
    selector-STARTING pointer gestures in the card's own event filter and
    never closes this transport.
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
    #: A refusal the mounted plot surface reported (``errorOccurred``).  The
    #: card only relays it: unconnected, those refusals were fully silent.
    plot_error = QtCore.pyqtSignal(str)

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
        #: The form's required content width in device pixels.  Carried to the
        #: popup sizer explicitly: the scroll body stays a width CONSUMER (its
        #: minimum is never pinned), so a screen-clamped popup reflows the
        #: rows instead of clipping their right edge under the scrollbar.
        self._settings_content_width: int | None = None
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
                # UI-only empty-card geometry.  Product choices/default arrive
                # from the plotting owner before a card enters the board.
                "size": "1x1",
                "interval_ms": 0,
                "title": str(title),
            }
        )
        self._parameter_surface: Mapping[str, object] = {}
        self._interval_choices: tuple[int, ...] = ()
        self._size_choices: tuple[str, ...] = ()
        self._cell_kind_choices: tuple[str, ...] = ()
        self._default_size = ""
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

        # The presenter marks the card a board-wide error line is about.  v1
        # reserves no status row in the card body, so the dot overlays the
        # title strip beside Setting -- geometry never changes -- and the full
        # message travels as the dot's tooltip.
        self.status_dot = FluentStatusDot(size=12, parent=self)
        self.status_dot.hide()
        # FigureViewer also embeds this card and exposes these lightweight
        # handles through its own port. TaskConsole state never reads them.
        self.title_edit = FluentLineEdit(str(title), parent=self)
        self.title_edit.hide()
        self.title_edit.editingFinished.connect(self._commit_title)
        self.signal_combo = FluentComboBox(parent=self)
        self.signal_combo.hide()
        self.signal_combo.currentIndexChanged[int].connect(self._signal_changed)
        self.size_combo = FluentComboBox(parent=self)
        self.size_combo.hide()
        self.size_combo.currentIndexChanged[int].connect(self._size_changed)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self._selectors_on = False
        #: Buttons whose press this card swallowed while Selectors is off, so
        #: the matching move/release stream is swallowed with it and the plot
        #: never sees half a gesture.
        self._gated_buttons: set[int] = set()
        self._surface_error_connected = False
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

    def set_interval_choices(self, intervals: object, default_interval: int) -> None:
        """Receive the scheduler's one finite refresh policy."""

        values = tuple(int(value) for value in tuple(intervals or ()))
        # The shared helper validates both the domain and the current state.
        default = int(default_interval)
        interval_form_field(values, default)
        self._interval_choices = values
        if not self._state_projection.get("kind"):
            self._state_projection["interval_ms"] = default
        self._rebuild_settings_form()

    def set_size_choices(self, sizes: object, default_size: str) -> None:
        """Receive the plotting owner's finite panel-size policy."""

        values = tuple(str(value) for value in tuple(sizes or ()))
        if not values or len(set(values)) != len(values):
            raise ValueError("panel size choices must be unique and non-empty")
        default = str(default_size)
        if default not in values:
            raise ValueError("default panel size must be one of the choices")
        self._size_choices = values
        self._default_size = default
        with signals_blocked(self.size_combo):
            self.size_combo.clear()
            for value in values:
                self.size_combo.addItem(value, value)
            current = str(self._state_projection.get("size") or default)
            self.size_combo.setCurrentIndex(self.size_combo.findData(current))
        if not self._state_projection.get("kind"):
            self._state_projection["size"] = default
        self._apply_card_size(str(self._state_projection["size"]))
        self._rebuild_settings_form()

    def set_cell_kind_choices(self, kinds: object) -> None:
        """Receive the grid cell vocabulary for the settings choice control."""

        values = tuple(str(value) for value in tuple(kinds or ()))
        if not values or len(set(values)) != len(values):
            raise ValueError("grid cell kinds must be unique and non-empty")
        self._cell_kind_choices = values
        self._rebuild_settings_form()

    def set_panel_state(self, state: object) -> None:
        """Project the one Workbench-owned state into this Setting view."""

        self._apply_panel_state(self._validated_panel_state(state))

    def _validated_panel_state(self, state: object) -> dict[str, object]:
        incoming = panel_state_document(state)
        if not self._size_choices:
            raise RuntimeError("panel size choices were not projected")
        if incoming["size"] not in self._size_choices:
            raise ValueError(
                f"unknown panel size {incoming['size']!r}; choose from "
                f"{', '.join(self._size_choices)}"
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
        # The strip names the panel AND says what it is drawing.  The name is
        # often the signal key and nothing else -- "@logic/<node>/frames" --
        # which says where the numbers came from and not what they are; the
        # rest is composed by the presenter from the shown dataset and this
        # panel's own use of it.  The editable name stays the name.
        summary = str(self._parameter_surface.get("data_summary") or "")
        self.setTitle(
            f"{self._base_title}  ·  {summary}" if summary else str(self._base_title)
        )
        with signals_blocked(self.title_edit, self.signal_combo, self.size_combo):
            self.title_edit.setText(self._base_title)
            signal_index = self.signal_combo.findData(incoming["signal"])
            if signal_index >= 0:
                self.signal_combo.setCurrentIndex(signal_index)
            size_index = self.size_combo.findData(incoming["size"])
            if size_index >= 0:
                self.size_combo.setCurrentIndex(size_index)
        if incoming["size"] != previous_size:
            self._apply_card_size(str(incoming["size"]))
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
        dot = getattr(self, "status_dot", None)
        if dot is not None:
            dot.move(
                max(0, button.x() - dot.width() - CARD_PAD),
                max(0, button.y() + (button.height() - dot.height()) // 2),
            )
            dot.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._place_settings_button()

    @property
    def surface(self) -> QtWidgets.QWidget | None:
        """The mounted view surface, so its owner can reuse or address it."""

        return self._surface

    def set_surface(self, widget: QtWidgets.QWidget | None) -> None:
        """Mount or replace an arbitrary view surface."""

        if widget is not None and not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("surface must be QWidget or None")
        if widget is not None and widget is self._surface:
            widget.show()
            return
        if widget is not None:
            # The transport stays OPEN regardless of the Selectors switch:
            # navigation (double-click focus, pan, zoom) is never the
            # switch's to drop.  Selector-starting gestures are gated in
            # this card's event filter instead.
            _set_interaction(widget, True)
        if self._surface is not None:
            self._surface.removeEventFilter(self)
            if self._surface_error_connected:
                try:
                    self._surface.errorOccurred.disconnect(self.plot_error)
                except (AttributeError, RuntimeError, TypeError):
                    pass
            self._surface_error_connected = False
            self._surface_layout.removeWidget(self._surface)
            self._surface.hide()
            self._surface.setParent(None)
        self._gated_buttons.clear()
        self._surface = widget
        if widget is None:
            self._placeholder.show()
            return
        self._placeholder.hide()
        widget.setParent(self)
        widget.installEventFilter(self)
        relay = getattr(widget, "errorOccurred", None)
        if hasattr(relay, "connect"):
            # A plot surface says what a gesture could not do through this
            # one signal; the card relays it so the console can report it.
            relay.connect(self.plot_error)
            self._surface_error_connected = True
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
            watched is self._settings_body
            and event.type() == QtCore.QEvent.LayoutRequest
        ):
            # The settings body re-measures whenever ITS OWN layout says its
            # geometry is stale -- however many layout turns nested rows take
            # to settle.  A deferred one-shot measurement was a timing guess,
            # and a guess that lost left the pinned body height short, with
            # the Edit row painted over the form's last row until the popup
            # was reopened.  Re-entry terminates: an unchanged measurement
            # writes nothing and so requests no further layout.
            self._sync_settings_body_size()
        if (
            watched is self._surface
            and event.type() == QtCore.QEvent.Wheel
            and not bool(getattr(self._surface, "interaction_enabled", False))
        ):
            # A surface with an OPEN plot transport owns its wheel (zoom).
            # Anything else -- a demo label, a surface whose transport a host
            # closed -- would drop the wheel dead, so it goes to the board
            # scroll instead.  Qt does not propagate the ignored wheel up
            # through this parent chain by itself.
            ancestor = self.parentWidget()
            while ancestor is not None:
                if isinstance(ancestor, QtWidgets.QAbstractScrollArea):
                    QtWidgets.QApplication.sendEvent(ancestor.viewport(), event)
                    return True
                ancestor = ancestor.parentWidget()
        if watched is self._surface and not self._selectors_on:
            # The Selectors switch gates SELECTOR CREATION only, never
            # navigation.  A left press starts (or drags) a selector and a
            # right press installs a crosshair, so those presses -- and the
            # rest of their own gesture stream -- are swallowed here.
            # Everything else stays routed: double-click (facet focus),
            # middle-button pan/zoom-reset, wheel zoom, hover, Escape.
            kind = event.type()
            if kind == QtCore.QEvent.MouseButtonPress and event.button() in (
                QtCore.Qt.LeftButton,
                QtCore.Qt.RightButton,
            ):
                self._gated_buttons.add(int(event.button()))
                return True
            if kind == QtCore.QEvent.MouseMove and any(
                int(event.buttons()) & button for button in self._gated_buttons
            ):
                return True
            if (
                kind == QtCore.QEvent.MouseButtonRelease
                and int(event.button()) in self._gated_buttons
            ):
                self._gated_buttons.discard(int(event.button()))
                return True
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
        """Say on the card itself what a board-wide status line says about it.

        These widgets existed hidden and nothing ever showed them, so a panel
        named in a red strip line could not be told apart by looking at the
        board.  The dot rides the title strip as an overlay, like the Setting
        button, so the v1 card body still reserves no status row; hovering it
        reads the full message.  An empty text clears the mark.
        """

        value = str(text)
        self.status_dot.set_color(RED if error else GREY)
        self.status_dot.setToolTip(value)
        self.status_dot.setVisible(bool(value))
        if value:
            self._place_settings_button()

    def set_selectors_enabled(self, enabled: bool) -> None:
        """Allow or suspend STARTING selectors on this panel's plot.

        Only that.  This used to close the plot widget's whole interaction
        transport, which silently dropped every pointer event -- including
        the double-click that is the only way to focus a facet cell, and
        pan/zoom -- until the operator flipped Selectors on.  Navigation is
        not the switch's to gate: the card's event filter swallows only the
        selector-starting presses while this flag is off.
        """

        self._selectors_on = bool(enabled)

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
        if state.get("kind") == "facet_grid" and self._cell_kind_choices:
            # A grid's cell kind is a panel PARAMETER, not a second plot
            # kind: empty means the data decides, a name is the operator's
            # explicit choice.
            fields.append(
                FormFieldProps(
                    "cell_kind",
                    "choice",
                    "Cell kind",
                    default=str(state.get("cell_kind") or ""),
                    choices=(
                        FormChoice("automatic (from data)", ""),
                        *(
                            FormChoice(value, value)
                            for value in self._cell_kind_choices
                        ),
                    ),
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
        if draws_image_surfaces(state):
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
                choices=tuple(FormChoice(value, value) for value in self._size_choices),
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
            "semantic": "Semantic",
            "display": "Display",
            "fit": "Fit",
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
            "size": str(self._state_projection.get("size") or self._default_size),
        }
        if draws_image_surfaces(self._state_projection):
            values["overlay_signal"] = str(
                self._state_projection.get("overlay_signal") or ""
            )
        if (
            self._state_projection.get("kind") == "facet_grid"
            and self._cell_kind_choices
        ):
            values["cell_kind"] = str(
                self._state_projection.get("cell_kind") or ""
            )
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
            spec = self._form_spec()
            self._settings_form.reconcile(spec, self._form_values())
            self._settings_form.refresh()
            self._settings_form.widget_for("signal").setEnabled(bool(self._groups))
            self._settings_form.widget_for("kind").setEnabled(False)
            for section in ("semantic", "display", "fit"):
                key = f"{section}_unavailable"
                if key in self._settings_form.spec.keys:
                    self._settings_form.widget_for(key).setEnabled(False)
            self._sync_settings_body_size()

    def _sync_settings_body_size(self) -> None:
        body = self._settings_body
        if body is None or body.layout() is None:
            return
        margins = body.layout().contentsMargins()
        required_width = (
            self._settings_form.minimum_content_width()
            + margins.left()
            + margins.right()
        )
        required_height = body.layout().sizeHint().height()
        geometry_changed = False
        if self._settings_content_width != required_width:
            self._settings_content_width = required_width
            geometry_changed = True
        if body.minimumHeight() != required_height:
            body.setMinimumHeight(required_height)
            geometry_changed = True
        popup = self._settings_popup
        if geometry_changed and popup is not None and popup.isVisible():
            self._show_settings_popup()

    def _settings_available_height(self) -> int:
        """Vertical room the popup may take: from below Setting to the card foot."""

        return max(
            1,
            self.rect().bottom()
            - self.settings_button.geometry().bottom()
            - popup_gap(),
        )

    def _show_settings_popup(self) -> None:
        """The card's ONE popup placement call, always with fresh geometry.

        Toggle-open and open-while-visible resize both come through here, so
        the popup can never be shown from a stale width measured before the
        form was rebuilt.
        """

        show_fluent_popup_for_anchor(
            self._settings_popup,
            self.settings_button,
            self._settings_body,
            minimum_width=1,
            minimum_height=320,
            maximum_height=self._settings_available_height(),
            content_width=self._settings_content_width,
        )

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
            # LayoutRequest-driven measurement: the body says when its own
            # geometry went stale (see eventFilter).
            body.installEventFilter(self)
            self._sync_settings_body_size()
            popup.hide()
        anchor = self._settings_anchor
        anchor.toggle(
            self._settings_body,
            prepare=self._rebuild_settings_form,
            # The card's own placement call: it reads the width the rebuild
            # just measured, where a value passed alongside ``prepare`` would
            # be the one measured BEFORE it ran.
            present=self._show_settings_popup,
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
            elif name in {"title", "signal", "overlay_signal", "size", "cell_kind"}:
                patch = {name: str(value or "")}
            else:
                return
        except (KeyError, TypeError, ValueError) as error:
            self.set_status(str(error) or type(error).__name__, error=True)
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
