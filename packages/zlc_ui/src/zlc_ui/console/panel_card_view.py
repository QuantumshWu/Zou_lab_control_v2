"""The compact console panel card.

The board owns placement; this widget owns only the titled Fluent surface and
the compact ``Setting`` / guarded ``×`` affordances.  Signal/size/update controls remain
available through the settings popup for the lightweight presenter API, but
they do not add an invented toolbar to the card face.
"""

from __future__ import annotations

from collections.abc import Mapping

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.fluent import (
    detach_widget,
    ACCENT,
    AXIS_GROUP_COLORS,
    BG,
    CARD_PAD,
    CARD_TITLE_PAD,
    FluentButton,
    FluentComboBox,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentOverlayFrame,
    FluentScrollArea,
    FluentSettingsPopupAnchor,
    FluentStatusDot,
    fluent_text_width,
    GREY,
    ORANGE,
    popup_gap,
    RADIUS,
    RED,
    scaled_px,
    show_fluent_popup_for_anchor,
    signals_blocked,
    FONT,
    fluent_font_size,)
from zlc_ui.form.form import FormChoice, FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm

from ._panel_projection import (
    interval_form_field,
    panel_state_document,
    parameter_edit_values,
    parameter_fields,
    parameter_form_spec,
    parameter_form_values,
    signal_form_runtime,
)


def _coordinate_text(value: object) -> str:
    """One pinned coordinate, as short as it can be said.

    The console resolves labelled coordinates to their display text before
    they arrive here; a bare number still prints as one.
    """

    if isinstance(value, str):
        return value
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _escaped(text: str) -> str:
    """Plain text as rich text: the strip quotes names it does not control."""

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "&nbsp;")
    )


def _set_interaction(surface: object | None, enabled: bool) -> None:
    """Project the TaskConsole selector gate onto a plot surface, when present.

    Duck-typed on purpose: a plot widget owns an interaction gate, and a label
    standing in for one in a demo does not.  Suspending the transport cancels
    any active plot gesture; the card separately sends an Off-state wheel to
    the surrounding page instead of letting the plot consume it.
    """

    gate = getattr(surface, "set_interaction_enabled", None)
    if callable(gate):
        gate(bool(enabled))


def data_structure_fragments(
    structure: object, valid: object = None
) -> tuple[tuple, tuple]:
    """The one coloured two-line rendering of a three-domain shape.

    ``valid`` maps a Repeat axis's name to how many of its samples have
    landed whole, and that number is what the axis reads where the others
    read their size: a repeat is a sample, not a coordinate, and the count
    a reader wants is the count of complete ones, which is also the number
    that moves while a run is playing.

    The separator is the multiplication SIGN, not the letter: an axis is
    very often called "x", and "(repeat)x(x)x()" asked the reader to work
    out which of those three x's was an axis name.

    A fragment is ``(text, colour, elide)``, where ``elide`` is the Qt elide
    mode this fragment gives up room by, or ``None`` for one that may not.
    The brackets and the separators are STRUCTURE -- they say how many groups
    there are and where each one ends -- so they are never shortened; only
    what a group contains is, and it goes from the MIDDLE so the first axis
    and the last one both stay legible.  That is what lets a strip too narrow
    to say everything still show how many groups there are, in which colours,
    paired line to line.

    A domain with no axes is not a factor of anything and contributes no
    group.  It used to render as an empty "()" behind its own multiplication
    sign, which reads as a bracket somebody forgot to fill.  The colour
    belongs to the DOMAIN rather than to the printed position, so the domains
    that do appear keep their own colour instead of shifting up into the
    missing one's.
    """

    landed = {str(name): int(count) for name, count in dict(valid or {}).items()}

    def count_text(name: object, size: object) -> str:
        return str(landed[str(name)]) if str(name) in landed else str(int(size))

    sizes: list[tuple[str, str | None, object]] = []
    names: list[tuple[str, str | None, object]] = []
    for index, group in enumerate(tuple(structure or ())):
        if not group:
            continue
        colour = AXIS_GROUP_COLORS[index % len(AXIS_GROUP_COLORS)]
        if sizes:
            sizes.append((" × ", None, None))
            names.append((" × ", None, None))
        for line, inner in (
            (sizes, " × ".join(count_text(name, size) for name, size in group)),
            (names, " × ".join(str(name) for name, _size in group)),
        ):
            line.append(("(", colour, None))
            line.append((inner, colour, QtCore.Qt.ElideMiddle))
            line.append((")", colour, None))
    return tuple(sizes), tuple(names)


def elide_fragments(fragments: object, metrics: object, available: int) -> tuple:
    """The same fragments, shortened to ``available`` with their structure kept.

    Eliding the LINE -- one ``elidedText`` over the whole strip's plain text --
    threw away the two things the strip is read by.  The colours pair the
    numbers above with the names below group by group, and a single escaped
    string has no fragments left to colour; the brackets say where a group
    ends, and ElideMiddle ate them from the middle outwards, so a three-domain
    shape read "(rep...ite)" -- one bracket around everything.  Measured on a
    15x15x15 scan, the names line asks for 1938 px and the widest card offers
    500, so this was not the narrow case: it was every case.

    The room is spent here instead.  Brackets and separators are kept whole,
    and the room left over is shared between the group CONTENTS by equal
    claim, not in proportion to what each asked for: a group that fits inside
    an equal share keeps its whole name and hands the surplus back, and only
    the groups that are genuinely long give anything up.  Splitting the width
    in proportion instead starved the short groups to nothing -- a scan's
    "(35) x (site)" became "() x ()" while the long one kept a stub -- which
    is the same bracket-with-nothing-in-it this strip is not allowed to show.
    """

    fragments = tuple(fragments)
    natural = tuple(
        fluent_text_width(metrics, text) for text, _colour, _elide in fragments
    )
    if available <= 0 or sum(natural) <= available:
        return fragments
    fixed = sum(
        width
        for width, (_text, _colour, elide) in zip(natural, fragments)
        if elide is None
    )
    claimants = [
        index
        for index, (_text, _colour, elide) in enumerate(fragments)
        if elide is not None and natural[index]
    ]
    if not claimants:
        return fragments
    # An ellipsis is the least a fragment may be shown as: below that the
    # bracket closes on nothing and the group reads as empty rather than as
    # abbreviated.
    floor = fluent_text_width(metrics, "\u2026")
    remaining = max(0, available - fixed)
    allotted: dict[int, int] = {}
    while claimants:
        share = remaining / len(claimants)
        satisfied = [index for index in claimants if natural[index] <= share]
        if not satisfied:
            for index in claimants:
                allotted[index] = max(int(share), floor)
            break
        for index in satisfied:
            allotted[index] = natural[index]
            remaining -= natural[index]
        claimants = [index for index in claimants if index not in set(satisfied)]
    return tuple(
        (text, colour, elide)
        if index not in allotted
        else (metrics.elidedText(text, elide, allotted[index]), colour, elide)
        for index, (text, colour, elide) in enumerate(fragments)
    )


class PanelCardView(FluentGroupBox):
    """A titled card with a replaceable QWidget surface."""

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
        # No Qt title: this card paints its own strip, two lines tall, as the
        # first row of its layout.  A QGroupBox title is a GEOMETRY
        # constraint -- what it says decides how wide the widget must be --
        # and a strip that names a signal and its shape took the card's
        # minimum width from 149 px to 704, so the board packed cards that
        # Qt then refused to shrink, and they overlapped.
        super().__init__("", parent, title_strip_px=0)
        self.panel_id = str(panel_id)
        self._base_title = str(title)
        self._surface: QtWidgets.QWidget | None = None
        self._settings_popup: FluentOverlayFrame | None = None
        self._settings_anchor: FluentSettingsPopupAnchor | None = None
        self._settings_drag_handle: FluentLabel | None = None
        self._settings_close_button: FluentButton | None = None
        self._settings_scroll: FluentScrollArea | None = None
        self._settings_body: QtWidgets.QWidget | None = None
        self._settings_form: FluentParameterForm | None = None
        #: What the card's status dot says, kept so the Setting frame can say
        #: it too, whenever it is opened.
        self._status_text = ""
        self._status_error = False
        self._settings_status: QtWidgets.QLabel | None = None
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
        self._press_at: QtCore.QPoint | None = None
        self._drag_offset: QtCore.QPoint | None = None
        self._settings_drag_offset: QtCore.QPoint | None = None
        #: Where the OPERATOR dragged this popup, for as long as it stays
        #: open.  Placement has two owners -- the anchor beside the Setting
        #: button, and the drag -- and the anchor used to re-assert itself
        #: on every content resize, so picking a fit model (which adds the
        #: parameter row) threw the popup back beside the button.

        #: Whether what this card shows will deliver again.  A live panel
        #: redraws on a beat and can be taken off the board; a saved figure
        #: does neither, and must not offer controls for both.
        self._editing_enabled = True
        self.settings_button = FluentButton("Setting", color=GREY)
        button_height = scaled_px(26, minimum=22)
        self.settings_button.setFixedSize(
            self.settings_button.fontMetrics().horizontalAdvance("Setting")
            + scaled_px(18, minimum=14),
            button_height,
        )
        self.settings_button.clicked.connect(self._open_settings)
        self.close_button = FluentButton("×", color=GREY)
        self.close_button.setFixedSize(
            scaled_px(28, minimum=24),
            button_height,
        )
        self.close_button.setToolTip(
            "Click once to arm; click again to remove this panel"
        )
        self._remove_armed = False
        self._remove_timer = QtCore.QTimer(self)
        self._remove_timer.setSingleShot(True)
        self._remove_timer.timeout.connect(self._disarm_header_remove)
        self.close_button.clicked.connect(self._header_remove_clicked)

        # The presenter marks the card a board-wide error line is about.  The
        # card body reserves no status row, so the dot rides the title strip
        # beside Setting and the full message travels as its tooltip.
        self.status_dot = FluentStatusDot(size=12)
        self.status_dot.hide()

        # The strip: one grey band across the whole top of the card, two
        # lines of text on the left and the card's one command on the right,
        # padded equally above, below and to the right.  It is a layout row
        # and not an overlay, so the card's height is the sum of what is in
        # it and the packer can simply ask.
        self._title_label = FluentLabel("")
        self._title_label.setTextFormat(QtCore.Qt.RichText)
        self._title_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._title_label.setStyleSheet("background: transparent; border: none;")
        # A strip may not decide how big a card is: it consumes whatever width
        # the card has and elides what does not fit.
        self._title_label.setMinimumWidth(0)
        self._title_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Fixed,
        )
        self._title_band = QtWidgets.QWidget(self)
        self._title_band.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._title_band.setStyleSheet(
            f"background: {BG};"
            f"border-top-left-radius: {RADIUS}px;"
            f"border-top-right-radius: {RADIUS}px;"
        )
        band_layout = QtWidgets.QHBoxLayout(self._title_band)
        band_layout.setContentsMargins(
            CARD_PAD, CARD_TITLE_PAD, CARD_TITLE_PAD, CARD_TITLE_PAD
        )
        band_layout.setSpacing(CARD_PAD)
        band_layout.addWidget(self._title_label, 1)
        band_layout.addWidget(self.status_dot, 0, QtCore.Qt.AlignVCenter)
        self._title_commands = QtWidgets.QWidget(self._title_band)
        command_layout = QtWidgets.QHBoxLayout(self._title_commands)
        command_layout.setContentsMargins(0, 0, 0, 0)
        command_layout.setSpacing(scaled_px(3, minimum=2))
        command_layout.addWidget(self.settings_button)
        command_layout.addWidget(self.close_button)
        band_layout.addWidget(self._title_commands, 0, QtCore.Qt.AlignVCenter)
        self._title_band.setFixedHeight(self._band_height())

        # The body is exactly CARD_PAD / 2 px / CARD_PAD around the surface,
        # with no invented header, footer, status row, or stretch spacer.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._title_band)
        holder = QtWidgets.QVBoxLayout()
        holder.setContentsMargins(
            CARD_PAD,
            scaled_px(2),
            CARD_PAD,
            CARD_PAD,
        )
        holder.setSpacing(0)
        outer.addLayout(holder)
        self._surface_layout = holder
        self._placeholder = FluentLabel("Pick a signal in Setting")
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        holder.addWidget(self._placeholder)
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
        # A bare reusable card has no TaskConsole switch, so its plot remains
        # interactive.  TaskConsoleHandle projects the global switch onto every
        # card before a mounted panel can be used.
        self._selectors_on = True
        self._surface_error_connected = False
        self._local_setting_error = ""
        self.set_status("", error=False)
        self.set_selectors_enabled(True)
        # An empty card still has a size: the frame it is.  Settle it now, so
        # a board can pack this card before anything is mounted in it.
        self._apply_card_size(str(self._state_projection["size"]))
        self._refresh_title_band()

    def _band_height(self) -> int:
        """Two lines of text, or the command beside them, plus its padding.

        Two LINES, whatever they say: a card whose height followed its text
        would reflow the whole board every time a signal was renamed.
        """

        lines = 2 * QtGui.QFontMetrics(self._title_label.font()).lineSpacing()
        return max(lines, self.settings_button.height()) + 2 * CARD_TITLE_PAD

    def _apply_card_size(self, size: str) -> None:
        """Reserve the room a picture of this preset needs, plus this card's own.

        Both halves are asked of whoever owns them, and both answers are here
        the moment the operator picks the preset.  How big a picture is at a
        preset is the PLOTTING package's fact -- it plans the canvas, and the
        answer does not depend on the kind or the cell count -- so the card
        asks instead of waiting for the picture to be drawn.  How much room
        the card itself takes is the card's own layout: its strip and its
        margins, measured, never the constant that was restated here and went
        wrong the moment the strip grew a second line.
        """

        from zlc_ui.board import (
            PLACEHOLDER_CELL_PX,
            panel_display_size,
            panel_size_cells,
        )

        if self._size_choices:
            width, height = panel_display_size(str(size))
        else:
            # Nobody has said which sizes exist yet, so there is no picture to
            # be the size of: this card is the empty frame it is, measured in
            # this package's own cell.
            rows, columns = panel_size_cells(str(size))
            width = columns * PLACEHOLDER_CELL_PX[0]
            height = rows * PLACEHOLDER_CELL_PX[1]
        self._reserve(width, height)

    def _reserve(self, width: int, height: int) -> None:
        """Become exactly a picture that size, framed by this card's chrome."""

        body = self._surface_layout.contentsMargins()
        outer = self.layout().contentsMargins()
        total_width = (
            int(width) + body.left() + body.right() + outer.left() + outer.right()
        )
        total_height = (
            int(height)
            + self._title_band.height()
            + body.top()
            + body.bottom()
            + outer.top()
            + outer.bottom()
        )
        if (self.width(), self.height()) == (total_width, total_height):
            return
        self.setFixedSize(total_width, total_height)
        self.geometry_changed.emit()


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

    def _apply_panel_state(
        self,
        incoming: Mapping[str, object],
        *,
        rebuild_form: bool = True,
    ) -> None:
        previous_size = str(self._state_projection.get("size") or "")
        self._state_projection = dict(incoming)
        self._base_title = incoming["title"] or "Panel"
        self._refresh_title_band()
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
        if rebuild_form:
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
        projected = dict(surface) if isinstance(surface, Mapping) else {}
        state_changed = incoming != self._state_projection
        # What the FORM is built from.  The three *_unavailable reasons used
        # to be here because the form declared them as fields; they are
        # messages, the console says them, and a message changing is not a
        # reason to rebuild a form.
        form_keys = (
            "semantic",
            "display",
            "fit",
            "science_locked",
            "paints_images",
        )
        form_changed = any(
            projected.get(key) != self._parameter_surface.get(key)
            for key in form_keys
        )
        self._parameter_surface = projected
        self._apply_panel_state(
            incoming,
            rebuild_form=state_changed or form_changed,
        )

    def _caption(self) -> str:
        """What the strip calls this panel: which card, showing what.

        DISPLAY ONLY.  The title is also the panel's editable name, and the
        console decides whether the operator has renamed a panel by asking
        whether its title still equals its signal -- so a decoration written
        into the state would permanently answer "renamed" and freeze
        auto-retitling when the signal changes.  This composes the caption at
        paint time and leaves ``_base_title`` alone; the rename field and the
        Setting form's "Panel name" go on showing the name itself.

        A signal-bound panel's title IS the signal's name, and every logic
        signal's name begins "@logic/" -- a segment that is the same on every
        panel and so tells the reader nothing.  It collapses to the "@" it
        already carries, and the card's own id goes in front, because "which
        of these cards am I looking at" is the question the strip could not
        answer before.

        The id is minted from a running serial and is never persisted, so it
        names a card within this board session, not across saves.
        """

        title = str(self._base_title)
        signal_prefix = "@logic/"
        if title.startswith(signal_prefix):
            return f"{self.panel_id}@{title[len(signal_prefix):]}"
        if title.startswith("@"):
            return f"{self.panel_id}{title}"
        # A panel with no signal is named for its kind ("Image 2"), and an
        # "@" in front of that would claim it is bound to something.
        return f"{self.panel_id} {title}"

    def _band_fragments(self) -> tuple[tuple, tuple]:
        """The strip's two lines as coloured fragments: sizes, then names.

        One line of numbers over one line of the axes they count, group by
        group in the same colour, so a reader matches "83x60" to
        "spatial-y x spatial-x" by eye instead of by counting brackets.  What
        the panel has PINNED closes the second line, because that is the part
        that changes under the operator's hands.
        """

        structure = tuple(self._parameter_surface.get("data_structure") or ())
        scope = tuple(self._parameter_surface.get("data_scope") or ())
        shape_sizes, shape_names = data_structure_fragments(
            structure, self._parameter_surface.get("data_valid")
        )
        # The caption is a NAME and names are read from the left, so it gives
        # up its tail; a shape is read from both ends and gives up its middle.
        sizes: list[tuple[str, str | None, object]] = [
            (self._caption(), None, QtCore.Qt.ElideRight)
        ]
        if shape_sizes:
            sizes.append((" ", None, None))
            sizes.extend(shape_sizes)
        names = list(shape_names)
        for label, value in scope:
            axis_colour = next(
                (
                    AXIS_GROUP_COLORS[index % len(AXIS_GROUP_COLORS)]
                    for index, group in enumerate(structure)
                    if any(str(name) == str(label) for name, _size in group)
                ),
                None,
            )
            names.append((", " if names else "", None, None))
            names.append(
                (
                    f"{label}={_coordinate_text(value)}",
                    axis_colour,
                    QtCore.Qt.ElideMiddle,
                )
            )
        return tuple(sizes), tuple(names)

    def _refresh_title_band(self) -> None:
        """Repaint the strip for the current state, projection and width."""

        band = getattr(self, "_title_band", None)
        if band is None:
            return
        label = self._title_label
        available = max(0, label.width())
        metrics = QtGui.QFontMetrics(label.font())
        lines = []
        plain_lines = []
        for fragments in self._band_fragments():
            # How wide the card must be is not the strip's to decide -- a
            # title that sets a minimum width packs cards the board then
            # cannot shrink -- so a strip too narrow to say everything gives
            # up TEXT, never structure and never colour.
            plain_lines.append("".join(text for text, _colour, _elide in fragments))
            lines.append(
                "".join(
                    _escaped(text)
                    if colour is None
                    else f'<span style="color:{colour}">{_escaped(text)}</span>'
                    for text, colour, _elide in elide_fragments(
                        fragments, metrics, available
                    )
                )
            )
        label.setText("<br>".join(lines))
        label.setToolTip("\n".join(line for line in plain_lines if line.strip()))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh_title_band()

    @property
    def surface(self) -> QtWidgets.QWidget | None:
        """The mounted view surface, so its owner can reuse or address it."""

        return self._surface

    def set_surface(self, widget: QtWidgets.QWidget | None) -> None:
        """Mount or replace an arbitrary view surface."""

        if widget is not None and not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("surface must be QWidget or None")
        if widget is not None and widget is self._surface:
            _set_interaction(widget, self._selectors_on)
            widget.show()
            return
        if widget is not None:
            _set_interaction(widget, self._selectors_on)
        if self._surface is not None:
            self._surface.removeEventFilter(self)
            if self._surface_error_connected:
                try:
                    self._surface.errorOccurred.disconnect(self.plot_error)
                except (AttributeError, RuntimeError, TypeError):
                    pass
            try:
                self._surface.surfaceChanged.disconnect(self._surface_resized)
            except (AttributeError, RuntimeError, TypeError):
                pass
            self._surface_error_connected = False
            self._surface_layout.removeWidget(self._surface)
            detach_widget(self._surface)
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
        resized = getattr(widget, "surfaceChanged", None)
        if hasattr(resized, "connect"):
            # A plot surface also says when its picture became a DIFFERENT
            # surface.  The preset already told this card how big that is, and
            # for every panel kind the two agree -- but a drawing may widen
            # its own margin (the pulse timeline does, by eight pixels), and
            # the card is the picture's frame, not the preset's.
            resized.connect(self._surface_resized)
        self._surface_layout.addWidget(widget)
        widget.show()

    def _surface_resized(self, _identity: object) -> None:
        """The mounted picture is a different surface now: frame that one."""

        surface = self._surface
        if surface is not None and surface.width() and surface.height():
            self._reserve(surface.width(), surface.height())

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        popup = self._settings_popup
        if watched is self._settings_drag_handle and popup is not None:
            if (
                event.type() == QtCore.QEvent.MouseButtonPress
                and event.button() == QtCore.Qt.LeftButton
            ):
                self._settings_drag_offset = (
                    event.globalPos() - popup.mapToGlobal(QtCore.QPoint(0, 0))
                )
                return True
            if (
                event.type() == QtCore.QEvent.MouseMove
                and self._settings_drag_offset is not None
            ):
                # Moving the frame IS the gesture -- inside its page.
                # Nothing else needs to remember where it went: no placement
                # runs again while the frame is open.  The clamp is the
                # page's edge: a Setting frame belongs to the panel area
                # and cannot be carried off it.
                popup.move(
                    self._clamped_overlay_position(
                        event.globalPos() - self._settings_drag_offset
                    )
                )
                return True
            if (
                event.type() == QtCore.QEvent.MouseButtonRelease
                and event.button() == QtCore.Qt.LeftButton
            ):
                self._settings_drag_offset = None
                return True
            if event.type() == QtCore.QEvent.Hide:
                # Closing ends the operator's placement: the next open is
                # a fresh gesture and belongs beside its button again.
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
            # Off means the plot owns no pointer gesture.  Qt does not
            # propagate an ignored wheel through this child chain, so deliver
            # it explicitly to the outer TaskConsole page.
            ancestor = self.parentWidget()
            while ancestor is not None:
                if isinstance(ancestor, QtWidgets.QAbstractScrollArea):
                    QtWidgets.QApplication.sendEvent(ancestor.viewport(), event)
                    return True
                ancestor = ancestor.parentWidget()
            return True
        if watched is self._surface and not self._selectors_on:
            if event.type() in {
                QtCore.QEvent.MouseButtonPress,
                QtCore.QEvent.MouseButtonRelease,
                QtCore.QEvent.MouseButtonDblClick,
                QtCore.QEvent.MouseMove,
            }:
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

        selected_groups = tuple(
            (str(producer), tuple((str(display), str(key)) for display, key in leaves))
            for producer, leaves in groups
        )
        selected_overlay_groups = tuple(
            (str(producer), tuple((str(display), str(key)) for display, key in leaves))
            for producer, leaves in overlay_groups
        )
        current = str(current or self._state_projection.get("signal") or "")
        overlay = str(
            overlay_current or self._state_projection.get("overlay_signal") or ""
        )
        if (
            selected_groups == self._groups
            and selected_overlay_groups == self._overlay_groups
            and current == str(self._state_projection.get("signal") or "")
            and overlay
            == str(self._state_projection.get("overlay_signal") or "")
        ):
            return
        self._groups = selected_groups
        self._overlay_groups = selected_overlay_groups
        if current:
            self._state_projection["signal"] = current
        self._state_projection["overlay_signal"] = overlay
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
        self._rebuild_settings_form(force=True)

    def set_status(self, text: str, *, error: bool) -> None:
        """Say on the card itself what a board-wide status line says about it.

        These widgets existed hidden and nothing ever showed them, so a panel
        named in a red strip line could not be told apart by looking at the
        board.  The dot rides the title strip beside the Setting button, so
        the card body still reserves no status row; hovering it reads the
        full message.  An empty text clears the mark.
        """

        value = str(text)
        self._status_text = value
        self._status_error = bool(error)
        self.status_dot.set_color(RED if error else GREY)
        self.status_dot.setToolTip(value)
        self.status_dot.setVisible(bool(value))
        self._project_settings_status()

    def _project_settings_status(self) -> None:
        """The same condition, written where the operator goes to fix it.

        A red dot with a tooltip names the panel; it does not tell the
        operator, who has opened Setting to put it right, WHAT is wrong.
        The condition heads the Setting frame in the dot's colour, and
        leaves with it.
        """

        label = self._settings_status
        if label is None:
            return
        label.setText(self._status_text)
        label.setStyleSheet(
            f'QLabel {{ color: {RED if self._status_error else GREY}; '
            f'font: {fluent_font_size()}pt "{FONT}"; background: transparent; '
            f"border: none; }}"
        )
        label.setVisible(bool(self._status_text))
        self._sync_settings_body_size()

    def set_selectors_enabled(self, enabled: bool) -> None:
        """Give the plot all pointer gestures when On and none when Off."""

        self._selectors_on = bool(enabled)
        _set_interaction(self._surface, self._selectors_on)

    @property
    def selectors_enabled(self) -> bool:
        return self._selectors_on

    def set_editing_enabled(self, enabled: bool) -> None:
        """Gate persisted panel edits without disabling plot interaction."""

        self._editing_enabled = bool(enabled)
        self.settings_button.setEnabled(self._editing_enabled)
        if not self._editing_enabled:
            self.retire_settings_popup()

    def retire_settings_popup(self) -> None:
        """Hide the page-owned Setting overlay without removing its Panel."""

        if self._settings_popup is not None:
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
        fields: list[FormFieldProps] = []
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
        if bool(self._parameter_surface.get("paints_images")):
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
        if self._interval_choices:
            fields.append(
                interval_form_field(
                    self._interval_choices,
                    state["interval_ms"],
                )
            )
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
                        unit=field.unit,
                        description=field.description,
                        minimum=field.minimum,
                        maximum=field.maximum,
                        choices=field.choices,
                        cycle_choices=field.cycle_choices,
                        cycle_label=field.cycle_label,
                        allow_blank=field.allow_blank,
                        unavailable_reason=field.unavailable_reason,
                        automatic=field.automatic,
                    )
                )
            # A REASON IS NOT A SETTING, so the form does not declare one.
            # Each section's "why this could not be applied" used to be
            # appended here as a field: kind "text", so it became a line
            # edit labelled with the bare section name, sitting under the
            # controls it was about and reading as another input.  The
            # console already has one place for what it could not do -- the
            # header line, and the standing mark on the card -- and that is
            # where these go.
        return FormSpec(tuple(fields))

    def _form_values(self) -> dict[str, object]:
        signal = str(self._state_projection.get("signal") or "") or None
        values: dict[str, object] = {
            "title": str(self._state_projection.get("title") or "Panel"),
            "signal": signal,
            "size": str(self._state_projection.get("size") or self._default_size),
        }
        if bool(self._parameter_surface.get("paints_images")):
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
        if self._interval_choices:
            values["interval_ms"] = int(self._state_projection["interval_ms"])
        for section in ("semantic", "display", "fit"):
            declared_fields = tuple(
                parameter_fields(self._parameter_surface, section)
            )
            declared_values = parameter_form_values(
                declared_fields,
            )
            for key, value in declared_values.items():
                values[f"{section}__{key}"] = value
        return values

    def _rebuild_settings_form(self, *, force: bool = False) -> None:
        if self._settings_form is not None:
            spec = self._form_spec()
            values = self._form_values()
            if force or not self._settings_form.adopt_projection(spec, values):
                self._settings_form.reconcile(spec, values)
            self._apply_settings_enabled_state()
            self._sync_settings_body_size()

    def _apply_settings_enabled_state(self) -> None:
        """Apply standing edit policy without one-frame enabled flicker."""

        form = self._settings_form
        if form is None:
            return
        science_locked = bool(self._parameter_surface.get("science_locked"))
        form.widget_for("signal").setEnabled(
            bool(self._groups) and not science_locked
        )
        for key in ("overlay_signal", "cell_kind"):
            if key in form.spec.keys:
                form.widget_for(key).setEnabled(not science_locked)
        if science_locked:
            for field in parameter_fields(self._parameter_surface, "semantic"):
                key = f"semantic__{field['key']}"
                if key in form.spec.keys:
                    form.widget_for(key).setEnabled(False)

    def _sync_settings_body_size(self) -> None:
        """Measure the content.  Do NOT move the frame it is sitting in.

        This used to re-run the whole placement -- size AND position, derived
        afresh from the content -- whenever the form's required width or
        height changed.  So adding one control resized the operator's frame
        underneath them, and everything in it moved.

        An open frame's geometry belongs to the operator: they opened it
        there, they may have carried it somewhere else, and a control
        appearing is not a reason to take that away.  The measurements below
        still matter -- the body's minimum height is what lets the scroll
        area reach the last row, and the width is what the NEXT open is
        sized from -- but they end here.  Content goes into the scroll.
        """

        body = self._settings_body
        scroll = self._settings_scroll
        if body is None or body.layout() is None or scroll is None:
            return
        margins = body.layout().contentsMargins()
        required_width = (
            self._settings_form.minimum_content_width()
            + margins.left()
            + margins.right()
        )
        self._settings_content_width = required_width
        required_height = body.layout().sizeHint().height()
        if body.minimumHeight() != required_height:
            body.setMinimumHeight(required_height)
        # Finish the job here rather than a layout turn later.  The
        # re-placement that used to follow was synchronous, so removing it
        # left the scroll content one turn stale -- one frame of the old
        # height on screen.  This is what widgetResizable would do next
        # turn, done now, and it touches the content only.
        body.resize(
            body.width(),
            max(required_height, scroll.viewport().height()),
        )
        popup = self._settings_popup
        if popup is None or not popup.isVisible():
            return
        # THE ONE EDGE THAT MAY MOVE.  The frame never moves and never
        # changes height while it is open, but the scroll area is
        # width-bounded with its horizontal bar off, so content that needs
        # more width than the viewport has is simply cut off.  Widen the
        # RIGHT edge by the shortfall: every row is laid out from the left,
        # so nothing already on screen moves.
        shortfall = required_width - scroll.viewport().width()
        if shortfall <= 0:
            return
        page = popup.parentWidget()
        limit = min(
            popup.width() + shortfall,
            max(1, page.width() - popup.x()),
        )
        if limit > popup.width():
            popup.resize(limit, popup.height())
        # The page is a hard wall now -- the frame cannot grow past the
        # panel area the way it once grew to the screen edge.  When the
        # form is wider than the page can afford, the content scrolls
        # sideways instead of being cut off.
        self._settings_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
            if limit < popup.width() + shortfall
            else QtCore.Qt.ScrollBarAlwaysOff
        )

    def _settings_available_height(self) -> int:
        """Vertical room the popup may take: from below Setting to the card foot."""

        anchor_bottom = self.settings_button.mapTo(
            self,
            QtCore.QPoint(0, self.settings_button.height()),
        ).y()
        return max(
            1,
            self.rect().bottom()
            - anchor_bottom
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
            # ONE height rule.  There used to be a second one -- no cap at
            # all once the operator had dragged the frame -- which only ever
            # ran on the while-visible re-placement, and made the frame grow
            # by 131 px the first time anything changed after a drag.
            # Placement now happens once, when the frame opens, and it opens
            # beside its button.
            maximum_height=self._settings_available_height(),
            content_width=self._settings_content_width,
        )

    def _open_settings(self) -> None:
        if self._settings_popup is None:
            # A frame INSIDE the page, not a window.  It lives in the panel
            # area's viewport, so it is clipped to that area, hides and
            # returns with the tab, and travels/minimises/closes with the
            # window without any desktop machinery.  The page owns the
            # widget; the card's death takes it along explicitly.
            popup = FluentOverlayFrame(self._settings_overlay_host())
            self.destroyed.connect(popup.deleteLater)
            layout = QtWidgets.QVBoxLayout(popup)
            pad = max(1, scaled_px(10))
            layout.setContentsMargins(pad, pad, 0, pad)
            # The title row sits in the same air above and below it.  With
            # no gap under it the row hugged the first field, and its close
            # button -- centred in the row -- read as sitting low in a blank
            # that was all above it.
            layout.setSpacing(pad)
            header = QtWidgets.QWidget(popup)
            header.setStyleSheet("background: transparent;")
            header_layout = QtWidgets.QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, pad, 0)
            header_layout.setSpacing(max(1, scaled_px(5)))
            drag_handle = FluentLabel(
                f"Setting · {self.panel_id}",
                header,
            )
            drag_handle.setCursor(QtCore.Qt.SizeAllCursor)
            drag_handle.installEventFilter(self)
            header_layout.addWidget(drag_handle, 1)
            close_button = FluentButton("×", header, color=GREY)
            close_button.setFixedSize(
                scaled_px(24, minimum=20),
                scaled_px(24, minimum=20),
            )
            close_button.setToolTip("Close settings")
            close_button.clicked.connect(self.retire_settings_popup)
            header_layout.addWidget(close_button, 0)
            layout.addWidget(header)
            scroll = FluentScrollArea(popup)
            body = QtWidgets.QWidget()
            body.setStyleSheet("background: transparent;")
            body_layout = QtWidgets.QVBoxLayout(body)
            body_layout.setContentsMargins(0, 0, pad, 0)
            body_layout.setSpacing(max(1, scaled_px(5)))
            status = FluentLabel("", body)
            status.setWordWrap(True)
            status.hide()
            body_layout.addWidget(status)
            self._settings_status = status
            self._settings_form = FluentParameterForm(
                self._form_spec(),
                self._form_values(),
                runtime=self._signal_runtime,
                parent=body,
            )
            self._settings_form.changed.connect(self._setting_changed)
            self._apply_settings_enabled_state()
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
            self._project_settings_status()
            self._settings_close_button = close_button
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

    def _settings_overlay_host(self) -> QtWidgets.QWidget:
        """The page area the Setting frame lives in and is clipped to.

        The nearest scroll viewport is the panel area of the tab page --
        the same ancestor the wheel forwarder walks to -- so the frame
        hides with the tab and cannot leave the panel region.  A bare
        card (tests, previews) is its own page.
        """

        ancestor = self.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QtWidgets.QAbstractScrollArea):
                return ancestor.viewport()
            ancestor = ancestor.parentWidget()
        return self.window()

    def _clamped_overlay_position(
        self, global_top_left: QtCore.QPoint
    ) -> QtCore.QPoint:
        """The frame's page-local position, kept wholly inside its page."""

        popup = self._settings_popup
        page = popup.parentWidget()
        local = page.mapFromGlobal(global_top_left)
        return QtCore.QPoint(
            min(max(local.x(), 0), max(0, page.width() - popup.width())),
            min(max(local.y(), 0), max(0, page.height() - popup.height())),
        )

    def _request_edit(self) -> None:
        self.retire_settings_popup()
        self.edit_requested.emit()

    def _request_remove(self) -> None:
        self.retire_settings_popup()
        self.remove_requested.emit()

    def _header_remove_clicked(self) -> None:
        if self._remove_armed:
            self._remove_timer.stop()
            self._disarm_header_remove()
            self.remove_requested.emit()
            return
        self._remove_armed = True
        self.close_button.set_color(RED)
        self.close_button.setToolTip("Click again to remove this panel")
        application = QtWidgets.QApplication.instance()
        interval = 400 if application is None else application.doubleClickInterval()
        self._remove_timer.start(max(250, int(interval)))

    def _disarm_header_remove(self) -> None:
        self._remove_armed = False
        self.close_button.set_color(GREY)
        self.close_button.setToolTip(
            "Click once to arm; click again to remove this panel"
        )

    def _setting_changed(self, key: str) -> None:
        if self._settings_form is None:
            return
        name = str(key)
        try:
            value = self._settings_form.read_value(name)
            section, separator, parameter_key = name.partition("__")
            if separator and section in {"semantic", "display", "fit"}:
                fields = tuple(
                    parameter_fields(self._parameter_surface, section)
                )
                selected = next(
                    field
                    for field in fields
                    if str(field["key"]) == parameter_key
                )
                owner = str(selected.get("edit_section") or section)
                owned = tuple(
                    field
                    for field in fields
                    if str(field.get("edit_section") or section) == owner
                )
                patch: dict[str, object] = {
                    owner: parameter_edit_values(
                        owned,
                        parameter_key,
                        lambda key: self._settings_form.read_value(
                            f"{section}__{key}"
                        ),
                    )
                }
            elif name == "interval_ms":
                patch = {name: int(value)}
            elif name in {"title", "signal", "overlay_signal", "size", "cell_kind"}:
                patch = {name: str(value or "")}
            else:
                return
        except (KeyError, TypeError, ValueError) as error:
            self._local_setting_error = str(error) or type(error).__name__
            self.set_status(self._local_setting_error, error=True)
            return
        if self._local_setting_error:
            self._local_setting_error = ""
            self.set_status("", error=False)
        self.state_changed.emit(patch)

    def _choice_groups(self, field: str):
        if str(field) == "signal":
            return self._groups
        if str(field) == "overlay_signal":
            return self._overlay_groups
        return ()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == QtCore.Qt.LeftButton:
            self._press_at = event.pos()
            self._drag_offset = None
            self.raise_()
            self.grabMouse()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._press_at is not None and event.buttons() & QtCore.Qt.LeftButton:
            if self._drag_offset is None:
                # A PRESS IS NOT A DRAG.  A zero-motion click emitted a
                # drop, and a drop records this card as the board's
                # anchor -- kept until the card leaves the board, so every
                # later repack placed it at the absolute (col, row) it
                # happened to hold and stacked the others around it.  One
                # click to look at a card, and widening the window
                # rearranged everything except that card.  Where the line
                # between a click and a drag is, is the window system's to
                # say, which is what the pulse schedule already asks.
                travelled = (event.pos() - self._press_at).manhattanLength()
                if travelled < QtWidgets.QApplication.startDragDistance():
                    super().mouseMoveEvent(event)
                    return
                self._drag_offset = self._press_at
                # The closed hand marks the grab, so it appears when the
                # card actually starts following the pointer.
                self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.move(self.mapToParent(event.pos() - self._drag_offset))
            point = (int(event.pos().x()), int(event.pos().y()))
            self.drag_started.emit(point)
            self.drag_moved.emit(point)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == QtCore.Qt.LeftButton and self._press_at is not None:
            dragged = self._drag_offset is not None
            self._press_at = None
            self._drag_offset = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            self.releaseMouse()
            # A press that reaches release without ever crossing the
            # threshold IS a click, and a click drops nothing.
            if dragged:
                self.dropped.emit((int(event.pos().x()), int(event.pos().y())))
        super().mouseReleaseEvent(event)


__all__ = ["PanelCardView"]
