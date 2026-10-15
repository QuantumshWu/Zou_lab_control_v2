"""The compact console panel card.

The board owns placement; this widget owns only the titled Fluent surface and
the compact ``Setting`` / guarded ``×`` affordances.  Signal, size and update
interval are edited in the Setting form and nowhere else: the card keeps no
second control for any of them.  A drawn card is carried by its title strip,
its body being its plot's; an empty card is carried by the whole of it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data.units import figure_padded
from zlc_ui.fluent import (
    detach_widget,
    AXIS_GROUP_COLORS,
    BG,
    CARD_TITLE_PAD,
    FIELD_BORDER_PX,
    FluentButton,
    FluentGroupBox,
    FluentLabel,
    FluentOverlayFrame,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingsPopupAnchor,
    FluentStatusDot,
    fluent_text_width,
    MUTED_INK,
    RADIUS,
    card_pad,
    control_height,
    row_gap,
    scaled_px,
    section_gap,
    set_status_ink,
    show_fluent_popup_for_anchor,)
from zlc_ui.form.form import FormChoice, FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm, _form_label_width, being_edited

from ._panel_projection import (
    interval_form_field,
    panel_state_document,
    parameter_edit_values,
    parameter_fields,
    parameter_form_spec,
    parameter_form_values,
    signal_form_runtime,
    size_form_field,
)


#: The sections a plot owner declares a panel's parameters in, after the
#: panel's own "Panel" section, each with its title.
PARAMETER_SECTIONS = (("semantic", "Semantic"), ("display", "Display"), ("fit", "Fit"))


def panel_section(
    title: str,
    form: FluentParameterForm,
    *,
    note: QtWidgets.QWidget | None = None,
) -> tuple[QtWidgets.QVBoxLayout, FluentSectionLabel]:
    """One titled section of a panel's settings, and its title: the title,
    the section's ``note`` if it has one, then its form, a row gap apart.

    THE one construction for both places a panel is set up.  The Edit page
    lays each section out as a card of its own, the Setting frame as a
    block of its body a section gap from the next; either way the title is
    the one section title, a control high like every card's title row, so
    a section is titled and starts its rows the same in both.
    """

    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(row_gap())
    heading = FluentSectionLabel(title)
    layout.addWidget(heading)
    if note is not None:
        layout.addWidget(note)
    layout.addWidget(form)
    return layout, heading


def share_label_column(forms, *, widths: tuple[int, ...] = ()) -> int:
    """Give ``forms`` one label column, the widest any of their rows needs,
    and answer it (0 when nothing needs one).

    Each form sizes a column to its own words, so sections stacked down one
    page would start their controls at as many x.  ``widths`` are the
    columns the page's other rows need (Save figure's rows, a producer
    contribution's labels); the caller gives those rows the answer.
    """

    forms = tuple(forms)
    column = max(
        (*widths, *(_form_label_width(form.spec.fields) for form in forms if form.spec.fields)),
        default=0,
    )
    if column:
        for form in forms:
            form.set_label_width(column)
    return column


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
    """Plain text as rich text: the strip quotes names it does not control.

    A figure space pads a climbing count to its axis size; rich text folds a
    run of them into one ordinary space unless it is kept as written.
    """

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "&nbsp;")
        .replace("\u2007", '<span style="white-space:pre">\u2007</span>')
    )


def relay_surface_errors(widget: object, relay: object) -> bool:
    """Connect a mounted plot surface's ``errorOccurred`` to ``relay``.

    ONE rule for every surface mount -- the card's and the editor's.  A
    plot surface says what a gesture could not do through this one signal
    (a pointer front that lost its surface, a refused RGBA) and the mount
    relays it so the console can report it; a mount that did not connect
    it kept those refusals fully silent.  Returns whether a connection was
    made, which is what its release needs to know.
    """

    source = getattr(widget, "errorOccurred", None)
    if not hasattr(source, "connect"):
        return False
    source.connect(relay)
    return True


def release_surface_errors(widget: object, relay: object, connected: bool) -> None:
    """Undo :func:`relay_surface_errors` when the surface leaves the mount,
    so a retired surface's refusals are not reported as the panel's."""

    if not connected:
        return
    try:
        widget.errorOccurred.disconnect(relay)
    except (AttributeError, RuntimeError, TypeError):
        pass


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


def _shows_unrecorded_pick(
    form: FluentParameterForm | None, state: Mapping[str, object]
) -> bool:
    """Whether a row naming WHAT the panel draws shows a pick its record lacks.

    A refused signal, overlay or cell-kind pick comes back as the record the
    presenter kept, so the projection equals the one the form was built from,
    and a form reconciled only on a changed projection went on showing the
    refused pick over a card drawing something else.  Asked by the rule
    reconcile writes by: a choice row always, a keyed row once the operator
    is no longer inside it.
    """

    if form is None:
        return False
    for field in form.spec.fields:
        if field.key not in ("signal", "overlay_signal", "cell_kind"):
            continue
        if field.kind != "choice" and being_edited(form.widget_for(field.key)):
            continue
        try:
            shown = form.read_value(field.key)
        except (KeyError, TypeError, ValueError):
            continue
        if str(shown or "") != str(state.get(field.key) or ""):
            return True
    return False


def data_structure_fragments(
    structure: object, valid: object = None
) -> tuple[tuple, tuple]:
    """The one coloured two-line rendering of a three-domain shape.

    ``valid`` is one integer conditional count per Repeat axis,
    in that group's own order. Other domains still read their full size.
    These counts come from the current data position, not plot selection;
    their product is not a total sample count. By POSITION within the
    Repeat group, never by name: an axis is very often called "repeat", two
    Repeat axes may both be, and so may a Point axis, and a name lookup
    handed the last count to every one of them.

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

    An axis-free domain still exists and has one scalar carrier. Its number
    is 1 and its name is a dash, not an invented axis. Keep all three domain
    groups and their colours, including after a manual axis deletion.
    """

    landed = tuple(valid or ())

    def count_text(group_index: int, position: int, size: object) -> str:
        if group_index == 0 and position < len(landed):
            # In the size's width: 9 -> 10 -> 100 moved every group after it.
            return figure_padded(int(landed[position]), int(size))
        return str(int(size))

    sizes: list[tuple[str, str | None, object]] = []
    names: list[tuple[str, str | None, object]] = []
    for index, group in enumerate(tuple(structure or ())):
        colour = AXIS_GROUP_COLORS[index % len(AXIS_GROUP_COLORS)]
        if sizes:
            sizes.append((" × ", None, None))
            names.append((" × ", None, None))
        for line, inner in (
            (
                sizes,
                " × ".join(
                    count_text(index, position, size)
                    for position, (_name, size) in enumerate(group)
                ) or "1",
            ),
            (names, " × ".join(str(name) for name, _size in group) or "—"),
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

    remove_requested = QtCore.pyqtSignal()
    edit_requested = QtCore.pyqtSignal()
    state_changed = QtCore.pyqtSignal(object)
    #: The three steps of carrying a card by its strip, each with the
    #: pointer's GLOBAL position: the board decides where a card lands from
    #: the pointer, and scrolls under a pointer that stands still, so a
    #: card-local point would be stale by the time it is read.
    #: ``drag_started`` once, when the press crosses the drag threshold;
    #: ``drag_moved`` after every move the card follows; ``dropped`` on the
    #: release that ends a drag (never on a click).
    drag_started = QtCore.pyqtSignal(tuple)
    drag_moved = QtCore.pyqtSignal(tuple)
    dropped = QtCore.pyqtSignal(tuple)
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
        #: Whether the mounted surface has shown a picture this card framed.
        #: From then on the picture is the card's size; a preset only sizes
        #: the empty frame.
        self._surface_framed = False
        self._settings_popup: FluentOverlayFrame | None = None
        self._settings_anchor: FluentSettingsPopupAnchor | None = None
        self._settings_drag_handle: FluentLabel | None = None
        self._settings_close_button: FluentButton | None = None
        self._settings_scroll: FluentScrollArea | None = None
        self._settings_body: QtWidgets.QWidget | None = None
        #: The Setting frame's sections, as on the Edit page: the Panel
        #: form, then one form per plot-owner section, each in its block.
        self._settings_form: FluentParameterForm | None = None
        self._settings_parameter_forms: dict[str, FluentParameterForm] = {}
        #: Each section's title, by section, shown with its form.
        self._settings_sections: dict[str, FluentSectionLabel] = {}
        # None means current; a deferred request retains whether keyed choices changed.
        self._settings_pending_rebuild: bool | None = None
        #: What the card's status dot says, kept so the Setting frame can say
        #: it too, whenever it is opened.
        self._status_text = ""
        self._status_error = False
        #: The presenter's standing condition for this panel, which a local
        #: Setting read error covers while it stands and never erases.
        self._presented_status: tuple[str, bool] = ("", False)
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
        #: {signal key: what a person calls it}, as the presenter says it.
        self._signal_texts: dict[str, str] = {}
        self._signal_runtime = signal_form_runtime(
            self._choice_groups, lambda: self._signal_texts
        )
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
        self.settings_button = FluentButton("Setting")
        # The strip's two commands are the console's controls: as tall as
        # every other button on the page (a FluentButton's own height), and
        # the one-glyph remover square at that height.  The strip's two text
        # lines are taller still, so the strip keeps its height.
        control = self.settings_button.minimumHeight()
        self.settings_button.setFixedHeight(control)
        self.settings_button.clicked.connect(self._open_settings)
        # A remover at rest is the quiet ghost every one-glyph × is -- red
        # only under the pointer.  Removing a panel takes two clicks: the
        # first arms the white-on-red confirm, the second removes.
        self.close_button = FluentButton(
            "×", role="destructive", confirm_twice="remove this panel"
        )
        self.close_button.setFixedSize(control, control)
        self.close_button.confirmed.connect(self.remove_requested.emit)

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
        # The band is a fill INSIDE the card's edge: the card paints its
        # own continuous edge, and an opaque band laid from (0, 0) painted
        # over the top and side of it -- with an unscaled corner that left
        # a fragment of the arc showing at three device pixels a point.  It
        # starts one edge-width in, with the inner, concentric radius.
        pad = self.strip_text_inset()
        title_pad = scaled_px(CARD_TITLE_PAD)
        inner_radius = max(0, scaled_px(RADIUS) - FIELD_BORDER_PX)
        self._title_band = QtWidgets.QWidget(self)
        self._title_band.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._title_band.setStyleSheet(
            f"background: {BG};"
            f"border-top-left-radius: {inner_radius}px;"
            f"border-top-right-radius: {inner_radius}px;"
        )
        band_layout = QtWidgets.QHBoxLayout(self._title_band)
        band_layout.setSpacing(pad)
        # Centred on the band, so a one-line title sits level with Setting.
        band_layout.addWidget(self._title_label, 1, QtCore.Qt.AlignVCenter)
        band_layout.addWidget(self.status_dot, 0, QtCore.Qt.AlignVCenter)
        self._title_commands = QtWidgets.QWidget(self._title_band)
        command_layout = QtWidgets.QHBoxLayout(self._title_commands)
        command_layout.setContentsMargins(0, 0, 0, 0)
        command_layout.setSpacing(row_gap())
        command_layout.addWidget(self.settings_button)
        command_layout.addWidget(self.close_button)
        band_layout.addWidget(self._title_commands, 0, QtCore.Qt.AlignVCenter)
        # The hand is shown over what carries the card (see handle_rect and
        # _show_grip); the strip's two commands are clicked, not carried.
        self._title_commands.setCursor(QtCore.Qt.ArrowCursor)
        band_height = self._band_height()
        self._title_band.setFixedHeight(band_height)
        # Setting and x sit concentric in the band's corner: as far from its
        # right end as from its top and bottom, not pressed against the edge.
        trailing = max(title_pad, (band_height - self._title_commands.sizeHint().height()) // 2)
        self._commands_trailing = trailing
        band_layout.setContentsMargins(pad - FIELD_BORDER_PX, title_pad, trailing, title_pad)

        # The body is exactly CARD_PAD / 2 px / CARD_PAD around the surface,
        # counted from the card's edge, with no invented header, footer,
        # status row, or stretch spacer.
        outer = QtWidgets.QVBoxLayout(self)
        edge = FIELD_BORDER_PX
        outer.setContentsMargins(edge, edge, edge, 0)
        outer.setSpacing(0)
        outer.addWidget(self._title_band)
        holder = QtWidgets.QVBoxLayout()
        holder.setContentsMargins(pad - edge, scaled_px(2), pad - edge, pad)
        holder.setSpacing(0)
        outer.addLayout(holder)
        # An empty card is taller than what it lays out -- a fixed strip and
        # an empty body -- and a layout shorter than its widget is CENTRED
        # in it unless told otherwise, which put an empty card's strip in
        # the middle of its frame.  A card with a picture is exactly its
        # layout, so this moves nothing there.
        outer.setAlignment(QtCore.Qt.AlignTop)
        self._surface_layout = holder
        # No body placeholder: an empty card is the blank frame its picture
        # will fill, and its strip's second line says why it holds none (see
        # _band_fragments).  A sentence centred in the body was a second
        # layout -- one strip line instead of two, the caption sitting lower
        # than every drawn neighbour's -- and it went on saying "pick a
        # signal" on a card whose strip already named one.
        # A bare reusable card has no TaskConsole switch, so its plot remains
        # interactive.  TaskConsoleHandle projects the global switch onto every
        # card before a mounted panel can be used.
        self._selectors_on = True
        self._surface_error_connected = False
        self._show_grip(QtCore.Qt.OpenHandCursor)
        #: (form, row, text) of a Setting value this card could not read;
        #: the row is kept so a re-projection knows whether it replaced it.
        self._local_setting_error: tuple[FluentParameterForm | None, str, str] = (None, "", "")
        self.set_status("", error=False)
        self.set_selectors_enabled(True)
        # An empty card still has a size: the frame it is.  Settle it now, so
        # a board can pack this card before anything is mounted in it.
        self._apply_card_size(str(self._state_projection["size"]))
        self._refresh_title_band()

    @staticmethod
    def strip_text_inset() -> int:
        """How far in from the card's left edge the strip's text starts.

        One card pad, the same as the picture below it.  A board lines its
        own words up with it (see ``ConsoleBoardView.text_left``).
        """

        return card_pad()

    def strip_rect(self) -> QtCore.QRect:
        """The card's title strip, across its whole width, in card coordinates."""

        band = self._title_band.geometry()
        return QtCore.QRect(0, 0, self.width(), band.bottom() + 1)

    def handle_rect(self) -> QtCore.QRect:
        """The part of the card it is carried by, in card coordinates.

        A drawn card is carried by its strip: its body is its plot's (a
        gesture with Selectors on, nothing with them off).  An empty card's
        body is a blank frame, so the whole card carries it -- the board
        drops a card in the row its strip is in, wherever on the card it is
        held.
        """

        return self.rect() if self._surface is None else self.strip_rect()

    def _show_grip(self, cursor: QtCore.Qt.CursorShape) -> None:
        """Show ``cursor`` -- the open hand, or the closed one while carried --
        over what carries this card: its strip, and an empty card's body."""

        self._title_band.setCursor(cursor)
        if self._surface is None:
            self.setCursor(cursor)
        else:
            self.unsetCursor()

    def keep_commands_within(self, right: int) -> None:
        """Keep Setting and × on screen: ``right`` (card x) is where what
        the viewport shows of this card ends.

        A card that runs past it has the two commands ride that edge, as far
        in from it as they stand from the card's own end, so a card wider
        than its window is edited and removed without scrolling sideways to
        find its corner; scrolled into view, they are back in the corner.
        They never ride past the strip's text edge."""

        band = self._title_band.layout()
        margins = band.contentsMargins()
        base = self._commands_trailing
        room = self._title_band.width() - (band.minimumSize().width() - margins.right() + base)
        trailing = base + max(0, min(self.width() - int(right), room))
        if margins.right() == trailing:
            return
        band.setContentsMargins(margins.left(), margins.top(), trailing, margins.bottom())
        band.activate()
        # The strip's words end where the commands now start.
        self._refresh_title_band()

    def follow_pointer(self, pointer: QtCore.QPoint) -> None:
        """Keep the point this card is held by under ``pointer`` (global).

        Asked on every move, and by a board that scrolled under a pointer
        that stood still.  A card not being carried stays where it is.
        """

        parent = self.parentWidget()
        if self._drag_offset is None or parent is None:
            return
        self.move(parent.mapFromGlobal(pointer) - self._drag_offset)

    def _band_height(self) -> int:
        """Two lines of text, or the command beside them, plus its padding.

        Two LINES, whatever they say: a card whose height followed its text
        would reflow the whole board every time a signal was renamed.  They
        are measured as the strip's rich text stacks them, which rounds each
        line UP to a whole pixel: twice the font's integer line spacing was
        two pixels short of that at the body font, and the second line's
        descenders -- the g of "Setting", the y of "spatial-y" -- were cut
        off by the label's edge.
        """

        probe = QtGui.QTextDocument()
        probe.setDocumentMargin(0)
        probe.setDefaultFont(self._title_label.font())
        probe.setHtml("&nbsp;<br>&nbsp;")
        lines = math.ceil(probe.size().height())
        return max(lines, self.settings_button.height()) + 2 * scaled_px(CARD_TITLE_PAD)

    def _apply_card_size(self, size: str) -> None:
        """Reserve the room a picture of this preset needs, plus this card's own.

        Only for a card that holds no picture yet: a board packs an empty
        card before anything is drawn, and how big a picture is at a preset
        is the PLOTTING package's fact -- it plans the canvas, and the answer
        does not depend on the kind or the cell count -- so the empty card
        asks.  A card that frames a picture is that picture's size and moves
        with it, never ahead of it: a preset picked by the operator resizes
        the card when the picture drawn at it lands, in the same event, so
        the picture, its frame and the strip change together -- not a frame
        that jumped first and stood around the old picture, strip re-elided,
        until the new one was painted.  How much room the card itself takes
        is the card's own layout: its strip and its margins, measured, never
        the constant that was restated here and went wrong the moment the
        strip grew a second line.
        """

        if self._surface_framed:
            return

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
        self._base_title = self._panel_name()
        self._refresh_title_band()
        if incoming["size"] != previous_size:
            self._apply_card_size(str(incoming["size"]))
        if rebuild_form:
            self._rebuild_settings_form()

    @property
    def panel_size(self) -> str:
        return str(self._state_projection["size"])

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
            # What "Panel name" shows: a renamed node renames its panels.
            "panel_name",
        )
        form_changed = any(
            projected.get(key) != self._parameter_surface.get(key)
            for key in form_keys
        )
        self._parameter_surface = projected
        self._apply_panel_state(
            incoming,
            rebuild_form=(
                state_changed
                or form_changed
                or _shows_unrecorded_pick(self._settings_form, incoming)
            ),
        )

    def _panel_name(self) -> str:
        """What the panel is called, as its presenter named it.

        DISPLAY ONLY, like the caption: an automatic title is the signal's
        key, which the presenter reads into the producer's name and output.
        A card nothing has projected yet is called by its record's title.
        """

        return str(
            self._parameter_surface.get("panel_name")
            or self._state_projection.get("title")
            or "Panel"
        )

    def _caption(self) -> str:
        """What the strip calls this panel, as its presenter captioned it.

        DISPLAY ONLY, and not this card's to compose: a signal-bound panel is
        captioned from its signal's name, and that name's grammar belongs to
        the runtime, which this package does not import.  The presenter reads
        it there and projects the caption beside the data shape, so the
        strip renders exactly what it was given.  The title is untouched --
        it is the panel's editable name, which the rename field and the
        Setting form's "Panel name" go on showing -- and a card nothing has
        captioned yet is called by that name.
        """

        return str(self._parameter_surface.get("caption") or self._base_title)

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
        # The caption is a NAME: a typed one gives up its tail, a signal's
        # producer its middle (below); a shape is read from both ends and
        # gives up its middle.  The card's own id leads it in muted ink: it
        # answers "which card", and the name after it is what the operator
        # reads.
        caption = self._caption()
        rest = (
            caption[len(self.panel_id):]
            if caption.startswith(self.panel_id) and len(caption) > len(self.panel_id)
            else ""
        )
        head, slash, output = rest.rpartition("/")
        sizes: list[tuple[str, str | None, object]]
        if rest.startswith("@") and slash and len(head) > 1:
            # A signal reads ``<producer>/<output>``.  The producer is a name
            # the operator may have made long, and two names may differ only
            # at their end: it gives up its middle, and the output -- which
            # of the producer's signals this is -- stays whole.
            sizes = [
                (self.panel_id, MUTED_INK, None),
                (head, None, QtCore.Qt.ElideMiddle),
                (slash + output, None, None),
            ]
        elif rest:
            sizes = [(self.panel_id, MUTED_INK, None), (rest, None, QtCore.Qt.ElideRight)]
        else:
            sizes = [(caption, None, QtCore.Qt.ElideRight)]
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
        if not names and self._surface is None:
            # A card with no picture says why on the line a drawn card names
            # its axes on, so the two strips are laid out alike: the caption
            # on the first line level with every neighbour's, not centred
            # alone in the band half a line lower.
            names.append((self._vacancy_note(), MUTED_INK, QtCore.Qt.ElideRight))
        return tuple(sizes), tuple(names)

    def _vacancy_note(self) -> str:
        """Why this card holds no picture, in the operator's next step."""

        if str(self._state_projection.get("signal") or ""):
            return "No data yet"
        return "Choose a signal in Setting"

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
        # A line with nothing to say is not drawn: a blank second line held
        # a one-line title at the top of the band, above its buttons.
        label.setText("<br>".join(
            line for line, plain in zip(lines, plain_lines) if plain.strip()
        ))
        label.setToolTip("\n".join(line for line in plain_lines if line.strip()))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh_title_band()

    @property
    def surface(self) -> QtWidgets.QWidget | None:
        """The mounted view surface, so its owner can reuse or address it."""

        return self._surface

    def set_surface(self, widget: QtWidgets.QWidget | None) -> None:
        """Mount or replace an arbitrary view surface; ``None`` empties the frame."""

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
            release_surface_errors(
                self._surface, self.plot_error, self._surface_error_connected
            )
            try:
                self._surface.surfaceChanged.disconnect(self._surface_resized)
            except (AttributeError, RuntimeError, TypeError):
                pass
            self._surface_error_connected = False
            self._surface_layout.removeWidget(self._surface)
            detach_widget(self._surface)
        self._surface = widget
        self._surface_framed = False
        self._show_grip(QtCore.Qt.OpenHandCursor)
        self._refresh_title_band()
        if widget is None:
            # The picture left, so the card is the empty frame its preset
            # plans again -- not the last picture's frame, which differs
            # wherever a drawing widens its own margin.
            self._apply_card_size(self.panel_size)
            return
        widget.setParent(self)
        widget.installEventFilter(self)
        self._surface_error_connected = relay_surface_errors(
            widget, self.plot_error
        )
        resized = getattr(widget, "surfaceChanged", None)
        if hasattr(resized, "connect"):
            # A plot surface says when its picture became a DIFFERENT
            # surface -- first painted, redrawn at another preset, widened
            # by its own margin (the pulse timeline does, by eight pixels).
            # That is the one moment this card changes size: the card is the
            # picture's frame, not the preset's.
            resized.connect(self._surface_resized)
        self._surface_layout.addWidget(widget)
        widget.show()
        if getattr(widget, "presented_front", None) is not None:
            # A widget made over a host that already had a front shows it
            # from its first moment -- before this card could listen -- so
            # its announcement is already past.  Frame what it shows now.
            self._surface_resized(None)

    def _surface_resized(self, _identity: object) -> None:
        """The mounted picture is a different surface now: frame that one."""

        surface = self._surface
        if surface is not None and surface.width() and surface.height():
            self._surface_framed = True
            self._reserve(surface.width(), surface.height())

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        popup = self._settings_popup
        if (
            watched is popup
            and event.type() == QtCore.QEvent.Show
            and self._settings_pending_rebuild is not None
        ):
            self._rebuild_settings_form(opening=True)
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
        texts: dict[str, str] | None = None,
    ) -> None:
        """Replace the signal domain used by the Setting form.

        ``texts`` is what a person calls each signal (the presenter's
        ``<producer>/<output>``), which the closed pickers read.
        """

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
        texts = {str(key): str(text) for key, text in dict(texts or {}).items()}
        if (
            selected_groups == self._groups
            and selected_overlay_groups == self._overlay_groups
            and texts == self._signal_texts
            and current == str(self._state_projection.get("signal") or "")
            and overlay
            == str(self._state_projection.get("overlay_signal") or "")
        ):
            return
        self._groups = selected_groups
        self._overlay_groups = selected_overlay_groups
        self._signal_texts = texts
        if current:
            self._state_projection["signal"] = current
        self._state_projection["overlay_signal"] = overlay
        self._rebuild_settings_form(force=True)

    def set_status(self, text: str, *, error: bool) -> None:
        """Say on the card itself what a board-wide status line says about it.

        These widgets existed hidden and nothing ever showed them, so a panel
        named in a red strip line could not be told apart by looking at the
        board.  The dot rides the title strip beside the Setting button, so
        the card body still reserves no status row; hovering it reads the
        full message.  An empty text clears the mark.

        A Setting value this card cannot read is its own condition: it is
        shown over this one while it stands, and fixing it shows this one
        again -- the presenter sends a condition once, when it changes.
        """

        self._presented_status = (str(text), bool(error))
        self._show_status()

    def _show_status(self) -> None:
        local = self._local_setting_error[2]
        value, error = (local, True) if local else self._presented_status
        self._status_text = value
        self._status_error = error
        self.status_dot.set_state("error" if error else "idle")
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
        set_status_ink(label, "error" if self._status_error else "idle")
        label.setVisible(bool(self._status_text))
        self._sync_settings_body_size()

    def set_selectors_enabled(self, enabled: bool) -> None:
        """Give the plot all pointer gestures when On and none when Off."""

        self._selectors_on = bool(enabled)
        _set_interaction(self._surface, self._selectors_on)

    @property
    def selectors_enabled(self) -> bool:
        return self._selectors_on

    def retire_settings_popup(self) -> None:
        """Hide the page-owned Setting overlay without removing its Panel."""

        if self._settings_popup is not None:
            self._settings_popup.hide()

    def _form_spec(self) -> FormSpec:
        """The Panel section: what the panel draws and how it is framed."""

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
            FormFieldProps("title", "text", "Panel name", default=self._panel_name()),
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
        fields.append(size_form_field(self._size_choices, state["size"]))
        if self._interval_choices:
            fields.append(
                interval_form_field(
                    self._interval_choices,
                    state["interval_ms"],
                )
            )
        return FormSpec(tuple(fields))

    def _form_values(self) -> dict[str, object]:
        signal = str(self._state_projection.get("signal") or "") or None
        values: dict[str, object] = {
            "title": self._panel_name(),
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
        return values

    def _parameter_projection(self, section: str) -> tuple[FormSpec, dict[str, object]]:
        """One plot-owner section's form: its declared fields and values.

        A REASON IS NOT A SETTING, so no section declares one: a section's
        "why this could not be applied" is a message, which the console
        says in its header line and the card's standing mark.
        """

        declared = tuple(parameter_fields(self._parameter_surface, section))
        if not declared:
            return FormSpec(()), {}
        return parameter_form_spec(declared), parameter_form_values(declared)

    def _settings_forms(self) -> tuple[FluentParameterForm, ...]:
        """Every form of the Setting frame, top to bottom."""

        if self._settings_form is None:
            return ()
        return (self._settings_form, *self._settings_parameter_forms.values())

    def _rebuild_settings_form(self, *, force: bool = False, opening: bool = False) -> None:
        if not opening and (
            self._settings_popup is None or not self._settings_popup.isVisible()
        ):
            self._settings_pending_rebuild = bool(force or self._settings_pending_rebuild)
            return
        if self._settings_form is not None:
            projections = [(self._settings_form, "panel", self._form_spec(), self._form_values())]
            projections.extend(
                (form, section, *self._parameter_projection(section))
                for section, form in self._settings_parameter_forms.items()
            )
            # Keyed choices changed (``force``) is the Panel form's alone:
            # it is the one whose rows read the signal domain.
            keyed = bool(force or self._settings_pending_rebuild)
            for form, section, spec, values in projections:
                if (
                    (keyed and form is self._settings_form)
                    or not form.adopt_projection(spec, values)
                ):
                    form.reconcile(spec, values)
                # A section with nothing declared is not shown, and takes
                # no section gap.
                self._settings_sections[section].setVisible(bool(spec.fields))
                form.setVisible(bool(spec.fields))
            share_label_column(self._settings_forms())
            # The projection now stands in every row the operator is not
            # typing in -- written with signals blocked, so no ``changed``
            # clears a read error its value replaced.  Left standing, the
            # red text outlived the value and hid every later presenter
            # condition.  A row still being typed in keeps both.
            form, row, _text = self._local_setting_error
            if form is not None and not (
                row in form.spec.keys and being_edited(form.widget_for(row))
            ):
                self._local_setting_error = (None, "", "")
                self._show_status()
            self._apply_settings_enabled_state()
            self._sync_settings_body_size(opening=opening)
            self._settings_pending_rebuild = None

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
        # A Task-protected panel keeps its semantics, as on the Edit page.
        self._settings_parameter_forms["semantic"].setEnabled(not science_locked)

    def _sync_settings_body_size(self, *, opening: bool = False) -> None:
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

        if not opening and (
            self._settings_popup is None or not self._settings_popup.isVisible()
        ):
            return
        body = self._settings_body
        scroll = self._settings_scroll
        if body is None or body.layout() is None or scroll is None:
            return
        margins = body.layout().contentsMargins()
        required_width = (
            max(form.minimum_content_width() for form in self._settings_forms())
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

    def _show_settings_popup(self) -> None:
        """The card's ONE popup placement call, always with fresh geometry.

        Toggle-open and open-while-visible resize both come through here, so
        the popup can never be shown from a stale width measured before the
        form was rebuilt.
        """

        popup = self._settings_popup
        # Placed off the STRIP, so the frame stands clear of it -- under its
        # foot where the page has the room, over its top where it has more
        # -- and never over the strip's second line.  ONE height rule: the
        # form's, bounded by the frame's PAGE, never by the card, whose foot
        # is not an edge of anything the frame belongs to.  Placement
        # happens once, when the frame opens.
        show_fluent_popup_for_anchor(
            popup,
            self._title_band,
            self._settings_body,
            minimum_width=1,
            minimum_height=320,
            content_width=self._settings_content_width,
        )
        # Beside its button: the frame's right edge is Setting's.
        right = self.settings_button.mapTo(
            popup.parentWidget(), QtCore.QPoint(self.settings_button.width(), 0)
        ).x()
        popup.move(max(0, right - popup.width()), popup.y())

    def _open_settings(self) -> None:
        created = self._settings_popup is None
        if created:
            # A frame INSIDE the page, not a window.  It lives in the panel
            # area's viewport, so it is clipped to that area, hides and
            # returns with the tab, and travels/minimises/closes with the
            # window without any desktop machinery.  The page owns the
            # widget; the card's death takes it along explicitly.
            popup = FluentOverlayFrame(self._settings_overlay_host())
            self.destroyed.connect(popup.deleteLater)
            layout = QtWidgets.QVBoxLayout(popup)
            pad = card_pad()
            layout.setContentsMargins(pad, pad, 0, pad)
            # The title row sits in the same air above and below it.  With
            # no gap under it the row hugged the first field, and its close
            # button -- centred in the row -- read as sitting low in a blank
            # that was all above it.
            layout.setSpacing(pad)
            scroll = FluentScrollArea(popup)
            header = QtWidgets.QWidget(popup)
            header.setStyleSheet("background: transparent;")
            header_layout = QtWidgets.QHBoxLayout(header)
            # The close x ends where the fields end: the body stops a pad
            # short of the scroll viewport, which stops the bar's gutter
            # short of the frame's edge.
            gutter = scroll.verticalScrollBar().sizeHint().width()
            header_layout.setContentsMargins(0, 0, pad + gutter, 0)
            header_layout.setSpacing(row_gap())
            drag_handle = FluentLabel(
                f"Setting · {self.panel_id}",
                header,
            )
            drag_handle.setCursor(QtCore.Qt.SizeAllCursor)
            drag_handle.installEventFilter(self)
            header_layout.addWidget(drag_handle, 1)
            # Closing the frame takes nothing away and commits nothing: the
            # quiet one-glyph look of a tab's close x, never the red of the
            # card's own remover, square at a control's height.
            close_button = FluentButton("×", header, role="quiet")
            close_button.setFixedSize(control_height(), control_height())
            close_button.setToolTip("Close settings")
            close_button.clicked.connect(self.retire_settings_popup)
            header_layout.addWidget(close_button, 0)
            layout.addWidget(header)
            body = QtWidgets.QWidget()
            body.setStyleSheet("background: transparent;")
            body_layout = QtWidgets.QVBoxLayout(body)
            body_layout.setContentsMargins(0, 0, pad, 0)
            # The sections of the Edit page, each under its title, a section
            # gap apart; a hidden one takes no gap.
            body_layout.setSpacing(section_gap())
            status = FluentLabel("", body)
            status.setWordWrap(True)
            status.hide()
            body_layout.addWidget(status)
            self._settings_status = status
            self._settings_form = FluentParameterForm(
                self._form_spec(),
                self._form_values(),
                runtime=self._signal_runtime,
            )
            self._settings_form.changed.connect(self._setting_changed)
            self._settings_form.value_normalized.connect(self._setting_changed)
            blocks = [("panel", "Panel", self._settings_form)]
            for section, title in PARAMETER_SECTIONS:
                form = FluentParameterForm(*self._parameter_projection(section))
                form.changed.connect(
                    lambda key, name=section: self._parameter_changed(name, str(key))
                )
                form.value_normalized.connect(
                    lambda key, name=section: self._parameter_changed(name, str(key))
                )
                self._settings_parameter_forms[section] = form
                blocks.append((section, title, form))
            for section, title, form in blocks:
                # A block of the body's own layout: its form is the body's
                # child, so a row it gains or loses reaches the body's own
                # layout request, where the frame is measured.
                block, heading = panel_section(title, form)
                body_layout.addLayout(block)
                # A section with nothing declared is not shown, and a block
                # all of whose widgets are hidden takes no section gap.
                heading.setVisible(bool(form.spec.fields))
                form.setVisible(bool(form.spec.fields))
                self._settings_sections[section] = heading
            share_label_column(self._settings_forms())
            self._settings_pending_rebuild = None
            self._apply_settings_enabled_state()
            buttons = QtWidgets.QHBoxLayout()
            buttons.setContentsMargins(0, 0, 0, 0)
            buttons.setSpacing(row_gap())
            # Here rather than on the card face, which is where every other
            # per-panel decision already lives.  They existed as hidden widgets
            # nothing ever showed, so a panel opened in the real window could
            # not be removed at all and its Edit signal could not be raised.
            self.edit_button = FluentButton("Edit", body)
            self.edit_button.clicked.connect(self._request_edit)
            # Removing a plot panel takes two clicks from wherever it is
            # removed -- this Remove as well as the header ×.
            self.remove_button = FluentButton(
                "Remove", body, role="destructive", confirm_twice="remove this panel"
            )
            self.remove_button.confirmed.connect(self._request_remove)
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
            popup.installEventFilter(self)
            self._sync_settings_body_size(opening=True)
            popup.hide()
        anchor = self._settings_anchor
        anchor.toggle(
            self._settings_body,
            prepare=None if created else lambda: self._rebuild_settings_form(opening=True),
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

    def _setting_changed(self, key: str) -> None:
        """A Panel row edited: its value, as one patch."""

        form = self._settings_form
        if form is None:
            return
        name = str(key)
        try:
            value = form.read_value(name)
            if name == "interval_ms":
                patch: dict[str, object] = {name: int(value)}
            elif name in {"title", "signal", "overlay_signal", "size", "cell_kind"}:
                patch = {name: str(value or "")}
            else:
                return
        except (KeyError, TypeError, ValueError) as error:
            self._setting_unreadable(form, name, error)
            return
        self._setting_read(patch)

    def _parameter_changed(self, section: str, key: str) -> None:
        """A plot-owner row edited: it and what is edited with it, as one patch."""

        form = self._settings_parameter_forms[section]
        try:
            patch = parameter_edit_values(
                parameter_fields(self._parameter_surface, section),
                section,
                key,
                form.read_value,
            )
        except (KeyError, TypeError, ValueError) as error:
            self._setting_unreadable(form, key, error)
            return
        self._setting_read(patch)

    def _setting_unreadable(self, form: FluentParameterForm, row: str, error: Exception) -> None:
        self._local_setting_error = (form, row, str(error) or type(error).__name__)
        self._show_status()

    def _setting_read(self, patch: dict[str, object]) -> None:
        if self._local_setting_error[0] is not None:
            self._local_setting_error = (None, "", "")
            self._show_status()
        self.state_changed.emit(patch)

    def _choice_groups(self, field: str):
        if str(field) == "signal":
            return self._groups
        if str(field) == "overlay_signal":
            return self._overlay_groups
        return ()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        # What carries the card is handle_rect: a drawn card's strip, an
        # empty card's whole frame.  A press on a strip command is the
        # command's; a press in a drawn card's body is its plot's.
        if (
            event.button() == QtCore.Qt.LeftButton
            and self.handle_rect().contains(event.pos())
        ):
            self._press_at = event.pos()
            self._drag_offset = None
            self.raise_()
            self.grabMouse()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._press_at is not None and event.buttons() & QtCore.Qt.LeftButton:
            pointer = (int(event.globalPos().x()), int(event.globalPos().y()))
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
                # card actually starts following the pointer -- over what
                # it is held by, which stays under the pointer while it does.
                self._show_grip(QtCore.Qt.ClosedHandCursor)
                self.drag_started.emit(pointer)
            self.follow_pointer(event.globalPos())
            self.drag_moved.emit(pointer)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == QtCore.Qt.LeftButton and self._press_at is not None:
            dragged = self._drag_offset is not None
            self._press_at = None
            self._drag_offset = None
            self._show_grip(QtCore.Qt.OpenHandCursor)
            self.releaseMouse()
            # A press that reaches release without ever crossing the
            # threshold IS a click, and a click drops nothing.
            if dragged:
                self.dropped.emit(
                    (int(event.globalPos().x()), int(event.globalPos().y()))
                )
        super().mouseReleaseEvent(event)


def new_panel_card(owner: Any, panel_id: str, title: str) -> PanelCardView:
    """One board card, its choice policy projected and its intents relayed.

    The console handle and the FigureViewer each own a board of these cards;
    they are made and wired here, once, so the two boards cannot drift into
    two kinds of card.  ``owner`` holds the projected ``_panel_sizes`` /
    ``_panel_default_size``, ``_panel_intervals`` /
    ``_panel_default_interval`` and ``_grid_cell_kinds``, and the four panel
    signals the card's intents travel on.
    """

    key = str(panel_id)
    if not owner._panel_sizes:
        raise RuntimeError("panel size choices were not projected")
    card = PanelCardView(key, str(title))
    card.set_size_choices(owner._panel_sizes, owner._panel_default_size)
    if owner._panel_intervals:
        card.set_interval_choices(
            owner._panel_intervals, owner._panel_default_interval
        )
    if owner._grid_cell_kinds:
        card.set_cell_kind_choices(owner._grid_cell_kinds)
    card.remove_requested.connect(
        lambda _=None, pid=key: owner.panel_remove_requested.emit(pid)
    )
    card.edit_requested.connect(
        lambda _=None, pid=key: owner.panel_edit_requested.emit(pid)
    )
    card.state_changed.connect(
        lambda patch, pid=key: owner.panel_state_changed.emit(pid, patch)
    )
    card.plot_error.connect(
        lambda message, pid=key: owner.panel_plot_error.emit(pid, str(message))
    )
    return card


__all__ = ["PanelCardView", "new_panel_card"]
