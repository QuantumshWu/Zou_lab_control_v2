"""Free-drag board backed by one two-dimensional gravity rule."""

from __future__ import annotations

from typing import NamedTuple

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.board import (
    BoardMetrics,
    GeomProxy,
    drop,
    first_free_slot,
    min_board_width,
    pack,
    slot_under,
)
from zlc_ui.fluent import ACCENT_TINT, RADIUS, SELECTION_EDGE, card_pad, retire_widget, scaled_px
from zlc_ui.fluent.fluent import frame_pen, frame_stroke_rect

from .panel_card_view import PanelCardView


class _DragFrame(NamedTuple):
    """The board a carried card is let go into, settled once per width."""

    #: Every card where it stood when the carried card was picked up, by
    #: panel id: the board the slot is chosen in and every drop starts from.
    board: dict[str, GeomProxy]
    #: The width it was settled at.
    board_w: int


class ConsoleBoardView(QtWidgets.QWidget):
    """Cards carried by their grip and put down where they are seen.

    One rule set places every card: every card has a place, a card let go
    onto another changes places with it, a card let go on empty space goes
    there, cards that overlap make way, and everything falls north-west.

    A drop records where it left EVERY card, for the board's column count
    (how many of the narrowest card fit side by side), and nothing else
    writes those places.  A card no drop has placed yet -- one just added --
    takes the first free slot after them, top row left to right, in board
    order, at the width the board has NOW: that is how Add tiles the board,
    and why panels added before the window's first layout do not stay in the
    one column Qt's default width had room for.  A column count no drop was
    made at shows the nearest wider count's arrangement packed narrower (with
    none wider, the nearest narrower one), so narrowing packs without
    destroying the arrangement and widening restores it.

    What is on screen is those places settled under north-west gravity for a
    board this wide (``pack``).  A card rises until a card above it stops it
    and slides left until a card beside it stops it; a card stopped by a
    card above it stays below it, rather than sliding out from under it to a
    hole beside it, and no card ever passes another.

    A drag shows the drop before it happens.  The slot is chosen in the
    board as it stood when the card was picked up, which stands still while
    the card is carried, so the choice cannot chase its own consequence: the
    place under the pointer, on the line of the carried card's strip,
    nearest where the card is drawn (``slot_under``) -- the column is where
    the hand is and the row the one the card is seen in, whatever part of
    it is held.  Let go there (``drop``), the card swaps with the card it is
    let go onto -- that card takes the place the carried one left, the
    carried one takes the slot -- or, let go on empty space, takes the slot;
    where their new places overlap, the card further up and left keeps its
    place, the other cards make way for both (right if the row has room and
    the move crosses none of them, else down with the rest of their column),
    and then everything falls.  Put back where it was
    picked up, it leaves every card where it stood.  While it is carried the
    other cards already stand where the drop will leave them -- preview and
    drop are one computation -- and the slot is drawn under it as an
    accent-tint card with a dashed edge.  Near the viewport's top or bottom
    edge the board scrolls under the pointer, so a card can be carried to a
    row off screen.
    """

    order_committed = QtCore.pyqtSignal(tuple)

    #: How often the board scrolls under a carried card held at the
    #: viewport's edge, in ms.
    _SCROLL_TICK_MS = 20

    def __init__(self, parent=None, *, metrics: BoardMetrics | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setStyleSheet("background: transparent;")
        # One card pad between cards and from the board's edges.  The host
        # mounts the board that pad short of its page's content edge, so the
        # first card stands on that edge, under the first tab's underline.
        self._metrics = metrics or BoardMetrics(gap=card_pad())
        self._cards: dict[str, PanelCardView] = {}
        self._order: tuple[str, ...] = ()
        #: Where the last drop at each column count left every card, by panel
        #: id.  Written by a drop and by nothing else: packing a narrower or
        #: wider board reads it and never writes it back, which would ratchet
        #: a narrow packing into the arrangement itself.
        self._arrangements: dict[int, dict[str, tuple[int, int]]] = {}
        self._wired_cards: set[PanelCardView] = set()
        self._active_card: PanelCardView | None = None
        #: While a card is carried: the board it is put back into.
        self._frame: _DragFrame | None = None
        #: The slot the carried card is aimed at, and the rectangle it comes
        #: to rest in from there -- the one the board draws.
        self._drop_slot: tuple[int, int] | None = None
        self._drop_rect: QtCore.QRect | None = None
        #: Where the pointer is, in global coordinates: the board scrolls
        #: under a pointer that stands still, so a board-local point goes
        #: stale on every scroll step.
        self._pointer: QtCore.QPoint | None = None
        self._scroll_timer = QtCore.QTimer(self)
        self._scroll_timer.setInterval(self._SCROLL_TICK_MS)
        self._scroll_timer.timeout.connect(self._scroll_step)

    def text_left(self) -> int:
        """Where the words on this board start, from its left edge.

        A card's strip text sits one card pad in from a card that sits one
        gap in from the board, so a note on an empty board set at this x
        begins where the first card's title will.
        """

        return self._metrics.gap + PanelCardView.strip_text_inset()

    def set_cards(self, cards: tuple[PanelCardView, ...]) -> None:
        incoming = tuple(cards)
        for card in incoming:
            if not isinstance(card, PanelCardView):
                raise TypeError("board cards must be PanelCardView instances")

        self._end_drag()
        # A card is retired when its panel leaves the board AND when another
        # object arrives under the same panel_id: the id is the key, and one
        # key is one card.  Kept, the replaced object stayed a shown, wired
        # child of the board behind the card that took its place.
        arriving = {card.panel_id: card for card in incoming}
        leaving = tuple(
            card
            for panel_id, card in self._cards.items()
            if arriving.get(panel_id) is not card
        )
        # The board's set changes BEFORE the leavers are emptied: an emptied
        # card returns to its preset frame and says so, and a card still
        # counted here would repack the board around itself on its way out.
        self._cards = arriving
        for card in leaving:
            card.retire_settings_popup()
            # The host owns its adapter until asynchronous shutdown has
            # drained replies. Retiring its temporary parent must not
            # destroy that QObject underneath a completing gesture.
            card.set_surface(None)
            self._wired_cards.discard(card)
            retire_widget(card)
        kept = (
            (columns, {panel_id: place for panel_id, place in places.items() if panel_id in self._cards})
            for columns, places in self._arrangements.items()
        )
        self._arrangements = {columns: places for columns, places in kept if places}
        for card in incoming:
            card.setParent(self)
            card.show()
            if card not in self._wired_cards:
                card.drag_started.connect(
                    lambda point, current=card: self._card_drag_started(current, point)
                )
                card.drag_moved.connect(
                    lambda point, current=card: self._card_drag_moved(current, point)
                )
                card.dropped.connect(
                    lambda point, current=card: self._card_dropped(current, point)
                )
                card.geometry_changed.connect(
                    lambda current=card: self._card_size_changed(current)
                )
                self._wired_cards.add(card)
        self._order = tuple(card.panel_id for card in incoming)
        self._pack_current()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._active_card is None:
            self._pack_current()
        elif event.size().width() != event.oldSize().width():
            # The frame a carried card's slot is chosen in was settled at
            # the old width.
            self._frame_drag()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        # A viewport that widens or narrows over a board wider than it moves
        # where the board is cut without moving or resizing the board.
        area = self._scroll_area()
        if area is not None:
            area.viewport().installEventFilter(self)
        self._keep_commands_in_view()

    def moveEvent(self, event) -> None:  # noqa: N802 - Qt API
        # The viewport scrolls by moving the board under it.
        super().moveEvent(event)
        self._keep_commands_in_view()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QtCore.QEvent.Resize:
            self._keep_commands_in_view()
        return super().eventFilter(watched, event)

    def _keep_commands_in_view(self) -> None:
        """Tell every card where the viewport's right edge cuts it, so its
        Setting and × ride that edge (``PanelCardView.keep_commands_within``)
        on a card wider than what the viewport shows of it."""

        area = self._scroll_area()
        if area is None:
            return
        viewport = area.viewport()
        right = self.mapFrom(viewport, QtCore.QPoint(viewport.width(), 0)).x()
        for card in self._cards.values():
            card.keep_commands_within(right - card.x())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        """The slot a carried card will land in, under the cards.

        The light face every target in this design wears (ACCENT_TINT) with
        a dashed SELECTION_EDGE, rounded like the card it stands for: dashes
        say "not a card yet", and the preview has moved every other card
        out of the way, so nothing lies on top of it but the card itself.
        """

        del event
        rect = self._drop_rect
        if rect is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        radius = float(scaled_px(RADIUS))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(ACCENT_TINT))
        painter.drawRoundedRect(QtCore.QRectF(rect), radius, radius)
        pen = frame_pen(self, SELECTION_EDGE)
        pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(frame_stroke_rect(self, QtCore.QRectF(rect)), radius, radius)
        painter.end()

    def _proxy(self, card: PanelCardView) -> GeomProxy:
        """One card's rectangle, asked of the card itself.

        The card has already RESERVED it -- the picture its preset plans plus
        its own strip and margins -- so this is what it will occupy, not what
        happens to be inside it at this instant.  Its layout's own hint is the
        latter, and lags by however long the renderer takes to deliver a
        picture of the new size.

        The packer used to be handed a preset NAME and a formula restating the
        same arithmetic: two answers kept in step by hand, and one of them
        could not follow a card whose strip grew a line.
        """

        size = card.size()
        return GeomProxy(size.width(), size.height())

    def _board_width(self, proxies) -> int:
        """How wide the board must be to hold these cards.

        Takes the PROXIES, not the order: every caller has already built
        one per card, and rebuilding them here made a resize drag allocate
        2N proxies and make 2N card.size() round trips per frame instead
        of N.  min_board_width reads only .width, so any copies answer the
        same question.
        """

        proxies = tuple(proxies)
        if not proxies:
            return 0
        return max(self.width(), min_board_width(proxies, self._metrics))

    def _columns(self, proxies, board_w: int) -> int:
        """How many of the narrowest card fit side by side on a board this wide."""

        gap = self._metrics.gap
        return max(1, (board_w - gap) // (min(proxy.width for proxy in proxies) + gap))

    def _places(self, proxies, board_w: int) -> dict[str, tuple[int, int]]:
        """The arrangement a board this wide shows: the one the last drop at
        its column count left, else the nearest wider one's (a narrower
        window packs it), else the nearest narrower one's."""

        if not self._arrangements:
            return {}
        columns = self._columns(proxies, board_w)
        if columns in self._arrangements:
            return self._arrangements[columns]
        wider = [count for count in self._arrangements if count > columns]
        return self._arrangements[min(wider) if wider else max(self._arrangements)]

    def _settle(self, order: tuple[str, ...], board_w: int) -> dict[str, GeomProxy]:
        """Where each card of ``order`` comes to rest, without moving any.

        A card the arrangement places starts there and falls; every other
        card -- one no drop has placed yet -- then takes the first free slot,
        in board order, at the width the board has now.
        """

        proxies = {panel_id: self._proxy(self._cards[panel_id]) for panel_id in order}
        if not proxies:
            return proxies
        places = self._places(proxies.values(), board_w)
        placed = []
        for panel_id in order:
            if panel_id in places:
                proxies[panel_id].col, proxies[panel_id].row = places[panel_id]
                placed.append(proxies[panel_id])
        pack(tuple(placed), self._metrics, board_w)
        for panel_id in order:
            if panel_id not in places:
                proxy = proxies[panel_id]
                proxy.col, proxy.row = first_free_slot(proxy, placed, board_w, self._metrics)
                placed.append(proxy)
        return proxies

    def _place(
        self,
        proxies: dict[str, GeomProxy],
        *,
        carried: PanelCardView | None = None,
    ) -> int:
        """Move every card but ``carried`` to its settled rectangle; return
        the lowest bottom edge among them all."""

        bottom = 0
        # ``updatesEnabled()`` is an effective state: it is also false while
        # an ancestor (not this board) temporarily suppresses painting during
        # a tab/scroll-area relayout.  Calling ``setUpdatesEnabled(False)`` in
        # that state would turn the ancestor's transient state into a sticky
        # board-local disable, leaving every card present but permanently
        # unpainted after the ancestor resumes.  Only acquire update
        # suppression when this board currently owns an enabled state; then
        # this scope also owns the matching re-enable.
        suppress_updates = self.updatesEnabled()
        if suppress_updates:
            self.setUpdatesEnabled(False)
        try:
            for panel_id, proxy in proxies.items():
                rect = QtCore.QRect(
                    int(proxy.col), int(proxy.row), proxy.width, proxy.height
                )
                bottom = max(bottom, rect.bottom() + 1)
                card = self._cards[panel_id]
                if card is not carried:
                    card.setGeometry(rect)
        finally:
            if suppress_updates:
                self.setUpdatesEnabled(True)
                self.update()
        self._keep_commands_in_view()
        return bottom

    def _pack_current(self) -> dict[str, GeomProxy]:
        if not self._order:
            self.setMinimumSize(0, 0)
            return {}
        proxies = self._settle(
            self._order,
            self._board_width(self._proxy(card) for card in self._cards.values()),
        )
        bottom = self._place(proxies)
        # Do not use the current right extent as the minimum width: that would
        # ratchet a wide two-column board and make a later window resize unable
        # to reflow it into one column.  The pure packer's one-card bound is
        # the only horizontal minimum; the current packed bottom may safely
        # determine the vertical scroll extent.
        self.setMinimumSize(
            min_board_width(tuple(proxies.values()), self._metrics),
            bottom + self._metrics.gap,
        )
        return proxies

    def _card_size_changed(self, card: PanelCardView) -> None:
        if card in self._cards.values() and self._active_card is None:
            self._pack_current()

    def _card_drag_started(self, card: PanelCardView, pointer: tuple[int, int]) -> None:
        if self._cards.get(card.panel_id) is not card:
            return
        self._active_card = card
        self._pointer = QtCore.QPoint(*pointer)
        card.raise_()
        self._frame_drag()
        self._steer_scroll()

    def _card_drag_moved(self, card: PanelCardView, pointer: tuple[int, int]) -> None:
        if self._cards.get(card.panel_id) is not card:
            return
        if card is not self._active_card:
            # The board's cards were replaced under a drag in progress, which
            # ended it; the card is still held, so it starts again here.
            self._card_drag_started(card, pointer)
            return
        self._pointer = QtCore.QPoint(*pointer)
        self._show_drop()
        self._steer_scroll()

    def _frame_drag(self) -> None:
        """The board the carried card is let go into: every card where it
        stands, for as long as the card is carried at this width."""

        board_w = self._board_width(self._proxy(each) for each in self._cards.values())
        self._frame = _DragFrame(self._settle(self._order, board_w), board_w)
        self._drop_slot = None
        self._show_drop()

    def _dropped_at(self, slot: tuple[int, int]) -> dict[str, GeomProxy]:
        """Every card at rest once the carried card is let go at ``slot``
        (``drop``), on copies of the frame."""

        frame = self._frame
        proxies = {
            panel_id: GeomProxy(proxy.width, proxy.height, proxy.col, proxy.row)
            for panel_id, proxy in frame.board.items()
        }
        drop(
            tuple(proxies[panel_id] for panel_id in self._order),
            proxies[self._active_card.panel_id],
            slot,
            self._metrics,
            frame.board_w,
        )
        return proxies

    def _show_drop(self) -> None:
        """Lay the board out as letting go here would, and draw the slot.

        Aimed at the pointer's x on the line of the carried card's strip --
        the column is where the hand is, the row the one the card is seen in
        -- and of the places that keep that point inside the card, the one
        nearest where the card is drawn (``slot_under``).  The card stands
        where the last move put it: follow_pointer runs before every
        drag_moved and every scroll step.
        """

        card = self._active_card
        frame = self._frame
        others = tuple(proxy for panel_id, proxy in frame.board.items() if panel_id != card.panel_id)
        at = self.mapFromGlobal(self._pointer)
        seen = card.pos()
        slot = slot_under(
            (at.x(), card.y() + card.strip_rect().center().y()),
            frame.board[card.panel_id],
            others,
            self._metrics,
            frame.board_w,
            seen=(seen.x(), seen.y()),
        )
        if slot == self._drop_slot:
            return
        self._drop_slot = slot
        preview = self._dropped_at(slot)
        bottom = self._place(preview, carried=card)
        target = preview[card.panel_id]
        if self._drop_rect is not None:
            self.update(self._drop_rect)
        self._drop_rect = QtCore.QRect(
            int(target.col), int(target.row), target.width, target.height
        )
        self.update(self._drop_rect)
        # Room for one more row while a card is carried: the slot after the
        # last card is somewhere the pointer must be able to reach, and to
        # scroll to on a board taller than its viewport.
        lowest = max((proxy.row + proxy.height for proxy in others), default=0)
        gap = self._metrics.gap
        self.setMinimumSize(
            min_board_width(tuple(preview.values()), self._metrics),
            max(bottom, lowest + gap + card.height()) + gap,
        )

    def _scroll_area(self) -> QtWidgets.QAbstractScrollArea | None:
        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, QtWidgets.QAbstractScrollArea):
            widget = widget.parentWidget()
        return widget

    def _scroll_speed(self) -> int:
        """How far the viewport moves per tick for the pointer where it is.

        Zero unless the pointer is within one title strip of the viewport's
        top or bottom edge -- a band the height of what a drawn card is held
        by, so the board moves once the hand reaches the edge -- or past it.
        Faster the deeper in, up to a quarter strip per tick at the edge and
        beyond: about a card's height a second, slow enough to stop on the
        row wanted.
        """

        card = self._active_card
        area = self._scroll_area()
        if card is None or area is None or self._pointer is None:
            return 0
        viewport = area.viewport()
        band = max(1, card.strip_rect().height())
        y = viewport.mapFromGlobal(self._pointer).y()
        if y < band:
            return -max(1, min(band - y, band) // 4)
        if y > viewport.height() - band:
            return max(1, min(y - (viewport.height() - band), band) // 4)
        return 0

    def _steer_scroll(self) -> None:
        if self._scroll_speed():
            if not self._scroll_timer.isActive():
                self._scroll_timer.start()
        else:
            self._scroll_timer.stop()

    def _scroll_step(self) -> None:
        card = self._active_card
        speed = self._scroll_speed()
        area = self._scroll_area()
        if card is None or not speed or area is None:
            self._scroll_timer.stop()
            return
        bar = area.verticalScrollBar()
        before = bar.value()
        bar.setValue(before + speed)
        if bar.value() == before:
            self._scroll_timer.stop()
            return
        # The board moved under a pointer that did not: the card keeps its
        # hold under the pointer, and the slot it is aimed at is a different
        # one.
        card.follow_pointer(self._pointer)
        self._show_drop()

    def _card_dropped(self, card: PanelCardView, pointer: tuple[int, int]) -> None:
        if self._cards.get(card.panel_id) is not card:
            # Retired mid-drag: the grabbing card still gets its release
            # before its deferred delete runs, and there is no place left
            # to drop it into.
            if card is self._active_card:
                self._end_drag()
            return
        # A drop always follows a drag that crossed the window system's
        # threshold; this covers a drag whose first move landed before this
        # view saw it, or whose cards were replaced under it.
        if card is not self._active_card:
            self._card_drag_started(card, pointer)
        self._pointer = QtCore.QPoint(*pointer)
        self._show_drop()
        preview = self._dropped_at(self._drop_slot)
        board_w = self._frame.board_w
        self._end_drag()
        # The drop records where it left EVERY card, at this column count.
        self._arrangements[self._columns(preview.values(), board_w)] = {
            panel_id: (int(proxy.col), int(proxy.row)) for panel_id, proxy in preview.items()
        }
        packed = self._pack_current()
        self._order = tuple(
            sorted(self._order, key=lambda panel_id: (packed[panel_id].row, packed[panel_id].col))
        )
        self.order_committed.emit(self._order)

    def _end_drag(self) -> None:
        self._active_card = None
        self._frame = None
        self._drop_slot = None
        self._pointer = None
        self._scroll_timer.stop()
        if self._drop_rect is not None:
            self.update(self._drop_rect)
            self._drop_rect = None


__all__ = ["ConsoleBoardView"]
