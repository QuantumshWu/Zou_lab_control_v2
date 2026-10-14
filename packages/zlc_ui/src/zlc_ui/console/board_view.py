"""Free-drag board backed by one two-dimensional gravity rule."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.board import (
    BoardMetrics,
    GeomProxy,
    first_free_slot,
    min_board_width,
    pack,
    slot_under,
)
from zlc_ui.fluent import ACCENT_TINT, RADIUS, SELECTION_EDGE, card_pad, retire_widget, scaled_px
from zlc_ui.fluent.fluent import frame_pen, frame_stroke_rect

from .panel_card_view import PanelCardView


class ConsoleBoardView(QtWidgets.QWidget):
    """Cards carried by their strip and put down in the slot under the pointer.

    A card the operator has put somewhere keeps that place: a drop records
    the slot and the board never writes over it.  Every other card FLOWS: on
    every pack it takes the first free slot -- top row left to right, then
    the next -- in board order, after the places drops hold.  That is how Add
    tiles the board, and it is recomputed at the width the board has NOW: a
    place recorded for a card when it arrived would be taken at whatever
    width the board had then -- before the window's first layout, Qt's
    default, one card wide -- and gravity, which only moves cards up and
    left, would keep that single column on a board two cards wide for good.

    What is on screen is those places settled under north-west gravity for a
    board this wide, so a narrow window packs into one column without
    destroying the arrangement and widening restores it, and a card put below
    another stays below it rather than flying to a hole beside it.

    A drag shows the drop before it happens.  The other cards stand where
    they will stand once the carried card is let go, and the slot it will land
    in is drawn under it as an accent-tint card with a dashed edge.  The slot
    is the one under the POINTER, chosen on the board as it is without the
    carried card: that frame does not move while the preview rearranges the
    others, so the choice cannot chase its own consequence, and it is the
    board the card is being put back into.  On the board as it stands, with
    the carried card's own hole in it, the last free slot would sit one place
    past the end, and a card put there would leave a hole behind the cards
    that close up ahead of it.  Near the viewport's top or bottom edge the
    board scrolls under the pointer, so a card can be carried to a row off
    screen.
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
        #: Where the operator PUT a card, by panel id: written by a drop and
        #: by nothing else.  The packed position is what this place comes to
        #: rest at on a board of the current width, and writing that back
        #: would ratchet a narrow packing into the arrangement itself.
        self._authored: dict[str, tuple[int, int]] = {}
        #: The card most recently released, which wins ties in the settling
        #: order: when two cards want one place, the operator's is the one
        #: that just asked for it.
        self._dropped_id: str | None = None
        self._wired_cards: set[PanelCardView] = set()
        self._active_card: PanelCardView | None = None
        #: While a card is carried: the places drops hold for the OTHER
        #: cards, the board settled without the carried card, and the width
        #: both were settled at.  The slot is chosen in this frame.
        self._frame: tuple[dict[str, tuple[int, int]], tuple[GeomProxy, ...], int] | None = None
        #: The slot under the pointer, and the rectangle the carried card
        #: comes to rest in from it -- the one the board draws.
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
        self._authored = {
            panel_id: place
            for panel_id, place in self._authored.items()
            if panel_id in self._cards
        }
        if self._dropped_id not in self._cards:
            self._dropped_id = None
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

    def _settle(
        self,
        order: tuple[str, ...],
        authored: dict[str, tuple[int, int]],
        dropped_id: str | None,
        board_w: int,
    ) -> dict[str, GeomProxy]:
        """Where each card of ``order`` comes to rest, without moving any.

        Every card falls from where it starts.  A card a drop put somewhere
        starts there; every other card starts in the first free slot, in
        board order, once those places are taken -- so the cards nobody
        placed tile the board at the width it has now.
        """

        proxies = {panel_id: self._proxy(self._cards[panel_id]) for panel_id in order}
        seeded: list[GeomProxy] = []
        for panel_id in order:
            place = authored.get(panel_id)
            if place is not None:
                proxies[panel_id].col, proxies[panel_id].row = place
                seeded.append(proxies[panel_id])
        for panel_id in order:
            if panel_id in authored:
                continue
            proxy = proxies[panel_id]
            proxy.col, proxy.row = first_free_slot(proxy, seeded, board_w, self._metrics)
            seeded.append(proxy)
        pack(
            tuple(proxies[panel_id] for panel_id in order),
            self._metrics,
            board_w,
            dropped=proxies.get(dropped_id),
        )
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
        return bottom

    def _pack_current(self) -> dict[str, GeomProxy]:
        if not self._order:
            self.setMinimumSize(0, 0)
            return {}
        proxies = self._settle(
            self._order,
            self._authored,
            self._dropped_id,
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
        """Settle the board without the carried card: the frame its slot is
        chosen in, for as long as it is carried at this width."""

        card = self._active_card
        authored = {
            panel_id: place
            for panel_id, place in self._authored.items()
            if panel_id != card.panel_id
        }
        others = tuple(panel_id for panel_id in self._order if panel_id != card.panel_id)
        board_w = self._board_width(self._proxy(each) for each in self._cards.values())
        base = self._settle(
            others,
            authored,
            None if self._dropped_id == card.panel_id else self._dropped_id,
            board_w,
        )
        self._frame = (authored, tuple(base.values()), board_w)
        self._drop_slot = None
        self._show_drop()

    def _show_drop(self) -> None:
        """Lay the board out as letting go here would, and draw the slot."""

        card = self._active_card
        authored, base, board_w = self._frame
        at = self.mapFromGlobal(self._pointer)
        slot = slot_under((at.x(), at.y()), self._proxy(card), base, self._metrics, board_w)
        if slot == self._drop_slot:
            return
        self._drop_slot = slot
        preview = self._settle(
            self._order, {**authored, card.panel_id: slot}, card.panel_id, board_w
        )
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
        lowest = max((proxy.row + proxy.height for proxy in base), default=0)
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

        Zero unless the pointer is within one strip of the viewport's top or
        bottom edge -- the height of what the card is held by, so the board
        moves once the hand reaches the edge -- or past it.  Faster the
        deeper in, up to a quarter strip per tick at the edge and beyond:
        about a card's height a second, slow enough to stop on the row
        wanted.
        """

        card = self._active_card
        area = self._scroll_area()
        if card is None or area is None or self._pointer is None:
            return 0
        viewport = area.viewport()
        band = max(1, card.handle_rect().height())
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
        # hold under the pointer, and the slot under it is a different one.
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
        slot = self._drop_slot
        self._end_drag()
        self._authored[card.panel_id] = slot
        self._dropped_id = card.panel_id
        others = tuple(panel_id for panel_id in self._order if panel_id != card.panel_id)
        packed = self._pack_current()
        pin = packed[card.panel_id]
        index = sum(
            (packed[panel_id].row, packed[panel_id].col) < (pin.row, pin.col)
            for panel_id in others
        )
        self._order = others[:index] + (card.panel_id,) + others[index:]
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
