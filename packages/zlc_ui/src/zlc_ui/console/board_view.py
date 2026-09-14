"""Free-drag board backed by one two-dimensional gravity rule."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.board import (
    BoardMetrics,
    GeomProxy,
    first_free_slot,
    nearest_anchor,
    min_board_width,
    pack,
)
from zlc_ui.fluent import retire_widget

from .panel_card_view import PanelCardView


class ConsoleBoardView(QtWidgets.QWidget):
    """Let cards move freely, then snap their position on mouse release.

    This is deliberately a two-phase interaction.  During a drag the card is
    an ordinary child widget at the operator's raw pointer position; the other
    cards do not move and there is no placeholder/ghost.  At release its
    top-left chooses the nearest grid anchor from the settled board, and that
    anchor becomes the place that card was PUT.

    Every card has one: authored by a drop, or seeded where it first appeared.
    The board keeps them and never writes over them -- what is on screen is
    those places settled under north-west gravity, for a board this wide.  So
    a narrow window packs into one column without destroying the arrangement,
    widening restores it, and a drop moves the card that was dropped rather
    than releasing everything else to reflow.
    """

    order_committed = QtCore.pyqtSignal(tuple)

    def __init__(self, parent=None, *, metrics: BoardMetrics | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setStyleSheet("background: transparent;")
        self._metrics = metrics or BoardMetrics(gap=8)
        self._cards: dict[str, PanelCardView] = {}
        self._order: tuple[str, ...] = ()
        #: Where each card was PUT, by panel id.  Authored by a drop, seeded
        #: on arrival, and never written by the packer: the packed position
        #: is what this place comes to rest at on a board of the current
        #: width, and writing that back would ratchet a narrow packing into
        #: the arrangement itself.
        self._authored: dict[str, tuple[int, int]] = {}
        #: The card most recently released, which wins ties in the settling
        #: order: when two cards want one place, the operator's is the one
        #: that just asked for it.
        self._dropped_id: str | None = None
        self._wired_cards: set[PanelCardView] = set()
        self._active_card: PanelCardView | None = None

    def set_cards(self, cards: tuple[PanelCardView, ...]) -> None:
        incoming = tuple(cards)
        for card in incoming:
            if not isinstance(card, PanelCardView):
                raise TypeError("board cards must be PanelCardView instances")

        self._cancel_drag()
        # A card is retired when its panel leaves the board AND when another
        # object arrives under the same panel_id: the id is the key, and one
        # key is one card.  Kept, the replaced object stayed a shown, wired
        # child of the board behind the card that took its place.
        arriving = {card.panel_id: card for card in incoming}
        for panel_id, card in tuple(self._cards.items()):
            if arriving.get(panel_id) is not card:
                card.retire_settings_popup()
                # The host owns its adapter until asynchronous shutdown has
                # drained replies. Retiring its temporary parent must not
                # destroy that QObject underneath a completing gesture.
                card.set_surface(None)
                self._wired_cards.discard(card)
                retire_widget(card)
        self._cards = arriving
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

    def _proxy(self, card: PanelCardView, *, placed: bool = False) -> GeomProxy:
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
        if placed:
            return GeomProxy(size.width(), size.height(), int(card.x()), int(card.y()))
        return GeomProxy(size.width(), size.height())

    def _board_width(self, proxies) -> int:
        """How wide the board must be to hold these cards.

        Takes the PROXIES, not the order: every caller has already built
        one per card, and rebuilding them here made a resize drag allocate
        2N proxies and make 2N card.size() round trips per frame instead
        of N.  min_board_width reads only .width, so the caller's copies --
        placed or not, pinned or not -- answer the same question.
        """

        proxies = tuple(proxies)
        if not proxies:
            return 0
        return max(self.width(), min_board_width(proxies, self._metrics))

    def _pack_current(self) -> dict[str, GeomProxy]:
        if not self._order:
            self.setMinimumSize(0, 0)
            return {}
        return self._apply_packed(self._order)

    def _apply_packed(self, order: tuple[str, ...]) -> dict[str, GeomProxy]:
        proxies = {
            panel_id: self._proxy(self._cards[panel_id]) for panel_id in order
        }
        board_w = self._board_width(proxies.values())
        # Every card falls from the place it was PUT.  A card that has never
        # been put anywhere takes the top-most then left-most free slot --
        # which is how Add tiles the top row and wraps -- and that becomes
        # its place, once this board has a real width to have tiled on: a
        # seed taken before the first layout would be a single column, and
        # gravity would then faithfully keep it one.
        seeded: list = []
        for panel_id in order:
            proxy = proxies[panel_id]
            place = self._authored.get(panel_id)
            if place is None:
                place = first_free_slot(proxy, seeded, board_w, self._metrics)
                if self.width() > 0:
                    self._authored[panel_id] = place
            proxy.col, proxy.row = place
            seeded.append(proxy)
        pack(
            tuple(proxies[panel_id] for panel_id in order),
            self._metrics,
            board_w,
            dropped=proxies.get(self._dropped_id),
        )
        by_id: dict[str, GeomProxy] = {}
        right = bottom = 0
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
            for panel_id in order:
                proxy = proxies[panel_id]
                by_id[panel_id] = proxy
                rect = QtCore.QRect(
                    int(proxy.col), int(proxy.row), proxy.width, proxy.height
                )
                right = max(right, rect.right() + 1)
                bottom = max(bottom, rect.bottom() + 1)
                card = self._cards[panel_id]
                card.setGeometry(rect)
        finally:
            if suppress_updates:
                self.setUpdatesEnabled(True)
                self.update()
        # Do not use the current right extent as the minimum width: that would
        # ratchet a wide two-column board and make a later window resize unable
        # to reflow it into one column.  The pure packer's one-card bound is
        # the only horizontal minimum; the current packed bottom may safely
        # determine the vertical scroll extent.
        self.setMinimumSize(
            min_board_width(tuple(proxies.values()), self._metrics),
            bottom + self._metrics.gap,
        )
        return by_id

    def _card_size_changed(self, card: PanelCardView) -> None:
        if card in self._cards.values() and self._active_card is None:
            self._pack_current()

    def _card_drag_started(self, card: PanelCardView, local_point: tuple[int, int]) -> None:
        if card.panel_id not in self._cards:
            return
        self._active_card = card
        card.raise_()
        # The card itself has already moved to the raw pointer position.  Do
        # not reflow the remaining cards or paint a dashed insertion ghost.

    def _card_dropped(self, card: PanelCardView, local_point: tuple[int, int]) -> None:
        # A drop now always follows a drag that crossed the window
        # system's threshold, so this no longer covers for a click
        # arriving as a drop -- it covers a drag whose first move landed
        # before this view saw it.
        if card is not self._active_card:
            self._active_card = card
        others = tuple(panel_id for panel_id in self._order if panel_id != card.panel_id)
        other_proxies = [
            self._proxy(self._cards[panel_id], placed=True) for panel_id in others
        ]
        dropped = self._proxy(card, placed=True)
        anchor = nearest_anchor(
            dropped,
            other_proxies,
            self._metrics,
            self._board_width((dropped, *other_proxies)),
        )
        self._authored[card.panel_id] = anchor
        self._dropped_id = card.panel_id
        self._active_card = None
        packed = self._pack_current()
        pin = packed[card.panel_id]
        index = sum(
            (packed[panel_id].row, packed[panel_id].col) < (pin.row, pin.col)
            for panel_id in others
        )
        self._order = others[:index] + (card.panel_id,) + others[index:]
        self.order_committed.emit(self._order)

    def _cancel_drag(self) -> None:
        self._active_card = None


__all__ = ["ConsoleBoardView"]
