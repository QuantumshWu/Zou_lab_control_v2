"""v1-style free-drag board backed by the pure north-west packer."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.board import (
    BoardMetrics,
    GeomProxy,
    drop_index,
    min_board_width,
    pack,
)
from zlc_ui.fluent import CARD_PAD, CARD_TITLE_PX, scaled_px

from .panel_card_view import PanelCardView


class ConsoleBoardView(QtWidgets.QWidget):
    """Let cards move freely, then snap their order on mouse release.

    This is deliberately a two-phase interaction.  During a drag the card is
    an ordinary child widget at the operator's raw pointer position; the other
    cards do not move and there is no placeholder/ghost.  At release the pure
    packer chooses the semantic insertion index and restores north-west gravity
    for the complete board, matching the v1 ``_snap_dropped_card`` path.
    """

    order_committed = QtCore.pyqtSignal(tuple)

    def __init__(self, parent=None, *, metrics: BoardMetrics | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setStyleSheet("background: transparent;")
        self._metrics = metrics or BoardMetrics(
            gap=8,
            card_size=self._default_card_size,
        )
        self._cards: dict[str, PanelCardView] = {}
        self._order: tuple[str, ...] = ()
        self._wired_cards: set[PanelCardView] = set()
        self._active_card: PanelCardView | None = None

    @staticmethod
    def _default_card_size(size: str) -> tuple[int, int]:
        """How big a card of this preset is -- asked of the card.

        The formula was written here as well, so the packer and the card could
        disagree about the size of the same card: cards laid out to one
        geometry and drawn to another, overlapping or leaving gaps.
        """

        value = str(size)
        if not value:
            raise ValueError("panel size was not projected")
        return PanelCardView.card_size(value)

    def set_cards(self, cards: tuple[PanelCardView, ...]) -> None:
        incoming = tuple(cards)
        for card in incoming:
            if not isinstance(card, PanelCardView):
                raise TypeError("board cards must be PanelCardView instances")

        self._cancel_drag()
        wanted_ids = {card.panel_id for card in incoming}
        for panel_id, card in tuple(self._cards.items()):
            if panel_id not in wanted_ids:
                card.setParent(None)
                card.hide()
        self._cards = {card.panel_id: card for card in incoming}
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

    def grab_board(self) -> QtGui.QPixmap:
        return self.grab()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._active_card is None:
            self._pack_current()

    def _size_key(self, card: PanelCardView) -> str:
        value = str(card.panel_size)
        if not value:
            raise ValueError("panel size was not projected")
        return value

    def _board_width(self, order: tuple[str, ...]) -> int:
        proxies = [GeomProxy(self._size_key(self._cards[panel_id])) for panel_id in order]
        if not proxies:
            return 0
        return max(self.width(), min_board_width(proxies, self._metrics))

    def _pack_current(self) -> None:
        if not self._order:
            self.setMinimumSize(0, 0)
            return
        self._apply_packed(self._order)

    def _apply_packed(self, order: tuple[str, ...]) -> dict[str, GeomProxy]:
        proxies = [GeomProxy(self._size_key(self._cards[panel_id])) for panel_id in order]
        pack(proxies, self._metrics, self._board_width(order))
        by_id: dict[str, GeomProxy] = {}
        right = bottom = 0
        updates_enabled = self.updatesEnabled()
        self.setUpdatesEnabled(False)
        try:
            for panel_id, proxy in zip(order, proxies):
                by_id[panel_id] = proxy
                width, height = self._metrics.card_size(proxy.size)
                rect = QtCore.QRect(int(proxy.col), int(proxy.row), int(width), int(height))
                right = max(right, rect.right() + 1)
                bottom = max(bottom, rect.bottom() + 1)
                card = self._cards[panel_id]
                card.setGeometry(rect)
        finally:
            self.setUpdatesEnabled(updates_enabled)
            if updates_enabled:
                self.update()
        # Do not use the current right extent as the minimum width: that would
        # ratchet a wide two-column board and make a later window resize unable
        # to reflow it into one column.  The pure packer's one-card bound is
        # the only horizontal minimum; the current packed bottom may safely
        # determine the vertical scroll extent.
        self.setMinimumSize(min_board_width(proxies, self._metrics), bottom + self._metrics.gap)
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

    def _card_drag_moved(self, card: PanelCardView, local_point: tuple[int, int]) -> None:
        # Kept as a signal seam for presenters; v1 does no live board layout
        # work while the pointer is down.
        return None

    def _card_dropped(self, card: PanelCardView, local_point: tuple[int, int]) -> None:
        if card is not self._active_card:
            self._active_card = card
        others = tuple(panel_id for panel_id in self._order if panel_id != card.panel_id)
        other_proxies = [GeomProxy(self._size_key(self._cards[panel_id])) for panel_id in others]
        index = drop_index(
            GeomProxy(self._size_key(card), int(card.x()), int(card.y())),
            other_proxies,
            self._metrics,
            self._board_width(self._order),
        )
        committed = others[:index] + (card.panel_id,) + others[index:]
        self._active_card = None
        self._order = committed
        self._pack_current()
        self.order_committed.emit(self._order)

    def _cancel_drag(self) -> None:
        self._active_card = None


__all__ = ["ConsoleBoardView"]
