"""Presentation-owned, capacity-one live transport.

The channel carries immutable role-axis snapshots but does not define their
schema.  It is deliberately separate from :mod:`zlc_data`, whose contract is
data ownership rather than UI cadence or latest-only scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from threading import Condition, RLock
from typing import Generic, TypeVar

from .errors import RevisionError


@dataclass(frozen=True, slots=True)
class IngressMetrics:
    published: int
    consumed: int
    coalesced_updates: int
    pending: bool
    last_published_revision: int | None
    last_consumed_revision: int | None


ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class RevisionedItem(Generic[ItemT]):
    revision: int
    payload: ItemT


class LatestRevisionChannel(Generic[ItemT]):
    """Thread-safe capacity-one handoff retaining only the newest item."""

    def __init__(self, *, initial_revision: int | None = None) -> None:
        if initial_revision is not None:
            self._validate_revision_value(initial_revision)
            initial_revision = int(initial_revision)
        self._condition = Condition(RLock())
        self._revision_floor = initial_revision
        self._pending: RevisionedItem[ItemT] | None = None
        self._last_revision: int | None = None
        self._last_consumed_revision: int | None = None
        self._published = 0
        self._consumed = 0
        self._coalesced = 0
        self._closed = False

    @staticmethod
    def _validate_revision_value(revision: object) -> None:
        if isinstance(revision, bool) or not isinstance(revision, Integral):
            raise RevisionError("revision must be an integer")
        if int(revision) < 0:
            raise RevisionError("revision must be non-negative")

    def _validate_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("latest revision channel is closed")

    def _validate_revision_locked(self, revision: object) -> int:
        self._validate_revision_value(revision)
        selected = int(revision)
        boundary = self._last_revision if self._last_revision is not None else self._revision_floor
        if boundary is not None and selected <= boundary:
            raise RevisionError(f"revision {selected} is not newer than {boundary}")
        return selected

    def _accept_item_locked(self, revision: object, payload: ItemT) -> RevisionedItem[ItemT]:
        self._validate_open_locked()
        selected = self._validate_revision_locked(revision)
        if self._pending is not None:
            self._coalesced += 1
        item = RevisionedItem(selected, payload)
        self._pending = item
        self._last_revision = selected
        self._published += 1
        self._condition.notify_all()
        return item

    def publish(self, revision: int, payload: ItemT) -> RevisionedItem[ItemT]:
        with self._condition:
            return self._accept_item_locked(revision, payload)

    def take_latest(self) -> RevisionedItem[ItemT] | None:
        with self._condition:
            return self._take_latest_locked()

    def _take_latest_locked(self) -> RevisionedItem[ItemT] | None:
        item = self._pending
        if item is None:
            return None
        self._pending = None
        self._consumed += 1
        self._last_consumed_revision = item.revision
        return item

    def wait_latest(self, timeout: float | None = None) -> RevisionedItem[ItemT] | None:
        with self._condition:
            if self._pending is None and not self._closed:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._closed,
                    timeout=timeout,
                )
            return self._take_latest_locked()

    def metrics(self) -> IngressMetrics:
        with self._condition:
            return IngressMetrics(
                published=self._published,
                consumed=self._consumed,
                coalesced_updates=self._coalesced,
                pending=self._pending is not None,
                last_published_revision=self._last_revision,
                last_consumed_revision=self._last_consumed_revision,
            )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()


__all__ = ["IngressMetrics", "LatestRevisionChannel", "RevisionedItem"]
