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

import numpy as np
from numpy.typing import ArrayLike
from zlc_data import DatasetSchema, OwnedSnapshot, owned_snapshot_from_arrays

from .data_contract import (
    schema_dtype,
    schema_equal,
    schema_shape,
    snapshot_revision,
    snapshot_schema,
    snapshot_validity,
    snapshot_values,
)
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


class LatestIngress:
    """Capacity-one ``OwnedSnapshot`` ingress with patch/rolling helpers."""

    def __init__(
        self,
        schema: DatasetSchema,
        *,
        initial: OwnedSnapshot | None = None,
    ) -> None:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be a zlc_data.DatasetSchema")
        if initial is not None and not isinstance(initial, OwnedSnapshot):
            raise TypeError("initial must be a zlc_data.OwnedSnapshot")
        self._schema = schema
        self._channel: LatestRevisionChannel[OwnedSnapshot] = LatestRevisionChannel()
        self._lock = RLock()
        self._state: OwnedSnapshot | None = None
        self._closed = False
        if initial is not None:
            self.publish(initial)

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    def _validate_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("latest ingress is closed")

    def _accept_locked(self, snapshot: OwnedSnapshot) -> OwnedSnapshot:
        if not schema_equal(self._schema, snapshot_schema(snapshot)):
            raise ValueError("live snapshot schema differs from ingress schema")
        self._channel.publish(snapshot_revision(snapshot), snapshot)
        self._state = snapshot
        return snapshot

    def publish(self, snapshot: OwnedSnapshot) -> OwnedSnapshot:
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be a zlc_data.OwnedSnapshot")
        with self._lock:
            self._validate_open_locked()
            return self._accept_locked(snapshot)

    def replace(
        self,
        values: ArrayLike,
        *,
        revision: int,
        validity: ArrayLike | None = None,
        metadata: object | None = None,
    ) -> OwnedSnapshot:
        del metadata  # metadata is not part of the zlc-data snapshot contract.
        with self._lock:
            self._validate_open_locked()
            return self._accept_locked(
                owned_snapshot_from_arrays(
                    schema=self._schema,
                    values=values,
                    revision=revision,
                    validity=validity,
                )
            )

    def patch(
        self,
        index: object,
        values: ArrayLike,
        *,
        revision: int,
        valid: ArrayLike | bool | None = None,
        metadata: object | None = None,
    ) -> OwnedSnapshot:
        del metadata
        with self._lock:
            self._validate_open_locked()
            state = self._state
            if state is None:
                raise RevisionError("patch requires an initial replace/publish")
            raw_values = np.asarray(values)
            if raw_values.dtype != schema_dtype(self._schema):
                raise ValueError(
                    f"patch dtype {raw_values.dtype} does not match schema dtype "
                    f"{schema_dtype(self._schema)}"
                )
            raw_valid = None if valid is None else np.asarray(valid)
            if raw_valid is not None and raw_valid.dtype != np.dtype(np.bool_):
                raise ValueError("patch validity must have boolean dtype")
            next_values = np.array(snapshot_values(state), copy=True)
            next_validity = np.array(snapshot_validity(state), copy=True)
            try:
                next_values[index] = raw_values
                if raw_valid is not None:
                    next_validity[index] = raw_valid
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError(f"invalid patch: {error}") from error
            return self._accept_locked(
                owned_snapshot_from_arrays(
                    schema=self._schema,
                    values=next_values,
                    revision=revision,
                    validity=next_validity,
                )
            )

    def rolling(
        self,
        samples: ArrayLike,
        *,
        revision: int,
        axis: int = 1,
        validity: ArrayLike | None = None,
        metadata: object | None = None,
    ) -> OwnedSnapshot:
        del metadata
        with self._lock:
            self._validate_open_locked()
            state = self._state
            if state is None:
                raise RevisionError("rolling requires an initial replace/publish")
            raw = np.asarray(samples)
            expected_shape = schema_shape(self._schema)
            ndim = len(expected_shape)
            if raw.ndim != ndim:
                raise ValueError(f"rolling samples must be {ndim}-D, got {raw.ndim}-D")
            if isinstance(axis, bool) or not isinstance(axis, Integral):
                raise ValueError("rolling axis must be an integer")
            normalized_axis = int(axis)
            if normalized_axis < 0:
                normalized_axis += ndim
            if normalized_axis < 0 or normalized_axis >= ndim:
                raise ValueError(f"rolling axis {axis} is outside dataset ndim {ndim}")
            for dimension, (actual, expected) in enumerate(zip(raw.shape, expected_shape, strict=True)):
                if dimension != normalized_axis and actual != expected:
                    raise ValueError(
                        f"rolling sample shape {raw.shape} is incompatible with {expected_shape}"
                    )
            count = raw.shape[normalized_axis]
            capacity = expected_shape[normalized_axis]
            if count <= 0 or count > capacity:
                raise ValueError("rolling sample count must be between one and axis capacity")
            if raw.dtype != schema_dtype(self._schema):
                raise ValueError(
                    f"rolling sample dtype {raw.dtype} does not match schema dtype "
                    f"{schema_dtype(self._schema)}"
                )
            next_values = np.roll(snapshot_values(state), -count, axis=normalized_axis)
            next_validity = np.roll(snapshot_validity(state), -count, axis=normalized_axis)
            tail = [slice(None)] * ndim
            tail[normalized_axis] = slice(capacity - count, capacity)
            tail_index = tuple(tail)
            try:
                next_values[tail_index] = raw
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid rolling samples: {error}") from error
            if validity is None:
                next_validity[tail_index] = True
            else:
                raw_validity = np.asarray(validity)
                if raw_validity.dtype != np.dtype(np.bool_) or raw_validity.shape != raw.shape:
                    raise ValueError("rolling validity must be boolean and match sample shape")
                next_validity[tail_index] = raw_validity
            return self._accept_locked(
                owned_snapshot_from_arrays(
                    schema=self._schema,
                    values=next_values,
                    revision=revision,
                    validity=next_validity,
                )
            )

    def take_latest(self) -> OwnedSnapshot | None:
        item = self._channel.take_latest()
        return None if item is None else item.payload

    def wait_latest(self, timeout: float | None = None) -> OwnedSnapshot | None:
        item = self._channel.wait_latest(timeout)
        return None if item is None else item.payload

    def metrics(self) -> IngressMetrics:
        return self._channel.metrics()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._state = None
            self._channel.close()


__all__ = ["IngressMetrics", "LatestIngress", "LatestRevisionChannel", "RevisionedItem"]
