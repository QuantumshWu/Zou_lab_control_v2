"""Immutable display revisions and atomic parameter updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .parameters import FrozenParameters, ParameterSchema, RenderEffect


def normalize_interaction(values: Mapping[str, object]) -> FrozenParameters:
    """Committed gestures use portable identities, never renderer objects."""

    if not isinstance(values, Mapping) or set(values) - {"series_lock"}:
        raise ValueError("interaction fields must be series_lock")
    result = dict(values)
    if result.get("series_lock") is not None:
        result["series_lock"] = _series_target(result["series_lock"], readout=False)
    return FrozenParameters(result)


def normalize_presentation(values: Mapping[str, object]) -> FrozenParameters:
    """The transient readout frozen with a saved picture, not a shared pointer."""

    if not isinstance(values, Mapping) or set(values) - {"series_readout"}:
        raise ValueError("presentation fields must be series_readout")
    result = dict(values)
    if result.get("series_readout") is not None:
        result["series_readout"] = _series_target(result["series_readout"], readout=True)
    return FrozenParameters(result)


def _series_target(value: object, *, readout: bool) -> FrozenParameters:
    expected = {"key", "facet_index"} | ({"label", "mode"} if readout else set())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("series identity fields differ from the interaction contract")
    key = tuple(tuple(part) for part in value["key"])
    if not key or any(len(part) != 3 or not all(isinstance(item, str) for item in part) for part in key):
        raise ValueError("series key must contain domain, axis and canonical coordinate identities")
    facet = value["facet_index"]
    if facet is not None and (type(facet) is not int or facet < 0):
        raise ValueError("series facet_index must be a nonnegative integer or None")
    result = {"key": key, "facet_index": facet}
    if readout:
        if value["mode"] not in {"hover", "locked"} or not isinstance(value["label"], str):
            raise ValueError("series readout requires a label and hover/locked mode")
        result.update(label=value["label"], mode=value["mode"])
    return FrozenParameters(result)


@dataclass(frozen=True, slots=True)
class DisplayState:
    """One immutable, monotonically numbered display configuration."""

    revision: int
    values: FrozenParameters
    changed_names: frozenset[str]
    effects: RenderEffect
    interaction: FrozenParameters = field(default_factory=lambda: FrozenParameters({"series_lock": None}))

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("display revision must be an integer")
        if self.revision < 0:
            raise ValueError("display revision must be non-negative")
        if not isinstance(self.values, FrozenParameters):
            object.__setattr__(self, "values", FrozenParameters(self.values))
        if not isinstance(self.changed_names, frozenset):
            object.__setattr__(self, "changed_names", frozenset(self.changed_names))
        if not all(isinstance(name, str) for name in self.changed_names):
            raise TypeError("changed_names must contain strings")
        if not isinstance(self.effects, RenderEffect):
            raise TypeError("effects must be RenderEffect")
        object.__setattr__(self, "interaction", normalize_interaction(self.interaction))

    def __getitem__(self, name: str) -> Any:
        return self.values[name]


class DisplayStateStore:
    """Thread-safe owner of the current :class:`DisplayState`.

    A prepared transition replaces the complete immutable state in one locked
    operation, so render workers can safely retain an earlier revision while
    the UI continues editing.
    """

    __slots__ = ("_schema", "_lock", "_state")

    def __init__(
        self,
        schema: ParameterSchema,
        initial: Mapping[str, object] | None = None,
        *,
        initial_revision: int = 0,
        initial_effects: RenderEffect = RenderEffect.LAYOUT,
        initial_interaction: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(schema, ParameterSchema):
            raise TypeError("schema must be ParameterSchema")
        if isinstance(initial_revision, bool) or not isinstance(initial_revision, int):
            raise TypeError("initial_revision must be an integer")
        if initial_revision < 0:
            raise ValueError("initial_revision must be non-negative")
        if not isinstance(initial_effects, RenderEffect):
            raise TypeError("initial_effects must be RenderEffect")
        values = schema.initial_values(initial)
        self._schema = schema
        self._lock = RLock()
        self._state = DisplayState(
            revision=initial_revision,
            values=values,
            changed_names=frozenset(schema.names),
            effects=initial_effects,
            interaction=normalize_interaction({"series_lock": None} if initial_interaction is None else initial_interaction),
        )

    @property
    def state(self) -> DisplayState:
        with self._lock:
            return self._state

    def _commit_prepared(
        self,
        previous: DisplayState,
        candidate: FrozenParameters,
    ) -> DisplayState:
        """Atomically commit a transition already prepared by this store's schema."""

        with self._lock:
            if self._state is not previous:
                raise RuntimeError("display state changed before prepared commit")
            changed = frozenset(
                name
                for name in self._schema.names
                if previous.values[name] != candidate[name]
            )
            if not changed:
                return previous
            self._state = DisplayState(
                revision=previous.revision + 1,
                values=candidate,
                changed_names=changed,
                effects=self._schema.effects_for(changed),
                interaction=previous.interaction,
            )
            return self._state

    def _commit_interaction(self, updates: Mapping[str, object]) -> DisplayState:
        with self._lock:
            current = self._state
            wanted = normalize_interaction({**current.interaction, **updates})
            if wanted == current.interaction:
                return current
            self._state = DisplayState(current.revision + 1, current.values,
                frozenset(), RenderEffect.OVERLAY, wanted)
            return self._state

    def _restore_prepared(
        self,
        current: DisplayState,
        previous: DisplayState,
    ) -> None:
        """Roll back one unpresented transition under compare-and-swap."""

        if not isinstance(current, DisplayState) or not isinstance(
            previous, DisplayState
        ):
            raise TypeError("display rollback requires DisplayState values")
        with self._lock:
            if self._state is not current:
                raise RuntimeError("display state changed before rollback")
            self._state = previous


__all__ = [
    "DisplayState",
]
