"""Pure lineage and coherent-front construction.

The plane owns the mutable generation registry and the private parent payload
store.  This module owns only the read-only algorithm that joins a requested
signal component, so the same-shot invariant can be tested without locks,
threads, or a live plane.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from .plane import SignalFront, SignalPublication


def _values(states_view: Mapping[object, object] | Iterable[object]) -> tuple[object, ...]:
    if isinstance(states_view, Mapping):
        return tuple(states_view.values())
    return tuple(states_view)


def _state_for_signal(states: tuple[object, ...], name: str) -> object | None:
    selected = None
    for state in states:
        if getattr(state, "retired") or name not in getattr(state, "output_names"):
            continue
        if selected is not None:
            raise RuntimeError(f"signal {name!r} has more than one generation owner")
        selected = state
    return selected


def _publication_roots(
    publication: SignalPublication,
    resolve_parents: Callable[[SignalPublication], Iterable[SignalPublication]],
) -> frozenset[object]:
    pending = [publication]
    roots: set[object] = set()
    seen: set[object] = set()
    while pending:
        current = pending.pop()
        if current.event_ref in seen:
            continue
        seen.add(current.event_ref)
        parents = tuple(resolve_parents(current))
        if parents:
            pending.extend(parents)
        else:
            roots.add(current.event_ref)
    return frozenset(roots)


def _collect_ancestry(
    publication: SignalPublication,
    resolve_parents: Callable[[SignalPublication], Iterable[SignalPublication]],
) -> Mapping[str, SignalPublication] | None:
    pending = [publication]
    by_name: dict[str, SignalPublication] = {}
    seen: set[object] = set()
    while pending:
        current = pending.pop()
        if current.event_ref in seen:
            continue
        seen.add(current.event_ref)
        for name in current.signals:
            previous = by_name.get(name)
            if previous is not None and previous.event_ref != current.event_ref:
                return None
            by_name[name] = current
        pending.extend(resolve_parents(current))
    return by_name


def _name_is_ancestor(
    ancestor: str,
    descendant: str,
    source_by_output: Mapping[str, str],
) -> bool:
    seen: set[str] = set()
    current = descendant
    while current in source_by_output and current not in seen:
        seen.add(current)
        current = source_by_output[current]
        if current == ancestor:
            return True
    return False


def build_front(
    states_view: Mapping[object, object] | Iterable[object],
    front_signals: Iterable[str],
    previous_front: SignalFront | None,
    resolve_parents: Callable[[SignalPublication], Iterable[SignalPublication]],
) -> SignalFront:
    """Build one coherent front from an immutable state view.

    ``states_view`` is read once; its members must expose the seven state
    fields used by the algorithm (retired, failure, publication, terminal,
    output_names, kind, and source_name).  ``resolve_parents`` is the plane's
    read-only parent lookup.  No object in the input view is mutated.
    """

    states = _values(states_view)
    requested_names = frozenset(front_signals)
    latest: dict[str, SignalPublication] = {}
    failures: dict[str, str] = {}
    active_names: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    source_by_output: dict[str, str] = {}

    for state in states:
        if getattr(state, "retired"):
            continue
        owner_id = getattr(state, "owner_id")
        failure = getattr(state, "failure")
        if failure is not None:
            failures[owner_id] = failure
        publication = getattr(state, "publication")
        if getattr(state, "terminal") and publication is not None:
            state_active_names = set(publication.signals)
        else:
            state_active_names = set(getattr(state, "output_names"))
        active_names.update(state_active_names)
        if publication is not None:
            for name in publication.signals:
                if name in latest:
                    raise RuntimeError(f"signal {name!r} has two publications")
                latest[name] = publication
            siblings = tuple(publication.signals)
            for name in siblings:
                adjacency.setdefault(name, set()).update(
                    candidate for candidate in siblings if candidate != name
                )
        if getattr(state, "kind") in {"processor", "continuous"}:
            source_name = getattr(state, "source_name")
            if source_name is not None:
                for output_name in getattr(state, "output_names"):
                    source_by_output[output_name] = source_name
                    adjacency.setdefault(output_name, set()).add(source_name)
                    adjacency.setdefault(source_name, set()).add(output_name)

    requested = requested_names.intersection(active_names)
    pending = set(requested)
    components: list[set[str]] = []
    while pending:
        seed = min(pending)
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour in adjacency.get(current, ()):
                if neighbour in active_names and neighbour not in component:
                    component.add(neighbour)
                    stack.append(neighbour)
        pending.difference_update(component)
        components.append(component)

    selected = dict(latest)
    previous_publications = (
        {} if previous_front is None else previous_front.publication_by_signal
    )
    for component in components:
        requested_component = requested.intersection(component)
        leaves = tuple(
            name
            for name in sorted(requested_component)
            if not any(
                other != name
                and _name_is_ancestor(name, other, source_by_output)
                for other in requested_component
            )
        )
        leaf_publications = tuple(latest.get(name) for name in leaves)
        coherent = bool(leaves) and all(
            publication is not None for publication in leaf_publications
        )
        ancestry: dict[str, SignalPublication] = {}
        if coherent:
            root_sets = {
                _publication_roots(publication, resolve_parents)
                for publication in leaf_publications
                if publication is not None
            }
            coherent = len(root_sets) == 1
        if coherent:
            for publication in leaf_publications:
                assert publication is not None
                current = _collect_ancestry(publication, resolve_parents)
                if current is None:
                    coherent = False
                    break
                for name, candidate in current.items():
                    previous = ancestry.get(name)
                    if previous is not None and previous.event_ref != candidate.event_ref:
                        coherent = False
                        break
                    ancestry[name] = candidate
                if not coherent:
                    break
        if coherent and any(name not in ancestry for name in requested_component):
            coherent = False

        if coherent:
            for name, publication in ancestry.items():
                if name in component and name in active_names:
                    selected[name] = publication
        else:
            for name in component:
                previous = previous_publications.get(name)
                current_state = _state_for_signal(states, name)
                if (
                    previous is None
                    or name not in active_names
                    or current_state is None
                    or getattr(current_state, "owner_id")
                    != previous.event_ref.stream_id.value
                    or getattr(current_state, "generation")
                    != previous.event_ref.generation
                ):
                    selected.pop(name, None)
                else:
                    selected[name] = previous

    signals: dict[str, object] = {}
    publications: dict[str, SignalPublication] = {}
    for name, publication in selected.items():
        if name not in active_names:
            continue
        value = publication.value(name)
        if value is None:
            raise RuntimeError("front publication lost one of its signals")
        signals[name] = value
        publications[name] = publication

    groups: dict[str, frozenset[str]] = {}
    for component in components:
        members = frozenset(name for name in requested.intersection(component) if name in signals)
        for name in members:
            groups[name] = members
    return SignalFront(signals, failures, publications, groups)


__all__ = ["build_front"]

