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
    collisions: dict[str, list[SignalPublication]] = {}
    parents: dict[object, tuple[SignalPublication, ...]] = {}
    while pending:
        current = pending.pop()
        if current.event_ref in parents:
            continue
        parents[current.event_ref] = tuple(resolve_parents(current))
        for name in current.signals:
            previous = by_name.get(name)
            if previous is not None and previous.event_ref != current.event_ref:
                collisions.setdefault(name, [previous]).append(current)
            else:
                by_name[name] = current
        pending.extend(parents[current.event_ref])
    # A run may derive from an older run of the same signal. Its ancestor
    # remains lineage, but does not compete as another visible value. Only
    # a candidate descending from every same-named event may replace them;
    # independent branches still have no coherent answer. Defer the choice
    # until traversal finishes so parent ordering cannot change the result.
    ancestors: dict[object, set[object]] = {}
    for name, candidates in collisions.items():
        required = {candidate.event_ref for candidate in candidates}
        for candidate in candidates:
            reachable = ancestors.get(candidate.event_ref)
            if reachable is None:
                reachable = set()
                pending = [candidate]
                while pending:
                    current = pending.pop()
                    if current.event_ref in reachable:
                        continue
                    reachable.add(current.event_ref)
                    pending.extend(parents[current.event_ref])
                ancestors[candidate.event_ref] = reachable
            if required.issubset(reachable):
                by_name[name] = candidate
                break
        else:
            return None
    return by_name


def _name_is_ancestor(
    ancestor: str,
    descendant: str,
    source_by_output: Mapping[str, str],
    bundle_of: Mapping[str, frozenset[str]],
) -> bool:
    """Whether ``descendant`` derives, along source routes, from ``ancestor``.

    Each step up a route lands on a SOURCE name, and that name is published
    with its siblings in one atomic event: the event that carries ``counts``
    carries ``occupied`` too, so a processor of ``counts`` descends from
    ``occupied`` as well.  Judged by the exact source name alone, a
    requested sibling was an independent leaf whose latest publication was
    set against the processor's older shot -- and a complete same-shot
    front the plane already held was reported as pending for as long as
    the processor lagged its source.
    """

    seen: set[str] = set()
    current = descendant
    while current in source_by_output and current not in seen:
        seen.add(current)
        current = source_by_output[current]
        if current == ancestor or ancestor in bundle_of.get(current, ()):
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
    fields used by the algorithm (retired, publication, terminal,
    output_names, kind, source_name, and coherent).  ``resolve_parents`` is
    the plane's read-only parent lookup.  No object in the input view is
    mutated.
    """

    states = _values(states_view)
    requested_names = frozenset(front_signals)
    latest: dict[str, SignalPublication] = {}
    active_names: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    source_by_output: dict[str, str] = {}
    #: Every active name -> the names one owner commits beside it.
    bundle_of: dict[str, frozenset[str]] = {}

    for state in states:
        if getattr(state, "retired"):
            continue
        publication = getattr(state, "publication")
        if getattr(state, "terminal") and publication is not None:
            state_active_names = set(publication.signals)
        else:
            state_active_names = set(getattr(state, "output_names"))
        active_names.update(state_active_names)
        bundle = frozenset(state_active_names)
        for name in bundle:
            bundle_of[name] = bundle
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
        if getattr(state, "kind") == "processor" and getattr(
            state, "coherent"
        ):
            # A presentation-paced follower (coherent=False) keeps lineage
            # but never joins its source's same-shot component: it advances
            # only after the source presents, so holding the source for it
            # would deadlock the whole component.
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
                and _name_is_ancestor(name, other, source_by_output, bundle_of)
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
            # Pending first/restarted outputs cannot be dropped from a group.
            # Reuse only a complete same-shot group in current generations;
            # otherwise leave the surface's last accepted picture untouched.
            fallback: dict[str, SignalPublication] = {}
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
                    continue
                fallback[name] = previous
            if not requested_component.issubset(fallback) or len({
                _publication_roots(fallback[name], resolve_parents)
                for name in requested_component
                if name in fallback
            }) != 1:
                fallback.clear()
            for name in component:
                selected.pop(name, None)
            selected.update(fallback)

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

    return SignalFront(signals, publications)


__all__ = ["build_front"]
