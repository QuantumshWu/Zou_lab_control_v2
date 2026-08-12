"""Turning a gesture on a plot into numbers the runtime can derive from.

Dragging a box across a panel should produce a new signal -- the region's
frames, its mean, a fitted centre -- that everything downstream consumes like
any other.  That is the last metre of the live analysis chain, and it was the
piece nothing implemented.

Ownership, decided before any of this was written, because it is exactly the
kind of code that drifts into the wrong package:

* PUBLISHING belongs to zlc_runtime.  ``SelectionBridge`` already owns the
  lineage: a derived value published in the same breath as its parent, so a
  front never shows a region cut from a frame you are no longer looking at.
* MEANING belongs to zlc_atom.  What a region's occupancy IS, is physics.
* This module TRANSLATES, and does nothing else.  zlc_plot speaks in selectors,
  axis refs and futures; zlc_runtime speaks in axis names and bounds.  This is
  the only place both vocabularies are known, so neither leaks into the other.

Two constraints shaped the implementation and are easy to undo by accident:

* A selection callback can arrive on the raster worker thread (an unattached
  session dispatches inline), so this must never call back into the plot host
  while handling one -- that submits work to the very thread that is waiting.
  Everything needed therefore comes from the event, and the state the bridge
  re-reads on commit is served from what the event stream already said.
* zlc_plot swallows exceptions raised by an application callback, on purpose,
  so that one bad observer cannot disable the others.  Raising from here is
  therefore invisible.  A selection this cannot carry records its reason on
  :attr:`PlotSelectionSource.last_error` instead of vanishing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from zlc_plot import NumericRange, SelectorKind

from zlc_runtime import (
    FitEventValue,
    SelectionBridge,
    SelectionChange,
    SelectionRange,
    SelectionState,
)
from zlc_runtime.selection_bridge import FacetCondition


__all__ = [
    "PlotSelectionSource",
    "attach_selection_bridge",
    "subscribe_committed_selection",
]


#: Plot kinds the runtime can derive from.  Rolling x is plot-owned history,
#: not an axis of the current source snapshot, so it deliberately is absent.
_PLOT_KINDS = {
    "image": "image",
    "curve": "curve",
    "histogram": "histogram",
}

#: Selector kinds that describe a region.
_SELECTOR_KINDS = {"area": "area", "x_range": "x_range"}

#: Selector kinds that mark a point or a level.  A crosshair or threshold
#: cuts no region, so nothing downstream can be derived from one and a
#: translation failure is not the right thing to record -- doing so painted
#: every crosshair click as a panel failure.  They are still gestures the
#: PANEL owns (a threshold decides which population every point belongs to),
#: which is what ``on_threshold`` carries -- a crosshair reads a number off
#: the picture and changes no analysis, while a threshold decides which
#: population every point belongs to, and with it the fractions and the
#: fidelity both of a panel's surfaces report.
_POINT_SELECTOR_KINDS = {"crosshair", "threshold"}


def _apply_panel_threshold(host: Any, selector: Any) -> object:
    """Put the panel's chosen classifier threshold on one plot surface.

    The value is canonical, so it crosses between two hosts holding different
    data revisions unchanged, and it lands on whichever cell that host has
    open -- the panel keeps both of its hosts on the same cell.
    """

    return host.set_threshold_selector(float(selector.value), display=False)


def _remove_panel_threshold(host: Any) -> object:
    """Let this surface go back to the threshold its own fit proposes."""

    try:
        return host.remove_selector(SelectorKind.THRESHOLD, emit_change=False)
    except KeyError:
        return None


def _same_panel_threshold(left: Any, right: Any) -> bool:
    """Whether two threshold gestures say the same thing about the same cell."""

    if left is None or right is None:
        return left is None and right is None
    return (
        float(left.value) == float(right.value)
        and left.facet_index == right.facet_index
    )


def _apply_panel_selection(host: Any, selection: SelectionState) -> object:
    """Project one panel-owned canonical selection onto a plot surface."""

    ranges = selection.ranges
    if selection.selector_kind == "area":
        x, y = ranges
        return host.set_area_selector(
            NumericRange(x.lower, x.upper),
            NumericRange(y.lower, y.upper),
            display=False,
            emit_change=False,
        )
    value = ranges[0]
    return host.set_x_selector(
        value.lower,
        value.upper,
        display=False,
        emit_change=False,
    )


def _remove_panel_selection(host: Any, selection: SelectionState) -> object:
    """Remove the same selector kind from the panel's other plot surface."""

    return host.remove_selector(
        SelectorKind(selection.selector_kind),
        emit_change=False,
    )


def _apply_panel_viewport(host: Any, viewport: object | None) -> object:
    if viewport is None:
        return host.reset_viewport(emit_change=False)
    return host.set_viewport(viewport.x, viewport.y, emit_change=False)


class _Unbridgeable(Exception):
    """A selection this translator cannot express, with the reason why."""


class PlotSelectionSource:
    """One plot's selections and fits, as the numbers the runtime accepts.

    Implements both runtime ports -- the event source and the data reader --
    because they are two views of one translation, and splitting them across
    two objects would mean two copies of the vocabulary.
    """

    def __init__(
        self,
        host: Any,
        *,
        on_threshold: Callable[[Any], object] | None = None,
    ) -> None:
        self._host = host
        self._states: dict[str, SelectionState] = {}
        self._releases: list[Callable[[], None]] = []
        self._last_error: Exception | None = None
        self._on_threshold = on_threshold

    # ------------------------------------------------------------- properties

    @property
    def last_error(self) -> Exception | None:
        """Why the most recent gesture derived nothing, or None.

        Exceptions raised inside a plot callback are swallowed by design, so
        without this a selection that cannot be carried would simply produce
        no signal and no complaint.
        """

        return self._last_error

    # ---------------------------------------------------------- runtime ports

    def subscribe_selection(
        self,
        callback: Callable[[SelectionChange, SelectionState], object],
    ) -> Callable[[], None]:
        """Report committed and removed selections as the runtime's values.

        A drag in progress is deliberately not reported: deriving from every
        intermediate position publishes a stream of answers nobody asked for
        and makes the plane's history unreadable.  The operator's answer is
        where they let go.  Intermediate positions are still translated, so the
        state this serves stays current, and a translation failure is visible
        before the operator commits rather than after.
        """

        def _on_selection(event: object) -> None:
            if _selector_kind_of(event) in _POINT_SELECTOR_KINDS:
                # Derives nothing, records no failure -- and is still an
                # answer about this panel, so whoever owns the panel hears it.
                if (
                    self._on_threshold is not None
                    and _selector_kind_of(event) == "threshold"
                ):
                    removed = _change_of(event) is SelectionChange.REMOVED
                    self._deliver(
                        self._on_threshold, None if removed else event.selector
                    )
                return
            if _name_of(event.subject.plot_kind) == "rolling":
                # Rolling x is the plot's cross-revision history ordinal, not
                # an axis of the current source snapshot.
                self._last_error = None
                self._states.pop(_selector_kind_of(event), None)
                return
            change = _change_of(event)
            try:
                state = self._translate(event)
            except _Unbridgeable as error:
                self._last_error = error
                self._states.pop(_selector_kind_of(event), None)
                return
            self._last_error = None
            if change is SelectionChange.REMOVED:
                self._states.pop(state.selector_kind, None)
            else:
                self._states[state.selector_kind] = state
            if change in {SelectionChange.COMMITTED, SelectionChange.REMOVED}:
                self._deliver(callback, change, state)

        return self._install(self._host.subscribe_selection, _on_selection)

    def subscribe_viewport(
        self,
        callback: Callable[[SelectionState, object | None], object],
    ) -> Callable[[], None]:
        """Report zoom/pan as the same canonical range used by a producer."""

        def _on_viewport(event: object) -> None:
            try:
                state, display = _viewport_state(event)
            except Exception as error:
                self._last_error = error
                return
            self._last_error = None
            if state is not None:
                self._deliver(callback, state, display)

        return self._install(self._host.subscribe_viewport, _on_viewport)

    def subscribe_fit(
        self,
        callback: Callable[[FitEventValue], object],
    ) -> Callable[[], None]:
        """Report accepted fits as one parameter table.

        A scalar fit and a facet batch are the same table with one or many
        samples, which is why the runtime models only the batch shape.
        """

        def _on_fit(event: object) -> None:
            try:
                value = _fit_value(event)
            except _Unbridgeable as error:
                self._last_error = error
                return
            self._last_error = None
            self._deliver(callback, value)

        return self._install(self._host.subscribe_fit, _on_fit)

    def _deliver(self, callback: Callable[..., object], *payload: object) -> None:
        """Hand one translated event downstream, keeping its failure visible.

        This is the last frame that can see the error.  One more return and it
        is inside zlc_plot's callback guard, which swallows it deliberately so
        that one bad observer cannot disable the others -- leaving an operator
        with a box that derives nothing and says nothing about why.
        """

        try:
            callback(*payload)
        except Exception as error:
            self._last_error = error

    def selector_data(self, kind: str) -> SelectionState:
        """The current state of one selector, as the runtime's value.

        Served from the event stream rather than by asking the plot, because
        the bridge calls this while handling a callback that may be running on
        the plot's own worker thread.  It is not a cache of a separate truth:
        the plot's every selector change passes through here first.
        """

        state = self._states.get(str(kind))
        if state is None:
            raise KeyError(f"no {kind} selector is current on this panel")
        return state

    # ------------------------------------------------------------- life cycle

    def close(self) -> None:
        for release in reversed(self._releases):
            release()
        self._releases.clear()
        self._states.clear()

    # ---------------------------------------------------------------- private

    def _install(
        self,
        subscribe: Callable[[Any], Any],
        listener: Callable[[object], None],
    ) -> Callable[[], None]:
        """Subscribe through a host that may answer with a future.

        Resolved here, on the caller's thread at attach time, so that nothing
        downstream ever has to wait on the plot from inside an event.
        """

        release = _resolved(subscribe(listener))
        released = False

        def _once() -> None:
            nonlocal released
            if released:
                return
            released = True
            _resolved(release())

        self._releases.append(_once)
        return _once

    def _translate(self, event: object) -> SelectionState:
        """One plot selection event, as the runtime's numbers."""

        selector = event.selector
        subject = event.subject
        plot_kind = _PLOT_KINDS.get(_name_of(subject.plot_kind))
        if plot_kind is None:
            raise _Unbridgeable(
                f"a {_name_of(subject.plot_kind)} plot has no upstream region to derive"
            )
        selector_kind = _SELECTOR_KINDS.get(_name_of(selector.kind))
        if selector_kind is None:
            raise _Unbridgeable(
                f"a {_name_of(selector.kind)} selector marks a point, not a region"
            )
        if selector_kind == "area":
            # A bound axis resolves to None when it cuts the measured VALUE
            # rather than a named upstream axis -- a curve's y, a histogram's
            # x.  That is a description, not a failure: the region the
            # operator can select is the interval on whichever axis IS named,
            # so one named axis translates as a single-range selection.  Only
            # a box naming nothing has nothing to select on.
            named = tuple(
                (axis, bounds, role)
                for axis, bounds, role in (
                    (subject.x, selector.value.x, "x"),
                    (subject.y, selector.value.y, "y"),
                )
                if axis is not None
            )
            if not named:
                raise _Unbridgeable(
                    "this plot's bounds cut no named upstream axis, so there "
                    "is nothing to select on"
                )
            ranges = tuple(
                _range(axis, bounds, role) for axis, bounds, role in named
            )
            if len(ranges) == 1:
                selector_kind = "x_range"
        else:
            ranges = (_range(subject.x, selector.value, "x"),)
        facets, repeat_index = self._facet_scope(selector)
        return SelectionState(
            plot_kind=plot_kind,
            selector_kind=selector_kind,
            ranges=ranges,
            facets=facets,
            repeat_index=repeat_index,
            revision=int(selector.revision),
        )

    def _facet_scope(
        self,
        selector: object,
    ) -> tuple[tuple[FacetCondition, ...], int | None]:
        """The focused cell's identity, as the runtime's facet restriction.

        A selector drawn inside a focused facet cell carries the cell index,
        and dropping it meant a box in ONE cell derived a region cut from ALL
        cells.  A repeat facet is structural -- cell k IS repeat row k, the
        repeat axis has no name anywhere -- so it crosses as ``repeat_index``.
        A named facet axis needs its cell's coordinate, which is not in the
        event, so it is read from the owning session's current facet
        projection: plain state reads on the thread that owns them, never a
        host dispatch, which from inside this callback would re-enter the
        worker's own state machine.
        """

        index = getattr(selector, "facet_index", None)
        if index is None:
            return (), None
        session = self._session_view()
        if session is None:
            raise _Unbridgeable(
                "the focused facet cell's identity is unreachable from this "
                "plot host, so a cell-scoped selection cannot be carried"
            )
        facet = getattr(session.spec, "facet", None)
        if facet is None:
            # A focused index without a facet spec: nothing narrower than the
            # whole surface exists, so there is no restriction to add.
            return (), None
        domain = _name_of(getattr(facet, "domain", None))
        if domain == "repeat":
            return (), int(index)
        axis_name = getattr(facet, "axis_id", None)
        if not isinstance(axis_name, str) or not axis_name:
            raise _Unbridgeable(
                f"a {domain} facet has no named upstream axis, so the focused "
                "cell cannot be carried"
            )
        return (FacetCondition(axis_name, self._facet_value(session, index)),), None

    def _session_view(self) -> Any | None:
        """The owning session, reachable only where its state may be read.

        A raster host's selection callbacks run on its worker thread -- the
        one thread allowed to touch the session directly, which its adapter
        enforces.  An unattached session dispatches callbacks inline, so the
        host IS the session there.
        """

        host = self._host
        adapter = getattr(host, "_worker_adapter", None)
        if adapter is not None:
            try:
                return adapter._session()
            except RuntimeError:
                return None
        return host if hasattr(host, "spec") else None

    @staticmethod
    def _facet_value(session: Any, index: int) -> int | float | str:
        """The canonical facet coordinate of one cell of the current grid."""

        payload = getattr(
            getattr(session, "_projection", None), "_payload", None
        )
        cells = getattr(payload, "cells", None)
        if cells is None:
            raise _Unbridgeable(
                "the focused facet cell has no facet projection to identify it"
            )
        cell = next(
            (item for item in cells if item.facet_index == index),
            None,
        )
        if cell is None:
            raise _Unbridgeable(
                f"facet cell {index} is not present in the current projection"
            )
        value = cell.facet_value_canonical
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise _Unbridgeable(
                f"facet cell {index} has a non-scalar facet coordinate"
            )
        return value


def attach_selection_bridge(
    plane: object,
    host: Any,
    source_signal: str,
    *,
    bridge_id: str,
    source_publication_for: Callable[[int], object | None] | None = None,
    on_committed: Callable[[SelectionState], object] | None = None,
    on_removed: Callable[[SelectionState], object] | None = None,
    on_viewport: Callable[[SelectionState, object | None], object] | None = None,
    on_threshold: Callable[[Any], object] | None = None,
) -> tuple[SelectionBridge, PlotSelectionSource]:
    """Connect one panel's gestures to the plane, deriving as they commit.

    ``source_publication_for`` resolves a fit's exact parent publication by
    data revision — the panel port that rendered the fit is the causal
    holder, so the console passes its port's resolver here.

    Returns both so the caller can close them: the bridge outlives any single
    selection, and the subscription outlives the bridge only if it leaks.
    """

    source = PlotSelectionSource(host, on_threshold=on_threshold)
    bridge = SelectionBridge(
        plane,
        source_signal,
        source,
        source,
        bridge_id=bridge_id,
        source_publication_for=source_publication_for,
    )
    bridge.start()
    if on_committed is not None or on_removed is not None:
        if on_committed is not None and not callable(on_committed):
            bridge.close()
            source.close()
            raise TypeError("on_committed must be callable or None")
        if on_removed is not None and not callable(on_removed):
            bridge.close()
            source.close()
            raise TypeError("on_removed must be callable or None")

        def _on_selection(
            change: SelectionChange,
            selection: SelectionState,
        ) -> None:
            if change is SelectionChange.COMMITTED and on_committed is not None:
                on_committed(selection)
            elif change is SelectionChange.REMOVED and on_removed is not None:
                on_removed(selection)

        source.subscribe_selection(_on_selection)
    if on_viewport is not None:
        source.subscribe_viewport(on_viewport)
    return bridge, source


def subscribe_committed_selection(
    host: Any,
    callback: Callable[[SelectionState], object],
    *,
    on_removed: Callable[[SelectionState], object] | None = None,
    on_viewport: Callable[[SelectionState, object | None], object] | None = None,
    on_threshold: Callable[[Any], object] | None = None,
) -> PlotSelectionSource:
    """Translate committed gestures without publishing a derived signal.

    Panel Edit uses a frozen plotting host: its selector updates the direct
    producer draft, but must not create a second runtime derivation beside the
    live monitor panel.  The returned source owns the host subscription and is
    the one object the caller closes with that frozen surface.
    """

    if not callable(callback):
        raise TypeError("callback must be callable")
    if on_removed is not None and not callable(on_removed):
        raise TypeError("on_removed must be callable or None")
    source = PlotSelectionSource(host, on_threshold=on_threshold)

    def _on_selection(
        change: SelectionChange,
        selection: SelectionState,
    ) -> None:
        if change is SelectionChange.COMMITTED:
            callback(selection)
        elif change is SelectionChange.REMOVED and on_removed is not None:
            on_removed(selection)

    try:
        source.subscribe_selection(_on_selection)
        if on_viewport is not None:
            source.subscribe_viewport(on_viewport)
    except Exception:
        source.close()
        raise
    return source


# --------------------------------------------------------------- translation


def _range(axis: object, bounds: object, role: str) -> SelectionRange:
    if axis is None:
        raise _Unbridgeable(
            f"this plot's {role} bounds cut no named upstream axis, so there is "
            "nothing to select on"
        )
    domain = _name_of(getattr(axis, "domain", ""))
    if domain in {"repeat", "point_row"}:
        return SelectionRange(
            axis="",
            lower=float(bounds.low),
            upper=float(bounds.high),
            domain=domain,
        )
    name = getattr(axis, "axis_id", None)
    if not isinstance(name, str) or not name:
        raise _Unbridgeable(f"the {role} axis of this plot has no upstream name")
    return SelectionRange(axis=name, lower=float(bounds.low), upper=float(bounds.high))


def _viewport_state(event: object) -> tuple[SelectionState | None, object | None]:
    canonical, display, subject = event
    if _name_of(subject.plot_kind) == "rolling":
        return None, display
    plot_kind = _PLOT_KINDS.get(_name_of(subject.plot_kind))
    if plot_kind is None:
        return None, display
    x = getattr(subject, "x", None)
    y = getattr(subject, "y", None)
    if x is None:
        return None, display
    if y is None:
        return (
            SelectionState(
                plot_kind=plot_kind,
                selector_kind="x_range",
                ranges=(_range(x, canonical.x, "x"),),
            ),
            display,
        )
    return (
        SelectionState(
            plot_kind=plot_kind,
            selector_kind="area",
            ranges=(
                _range(x, canonical.x, "x"),
                _range(y, canonical.y, "y"),
            ),
        ),
        display,
    )


def _fit_value(event: object) -> FitEventValue:
    """One accepted fit, as the runtime's parameter table."""

    result = event.result
    results = getattr(result, "results", None)
    if results is None:
        return _scalar_fit_value(result)
    return _batch_fit_value(result)


def _scalar_fit_value(result: object) -> FitEventValue:
    names = tuple(result.parameter_names)
    units = dict(result.parameter_units)
    success = bool(result.success)
    values = np.asarray(result.parameter_values, dtype=np.float64).reshape(-1)
    errors = np.asarray(result.standard_errors, dtype=np.float64).reshape(-1)
    return FitEventValue(
        parameter_names=names,
        parameter_units={name: units.get(name, "") for name in names},
        parameter_values={
            name: np.array([values[index] if success else np.nan])
            for index, name in enumerate(names)
        },
        parameter_errors={
            name: np.array([errors[index]]) for index, name in enumerate(names)
        },
        success=np.array([success]),
        sample_axis_name="",
        sample_coordinates=np.array([0.0]),
        sample_unit="",
        sample_labels=None,
        source_revision=int(result.source_revision),
        batch_revision=int(result.batch_revision),
    )


def _batch_fit_value(batch: object) -> FitEventValue:
    names = tuple(batch.parameter_names)
    units = dict(batch.parameter_units)
    outcomes = tuple(batch.results)
    success = np.array(
        [item is not None and bool(item.success) for item in outcomes],
        dtype=np.bool_,
    )
    coordinates = batch.sample_coordinates
    if coordinates is None:
        raise _Unbridgeable("a facet fit batch without sample coordinates cannot be placed")

    def _column(index: int, attribute: str) -> np.ndarray:
        return np.array(
            [
                (
                    np.nan
                    if item is None or not item.success
                    else float(np.asarray(getattr(item, attribute)).reshape(-1)[index])
                )
                for item in outcomes
            ],
            dtype=np.float64,
        )

    return FitEventValue(
        parameter_names=names,
        parameter_units={name: units.get(name, "") for name in names},
        parameter_values={
            name: _column(index, "parameter_values") for index, name in enumerate(names)
        },
        parameter_errors={
            name: _column(index, "standard_errors") for index, name in enumerate(names)
        },
        success=success,
        sample_axis_name=str(batch.sample_axis_name),
        sample_coordinates=np.asarray(coordinates, dtype=np.float64).reshape(-1),
        sample_unit=str(batch.sample_unit),
        sample_labels=batch.sample_labels,
        source_revision=int(batch.source_revision),
        batch_revision=int(batch.batch_revision),
    )


# ------------------------------------------------------------------ plumbing


def _resolved(value: object) -> Any:
    """The value behind a host answer, whether or not it came as a future."""

    if hasattr(value, "result"):
        value = value.result()
    return getattr(value, "value", value)


def _name_of(value: object) -> str:
    return str(getattr(value, "value", value))


def _change_of(event: object) -> SelectionChange:
    return SelectionChange(_name_of(event.change))


def _selector_kind_of(event: object) -> str:
    return _name_of(event.selector.kind)
