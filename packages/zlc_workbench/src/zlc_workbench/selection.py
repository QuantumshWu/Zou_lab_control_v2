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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from zlc_plot import (
    NumericRange,
    SelectionSubject,
    SelectorKind,
)
from zlc_plot.selectors import RectangleRange, SelectorState as PlotSelectorState

from zlc_runtime import (
    FitEventValue,
    SelectionBridge,
    SelectionChange,
    SignalPublication,
    SelectionRange,
    SelectionState,
    selection_output_catalog,
)
from zlc_runtime.selection_bridge import FacetCondition


__all__ = [
    "PlotSelectionSource",
    "PlotSelectionObservation",
    "panel_selection_document",
    "panel_selection_from_document",
    "panel_selection_matches_subject",
    "panel_selection_derives_signal",
    "panel_selection_output_catalog",
    "observation_matches_plot_input",
    "plot_identity_matches_plot_input",
    "panel_plot_selectors",
    "attach_selection_bridge",
]


@dataclass(frozen=True, slots=True)
class PlotSelectionObservation:
    """One panel selector event plus the exact data moment it was drawn on."""

    change: SelectionChange
    state: SelectionState
    subject: SelectionSubject
    data_generation: str | None
    data_revision: int


def observation_matches_plot_input(
    observation: object,
    plot_input: object,
) -> bool:
    """Whether one Plot observation names this exact accepted Dataset."""

    return plot_identity_matches_plot_input(
        plot_input,
        getattr(observation, "data_generation", None),
        getattr(observation, "data_revision", -1),
    )


def plot_identity_matches_plot_input(
    plot_input: object,
    data_generation: object,
    data_revision: object,
) -> bool:
    """Whether plain generation/revision values name this exact Dataset."""

    snapshot = getattr(plot_input, "snapshot", plot_input)
    ref = getattr(snapshot, "ref", None)
    generation = getattr(getattr(ref, "stream_generation", None), "value", None)
    revision = getattr(getattr(ref, "revision", None), "value", None)
    try:
        selected_revision = int(data_revision)
    except (TypeError, ValueError):
        return False
    return bool(
        generation == str(data_generation)
        and revision == selected_revision
    )


#: Plot kinds the runtime can derive from.  Rolling x is a plot-owned display
#: ordinal over Runtime history, not an upstream source axis, so it is absent.
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


def panel_selection_document(selection: SelectionState | None) -> dict[str, Any]:
    """One committed region, as the plain numbers a layout can hold.

    Exact axis domains/ids and canonical bounds are what the region means.
    The revision it was drawn on is not part of it -- that is the moment, not
    the choice.
    """

    if selection is None:
        return {}
    return {
        "plot_kind": str(selection.plot_kind),
        "selector_kind": str(selection.selector_kind),
        "ranges": [
            {
                "axis": str(item.axis),
                "lower": float(item.lower),
                "upper": float(item.upper),
                "coordinate_frame": (
                    None
                    if item.coordinate_frame is None
                    else str(item.coordinate_frame)
                ),
                "domain": str(item.domain),
            }
            for item in selection.ranges
        ],
        "facets": [
            {
                "axis": str(item.axis),
                "value": item.value,
                "domain": str(item.domain),
            }
            for item in selection.facets
        ],
        "repeat_index": (
            None if selection.repeat_index is None else int(selection.repeat_index)
        ),
    }


def panel_selection_from_document(document: Mapping[str, Any]) -> SelectionState | None:
    """The region a layout wrote down, back in the runtime's own words."""

    if not document:
        return None
    expected = {"plot_kind", "selector_kind", "ranges", "facets", "repeat_index"}
    if set(document) != expected:
        raise ValueError("panel selector fields do not match the current grammar")
    raw_ranges = document["ranges"]
    raw_facets = document["facets"]
    if not isinstance(raw_ranges, (list, tuple)):
        raise TypeError("panel selector ranges must be a sequence")
    if not isinstance(raw_facets, (list, tuple)):
        raise TypeError("panel selector facets must be a sequence")
    for item in raw_ranges:
        if not isinstance(item, Mapping) or set(item) != {
            "axis",
            "lower",
            "upper",
            "coordinate_frame",
            "domain",
        }:
            raise ValueError("panel selector range fields do not match the current grammar")
    for item in raw_facets:
        if not isinstance(item, Mapping) or set(item) != {
            "axis",
            "value",
            "domain",
        }:
            raise ValueError("panel selector facet fields do not match the current grammar")
    return SelectionState(
        plot_kind=str(document["plot_kind"]),
        selector_kind=str(document["selector_kind"]),
        ranges=tuple(
            SelectionRange(
                axis=str(item["axis"]),
                lower=float(item["lower"]),
                upper=float(item["upper"]),
                coordinate_frame=item.get("coordinate_frame"),
                domain=str(item["domain"]),
            )
            for item in raw_ranges
        ),
        facets=tuple(
            FacetCondition(
                str(item["axis"]),
                item["value"],
                str(item["domain"]),
            )
            for item in raw_facets
        ),
        repeat_index=document["repeat_index"],
    )


def panel_plot_selectors(
    selection: SelectionState | None,
    *,
    facet_index: int | None,
) -> tuple[PlotSelectorState, ...]:
    """Translate one saved panel target into zlc_plot's native selectors."""

    states: list[PlotSelectorState] = []
    if selection is not None:
        ranges = tuple(selection.ranges)
        if selection.selector_kind == "area":
            if len(ranges) != 2:
                raise ValueError("an area panel selection requires x and y ranges")
            states.append(PlotSelectorState(
                SelectorKind.AREA,
                RectangleRange(
                    NumericRange(ranges[0].lower, ranges[0].upper),
                    NumericRange(ranges[1].lower, ranges[1].upper),
                ),
                facet_index=facet_index,
            ))
        elif selection.selector_kind == "x_range":
            if len(ranges) != 1:
                raise ValueError("an x-range panel selection requires one range")
            states.append(PlotSelectorState(
                SelectorKind.X_RANGE,
                NumericRange(ranges[0].lower, ranges[0].upper),
                facet_index=facet_index,
            ))
        else:
            raise ValueError(
                f"unsupported panel selector kind {selection.selector_kind!r}"
            )
    return tuple(states)


def panel_selection_matches_subject(
    selection: SelectionState,
    subject: SelectionSubject,
) -> bool:
    """Whether a stored canonical selection still names this exact surface.

    Plot owns the subject.  Workbench only translates that immutable subject
    into Runtime's neutral range/facet vocabulary and compares the result;
    it never reconstructs Histogram, scope or facet semantics from a spec.
    """

    if not isinstance(selection, SelectionState):
        raise TypeError("selection must be SelectionState")
    if not isinstance(subject, SelectionSubject):
        raise TypeError("subject must be SelectionSubject")
    plot_kind = _PLOT_KINDS.get(_name_of(subject.plot_kind))
    if plot_kind is None or selection.plot_kind != plot_kind:
        return False
    dummy = NumericRange(0.0, 1.0)
    try:
        if selection.selector_kind == "x_range":
            expected_ranges = (
                _range(
                    subject.x,
                    dummy,
                    "x",
                    subject.x_coordinate_frame,
                ),
            )
        elif selection.selector_kind == "area":
            if subject.x is None or subject.y is None:
                return False
            expected_ranges = (
                _range(subject.x, dummy, "x", subject.x_coordinate_frame),
                _range(subject.y, dummy, "y", subject.y_coordinate_frame),
            )
        else:
            return False
        expected_facets, expected_repeat = _subject_scope(subject)
    except _Unbridgeable:
        return False

    def range_identity(item: SelectionRange) -> tuple[str, str, str | None]:
        return item.domain, item.axis, item.coordinate_frame

    actual_facets = {
        (condition.domain, condition.axis): condition.value
        for condition in selection.facets
    }
    wanted_facets = {
        (condition.domain, condition.axis): condition.value
        for condition in expected_facets
    }
    return bool(
        tuple(map(range_identity, selection.ranges))
        == tuple(map(range_identity, expected_ranges))
        and len(actual_facets) == len(selection.facets)
        and actual_facets == wanted_facets
        and selection.repeat_index == expected_repeat
    )


def panel_selection_output_catalog(
    subject: SelectionSubject | None,
) -> tuple[tuple[str, str], ...]:
    """Runtime outputs this accepted surface can actually derive."""

    if subject is None:
        return ()
    if not isinstance(subject, SelectionSubject):
        raise TypeError("subject must be SelectionSubject or None")
    if subject.x is None:
        return ()
    selector_kind = "area" if subject.y is not None else "x_range"
    return selection_output_catalog(selector_kind)


def panel_selection_derives_signal(selection: SelectionState) -> bool:
    """Whether this panel-local selector also names upstream Dataset axes."""

    if not isinstance(selection, SelectionState):
        raise TypeError("selection must be SelectionState")
    return all(item.domain != "value" for item in selection.ranges)


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
        on_threshold: Callable[[object], object] | None = None,
        on_crosshair: Callable[[object], object] | None = None,
    ) -> None:
        self._host = host
        self._releases: list[Callable[[], None]] = []
        self._subscription_lock = Lock()
        self._closed = False
        self._last_error: Exception | None = None
        self._on_threshold = on_threshold
        self._on_crosshair = on_crosshair

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

    def subscribe_observation(
        self,
        callback: Callable[[PlotSelectionObservation], object],
    ) -> Callable[[], None]:
        """Report panel-owned regions with their exact rendered data identity."""

        if not callable(callback):
            raise TypeError("selection observation callback must be callable")

        def _on_selection(event: object) -> None:
            if _selector_kind_of(event) in _POINT_SELECTOR_KINDS:
                if (
                    self._on_threshold is not None
                    and _selector_kind_of(event) == "threshold"
                ):
                    self._deliver(
                        self._on_threshold,
                        event,
                    )
                elif (
                    self._on_crosshair is not None
                    and _selector_kind_of(event) == "crosshair"
                ):
                    # A crosshair cuts no region either, but it IS the
                    # panel's marker: both of a panel's surfaces point at
                    # the same place or the operator is reading two
                    # different numbers off "one" panel.
                    self._deliver(
                        self._on_crosshair,
                        event,
                    )
                return
            if _name_of(event.subject.plot_kind) == "rolling":
                self._last_error = None
                return
            change = _change_of(event)
            try:
                state = self._translate(event)
            except _Unbridgeable as error:
                self._last_error = error
                return
            self._last_error = None
            if change in {SelectionChange.COMMITTED, SelectionChange.REMOVED}:
                self._deliver(
                    callback,
                    PlotSelectionObservation(
                        change,
                        state,
                        event.subject,
                        (
                            None
                            if event.data_generation is None
                            else str(event.data_generation)
                        ),
                        int(event.data_revision),
                    ),
                )

        return self._install(self._host.subscribe_selection, _on_selection)

    def subscribe_viewport_observation(
        self,
        callback: Callable[[object], object],
    ) -> Callable[[], None]:
        """Report a viewport with the exact rendered data identity."""

        if not callable(callback):
            raise TypeError("viewport observation callback must be callable")

        def _on_viewport(event: object) -> None:
            self._last_error = None
            self._deliver(callback, event)

        return self._install(self._host.subscribe_viewport, _on_viewport)

    def subscribe_focus_observation(
        self,
        callback: Callable[
            [int | None, SelectionSubject, str | None, int],
            object,
        ],
    ) -> Callable[[], None]:
        """Report accepted facet focus with its exact rendered data identity."""

        if not callable(callback):
            raise TypeError("focus observation callback must be callable")

        def _on_focus(
            focused_index: int | None,
            subject: SelectionSubject,
            data_generation: str | None,
            data_revision: int,
        ) -> None:
            self._last_error = None
            self._deliver(
                callback,
                focused_index,
                subject,
                data_generation,
                data_revision,
            )

        return self._install(self._host.subscribe_facet_focus, _on_focus)

    def subscribe_fit(
        self,
        callback: Callable[[FitEventValue | None], object],
    ) -> Callable[[], None]:
        """Report accepted fits as one parameter table, and their withdrawal.

        A scalar fit and a facet batch are the same table with one or many
        samples, which is why the runtime models only the batch shape.
        ``None`` means the fit is gone -- the answer is no longer on screen,
        so it is no longer published either.
        """

        def _on_fit(event: object) -> None:
            if event is None:
                self._last_error = None
                self._deliver(callback, None)
                return
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

        with self._subscription_lock:
            if self._closed:
                return
        try:
            callback(*payload)
        except Exception as error:
            self._last_error = error

    # ------------------------------------------------------------- life cycle

    def close(self) -> None:
        with self._subscription_lock:
            if self._closed:
                return
            self._closed = True
            releases = tuple(reversed(self._releases))
            self._releases.clear()
        for release in releases:
            release()

    # ---------------------------------------------------------------- private

    def _install(
        self,
        subscribe: Callable[[Any], Any],
        listener: Callable[[object], None],
    ) -> Callable[[], None]:
        """Install and retire a plot subscription without waiting on its worker."""

        with self._subscription_lock:
            if self._closed:
                raise RuntimeError("plot selection source is closed")
        active = True
        installed: Callable[[], object] | None = None

        def settle_release(release: Callable[[], object]) -> None:
            try:
                answer = release()
            except Exception as error:
                self._last_error = error
                return
            add_done = getattr(answer, "add_done_callback", None)
            if not callable(add_done):
                return

            def released(done: object) -> None:
                try:
                    done.result()
                except Exception as error:
                    self._last_error = error

            add_done(released)

        def _once() -> None:
            nonlocal active, installed
            with self._subscription_lock:
                if not active:
                    return
                active = False
                release, installed = installed, None
            if release is not None:
                settle_release(release)

        def subscribed(answer: object) -> None:
            nonlocal installed
            try:
                result = answer.result() if hasattr(answer, "result") else answer
                release = getattr(result, "value", result)
                if not callable(release):
                    raise TypeError("plot subscription did not return a release callable")
            except Exception as error:
                self._last_error = error
                return
            with self._subscription_lock:
                if active and not self._closed:
                    installed = release
                    release = None
            if release is not None:
                settle_release(release)

        with self._subscription_lock:
            self._releases.append(_once)
        try:
            answer = subscribe(listener)
        except BaseException:
            with self._subscription_lock:
                if _once in self._releases:
                    self._releases.remove(_once)
                active = False
            raise
        add_done = getattr(answer, "add_done_callback", None)
        if callable(add_done):
            add_done(subscribed)
        else:
            subscribed(answer)
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
                (axis, bounds, role, frame)
                for axis, bounds, role, frame in (
                    (
                        subject.x,
                        selector.value.x,
                        "x",
                        subject.x_coordinate_frame,
                    ),
                    (
                        subject.y,
                        selector.value.y,
                        "y",
                        subject.y_coordinate_frame,
                    ),
                )
                if axis is not None
            )
            if not named:
                raise _Unbridgeable(
                    "this plot's bounds cut no named upstream axis, so there "
                    "is nothing to select on"
                )
            ranges = tuple(
                _range(axis, bounds, role, frame)
                for axis, bounds, role, frame in named
            )
            if len(ranges) == 1:
                selector_kind = "x_range"
        else:
            ranges = (
                _range(
                    subject.x,
                    selector.value,
                    "x",
                    subject.x_coordinate_frame,
                ),
            )
        facets, repeat_index = _subject_scope(subject)
        return SelectionState(
            plot_kind=plot_kind,
            selector_kind=selector_kind,
            ranges=ranges,
            facets=facets,
            repeat_index=repeat_index,
            revision=int(selector.revision),
        )


def attach_selection_bridge(
    plane: object,
    host: Any,
    source_signal: str,
    *,
    bridge_id: str,
    source_publication_for: (
        Callable[[str, int], object | None] | None
    ) = None,
    request_owner_wake: Callable[[], object] | None = None,
    initial_selection: SelectionState | None = None,
    initial_publication: SignalPublication | None = None,
    on_observation: Callable[
        [SelectionBridge, PlotSelectionObservation, SignalPublication],
        object,
    ],
    on_threshold: Callable[[Any], object] | None = None,
    on_crosshair: Callable[[Any], object] | None = None,
) -> tuple[SelectionBridge, PlotSelectionSource]:
    """Resolve exact observations for the interaction owner that commits them.

    ``source_publication_for`` resolves an event's exact parent publication by
    data generation and revision.  Without a presentation-side holder, only the plane's
    current publication may answer, and only when both its generation and
    sequence match the observation exactly.

    Returns both so the caller can close them: the bridge outlives any single
    selection, and the subscription outlives the bridge only if it leaks.
    """

    source = PlotSelectionSource(
        host, on_threshold=on_threshold, on_crosshair=on_crosshair
    )
    if not callable(on_observation):
        raise TypeError("on_observation must be callable")
    bridge = SelectionBridge(
        plane,
        source_signal,
        source,
        bridge_id=bridge_id,
        source_publication_for=source_publication_for,
        request_owner_wake=request_owner_wake,
    )

    def exact_publication(observation: PlotSelectionObservation):
        publication = (
            source_publication_for(
                str(observation.data_generation),
                observation.data_revision,
            )
            if source_publication_for is not None
            and observation.data_generation is not None
            else plane.latest_publication(source_signal)
        )
        if publication is None:
            return None
        if not isinstance(publication, SignalPublication):
            raise TypeError(
                "selection publication resolver must return SignalPublication or None"
            )
        source_value = publication.value(source_signal)
        if source_value is None or not observation_matches_plot_input(
            observation,
            source_value.snapshot,
        ):
            return None
        return publication

    def route_observation(observation: PlotSelectionObservation) -> None:
        publication = exact_publication(observation)
        if publication is None:
            return
        on_observation(bridge, observation, publication)

    try:
        bridge.start(
            initial_selection=initial_selection,
            initial_publication=initial_publication,
        )
        source.subscribe_observation(route_observation)
    except BaseException:
        bridge.close()
        source.close()
        raise
    return bridge, source


# --------------------------------------------------------------- translation


def _range(
    axis: object,
    bounds: object,
    role: str,
    coordinate_frame: str | None = None,
) -> SelectionRange:
    if axis is None:
        if role == "x":
            return SelectionRange(
                axis="",
                lower=float(bounds.low),
                upper=float(bounds.high),
                coordinate_frame=coordinate_frame,
                domain="value",
            )
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
            coordinate_frame=coordinate_frame,
            domain=domain,
        )
    name = getattr(axis, "axis_id", None)
    if not isinstance(name, str) or not name:
        raise _Unbridgeable(f"the {role} axis of this plot has no upstream name")
    return SelectionRange(
        axis=name,
        lower=float(bounds.low),
        upper=float(bounds.high),
        domain=domain,
        coordinate_frame=coordinate_frame,
    )


def _subject_scope(
    subject: object,
) -> tuple[tuple[FacetCondition, ...], int | None]:
    """Canonical panel/facet scope frozen into one plot event."""

    conditions: list[FacetCondition] = []
    for ref, value in subject.scope:
        domain = _name_of(ref.domain)
        axis_name = "" if ref.axis_id is None else ref.axis_id
        if domain != "point_row" and (
            not isinstance(axis_name, str) or not axis_name
        ):
            raise _Unbridgeable(
                f"a {domain} scoped axis has no upstream name, so the "
                "selection cannot be carried"
            )
        conditions.append(FacetCondition(axis_name, value, domain))
    return tuple(conditions), subject.repeat_index


def _fit_value(event: object) -> FitEventValue:
    """One accepted fit, as the runtime's parameter table."""

    result = event.result
    results = getattr(result, "results", None)
    if results is None:
        return _scalar_fit_value(result, event.source_generation)
    return _batch_fit_value(result, event.source_generation)


def _scalar_fit_value(result: object, source_generation: str) -> FitEventValue:
    names = tuple(result.parameter_names)
    units = dict(result.parameter_units)
    success = bool(result.success)
    values = np.asarray(result.parameter_values, dtype=np.float64).reshape(-1)
    errors = np.asarray(result.standard_errors, dtype=np.float64).reshape(-1)
    error_validity = dict(result.parameter_error_validity)
    return FitEventValue(
        parameter_names=names,
        parameter_units={name: units.get(name, "") for name in names},
        parameter_values={
            name: np.array([values[index] if success else np.nan])
            for index, name in enumerate(names)
        },
        parameter_errors={
            name: np.array([
                errors[index] if error_validity[name] else np.nan
            ])
            for index, name in enumerate(names)
        },
        success=np.array([success]),
        sample_axis_domain="",
        sample_axis_id="",
        sample_axis_name="",
        sample_coordinates=np.array([0.0]),
        sample_unit="",
        sample_labels=None,
        source_generation=str(source_generation),
        source_revision=int(result.source_revision),
        batch_revision=int(result.batch_revision),
    )


def _batch_fit_value(batch: object, source_generation: str) -> FitEventValue:
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
    parameter_values = batch.parameter_values
    parameter_errors = batch.parameter_errors

    return FitEventValue(
        parameter_names=names,
        parameter_units={name: units.get(name, "") for name in names},
        parameter_values={
            name: np.asarray(parameter_values[name], dtype=np.float64)
            for name in names
        },
        parameter_errors={
            name: np.asarray(parameter_errors[name], dtype=np.float64)
            for name in names
        },
        success=success,
        sample_axis_domain=str(batch.facet.domain.value),
        sample_axis_id=str(batch.facet.axis_id or ""),
        sample_axis_name=str(batch.sample_axis_name),
        sample_coordinates=np.asarray(coordinates, dtype=np.float64).reshape(-1),
        sample_unit=str(batch.sample_unit),
        sample_labels=batch.sample_labels,
        source_generation=str(source_generation),
        source_revision=int(batch.source_revision),
        batch_revision=int(batch.batch_revision),
    )


# ------------------------------------------------------------------ plumbing


def _name_of(value: object) -> str:
    return str(getattr(value, "value", value))


def _change_of(event: object) -> SelectionChange:
    return SelectionChange(_name_of(event.change))


def _selector_kind_of(event: object) -> str:
    return _name_of(event.selector.kind)
