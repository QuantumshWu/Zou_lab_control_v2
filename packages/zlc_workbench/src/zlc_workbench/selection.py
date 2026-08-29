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
    DrawnRegion,
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
    "panel_selection_binds_a_revision",
    "panel_selection_output_catalog",
    "observation_matches_plot_input",
    "observation_predates_plot_input",
    "same_plot_run",
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


def same_plot_run(left: object, right: object) -> bool:
    """Whether two datasets are two moments of the SAME stream generation."""

    def generation(carrier: object) -> object | None:
        snapshot = getattr(carrier, "snapshot", carrier)
        ref = getattr(snapshot, "ref", None)
        return getattr(getattr(ref, "stream_generation", None), "value", None)

    first = generation(left)
    return first is not None and first == generation(right)


def observation_predates_plot_input(
    observation: object,
    plot_input: object,
) -> bool:
    """Whether a region was drawn on an OLDER picture than this Dataset.

    Staleness is being older, not being different.  A host renders every
    revision it is handed, while the bookkeeping that ACCEPTS a surface
    runs on the board's beat -- so the picture a hand draws on is routinely
    AHEAD of the one the panel has accepted.  Measured on a live camera
    panel, every single committed region arrived exactly one revision
    ahead, idle and streaming alike.

    Demanding the exact revision therefore refused every region an operator
    committed on a live panel, for ever: the mark appeared on the plot, the
    panel never heard of it, nothing was cut from the new region, and the
    remembered old one was re-applied over the top -- which is what "I
    moved it and it went back" looks like from the outside, and why the
    published ROI never changed shape again after the first one.
    """

    snapshot = getattr(plot_input, "snapshot", plot_input)
    ref = getattr(snapshot, "ref", None)
    revision = getattr(getattr(ref, "revision", None), "value", None)
    try:
        drawn_on = int(getattr(observation, "data_revision", None))
        accepted = int(revision)
    except (TypeError, ValueError):
        # Nothing to compare is not evidence of currency.
        return True
    return drawn_on < accepted


def same_plot_generation(observation: object, plot_input: object) -> bool:
    """Whether an observation was drawn on the same RUN as this Dataset.

    The weaker half of :func:`plot_identity_matches_plot_input`: the same
    stream generation, whatever revision it has reached.  It is the right
    question for a region that derives nothing -- what such a region names
    is a place on a picture, and a later shot does not move it -- while a
    region something IS cut from still has to name the exact revision it
    was cut from.
    """

    snapshot = getattr(plot_input, "snapshot", plot_input)
    ref = getattr(snapshot, "ref", None)
    generation = getattr(getattr(ref, "stream_generation", None), "value", None)
    if generation is None:
        return False
    return generation == str(getattr(observation, "data_generation", None))


#: Plot kinds the runtime can derive from -- all of them.  A region cuts
#: the signal it was drawn on whatever surface drew it.
_PLOT_KINDS = {
    "image": "image",
    "curve": "curve",
    "histogram": "histogram",
    "rolling": "rolling",
}

#: Selector kinds that describe a region.
_SELECTOR_KINDS = {"area": "area", "x_range": "x_range"}

#: Bounds that name no upstream axis: the measured value and the shot
#: ordinal.  They still cut the signal -- a value band decides which cells
#: count, a shot window which publications answer -- but they cut it the
#: same way on every revision, so a region made only of them is not tied to
#: the exact picture it was drawn on.
_NON_AXIS_DOMAINS = frozenset({"value", "shot"})


def _rolling_ranges(
    selector_kind: str,
    x_bounds: object,
    y_bounds: object = None,
) -> tuple[SelectionRange, ...]:
    """A rolling region's bounds: the shot ordinal, then the value.

    THE rule, in one place, because it is needed twice -- to BUILD a region
    from a gesture and to RECOGNISE a stored one as belonging to the surface
    in front of you.  Written twice, the two drifted the moment rolling was
    added to one of them, and a region that translated perfectly was thrown
    away by the matcher a few frames later.
    """

    pairs = ((x_bounds, "shot"), (y_bounds, "value"))
    return tuple(
        SelectionRange(
            axis="",
            lower=float(bounds.low),
            upper=float(bounds.high),
            domain=domain,
        )
        for bounds, domain in pairs
        if bounds is not None
    )

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
        "drawn": (
            None
            if selection.drawn is None
            else {
                "kind": str(selection.drawn.kind),
                "lower": (
                    None if selection.drawn.lower is None
                    else float(selection.drawn.lower)
                ),
                "upper": (
                    None if selection.drawn.upper is None
                    else float(selection.drawn.upper)
                ),
            }
        ),
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
    expected = {
        "plot_kind", "selector_kind", "drawn", "ranges", "facets", "repeat_index",
    }
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
    raw_drawn = document["drawn"]
    if raw_drawn is not None and (
        not isinstance(raw_drawn, Mapping)
        or set(raw_drawn) != {"kind", "lower", "upper"}
    ):
        raise ValueError("panel selector drawn fields do not match the current grammar")
    return SelectionState(
        plot_kind=str(document["plot_kind"]),
        selector_kind=str(document["selector_kind"]),
        drawn=(
            None
            if raw_drawn is None
            else DrawnRegion(
                str(raw_drawn["kind"]),
                None if raw_drawn["lower"] is None else float(raw_drawn["lower"]),
                None if raw_drawn["upper"] is None else float(raw_drawn["upper"]),
            )
        ),
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


def _surface_geometry(
    selection: SelectionState,
) -> tuple[str, tuple[float, float], tuple[float, float] | None]:
    """What a plot surface shows for this region: kind, x bounds, y bounds.

    ONE owner, because three callers need the same answer -- mounting a
    surface, mirroring onto the panel's OTHER surface, and taking the
    region away.  While each read ``selector_kind`` directly, a histogram's
    box came back as a full-height band on the Setting editor while the
    card still showed the rectangle, and the removal asked the card for a
    kind it did not have.
    """

    ranges = tuple(selection.ranges)
    x_bounds = (float(ranges[0].lower), float(ranges[0].upper))
    drawn = selection.drawn
    if drawn is not None and drawn.kind == "area":
        if drawn.lower is None or drawn.upper is None:
            raise ValueError("a drawn area needs the bound its ranges do not carry")
        return "area", x_bounds, (float(drawn.lower), float(drawn.upper))
    if selection.selector_kind == "area":
        if len(ranges) != 2:
            raise ValueError("an area panel selection requires x and y ranges")
        return "area", x_bounds, (float(ranges[1].lower), float(ranges[1].upper))
    if selection.selector_kind == "x_range":
        if len(ranges) != 1:
            raise ValueError("an x-range panel selection requires one range")
        return "x_range", x_bounds, None
    raise ValueError(
        f"unsupported panel selector kind {selection.selector_kind!r}"
    )


def panel_plot_selectors(
    selection: SelectionState | None,
    *,
    facet_index: int | None,
) -> tuple[PlotSelectorState, ...]:
    """Translate one saved panel target into zlc_plot's native selectors."""

    states: list[PlotSelectorState] = []
    if selection is not None:
        kind, x_bounds, y_bounds = _surface_geometry(selection)
        if kind == "area":
            assert y_bounds is not None
            states.append(PlotSelectorState(
                SelectorKind.AREA,
                RectangleRange(
                    NumericRange(*x_bounds),
                    NumericRange(*y_bounds),
                ),
                facet_index=facet_index,
            ))
        else:
            states.append(PlotSelectorState(
                SelectorKind.X_RANGE,
                NumericRange(*x_bounds),
                facet_index=facet_index,
            ))
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
    if plot_kind == "rolling":
        # Same rule that built it, so a rolling region is recognised on the
        # surface that drew it instead of being dropped one frame later.
        expected = _rolling_ranges(
            selection.selector_kind,
            dummy,
            dummy if selection.selector_kind == "area" else None,
        )
        expected_facets, expected_repeat = _subject_scope(subject)
        return bool(
            tuple(
                (item.domain, item.axis, item.coordinate_frame)
                for item in selection.ranges
            )
            == tuple(
                (item.domain, item.axis, item.coordinate_frame)
                for item in expected
            )
            and selection.repeat_index == expected_repeat
        )
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
    """Runtime outputs this accepted surface can actually derive.

    An axis bound slices and a value band invalidates what falls outside
    it, so an image, a curve and a histogram all derive from a region.  A
    ROLLING trace does not: its x says how many shots back a point is, and
    the derivation only ever sees the newest publication, so nothing it
    could publish would be the shots the operator boxed.  The region is
    still drawn and still kept -- Fit and restore read it -- it simply
    offers nothing downstream.  Runtime owns that rule; this asks it.
    """

    if subject is None:
        return ()
    if not isinstance(subject, SelectionSubject):
        raise TypeError("subject must be SelectionSubject or None")
    selector_kind = "area" if subject.y is not None else "x_range"
    return selection_output_catalog(
        selector_kind,
        _name_of(subject.plot_kind),
    )


def panel_selection_binds_a_revision(selection: SelectionState) -> bool:
    """Whether this region's meaning depends on the picture it was drawn on.

    Answered by what the ranges NAME, not by which kind drew them.  A bound
    on a Dataset axis is a bound on coordinates that can move between
    revisions, so the region has to be checked against the exact
    publication it was cut from.  A bound on the measured value or on the
    shot ordinal means the same thing on every revision: it restricts what
    counts, not where -- so demanding a matching revision of it only ever
    threw the region away.

    This is NOT "can anything be derived from it".  Every region cuts the
    signal it was drawn on: an axis bound slices, a value band invalidates
    what falls outside it, a shot window decides which publications answer.
    """

    if not isinstance(selection, SelectionState):
        raise TypeError("selection must be SelectionState")
    return any(item.domain not in _NON_AXIS_DOMAINS for item in selection.ranges)


def _apply_panel_selection(host: Any, selection: SelectionState) -> object:
    """Project one panel-owned canonical selection onto a plot surface."""

    kind, x_bounds, y_bounds = _surface_geometry(selection)
    if kind == "area":
        assert y_bounds is not None
        return host.set_area_selector(
            NumericRange(*x_bounds),
            NumericRange(*y_bounds),
            display=False,
            emit_change=False,
        )
    return host.set_x_selector(
        x_bounds[0],
        x_bounds[1],
        display=False,
        emit_change=False,
    )


def _remove_panel_selection(host: Any, selection: SelectionState) -> object:
    """Remove the same selector kind from the panel's other plot surface."""

    kind, _x_bounds, _y_bounds = _surface_geometry(selection)
    return host.remove_selector(
        SelectorKind(kind),
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

    def subscribe_display_observation(
        self,
        callback: Callable[[object], object],
    ) -> Callable[[], None]:
        """Report display-state changes through the SAME release list.

        This one used to be installed straight on the host, so its
        unsubscribe was dropped on the floor: every rebind of an unchanged
        Edit host added another identical display callback and released
        none, and each colour-limit drag afterwards enqueued one panel
        interaction per Refresh the operator had pressed.
        """

        if not callable(callback):
            raise TypeError("display observation callback must be callable")

        def _on_display(state: object) -> None:
            self._last_error = None
            self._deliver(callback, state)

        return self._install(self._host.subscribe_display, _on_display)

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
        if plot_kind == "rolling":
            # A rolling trace's own two bounds: the shot ordinal it counts
            # publications on, and the measured value.  The Dataset names
            # neither, which is why the subject reports no axes -- and why
            # this region derives nothing.  It is still exactly the region
            # the operator drew, so it is carried whole: both bounds, in
            # the order the surfaces apply them, so the mark the card shows
            # and the mark the Setting editor shows are the same mark.
            ranges = (
                _rolling_ranges(
                    selector_kind, selector.value.x, selector.value.y
                )
                if selector_kind == "area"
                else _rolling_ranges(selector_kind, selector.value)
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
        drawn: DrawnRegion | None = None
        if selector_kind == "area":
            # A drag always draws a box; what the box MEANS is the surface's
            # business.  Its x bound always says something -- the interval on
            # a named axis, or, when the x is the measured value as a
            # histogram's is, a band on that value.  Its y bound says
            # something only when it names an axis: a curve's y is the value
            # its x already carries and a histogram's y is a bin COUNT, which
            # restricts nothing upstream.  So one named y makes it an area,
            # and everything else is the x range -- the same rule
            # ``panel_selection_output_catalog`` offers outputs by.
            ranges = [
                _range(
                    subject.x,
                    selector.value.x,
                    "x",
                    subject.x_coordinate_frame,
                )
            ]
            if subject.y is not None:
                ranges.append(
                    _range(
                        subject.y,
                        selector.value.y,
                        "y",
                        subject.y_coordinate_frame,
                    )
                )
            else:
                # The DERIVATION reads an x range -- that is what
                # ``selector_kind`` answers.  What the hand DREW is still a
                # box, and that is what both of this panel's surfaces must
                # show and must be asked to remove.  Rewriting the one word
                # to mean the other is what put a full-height band in the
                # Setting editor while the card kept the rectangle, and made
                # "clear the region" ask a surface to drop a kind it never
                # had.
                drawn = DrawnRegion(
                    "area",
                    float(selector.value.y.low),
                    float(selector.value.y.high),
                )
                selector_kind = "x_range"
            ranges = tuple(ranges)
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
            drawn=drawn,
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
