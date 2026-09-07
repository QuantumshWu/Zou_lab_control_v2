"""A coalesced configure supersedes what it replaces, and drops nothing.

Every configure shares one coalesce key, so a queued target is REPLACED by
the next -- and the console sends two inside a single busy worker window:
the Setting form's semantic/parameters/fit, and a sibling gesture's
viewport or facet_focus.  Whichever arrived first had its fields silently
discarded, with nothing told: an edit in Setting simply did not take, or a
mirrored viewport did not follow.
"""

from __future__ import annotations

from threading import Event

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, REPEAT
from zlc_plot import AxisRef, CurvePlot
from zlc_plot.raster import RasterPlotHost
from zlc_plot.selectors import CrosshairPoint, NumericRange, RectangleRange, SelectorKind, SelectorState


def _snapshot() -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
    )
    return make_snapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=0)


def test_a_later_configure_carries_the_queued_one_forward() -> None:
    """Queued fields and kind-local selector edits land as one description."""

    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    gate = Event()
    started = Event()
    try:
        host.wait_for_front(timeout=10)

        def block() -> None:
            started.set()
            gate.wait(5.0)

        host.dispatch_control(block)
        assert started.wait(2.0)

        crosshair = SelectorState(SelectorKind.CROSSHAIR, CrosshairPoint(1.0, 2.0))
        newer_crosshair = SelectorState(SelectorKind.CROSSHAIR, CrosshairPoint(1.5, 2.5))
        x_range = SelectorState(SelectorKind.X_RANGE, NumericRange(0.5, 1.5))
        newer_range = SelectorState(SelectorKind.X_RANGE, NumericRange(0.25, 1.75))
        area = SelectorState(SelectorKind.AREA,
                             RectangleRange(NumericRange(0.5, 1.5), NumericRange(1.5, 2.5)))
        newer_area = SelectorState(SelectorKind.AREA,
                                   RectangleRange(NumericRange(0.25, 1.75), NumericRange(1.25, 2.75)))
        # This setter has not run when the patch is submitted. The patch
        # must read execution-time state, not replay an earlier empty tuple.
        host.set_crosshair_selector(1.0, 2.0, display=False)
        first = host.configure(parameters={"title": "from Setting"},
                               selector_updates={SelectorKind.X_RANGE: x_range})
        viewport = RectangleRange(NumericRange(0.5, 1.5), NumericRange(1.5, 2.5))
        second = host.configure(viewport=viewport)
        gate.set()
        second.result(timeout=10)

        # The first was coalesced away -- that is what the key is for --
        # but its target went with the one that replaced it.
        assert first.cancelled() or first.done()
        description = host.describe_display().result(timeout=10).value
        assert description.display_state["title"] == "from Setting"
        assert description.viewport == viewport
        assert {state.kind: state.value for state in description.selectors} == {
            SelectorKind.CROSSHAIR: crosshair.value, SelectorKind.X_RANGE: x_range.value}

        for targets, expected in (
            (({"selector_updates": {SelectorKind.X_RANGE: x_range}},
              {"selector_updates": {SelectorKind.CROSSHAIR: newer_crosshair}},
              {"selector_updates": {SelectorKind.X_RANGE: newer_range}}),
             {SelectorKind.X_RANGE: newer_range.value, SelectorKind.CROSSHAIR: newer_crosshair.value}),
            (({"selector_updates": {SelectorKind.X_RANGE: x_range}},
              {"selectors": (crosshair,), "selector_updates": {SelectorKind.AREA: area}},
              {"selector_updates": {SelectorKind.CROSSHAIR: None, SelectorKind.AREA: newer_area}}),
             {SelectorKind.AREA: newer_area.value}),
        ):
            gate.clear()
            started.clear()
            host.dispatch_control(block)
            assert started.wait(2.0)
            operations = [host.configure(**target) for target in targets]
            gate.set()
            answer = operations[-1].result(timeout=10)
            assert {state.kind: state.value for state in answer.value.selectors} == expected
            assert answer.front.interaction.selectors == answer.value.selectors
            assert answer.front.identity.data_revision == 0
            assert answer.value.display_state["title"] == "from Setting"

        before = host.front
        noop = host.configure(selector_updates={SelectorKind.X_RANGE: None}).result(timeout=10)
        assert noop.front is before, "removing an absent kind is a configure no-op"
        assert {state.kind: state.value for state in noop.value.selectors} == {SelectorKind.AREA: newer_area.value}
        for patch, message in (
            ({"area": None}, "keys must be SelectorKind"),
            ({SelectorKind.X_RANGE: crosshair}, "kind does not match"),
            ({SelectorKind.AREA: "not a state"}, "must be SelectorState or None"),
        ):
            with pytest.raises((TypeError, ValueError), match=message):
                host.configure(parameters={"title": "must not land"}, selector_updates=patch).result(timeout=10)
            assert host.front is before
            assert host.describe_display().result(timeout=10).value.display_state["title"] == "from Setting"
    finally:
        gate.set()
        host.close(timeout=10)
