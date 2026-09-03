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
from zlc_plot.selectors import NumericRange, RectangleRange


def _snapshot() -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
    )
    return make_snapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=0)


def test_a_later_configure_carries_the_queued_one_forward() -> None:
    """The title edit and the mirrored viewport both land."""

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

        first = host.configure(parameters={"title": "from Setting"})
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
    finally:
        gate.set()
        host.close(timeout=10)
