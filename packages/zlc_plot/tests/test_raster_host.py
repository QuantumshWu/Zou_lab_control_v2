from __future__ import annotations

from threading import Event
from concurrent.futures import Future

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from test_facet_live_fit import _facet_snapshot, _spec as facet_spec
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot
from zlc_plot.fit import FacetFitBatchResult
from zlc_plot.raster import RasterPlotHost


def _snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
        generation="raster-host-test",
    )
    return DatasetSnapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=0)


def test_host_coalesces_same_key_and_front_sequences_advance() -> None:
    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    gate = Event()
    started = Event()
    try:
        first = host.wait_for_front(timeout=10)
        assert first.identity.sequence == 1

        def block() -> None:
            started.set()
            gate.wait(5.0)

        blocker = host.dispatch_control(block)
        assert started.wait(2.0)
        superseded = host.set_parameter("title", "first")
        newest = host.set_parameter("title", "second")
        assert superseded.cancelled()
        gate.set()
        blocker.result(timeout=10)
        operation = newest.result(timeout=10)
        assert operation.front.identity.sequence > first.identity.sequence
        assert host.front is not None
        assert host.front.identity.sequence == operation.front.identity.sequence
    finally:
        gate.set()
        host.close(timeout=10)


def test_close_cancels_queued_tasks() -> None:
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
        pending = host.set_parameter("title", "queued")
        host.close(timeout=0.05)
        assert pending.cancelled()
        gate.set()
        host.close(timeout=10)
    finally:
        gate.set()
        host.close(timeout=10)


def test_press_relocates_to_latest_front_after_live_revision() -> None:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
        generation="raster-press-race",
    )
    first_data = DatasetSnapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=0)
    next_data = DatasetSnapshot(schema, np.array([[2.0, 3.0, 4.0]]), revision=1)
    host = RasterPlotHost.from_plot(first_data, CurvePlot(AxisRef.point("x")))
    try:
        stale = host.wait_for_front(timeout=10)
        host.update_data(next_data).result(timeout=10)
        latest = host.front
        assert latest is not None
        assert latest.identity.sequence > stale.identity.sequence

        operation = host._pointer_event(
            "press",
            0.45,
            0.45,
            button=1,
            identity=stale.identity,
            axes=stale.interaction.axes[0],
            interaction=stale.interaction,
        ).result(timeout=10)

        assert operation.value.candidate is not None
        assert operation.value.role == "main"
    finally:
        host.close(timeout=10)


def test_host_facet_live_fit_promotes_one_batch_front_and_future() -> None:
    """Facet analysis must publish through the same source-revision contract."""

    spec = facet_spec()
    assert isinstance(spec, FacetGridPlot)
    host = RasterPlotHost.from_plot(_facet_snapshot(), spec)
    try:
        first = host.wait_for_front(timeout=10)
        operation = host.fit("gaussian_offset", live=True).result(timeout=30)
        assert isinstance(operation.value, FacetFitBatchResult)
        assert operation.value.source_revision == operation.front.identity.data_revision
        assert operation.front.identity.sequence > first.identity.sequence
        assert host.front is operation.front
    finally:
        host.close(timeout=10)


def test_host_fit_callback_exceptions_complete_the_future(monkeypatch) -> None:
    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        host.wait_for_front(timeout=10)

        def broken_fit(*_args, **_kwargs):
            result: Future[object] = Future()
            result.set_result(object())
            return result

        monkeypatch.setattr(host._worker_adapter, "fit", broken_fit)
        completion = host.fit("gaussian_offset", live=False)
        with pytest.raises(AttributeError):
            completion.result(timeout=10)
    finally:
        host.close(timeout=10)


def test_one_semantic_edit_is_composed_where_the_spec_and_schema_live() -> None:
    """A caller supplies only what the operator changed.

    Composing the candidate needs the current spec and the schema of what is
    being drawn.  Both belong to the session, so asking an embedder for them is
    asking it to keep a copy of state it does not own -- and every embedder that
    did kept a copy that could drift from the session actually rendering.
    """

    from zlc_plot import PlotKind

    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        host.wait_for_front(timeout=10)

        described = host.apply_semantic("kind", PlotKind.HISTOGRAM).result(timeout=10)

        assert described.value is not None
        # It really replaced the spec: the next edit composes against the new
        # one, not against the curve it started as.
        again = host.describe_semantics().result(timeout=10).value
        assert again.kind is PlotKind.HISTOGRAM
    finally:
        host.close(timeout=10)


def test_a_semantic_edit_that_changes_nothing_does_not_rebuild() -> None:
    """Re-selecting the kind a plot already is must not restart the render."""

    from zlc_plot import PlotKind

    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        first = host.wait_for_front(timeout=10)

        described = host.apply_semantic("kind", PlotKind.CURVE).result(timeout=10)

        assert described.value is not None
        # The figure was not thrown away and rebuilt: a replace bumps the
        # display and layout revisions, and neither moved.
        assert host.front.identity.display_revision == first.identity.display_revision
        assert host.front.identity.layout_revision == first.identity.layout_revision
    finally:
        host.close(timeout=10)
