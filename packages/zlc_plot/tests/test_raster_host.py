from __future__ import annotations

from threading import Event
from concurrent.futures import Future
from pathlib import Path
import time

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from test_facet_live_fit import _facet_snapshot, _spec as facet_spec
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    SelectorKind,
    parameter_controls,
)
from zlc_plot.fit import (
    FacetFitBatchResult,
    FitCancelled,
    FitDeadlineExceeded,
    FitEngine,
)
from zlc_plot.raster import RasterBuffer, RasterPlotHost
from zlc_plot.rendering import MatplotlibRenderer
from zlc_plot.ui import ControlKind


def _snapshot() -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
        generation="raster-host-test",
    )
    return DatasetSnapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=0)


def _site_distribution_snapshot() -> DatasetSnapshot:
    samples = np.linspace(-3.0, 3.0, 80)
    values = np.column_stack(
        (
            np.where(samples < 0.0, samples - 2.0, samples + 2.0),
            np.where(samples < 0.0, samples - 1.0, samples + 3.0),
        )
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=values.shape[0]),
        PointTable.from_columns({"site": (0.0, 1.0)}),
        dtype=np.float64,
        generation="raster-site-distribution",
    )
    return DatasetSnapshot(schema, values, revision=0)


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


def test_live_fit_pairs_every_front_and_coalesces_queued_frames(
    monkeypatch,
) -> None:
    """A fit-armed frame is data@N with fit@N, born complete.

    The publish gate: while revision 1's solve runs, no new front appears —
    a pair never half-shows.  No preemption: newer revisions arriving during
    the solve replace only the QUEUED frame (revision 2's future cancels,
    its solve never starts), while the in-flight pair runs to completion and
    publishes.  The next pair then jumps to the newest revision, so under
    solve > period the pair rate drops and shots skip, but pairs never
    split.
    """

    x = np.linspace(-4.0, 4.0, 81)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x}),
        dtype=np.float64,
        generation="raster-live-fit-pairs",
    )

    def snapshot(revision: int, center: float) -> DatasetSnapshot:
        values = 2.0 * np.exp(-0.5 * ((x - center) / 0.9) ** 2) + 0.1
        return DatasetSnapshot(schema, values.reshape(1, -1), revision=revision)

    engine = FitEngine()
    original_fit = engine.fit
    solved_revisions: list[int] = []
    first_started = Event()
    release_first = Event()

    def controlled_fit(model, coordinates, observations=None, **kwargs):
        revision = int(kwargs["data_revision"])
        solved_revisions.append(revision)
        if revision == 1:
            first_started.set()
            if not release_first.wait(5.0):
                raise AssertionError("revision 1 fit was never released")
        return original_fit(model, coordinates, observations, **kwargs)

    monkeypatch.setattr(engine, "fit", controlled_fit)
    host = RasterPlotHost.from_plot(
        snapshot(0, 0.0),
        CurvePlot(AxisRef.point("x")),
        fit_engine=engine,
    )
    accepted: list[int] = []
    release_subscription = None
    try:
        host.wait_for_front(timeout=10)
        host.fit("gaussian_offset", live=True).result(timeout=30)
        release_subscription = host.subscribe_fit(
            lambda event: accepted.append(int(event.result.source_revision))
        ).result(timeout=10).value

        first = host.update_data(snapshot(1, 0.1))
        assert first_started.wait(2.0)
        # The publish gate: the pair is incomplete, so the front holds.
        time.sleep(0.05)
        assert host.front is not None
        assert host.front.identity.data_revision == 0

        second = host.update_data(snapshot(2, 0.2))
        third = host.update_data(snapshot(3, 0.3))
        # Latest-only queueing: revision 2 was replaced before it started.
        assert second.cancelled()

        release_first.set()
        # No preemption: the in-flight pair completed and published even
        # though newer data had arrived.
        first.result(timeout=10)
        third.result(timeout=10)
        assert host.front is not None
        assert host.front.identity.data_revision == 3
        # Every published front carried its own shot's overlay; revision 2
        # was skipped entirely — never solved, never shown.
        assert accepted == [1, 3]
        assert 2 not in solved_revisions
    finally:
        release_first.set()
        if release_subscription is not None:
            release_subscription().result(timeout=10)
        host.close(timeout=10)


def test_a_regular_bimodal_fit_does_not_create_a_threshold_classifier() -> None:
    host = RasterPlotHost.from_plot(
        _site_distribution_snapshot(),
        FacetGridPlot(AxisRef.point("site"), HistogramPlot()),
    )
    try:
        host.wait_for_front(timeout=10)
        host.fit("bimodal_gaussian", live=False).result(timeout=30)

        with pytest.raises(KeyError):
            host.selector_state(SelectorKind.THRESHOLD).result(timeout=10)
    finally:
        host.close(timeout=10)


def test_threshold_classifier_is_independent_and_covers_every_facet() -> None:
    """The Distribution switch owns its fit, threshold, and compact cell text."""

    host = RasterPlotHost.from_plot(
        _site_distribution_snapshot(),
        FacetGridPlot(AxisRef.point("site"), HistogramPlot()),
    )
    try:
        initial = host.wait_for_front(timeout=10)
        configured = host.configure(
            parameters={"threshold_classifier": True},
        ).result(timeout=30)
        assert configured.value.display_state.values["threshold_classifier"] is True
        classifier_control = next(
            control
            for control in parameter_controls(
                configured.value.parameter_schema,
                configured.value.display_state.values,
            )
            if control.name == "threshold_classifier"
        )
        assert classifier_control.kind is ControlKind.BOOLEAN
        assert configured.front.identity.sequence > initial.identity.sequence
        assert configured.front.buffer.pixels != initial.buffer.pixels

        status = host.dispatch_control(
            lambda: host._worker_adapter._session().fit_status
        ).result(timeout=10)
        assert status.value is None

        details, facet_font = host.dispatch_control(
            lambda: (
                tuple(
                    (
                        len(artists[0]),
                        artists[2].get_text(),
                        artists[2].get_fontsize(),
                    )
                    for artists in host._worker_adapter._session()._renderer._classifier_artists.values()
                ),
                host._worker_adapter._session()._defaults.style.fonts.facet_fit_annotation_pt,
            )
        ).result(timeout=10).value
        assert len(details) == 2
        assert all(line_count == 3 for line_count, _label, _font in details)
        assert all("Threshold" in label for _line_count, label, _font in details)
        assert all("L/R" in label for _line_count, label, _font in details)
        assert all("Fidelity" in label for _line_count, label, _font in details)
        assert all(font == facet_font for _line_count, _label, font in details)
        for _line_count, label, _font in details:
            left_text, right_text = label.splitlines()[1].removeprefix("L/R ").split("/")
            assert float(left_text.removesuffix("%")) + float(
                right_text.removesuffix("%")
            ) == pytest.approx(100.0)

        first_cell = next(
            axes
            for axes in configured.front.interaction.axes
            if axes.role == "facet_cell" and axes.cell_index == 0
        )
        host.focus_facet(configured.front.identity, first_cell).result(timeout=10)
        optimum = host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value
        moved = host.set_threshold_selector(
            float(optimum.value) + 0.25,
            display=False,
        ).result(timeout=10).value
        assert moved.value != optimum.value
        moved_label = host.dispatch_control(
            lambda: host._worker_adapter._session()._renderer._classifier_labels[0]
        ).result(timeout=10).value
        assert moved_label != details[0][1]
        assert "Fidelity" in moved_label
        moved_left, moved_right = (
            moved_label.splitlines()[1].removeprefix("L/R ").split("/")
        )
        assert float(moved_left.removesuffix("%")) + float(
            moved_right.removesuffix("%")
        ) == pytest.approx(100.0)

        host.fit("bimodal_gaussian", live=False).result(timeout=30)
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == moved.value
        host.clear_fit().result(timeout=10)
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == moved.value

        configured = host.configure(
            parameters={"threshold_classifier": True},
            classifier_thresholds=(-0.25, 0.75),
        ).result(timeout=10)
        assert configured.value.display_state.values["threshold_classifier"] is True
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == -0.25
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


def test_one_complete_configuration_is_differenced_by_the_plot_owner() -> None:
    """An embedder states the desired plot; it does not choose the render path."""

    from zlc_plot import PlotKind

    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        first = host.wait_for_front(timeout=10)
        identity = host.host_id
        initial = host.describe_display().result(timeout=10)
        assert initial.value.display_state.values["title"] is None
        assert initial.value.display_state.values["x_label"] is None

        configured = host.configure(
            semantic={"kind": PlotKind.CURVE},
            parameters={"title": "Configured once", "show_grid": True},
            size="2x2",
        ).result(timeout=10)

        assert host.host_id == identity
        assert configured.value.display_state.values["title"] == "Configured once"
        assert configured.value.display_state.values["show_grid"] is True
        assert configured.value.size == "2x2"
        assert configured.front.identity.sequence == first.identity.sequence + 1
        assert configured.front.identity.display_revision > first.identity.display_revision
        assert configured.front.identity.layout_revision == first.identity.layout_revision

        reshaped = host.configure(
            semantic={"kind": PlotKind.HISTOGRAM},
            parameters={"title": "Distribution", "bin_count": 8},
            size="2x4",
        ).result(timeout=10)
        assert reshaped.value.kind is PlotKind.HISTOGRAM
        assert reshaped.value.display_state.values["title"] == "Distribution"
        assert reshaped.value.display_state.values["bin_count"] == 8
        assert reshaped.value.size == "2x4"
        assert reshaped.front.identity.sequence == configured.front.identity.sequence + 1
        assert reshaped.front.identity.layout_revision > configured.front.identity.layout_revision

        unchanged = host.configure(
            semantic={"kind": PlotKind.HISTOGRAM},
            parameters={"title": "Distribution", "bin_count": 8},
            size="2x4",
        ).result(timeout=10)
        assert unchanged.value.display_state.revision == reshaped.value.display_state.revision
        assert unchanged.value.size == reshaped.value.size

        automatic = host.configure(
            parameters={"title": None},
        ).result(timeout=10)
        assert automatic.value.display_state.values["title"] is None
        assert automatic.front.identity.sequence == unchanged.front.identity.sequence + 1
    finally:
        host.close(timeout=10)


def test_a_host_that_could_not_start_says_why_not_that_it_is_closing() -> None:
    """The refusal must carry the reason, not the symptom.

    A startup failure sets ``_closing`` and records itself in
    ``_startup_error``.  Every refusal that read only ``_closing`` answered
    "raster plot host is closing" -- so a panel asked to draw something the
    plot kind cannot accept reported a host that was shutting down, and the
    real sentence was recorded and never read by anyone.  It cost a whole
    investigation to find a message the program already had.
    """

    # A calibration artifact is not a snapshot; CurvePlot refuses it, which is
    # correct -- what matters is whether the caller is told THAT.
    host = RasterPlotHost.from_plot(object(), CurvePlot(AxisRef.point("x")))
    try:
        with pytest.raises(RuntimeError) as raised:
            host.wait_for_front(timeout=10)
        assert "failed to start" in str(raised.value), str(raised.value)
        assert "OwnedSnapshot" in str(raised.value), (
            "the refusal must name the real reason: " + str(raised.value)
        )
        assert raised.value.__cause__ is not None, "and keep the original as its cause"

        # Every later refusal says the same thing, not "closing".
        with pytest.raises(RuntimeError) as again:
            host.subscribe_front(lambda front: None)
        assert "failed to start" in str(again.value), str(again.value)
    finally:
        host.close()


def test_host_save_preserves_existing_file_when_renderer_fails_after_partial_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed export must never expose the renderer's partial output."""

    target = tmp_path / "panel.png"
    original = b"existing-production-image"
    target.write_bytes(original)
    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        host.wait_for_front(timeout=10)
        host.save(target).result(timeout=10)
        assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        target.write_bytes(original)
        observed: list[tuple[object, dict[str, object]]] = []

        def fail_after_partial_write(
            _renderer: object,
            output: object,
            **_options: object,
        ) -> None:
            observed.append((output, dict(_options)))
            if hasattr(output, "write"):
                output.write(b"partial-render")  # type: ignore[attr-defined]
            else:
                Path(output).write_bytes(b"partial-render")  # type: ignore[arg-type]
            raise RuntimeError("renderer failed after a partial write")

        monkeypatch.setattr(
            MatplotlibRenderer,
            "save",
            fail_after_partial_write,
        )

        with pytest.raises(RuntimeError, match="partial write"):
            host.save(target).result(timeout=10)

        assert target.read_bytes() == original
        assert tuple(tmp_path.iterdir()) == (target,)
        assert len(observed) == 1
        output, options = observed[0]
        assert not isinstance(output, (str, Path))
        assert options["format"] == "png"
    finally:
        host.close(timeout=10)


def test_raster_buffer_save_preserves_existing_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical-pixel export uses the same durable replacement owner."""

    import zlc_durable.durability as durability

    target = tmp_path / "front.png"
    original = b"existing-production-image"
    target.write_bytes(original)
    buffer = RasterBuffer(1, 1, b"\x10\x20\x30\xff")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(durability.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        buffer.save(target)

    assert target.read_bytes() == original
    assert tuple(tmp_path.iterdir()) == (target,)
