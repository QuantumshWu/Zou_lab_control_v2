from __future__ import annotations

from concurrent.futures import Future
from threading import Event
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
    PlotSession,
    Reduction,
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
from zlc_plot.selectors import NumericRange, SelectorState
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


def _fit_curve_series(generation: str, *, offset: float = 0.0):
    x = np.linspace(-4.0, 4.0, 81)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x}),
        dtype=np.float64,
        generation=generation,
    )

    def snapshot(revision: int, center: float | None = None) -> DatasetSnapshot:
        selected = revision * 0.05 if center is None else center
        values = 2.0 * np.exp(-0.5 * ((x - selected) / 0.9) ** 2) + offset
        return DatasetSnapshot(schema, values.reshape(1, -1), revision=revision)

    return snapshot


@pytest.fixture
def blocked_fit_host(monkeypatch):
    owned = []

    def build(
        generation: str,
        *,
        offset: float = 0.0,
        fail_revision=None,
        block_revision: int = 1,
        cooperative: bool = True,
    ):
        snapshots = _fit_curve_series(generation, offset=offset)
        engine = FitEngine()
        solve = engine.fit
        started, release = Event(), Event()
        solved = []

        def controlled(model, coordinates, observations=None, **kwargs):
            revision = int(kwargs["data_revision"])
            solved.append(revision)
            if revision == block_revision:
                started.set()
                cancelled = kwargs.get("cancelled")
                deadline = time.monotonic() + 5.0
                while not release.wait(0.005):
                    if cooperative and callable(cancelled) and cancelled():
                        raise FitCancelled("forced cooperative cancellation")
                    assert time.monotonic() < deadline
            if revision == fail_revision:
                raise RuntimeError(f"forced revision-{revision} failure")
            return solve(model, coordinates, observations, **kwargs)

        monkeypatch.setattr(engine, "fit", controlled)
        host = RasterPlotHost.from_plot(
            snapshots(0),
            CurvePlot(AxisRef.point("x")),
            fit_engine=engine,
        )
        owned.append((host, release))
        return snapshots, host, started, release, solved

    yield build
    for host, release in reversed(owned):
        release.set()
        host.close(timeout=10)


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


def test_initial_front_precedes_a_startup_noop_configuration() -> None:
    snapshot = _snapshot()
    factory_started = Event()
    release_factory = Event()

    def make_session() -> PlotSession:
        factory_started.set()
        release_factory.wait(5.0)
        return PlotSession(
            snapshot,
            CurvePlot(AxisRef.point("x")),
            size="2x2",
        )

    host = RasterPlotHost(make_session)
    try:
        assert factory_started.wait(2.0)
        configured = host.configure(parameters={}, size="2x2")
        release_factory.set()

        first = host.wait_for_front(timeout=10)
        operation = configured.result(timeout=10)

        assert first.identity.sequence == 1
        assert operation.front.identity == first.identity
        assert host.front is first
    finally:
        release_factory.set()
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


def test_live_fit_keeps_only_the_latest_successor_while_active(
    blocked_fit_host,
) -> None:
    """Direct callers get capacity-one success and close semantics."""

    snapshot, normal, started, release, solved = blocked_fit_host(
        "raster-live-fit-latest-success", offset=0.1
    )
    events: list[object] = []
    try:
        normal.wait_for_front(timeout=10)
        normal.fit("gaussian_offset", live=True).result(timeout=30)
        normal.subscribe_fit(events.append).result(timeout=10)
        active = normal.update_data(snapshot(1, 0.1))
        assert started.wait(2.0)
        superseded = normal.update_data(snapshot(2, 0.2))
        latest = normal.update_data(snapshot(3, 0.3))
        assert superseded.cancelled()
        release.set()
        active.result(timeout=10)
        latest.result(timeout=10)
        assert [event.result.source_revision for event in events] == [1, 3]
        assert solved[-2:] == [1, 3]
        assert normal.front is not None
        assert normal.front.identity.data_revision == 3
    finally:
        release.set()
        normal.close(timeout=10)

    snapshot, host, first_started, release_first, solved_revisions = blocked_fit_host(
        "raster-live-fit-pairs", offset=0.1, cooperative=False
    )
    fit_events: list[object] = []
    try:
        host.wait_for_front(timeout=10)
        host.fit("gaussian_offset", live=True).result(timeout=30)
        host.subscribe_fit(fit_events.append).result(timeout=10)

        first = host.update_data(snapshot(1, 0.1))
        assert first_started.wait(2.0)
        # The publish gate: the pair is incomplete, so the front holds.
        time.sleep(0.05)
        assert host.front is not None
        assert host.front.identity.data_revision == 0

        superseded = host.update_data(snapshot(2, 0.2))
        latest = host.update_data(snapshot(3, 0.3))
        assert superseded.cancelled()
        assert not latest.done()

        close_started = time.monotonic()
        assert not host.close(timeout=0.05)
        assert time.monotonic() - close_started < 0.5
        assert first.cancelled()
        assert latest.cancelled()
        release_first.set()
        assert host.close(timeout=2.0)
        assert fit_events == []
        assert solved_revisions == [0, 1]
    finally:
        release_first.set()
        host.close(timeout=10)


def test_active_fit_times_out_without_a_successor_and_recovers(
    monkeypatch,
    blocked_fit_host,
) -> None:
    """Only solve has a deadline; slow data prepare completes normally."""

    snapshot, host, first_started, release_first, solved = blocked_fit_host(
        "active-fit-deadline", block_revision=2
    )
    release_subscription = None
    accepted: list[tuple[int, bool]] = []
    fit_events: list = []

    def observe_fit(event) -> None:
        fit_events.append(event)
        accepted.append(
            (int(event.result.source_revision), bool(event.result.success))
        )

    try:
        host.wait_for_front(timeout=10)
        session = host._session
        assert session is not None
        original_prepare = PlotSession.prepare_live_frame
        prepared = original_prepare(session, snapshot(1)).result(timeout=10)
        slow_prepare: Future = Future()
        monkeypatch.setattr(
            PlotSession,
            "prepare_live_frame",
            lambda _session, _data, **_kwargs: slow_prepare,
        )
        data_only = host.update_data(snapshot(1))
        time.sleep(1.1)
        assert not data_only.done()
        slow_prepare.set_result(prepared)
        data_only.result(timeout=10)
        assert host.front is not None
        assert host.front.identity.data_revision == 1
        monkeypatch.setattr(PlotSession, "prepare_live_frame", original_prepare)

        host.fit("gaussian_offset", live=True).result(timeout=30)
        release_subscription = host.subscribe_fit(observe_fit).result(timeout=10).value

        started_at = time.monotonic()
        first = host.update_data(snapshot(2))
        assert first_started.wait(2.0)
        with pytest.raises(RuntimeError, match="active deadline"):
            first.result(timeout=2.0)
        elapsed = time.monotonic() - started_at
        assert 0.9 <= elapsed < 1.8, elapsed
        assert not release_first.is_set()
        latest = host.update_data(snapshot(3))
        latest.result(timeout=10)
        assert accepted == [
            (2, False),
            (3, True),
        ]
        assert np.isnan(fit_events[0].result.parameter_values).all()
        assert host.front is not None
        assert host.front.identity.data_revision == 3
        assert solved[-2:] == [2, 3]

        release_subscription().result(timeout=10)
        release_subscription = None
        close_started = time.monotonic()
        assert host.close(timeout=2.0)
        assert time.monotonic() - close_started < 2.0
    finally:
        release_first.set()
        if release_subscription is not None:
            release_subscription().result(timeout=10)


def test_solver_failure_is_loud_for_that_revision_and_the_tail_continues(
    blocked_fit_host,
) -> None:
    """A failed pair leaves the old front, reports its gap, then recovers."""

    snapshot, host, first_started, release_first, solved = blocked_fit_host(
        "exact-fit-solver-failure", fail_revision=2
    )
    release_subscription = None
    accepted: list[tuple[int, bool]] = []
    fit_events: list = []

    def observe_fit(event) -> None:
        fit_events.append(event)
        accepted.append(
            (int(event.result.source_revision), bool(event.result.success))
        )

    try:
        host.wait_for_front(timeout=10)
        host.fit("gaussian_offset", live=True).result(timeout=30)
        release_subscription = host.subscribe_fit(observe_fit).result(timeout=10).value

        first = host.update_data(snapshot(1))
        assert first_started.wait(2.0)
        release_first.set()
        first.result(timeout=10)

        second = host.update_data(snapshot(2))
        with pytest.raises(RuntimeError, match="forced revision-2 failure"):
            second.result(timeout=10)
        third = host.update_data(snapshot(3))
        third.result(timeout=10)
        host.update_data(snapshot(4)).result(timeout=10)

        assert host.front is not None
        assert host.front.identity.data_revision == 4
        assert accepted == [(1, True), (2, False), (3, True), (4, True)]
        failed = fit_events[1].result
        assert np.isnan(failed.parameter_values).all()
        assert np.isnan(failed.standard_errors).all()
        assert solved[-4:] == [1, 2, 3, 4]
    finally:
        release_first.set()
        if release_subscription is not None:
            release_subscription().result(timeout=10)


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

        host.focus_facet(0).result(timeout=10)
        optimum = host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value
        moved = host.set_threshold_selector(
            float(optimum.value) + 0.25,
            display=False,
        ).result(timeout=10)
        assert moved.value.value != optimum.value
        assert moved.front.identity.sequence > configured.front.identity.sequence

        host.fit("bimodal_gaussian", live=False).result(timeout=30)
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == moved.value.value
        host.clear_fit().result(timeout=10)
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == moved.value.value

        configured = host.configure(
            parameters={"threshold_classifier": True},
            classifier_thresholds=(
                {
                    "value": -0.25,
                    "scope": (
                        {
                            "domain": "point_coordinate",
                            "axis_id": "site",
                            "coordinate": 0,
                        },
                    ),
                    "repeat_index": None,
                },
                {
                    "value": 0.75,
                    "scope": (
                        {
                            "domain": "point_coordinate",
                            "axis_id": "site",
                            "coordinate": 1,
                        },
                    ),
                    "repeat_index": None,
                },
            ),
        ).result(timeout=10)
        assert configured.value.display_state.values["threshold_classifier"] is True
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == -0.25
    finally:
        host.close(timeout=10)


def test_one_complete_configuration_is_differenced_by_the_plot_owner() -> None:
    """An embedder states the desired plot; it does not choose the render path."""

    from zlc_plot import PlotKind

    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        first = host.wait_for_front(timeout=10)
        configured = host.configure(
            semantic={"kind": PlotKind.CURVE},
            parameters={"title": "Configured once", "show_grid": True},
            size="2x2",
        ).result(timeout=10)

        assert configured.value.display_state.values["title"] == "Configured once"
        assert configured.front.identity.sequence == first.identity.sequence + 1
        assert configured.front.identity.display_revision > first.identity.display_revision

        reshaped = host.configure(
            semantic={"kind": PlotKind.HISTOGRAM},
            parameters={"title": "Distribution", "bin_count": 8},
            size="2x4",
        ).result(timeout=10)
        assert reshaped.value.kind is PlotKind.HISTOGRAM
        assert reshaped.value.display_state.values["title"] == "Distribution"
        assert reshaped.value.size == "2x4"
        assert reshaped.front.identity.sequence == configured.front.identity.sequence + 1

        unchanged = host.configure(
            semantic={"kind": PlotKind.HISTOGRAM},
            parameters={"title": "Distribution", "bin_count": 8},
            size="2x4",
        ).result(timeout=10)
        assert unchanged.value.display_state.revision == reshaped.value.display_state.revision
        assert unchanged.front is reshaped.front

        automatic = host.configure(
            parameters={"title": None},
        ).result(timeout=10)
        assert automatic.value.display_state.values["title"] is None
        assert automatic.front.identity.sequence == unchanged.front.identity.sequence + 1
    finally:
        host.close(timeout=10)


def test_complete_configuration_fits_clears_and_noops_as_one_front(
    monkeypatch,
) -> None:
    """One desired target has one solve/front; replaying it has neither."""

    snapshots = _fit_curve_series("atomic-configuration", offset=0.1)
    snapshot = snapshots(0, 0.3)
    engine = FitEngine()
    solve_count = 0
    solve = engine.fit

    def counted_solve(*args, **kwargs):
        nonlocal solve_count
        solve_count += 1
        return solve(*args, **kwargs)

    monkeypatch.setattr(engine, "fit", counted_solve)
    host = RasterPlotHost.from_plot(
        snapshot,
        CurvePlot(AxisRef.point("x")),
        fit_engine=engine,
    )
    try:
        initial = host.wait_for_front(timeout=10)
        fit_target = {
            "model": "gaussian_offset",
            "initial": {
                "amplitude": 2.0,
                "center": 0.3,
                "sigma": 0.9,
                "offset": 0.1,
            },
            "bounds": {"sigma": (0.1, 3.0)},
        }
        desired = dict(
            parameters={"title": "Atomic"},
            selectors=(),
            viewport=None,
            facet_focus=None,
        )
        fitted = host.configure(fit=fit_target, **desired).result(timeout=30)
        assert fitted.front.identity.sequence == initial.identity.sequence + 1
        assert solve_count == 1

        unchanged = host.configure(fit=fit_target, **desired).result(timeout=30)
        assert unchanged.front is fitted.front
        assert solve_count == 1

        cleared = host.configure(fit={}, **desired).result(timeout=30)
        assert cleared.front.identity.sequence == fitted.front.identity.sequence + 1

        clear_noop = host.configure(fit={}, **desired).result(timeout=30)
        assert clear_noop.front is cleared.front
    finally:
        host.close(timeout=10)


@pytest.mark.parametrize(
    ("invalid_target", "error", "match", "solver_failure"),
    (
        ({"facet_focus": 0, "fit": {}}, TypeError, "facet", False),
        (
            {
                "selectors": (
                    SelectorState(SelectorKind.X_RANGE, NumericRange(1.0, 1.0)),
                ),
                "fit": {},
            },
            ValueError,
            "non-degenerate",
            False,
        ),
        (
            {"fit": {"model": "gaussian_offset", "obsolete": True}},
            TypeError,
            "unknown fit target fields",
            False,
        ),
        ({"fit": {"model": "gaussian_offset"}}, RuntimeError, "atomic solver", True),
    ),
    ids=("facet-focus", "selector", "fit", "solver"),
)
def test_complete_configuration_rejects_a_late_target_without_partial_state(
    invalid_target,
    error,
    match,
    solver_failure,
    monkeypatch,
) -> None:
    engine = FitEngine()
    if solver_failure:
        def fail_fit(*_args, **_kwargs):
            raise RuntimeError("atomic solver failed")

        monkeypatch.setattr(engine, "fit", fail_fit)
    host = RasterPlotHost.from_plot(
        _snapshot(), CurvePlot(AxisRef.point("x")), fit_engine=engine
    )
    try:
        initial = host.wait_for_front(timeout=10)
        before = host.describe_display().result(timeout=10).value

        with pytest.raises(error, match=match):
            host.configure(
                parameters={"title": "must not leak"},
                **invalid_target,
            ).result(timeout=10)

        after = host.describe_display().result(timeout=10).value
        assert after.display_state == before.display_state
        assert host.front is initial
    finally:
        host.close(timeout=10)


@pytest.mark.parametrize(
    ("fit_target", "semantic"),
    (
        pytest.param(
            {"model": "gaussian_offset"},
            {"reduction": Reduction.MAX},
            id="rearm",
        ),
        pytest.param({}, None, id="clear"),
    ),
)
def test_failed_final_configuration_does_not_retire_the_previous_fit_authority(
    monkeypatch,
    fit_target,
    semantic,
) -> None:
    """Irreversible fit retirement belongs after the final render commit."""

    snapshot = _fit_curve_series(
        f"atomic-fit-{fit_target or 'clear'}-rollback",
        offset=0.1,
    )

    host = RasterPlotHost.from_plot(snapshot(0, 0.2), CurvePlot(AxisRef.point("x")))
    fail_final_render = False
    present = MatplotlibRenderer.present

    def controlled_present(renderer, *args, **kwargs):
        if fail_final_render:
            raise RuntimeError("forced final configuration render failure")
        return present(renderer, *args, **kwargs)

    monkeypatch.setattr(MatplotlibRenderer, "present", controlled_present)
    fit_events = []
    release_subscription = None
    try:
        host.wait_for_front(timeout=10)
        host.fit("gaussian_offset", live=True).result(timeout=30)
        release_subscription = host.subscribe_fit(fit_events.append).result(timeout=10).value
        old_front = host.front
        assert old_front is not None
        fail_final_render = True
        with pytest.raises(
            RuntimeError,
            match="forced final configuration render failure",
        ):
            host.configure(
                semantic=semantic,
                parameters={"title": "must roll back"},
                fit=fit_target,
            ).result(timeout=30)

        assert host.front is old_front
        fail_final_render = False
        host.update_data(snapshot(1, 0.25)).result(timeout=30)
        assert fit_events[-1].result.model.model_id == "gaussian_offset"
        assert fit_events[-1].result.source_revision == 1
        host.configure(
            semantic=semantic,
            parameters={"title": "committed"},
            fit=fit_target,
        ).result(timeout=30)
    finally:
        fail_final_render = False
        if release_subscription is not None:
            release_subscription().result(timeout=10)
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
