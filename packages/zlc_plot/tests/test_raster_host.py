from __future__ import annotations

from concurrent.futures import CancelledError, Future
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
from zlc_plot._axis_transform import canvas_physical_size
from zlc_plot.selectors import CrosshairPoint, NumericRange, SelectorState
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


def test_equal_device_pixel_ratio_reuses_the_current_front() -> None:
    host = RasterPlotHost.from_plot(
        _snapshot(),
        CurvePlot(AxisRef.point("x")),
        device_pixel_ratio=2.0,
    )
    try:
        first = host.wait_for_front(timeout=10)
        unchanged = host.set_device_pixel_ratio(2.0).result(timeout=10)
        assert unchanged.front is first
        changed = host.set_device_pixel_ratio(1.0).result(timeout=10)
        assert changed.front.identity.sequence == first.identity.sequence + 1
        assert changed.front.device_pixel_ratio == 1.0
    finally:
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


def test_press_lands_on_the_painted_transform_it_carries() -> None:
    """The front a press arrives with IS what the operator saw.

    The widget swaps pixels and identity atomically, so the transform in
    the event always matches the picture that was pressed on, and the
    gesture layer interprets the press THROUGH it into canonical
    coordinates.  Even after live autoscale moved the current limits,
    the stale-front press is self-consistent and must be accepted --
    rejecting it bounced the first press after every commit for as long
    as the frontend ran one front behind.
    """

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
        before = host.describe_display().result(timeout=10).value
        updated = host.update_data(next_data).result(timeout=10)
        latest = host.front
        assert latest is not None
        assert updated.front is latest
        assert updated.value == host.describe_display().result(timeout=10).value
        assert updated.value.limits != before.limits
        assert latest.identity.sequence > stale.identity.sequence

        state = host._pointer_event(
            "press",
            0.45,
            0.45,
            button=1,
            identity=stale.identity,
            axes=stale.interaction.axes[0],
            interaction=stale.interaction,
        ).result(timeout=10)
        assert state is not None
        host._pointer_event("cancel", 0.45, 0.45, button=1).result(timeout=10)
    finally:
        host.close(timeout=10)


def test_press_accepts_a_live_revision_that_held_the_geometry_still() -> None:
    """A live frame that moves no limits must not reject the press.

    This is the acquisition steady state: data revisions advance with
    every published frame, retention holds the view still, and the
    operator's press lands on exactly the geometry they saw.  Rejecting
    it made selectors and camera gestures unusable during live runs.
    """

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
        generation="raster-press-live-hold",
    )
    first_data = DatasetSnapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=0)
    same_data = DatasetSnapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=1)
    host = RasterPlotHost.from_plot(first_data, CurvePlot(AxisRef.point("x")))
    try:
        stale = host.wait_for_front(timeout=10)
        host.update_data(same_data).result(timeout=10)
        latest = host.front
        assert latest is not None
        assert latest.identity.data_revision != stale.identity.data_revision

        state = host._pointer_event(
            "press",
            0.45,
            0.45,
            button=1,
            identity=stale.identity,
            axes=stale.interaction.axes[0],
            interaction=stale.interaction,
        ).result(timeout=10)
        assert state is not None
        host._pointer_event(
            "cancel",
            0.45,
            0.45,
            button=1,
        ).result(timeout=10)
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

        weighted_crossing = 1.0 + 0.5 * float(np.log(0.7 / 0.3))
        configured = host.configure(
            parameters={"threshold_classifier": True},
            classifier_thresholds=(
                {
                    "value": weighted_crossing,
                    "scope": (
                        {
                            "domain": "point_coordinate",
                            "axis_id": "site",
                            "coordinate": 0,
                        },
                    ),
                    "repeat_index": None,
                    "gaussian_components": {
                        "left_mean": 0.0,
                        "left_sigma": 1.0,
                        "left_weight": 0.7,
                        "right_mean": 2.0,
                        "right_sigma": 1.0,
                        "right_weight": 0.3,
                    },
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
                    "gaussian_components": None,
                },
            ),
        ).result(timeout=10)
        assert configured.value.display_state.values["threshold_classifier"] is True
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == weighted_crossing
        session = host._require_session()
        first = session._classifier_results[0]
        assert first is not None
        values = first.parameters
        left_area = values["left_amplitude"] * values["left_sigma"]
        right_area = values["right_amplitude"] * values["right_sigma"]
        assert left_area / (left_area + right_area) == pytest.approx(0.7)
        left_curve = values["left_amplitude"] * np.exp(
            -0.5
            * ((weighted_crossing - 0.0) / values["left_sigma"]) ** 2
        )
        right_curve = values["right_amplitude"] * np.exp(
            -0.5
            * ((weighted_crossing - 2.0) / values["right_sigma"]) ** 2
        )
        assert left_curve == pytest.approx(right_curve)
        assert session._classifier_results[1] is None
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
            selectors=(
                SelectorState(
                    SelectorKind.CROSSHAIR,
                    CrosshairPoint(1.0, 2.0),
                ),
            ),
            viewport=None,
            facet_focus=None,
        )
        fitted = host.configure(fit=fit_target, **desired).result(timeout=30)
        assert fitted.front.identity.sequence == initial.identity.sequence + 1
        assert fitted.value.selectors == desired["selectors"]
        assert solve_count == 1

        unchanged = host.configure(fit=fit_target, **desired).result(timeout=30)
        assert unchanged.front is fitted.front
        assert solve_count == 1

        expressed = host.configure(
            fit={
                "model": "gaussian_offset",
                "expression": "center=0.3, sigma=guess(0.9)",
            },
            **desired,
        ).result(timeout=30)
        assert expressed.value.fit["fixed"] == {"center": 0.3}
        assert expressed.value.fit["initial"] == {"sigma": 0.9}
        assert expressed.value.fit_expression == (
            "sigma=guess(0.9), center=0.3"
        )
        assert expressed.value.fit_expression_error == ""

        automatic = host.configure(
            fit={
                "model": "gaussian_offset",
                "expression": "missing=1",
            },
            **desired,
        ).result(timeout=30)
        assert automatic.value.fit == {"model": "gaussian_offset"}
        assert automatic.value.fit_expression == "missing=1"
        assert "fit parameter" in automatic.value.fit_expression_error
        assert automatic.front is not expressed.front

        cleared = host.configure(fit={}, **desired).result(timeout=30)
        assert cleared.front.identity.sequence == automatic.front.identity.sequence + 1

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


def test_envelope_preserves_column_extremes_and_gaps() -> None:
    """The envelope is the drawing's contract: extremes and gaps survive."""

    from zlc_plot.rendering import _envelope_decimated

    n = 50_000
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(x * 40.0)
    y[12_345] = 9.0
    y[34_567] = -7.0
    y[20_000:20_400] = np.nan
    columns = 256
    enveloped = _envelope_decimated(x, y, (0.0, 1.0), columns)
    assert enveloped is not None
    out_x, out_y = enveloped
    assert out_x.size <= columns * 3 + 2
    finite = np.isfinite(out_y)
    assert float(np.max(out_y[finite])) == 9.0
    assert float(np.min(out_y[finite])) == -7.0
    # The invalid run must still break the stroke: some separator in the gap
    # columns is NaN.
    gap_zone = (out_x >= x[20_000]) & (out_x <= x[20_399])
    assert bool(np.any(~np.isfinite(out_y[gap_zone])))


def test_envelope_declines_sparse_windows() -> None:
    from zlc_plot.rendering import _envelope_decimated

    x = np.linspace(0.0, 1.0, 1_000)
    y = np.sin(x)
    assert _envelope_decimated(x, y, (0.0, 1.0), 256) is None
    dense_x = np.linspace(0.0, 1.0, 200_000)
    dense_y = np.sin(dense_x)
    # A deep zoom leaves fewer samples than the envelope needs: raw points.
    assert _envelope_decimated(dense_x, dense_y, (0.5, 0.5005), 256) is None
    assert _envelope_decimated(dense_x, dense_y, (0.0, 1.0), 256) is not None


def test_dense_curve_hands_display_resolution_polyline_to_the_artist() -> None:
    n = 300_000
    x = np.linspace(0.0, 1.0, n)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x}),
        dtype=np.float64,
        generation="envelope-host",
    )
    values = np.sin(x * 20.0).reshape(1, -1)
    host = RasterPlotHost.from_plot(
        DatasetSnapshot(schema, values, revision=0),
        CurvePlot(AxisRef.point("x")),
    )
    try:
        host.wait_for_front(timeout=30)
        host.update_data(
            DatasetSnapshot(schema, values + 0.001, revision=1)
        ).result(timeout=60)
        renderer = host._session._renderer
        lines = next(
            (
                value
                for value in renderer._artists.values()
                if isinstance(value, list)
                and value
                and hasattr(value[0], "get_xdata")
            ),
            None,
        )
        assert lines, "curve series artist must exist"
        drawn = np.asarray(lines[0].get_xdata())
        assert drawn.size < n / 10
    finally:
        host.close(timeout=30)


def test_curve_series_inspector_is_stable_sticky_and_redraw_bounded(
    monkeypatch, tmp_path,
) -> None:
    candidate = np.tile(np.arange(7.0), 2)
    site = np.repeat((17.0, 23.0), 7)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"candidate": candidate, "site": site}),
        dtype=np.float64,
        generation="series-inspector",
    )
    values = np.concatenate((1.0 + np.arange(7.0), 11.0 + np.arange(7.0)))[None]
    session = PlotSession(
        DatasetSnapshot(schema, values, 0),
        CurvePlot(AxisRef.point("candidate"), group=AxisRef.point("site")),
    )
    try:
        renderer = session._renderer
        axes = renderer.primary_axes
        width, height = canvas_physical_size(renderer.figure.canvas)

        def pointer(action, x, y, *, button=None, key=None):
            px, py = axes.transData.transform((x, y))
            transform = next(item for item in session._raster_axes_snapshot()
                             if item.role == "main")
            return session._raster_pointer_event(
                action, px / width, 1.0 - py / height,
                button=button, key=key, axes_snapshot=transform,
            )

        lines = renderer._artists["curve"]
        colors = {line.get_label(): line.get_color() for line in lines}
        generation = renderer.raster_generation
        hovered = pointer("move", 2.0, 3.0)
        assert hovered.publish_front
        assert renderer.raster_generation == generation + 1
        assert sorted(line.get_alpha() for line in lines) == [0.8, 1.0]
        base_width = renderer.style.artists.curve.linewidth
        np.testing.assert_allclose(
            sorted(line.get_linewidth() for line in lines),
            (base_width, 1.45 * base_width),
        )
        annotation = next(item for item in renderer._series_annotations.values()
                          if item.get_visible())
        assert "site=17" in annotation.get_text()
        assert not annotation.get_text().startswith("*")
        assert annotation.get_position() == (0.98, 0.98)
        assert annotation.get_bbox_patch() is None

        generation = renderer.raster_generation
        assert not pointer("move", 4.0, 5.0).publish_front
        assert renderer.raster_generation == generation

        pointer("press", 2.0, 3.0, button=1)
        pointer("release", 2.0, 3.0, button=1)
        assert session.selectors == ()
        locked = renderer._series_locked[1]
        assert sorted(line.get_alpha() for line in lines) == [0.18, 1.0]
        assert annotation.get_text().startswith("* ")
        pointer("move", 2.0, 13.0)
        assert renderer._series_locked[1] == locked

        pointer("press", 2.0, 3.0, button=1)
        pointer("release", 2.0, 3.0, button=1)
        assert renderer._series_locked is None and session.selectors == ()
        pointer("move", 2.0, 3.0)
        pointer("press", 2.0, 3.0, button=1)
        pointer("release", 2.0, 3.0, button=1)
        pointer("press", 2.0, 7.0, button=1)
        pointer("release", 2.0, 7.0, button=1)
        assert renderer._series_locked is None and session.selectors == ()

        pointer("move", 2.0, 3.0)
        pointer("press", 2.0, 3.0, button=1)
        pointer("release", 2.0, 3.0, button=1)
        pointer("key", 0.0, 0.0, key="escape")
        assert renderer._series_locked is None

        observed = []
        pointer("move", 2.0, 3.0)
        pointer("press", 2.0, 3.0, button=1)
        pointer("release", 2.0, 3.0, button=1)
        monkeypatch.setattr(
            renderer.figure, "savefig",
            lambda *_args, **_kwargs: observed.append(tuple(line.get_alpha() for line in lines)),
        )
        session.save(tmp_path / "neutral.png")
        assert observed == [(0.8, 0.8)]
        assert renderer._series_locked is not None

        session.update_data(DatasetSnapshot(schema, values + 0.25, 1))
        assert {line.get_label(): line.get_color() for line in lines} == colors

        pointer("key", 0.0, 0.0, key="escape")
        validity = np.ones(values.shape, dtype=bool)
        validity[0, :7] = False
        validity[0, 2] = True
        session.update_data(
            DatasetSnapshot(
                schema,
                values + 0.25,
                2,
                validity=validity,
            )
        )
        isolated = next(line for line in lines if "site=17" in line.get_label())
        connected = next(line for line in lines if "site=23" in line.get_label())
        assert isolated.get_marker() == "_"
        assert np.asarray(isolated.get_markevery()).size == 1
        assert connected.get_marker() in (None, "None", "")
        assert pointer("move", 2.0, 3.25).publish_front
        assert "site=17" in renderer._series_hover[2]
        pointer("press", 2.0, 3.25, button=1)
        pointer("release", 2.0, 3.25, button=1)
        assert "site=17" in renderer._series_locked[2]
        assert isolated.get_markeredgewidth() == pytest.approx(2.0 * base_width)
    finally:
        session.close()


def test_curve_series_picker_never_uses_raw_dense_line_on_deep_zoom() -> None:
    count = 200_000
    x = np.linspace(0.0, 1.0, count)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x}),
        dtype=np.float64,
        generation="series-inspector-dense",
    )
    session = PlotSession(
        DatasetSnapshot(schema, np.sin(x)[None], 0),
        CurvePlot(AxisRef.point("x")),
    )
    try:
        renderer = session._renderer
        axes = renderer.primary_axes
        line = renderer._series_lines[id(axes)][0][0]
        renderer._set_xlim(axes, 0.5, 0.5005)
        assert np.asarray(line.get_xdata()).size == count
        px, py = axes.transData.transform((0.50025, np.sin(0.50025)))
        start = time.perf_counter()
        assert renderer._series_hit(axes, px, py, 10.0) is not None
        assert time.perf_counter() - start < 0.25
    finally:
        session.close()


def test_locked_curve_wheel_steps_canonical_series_without_zoom() -> None:
    candidate = np.tile(np.arange(7.0), 2)
    site = np.repeat((17.0, 23.0), 7)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"candidate": candidate, "site": site}),
        dtype=np.float64,
        generation="series-wheel",
    )
    values = np.concatenate((1.0 + np.arange(7.0), 11.0 + np.arange(7.0)))[None]
    session = PlotSession(
        DatasetSnapshot(schema, values, 0),
        CurvePlot(AxisRef.point("candidate"), group=AxisRef.point("site")),
    )
    try:
        renderer = session._renderer
        axes = renderer.primary_axes
        width, height = canvas_physical_size(renderer.figure.canvas)

        def event(action, x, y, *, button=None, step=0.0):
            px, py = axes.transData.transform((x, y))
            transform = next(item for item in session._raster_axes_snapshot()
                             if item.role == "main")
            return session._raster_pointer_event(
                action, px / width, 1.0 - py / height,
                button=button, step=step, axes_snapshot=transform,
            )

        event("move", 2.0, 13.0)
        event("press", 2.0, 13.0, button=1)
        event("release", 2.0, 13.0, button=1)
        assert "site=23" in renderer._series_locked[2]
        original_xlim = tuple(axes.get_xlim())

        changed = event("scroll", 2.0, 13.0, step=1.0)
        assert changed.publish_front
        assert "site=17" in renderer._series_locked[2]
        assert tuple(axes.get_xlim()) == original_xlim

        generation = renderer.raster_generation
        clamped = event("scroll", 2.0, 3.0, step=1.0)
        assert not clamped.publish_front
        assert renderer.raster_generation == generation
        assert tuple(axes.get_xlim()) == original_xlim

        event("scroll", 2.0, 3.0, step=-1.0)
        assert "site=23" in renderer._series_locked[2]
        assert tuple(axes.get_xlim()) == original_xlim
        event("scroll", 2.0, 13.0, step=1.0)
        assert "site=17" in renderer._series_locked[2]

        event("press", 2.0, 3.0, button=1)
        event("release", 2.0, 3.0, button=1)
        assert renderer._series_locked is None
        event("scroll", 2.0, 3.0, step=1.0)
        assert tuple(axes.get_xlim()) != original_xlim
    finally:
        session.close()


def test_axis_resolution_grabs_what_is_visible() -> None:
    """Nearest-axis pointer resolution within the selector handle radius.

    A guide painted ON an axes boundary spills its visible linewidth
    outside the box; the resolver must grab it from either side, and the
    nearest axis must win inside the gaps between adjacent axes."""

    from types import SimpleNamespace

    from zlc_plot._axis_transform import AxisTransform
    from zlc_plot.backends import _axis_at_normalized

    def transform(role, left, top, right, bottom):
        return AxisTransform(
            role, None, (left, top, right, bottom),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        )

    image = transform("image", 0.10, 0.10, 0.70, 0.90)
    rail = transform("distribution", 0.72, 0.10, 0.80, 0.90)
    front = SimpleNamespace(
        logical_size=(1000, 1000),
        interaction=SimpleNamespace(axes=(image, rail)),
    )

    inside = _axis_at_normalized(front, 0.75, 0.5, tolerance_px=10.0)
    assert inside is rail
    # 3 px ABOVE the rail's top edge: the outer half of an edge guide.
    above = _axis_at_normalized(front, 0.75, 0.10 - 0.003, tolerance_px=10.0)
    assert above is rail
    # In the gap between image and rail, the nearer one wins.
    near_rail = _axis_at_normalized(front, 0.715, 0.5, tolerance_px=10.0)
    assert near_rail is rail
    near_image = _axis_at_normalized(front, 0.705, 0.5, tolerance_px=10.0)
    assert near_image is image
    # Beyond the radius resolves to nothing, exactly as before.
    assert _axis_at_normalized(front, 0.75, 0.05, tolerance_px=10.0) is None
    assert _axis_at_normalized(front, 0.75, 0.099, tolerance_px=0.0) is None


def test_press_ignores_the_crosshair_marker_in_the_painted_interaction() -> None:
    """A pick republishes a front carrying its crosshair; the NEXT press
    arrives with the previous front for as long as the frontend lags one
    behind.  The crosshair is a marker nothing can grab, so it is not
    part of press currency -- rejecting on it froze every orbit that
    followed a pick."""

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
        generation="raster-press-marker",
    )
    data = DatasetSnapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=0)
    host = RasterPlotHost.from_plot(data, CurvePlot(AxisRef.point("x")))
    try:
        stale = host.wait_for_front(timeout=10)
        host.set_crosshair_selector(1.0, 2.0).result(timeout=10)
        latest = host.front
        assert latest is not None
        assert latest.interaction.selectors != stale.interaction.selectors

        state = host._pointer_event(
            "press",
            0.45,
            0.45,
            button=2,
            identity=stale.identity,
            axes=stale.interaction.axes[0],
            interaction=stale.interaction,
        ).result(timeout=10)
        assert state is not None
        host._pointer_event("cancel", 0.45, 0.45, button=2).result(timeout=10)
    finally:
        host.close(timeout=10)


def test_scroll_is_self_relative_and_needs_no_front_currency() -> None:
    """A 3D wheel tick commits the camera and bumps the display revision;
    the frontend is one front behind for a beat, and demanding identity
    currency on the NEXT tick bounced continuous zooming.  A scroll is
    self-relative view navigation: it rides whatever front it saw."""

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
        generation="raster-scroll-currency",
    )
    data = DatasetSnapshot(schema, np.array([[1.0, 2.0, 3.0]]), revision=0)
    host = RasterPlotHost.from_plot(data, CurvePlot(AxisRef.point("x")))
    try:
        stale = host.wait_for_front(timeout=10)
        host.set_size("4x4").result(timeout=10)
        latest = host.front
        assert latest is not None
        assert latest.identity != stale.identity

        state = host._pointer_event(
            "scroll",
            0.45,
            0.45,
            step=1.0,
            identity=stale.identity,
            axes=stale.interaction.axes[0],
        ).result(timeout=10)
        assert state is not None
    finally:
        host.close(timeout=10)


def test_a_moving_hand_stands_down_every_speculative_frame() -> None:
    """One machine: the drag wins while it moves, on every surface.

    A panel's Edit surface and its live card render on separate worker
    threads and compete for the same cores.  Measured over 1024x1024
    data, the card's committed frames stalled a drag on the Edit surface
    to a per-move p90 of 354 ms (46 ms with the card idle).

    That included the panel being dragged.  A frame already running on the
    dragged host itself cannot be pre-empted, so a move that arrived
    during one waited out its whole render -- measured on a live console,
    the first move of an orbit took 192 ms against 134 once its own
    panel's frames stood aside too.  A data frame is speculative wherever
    it runs: the next publication supersedes it and nobody waits for this
    one in particular.  A hand is not.

    Only speculative work yields: a caller blocked on a control answer,
    the pointer work itself, and a surface that has no front to show yet
    never wait on a hand.
    """

    from zlc_plot.raster import _HANDS, _DispatchMode, _WorkerTask

    hand = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    other = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        hand.wait_for_front(timeout=10)
        other.wait_for_front(timeout=10)

        def task(mode: _DispatchMode) -> _WorkerTask:
            return _WorkerTask(lambda: None, Future(), mode, None, None, None)

        assert not other._yields_to_hand(task(_DispatchMode.PUBLISH))
        _HANDS.grip(hand.host_id)
        try:
            assert other._yields_to_hand(task(_DispatchMode.PUBLISH))
            assert other._yields_to_hand(task(_DispatchMode.PRESENTATION))
            # a blocked caller is not speculative work
            assert not other._yields_to_hand(task(_DispatchMode.CONTROL))
            assert not other._yields_to_hand(task(_DispatchMode.ADAPTIVE))
            # including the dragged host's own speculative frames
            assert hand._yields_to_hand(task(_DispatchMode.PUBLISH))
            # nor is a surface still reaching its first front: opening Edit
            # during someone's drag must not wait on it
            fresh = RasterPlotHost.from_plot(
                _snapshot(), CurvePlot(AxisRef.point("x"))
            )
            try:
                assert fresh.wait_for_front(timeout=10) is not None
            finally:
                fresh.close(timeout=10)
        finally:
            _HANDS.ungrip(hand.host_id, 0.0)

        # the hold outlives the grip by what the pointer work cost, so a
        # sibling cannot slip a frame into the gap between two moves
        _HANDS.touch(hand.host_id, 0.2)
        assert other._yields_to_hand(task(_DispatchMode.PUBLISH))
        _HANDS.forget(hand.host_id)
        assert not other._yields_to_hand(task(_DispatchMode.PUBLISH))
    finally:
        hand.close(timeout=10)
        other.close(timeout=10)


def test_yielded_frames_are_never_lost_only_deferred() -> None:
    """Standing down costs freshness, nothing else.

    A data frame retains only its latest successor, so work deferred
    while a hand moves collapses to one frame the moment it stops -- the
    yield must not queue up a burst to replay, and must not drop the
    newest picture either.
    """

    from zlc_plot.raster import _HANDS

    hand = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    other = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        hand.wait_for_front(timeout=10)
        first = other.wait_for_front(timeout=10)
        schema = _snapshot().block.schema
        _HANDS.grip(hand.host_id)
        pending = [
            other.update_data(
                DatasetSnapshot(
                    schema, np.array([[float(revision), 2.0, 3.0]]), revision=revision
                )
            )
            for revision in range(1, 4)
        ]
        time.sleep(0.15)
        assert other.front is not None
        assert other.front.identity == first.identity, "a frame ran under a hand"
        _HANDS.ungrip(hand.host_id, 0.0)
        _HANDS.forget(hand.host_id)
        pending[-1].result(timeout=10)
        latest = other.front
        assert latest is not None
        assert latest.identity.data_revision == 3
    finally:
        hand.close(timeout=10)
        other.close(timeout=10)


def test_a_revision_the_session_already_holds_is_nothing_to_do() -> None:
    """Handing the same data twice is a no-op, not a fault on the card.

    The submitter records what it handed over only once the render
    COMPLETES, and a gesture makes the arbiter supersede renders on purpose
    -- so a frame that committed but whose future was cancelled looked free
    and went round again.  ``prepare_live_frame`` refuses it, rightly, for a
    caller asking to advance; here the refusal reached the operator as "data
    revision must increase" on a panel they were only turning.

    A frame the session already holds is superseded.  That is a cancelled
    future, which the pipeline already knows how to ignore, not an error.
    """

    host = RasterPlotHost.from_plot(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        host.wait_for_front(10.0)
        schema = DatasetSchema.create(
            Axis.create("repeat", size=1),
            PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
            dtype=np.float64,
            generation="raster-host-test",
        )
        fresh = DatasetSnapshot(schema, np.array([[2.0, 3.0, 4.0]]), revision=1)
        host.update_data(fresh).result(10.0)

        repeated = host.update_data(
            DatasetSnapshot(schema, np.array([[9.0, 9.0, 9.0]]), revision=1)
        )
        try:
            repeated.result(10.0)
        except CancelledError:
            pass
        except Exception as error:  # pragma: no cover - the regression
            raise AssertionError(
                "re-handing a held revision surfaced %r instead of being "
                "ignored" % (error,)
            ) from None
    finally:
        host.close(timeout=10.0)
