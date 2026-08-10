from __future__ import annotations

import time

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from test_facet_live_fit import _facet_snapshot, _spec
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, HistogramPlot, PlotSession
from zlc_plot.fit import FacetFitBatchResult, FitResult
from zlc_plot.fit import FitEngine


class _RecordingFitEngine(FitEngine):
    def __init__(self) -> None:
        super().__init__()
        self.initials: list[np.ndarray | None] = []
        self.warm_starts: list[np.ndarray | None] = []

    def fit(self, model, coordinates, observations=None, **kwargs):  # type: ignore[no-untyped-def]
        initial = kwargs.get("initial")
        self.initials.append(
            None if initial is None else np.asarray(tuple(initial), dtype=float)
        )
        warm_start = kwargs.get("warm_start")
        self.warm_starts.append(
            None if warm_start is None else np.asarray(tuple(warm_start), dtype=float)
        )
        return super().fit(model, coordinates, observations, **kwargs)


class _FailOnceFitEngine(_RecordingFitEngine):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next = False

    def fit(self, model, coordinates, observations=None, **kwargs):  # type: ignore[no-untyped-def]
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("forced warm-start failure")
        return super().fit(model, coordinates, observations, **kwargs)


def _dense_facet_snapshot(*, revision: int = 0, scale: float = 1.0) -> DatasetSnapshot:
    x = np.linspace(-3.0, 3.0, 41)
    facet = np.repeat([0.0, 1.0], x.size)
    coordinates = np.tile(x, 2)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": coordinates, "facet": facet}),
        dtype=np.float64,
        generation="fit-warm-dense",
    )
    values = np.tile(
        2.0 * np.exp(-0.5 * ((x - 0.2) / 1.0) ** 2) + 0.2,
        2,
    )
    return DatasetSnapshot(schema, (values * scale).reshape(1, -1), revision=revision)


def _dense_spec() -> FacetGridPlot:
    return FacetGridPlot(AxisRef.point("facet"), CurvePlot(AxisRef.point("x")))


def _present_and_wait(
    session: PlotSession,
    snapshot: DatasetSnapshot,
    revision: int,
) -> FitResult | FacetFitBatchResult:
    prepared = session.prepare_live_frame(snapshot).result(timeout=10.0)
    finalization = session.commit_live_frame(prepared)
    assert finalization is not None
    session.finalize_live_frame(finalization)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        result = session.last_fit
        if result is not None and result.source_revision == revision:
            return result
        time.sleep(0.005)
    raise AssertionError(f"fit revision {revision} was not accepted")


def test_live_facet_revision_reuses_last_accepted_cell_parameters() -> None:
    engine = _RecordingFitEngine()
    session = PlotSession(_facet_snapshot(), _spec(), fit_engine=engine)
    try:
        first = session.fit("gaussian_offset", live=True)
        assert isinstance(first, FacetFitBatchResult)
        first_initial_count = len(engine.initials)
        assert first_initial_count == 2
        assert all(initial is None for initial in engine.initials)
        assert all(warm is None for warm in engine.warm_starts)

        _present_and_wait(
            session,
            _facet_snapshot(revision=1, scale=1.001),
            1,
        )
        assert len(engine.initials) == first_initial_count + 2
        for index, warm in enumerate(engine.warm_starts[-2:]):
            assert warm is not None
            assert np.allclose(warm, first.results[index].parameter_values, rtol=1e-12)
    finally:
        session.close()


def test_live_warm_start_keeps_the_facet_result_within_solver_tolerance() -> None:
    data = _dense_facet_snapshot()
    revision = _dense_facet_snapshot(revision=1, scale=1.001)

    cold_session = PlotSession(data, _dense_spec())
    try:
        cold_session.fit("gaussian_offset", live=True)
        cold_session._fit_warm_starts.clear()
        cold = _present_and_wait(cold_session, revision, 1)
    finally:
        cold_session.close()

    warm_session = PlotSession(data, _dense_spec())
    try:
        warm_session.fit("gaussian_offset", live=True)
        warm = _present_and_wait(warm_session, revision, 1)
        assert isinstance(cold, FacetFitBatchResult)
        assert isinstance(warm, FacetFitBatchResult)
        for cold_result, warm_result in zip(cold.results, warm.results, strict=True):
            assert cold_result is not None and warm_result is not None
            assert np.allclose(
                cold_result.parameter_values,
                warm_result.parameter_values,
                rtol=1e-6,
                atol=1e-8,
            )
            assert np.allclose(
                cold_result.standard_errors,
                warm_result.standard_errors,
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            )
    finally:
        warm_session.close()


def _bimodal_snapshot(*, revision: int = 0) -> DatasetSnapshot:
    rng = np.random.default_rng(3 + revision)
    values = np.concatenate(
        (rng.normal(-2.0, 0.6, 150), rng.normal(2.0, 0.7, 150))
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=values.size),
        PointTable.from_columns({"sample": (0.0,)}),
        dtype=np.float64,
        generation="classifier-warm",
    )
    return DatasetSnapshot(schema, values[:, None], revision=revision)


def test_threshold_classifier_refresh_warm_starts_from_prior_solution() -> None:
    """Classifier seeds persist under a stable generation across refreshes."""

    session = PlotSession(
        _bimodal_snapshot(),
        HistogramPlot(),
        parameters={"threshold_classifier": True},
    )
    try:
        assert len(session._classifier_thresholds) == 1
        assert session._classifier_thresholds[0] is not None
        key = (-1, "bimodal_gaussian", None)
        assert key in session._fit_warm_starts
        seeded = session._fit_warm_starts[key]
        first_threshold = session._classifier_thresholds[0]
        session._refresh_threshold_classifier()
        # The warm refresh re-solves from the prior solution; the threshold
        # is reproducible to the classifier's own scalar-optimizer tolerance.
        assert session._classifier_thresholds[0] == pytest.approx(
            first_threshold, rel=1e-2, abs=1e-3
        )
        assert session._fit_warm_starts[key] == pytest.approx(
            seeded, rel=1e-3, abs=1e-6
        )
    finally:
        session.close()


def test_live_fit_overlay_lands_in_the_same_committed_front() -> None:
    """The armed live fit solves inside commit: no finalize, no polling."""

    session = PlotSession(
        _dense_facet_snapshot(),
        CurvePlot(AxisRef.point("x")),
    )
    try:
        first = session.fit("gaussian_offset", live=True)
        assert first.success
        prepared = session.prepare_live_frame(
            _dense_facet_snapshot(revision=1, scale=1.001)
        ).result(timeout=10.0)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        accepted = session.last_fit
        assert isinstance(accepted, FitResult)
        assert accepted.source_revision == 1
        assert session.fit_status == "current"
        # finalize must not schedule a second solve for the same front
        session.finalize_live_frame(finalization)
        assert session.last_fit is accepted
    finally:
        session.close()


class _DeadlineOnceFitEngine(_RecordingFitEngine):
    """Raise FitDeadlineExceeded for exactly one solve, then delegate."""

    def __init__(self) -> None:
        super().__init__()
        self.deadline_next = False

    def fit(self, model, coordinates, observations=None, **kwargs):  # type: ignore[no-untyped-def]
        if self.deadline_next:
            self.deadline_next = False
            from zlc_plot.fit import FitDeadlineExceeded

            raise FitDeadlineExceeded("forced live-fit deadline")
        return super().fit(model, coordinates, observations, **kwargs)


def test_live_fit_deadline_falls_back_to_the_async_path() -> None:
    """A deadline publishes the data front and the executor finishes the fit."""

    engine = _DeadlineOnceFitEngine()
    session = PlotSession(
        _dense_facet_snapshot(),
        CurvePlot(AxisRef.point("x")),
        fit_engine=engine,
    )
    try:
        first = session.fit("gaussian_offset", live=True)
        assert first.success
        engine.deadline_next = True
        prepared = session.prepare_live_frame(
            _dense_facet_snapshot(revision=1, scale=1.001)
        ).result(timeout=10.0)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        # The synchronous attempt hit its deadline: the data front published
        # without a revision-1 overlay.
        lagging = session.last_fit
        assert isinstance(lagging, FitResult)
        assert lagging.source_revision == 0
        session.finalize_live_frame(finalization)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            result = session.last_fit
            if result is not None and result.source_revision == 1:
                break
            time.sleep(0.005)
        assert session.last_fit.source_revision == 1
    finally:
        session.close()


def test_fit_warm_cache_is_cleared_after_solver_exception() -> None:
    engine = _FailOnceFitEngine()
    session = PlotSession(
        _dense_facet_snapshot(),
        CurvePlot(AxisRef.point("x")),
        fit_engine=engine,
    )
    try:
        first = session.fit("gaussian_offset", live=True)
        assert first.success
        engine.fail_next = True
        prepared = session.prepare_live_frame(
            _dense_facet_snapshot(revision=1, scale=1.001)
        ).result(timeout=10.0)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        session.finalize_live_frame(finalization)
        deadline = time.monotonic() + 10.0
        while engine.fail_next and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not engine.fail_next
        _present_and_wait(
            session,
            _dense_facet_snapshot(revision=2, scale=1.002),
            2,
        )
        assert engine.warm_starts[-1] is None
    finally:
        session.close()
