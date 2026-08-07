from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from test_facet_live_fit import _facet_snapshot, _spec
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, PlotSession
from zlc_plot.fit import FacetFitBatchResult
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

        prepared = session.prepare_live_frame(
            _facet_snapshot(revision=1, scale=1.001)
        ).result(timeout=10.0)
        assert prepared.fit is not None
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
        cold_prepared = cold_session.prepare_live_frame(revision).result(timeout=10.0)
        assert cold_prepared.fit is not None
        cold = cold_prepared.fit.result
    finally:
        cold_session.close()

    warm_session = PlotSession(data, _dense_spec())
    try:
        warm_session.fit("gaussian_offset", live=True)
        warm_prepared = warm_session.prepare_live_frame(revision).result(timeout=10.0)
        assert warm_prepared.fit is not None
        warm = warm_prepared.fit.result
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
        with np.testing.assert_raises(RuntimeError):
            session.prepare_live_frame(
                _dense_facet_snapshot(revision=1, scale=1.001)
            ).result(timeout=10.0)
        session.prepare_live_frame(
            _dense_facet_snapshot(revision=2, scale=1.002)
        ).result(timeout=10.0)
        assert engine.warm_starts[-1] is None
    finally:
        session.close()
