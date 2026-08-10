from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from test_facet_live_fit import _facet_snapshot, _spec
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSession,
)
from zlc_plot.fit import FacetFitBatchResult, FitResult
from zlc_plot.fit import FitDeadlineExceeded, FitEngine
from zlc_plot.live import _PacedLiveFrame


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
        seeded = session._fit_warm_starts[key].parameters
        first_threshold = session._classifier_thresholds[0]
        session._refresh_threshold_classifier()
        # The warm refresh re-solves from the prior solution; the threshold
        # is reproducible to the classifier's own scalar-optimizer tolerance.
        assert session._classifier_thresholds[0] == pytest.approx(
            first_threshold, rel=1e-2, abs=1e-3
        )
        assert session._fit_warm_starts[key].parameters == pytest.approx(
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


# --- live image re-fits: the warm seed competes, it never short-circuits ----

_IMAGE_W, _IMAGE_H = 128, 96


def _blob_image_snapshot(
    centers: tuple[tuple[float, float, float], ...],
    *,
    revision: int,
    x_unit: str = "m",
    y_unit: str = "m",
    sx: float = 6.0,
    sy: float = 6.0,
) -> DatasetSnapshot:
    """One 128x96 image frame with Gaussian blobs at (cx, cy, amplitude)."""

    x = np.arange(_IMAGE_W, dtype=np.float64)
    y = np.arange(_IMAGE_H, dtype=np.float64)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"sample": [0.0]}),
        data_axes=(
            Axis.create("x", values=x, canonical_unit=x_unit),
            Axis.create("y", values=y, canonical_unit=y_unit),
        ),
        dtype=np.float64,
        canonical_unit="1",
        generation=f"warm-image-{x_unit}-{y_unit}",
    )
    xx, yy = np.meshgrid(x, y)
    values = np.full(xx.shape, 100.0)
    for cx, cy, amplitude in centers:
        values += amplitude * np.exp(
            -(((xx - cx) ** 2) / sx**2 + ((yy - cy) ** 2) / sy**2)
        )
    return DatasetSnapshot(schema, values.T[None, None, :, :], revision=revision)


def _image_spec() -> ImagePlot:
    return ImagePlot(AxisRef.data("x"), AxisRef.data("y"))


@pytest.mark.parametrize(
    ("model", "y_unit"),
    [("radial_gaussian_center", "m"), ("anisotropic_gaussian_center", "s")],
)
def test_live_image_refit_lands_on_a_blob_displaced_beyond_two_sigma(
    model: str,
    y_unit: str,
) -> None:
    """A moved blob must win over descent from the stale warm basin."""

    first = _blob_image_snapshot(((64.0, 48.0, 3000.0),), revision=0, y_unit=y_unit)
    session = PlotSession(first, _image_spec())
    try:
        initial = session.fit(model, live=True)
        assert initial.success
        assert initial.parameters["center_x"] == pytest.approx(64.0, abs=0.1)
        # ~8 sigma displacement: far outside the warm basin.
        moved = _blob_image_snapshot(
            ((24.0, 78.0, 3000.0),), revision=1, y_unit=y_unit
        )
        result = _present_and_wait(session, moved, 1)
        assert isinstance(result, FitResult)
        assert result.success
        assert result.parameters["center_x"] == pytest.approx(24.0, abs=0.1)
        assert result.parameters["center_y"] == pytest.approx(78.0, abs=0.1)
        # Recovery stays exact on the next static frame: the accept above is
        # also the seed for this solve.
        static = _blob_image_snapshot(
            ((24.0, 78.0, 3000.0),), revision=2, y_unit=y_unit
        )
        result = _present_and_wait(session, static, 2)
        assert result.parameters["center_x"] == pytest.approx(24.0, abs=0.1)
        assert result.parameters["center_y"] == pytest.approx(78.0, abs=0.1)
    finally:
        session.close()


def test_live_image_refit_matches_cold_across_occupancy_resamples() -> None:
    """Site-camera style: blobs appear and vanish between revisions.

    Every accepted live re-fit must match a cold fit of the same frame (or
    beat it on reduced chi-square): the warm seed may accelerate the solve
    but never change which solution wins.
    """

    sites = ((32.0, 24.0), (96.0, 24.0), (32.0, 72.0), (96.0, 72.0), (64.0, 48.0))
    rng = np.random.default_rng(11)

    def occupancy_frame(revision: int) -> DatasetSnapshot:
        occupied = rng.random(len(sites)) < 0.6
        if not occupied.any():
            occupied[int(rng.integers(len(sites)))] = True
        centers = tuple(
            (cx, cy, 2000.0 + 250.0 * index)
            for index, ((cx, cy), lit) in enumerate(zip(sites, occupied))
            if lit
        )
        return _blob_image_snapshot(centers, revision=revision, sx=5.0, sy=5.0)

    frames = [occupancy_frame(revision) for revision in range(7)]
    session = PlotSession(frames[0], _image_spec())
    try:
        assert session.fit("radial_gaussian_center", live=True).success
        for revision in range(1, len(frames)):
            live = _present_and_wait(session, frames[revision], revision)
            cold_session = PlotSession(frames[revision], _image_spec())
            try:
                cold = cold_session.fit("radial_gaussian_center", live=False)
            finally:
                cold_session.close()
            error = float(
                np.hypot(
                    live.parameters["center_x"] - cold.parameters["center_x"],
                    live.parameters["center_y"] - cold.parameters["center_y"],
                )
            )
            assert error < 0.5 or (
                live.reduced_chi_square <= cold.reduced_chi_square
            ), f"revision {revision}: live drifted {error:.2f} px from cold"
    finally:
        session.close()


# --- warm-seed memory hardening ---------------------------------------------


def test_degenerate_image_accepts_are_never_remembered_as_seeds() -> None:
    """Frame-sized radii and vanished amplitudes must not seed later solves."""

    session = PlotSession(
        _blob_image_snapshot(((64.0, 48.0, 3000.0),), revision=0),
        _image_spec(),
    )
    try:
        selection = session.fit_selection("radial_gaussian_center")
        good = session.fit("radial_gaussian_center", live=False)
        assert session._propose_warm_seed(good, selection) is not None
        values = np.asarray(good.parameter_values).copy()
        radius_index = good.parameter_names.index("one_over_e_radius")
        values[radius_index] = 0.6 * float(np.ptp(selection.regular_image.x_coordinates))
        pinned = replace(good, parameter_values=values)
        assert session._propose_warm_seed(pinned, selection) is None
        values = np.asarray(good.parameter_values).copy()
        values[good.parameter_names.index("amplitude")] = 1e-4
        flat = replace(good, parameter_values=values)
        assert session._propose_warm_seed(flat, selection) is None
        assert session._propose_warm_seed(replace(good, success=False), selection) is None
    finally:
        session.close()


def test_chi_square_blowup_drops_the_remembered_seed() -> None:
    """A drastically worse accept clears the memory; the next solve runs cold."""

    session = PlotSession(
        _blob_image_snapshot(((64.0, 48.0, 3000.0),), revision=0),
        _image_spec(),
    )
    try:
        accepted = session.fit("radial_gaussian_center", live=True)
        assert accepted.success
        generation = session._fit_request_generation
        key = (generation, "radial_gaussian_center", None)
        assert key in session._fit_warm_starts
        selection = session.fit_selection("radial_gaussian_center")
        remembered = session._fit_warm_starts[key]
        # A modest degradation (inside two orders of magnitude above the
        # noise floor) keeps the memory fresh.
        modest = replace(
            accepted, reduced_chi_square=max(accepted.reduced_chi_square, 1e-9)
        )
        session._remember_fit_warm_starts(
            modest, request_generation=generation, selections=(selection,)
        )
        assert key in session._fit_warm_starts
        # A blow-up far beyond the remembered accept drops the seed entirely.
        blown = replace(
            accepted,
            reduced_chi_square=1e6
            * max(remembered.reduced_chi_square, remembered.noise_floor, 1e-6),
        )
        session._remember_fit_warm_starts(
            blown, request_generation=generation, selections=(selection,)
        )
        assert key not in session._fit_warm_starts
    finally:
        session.close()


# --- commit-time budget follows the controller's actual cadence -------------


class _PacedFitEngine(_RecordingFitEngine):
    """Honor commit deadlines like the real solver, at a scripted duration."""

    def __init__(self, solve_seconds: float) -> None:
        super().__init__()
        self.solve_seconds = solve_seconds
        self.deadlines: list[float | None] = []

    def fit(self, model, coordinates, observations=None, **kwargs):  # type: ignore[no-untyped-def]
        options = kwargs.get("options")
        deadline = None if options is None else options.deadline_seconds
        self.deadlines.append(deadline)
        if deadline is not None:
            if deadline < self.solve_seconds:
                raise FitDeadlineExceeded("scripted solve exceeds the budget")
            time.sleep(self.solve_seconds)
        return super().fit(model, coordinates, observations, **kwargs)


def test_slow_solve_lands_same_frame_at_a_wide_cadence() -> None:
    """At a 400 ms cadence a ~150 ms solve must not be aborted to async."""

    engine = _PacedFitEngine(solve_seconds=0.15)
    session = PlotSession(
        _dense_facet_snapshot(),
        CurvePlot(AxisRef.point("x")),
        fit_engine=engine,
    )
    try:
        assert session.fit("gaussian_offset", live=True).success
        prepared = session.prepare_live_frame(
            _dense_facet_snapshot(revision=1, scale=1.001)
        ).result(timeout=10.0)
        finalization = session.commit_live_frame(_PacedLiveFrame(prepared, 400))
        assert finalization is not None
        accepted = session.last_fit
        assert isinstance(accepted, FitResult)
        assert accepted.source_revision == 1, "solve was aborted to async"
        assert engine.deadlines[-1] is not None and engine.deadlines[-1] > 0.15
        session.finalize_live_frame(finalization)
        assert session.last_fit is accepted
    finally:
        session.close()


def test_slow_solve_still_defers_to_async_at_the_default_cadence() -> None:
    """Without a paced envelope the 100 ms default budget aborts the solve."""

    engine = _PacedFitEngine(solve_seconds=0.15)
    session = PlotSession(
        _dense_facet_snapshot(),
        CurvePlot(AxisRef.point("x")),
        fit_engine=engine,
    )
    try:
        assert session.fit("gaussian_offset", live=True).success
        prepared = session.prepare_live_frame(
            _dense_facet_snapshot(revision=1, scale=1.001)
        ).result(timeout=10.0)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        assert engine.deadlines[-1] is not None and engine.deadlines[-1] <= 0.1
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
