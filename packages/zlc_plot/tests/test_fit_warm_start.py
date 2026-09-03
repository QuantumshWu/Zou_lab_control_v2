from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, REPEAT, SPATIAL_X, SPATIAL_Y
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
from zlc_plot.fit import FitDeadlineExceeded, FitEngine, FitOptions


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


def _dense_facet_snapshot(*, revision: int = 0, scale: float = 1.0) -> OwnedSnapshot:
    x = np.linspace(-3.0, 3.0, 41)
    facet = np.repeat([0.0, 1.0], x.size)
    coordinates = np.tile(x, 2)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": coordinates, "facet": facet}),
        dtype=np.float64,
    )
    values = np.tile(
        2.0 * np.exp(-0.5 * ((x - 0.2) / 1.0) ** 2) + 0.2,
        2,
    )
    return make_snapshot(schema, (values * scale).reshape(1, -1), revision=revision)


def _dense_spec() -> FacetGridPlot:
    return FacetGridPlot(AxisRef.point("facet"), CurvePlot(AxisRef.point("x")))


def _present_and_wait(
    session: PlotSession,
    snapshot: OwnedSnapshot,
    revision: int,
) -> FitResult | FacetFitBatchResult:
    """Drive one complete data/fit pair through the live protocol.

    The commit installs data@N and fit@N together, so the accepted result is
    available synchronously afterwards — no finalize step, no polling.
    """

    prepared = session.prepare_live_frame(snapshot).result(timeout=10.0)
    solve = session.solve_live_frame(prepared)
    solved = None if solve is None else solve.result(timeout=10.0)
    finalization = session.commit_live_frame(prepared, solved)
    assert finalization is not None
    result = session.last_fit
    assert result is not None and result.source_revision == revision, (
        f"fit revision {revision} was not accepted with its own data frame"
    )
    return result


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


def _bimodal_snapshot(*, revision: int = 0) -> OwnedSnapshot:
    rng = np.random.default_rng(3 + revision)
    values = np.concatenate(
        (rng.normal(-2.0, 0.6, 150), rng.normal(2.0, 0.7, 150))
    )
    schema = make_dataset_schema(
        repeat_domain(size=values.size),
        mapped_domain_from_columns({"sample": (0.0,)}),
        dtype=np.float64,
    )
    return make_snapshot(schema, values[:, None], revision=revision)


def test_threshold_classifier_refresh_warm_starts_from_prior_solution() -> None:
    """Classifier seeds persist under a stable generation across refreshes."""

    session = PlotSession(
        _bimodal_snapshot(),
        HistogramPlot(),
        parameters={"threshold_classifier": True},
    )
    try:
        # Read where the classifier actually cuts: nobody has chosen a
        # threshold here, so that is the one this fit proposes.
        settled = session._classifier_thresholds_settled()
        assert len(settled) == 1
        assert settled[0] is not None
        key = (-1, "bimodal_gaussian", None)
        assert key in session._fit_warm_starts
        seeded = session._fit_warm_starts[key].parameters
        first_threshold = settled[0]
        session._refresh_threshold_classifier()
        # The warm refresh re-solves from the prior solution; the threshold
        # is reproducible to the classifier's own scalar-optimizer tolerance.
        assert session._classifier_thresholds_settled()[0] == pytest.approx(
            first_threshold, rel=1e-2, abs=1e-3
        )
        assert session._fit_warm_starts[key].parameters == pytest.approx(
            seeded, rel=1e-3, abs=1e-6
        )
    finally:
        session.close()


def test_live_fit_overlay_lands_in_the_same_committed_front() -> None:
    """A fit-armed frame is a pair: data@N and fit@N install in one commit."""

    session = PlotSession(
        _dense_facet_snapshot(),
        CurvePlot(AxisRef.point("x")),
    )
    surfaces: list[int] = []
    release_surface = None
    try:
        first = session.fit("gaussian_offset", live=True)
        assert first.success
        release_surface = session.subscribe_surface(
            lambda: surfaces.append(session.data_revision)
        )
        accepted = _present_and_wait(
            session,
            _dense_facet_snapshot(revision=1, scale=1.001),
            1,
        )
        assert isinstance(accepted, FitResult)
        assert session.fit_status == "current"
        assert surfaces == [1], "one exact data+fit pair rendered more than once"
    finally:
        if release_surface is not None:
            release_surface()
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


def test_deadline_exceeded_pair_is_loud_and_keeps_the_previous_pair() -> None:
    """A deadline cannot turn an armed revision into an unpaired data front."""

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
        solve = session.solve_live_frame(prepared)
        assert solve is not None
        with pytest.raises(FitDeadlineExceeded, match="forced live-fit deadline"):
            solve.result(timeout=10.0)
        assert session.data_revision == 0
        assert session.last_fit is first
        _present_and_wait(
            session,
            _dense_facet_snapshot(revision=1, scale=1.002),
            1,
        )
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
) -> OwnedSnapshot:
    """One 128x96 image frame with Gaussian blobs at (cx, cy, amplitude)."""

    x = np.arange(_IMAGE_W, dtype=np.float64)
    y = np.arange(_IMAGE_H, dtype=np.float64)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"sample": [0.0]}),
        cell_axes=(
            axis("x", values=x, unit=x_unit, role=SPATIAL_X),
            axis("y", values=y, unit=y_unit, role=SPATIAL_Y),
        ),
        dtype=np.float64,
        value_unit="1",
    )
    xx, yy = np.meshgrid(x, y)
    values = np.full(xx.shape, 100.0)
    for cx, cy, amplitude in centers:
        values += amplitude * np.exp(
            -(((xx - cx) ** 2) / sx**2 + ((yy - cy) ** 2) / sy**2)
        )
    return make_snapshot(schema, values.T[None, None, :, :], revision=revision)


def _image_spec() -> ImagePlot:
    return ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y"))


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

    def occupancy_frame(revision: int) -> OwnedSnapshot:
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


# --- the pair solve runs unbudgeted; only caller deadlines apply -------------


def test_pair_solves_carry_no_library_deadline() -> None:
    """The pair engine solves to completion: no cadence budget exists.

    A caller-authored ``FitOptions.deadline_seconds`` still travels with the
    request; without one, the solver sees no deadline at all.
    """

    class _DeadlineRecorder(_RecordingFitEngine):
        def __init__(self) -> None:
            super().__init__()
            self.deadlines: list[float | None] = []

        def fit(self, model, coordinates, observations=None, **kwargs):  # type: ignore[no-untyped-def]
            options = kwargs.get("options")
            self.deadlines.append(
                None if options is None else options.deadline_seconds
            )
            return super().fit(model, coordinates, observations, **kwargs)

    engine = _DeadlineRecorder()
    session = PlotSession(
        _dense_facet_snapshot(),
        CurvePlot(AxisRef.point("x")),
        fit_engine=engine,
    )
    try:
        assert session.fit("gaussian_offset", live=True).success
        assert engine.deadlines == [None]
        _present_and_wait(
            session,
            _dense_facet_snapshot(revision=1, scale=1.001),
            1,
        )
        assert engine.deadlines == [None, None]

        session.clear_fit()
        assert session.fit(
            "gaussian_offset",
            live=True,
            options=FitOptions(deadline_seconds=2.5),
        ).success
        assert engine.deadlines[-1] == 2.5
        _present_and_wait(
            session,
            _dense_facet_snapshot(revision=2, scale=1.002),
            2,
        )
        assert engine.deadlines[-1] == 2.5
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
        solve = session.solve_live_frame(prepared)
        assert solve is not None
        with pytest.raises(RuntimeError, match="forced warm-start failure"):
            solve.result(timeout=10.0)
        assert not engine.fail_next
        assert session.data_revision == 0
        assert session.last_fit is first
        _present_and_wait(
            session,
            _dense_facet_snapshot(revision=1, scale=1.002),
            1,
        )
        assert engine.warm_starts[-1] is None
    finally:
        session.close()
