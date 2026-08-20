from __future__ import annotations

import numpy as np

from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, PlotSession
from zlc_plot._fit_projection import FitScope
from zlc_plot.fit import FacetFitBatchResult, FitEngine
from zlc_plot.selectors import NumericRange
from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


def _facet_snapshot(
    *,
    revision: int = 0,
    scale: float = 1.0,
    noisy: bool = False,
    facet_unit: str | None = None,
) -> DatasetSnapshot:
    x = np.linspace(-3.0, 3.0, 20)
    facets = np.repeat([0.0, 1.0], 10)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns(
            {"x": x, "facet": facets},
            units=None if facet_unit is None else {"facet": facet_unit},
        ),
        dtype=np.float64,
        generation="facet-live-fit",
    )
    values = (2.0 * np.exp(-0.5 * ((x - 0.2) / 1.0) ** 2) + 0.2)[None, :]
    if noisy:
        values = values + (
            0.04 * np.sin(np.arange(x.size, dtype=float) * 1.17)
            + 0.025 * np.cos(np.arange(x.size, dtype=float) * 0.37)
        )[None, :]
    return DatasetSnapshot(schema, values * scale, revision=revision)


def _spec() -> FacetGridPlot:
    return FacetGridPlot(
        AxisRef.point("facet"),
        CurvePlot(AxisRef.point("x")),
    )


def test_facet_live_fit_paints_every_cell_and_focus_keeps_annotation() -> None:
    session = PlotSession(_facet_snapshot(), _spec())
    try:
        result = session.fit("gaussian_offset", live=True)
        assert isinstance(result, FacetFitBatchResult)
        assert len(result.overlays) == 2
        assert all(overlay.polylines for overlay in result.overlays)

        for index, axis in enumerate(session._renderer.axes["facet_cell"]):
            assert any(
                artist.axes is axis
                for artist in session._renderer._fit_artists
            )
            assert len(axis.texts) == 1
            text = axis.texts[0].get_text()
            assert "$x_0$" in text
            assert "±" in text
            assert "\n" not in text
            assert "$f(" not in text
            inset = session._defaults.style.render.axes_text_inset_fraction
            assert axis.texts[0].get_position() == (inset, 1.0 - inset)
            assert axis.texts[0].get_fontsize() == session._defaults.style.fonts.facet_fit_annotation_pt
            assert axis.texts[0].get_zorder() == session._defaults.style.artists.fit_annotation_zorder
            assert "headline=" not in axis.get_title()
            assert result.overlays[index].facet_index == index

        session.focus_facet(0)
        focused = session._renderer.axes["facet_cell"][0]
        assert focused.texts
        assert "x_0" in focused.texts[0].get_text()
        assert "\n" in focused.texts[0].get_text()
        assert "$f(" in focused.texts[0].get_text()
    finally:
        session.close()


def test_facet_fit_overview_keeps_failure_diagnostic_inside_cell() -> None:
    session = PlotSession(_facet_snapshot(), _spec())
    try:
        result = session.fit("gaussian_offset", live=True)
        assert isinstance(result, FacetFitBatchResult)
        # Replace the accepted result with a deliberately bounded failure overlay
        # through the renderer's public presentation transaction.
        from zlc_plot._fit_scene import FitOverlay

        failure = FitOverlay(
            success=False,
            diagnostic="this diagnostic is intentionally much longer than one cell",
            facet_index=0,
        )
        session._renderer._update_facet_fit_overview(
            (failure, *result.overlays[1:]),
            "gaussian_offset",
        )
        axis = session._renderer.axes["facet_cell"][0]
        assert axis.texts
        assert len(axis.texts[0].get_text()) <= 24
        inset = session._defaults.style.render.axes_text_inset_fraction
        assert axis.texts[0].get_position() == (inset, 1.0 - inset)
        assert "headline=" not in axis.get_title()
    finally:
        session.close()


class _FailFirstFitEngine(FitEngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def fit(self, model, coordinates, observations=None, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("forced facet cell failure")
        return super().fit(model, coordinates, observations, **kwargs)


def test_facet_result_publishes_mixed_success_and_explicit_error_validity() -> None:
    session = PlotSession(
        _facet_snapshot(facet_unit="m"),
        _spec(),
        fit_engine=_FailFirstFitEngine(),
    )
    try:
        result = session.fit("gaussian_offset", live=True)
        assert isinstance(result, FacetFitBatchResult)
        assert np.array_equal(result.success, [False, True])
        assert result.sample_unit == "m"
        for name in result.parameter_names:
            assert np.isnan(result.parameter_values[name][0])
            assert np.isnan(result.parameter_errors[name][0])
            assert not bool(result.parameter_error_validity[name][0])
            assert bool(result.parameter_error_validity[name][1])
    finally:
        session.close()


def test_facet_live_fit_pairs_the_whole_batch_with_its_data_frame() -> None:
    """A facet pair commits data@1 and the whole batch fit@1 as one front."""

    initial = _facet_snapshot()
    session = PlotSession(initial, _spec())
    surfaces: list[int] = []
    release_surface = None
    try:
        session.fit("gaussian_offset", live=True)
        release_surface = session.subscribe_surface(
            lambda: surfaces.append(session.data_revision)
        )
        original_artists = tuple(session._renderer._fit_artists)
        assert original_artists
        prepared = session.prepare_live_frame(
            _facet_snapshot(revision=1, scale=1.1)
        ).result(timeout=10.0)
        solve = session.solve_live_frame(prepared)
        assert solve is not None
        solved = solve.result(timeout=10.0)
        finalization = session.commit_live_frame(prepared, solved)
        assert finalization is not None

        accepted = session.last_fit
        assert isinstance(accepted, FacetFitBatchResult)
        assert accepted.source_revision == 1
        assert len(accepted.overlays) == 2
        assert all(overlay.polylines for overlay in accepted.overlays)
        # The pair repainted the whole batch inside the one commit: every
        # cell carries exactly its revision-1 overlay artists.
        assert session._renderer._fit_artists
        assert session.fit_status == "current"
        assert surfaces == [1], "facet data+fit pair rendered more than once"

        session.fit("lorentzian", live=True)
        assert session.last_fit.model.model_id == "lorentzian"
    finally:
        if release_surface is not None:
            release_surface()
        session.close()


def test_one_region_gives_every_cell_of_the_batch_the_same_domain() -> None:
    """A region is drawn in one cell; the whole batch is solved over it.

    The region belongs to the cell the operator drew it in, so every other
    cell used to fall through to the viewport -- a batch whose cells were
    silently fitted over different data.
    """

    session = PlotSession(_facet_snapshot(), _spec())
    try:
        session.set_x_selector(-1.0, 1.0)
        session.set_viewport(NumericRange(-3.0, 0.0), NumericRange(-10.0, 10.0))
        result = session.fit("gaussian_offset", live=False, fit_all_facets=True)
        assert isinstance(result, FacetFitBatchResult)

        selections = session._accepted_fit.selections
        assert len(selections) == 2
        assert {selection.scope for selection in selections} == {FitScope.SELECTOR}
        assert len({selection.observations.size for selection in selections}) == 1
    finally:
        session.close()
