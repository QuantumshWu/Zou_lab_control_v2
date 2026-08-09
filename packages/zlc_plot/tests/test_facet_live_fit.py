from __future__ import annotations

import time

import numpy as np

from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, PlotSession
from zlc_plot.fit import FacetFitBatchResult, FitEngine
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


def _wait_for_fit_revision(
    session: PlotSession,
    revision: int,
    *,
    timeout: float = 10.0,
) -> FacetFitBatchResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = session.last_fit
        if (
            isinstance(result, FacetFitBatchResult)
            and result.source_revision == revision
        ):
            return result
        time.sleep(0.005)
    raise AssertionError(f"fit revision {revision} was not accepted")


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
        session._renderer._update_facet_fit_overview((failure, *result.overlays[1:]))
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


def test_facet_table_publishes_mixed_success_and_explicit_error_validity() -> None:
    session = PlotSession(
        _facet_snapshot(facet_unit="m"),
        _spec(),
        fit_engine=_FailFirstFitEngine(),
    )
    try:
        result = session.fit("gaussian_offset", live=True)
        assert isinstance(result, FacetFitBatchResult)
        table = result.table
        assert np.array_equal(table.success, [False, True])
        assert table.sample_unit == "m"
        for name in table.parameter_names:
            assert np.isnan(table.parameter_values[name][0])
            assert np.isnan(table.parameter_errors[name][0])
            assert not bool(table.parameter_error_validity[name][0])
            assert bool(table.parameter_error_validity[name][1])
    finally:
        session.close()


def test_facet_live_fit_replaces_the_whole_batch_on_new_revision() -> None:
    initial = _facet_snapshot()
    session = PlotSession(initial, _spec())
    try:
        session.fit("gaussian_offset", live=True)
        prepared = session.prepare_live_frame(
            _facet_snapshot(revision=1, scale=1.1)
        ).result(timeout=10.0)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        session.finalize_live_frame(finalization)

        accepted = _wait_for_fit_revision(session, 1)
        assert len(accepted.overlays) == 2
        assert all(overlay.polylines for overlay in accepted.overlays)
        assert session.fit_status == "current"
    finally:
        session.close()
