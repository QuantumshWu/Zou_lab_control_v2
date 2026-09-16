"""The window mask and the rolling trace read ONE history layout.

"Which rows belong to the last N shots" used to be answered twice: the
rolling trace mapped the shot column through the domain machinery while
every histogram and facet path walked the rows in Python and asked numpy
to compare objects.  Both now read the schema's indexed-history layout, so
a grid of site histograms selects exactly the shots the rolling trace draws.
"""

from __future__ import annotations

import numpy as np
from zlc_data import (
    PRIMARY_INDEX,
    READOUT_EVENT,
    REPEAT,
    SITE,
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from zlc_plot import AxisRef
from zlc_plot.data_view import DataView

SITES = 3
FRAMES = 2


def _history(shots: int) -> DataView:
    """A Runtime-shaped history: shots x frames rows, sites in the cell."""

    offsets = tuple(int(offset) for offset in range(-shots + 1, 1))
    primary = AxisSpec(
        PRIMARY_INDEX_AXIS_ID, "source index", PRIMARY_INDEX, shots, offsets
    )
    frame = AxisSpec(
        AxisId("frame"),
        "frame",
        READOUT_EVENT,
        FRAMES,
        tuple(range(FRAMES)),
        coordinate_labels=("before", "after"),
    )
    site = AxisSpec(AxisId("site"), "site", SITE, SITES, tuple(range(SITES)))
    schema = DatasetSchema(
        DomainSpec(
            (1,),
            (AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,)),),
            ((0,),),
        ),
        DomainSpec(
            (shots * FRAMES,),
            (primary, frame),
            (
                tuple(int(code) for code in np.repeat(np.arange(shots), FRAMES)),
                tuple(range(FRAMES)) * shots,
            ),
        ),
        DomainSpec((SITES,), (site,)),
        ValueSchema(
            ValidityContract.components(AxisId("site")), np.dtype("<f8"), "count"
        ),
    )
    values = np.arange(shots * FRAMES * SITES, dtype=np.float64).reshape(
        1, shots * FRAMES, SITES
    )
    return DataView(owned_snapshot_from_arrays(schema, values, 1, stream_generation="g"))


def test_the_window_mask_selects_the_shots_the_rolling_trace_draws() -> None:
    view = _history(5)
    history = view.rolling_history(group=AxisRef.cell_data("site"))
    assert history.source_indices.tolist() == [-4, -3, -2, -1, 0]
    for window in (1, 2, 5, 9):
        mask = view.history_validity(window)
        assert mask.shape == (1, 5 * FRAMES, SITES)
        rows = np.flatnonzero(mask[0, :, 0])
        shots_kept = history.source_indices[-window:]
        # Every row of a kept shot, and only those, is inside the window.
        assert rows.tolist() == [
            row for row in range(5 * FRAMES) if -(5 - 1) + row // FRAMES in shots_kept
        ]
        # And the trace's shot planes are the same rows, reduced per site.
        expected = view.samples.value.canonical[0].reshape(5, FRAMES, SITES).mean(axis=1)
        np.testing.assert_allclose(np.asarray(history.values), expected)


def test_the_mask_is_built_once_per_view_however_many_projections_ask() -> None:
    view = _history(4)
    first = view.history_validity(3)
    assert view.history_validity(3) is first
    assert view.history_validity(2) is not first


def test_a_labelled_point_axis_names_each_distinct_value_from_its_first_row() -> None:
    view = _history(3)
    domain = view._domain(AxisRef.point("frame"), view._all_positions())
    assert [value.canonical for value in domain.values] == [0, 1]
    assert [value.label for value in domain.values] == ["frame=before", "frame=after"]


def test_continuous_records_share_real_sample_coordinates_with_selection_and_figure(tmp_path) -> None:
    from types import SimpleNamespace
    from zlc_data import SAMPLE_TIME, Selection, DatasetRevisionRef, BlockId
    from zlc_data.figure_archive import read_archive
    from zlc_data.snapshot_projection import SHOT_TIME_AXIS_ID, indexed_history_layout, restrict_snapshot
    from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage, SignalDataPlane
    from zlc_runtime.selection_bridge import SelectionBridge
    from zlc_plot import CurvePlot, FacetGridPlot, HistogramPlot, PlotKind, PlotSession, SelectorKind, read_figure_plot, save_figure_artifact
    from zlc_plot._kinds import default_spec
    from zlc_plot.specs import history_window_requirement, parameter_schema_for
    from zlc_plot.config import DEFAULTS
    from zlc_plot import SelectionChange
    from zlc_workbench.selection import PlotSelectionSource, panel_selection_matches_subject

    sample = AxisSpec(AxisId("sample"), "sample time", SAMPLE_TIME, 3, (0.0, 0.1, 0.2), unit="s")
    channel = AxisSpec(AxisId("channel"), "channel", SITE, 2, (0, 1))
    schema = DatasetSchema(
        DomainSpec((1,), (), ()), DomainSpec((3,), (sample,), ((0, 1, 2),)),
        DomainSpec((2,), (channel,)), ValueSchema.scalar(np.dtype("f8"), "V", name="Voltage"),
    )
    declaration = DatasetOutputDeclaration("value", "test.waveform", index_by_source=True)
    node = SimpleNamespace(instance_id="waveform", dataset_output_declarations=(declaration,),
                           signal_key=lambda name: f"waveform/{name}")
    plane = SignalDataPlane()
    lease = None
    session = None
    bridges, sources = [], []
    try:
        plane.begin_generation(node)
        plane.set_front_signals({"waveform/value", "@logic/window_curve/roi_frame", "@logic/window_curve/roi_mean", "@logic/window_hist/roi_frame"})
        lease = plane.acquire_indexed_history("waveform/value", 3)
        for record in range(3):
            values = np.arange(record * 6.0, record * 6.0 + 6).reshape(1, 3, 2)
            event = owned_snapshot_from_arrays(schema, values, record, stream_generation="source")
            plane.commit_live(node, {"value": LiveDatasetOutput(
                declaration, event, MonitorCoverage(3, 3), shot_time_seconds=record * 0.3,
            )})
        snapshot = plane.current_dataset("waveform/value")
        ref = AxisRef.point("sample")
        spec = default_spec(snapshot.block.schema, PlotKind.CURVE)
        assert spec == CurvePlot(ref, group=AxisRef.cell_data("channel"))
        assert history_window_requirement(spec, {"window": 1}) is None
        assert history_window_requirement(spec, {"window": 3}) == 3
        assert parameter_schema_for(FacetGridPlot(None, spec), style=DEFAULTS.style).initial_values()["window"] == 1
        session = PlotSession(snapshot, spec, parameters={"window": 3}, size="2x2")
        source = PlotSelectionSource(session)
        bridge = SelectionBridge(plane, "waveform/value", source, bridge_id="window_curve")
        bridge.start()
        source.subscribe_observation(lambda event: bridge.commit_selection(
            event.state, source_publication=plane.latest_publication("waveform/value"),
        ))
        bridges.append(bridge)
        sources.append(source)
        series = session._payload.series
        assert len(series) == 2
        np.testing.assert_allclose(series[0].x.canonical, np.arange(9) * 0.1)
        np.testing.assert_array_equal(series[0].y.canonical, np.arange(0.0, 18.0, 2))
        session.set_x_selector(0.15, 0.65)
        session._emit_selection(SelectionChange.COMMITTED, session.selector_state(SelectorKind.X_RANGE, display=False))
        assert source.last_error is None
        assert bridge.last_error is None and bridge.last_condition == ""
        np.testing.assert_array_equal(
            session.selector_data(SelectorKind.X_RANGE).canonical_values, np.arange(4.0, 14.0),
        )
        fit_input = session.fit_selection("gaussian_offset", selector_kind=SelectorKind.X_RANGE)
        np.testing.assert_allclose(fit_input.coordinates[0], np.arange(2, 7) * 0.1)
        np.testing.assert_array_equal(plane.current_dataset("@logic/window_curve/roi_frame").block.values.reshape(-1), np.arange(4.0, 14.0))
        second = PlotSession(snapshot, FacetGridPlot(None, spec), parameters={"window": 2}, size="2x2")
        try:
            np.testing.assert_allclose(second._payload.cells[0].payload.series[0].x.canonical, np.arange(3, 9) * 0.1)
            second.set_parameter("window", 1)
            np.testing.assert_allclose(second._payload.cells[0].payload.series[0].x.canonical, np.arange(6, 9) * 0.1)
            assert len(session._payload.series[0].x.canonical) == 9
        finally:
            second.close()
        session.set_parameter("window", 2)
        np.testing.assert_array_equal(session.selector_data(SelectorKind.X_RANGE).canonical_values, np.arange(6.0, 14.0))
        np.testing.assert_array_equal(plane.current_dataset("@logic/window_curve/roi_frame").block.values.reshape(-1), np.arange(6.0, 14.0))
        np.testing.assert_array_equal(plane.current_dataset("@logic/window_curve/roi_mean").block.values.reshape(-1), np.arange(6.5, 14.0, 2))
        session.set_parameter("window", 3)
        histogram = PlotSession(snapshot, HistogramPlot(), parameters={"window": 2}, size="2x2")
        hist_source = PlotSelectionSource(histogram)
        hist_bridge = SelectionBridge(plane, "waveform/value", hist_source, bridge_id="window_hist")
        hist_bridge.start()
        hist_source.subscribe_observation(lambda event: hist_bridge.commit_selection(
            event.state, source_publication=plane.latest_publication("waveform/value"),
        ))
        bridges.append(hist_bridge)
        sources.append(hist_source)
        try:
            histogram.set_x_selector(-1.0, 100.0)
            histogram._emit_selection(SelectionChange.COMMITTED, histogram.selector_state(SelectorKind.X_RANGE, display=False))
            np.testing.assert_array_equal(histogram.selector_data(SelectorKind.X_RANGE).canonical_values, np.arange(6.0, 18.0))
            np.testing.assert_array_equal(plane.current_dataset("@logic/window_hist/roi_frame").block.values.reshape(-1), np.arange(6.0, 18.0))
            histogram.set_parameter("window", 1)
            np.testing.assert_array_equal(plane.current_dataset("@logic/window_hist/roi_frame").block.values.reshape(-1), np.arange(12.0, 18.0))
        finally:
            histogram.close()
        cropped = restrict_snapshot(
            snapshot, Selection.coordinate_range(sample.axis_id, 0.15, 0.65, coordinate_frame=None),
            reference_for=lambda selected: DatasetRevisionRef(
                BlockId("cropped"), snapshot.ref.stream_generation, selected.fingerprint, snapshot.ref.revision,
            ),
        )
        layout = indexed_history_layout(cropped.block.schema)
        assert layout.inner_count is None and layout.codes().tolist() == [0, 1, 1, 1, 2]
        np.testing.assert_allclose(DataView(cropped).curve(ref).series[0].x.canonical, np.arange(2, 7) * 0.1)
        rolling = PlotSession(snapshot, default_spec(snapshot.block.schema, PlotKind.ROLLING),
                              parameters={"window": 3}, size="2x2")
        try:
            events = []
            rolling.subscribe_selection(events.append)
            rolling.set_x_selector(0.15, 0.65)
            selected = PlotSelectionSource._translate(events[-1])
            assert selected.ranges[0].axis == str(SHOT_TIME_AXIS_ID)
            assert selected.ranges[0].domain == "point"
            assert panel_selection_matches_subject(selected, events[-1].subject)
            np.testing.assert_array_equal(
                rolling.selector_data(SelectorKind.X_RANGE).canonical_values, np.arange(6.0, 18.0),
            )
            rolling.set_axis_unit(AxisRef.point(str(SHOT_TIME_AXIS_ID)), "ms")
            np.testing.assert_allclose(rolling._payload.series[0].x.display, [0, 300, 600])
            rolling.set_crosshair_selector(300.0, 8.0, display=True)
            assert rolling.selector_state(SelectorKind.CROSSHAIR, display=False).value.x == 0.3
            assert rolling.selector_state(SelectorKind.CROSSHAIR, display=True).value.x == 300.0
        finally:
            rolling.close()
        image, archive = save_figure_artifact(tmp_path / "continuous", plot_input=snapshot,
                                             spec=spec, parameters={"window": 3}, size="2x2")
        restored, recipe = read_figure_plot(*read_archive(archive), "data")
        assert image.is_file() and recipe["spec"] == spec
        np.testing.assert_allclose(DataView(restored).curve(ref).series[0].x.canonical, np.arange(9) * 0.1)
    finally:
        for source in sources:
            source.close()
        for bridge in bridges:
            bridge.close()
        if session is not None:
            session.close()
        if lease is not None:
            lease.close()
        plane.close()
