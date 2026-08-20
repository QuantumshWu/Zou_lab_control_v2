"""A selection event says which upstream quantities its bounds cut.

Bounds alone are not actionable.  "x from 1 to 5" is only a statement about
data once you know what x is, and x is a property of the projection, not of the
selector: a histogram's x bounds cut the value quantity while a curve's cut the
x coordinate, and a semantic edit moves either one while the operator works.

A consumer could ask the session afterwards, but that answer belongs to
whatever the projection is by then -- which may not be what was drawn on.  That
failure is silent and produces a plausible number from the wrong axis, so the
resolved subject travels with the event.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zlc_data import CoordinateFrameId, owned_snapshot_from_arrays
from zlc_plot import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotKind,
    PlotSession,
)
from zlc_plot.kinds import AxisRef
from zlc_plot.selectors import NumericRange, SelectorKind
from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


def _curve_snapshot() -> object:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=3),
        PointTable.from_columns({"detuning": np.arange(5.0)}),
        dtype=np.float64,
        generation="selection-subject",
    )
    return DatasetSnapshot(schema, np.arange(15.0).reshape(3, 5), revision=0)


def _image_snapshot() -> object:
    columns = np.linspace(0.0, 5.0, 6)
    rows = np.linspace(0.0, 3.0, 4)
    pixel_frame = CoordinateFrameId("camera-pixel")
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": [0.0]}),
        data_axes=(
            replace(
                Axis.create("column", values=columns),
                coordinate_frame=pixel_frame,
            ),
            replace(
                Axis.create("row", values=rows),
                coordinate_frame=pixel_frame,
            ),
        ),
        dtype=np.float64,
        generation="selection-subject",
    )
    return DatasetSnapshot(schema, np.arange(6 * 4.0).reshape(1, 1, 6, 4), revision=0)


def _events(session: PlotSession, apply) -> list:
    events: list = []
    release = session.subscribe_selection(events.append)
    try:
        apply()
    finally:
        release()
    assert events, "no selection event was emitted"
    return events


def _subjects(session: PlotSession, apply) -> list:
    return [event.subject for event in _events(session, apply)]


def _named_facet_snapshot() -> object:
    detuning = np.tile(np.arange(5.0), 2)
    site = np.repeat((10.0, 20.0), 5)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=3),
        PointTable.from_columns({"detuning": detuning, "site": site}),
        dtype=np.float64,
        generation="selection-subject-facet",
    )
    return owned_snapshot_from_arrays(
        schema,
        np.arange(30.0).reshape(3, 10, 1),
        revision=7,
        stream_generation="selection-subject-run",
    )


def _threshold_facet_snapshot(*, inserted_cell: bool = False) -> object:
    samples = np.linspace(-3.0, 3.0, 80)
    columns = (
        np.where(samples < 0.0, samples - 2.0, samples + 2.0),
        np.where(samples < 0.0, samples - 1.0, samples + 3.0),
        np.where(samples < 0.0, samples - 3.0, samples + 1.0),
    )
    sites = (10.0, 20.0, 5.0)
    order = (2, 0, 1) if inserted_cell else (0, 1)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=len(samples)),
        PointTable.from_columns(
            {"site": tuple(sites[index] for index in order)}
        ),
        dtype=np.float64,
        generation=f"threshold-target-{int(inserted_cell)}",
    )
    return DatasetSnapshot(
        schema,
        np.column_stack(tuple(columns[index] for index in order)),
        revision=0,
    )


def test_a_curve_names_the_axis_its_x_bounds_cut() -> None:
    session = PlotSession(_curve_snapshot(), CurvePlot(x=AxisRef.point("detuning")))
    try:
        subject = _subjects(
            session,
            lambda: session.set_x_selector(1.0, 3.0),
        )[-1]
        assert subject.plot_kind is PlotKind.CURVE
        assert subject.x == AxisRef.point("detuning")
        # A curve's y is the measured value, not an axis anyone can slice by.
        assert subject.y is None
    finally:
        session.close()


def test_an_image_names_both_axes_its_area_cuts() -> None:
    session = PlotSession(
        _image_snapshot(),
        ImagePlot(AxisRef.data("column"), AxisRef.data("row")),
    )
    try:
        subject = _subjects(
            session,
            lambda: session.set_area_selector(NumericRange(1.0, 3.0), NumericRange(0.0, 2.0)),
        )[-1]
        assert subject.plot_kind is PlotKind.IMAGE
        assert subject.x == AxisRef.data("column")
        assert subject.y == AxisRef.data("row")
        assert subject.x_coordinate_frame == "camera-pixel"
        assert subject.y_coordinate_frame == "camera-pixel"
    finally:
        session.close()


def test_a_histogram_reports_no_axis_because_its_bounds_cut_values() -> None:
    """The case that makes re-asking the session wrong instead of merely late.

    The same gesture on the same data means a different thing here, and the
    honest answer is that there is no upstream axis to name.
    """

    session = PlotSession(_curve_snapshot(), HistogramPlot())
    try:
        subject = _subjects(
            session,
            lambda: session.set_x_selector(1.0, 3.0),
        )[-1]
        assert subject.plot_kind is PlotKind.HISTOGRAM
        assert subject.x is None
        assert subject.y is None
    finally:
        session.close()


def test_the_subject_follows_a_semantic_edit_within_one_session() -> None:
    """Which is the whole point: it is resolved per event, not once."""

    session = PlotSession(_curve_snapshot(), CurvePlot(x=AxisRef.point("detuning")))
    try:
        first = _subjects(session, lambda: session.set_x_selector(1.0, 3.0))[-1]
        assert first.x == AxisRef.point("detuning")
        session.replace_spec(HistogramPlot())
        second = _subjects(session, lambda: session.set_x_selector(1.0, 3.0))[-1]
        assert second.plot_kind is PlotKind.HISTOGRAM
        assert second.x is None
    finally:
        session.close()


def test_every_selector_kind_carries_a_subject() -> None:
    """Including the ones that cannot be bridged, which must still say so."""

    session = PlotSession(_curve_snapshot(), CurvePlot(x=AxisRef.point("detuning")))
    try:
        for subject in _subjects(
            session,
            lambda: (
                session.set_x_selector(1.0, 3.0),
                session.set_threshold_selector(2.0),
                session.set_crosshair_selector(1.0, 2.0),
            ),
        ):
            assert subject is not None
            assert subject.plot_kind is PlotKind.CURVE
        assert SelectorKind.THRESHOLD in {state.kind for state in session.selectors}
    finally:
        session.close()

@pytest.mark.parametrize("focused", (False, True), ids=("panel-scope", "focused"))
@pytest.mark.parametrize("named", (False, True), ids=("repeat", "named"))
def test_panel_and_focused_scope_carry_canonical_event_meaning(
    focused: bool,
    named: bool,
) -> None:
    snapshot = _named_facet_snapshot() if named else _curve_snapshot()
    scope_axis = AxisRef.point("site") if named else AxisRef.repeat()
    scope_value = 20.0 if named else 1.0
    cell = CurvePlot(x=AxisRef.point("detuning"))
    spec = (
        FacetGridPlot(scope_axis, cell)
        if focused
        else replace(cell, scope=((scope_axis, scope_value),))
    )
    session = PlotSession(snapshot, spec)
    try:
        if focused:
            session.focus_facet(1)
        event = _events(
            session,
            lambda: session.set_x_selector(1.0, 3.0, display=False),
        )[-1]
        assert event.subject.scope == (
            ((AxisRef.point("site"), 20.0),) if named else ()
        )
        assert event.subject.repeat_index == (None if named else 1)
        if named and not focused:
            assert event.data_revision == 7
            assert event.data_generation == "selection-subject-run"
            assert event.selector.value == NumericRange(1.0, 3.0)
            assert event.display_selector.value == NumericRange(1.0, 3.0)
    finally:
        session.close()


def test_classifier_threshold_targets_follow_coordinates_when_facets_reorder() -> None:
    spec = FacetGridPlot(AxisRef.point("site"), HistogramPlot())
    authored = PlotSession(_threshold_facet_snapshot(), spec)
    try:
        authored.configure(parameters={"threshold_classifier": True})
        authored.focus_facet(0)
        event_a = _events(
            authored,
            lambda: authored.set_threshold_selector(-0.25, display=False),
        )
        target_a = event_a[-1].classifier_thresholds[0]
        authored.focus_facet(1)
        event_b = _events(
            authored,
            lambda: authored.set_threshold_selector(0.75, display=False),
        )
        target_b = next(
            target
            for target in event_b[-1].classifier_thresholds
            if target["scope"][0]["coordinate"] == 20.0
        )
        assert target_a["scope"] == (
            {
                "domain": "point_coordinate",
                "axis_id": "site",
                "coordinate": 10.0,
            },
        )
        assert target_b["scope"][0]["coordinate"] == 20.0
        assert event_b[-1].classifier_thresholds == (target_a, target_b)
    finally:
        authored.close()

    restored = PlotSession(_threshold_facet_snapshot(inserted_cell=True), spec)
    try:
        restored.configure(
            parameters={"threshold_classifier": True},
            classifier_thresholds=(target_a, target_b),
        )
        restored.focus_facet(1)
        assert restored.selector_state(SelectorKind.THRESHOLD).value == -0.25
        restored.focus_facet(2)
        assert restored.selector_state(SelectorKind.THRESHOLD).value == 0.75
    finally:
        restored.close()


def test_repeat_facet_threshold_target_uses_the_source_repeat_row() -> None:
    samples = np.linspace(-3.0, 3.0, 80)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns({"sample": samples}),
        dtype=np.float64,
        generation="repeat-threshold-target",
    )
    values = np.vstack(
        (
            np.where(samples < 0.0, samples - 2.0, samples + 2.0),
            np.where(samples < 0.0, samples - 1.0, samples + 3.0),
        )
    )
    session = PlotSession(
        DatasetSnapshot(schema, values, revision=0),
        FacetGridPlot(AxisRef.repeat(), HistogramPlot()),
    )
    try:
        session.configure(parameters={"threshold_classifier": True})
        session.focus_facet(1)
        target = _events(
            session,
            lambda: session.set_threshold_selector(0.5, display=False),
        )[-1].classifier_thresholds[0]
        assert target == {"value": 0.5, "scope": (), "repeat_index": 1}
    finally:
        session.close()
