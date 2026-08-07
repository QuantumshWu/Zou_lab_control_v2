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

import numpy as np

from zlc_plot import CurvePlot, HistogramPlot, ImagePlot, PlotKind, PlotSession
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
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": [0.0]}),
        data_axes=(
            Axis.create("column", values=columns),
            Axis.create("row", values=rows),
        ),
        dtype=np.float64,
        generation="selection-subject",
    )
    return DatasetSnapshot(schema, np.arange(6 * 4.0).reshape(1, 1, 6, 4), revision=0)


def _subjects(session: PlotSession, apply) -> list:
    events: list = []
    release = session.subscribe_selection(events.append)
    try:
        apply()
    finally:
        release()
    assert events, "no selection event was emitted"
    return [event.subject for event in events]


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
