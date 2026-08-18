"""A box dragged on a real panel becomes a real signal.

Driven through an actual raster host over actual camera frames on an actual
plane, rather than two fakes agreeing with each other.  The whole value of this
module is that it sits between packages written independently, so a test where
both sides are mine proves only that I am self-consistent.

What is asserted:

* a committed box publishes a derived signal, cut from the axes the operator
  actually drew on;
* only numbers cross -- no plot object reaches the runtime;
* a gesture this cannot carry says why, because zlc_plot swallows exceptions
  raised inside an application callback and a raise here would be invisible;
* removing the selector retires what it derived, and closing releases the
  subscription -- a panel that keeps deriving after it is gone derives forever.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_data import (
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema as RawSchema,
    GridTopology,
    OwnedSnapshot,
    PointColumn,
    PointTable as RawPointTable,
    StreamGenerationId,
    ValueSchema,
)
from zlc_runtime.dataset import MonitorCoverage
from zlc_runtime.dataset_output import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime.plane import SignalDataPlane
from zlc_workbench.selection import (
    PlotSelectionSource,
    attach_selection_bridge,
    subscribe_committed_selection,
)
from zlc_workbench.session import ExperimentSession
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


@pytest.fixture
def session(tmp_path):
    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def frames(session):
    """One real shot: the frames signal and the snapshot a panel opens on."""

    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 1, CAMERA_WINDOWS),
        signal_plane=session.signal_plane,
        producer="cm",
    )
    capture = node.prepare()
    session.fire(shots=1)
    result = capture.collect()
    signal = node.signal_key("frames")
    return signal, result.publication.value(signal).snapshot


@pytest.fixture
def image_panel(frames):
    """A panel showing the frames, exactly as the console builds one."""

    plot = pytest.importorskip("zlc_plot")
    _signal, snapshot = frames
    host = plot.RasterPlotHost.from_plot(
        snapshot,
        plot.ImagePlot(
            plot.AxisRef.data("spatial-x"),
            plot.AxisRef.data("spatial-y"),
        ),
    )
    try:
        yield host
    finally:
        host.close()


def _draw_area(host, *, span: tuple[float, float, float, float] = (0.3, 0.3, 0.7, 0.7)) -> None:
    """Drag a box across the panel and let go.

    The real path: press, move, release, delivered as normalized pointer events
    against the exact painted front -- the same three calls the Qt canvas makes.
    Setting the selector through the API instead would skip the gesture
    lifecycle, which is where a commit actually happens.
    """

    front = host.wait_for_front(5.0)
    axes = front.interaction.axes[0]
    left, bottom, right, top = axes.bounds

    def _at(fraction_x: float, fraction_y: float) -> tuple[float, float]:
        return (
            left + (right - left) * fraction_x,
            bottom + (top - bottom) * fraction_y,
        )

    start = _at(span[0], span[1])
    end = _at(span[2], span[3])
    for action, point in (("press", start), ("move", end), ("release", end)):
        host._pointer_event(
            action,
            point[0],
            point[1],
            button=1,
            identity=front.identity,
            axes=axes,
            interaction=front.interaction,
        ).result()


def _commit_x_range(host, low: float, high: float) -> None:
    """Install through the public selector API, then deliver its commit event."""

    operation = host._submit(
        lambda: host._require_session().set_x_selector(
            low, high, display=False, emit_change=False
        ),
        mode=_control_mode(),
    ).result(timeout=15)
    _emit(host, __import__("zlc_plot").SelectionChange.COMMITTED, operation.value)


def test_a_committed_box_publishes_a_signal_cut_from_the_drawn_axes(
    session, frames, image_panel
) -> None:
    signal, _snapshot = frames
    bridge, source = attach_selection_bridge(
        session.signal_plane, image_panel, signal, bridge_id="panel-1"
    )
    try:
        _draw_area(image_panel)
        assert source.last_error is None, source.last_error
        assert bridge.selection is not None, bridge.last_error
        assert [entry.axis for entry in bridge.selection.ranges] == [
            "spatial-x",
            "spatial-y",
        ]
        assert bridge.signal_keys(), f"a committed box derived nothing: {bridge.last_error}"
    finally:
        bridge.close()
        source.close()


def test_only_numbers_cross_the_boundary(image_panel) -> None:
    """An axis name and two bounds.  No plot object reaches the runtime."""

    source = PlotSelectionSource(image_panel)
    seen: list = []
    source.subscribe_selection(lambda _change, state: seen.append(state))
    try:
        _draw_area(image_panel)
        assert seen, f"a committed box reported nothing: {source.last_error}"
        state = seen[-1]
        assert state.plot_kind == "image"
        assert state.selector_kind == "area"
        for entry in state.ranges:
            assert isinstance(entry.axis, str)
            assert isinstance(entry.lower, float) and isinstance(entry.upper, float)
    finally:
        source.close()


def test_an_unfinished_drag_derives_nothing(image_panel) -> None:
    """The operator's answer is where they let go, not everywhere they passed.

    Deriving from every intermediate position would publish a stream of answers
    nobody asked for and make the plane's history unreadable.
    """

    from zlc_plot._session_state import SelectionChange
    from zlc_plot.selectors import (
        NumericRange,
        RectangleRange,
        SelectorKind,
        SelectorState,
    )
    from zlc_runtime.selection_bridge import SelectionChange as RuntimeChange

    source = PlotSelectionSource(image_panel)
    seen: list = []
    source.subscribe_selection(lambda change, _state: seen.append(change))
    try:
        state = SelectorState(
            SelectorKind.AREA,
            RectangleRange(NumericRange(2.0, 8.0), NumericRange(1.0, 6.0)),
            revision=1,
        )
        for change in (SelectionChange.ADDED, SelectionChange.UPDATED):
            _emit(image_panel, change, state)
        assert seen == []
        _emit(image_panel, SelectionChange.COMMITTED, state)
        assert seen == [RuntimeChange.COMMITTED]
    finally:
        source.close()


def test_the_state_the_bridge_re_reads_tracks_an_unfinished_drag(image_panel) -> None:
    """Intermediate positions do not derive, but they are not ignored either.

    The bridge re-reads the current selector when a commit arrives and refuses
    a disagreement.  If this only tracked commits, that re-read would answer
    with a stale box after any drag that moved.
    """

    from zlc_plot._session_state import SelectionChange
    from zlc_plot.selectors import (
        NumericRange,
        RectangleRange,
        SelectorKind,
        SelectorState,
    )

    source = PlotSelectionSource(image_panel)
    source.subscribe_selection(lambda *_args: None)
    try:
        moved = SelectorState(
            SelectorKind.AREA,
            RectangleRange(NumericRange(3.0, 9.0), NumericRange(2.0, 7.0)),
            revision=2,
        )
        _emit(image_panel, SelectionChange.UPDATED, moved)
        current = source.selector_data("area")
        assert (current.ranges[0].lower, current.ranges[0].upper) == (3.0, 9.0)
    finally:
        source.close()


def test_a_gesture_that_cannot_be_carried_says_why(frames) -> None:
    """A histogram's bounds cut values, so there is no axis to select on.

    zlc_plot swallows exceptions raised inside an application callback so that
    one bad observer cannot disable the others.  A raise here would therefore
    be invisible, and the operator would see a box that derives nothing and
    explains nothing.
    """

    plot = pytest.importorskip("zlc_plot")
    _signal, snapshot = frames
    host = plot.RasterPlotHost.from_plot(snapshot, plot.HistogramPlot())
    source = PlotSelectionSource(host)
    source.subscribe_selection(lambda *_args: None)
    try:
        host.set_x_selector(1.0, 3.0).result()
        assert source.last_error is not None
        assert "upstream axis" in str(source.last_error)
    finally:
        source.close()
        host.close()


def test_a_point_selector_is_a_readout_not_an_error(image_panel) -> None:
    """A crosshair or threshold marks a point: it derives nothing AND is no
    failure.

    These used to be recorded into ``last_error`` on every click, so the
    console painted a panel error each time an operator read a value off the
    plot -- the user-visible "crosshair has no data".
    """

    source = PlotSelectionSource(image_panel)
    derived: list = []
    source.subscribe_selection(lambda _change, state: derived.append(state))
    try:
        image_panel.set_threshold_selector(2.0).result()
        image_panel.set_crosshair_selector(2.0, 3.0).result()
        assert derived == []
        assert source.last_error is None
    finally:
        source.close()


def test_removing_the_selector_retires_what_it_derived(
    session, frames, image_panel
) -> None:
    from zlc_plot.selectors import SelectorKind

    signal, _snapshot = frames
    bridge, source = attach_selection_bridge(
        session.signal_plane, image_panel, signal, bridge_id="panel-1"
    )
    try:
        _draw_area(image_panel)
        assert bridge.signal_keys()
        image_panel.remove_selector(SelectorKind.AREA).result()
        assert bridge.selection is None
        assert not bridge.signal_keys()
    finally:
        bridge.close()
        source.close()


def test_closing_releases_every_subscription(image_panel) -> None:
    """A removed panel that keeps deriving derives forever."""

    source = PlotSelectionSource(image_panel)
    seen: list = []
    source.subscribe_selection(lambda *args: seen.append(args))
    source.close()
    _draw_area(image_panel)
    assert seen == []


def test_this_module_decides_no_physics() -> None:
    """The rule that keeps this file in the right package.

    Publishing belongs to the runtime and meaning belongs to the domain; this
    carries a gesture across and nothing else.  A calculation appearing here is
    a calculation in the wrong package.
    """

    import ast

    import zlc_workbench.selection as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"mean", "sum", "fit", "calibrate", "detect", "std", "median"}


def _emit(host, change, state) -> None:
    """Deliver one selector lifecycle event the way the plot itself does."""

    host._submit(
        lambda: host._require_session()._emit_selection(change, state),
        mode=_control_mode(),
    ).result()


def _control_mode():
    from zlc_plot.raster import _DispatchMode

    return _DispatchMode.CONTROL


def test_every_derived_signal_can_be_drawn_by_the_panel_that_derived_it(
    session, frames, image_panel
) -> None:
    """A box that publishes a signal nothing can plot is a dead end.

    The console offers every published signal in its card picker, so the picker
    is a promise: what it lists, Add Panel must be able to draw.  Panels used to
    be built as an ImagePlot against two named spatial axes no matter what the
    signal was, so pointing a card at an ROI total or a fit centre raised out of
    the Qt slot that asked for it.  The kind now comes from the signal's own
    schema, and this walks everything one real gesture publishes.
    """

    plot = pytest.importorskip("zlc_plot")
    signal, _snapshot = frames
    bridge, source = attach_selection_bridge(
        session.signal_plane, image_panel, signal, bridge_id="panel-1"
    )
    try:
        _draw_area(image_panel)
        derived = tuple(bridge.signal_keys())
        assert derived, f"a committed box derived nothing: {bridge.last_error}"

        frozen = session.signal_plane.freeze()
        undrawable = []
        for key in derived:
            value = frozen.value(key)
            if value is None:
                continue
            if plot.fitting_spec(value.snapshot.block.schema) is None:
                undrawable.append(key)
        assert not undrawable, (
            f"published but unplottable: {undrawable}"
        )
    finally:
        bridge.close()
        source.close()


def test_frozen_editor_subscription_reports_commits_without_a_runtime_bridge(
    image_panel,
) -> None:
    """Panel Edit translates one answer and owns only a host subscription."""

    seen: list = []
    source = subscribe_committed_selection(image_panel, seen.append)
    try:
        _draw_area(image_panel)
        assert len(seen) == 1
        assert seen[0].selector_kind == "area"
    finally:
        source.close()


# --------------------------------------------------------- facet cell gestures
#
# Everything below drives REAL pointer gestures on facet-grid hosts over raw
# runtime snapshots -- the scan-heatmap chain that used to die silently with
# "image area axes must be source data axes", and the curve cells whose every
# box was refused because a curve's y is the measured value, not an axis.


@dataclass
class _Source:
    declaration: DatasetOutputDeclaration

    instance_id = "camera"

    @property
    def dataset_output_declarations(self):
        return (self.declaration,)

    def signal_key(self, name: str) -> str:
        return f"camera/{name}"


class _Slot:
    notification_failure = None

    def __init__(self, state: dict) -> None:
        self.state = state

    def freeze_live_outputs(self):
        return dict(self.state)

    def close(self) -> None:
        return None


def _heatmap_snapshot(repeats: int = 2) -> OwnedSnapshot:
    """A 3x3 scan grid: scalar cells over dimensions (bias_x, grad)."""

    repeat = AxisSpec(
        AxisId("repeat"), "repeat", REPEAT, repeats, tuple(range(repeats))
    )
    bias = PointColumn(
        AxisId("bias_x"),
        "bias_x",
        SCAN_POINT,
        PointColumn.NUMERIC,
        (-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    )
    grad = PointColumn(
        AxisId("grad"),
        "grad",
        SCAN_POINT,
        PointColumn.NUMERIC,
        (10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0),
    )
    topology = GridTopology(
        (AxisId("bias_x"), AxisId("grad")),
        ((-1.0, 0.0, 1.0), (10.0, 20.0, 30.0)),
        tuple((i, j) for i in range(3) for j in range(3)),
    )
    schema = RawSchema(
        repeat,
        RawPointTable(9, (bias, grad)),
        topology,
        ValueSchema.scalar(np.dtype("float64"), None),
    )
    values = np.arange(repeats * 9, dtype=np.float64).reshape(repeats, 9, 1)
    block = DataBlock(
        BlockId("scan-1"),
        DatasetRevision(1),
        values,
        CellValidity(np.ones((repeats, 9), dtype=np.bool_)),
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("scan-generation")), block)


def _curve_scan_snapshot(repeats: int = 2) -> OwnedSnapshot:
    """A 1D scan: scalar cells over one point coordinate, repeats stacked."""

    repeat = AxisSpec(
        AxisId("repeat"), "repeat", REPEAT, repeats, tuple(range(repeats))
    )
    detuning = PointColumn(
        AxisId("detuning"),
        "detuning",
        SCAN_POINT,
        PointColumn.NUMERIC,
        (0.0, 1.0, 2.0, 3.0, 4.0),
    )
    schema = RawSchema(
        repeat,
        RawPointTable(5, (detuning,)),
        None,
        ValueSchema.scalar(np.dtype("float64"), None),
    )
    values = np.arange(repeats * 5, dtype=np.float64).reshape(repeats, 5, 1)
    block = DataBlock(
        BlockId("curve-1"),
        DatasetRevision(1),
        values,
        CellValidity(np.ones((repeats, 5), dtype=np.bool_)),
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("curve-generation")), block)


def _plane_for(snapshot: OwnedSnapshot, front_signals: set[str]):
    declaration = DatasetOutputDeclaration("frame", "test.camera.frame")
    total = (
        snapshot.block.schema.repeat_axis.size
        * snapshot.block.schema.point_table.row_count
    )
    source_node = _Source(declaration)
    slot = _Slot(
        {"frame": LiveDatasetOutput(declaration, snapshot, MonitorCoverage(total, total))}
    )
    plane = SignalDataPlane()
    plane.reserve(source_node)
    plane.attach(source_node, slot)
    plane.mark_changed(source_node, slot)
    plane.freeze()
    plane.set_front_signals({"camera/frame", *front_signals})
    return plane, source_node


def _gesture_area(host, front, transform, *, span=(0.25, 0.25, 0.75, 0.75)) -> None:
    """Press, move, release across a fraction of one painted transform."""

    left, top, right, bottom = transform.bounds

    def _at(fraction_x: float, fraction_y: float) -> tuple[float, float]:
        return (
            left + fraction_x * (right - left),
            top + fraction_y * (bottom - top),
        )

    start = _at(span[0], span[1])
    end = _at(span[2], span[3])
    for action, point in (("press", start), ("move", end), ("release", end)):
        host._pointer_event(
            action,
            point[0],
            point[1],
            button=1,
            identity=front.identity,
            axes=transform,
            interaction=front.interaction,
        ).result(timeout=15)


def _focused_cell_front(host, cell_index: int):
    """Focus one facet cell the way a double-click does, on the real front."""

    host.wait_for_front(20.0)
    host.focus_facet(cell_index).result(timeout=15)
    front = host.wait_for_front(20.0)
    transform = next(
        item for item in front.interaction.axes if item.role == "facet_cell"
    )
    return front, transform


def _wait_published(plane, name: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while True:
        value = plane.freeze().value(name)
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{name} was never published")
        time.sleep(0.02)


def test_a_box_on_a_focused_scan_heatmap_cell_publishes_the_focused_subgrid() -> None:
    """The headline chain: heatmap facet cell -> gesture -> derived signal.

    The auto default for a repeated 2D scalar scan is a facet grid of
    scan-heatmap image cells whose axes are grid-topology DIMENSIONS.  Every
    committed box on one used to die silently in the bridge with 'image area
    axes must be source data axes' -- nothing published, nothing said.  And
    the focused cell's identity used to be dropped, so the cut spanned every
    repeat instead of the one on screen.
    """

    plot = pytest.importorskip("zlc_plot")
    from zlc_plot._kinds.facet_grid import default_spec as facet_default_spec

    snapshot = _heatmap_snapshot()
    schema = snapshot.block.schema
    values = snapshot.block.values
    spec = facet_default_spec(schema)
    assert spec is not None and isinstance(spec.cell, plot.ImagePlot), spec

    plane, source_node = _plane_for(
        snapshot, {"@logic/panel-1/roi_frame", "@logic/panel-1/roi_mean"}
    )
    host = plot.RasterPlotHost.from_plot(snapshot, spec, size="4x4")
    bridge = source = None
    try:
        bridge, source = attach_selection_bridge(
            plane, host, "camera/frame", bridge_id="panel-1"
        )
        front, transform = _focused_cell_front(host, 1)
        _gesture_area(host, front, transform)

        assert source.last_error is None, source.last_error
        assert bridge.last_error is None, bridge.last_error
        selection = bridge.selection
        assert selection is not None
        # The focused cell crosses structurally: repeat row 1, no named axis.
        assert selection.repeat_index == 1
        assert selection.facets == ()

        # The derived data equals the slice the committed ranges + focused
        # repeat select from the source, computed independently here.
        columns = {
            column.name: column.values for column in schema.point_table.columns
        }
        rows = [
            row
            for row in range(schema.point_table.row_count)
            if all(
                entry.lower <= columns[entry.axis][row] <= entry.upper
                for entry in selection.ranges
            )
        ]
        assert 1 <= len(rows) < schema.point_table.row_count, rows
        expected = values[1:2][:, rows]

        roi_frame = _wait_published(plane, "@logic/panel-1/roi_frame")
        roi_mean = _wait_published(plane, "@logic/panel-1/roi_mean")
        np.testing.assert_array_equal(roi_frame.snapshot.block.values, expected)
        assert roi_frame.snapshot.block.schema.repeat_axis.size == 1
        # roi_mean is ONE value per (repeat, point): the reduction consumes the
        # image axes and nothing else.  A square cell makes the same fractional
        # drag cover several scan points, so the comparison must be per point --
        # collapsing it to a single number only ever held for a one-point ROI.
        np.testing.assert_array_equal(
            roi_mean.snapshot.block.values.reshape(-1),
            np.mean(expected, axis=-1).reshape(-1),
        )
    finally:
        if bridge is not None:
            bridge.close()
        if source is not None:
            source.close()
        host.close()
        plane.retire(source_node)
        plane.close()


def test_an_area_on_a_plain_curve_panel_derives_an_x_range() -> None:
    """A box on a curve translates to the interval on its one named axis.

    The y of a curve is the measured VALUE, not an axis; building a y range
    anyway made every box unbridgeable.  One named axis is a description,
    not a failure.
    """

    plot = pytest.importorskip("zlc_plot")

    snapshot = _curve_scan_snapshot()
    host = plot.RasterPlotHost.from_plot(
        snapshot, plot.CurvePlot(x=plot.AxisRef.point("detuning"))
    )
    source = PlotSelectionSource(host)
    committed: list = []
    source.subscribe_selection(lambda _change, state: committed.append(state))
    try:
        front = host.wait_for_front(20.0)
        transform = next(
            item for item in front.interaction.axes if item.role in ("main", "image")
        )
        _gesture_area(host, front, transform)
        assert source.last_error is None, source.last_error
        assert committed, "a committed box reported nothing"
        state = committed[-1]
        assert state.plot_kind == "curve"
        assert state.selector_kind == "x_range"
        assert [entry.axis for entry in state.ranges] == ["detuning"]
        assert state.facets == () and state.repeat_index is None
    finally:
        source.close()
        host.close()


def test_structural_curve_axes_select_the_source_rows_without_a_fake_name() -> None:
    """Repeat and point-row axes are structural, not unnamed failures."""

    plot = pytest.importorskip("zlc_plot")
    snapshot = _curve_scan_snapshot(repeats=4)
    values = snapshot.block.values
    cases = (
        (plot.AxisRef.repeat(), "repeat", values[1:3, :, :]),
        (plot.AxisRef.point_rows(), "point_row", values[:, 1:3, :]),
    )

    for axis, domain, expected in cases:
        plane, source_node = _plane_for(
            snapshot, {"@logic/panel-1/roi_mean"}
        )
        host = plot.RasterPlotHost.from_plot(snapshot, plot.CurvePlot(axis))
        bridge = source = None
        try:
            bridge, source = attach_selection_bridge(
                plane, host, "camera/frame", bridge_id="panel-1"
            )
            host.wait_for_front(20.0)
            viewports: list = []
            source.subscribe_viewport(
                lambda state, _display: viewports.append(state)
            )
            host.set_viewport(
                plot.NumericRange(1.0, 2.0),
                plot.NumericRange(-1.0, 20.0),
            ).result(timeout=15)
            assert viewports and viewports[-1].ranges[0].domain == domain
            _commit_x_range(host, 1.0, 2.0)

            assert source.last_error is None, source.last_error
            assert bridge.last_error is None, bridge.last_error
            assert bridge.selection is not None
            assert bridge.selection.ranges[0].domain == domain
            derived = _wait_published(plane, "@logic/panel-1/roi_mean")
            np.testing.assert_array_equal(derived.snapshot.block.values, expected)
        finally:
            if bridge is not None:
                bridge.close()
            if source is not None:
                source.close()
            host.close()
            plane.retire(source_node)
            plane.close()


def test_rolling_history_is_not_claimed_as_an_upstream_point_axis() -> None:
    """Rolling x belongs to plot history, so it derives nothing and is no error."""

    plot = pytest.importorskip("zlc_plot")
    snapshot = _curve_scan_snapshot()
    host = plot.RasterPlotHost.from_plot(snapshot, plot.RollingPlot())
    source = PlotSelectionSource(host)
    seen: list = []
    viewports: list = []
    source.subscribe_selection(lambda *_args: seen.append(True))
    source.subscribe_viewport(lambda *_args: viewports.append(True))
    try:
        host.wait_for_front(20.0)
        host.set_viewport(
            plot.NumericRange(0.0, 1.0),
            plot.NumericRange(-1.0, 10.0),
        ).result(timeout=15)
        _commit_x_range(host, 0.0, 1.0)
        assert seen == []
        assert viewports == []
        assert source.last_error is None
    finally:
        source.close()
        host.close()


def test_an_area_on_a_focused_curve_facet_cell_carries_the_cell() -> None:
    """The same translation inside a focused facet cell keeps the cell."""

    plot = pytest.importorskip("zlc_plot")

    snapshot = _curve_scan_snapshot()
    spec = plot.FacetGridPlot(
        plot.AxisRef.repeat(), plot.CurvePlot(x=plot.AxisRef.point("detuning"))
    )
    host = plot.RasterPlotHost.from_plot(snapshot, spec, size="4x4")
    source = PlotSelectionSource(host)
    committed: list = []
    source.subscribe_selection(lambda _change, state: committed.append(state))
    try:
        front, transform = _focused_cell_front(host, 1)
        _gesture_area(host, front, transform)
        assert source.last_error is None, source.last_error
        assert committed, "a committed box reported nothing"
        state = committed[-1]
        assert state.selector_kind == "x_range"
        assert [entry.axis for entry in state.ranges] == ["detuning"]
        assert state.repeat_index == 1
        assert state.facets == ()
    finally:
        source.close()
        host.close()


def test_a_box_on_a_focused_frame_cell_derives_only_that_frame(
    session, frames
) -> None:
    """Frames on the point axis: the focused frame's identity crosses.

    A camera cycle publishes its frames as POINT rows with a 'frame' column,
    and its auto default facets those frames side by side.  A box drawn on
    one focused frame must derive that frame's region only -- the frame is a
    named point coordinate, so unlike a repeat facet it crosses as a facet
    condition on its own axis.
    """

    plot = pytest.importorskip("zlc_plot")
    from zlc_plot._kinds.facet_grid import default_spec as facet_default_spec

    signal, snapshot = frames
    schema = snapshot.block.schema
    assert schema.point_table.row_count == CAMERA_WINDOWS
    frame_column = schema.point_table.columns[0]
    assert frame_column.name == "frame"
    spec = facet_default_spec(schema)
    assert spec is not None and isinstance(spec.cell, plot.ImagePlot), spec

    host = plot.RasterPlotHost.from_plot(snapshot, spec, size="4x4")
    bridge = source = None
    try:
        bridge, source = attach_selection_bridge(
            session.signal_plane, host, signal, bridge_id="panel-1"
        )
        front, transform = _focused_cell_front(host, 1)
        _gesture_area(host, front, transform)

        assert source.last_error is None, source.last_error
        assert bridge.last_error is None, bridge.last_error
        selection = bridge.selection
        assert selection is not None
        assert selection.repeat_index is None
        assert [facet.axis for facet in selection.facets] == [
            frame_column.coordinate_id.value
        ]
        assert selection.facets[0].value == 1.0

        roi_frame = _wait_published(
            session.signal_plane, "@logic/panel-1/roi_frame"
        )
        derived = roi_frame.snapshot.block
        assert derived.schema.point_table.row_count == 1
        derived_column = derived.schema.point_table.column(
            frame_column.coordinate_id
        )
        assert tuple(derived_column.values) == (frame_column.values[1],)

        # The derived pixels are the same crop of frame 1, located through
        # the derived axes' own origins.
        starts = tuple(
            axis.index_origin
            if axis.coordinates is None
            else int(axis.coordinates[0])
            for axis in derived.schema.cell_schema.data_axes
        )
        sizes = tuple(
            axis.size for axis in derived.schema.cell_schema.data_axes
        )
        assert all(
            0 < size < full.size
            for size, full in zip(sizes, schema.cell_schema.data_axes)
        )
        window = tuple(
            slice(start, start + size) for start, size in zip(starts, sizes)
        )
        np.testing.assert_array_equal(
            derived.values,
            snapshot.block.values[(slice(None), slice(1, 2), *window)],
        )
    finally:
        if bridge is not None:
            bridge.close()
        if source is not None:
            source.close()
        host.close()
