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
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_workbench.selection import (
    PlotSelectionSource,
    attach_selection_bridge,
    subscribe_committed_selection,
)
from zlc_workbench.session import ExperimentSession

ATOM_ROOT = Path(__file__).resolve().parents[2] / "zlc_atom"


@pytest.fixture
def session(tmp_path):
    pulses = tmp_path / "pulses"
    pulses.mkdir()
    shutil.copy(ATOM_ROOT / "pulses" / "calibration.py", pulses / "calibration.py")
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def frames(session):
    """One real shot: the frames signal and the snapshot a panel opens on."""

    pulse = session.load_pulse("calibration")
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest(
            "camera", 0.02, None, 1, int(pulse["camera_windows"]), 2.0
        ),
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


def test_a_point_selector_reports_rather_than_deriving_nonsense(image_panel) -> None:
    """A threshold marks a value, not a region.  It has nothing to slice with."""

    source = PlotSelectionSource(image_panel)
    derived: list = []
    source.subscribe_selection(lambda _change, state: derived.append(state))
    try:
        image_panel.set_threshold_selector(2.0).result()
        assert derived == []
        assert "point" in str(source.last_error)
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
