from __future__ import annotations

import base64
import json
import struct
import time
from threading import Event

import numpy as np

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)

from test_facet_live_fit import _facet_snapshot, _spec as facet_spec
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    NumericRange,
    PlotSession,
    RasterPlotHost,
    SelectorKind,
)
from zlc_plot.selectors import RectangleRange, SelectorState
from zlc_plot.notebook import (
    _WIDGET_ESM,
    _front_packet,
    _selector_state_to_dict,
    _snapshot_display_data,
    NotebookView,
)

def _session() -> PlotSession:
    x = np.arange(10.0)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": x}),
        dtype=np.float64,
    )
    return PlotSession(
        make_snapshot(schema, x.reshape(1, -1), revision=0),
        CurvePlot(AxisRef.point("x")),
    )

def test_front_selector_state_json_preserves_display_geometry_for_browser_preview() -> None:
    state = SelectorState(
        SelectorKind.AREA,
        RectangleRange(NumericRange(1.0, 2.0), NumericRange(3.0, 4.0)),
        facet_index=2,
    )
    assert _selector_state_to_dict(state) == {
        "kind": "area",
        "value": {"x": [1.0, 2.0], "y": [3.0, 4.0]},
        "facet_index": 2,
    }

def test_notebook_front_packet_accepts_jupyter_dataview_buffers() -> None:
    """The browser adapter must decode one complete DataView frame packet."""

    assert "export default" in _WIDGET_ESM
    assert "render({ model, el })" in _WIDGET_ESM
    assert "ArrayBuffer.isView(value)" in _WIDGET_ESM
    assert "value.byteOffset, value.byteLength" in _WIDGET_ESM
    assert "change:frame_packet" in _WIDGET_ESM
    assert "_decodeFrame" in _WIDGET_ESM
    assert "define(" not in _WIDGET_ESM

def test_notebook_widget_uses_anywidget_registry_not_a_stale_matplotlib_module() -> None:
    """A saved notebook must resolve the current AnyWidget view class.

    Older prototypes emitted ``zlc_plot_raster``/``MPLCanvasView`` metadata.
    JupyterLab keeps that metadata in an output cell, so merely changing the
    kernel renderer cannot repair an already-saved widget view.  The current
    adapter must advertise the standard anywidget registry explicitly.
    """

    from zlc_plot.notebook import _widget_class

    widget = _widget_class()()
    assert widget._model_module == "anywidget"
    assert widget._view_module == "anywidget"
    assert widget._model_name == "AnyModel"
    assert widget._view_name == "AnyView"
    assert "zlc_plot_raster" not in _WIDGET_ESM
    assert "MPLCanvasView" not in _WIDGET_ESM

def test_widget_esm_is_a_pure_frame_blitter_and_input_normalizer() -> None:
    """The kernel renders everything; the browser never paints geometry.

    Transient gesture candidates are baked into fronts by the same
    matplotlib artists that draw committed selectors, so any browser-side
    scene painting would reintroduce a second renderer that can drift.
    """

    # No scene traffic, no browser text, no second canvas.
    assert "scene_json" not in _WIDGET_ESM
    assert "overlay" not in _WIDGET_ESM
    assert "FontFace" not in _WIDGET_ESM
    assert "fillText" not in _WIDGET_ESM
    # Presentation styling lives on a view-owned wrapper; ipywidgets' async
    # default LayoutView rewrites Layout-managed inline styles on the root.
    assert "this.surface" in _WIDGET_ESM
    assert "this.el.style.display" not in _WIDGET_ESM
    assert "this.el.style.width" not in _WIDGET_ESM
    # Input normalization contract: strict-mode-safe sends, one primary
    # pointer, buttons 1..3, drag-gated wheel, synthesized left doubles,
    # JupyterLab context menu opt-out, environment reporting.
    assert "e.button=null" not in _WIDGET_ESM
    assert "e.button = null" not in _WIDGET_ESM
    assert "e.isPrimary" in _WIDGET_ESM
    assert "e.button > 2" in _WIDGET_ESM
    assert "this._dragging || !e.deltaY" in _WIDGET_ESM
    assert "_isDouble" in _WIDGET_ESM
    assert "data-jp-suppress-context-menu" in _WIDGET_ESM
    assert "type: 'environment'" in _WIDGET_ESM
    assert "device_pixel_ratio: window.devicePixelRatio" in _WIDGET_ESM
    # A mismatched-density first frame is held briefly so plots do not paint
    # blurry and then jump once the environment report round-trips.
    assert "_acceptFrame" in _WIDGET_ESM
    assert "_graceDeadline" in _WIDGET_ESM

def test_session_mutation_republishes_front_and_keeps_pointer_compatible() -> None:
    """Every external setter publishes changed pixels through the host front."""

    session = _session()
    host = RasterPlotHost.from_session(session)
    published = Event()
    unsubscribe = host.subscribe_front(lambda _front: published.set())
    try:
        host.wait_for_front(timeout=5.0)
        setters = (
            lambda: session.set_parameter("title", "Mutated"),
            lambda: session.set_labels(x="X2", y="Y2"),
            lambda: session.set_parameter("relim_mode", "tight"),
            lambda: session.set_x_limits(2.0, 7.0),
            lambda: session.set_size("2x4"),
            lambda: session.set_device_pixel_ratio(1.5),
        )
        for setter in setters:
            previous = host.front
            assert previous is not None
            before = previous.buffer.as_rgba(copy=True)
            published.clear()
            completion = setter()
            if hasattr(completion, "result"):
                completion.result(timeout=5.0)
            assert published.wait(5.0)
            front = host.front
            assert front is not None
            assert front.identity.sequence > previous.identity.sequence
            assert not np.array_equal(before, front.buffer.as_rgba())

        assert front is not None
        axis = front.interaction.axes[0]
        operation = host._pointer_event(
            "scroll",
            0.5,
            0.5,
            button=None,
            step=1.0,
            identity=front.identity,
            axes=axis,
        ).result(timeout=5.0)
        assert operation.front is not None
        assert operation.front.identity.sequence > front.identity.sequence
    finally:
        unsubscribe()
        host.close(timeout=5.0)

def test_notebook_host_presents_facet_grid_front() -> None:
    """The notebook transport uses the same multi-axis host path as GUI."""

    spec = facet_spec()
    assert isinstance(spec, FacetGridPlot)
    host = RasterPlotHost.from_plot(_facet_snapshot(), spec)
    try:
        front = host.wait_for_front(timeout=5.0)
        assert front.identity.kind == "facet_grid"
        assert len(front.interaction.axes) >= 2
    finally:
        host.close(timeout=5.0)

def test_middle_double_click_zooms_to_area_selection_then_resets() -> None:
    """Double middle-click zooms to the committed range; without one it homes."""

    def settled_front(host, previous):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            front = host.front
            if front is not None and front.identity.sequence > previous.identity.sequence:
                return front
            time.sleep(0.05)
        raise AssertionError("host never republished after the session mutation")

    session = _session()
    host = RasterPlotHost.from_session(session)
    try:
        initial = host.wait_for_front(timeout=5.0)
        session.set_area_selector(
            NumericRange(2.0, 6.0), NumericRange(0.5, 4.5), display=True
        )
        front = settled_front(host, initial)
        axis = front.interaction.axes[0]
        host._pointer_event(
            "press",
            0.5,
            0.5,
            button=2,
            double=True,
            identity=front.identity,
            axes=axis,
            interaction=front.interaction,
        ).result(timeout=5.0)
        viewport = session.viewport
        assert viewport is not None
        assert viewport.x.low == 2.0 and viewport.x.high == 6.0
        assert viewport.y.low == 0.5 and viewport.y.high == 4.5

        zoomed = host.front
        session.remove_selector(SelectorKind.AREA)
        front = settled_front(host, zoomed)
        host._pointer_event(
            "press",
            0.5,
            0.5,
            button=2,
            double=True,
            identity=front.identity,
            axes=front.interaction.axes[0],
            interaction=front.interaction,
        ).result(timeout=5.0)
        assert session.viewport is None
    finally:
        host.close(timeout=5.0)

def test_pulse_selectors_paint_in_source_units_and_fit_catalogue_is_empty() -> None:
    """Pulse paints selectors in its axes' source-time space, factor once."""

    from zlc_plot import (
        PlotLabels,
        PulseBlock,
        PulseChannel,
        PulseTimelineData,
        pulse_timeline,
    )

    data = PulseTimelineData(
        channels=(PulseChannel("laser", "Laser"),),
        blocks=(PulseBlock("laser", 0.0, 4.0e-6, label="Init"),),
        time_unit="s",
        total_duration=10.0e-6,
    )
    session = pulse_timeline(data, labels=PlotLabels(title="Pulse"))
    host = RasterPlotHost.from_session(session)
    try:
        assert session.fit_models == ()
        front = host.wait_for_front(timeout=5.0)
        axis = front.interaction.axes[0]
        host._pointer_event(
            "press",
            0.5,
            0.5,
            button=3,
            identity=front.identity,
            axes=axis,
            interaction=front.interaction,
        ).result(timeout=5.0)
        released = host._pointer_event(
            "release",
            0.5,
            0.5,
            button=3,
            identity=front.identity,
            axes=axis,
        ).result(timeout=5.0)
        crosshair = next(
            state
            for state in released.front.interaction.selectors
            if state.kind is SelectorKind.CROSSHAIR
        )
        # Painted (front) space is the axes' source-time domain, not the
        # microsecond display projection: the value must sit inside the
        # 10 microsecond total duration.
        assert 0.0 < crosshair.value.x < 10.0e-6
        low, high = axis.x_limits
        assert low <= crosshair.value.x <= high
    finally:
        host.close(timeout=5.0)
        session.close()

def test_environment_message_rescales_raster_to_reported_pixel_ratio() -> None:
    """The browser's pixel-density report must drive the kernel raster size."""

    view = NotebookView(_session(), close_session_on_close=True)
    try:
        first = view.host.wait_for_front(timeout=5.0)
        assert first.device_pixel_ratio == 1.0
        view._on_widget_message(
            None, {"type": "environment", "device_pixel_ratio": 2.0}, None
        )
        deadline = time.monotonic() + 5.0
        front = first
        while time.monotonic() < deadline:
            front = view.host.front
            if front is not None and front.device_pixel_ratio == 2.0:
                break
            time.sleep(0.05)
        assert front is not None
        assert front.device_pixel_ratio == 2.0
        assert front.buffer.width > first.buffer.width
        assert front.logical_size == first.logical_size
        # Malformed reports must be ignored, never raise inside the comm
        # callback.
        view._on_widget_message(
            None, {"type": "environment", "device_pixel_ratio": None}, None
        )
        view._on_widget_message(
            None, {"type": "environment", "device_pixel_ratio": -1.0}, None
        )
    finally:
        view.close()

def test_environment_ratio_is_remembered_for_later_views(monkeypatch) -> None:
    """A new view adopts the last reported density for its first front."""

    import zlc_plot.notebook as notebook_module

    monkeypatch.setattr(notebook_module, "_LAST_DEVICE_PIXEL_RATIO", None)
    reporter = NotebookView(_session(), close_session_on_close=True)
    try:
        reporter.host.wait_for_front(timeout=5.0)
        reporter._on_widget_message(
            None, {"type": "environment", "device_pixel_ratio": 2.0}, None
        )
    finally:
        reporter.close()
    assert notebook_module._LAST_DEVICE_PIXEL_RATIO == 2.0

    adopter = NotebookView(_session(), close_session_on_close=True)
    try:
        assert adopter._pending_environment is not None
        adopter._pending_environment.result(timeout=5.0)
        front = adopter.host.front
        assert front is not None
        assert front.device_pixel_ratio == 2.0
    finally:
        adopter.close()

def test_snapshot_display_data_encodes_front_as_png_at_logical_size() -> None:
    host = RasterPlotHost.from_session(_session())
    try:
        front = host.wait_for_front(timeout=5.0)
        data, metadata = _snapshot_display_data(front)
        png = base64.b64decode(data["image/png"])
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert metadata["image/png"] == {
            "width": front.logical_size[0],
            "height": front.logical_size[1],
        }
        assert "text/plain" in data
    finally:
        host.close(timeout=5.0)

def test_display_publishes_widget_view_with_png_fallback_through_display_id(
    monkeypatch,
) -> None:
    """The bundle must carry the widget view, a PNG fallback and a handle."""

    import IPython.core.interactiveshell as interactiveshell
    import IPython.display as ipython_display

    class _Handle:
        def __init__(self) -> None:
            self.calls: list[tuple[dict, dict]] = []

        def update(self, data: object, *, raw: bool, metadata: object) -> None:
            assert raw is True
            self.calls.append((data, metadata))

    recorded: dict[str, object] = {}
    handle = _Handle()

    def fake_display(data=None, *, raw=False, metadata=None, display_id=None):
        recorded.update(data=data, raw=raw, metadata=metadata, display_id=display_id)
        return handle

    monkeypatch.setattr(ipython_display, "display", fake_display)
    monkeypatch.setattr(
        interactiveshell.InteractiveShell, "initialized", staticmethod(lambda: True)
    )
    view = NotebookView(_session(), close_session_on_close=True)
    try:
        view.display()
        model_id = view.widget.model_id
        assert view._display_handle is handle
    finally:
        view.close()
    data = recorded["data"]
    assert recorded["raw"] is True
    assert recorded["display_id"] is True
    assert data["application/vnd.jupyter.widget-view+json"] == {
        "version_major": 2,
        "version_minor": 0,
        "model_id": model_id,
    }
    assert base64.b64decode(data["image/png"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert set(recorded["metadata"]["image/png"]) == {"width", "height"}
    assert len(handle.calls) == 1
    final_data, _ = handle.calls[0]
    assert "application/vnd.jupyter.widget-view+json" not in final_data
    assert base64.b64decode(final_data["image/png"]).startswith(b"\x89PNG\r\n\x1a\n")

def test_display_without_running_shell_keeps_compact_widget_repr(
    monkeypatch, capsys
) -> None:
    """The degrade path must not dump the base64 PNG bundle to stdout."""

    import IPython.core.interactiveshell as interactiveshell

    monkeypatch.setattr(
        interactiveshell.InteractiveShell, "initialized", staticmethod(lambda: False)
    )
    view = NotebookView(_session(), close_session_on_close=True)
    try:
        view.display()
        assert view._display_handle is None
    finally:
        view.close()
    assert "image/png" not in capsys.readouterr().out

def test_close_replaces_widget_output_with_final_frame_png() -> None:
    """A closed view must leave the last frame in the cell, not a blank output."""

    class _Handle:
        def __init__(self) -> None:
            self.calls: list[tuple[dict, dict]] = []

        def update(self, data: object, *, raw: bool, metadata: object) -> None:
            assert raw is True
            self.calls.append((data, metadata))

    view = NotebookView(_session(), close_session_on_close=True)
    handle = _Handle()
    try:
        front = view.host.wait_for_front(timeout=5.0)
        view._front = front
        view._display_handle = handle
    finally:
        view.close()
    assert len(handle.calls) == 1
    data, metadata = handle.calls[0]
    assert base64.b64decode(data["image/png"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert metadata["image/png"] == {
        "width": front.logical_size[0],
        "height": front.logical_size[1],
    }
    view.close()
    assert len(handle.calls) == 1

def test_raster_pointer_motion_bakes_candidate_into_published_fronts() -> None:
    """Every frontend blits fronts; one renderer draws transient candidates.

    A drag must publish a fresh raster per preview (the matplotlib artists
    bake the candidate), and the committed selector must land in the released
    front's interaction map.  There is no frontend vector scene.
    """

    session = _session()
    session.fit("gaussian_offset")
    host = RasterPlotHost.from_session(session)
    try:
        front = host.wait_for_front(timeout=5.0)
        axis = front.interaction.axes[0]
        pressed = host._pointer_event(
            "press",
            0.4,
            0.4,
            button=1,
            identity=front.identity,
            axes=axis,
            interaction=front.interaction,
        ).result()
        assert pressed.front is front
        assert pressed.value.candidate is None
        assert not pressed.value.publish_front
        moved = host._pointer_event(
            "move",
            0.7,
            0.7,
            button=1,
            identity=front.identity,
            axes=axis,
        ).result()
        assert moved.front.identity.sequence > front.identity.sequence
        assert not hasattr(moved.value, "scene")
        committed = host._pointer_event(
            "release",
            0.7,
            0.7,
            button=1,
            identity=front.identity,
            axes=axis,
        ).result()
        assert committed.front.identity.sequence > moved.front.identity.sequence
        assert tuple(state.kind for state in committed.front.interaction.selectors) == (
            SelectorKind.AREA,
        )
    finally:
        host.close(timeout=5.0)

def test_an_unmoved_area_press_or_double_click_never_becomes_a_gesture() -> None:
    """Press only arms Area; motion is the sole edge that starts it.

    Qt delivers the first half of a double click as an ordinary
    press/release.  Starting the rubber band on press therefore flashed a
    zero-size Area even when the pointer never moved, and grabbing an existing
    Area handle needlessly repainted and revised the selector on a click.
    The backend-neutral host path is shared by Qt and Notebook, so prove the
    contract here once.
    """

    session = _session()
    host = RasterPlotHost.from_session(session)
    events = []
    unsubscribe = session.subscribe_selection(events.append)
    try:
        front = host.wait_for_front(timeout=5.0)
        axis = front.interaction.axes[0]

        def pointer(action, x, y, *, double=False):
            return host._pointer_event(
                action,
                x,
                y,
                button=1,
                double=double,
                identity=front.identity,
                axes=axis,
                interaction=front.interaction if action == "press" else None,
            ).result(timeout=5.0)

        for double in (False, True):
            pressed = pointer("press", 0.4, 0.4, double=double)
            released = pointer("release", 0.4, 0.4)
            assert pressed.front is front and released.front is front
            assert not pressed.value.publish_front
            assert not released.value.publish_front
            assert pressed.value.candidate is None
            assert released.value.candidate is None
            assert session.selectors == ()
            assert events == []

        session.set_area_selector(
            NumericRange(2.0, 6.0), NumericRange(1.0, 5.0)
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and host.front is front:
            time.sleep(0.01)
        area_front = host.front
        assert area_front is not None and area_front is not front
        area_axis = area_front.interaction.axes[0]
        nx, ny = area_axis.display_to_normalized(4.0, 3.0)
        before = session.selectors
        events.clear()

        pressed = host._pointer_event(
            "press",
            nx,
            ny,
            button=1,
            identity=area_front.identity,
            axes=area_axis,
            interaction=area_front.interaction,
        ).result(timeout=5.0)
        released = host._pointer_event(
            "release",
            nx,
            ny,
            button=1,
            identity=area_front.identity,
            axes=area_axis,
        ).result(timeout=5.0)
        assert pressed.front is area_front and released.front is area_front
        assert not pressed.value.publish_front
        assert not released.value.publish_front
        assert session.selectors == before
        assert events == []

        blank_x, blank_y = area_axis.display_to_normalized(8.0, 8.0)
        host._pointer_event(
            "press",
            blank_x,
            blank_y,
            button=1,
            identity=area_front.identity,
            axes=area_axis,
            interaction=area_front.interaction,
        ).result(timeout=5.0)
        host._pointer_event(
            "release",
            blank_x,
            blank_y,
            button=1,
            identity=area_front.identity,
            axes=area_axis,
        ).result(timeout=5.0)
        assert session.selectors == ()
        assert len(events) == 1
    finally:
        unsubscribe()
        host.close(timeout=5.0)

def test_fit_area_pointer_sequence_never_promotes_a_blank_front() -> None:
    session = _session()
    session.fit("gaussian_offset")
    host = RasterPlotHost.from_session(session)
    try:
        front = host.wait_for_front(timeout=5.0)
        axis = front.interaction.axes[0]
        operations = [
            host._pointer_event(
                "press",
                0.32,
                0.35,
                button=1,
                identity=front.identity,
                axes=axis,
                interaction=front.interaction,
            ),
            host._pointer_event(
                "move",
                0.72,
                0.75,
                button=1,
                identity=front.identity,
                axes=axis,
            ),
            host._pointer_event(
                "release",
                0.72,
                0.75,
                button=1,
                identity=front.identity,
                axes=axis,
            ),
        ]
        for operation in operations:
            rgba = np.asarray(operation.result(timeout=5.0).front.buffer.as_rgba())
            assert rgba.std() > 0.0
            assert np.count_nonzero(rgba) > 0
    finally:
        host.close(timeout=5.0)
