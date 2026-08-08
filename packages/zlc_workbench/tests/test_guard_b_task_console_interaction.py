import zou_lab_control_v2

import os
import time

import pytest

import zlc_workbench.console as tested_module


print(tested_module.__file__)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_workbench.console import ConsolePresenter
from zlc_workbench.logic import stable_signal_key
from zlc_workbench.session import ExperimentSession

from test_console_presenter import _ConsoleView
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


def _wait_until(predicate, presenter, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        presenter.poll_logic()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("TaskConsole interaction did not settle before timeout")


def _commit_area(host) -> None:
    """Commit one real pointer gesture through the raster interaction seam."""

    front = host.wait_for_front(5.0)
    axes = front.interaction.axes[0]
    left, bottom, right, top = axes.bounds

    def point(x_fraction: float, y_fraction: float) -> tuple[float, float]:
        return (
            left + (right - left) * x_fraction,
            bottom + (top - bottom) * y_fraction,
        )

    start, end = point(0.25, 0.25), point(0.75, 0.75)
    for action, location in (("press", start), ("move", end), ("release", end)):
        host._pointer_event(
            action,
            location[0],
            location[1],
            button=1,
            identity=front.identity,
            axes=axes,
            interaction=front.interaction,
        ).result()


def test_guard_b_task_console_selector_updates_shared_draft_and_apply_restarts(
    tmp_path,
) -> None:
    plot = pytest.importorskip("zlc_plot")
    session = ExperimentSession.open(tmp_path, template="virtual")
    write_ordinary_pulse(tmp_path)

    def kind_of(name):
        return next((item for item in plot.PlotKind if item.value == str(name)), None)

    def spec_for(snapshot, kind=""):
        return plot.fitting_spec(snapshot.block.schema, kind_of(kind))

    def make_host(initial, _signal, kind=""):
        return plot.RasterPlotHost.from_plot(initial, spec_for(initial, kind))

    presenter = ConsolePresenter(
        session,
        _ConsoleView(),
        make_host=make_host,
        panel_kinds=plot.panel_kinds,
        spec_for=spec_for,
    )
    try:
        session.load_pulse(PULSE_NAME)
        node_id = presenter.add_logic("camera_measurement")
        assert presenter.view.focused_logic_editor == node_id

        authored_roi = (4, 3, 32, 24)
        presenter.view.logic_draft_changed.emit(
            node_id,
            {
                "device_keys": {"camera": "camera"},
                "values": {
                    "exposure_seconds": 0.013,
                    "roi_x": authored_roi[0],
                    "roi_y": authored_roi[1],
                    "roi_width": authored_roi[2],
                    "roi_height": authored_roi[3],
                    "repeat": 0,
                    "frames_per_cycle": CAMERA_WINDOWS,
                },
            },
        )
        assert presenter.start_logic(node_id) is True
        old_host = presenter.logic[node_id].host
        assert old_host is not None and old_host.running
        signal_key = stable_signal_key(node_id, "frames")
        old_generation = old_host.generation

        _wait_until(lambda: session.camera.capture_state(), presenter)
        session.fire(shots=1)
        _wait_until(
            lambda: session.signal_plane.freeze().publication(signal_key) is not None,
            presenter,
        )
        panel = presenter.add_selected_panel("image")
        assert panel is not None and panel.kind == "image"
        assert presenter.edit_panel(panel.panel_id) is True
        assert panel.editor_host is not None and panel.editor_host is not panel.host

        _commit_area(panel.editor_host)
        draft = presenter.logic[node_id].draft.values
        selected_roi = tuple(
            int(draft[name])
            for name in ("roi_x", "roi_y", "roi_width", "roi_height")
        )
        assert selected_roi != authored_roi, (
            "committing the Image Area selector left the producer row ROI draft unchanged"
        )
        assert tuple(
            presenter.view.logic_editors[node_id]["form_values"][name]
            for name in ("roi_x", "roi_y", "roi_width", "roi_height")
        ) == selected_roi

        assert presenter.apply_panel_producer(panel.panel_id) is True
        _wait_until(lambda: presenter.logic[node_id].host is not old_host, presenter)
        replacement = presenter.logic[node_id].host
        assert replacement is not None and replacement.running
        assert replacement.signal_key("frames") == signal_key
        assert replacement.generation != old_generation
        assert replacement.node.request.roi_xywh == selected_roi
    finally:
        presenter.close()
        session.close()
