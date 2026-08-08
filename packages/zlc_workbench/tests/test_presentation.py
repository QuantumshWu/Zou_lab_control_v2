"""The scheduler and the plotting host meet, and a live panel updates.

These two were designed to fit and had never been connected: zlc_runtime
schedules board-coherent updates and cannot draw, zlc_plot draws and does not
schedule.  Nothing exercised the seam, so nothing could have caught them
disagreeing about it.

What is proved here: a tick freezes a front, the panel prepares from it, and the
batch commits -- with real live data from a real camera, through a real plotting
host, driven by the real scheduler.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from zlc_data import StreamGenerationId

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.install import create_installation
from zlc_atom.nodes.calibration import FrameContract, ReadoutModel, SiteMap, TrapCalibration
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_runtime.plane import SignalDataPlane
from zlc_runtime.streams import EventRef
from zlc_runtime.presentation import (
    BoardScheduler,
    HarmonicClock,
    OwnerChannels,
    SurfaceBatchArbiter,
)
from zlc_workbench.presentation import PlotPanelPort
from zlc_workbench.image_overlay import ImageOverlayResolver

ATOM_ROOT = Path(__file__).resolve().parents[2] / "zlc_atom"
if str(ATOM_ROOT) not in sys.path:
    sys.path.insert(0, str(ATOM_ROOT))

from pulses.calibration import build  # noqa: E402


@pytest.fixture
def live_bench():
    """A camera monitoring live, with its frames on the plane."""

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    monitor = None
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build()
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)

        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest("camera", 0.02, None, 0, 1, 2.0),
            signal_plane=plane,
            producer="cm",
        )
        monitor = node.monitor(buffer_frames=4)
        sequencer.fire()
        sequencer.wait_done(1.0)
        deadline = time.monotonic() + 5.0
        while monitor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        yield plane, node, sequencer, monitor
    finally:
        if monitor is not None:
            monitor.close()
        installation.close()
        plane.close()


def test_the_scheduler_drives_a_real_plotting_host(live_bench) -> None:
    plot = pytest.importorskip("zlc_plot")
    plane, node, sequencer, monitor = live_bench

    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    assert value is not None, "the monitor published nothing to present"

    host = plot.RasterPlotHost.from_plot(
        value.snapshot,
        plot.ImagePlot(
            plot.AxisRef.data("spatial-x"),
            plot.AxisRef.data("spatial-y"),
            labels=plot.PlotLabels("live", "x", "y"),
        ),
    )
    try:
        port = PlotPanelPort("panel-1", signal, host, display_interval_ms=100)
        class _Wake:
            """The shape a Qt shim implements: coalesce, then wake the owner."""

            def __init__(self) -> None:
                self.pending = Event()

            def request_owner_wake(self) -> None:
                self.pending.set()

        wake = _Wake()
        channels = OwnerChannels(wake)
        arbiter = SurfaceBatchArbiter(channels)
        clock = HarmonicClock((100, 200, 400, 800), 100)
        scheduler = BoardScheduler(plane, clock, arbiter, lambda: (port,))

        presented = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            sequencer.fire()
            sequencer.wait_done(1.0)
            monitor.poll()
            scheduler.on_tick()
            # The owner thread drains what the tick staged; a Qt shim would do
            # this on the GUI thread when its wake fires.
            arbiter.drain(lambda panel_id: port if panel_id == port.panel_id else None)
            presented = port.presented_publication()
            if presented is not None:
                break
            time.sleep(0.05)

        assert presented is not None, (
            f"the board never committed a surface; panel missing={port.missing}"
        )
    finally:
        host.close()


def test_a_panel_refuses_a_surface_prepared_for_a_different_host(live_bench) -> None:
    """Board coherence: a reconfigured panel abandons the batch, not just itself.

    Showing one stale panel beside fresh ones is worse than showing nothing --
    it is the failure a coherent board exists to prevent, and it looks like real
    data.
    """

    plot = pytest.importorskip("zlc_plot")
    plane, node, _sequencer, _monitor = live_bench

    signal = node.signal_key("frames")
    value = plane.freeze().value(signal)
    assert value is not None
    publication = plane.latest_publication(signal)
    assert publication is not None

    spec = plot.ImagePlot(
        plot.AxisRef.data("spatial-x"),
        plot.AxisRef.data("spatial-y"),
        labels=plot.PlotLabels("live", "x", "y"),
    )
    host = plot.RasterPlotHost.from_plot(value.snapshot, spec)
    other = plot.RasterPlotHost.from_plot(value.snapshot, spec)
    try:
        port = PlotPanelPort("panel-1", signal, host, display_interval_ms=100)
        update = port.prepare(value, publication)
        assert port.can_accept(update, None)

        import dataclasses

        stale = dataclasses.replace(update, host_token=other.host_id)
        assert not port.can_accept(stale, None)
        assert not port.accept(stale, None)
        assert port.presented_publication() is None
    finally:
        host.close()
        other.close()


def test_a_new_generation_replaces_the_plot_host_even_at_the_same_revision(
    live_bench,
) -> None:
    """Revision orders one run; generation separates two runs both at one."""

    plot = pytest.importorskip("zlc_plot")
    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    spec = plot.ImagePlot(
        plot.AxisRef.data("spatial-x"),
        plot.AxisRef.data("spatial-y"),
    )
    first = plot.RasterPlotHost.from_plot(value.snapshot, spec)
    replacements: list[object] = []

    def replace_host(plot_input, _value, _publication):
        host = plot.RasterPlotHost.from_plot(plot_input, spec)
        replacements.append(host)
        return host

    try:
        port = PlotPanelPort(
            "panel-1",
            signal,
            first,
            display_interval_ms=100,
            shown=value.snapshot,
            replace_host=replace_host,
        )
        assert port.prepare(value, publication) is None
        restarted = replace(
            publication,
            event_ref=EventRef(
                publication.event_ref.stream_id,
                StreamGenerationId("replacement-run"),
                publication.event_ref.sequence,
            ),
        )
        assert port.prepare(value, restarted) is None
        assert len(replacements) == 1
        assert port.host is replacements[0]
        assert port.presented_publication() is restarted
    finally:
        first.close()
        for host in replacements:
            host.close()


def test_image_overlay_resolves_one_exact_occupancy_status_row(
    live_bench,
    tmp_path,
) -> None:
    """The overlay uses SiteMap identity and sibling occupied/valid values."""

    from zlc_data import SITE
    from zlc_plot.primitives import PointStatus
    from zlc_workbench.logic import stable_signal_key

    plane, node, _sequencer, _monitor = live_bench
    source = plane.freeze().value(node.signal_key("frames"))
    assert source is not None
    shape = tuple(int(value) for value in source.snapshot.block.values.shape[-2:])
    site_ids = ("a", "b", "c")
    calibration = TrapCalibration(
        SiteMap(
            site_ids,
            np.asarray(((2.0, 3.0), (5.0, 7.0), (9.0, 11.0))),
            np.asarray((True, True, False)),
            np.asarray((1.0, 1.0, 0.0)),
        ),
        ReadoutModel(
            site_ids,
            np.asarray((1.0, 1.0, 1.0)),
            np.asarray((True, True, True)),
            np.asarray((1.0, 1.0, 1.0)),
        ),
        FrameContract(shape),
    )
    path = calibration.save(tmp_path / "calibration.json")
    node_id = "occupancy-test"
    frame_key = stable_signal_key(node_id, "frame_judged")
    occupied_key = stable_signal_key(node_id, "occupied")
    valid_key = stable_signal_key(node_id, "valid")
    table = SimpleNamespace(
        row_count=3,
        columns=(SimpleNamespace(role=SITE),),
    )
    frame_value = SimpleNamespace(name=frame_key, snapshot=source.snapshot)
    occupied_value = SimpleNamespace(
        name=occupied_key,
        values=np.asarray([[[False], [True], [True]]]),
        schema=SimpleNamespace(point_table=table),
    )
    valid_value = SimpleNamespace(
        name=valid_key,
        values=np.asarray([[[True], [True], [False]]]),
        schema=SimpleNamespace(point_table=table),
    )
    values = {
        frame_key: frame_value,
        occupied_key: occupied_value,
        valid_key: valid_value,
    }
    publication = SimpleNamespace(
        run_record={
            "node": node_id,
            "parameters": {"calibration_path": str(path)},
        },
        value=values.get,
    )

    resolved = ImageOverlayResolver().resolve(
        frame_value,
        publication,
        mode="occupancy",
        overlay_revision=7,
    )

    assert resolved.resolved_mode == "occupancy"
    assert resolved.frame.overlay.point_ids == site_ids
    assert resolved.frame.overlay.statuses == (
        PointStatus.EMPTY,
        PointStatus.OCCUPIED,
        PointStatus.INVALID,
    )
    np.testing.assert_array_equal(
        resolved.frame.overlay.coordinates,
        calibration.site_map.centers_xy,
    )


def test_the_live_board_ticks_and_commits_through_one_object(live_bench) -> None:
    """What an application actually holds: a board, a tick, a commit.

    Nothing ticked and nothing polled before this.  The scheduler existed, the
    wake primitive existed, and no code connected them -- so a live panel could
    never update however correct either side was.
    """

    plot = pytest.importorskip("zlc_plot")
    from zlc_workbench.board import LiveBoard

    plane, node, sequencer, monitor = live_bench
    signal = node.signal_key("frames")
    value = plane.freeze().value(signal)
    assert value is not None

    host = plot.RasterPlotHost.from_plot(
        value.snapshot,
        plot.ImagePlot(
            plot.AxisRef.data("spatial-x"),
            plot.AxisRef.data("spatial-y"),
            labels=plot.PlotLabels("live", "x", "y"),
        ),
    )
    try:
        port = PlotPanelPort("panel-1", signal, host, display_interval_ms=100)
        woken: list[int] = []
        board = LiveBoard(
            plane,
            lambda: (port,),
            intervals=(100, 200, 400, 800),
            default_interval_ms=100,
            notify=lambda: woken.append(1),
        )
        try:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and port.presented_publication() is None:
                sequencer.fire()
                sequencer.wait_done(1.0)
                monitor.poll()
                board.tick()
                board.commit()
                time.sleep(0.05)

            assert port.presented_publication() is not None, (
                f"the board never presented; missing={port.missing}"
            )
            assert woken, "the runtime never asked for an owner turn"
        finally:
            board.close()
    finally:
        host.close()


def test_the_wake_coalesces_so_a_burst_costs_one_turn() -> None:
    """A burst of finished surfaces must not queue a turn each."""

    from zlc_workbench.board import OwnerWake

    notifications: list[int] = []
    wake = OwnerWake(lambda: notifications.append(1))
    for _ in range(5):
        wake.request_owner_wake()

    assert len(notifications) == 1, "each request notified separately"
    assert wake.take() is True
    assert wake.take() is False, "a claimed wake must not fire twice"

    wake.request_owner_wake()
    assert len(notifications) == 2, "a wake after a turn must notify again"
