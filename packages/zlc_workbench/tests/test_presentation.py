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
from pathlib import Path
from threading import Event

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera_measurement.measurement import CameraMeasurementNode
from zlc_runtime.plane import SignalDataPlane
from zlc_runtime.presentation import (
    BoardScheduler,
    HarmonicClock,
    OwnerChannels,
    SurfaceBatchArbiter,
)
from zlc_workbench.presentation import PlotPanelPort

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

        node = CameraMeasurementNode(camera=camera, signal_plane=plane, producer="cm")
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
