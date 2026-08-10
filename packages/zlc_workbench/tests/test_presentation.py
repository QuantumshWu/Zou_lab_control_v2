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
import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from zlc_data import StreamGenerationId

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.install import create_installation
from zlc_atom.nodes.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
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
from zlc_workbench.image_overlay import image_frame_from_publication
from zlc_workbench.session import read_pulse
from pulse_fixtures import write_ordinary_pulse


@pytest.fixture
def live_bench(tmp_path):
    """A camera monitoring live, with its frames on the plane."""

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    monitor = None
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        from zlc_pulse import compile_sequence, load_streamer_config

        sequence = read_pulse(write_ordinary_pulse(tmp_path)).sequence
        config = load_streamer_config()
        program = compile_sequence(sequence, config["params"], config["clock_hz"])
        sequencer.camera_trigger_channel = "emCCD"
        sequencer.load(program, source=sequence)

        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest("camera", 0.02, None, 0, 1),
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

    signal = node.signal_key("frame_0")
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

        models = host.fit_models().result(timeout=10).value
        assert models
        # Arm without blocking, exactly as the console does: the future
        # resolves at the first accepted pair — the arming solve itself, or
        # the next frame's pair when new data raced the arming solve.
        armed = host.fit(models[0].model_id, live=True)
        previous = presented
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            sequencer.fire()
            sequencer.wait_done(1.0)
            monitor.poll()
            scheduler.on_tick()
            arbiter.drain(
                lambda panel_id: port if panel_id == port.panel_id else None
            )
            presented = port.presented_publication()
            if (
                armed.done()
                and presented is not None
                and presented is not previous
            ):
                break
            time.sleep(0.05)
        assert presented is not None and presented is not previous
        assert armed.done()
        assert armed.result(timeout=0).value is not None
        assert port.last_error is None
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

    signal = node.signal_key("frame_0")
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


def test_publication_for_revision_resolves_bare_integer_revisions(
    live_bench,
) -> None:
    """A fit event names its parent by bare int; snapshots carry DatasetRevision.

    Comparing the raw objects silently never matched, so once a panel had
    staged a newer frame every trailing fit publication was dropped as
    superseded — the rolling trace of a fit froze whenever the camera ran
    faster than the solve, while every other panel kept flowing.
    """

    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frame_0")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    revision_object = value.snapshot.block.revision
    revision_number = value.snapshot.ref.revision.value
    assert not isinstance(revision_object, int)

    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda snapshot, **_render: Future(),
    )
    port = PlotPanelPort("panel-1", signal, host, display_interval_ms=100)
    update = port.prepare(value, publication)
    assert update is not None

    # Pending: the bare integer the fit event carries must resolve.
    assert port.publication_for_revision(revision_number) is publication
    assert port.publication_for_revision(revision_object) is publication
    assert port.publication_for_revision(revision_number + 999) is None

    # Presented: same identity rule after the batch accepted the update.
    assert port.accept(update, None)
    assert port.publication_for_revision(revision_number) is publication


def test_one_publication_is_submitted_once_while_its_surface_is_pending(
    live_bench,
) -> None:
    """A display beat cannot enqueue the same immutable publication twice."""

    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frame_0")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None

    calls: list[object] = []
    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda snapshot, **_render: calls.append(snapshot) or Future(),
    )
    port = PlotPanelPort("panel-1", signal, host, display_interval_ms=100)
    first = port.prepare(value, publication)
    assert first is not None
    assert port.prepare(value, publication) is None
    assert len(calls) == 1

    port.finish_unpresented(first)
    retry = port.prepare(value, publication)
    assert retry is not None
    assert len(calls) == 2
    port.finish_unpresented(retry)

    host.update_data = lambda _snapshot, **_render: (_ for _ in ()).throw(
        RuntimeError("synchronous render rejection")
    )
    with pytest.raises(RuntimeError, match="synchronous render rejection"):
        port.prepare(value, publication)
    host.update_data = lambda snapshot, **_render: calls.append(snapshot) or Future()
    recovered = port.prepare(value, publication)
    assert recovered is not None
    port.finish_unpresented(recovered)


def test_frames_outpacing_the_render_worker_are_skipped_without_an_error(
    live_bench,
) -> None:
    """Latest-only coalescing is flow control, not failure.

    When the camera publishes faster than the worker draws, the queued render
    is superseded and its future cancelled.  The arbiter used to turn that
    cancellation into a batch error, so every live camera panel went red once
    per coalesced frame -- with an empty message, because CancelledError
    stringifies to nothing.  A superseded update must simply finish, and the
    newer frame must present.
    """

    from zlc_data import owned_snapshot_from_arrays
    from zlc_runtime.plane import SignalFront

    plot = pytest.importorskip("zlc_plot")
    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frame_0")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None

    host = plot.RasterPlotHost.from_plot(
        value.snapshot,
        plot.ImagePlot(
            plot.AxisRef.data("spatial-x"),
            plot.AxisRef.data("spatial-y"),
        ),
    )
    try:
        port = PlotPanelPort(
            "panel-1",
            signal,
            host,
            display_interval_ms=100,
            shown=value.snapshot,
        )
        channels = OwnerChannels(SimpleNamespace(request_owner_wake=lambda: None))
        arbiter = SurfaceBatchArbiter(channels)

        def moment(step: int) -> SignalFront:
            snapshot = owned_snapshot_from_arrays(
                schema=value.snapshot.block.schema,
                values=value.snapshot.block.values,
                revision=value.snapshot.block.revision.value + step,
                validity=value.snapshot.block.validity,
                block_id=value.snapshot.block.block_id,
                stream_generation=value.snapshot.ref.stream_generation,
            )
            stepped_value = replace(value, snapshot=snapshot, transient=True)
            stepped = replace(
                publication,
                event_ref=replace(
                    publication.event_ref,
                    sequence=publication.event_ref.sequence + step,
                ),
                signals={**publication.signals, signal: stepped_value},
            )
            return SignalFront(
                {signal: stepped_value},
                {},
                {signal: stepped},
                {signal: frozenset((signal,))},
            )

        # Anchor the identity of the snapshot the host was built from.
        assert not arbiter.enqueue_group((port,), moment(0))

        # Occupy the serial render worker so the next update stays queued.
        gate = Event()
        blocker = host.dispatch_control(lambda: gate.wait(10))
        newer, newest = moment(1), moment(2)
        assert arbiter.enqueue_group((port,), newer)
        assert arbiter.enqueue_group((port,), newest)
        gate.set()
        blocker.result(timeout=10)

        deadline = time.monotonic() + 10.0
        while arbiter.pending_batches and time.monotonic() < deadline:
            arbiter.drain(
                lambda panel_id: port if panel_id == port.panel_id else None
            )
            time.sleep(0.01)

        assert arbiter.pending_batches == 0
        assert port.last_error is None, (
            f"a coalesced frame became an error: {port.last_error!r}"
        )
        assert port.presented_publication() is newest.publication(signal)
    finally:
        host.close()


def test_a_cancelled_render_is_never_remembered_as_a_panel_error(
    live_bench,
) -> None:
    """Whatever path hands reject() a CancelledError, it is not a failure."""

    from concurrent.futures import CancelledError, Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frame_0")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None

    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda _snapshot, **_render: Future(),
    )
    port = PlotPanelPort("panel-1", signal, host, display_interval_ms=100)
    update = port.prepare(value, publication)
    assert update is not None

    port.reject(update, CancelledError())
    assert port.last_error is None

    retry = port.prepare(value, publication)
    assert retry is not None, "the superseded update did not release its slot"
    port.reject(retry, RuntimeError("a real render failure"))
    assert isinstance(port.last_error, RuntimeError)


def test_same_snapshot_final_reanchors_pending_and_presented_identity(
    live_bench,
) -> None:
    """A reused FINAL snapshot neither redraws nor regresses to LIVE identity."""

    from concurrent.futures import Future
    from zlc_data import owned_snapshot_from_arrays

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frame_0")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None

    calls: list[object] = []
    completion: Future[object] = Future()
    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda snapshot, **_render: calls.append(snapshot) or completion,
    )
    presented: list[object] = []
    port = PlotPanelPort(
        "panel-1",
        signal,
        host,
        display_interval_ms=100,
        shown=value.snapshot,
        on_presented=lambda selected, _input: presented.append(selected),
    )
    assert port.prepare(value, publication) is None

    snapshot = owned_snapshot_from_arrays(
        schema=value.snapshot.block.schema,
        values=value.snapshot.block.values,
        revision=value.snapshot.block.revision.value + 1,
        validity=value.snapshot.block.validity,
        block_id=value.snapshot.block.block_id,
        stream_generation=value.snapshot.ref.stream_generation,
    )
    live_value = replace(value, snapshot=snapshot, transient=True)
    live = replace(
        publication,
        event_ref=replace(
            publication.event_ref,
            sequence=publication.event_ref.sequence + 1,
        ),
        signals={**publication.signals, signal: live_value},
    )
    update = port.prepare(live_value, live)
    assert update is not None

    final_value = replace(live_value, transient=False)
    final = replace(
        live,
        event_ref=replace(live.event_ref, sequence=live.event_ref.sequence + 1),
        signals={**live.signals, signal: final_value},
    )
    assert port.prepare(final_value, final) is None
    assert calls == [snapshot], "FINAL must coalesce with the identical pending render"

    completion.set_result(object())
    assert port.accept(update, object()) is True
    assert port.presented_publication() is final
    assert port.presented_publication().value(signal).transient is False
    assert presented[-1] is final

    terminal_record = replace(
        final,
        event_ref=replace(final.event_ref, sequence=final.event_ref.sequence + 1),
    )
    assert port.prepare(final_value, terminal_record) is None
    assert port.presented_publication() is terminal_record
    assert calls == [snapshot], "identity-only reanchor must not redraw pixels"


def test_a_new_generation_replaces_the_plot_host_even_at_the_same_revision(
    live_bench,
) -> None:
    """Revision orders one run; generation separates two runs both at one."""

    plot = pytest.importorskip("zlc_plot")
    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frame_0")
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
) -> None:
    """The image consumes one explicit typed overlay sibling, not an artifact."""

    from zlc_atom.nodes.occupancy.processor import OccupancyProcessor
    from zlc_plot.primitives import PointStatus
    from zlc_workbench.logic import stable_signal_key

    plane, node, _sequencer, _monitor = live_bench
    source = plane.freeze().value(node.signal_key("frame_0"))
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
        (
            ReadoutModel(
                site_ids,
                np.asarray((1.0e20, -1.0e20, 0.0)),
                np.asarray((True, True, True)),
                np.asarray((1.0, 1.0, 1.0)),
            ),
        ),
        ReadoutModelKind.BOX,
        FrameContract(shape),
    )
    node_id = "occupancy-test"
    frame_key = stable_signal_key(node_id, "frame_judged")
    overlay_key = stable_signal_key(node_id, "site_overlay")
    processor = OccupancyProcessor(calibration, producer=node_id)
    result = processor.process(
        np.asarray(source.values).reshape((-1, *shape))[:1],
        generation="one-frame",
        revision=1,
    )
    frame_value = SimpleNamespace(name=frame_key, snapshot=source.snapshot)
    overlay_value = SimpleNamespace(
        name=overlay_key,
        snapshot=result.artifacts["site_overlay"],
    )
    values = {
        frame_key: frame_value,
        overlay_key: overlay_value,
    }
    publication = SimpleNamespace(value=values.get)

    resolved = image_frame_from_publication(
        frame_value,
        publication,
        overlay_signal=overlay_key,
        overlay_revision=7,
    )

    assert resolved.overlay.point_ids == site_ids
    assert resolved.overlay.labels == ("1", "2", "3")
    assert resolved.overlay.statuses == (
        PointStatus.EMPTY,
        PointStatus.OCCUPIED,
        PointStatus.INVALID,
    )
    np.testing.assert_array_equal(
        resolved.overlay.coordinates,
        calibration.site_map.centers_xy,
    )
    assert resolved.snapshot is source.snapshot


def test_the_live_board_ticks_and_commits_through_one_object(live_bench) -> None:
    """What an application actually holds: a board, a tick, a commit.

    Nothing ticked and nothing polled before this.  The scheduler existed, the
    wake primitive existed, and no code connected them -- so a live panel could
    never update however correct either side was.
    """

    plot = pytest.importorskip("zlc_plot")
    from zlc_workbench.board import LiveBoard

    plane, node, sequencer, monitor = live_bench
    signal = node.signal_key("frame_0")
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
            # The cadence the Qt beat must be driven at: the clock's own base.
            assert board.base_interval_ms == 100
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
