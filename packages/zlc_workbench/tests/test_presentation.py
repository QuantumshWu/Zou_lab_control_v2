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

import pytest
from zlc_data import StreamGenerationId, owned_snapshot_from_arrays

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.install import create_installation
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
from zlc_workbench.session import read_pulse
from pulse_fixtures import write_ordinary_pulse


def _submit_now(work):
    from concurrent.futures import Future

    completed = Future()
    try:
        completed.set_result(work())
    except BaseException as error:
        completed.set_exception(error)
    return completed


def _without_deadlock(work, *, timeout: float = 2.0):
    """Run one lock-sensitive call without letting a regression hang pytest."""

    from threading import Thread

    finished = Event()
    outcome: list[object] = []

    def invoke() -> None:
        try:
            outcome.append(work())
        except BaseException as error:
            outcome.append(error)
        finally:
            finished.set()

    Thread(target=invoke, daemon=True).start()
    assert finished.wait(timeout), "presentation call deadlocked"
    result = outcome[0]
    if isinstance(result, BaseException):
        raise result
    return result


def _ready(value=None):
    from concurrent.futures import Future

    future = Future()
    future.set_result(value)
    return future


def _initial_then(host, later=None):
    first = True

    def replace_host(plot_input, value, publication):
        nonlocal first
        if first:
            first = False
            return host, _ready()
        if later is None:
            raise AssertionError("test did not declare a generation replacement")
        return later(plot_input, value, publication)

    return replace_host


def _mount(port, value, publication, front):
    update = port.prepare(value, publication, front)
    assert update is not None
    operation = update.future.result(timeout=10.0)
    assert port.accept(update, operation)
    assert port.presented_publication() is publication
    return update


def _advanced(value, publication, signal: str, step: int = 1):
    snapshot = owned_snapshot_from_arrays(
        value.snapshot.block.schema,
        value.snapshot.block.values,
        value.snapshot.block.revision.value + step,
        validity=value.snapshot.block.validity,
        block_id=value.snapshot.block.block_id,
        stream_generation=value.snapshot.ref.stream_generation,
    )
    advanced_value = replace(value, snapshot=snapshot)
    advanced_publication = replace(
        publication,
        event_ref=replace(
            publication.event_ref,
            sequence=publication.event_ref.sequence + step,
        ),
        signals={**publication.signals, signal: advanced_value},
    )
    return advanced_value, advanced_publication


class _ClosingHost:
    def __init__(self, name: str) -> None:
        self.host_id = name
        self.closed = False

    def close(self, *, timeout=0.0):
        del timeout
        self.closed = True
        return True


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
        monitor = node.monitor()
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
        port = PlotPanelPort(
            "panel-1",
            signal,
            display_interval_ms=100,
            submit_projection=_submit_now,
            replace_host=_initial_then(host),
        )
        class _Wake:
            """The shape a Qt shim implements: coalesce, then wake the owner."""

            def __init__(self) -> None:
                self.pending = Event()

            def request_owner_wake(self) -> None:
                self.pending.set()

        wake = _Wake()
        channels = OwnerChannels(wake)
        arbiter = SurfaceBatchArbiter(channels)
        clock = HarmonicClock((100, 200, 400, 800))
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

    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
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
        port = PlotPanelPort(
            "panel-1",
            signal,
            display_interval_ms=100,
            submit_projection=_submit_now,
            replace_host=_initial_then(host),
        )
        _mount(port, value, publication, front)
        next_value, next_publication = _advanced(value, publication, signal)
        update = port.prepare(next_value, next_publication, front)
        operation = update.future.result(timeout=10.0)
        assert port.can_accept(update, operation)

        import dataclasses

        stale = dataclasses.replace(update, host_token=other.host_id)
        assert not port.can_accept(stale, None)
        assert not port.accept(stale, None)
        assert port.presented_publication() is publication
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
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    revision_object = value.snapshot.block.revision
    revision_number = value.snapshot.ref.revision.value
    assert not isinstance(revision_object, int)

    host = SimpleNamespace(host_id=object())
    rendered = Future()
    port = PlotPanelPort(
        "panel-1",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=lambda _input, _value, _publication: (host, rendered),
    )
    update = port.prepare(value, publication, front)
    assert update is not None

    # Pending: the bare integer the fit event carries must resolve.
    assert port.publication_for_revision(revision_number) is publication
    assert port.publication_for_revision(revision_object) is publication
    assert port.publication_for_revision(revision_number + 999) is None

    # Presented: same identity rule after the batch accepted the update.
    rendered.set_result("mounted")
    assert port.accept(update, update.future.result(timeout=0))
    assert port.publication_for_revision(revision_number) is publication


def test_closing_a_port_cancels_queued_projection(live_bench) -> None:
    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    queued = Future()
    port = PlotPanelPort(
        "panel",
        signal,
        display_interval_ms=100,
        submit_projection=lambda _work: queued,
        replace_host=_initial_then(SimpleNamespace(host_id=object())),
    )

    update = port.prepare(value, publication, front)
    assert update is not None and port.has_pending
    port.close()

    assert queued.cancelled()
    assert update.future.cancelled()
    assert not port.has_pending


def test_close_does_not_wait_for_initial_host_staging(live_bench) -> None:
    from concurrent.futures import Future, ThreadPoolExecutor

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    entered = Event()
    release = Event()
    staged = SimpleNamespace(host_id=object())
    retired: list[object] = []

    def replace_host(_input, _value, _publication):
        entered.set()
        assert release.wait(5.0)
        rendered = Future()
        rendered.set_result(None)
        return staged, rendered

    projector = ThreadPoolExecutor(max_workers=1)
    try:
        port = PlotPanelPort(
            "panel",
            signal,
            display_interval_ms=100,
            submit_projection=projector.submit,
            replace_host=replace_host,
            retire_host=retired.append,
        )
        update = port.prepare(value, publication, front)
        assert update is not None and entered.wait(5.0)

        _without_deadlock(port.close)
        release.set()
        deadline = time.monotonic() + 5.0
        while not retired and time.monotonic() < deadline:
            time.sleep(0.001)

        assert update.future.cancelled()
        assert retired == [staged]
    finally:
        release.set()
        projector.shutdown(wait=True, cancel_futures=True)


def test_close_does_not_wait_for_running_projection(live_bench) -> None:
    from concurrent.futures import Future, ThreadPoolExecutor

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    entered = Event()
    release = Event()

    def project_input(plot_value, _publication, _front):
        entered.set()
        assert release.wait(5.0)
        return plot_value.snapshot

    rendered = Future()
    projector = ThreadPoolExecutor(max_workers=1)
    try:
        host = SimpleNamespace(
            host_id=object(),
            update_data=lambda _snapshot: rendered,
        )
        port = PlotPanelPort(
            "panel",
            signal,
            display_interval_ms=100,
            project_input=project_input,
            submit_projection=projector.submit,
            replace_host=lambda _input, _value, _publication: (host, rendered),
        )
        update = port.prepare(value, publication, front)
        assert update is not None and entered.wait(5.0)

        _without_deadlock(port.close)

        assert update.future.cancelled()
        assert not port.has_pending
    finally:
        release.set()
        projector.shutdown(wait=True, cancel_futures=True)


def test_already_completed_render_does_not_reenter_state_lock(live_bench) -> None:
    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    rendered = Future()
    operation = object()
    rendered.set_result(operation)
    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda _snapshot: rendered,
    )
    port = PlotPanelPort(
        "panel",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(host),
    )
    _mount(port, value, publication, front)
    next_value, next_publication = _advanced(value, publication, signal)

    update = _without_deadlock(
        lambda: port.prepare(next_value, next_publication, front)
    )

    assert update is not None
    assert update.future.result(timeout=0) is operation


def test_already_completed_replacement_does_not_reenter_state_lock(
    live_bench,
) -> None:
    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    old = SimpleNamespace(host_id=object())
    replacement = SimpleNamespace(host_id=object())
    rendered = Future()
    operation = object()
    rendered.set_result(operation)
    later = lambda _input, _value, _publication: (replacement, rendered)
    port = PlotPanelPort(
        "panel",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(old, later),
    )
    _mount(port, value, publication, front)
    restarted = replace(
        publication,
        event_ref=EventRef(
            publication.event_ref.stream_id,
            StreamGenerationId("completed-replacement"),
            publication.event_ref.sequence,
        ),
    )

    update = _without_deadlock(lambda: port.prepare(value, restarted, front))

    assert update is not None
    assert update.future.result(timeout=0) is operation
    assert port.accept(update, operation)
    assert port.host is replacement


def test_companion_only_change_updates_overlay_and_composite_currency(
    live_bench,
) -> None:
    from concurrent.futures import Future
    from zlc_plot.primitives import ImageFrame, ImagePointOverlay
    from zlc_runtime.plane import SignalFront

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    companion = "@logic/occupancy/occupied"
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    companion_value = replace(value, name=companion)

    def with_companion(sequence: int) -> tuple[SignalFront, object]:
        companion_publication = replace(
            publication,
            event_ref=replace(publication.event_ref, sequence=sequence),
            signals={companion: companion_value},
        )
        return (
            SignalFront(
                {signal: value, companion: companion_value},
                {},
                {
                    signal: publication,
                    companion: companion_publication,
                },
            ),
            companion_publication,
        )

    first_front, first_companion = with_companion(101)
    second_front, second_companion = with_companion(102)
    overlay_calls: list[object] = []
    completion = Future()
    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda _input: (_ for _ in ()).throw(
            AssertionError("companion-only change redrew unchanged data")
        ),
        update_image_overlay=lambda overlay: (
            overlay_calls.append(overlay) or completion
        ),
    )

    def project(primary, _publication, exact_front):
        companion_publication = exact_front.publication(companion)
        return ImageFrame(
            primary.snapshot,
            ImagePointOverlay.empty(companion_publication.event_ref.sequence),
        )

    port = PlotPanelPort(
        "panel",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(host),
        companion_signals=lambda: (companion,),
        project_input=project,
    )
    _mount(port, value, publication, first_front)
    assert port.presented_front_refs() == (
        publication.event_ref,
        first_companion.event_ref,
    )

    update = port.prepare(value, publication, second_front)
    assert update is not None
    assert len(overlay_calls) == 1
    completion.set_result("overlay")
    assert port.accept(update, "overlay")
    assert port.presented_front_refs() == (
        publication.event_ref,
        second_companion.event_ref,
    )


def test_a_completed_render_skipped_with_its_cohort_is_never_restaged(
    live_bench,
) -> None:
    """The host committed the frame even though its shot was abandoned.

    The port is the host's sole feeder and host revisions are strictly
    monotonic, so re-offering the same publication could only bounce off the
    host as a "data revision must increase" refusal — which showed on the
    card once per beat until the next shot arrived.  The staged mark records
    the host's true holdings; only a NEWER revision changes the panel.  A
    render that never completed (coalesced away while queued) leaves no
    mark: the host never drew it, so it may be staged again.
    """

    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None

    renders: list[Future] = []
    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda _snapshot, **_render: renders.append(Future())
        or renders[-1],
    )
    port = PlotPanelPort(
        "panel-1",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(host),
    )
    _mount(port, value, publication, front)
    shown = port.presented_publication()
    next_value, next_publication = _advanced(value, publication, signal)
    update = port.prepare(next_value, next_publication, front)
    assert update is not None

    # The render completed (the host holds the revision), then the whole
    # cohort was abandoned: the decision is final for this revision.
    renders[-1].set_result(object())
    update.future.result(timeout=0)
    port.finish_unpresented(update)
    assert port.prepare(next_value, next_publication, front) is None
    assert port.presented_publication() is shown
    assert port.last_error is None


def test_one_publication_is_submitted_once_while_its_surface_is_pending(
    live_bench,
) -> None:
    """A display beat cannot enqueue the same immutable publication twice."""

    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None

    calls: list[object] = []
    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda snapshot, **_render: calls.append(snapshot) or Future(),
    )
    port = PlotPanelPort(
        "panel-1",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(host),
    )
    _mount(port, value, publication, front)
    next_value, next_publication = _advanced(value, publication, signal)
    first = port.prepare(next_value, next_publication, front)
    assert first is not None
    assert port.prepare(next_value, next_publication, front) is None
    assert len(calls) == 1

    port.finish_unpresented(first)
    retry = port.prepare(next_value, next_publication, front)
    assert retry is not None
    assert len(calls) == 2
    port.finish_unpresented(retry)

    host.update_data = lambda _snapshot, **_render: (_ for _ in ()).throw(
        RuntimeError("synchronous render rejection")
    )
    refused = port.prepare(next_value, next_publication, front)
    assert refused is not None
    with pytest.raises(RuntimeError, match="synchronous render rejection"):
        refused.future.result()
    port.reject(refused, refused.future.exception())
    host.update_data = lambda snapshot, **_render: calls.append(snapshot) or Future()
    recovered = port.prepare(next_value, next_publication, front)
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

    from zlc_runtime.plane import SignalFront

    plot = pytest.importorskip("zlc_plot")
    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
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
            display_interval_ms=100,
            submit_projection=_submit_now,
            replace_host=_initial_then(host),
        )
        channels = OwnerChannels(SimpleNamespace(request_owner_wake=lambda: None))
        arbiter = SurfaceBatchArbiter(channels)

        def moment(step: int) -> SignalFront:
            stepped_value, stepped = _advanced(
                value, publication, signal, step
            )
            return SignalFront(
                {signal: stepped_value},
                {},
                {signal: stepped},
            )

        _mount(port, value, publication, moment(0))

        # Occupy the serial render worker so the next update stays queued.
        gate = Event()
        blocker = host.dispatch_control(lambda: gate.wait(10))
        newer, newest = moment(1), moment(2)
        assert arbiter.enqueue_group((port,), newer)
        assert arbiter.enqueue_group((port,), newest)
        gate.set()
        blocker.result(timeout=10)

        deadline = time.monotonic() + 10.0
        while arbiter.pending_cohorts and time.monotonic() < deadline:
            arbiter.drain(
                lambda panel_id: port if panel_id == port.panel_id else None
            )
            time.sleep(0.01)

        assert arbiter.pending_cohorts == 0
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
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None

    host = SimpleNamespace(
        host_id=object(),
        update_data=lambda _snapshot, **_render: Future(),
    )
    port = PlotPanelPort(
        "panel-1",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(host),
    )
    _mount(port, value, publication, front)
    next_value, next_publication = _advanced(value, publication, signal)
    update = port.prepare(next_value, next_publication, front)
    assert update is not None

    port.reject(update, CancelledError())
    assert port.last_error is None

    retry = port.prepare(next_value, next_publication, front)
    assert retry is not None, "the superseded update did not release its slot"
    port.reject(retry, RuntimeError("a real render failure"))
    assert isinstance(port.last_error, RuntimeError)


def test_same_snapshot_terminal_reanchors_pending_and_presented_identity(
    live_bench,
) -> None:
    """A terminal publication reusing committed bytes only reanchors identity."""

    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
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
    port = None

    def on_presented(selected: object, plot_input: object) -> None:
        presented.append(selected)
        assert port.presented_publication() is selected
        assert port.presented_input() is plot_input

    port = PlotPanelPort(
        "panel-1",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(host),
        on_presented=on_presented,
    )
    _without_deadlock(lambda: _mount(port, value, publication, front))

    live_value, live = _advanced(value, publication, signal)
    snapshot = live_value.snapshot
    update = port.prepare(live_value, live, front)
    assert update is not None

    final_value = live_value
    final = replace(
        live,
        event_ref=replace(live.event_ref, sequence=live.event_ref.sequence + 1),
        signals={**live.signals, signal: final_value},
    )
    assert _without_deadlock(lambda: port.prepare(final_value, final, front)) is None
    assert calls == [snapshot], "terminal identity must reuse the pending render"

    completion.set_result(object())
    assert port.accept(update, object()) is True
    assert port.presented_publication() is final
    assert port.presented_front_refs() == (final.event_ref,)
    assert presented[-1] is final

    terminal_record = replace(
        final,
        event_ref=replace(final.event_ref, sequence=final.event_ref.sequence + 1),
    )
    assert _without_deadlock(
        lambda: port.prepare(final_value, terminal_record, front)
    ) is None
    assert port.presented_publication() is terminal_record
    assert calls == [snapshot], "identity-only reanchor must not redraw pixels"


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
    accepted: list[tuple[object, object]] = []

    def replace_host(plot_input, _value, _publication):
        host = plot.RasterPlotHost.from_plot(plot_input, spec)
        replacements.append(host)
        return host, host.configure()

    try:
        def accepted_replacement(old, new, _publication, _input):
            if old is not None:
                accepted.append((old, new))

        port = PlotPanelPort(
            "panel-1",
            signal,
            display_interval_ms=100,
            submit_projection=_submit_now,
            replace_host=_initial_then(first, replace_host),
            accept_host=accepted_replacement,
        )
        _mount(port, value, publication, front)
        restarted = replace(
            publication,
            event_ref=EventRef(
                publication.event_ref.stream_id,
                StreamGenerationId("replacement-run"),
                publication.event_ref.sequence,
            ),
        )
        update = port.prepare(value, restarted, front)
        assert update is not None
        assert len(replacements) == 1
        assert port.host is first
        assert port.presented_publication() is publication
        operation = update.future.result(timeout=10.0)
        assert port.accept(update, operation)
        assert accepted == [(first, replacements[0])]
        assert port.host is replacements[0]
        assert port.presented_publication() is restarted
    finally:
        first.close()
        for host in replacements:
            host.close()


def test_two_panel_generation_replacements_wait_for_one_cohort_accept(
    live_bench,
) -> None:
    from concurrent.futures import Future
    from zlc_runtime.plane import SignalFront

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    restarted = replace(
        publication,
        event_ref=EventRef(
            publication.event_ref.stream_id,
            StreamGenerationId("cohort-replacement-run"),
            publication.event_ref.sequence,
        ),
    )
    restarted_front = SignalFront(
        {signal: value},
        {},
        {signal: restarted},
    )

    old = (_ClosingHost("old-a"), _ClosingHost("old-b"))
    staged: list[_ClosingHost] = []
    completions: list[Future] = []
    accepted: list[str] = []

    def stage(name: str):
        def create(_input, _value, _publication):
            host = _ClosingHost(name)
            future = Future()
            staged.append(host)
            completions.append(future)
            return host, future

        return create

    ports = (
        PlotPanelPort(
            "a",
            signal,
            display_interval_ms=100,
            submit_projection=_submit_now,
            replace_host=_initial_then(old[0], stage("new-a")),
            accept_host=lambda previous, new, _pub, _input: (
                accepted.append(new.host_id) if previous is not None else None
            ),
        ),
        PlotPanelPort(
            "b",
            signal,
            display_interval_ms=100,
            submit_projection=_submit_now,
            replace_host=_initial_then(old[1], stage("new-b")),
            accept_host=lambda previous, new, _pub, _input: (
                accepted.append(new.host_id) if previous is not None else None
            ),
        ),
    )
    for port in ports:
        _mount(port, value, publication, front)

    channels = OwnerChannels(SimpleNamespace(request_owner_wake=lambda: None))
    arbiter = SurfaceBatchArbiter(channels)
    assert arbiter.enqueue_group(ports, restarted_front)
    assert tuple(port.host for port in ports) == old
    assert not accepted

    completions[0].set_result("a")
    arbiter.drain(lambda panel_id: ports[0] if panel_id == "a" else ports[1])
    assert tuple(port.host for port in ports) == old
    assert not accepted

    completions[1].set_result("b")
    arbiter.drain(lambda panel_id: ports[0] if panel_id == "a" else ports[1])
    assert tuple(host.host_id for host in (port.host for port in ports)) == (
        "new-a",
        "new-b",
    )
    assert accepted == ["new-a", "new-b"]
    assert not any(host.closed for host in staged)


def test_two_panel_replacement_staging_failure_swaps_neither_host(
    live_bench,
) -> None:
    from concurrent.futures import Future
    from zlc_runtime.plane import SignalFront

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    restarted = replace(
        publication,
        event_ref=EventRef(
            publication.event_ref.stream_id,
            StreamGenerationId("failed-cohort-replacement"),
            publication.event_ref.sequence,
        ),
    )
    restarted_front = SignalFront(
        {signal: value},
        {},
        {signal: restarted},
    )

    old = (_ClosingHost("old-a"), _ClosingHost("old-b"))
    staged = (_ClosingHost("new-a"), _ClosingHost("new-b"))
    completions = (Future(), Future())
    accepted: list[str] = []
    ports = tuple(
        PlotPanelPort(
            panel_id,
            signal,
            display_interval_ms=100,
            submit_projection=_submit_now,
            replace_host=_initial_then(
                previous,
                lambda _input, _value, _publication, replacement=replacement,
                completion=completion: (replacement, completion),
            ),
            accept_host=lambda mounted, new, _pub, _input: (
                accepted.append(new.host_id) if mounted is not None else None
            ),
        )
        for panel_id, previous, replacement, completion in zip(
            ("a", "b"),
            old,
            staged,
            completions,
            strict=True,
        )
    )
    for port in ports:
        _mount(port, value, publication, front)
    channels = OwnerChannels(SimpleNamespace(request_owner_wake=lambda: None))
    arbiter = SurfaceBatchArbiter(channels)
    assert arbiter.enqueue_group(ports, restarted_front)
    completions[0].set_result("ready")
    completions[1].set_exception(RuntimeError("configuration failed"))
    arbiter.drain(lambda panel_id: ports[0] if panel_id == "a" else ports[1])

    assert tuple(port.host for port in ports) == old
    assert not accepted
    assert all(host.closed for host in staged)


@pytest.mark.parametrize("ending", ("finish", "reject"))
def test_abandoned_generation_replacement_closes_only_the_staged_host(
    live_bench,
    ending,
) -> None:
    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    restarted = replace(
        publication,
        event_ref=EventRef(
            publication.event_ref.stream_id,
            StreamGenerationId("abandoned-replacement-run"),
            publication.event_ref.sequence,
        ),
    )

    old = _ClosingHost("old")
    staged = _ClosingHost("staged")
    completion = Future()
    later = lambda _input, _value, _publication: (staged, completion)
    port = PlotPanelPort(
        "panel",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(old, later),
    )
    _mount(port, value, publication, front)
    update = port.prepare(value, restarted, front)
    assert update is not None
    completion.set_result("ready")
    if ending == "finish":
        port.finish_unpresented(update)
    else:
        port.reject(update, RuntimeError("staged render failed"))
    assert port.host is old
    assert port.presented_publication() is publication
    assert not old.closed
    assert staged.closed


def test_releasing_port_cancels_pending_replacement_without_swapping_host(
    live_bench,
) -> None:
    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    restarted = replace(
        publication,
        event_ref=EventRef(
            publication.event_ref.stream_id,
            StreamGenerationId("removed-panel-replacement"),
            publication.event_ref.sequence,
        ),
    )

    old = _ClosingHost("shown")
    staged = _ClosingHost("staged")
    operation = Future()
    later = lambda _input, _value, _publication: (staged, operation)
    port = PlotPanelPort(
        "panel",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(old, later),
    )
    _mount(port, value, publication, front)
    update = port.prepare(value, restarted, front)
    assert update is not None

    port.close()

    assert operation.cancelled()
    assert staged.closed
    assert port.host is old
    assert port.presented_publication() is publication
    assert not old.closed
    assert not port.has_pending


def test_board_close_releases_pending_generation_replacement(
    live_bench,
) -> None:
    from concurrent.futures import Future
    from zlc_runtime.plane import SignalFront

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    restarted = replace(
        publication,
        event_ref=EventRef(
            publication.event_ref.stream_id,
            StreamGenerationId("closed-board-replacement"),
            publication.event_ref.sequence,
        ),
    )
    restarted_front = SignalFront({signal: value}, {}, {signal: restarted})

    old = _ClosingHost("shown")
    staged = _ClosingHost("staged")
    operation = Future()
    later = lambda _input, _value, _publication: (staged, operation)
    port = PlotPanelPort(
        "panel",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(old, later),
    )
    _mount(port, value, publication, front)
    arbiter = SurfaceBatchArbiter(
        OwnerChannels(SimpleNamespace(request_owner_wake=lambda: None))
    )
    assert arbiter.enqueue_group((port,), restarted_front)

    arbiter.close()

    assert operation.cancelled()
    assert staged.closed
    assert port.host is old
    assert port.presented_publication() is publication
    assert not old.closed
    assert not port.has_pending


def test_staged_host_that_cannot_close_immediately_uses_retired_host_path(
    live_bench,
) -> None:
    from concurrent.futures import Future

    plane, node, _sequencer, _monitor = live_bench
    signal = node.signal_key("frames")
    front = plane.freeze()
    value = front.value(signal)
    publication = front.publication(signal)
    assert value is not None and publication is not None
    restarted = replace(
        publication,
        event_ref=EventRef(
            publication.event_ref.stream_id,
            StreamGenerationId("retired-staged-host"),
            publication.event_ref.sequence,
        ),
    )

    staged = SimpleNamespace(host_id="staged")
    operation = Future()
    retired: list[object] = []
    old = SimpleNamespace(host_id="shown")
    later = lambda _input, _value, _publication: (staged, operation)
    port = PlotPanelPort(
        "panel",
        signal,
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_initial_then(old, later),
        retire_host=retired.append,
    )
    _mount(port, value, publication, front)
    assert port.prepare(value, restarted, front) is not None

    port.close()

    assert retired == [staged]
    assert port.host is old


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
