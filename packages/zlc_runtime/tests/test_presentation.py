"""Pure presentation scheduler contracts."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
    SCALAR_DOMAIN,
    StreamGenerationId,
    ValueSchema,
)
from zlc_runtime.dataset_output import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime.plane import SignalFront, SignalPublication, SignalValue
from zlc_runtime.presentation import (
    BoardScheduler,
    HarmonicClock,
    SurfaceBatchArbiter,
    SurfaceUpdate,
)
from zlc_runtime.streams import EventRef, StreamId


def _front(
    name: str = "camera/frame", sequence: int = 1, *, valid: bool = True
) -> SignalFront:
    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((1,), (), ()),
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("<f8")),
    )
    block = DataBlock(
        BlockId(f"{name}-{sequence}"),
        DatasetRevision(sequence),
        np.asarray([[[float(sequence)]]], dtype=np.float64),
        VALID if valid else CellValidity(np.zeros((1, 1), dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId(f"{name}-generation")),
        block,
    )
    value = SignalValue(name, snapshot, None)
    publication = SignalPublication(
        EventRef(StreamId("camera"), StreamGenerationId("generation"), sequence),
        {name: value},
        object(),
    )
    return SignalFront(
        {name: value},
        {name: publication},
    )


class _Sink:
    def __init__(self) -> None:
        self.calls = 0

    def request_owner_wake(self) -> None:
        self.calls += 1


def test_harmonic_clock_uses_the_global_smallest_tick_and_group_maximum() -> None:
    clock = HarmonicClock((100, 200, 800))
    assert clock.base_ms == 100
    assert clock.advance() == 100
    assert not clock.group_due(100, (100, 800))
    assert clock.group_due(800, (100, 800))
    with pytest.raises(ValueError):
        HarmonicClock((100, 250))


class _Port:
    def __init__(
        self,
        panel_id: str,
        signal_name: str,
        *,
        interval: int = 100,
        fail_prepare: int = 0,
        companions: tuple[str, ...] = (),
    ) -> None:
        self.panel_id = panel_id
        self.signal_name = signal_name
        self.front_signals = (signal_name, *companions)
        self.display_interval_ms = interval
        self.presented = None
        self.presented_refs: tuple[EventRef, ...] = ()
        self.presentation_current = True
        self.fail_prepare = fail_prepare
        self.acceptable = True
        self.updates = []
        self.pending = []
        self.fronts = []
        self.accepted = []
        self.rejected = []
        self.finished = []
        self.waiting = []
        self.futures = []

    def presented_front_refs(self):
        return self.presented_refs

    @property
    def surface_busy(self):
        return bool(self.pending)

    def prepare(self, value, publication, front):
        # The front is the coherent freeze this update is drawn from; a port
        # that reads a companion signal reads it from HERE, never from the
        # plane's latest.
        self.fronts.append(front)
        if self.fail_prepare:
            self.fail_prepare -= 1
            raise ValueError("synthetic prepare failure")
        # The real port never hands the same publication to its host twice
        # while a staged render for it is still travelling toward the batch.
        if any(
            update.publication is publication for update in self.pending
        ):
            return None
        future = Future()
        update = SurfaceUpdate(
            self.panel_id,
            len(self.updates) + 1,
            object(),
            publication,
            value,
            tuple(
                front.publication(name).event_ref
                for name in self.front_signals
            ),
            future,
        )
        self.updates.append(update)
        self.pending.append(update)
        self.futures.append(future)
        return update

    def can_accept(self, _update, _operation):
        return self.acceptable

    def accept(self, update, operation):
        self.presented = update.publication
        self.presented_refs = update.front_refs
        self.accepted.append((update, operation))
        if update in self.pending:
            self.pending.remove(update)
        return True

    def reject(self, update, error):
        self.rejected.append((update, error))
        if update in self.pending:
            self.pending.remove(update)

    def finish_unpresented(self, update):
        self.finished.append(update)
        if update in self.pending:
            self.pending.remove(update)

    def report_waiting(self, missing_signal):
        self.waiting.append(missing_signal)


def test_companion_only_currency_change_schedules_the_panel_again() -> None:
    primary = _front("camera/frame", sequence=7)
    first_companion = _front("occupancy/occupied", sequence=11)

    def combined(companion: SignalFront) -> SignalFront:
        signals = {
            "camera/frame": primary.value("camera/frame"),
            "occupancy/occupied": companion.value("occupancy/occupied"),
        }
        publications = {
            "camera/frame": primary.publication("camera/frame"),
            "occupancy/occupied": companion.publication("occupancy/occupied"),
        }
        return SignalFront(signals, publications)

    plane = _Plane(combined(first_companion))
    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    port = _Port(
        "camera",
        "camera/frame",
        companions=("occupancy/occupied",),
    )
    scheduler = BoardScheduler(
        plane,
        HarmonicClock((100, 200, 400, 800)),
        arbiter,
        lambda: (port,),
    )

    scheduler.on_tick()
    port.futures[-1].set_result("first")
    arbiter.drain(lambda _panel_id: port)
    assert len(port.updates) == 1

    plane.front = combined(_front("occupancy/occupied", sequence=12))
    scheduler.on_tick()
    assert len(port.updates) == 2, (
        "a companion EventRef changed while the primary stayed current"
    )


def test_surface_arbiter_is_all_or_nothing_and_wakes_when_done() -> None:
    sink = _Sink()
    channels = sink
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")

    assert arbiter.enqueue_group((first, second), front)
    first.futures[0].set_result("first")
    second.futures[0].set_result("second")
    assert sink.calls == 2
    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))
    assert len(first.accepted) == len(second.accepted) == 1
    assert not first.rejected and not second.rejected

    failing = _Port("failing", "camera/frame")
    failing_second = _Port("failing-second", "camera/frame", fail_prepare=1)
    assert not arbiter.enqueue_group((failing, failing_second), front)
    assert len(failing.finished) == 1


def test_surface_arbiter_releases_origin_when_panel_disappears() -> None:
    front = _front()
    port = _Port("panel", "camera/frame")
    arbiter = SurfaceBatchArbiter(_Sink())

    assert arbiter.enqueue_group((port,), front)
    port.futures[0].set_result("ready")
    arbiter.drain(lambda _panel_id: None)

    assert port.finished == [port.updates[0]]
    assert not port.pending


def test_surface_arbiter_close_releases_running_origin_operation() -> None:
    front = _front()
    port = _Port("panel", "camera/frame")
    arbiter = SurfaceBatchArbiter(_Sink())

    assert arbiter.enqueue_group((port,), front)
    assert port.futures[0].set_running_or_notify_cancel()
    arbiter.close()

    assert not port.futures[0].cancelled()
    assert port.finished == [port.updates[0]]
    assert not port.pending


def test_same_shot_siblings_commit_together_in_one_cohort() -> None:
    """Sibling signals of one publication present as one atomic cohort.

    The camera now publishes ONE frames signal, so the sibling family this
    exercises is the occupancy processor's: one publication carrying several
    derived signals that must land on screen together.
    """

    first = _front("occupancy/rate")
    first_value = first.value("occupancy/rate")
    assert first_value is not None
    second_value = SignalValue(
        "occupancy/counts",
        first_value.snapshot,
        None,
        run_record=first_value.run_record,
    )
    publication = SignalPublication(
        first.publication("occupancy/rate").event_ref,
        {
            "occupancy/rate": first_value,
            "occupancy/counts": second_value,
        },
        object(),
    )
    siblings = frozenset(("occupancy/rate", "occupancy/counts"))
    front = SignalFront(
        dict(publication.signals),
        {name: publication for name in siblings},
    )
    ports = (
        _Port("first", "occupancy/rate", interval=400),
        _Port("second", "occupancy/counts", interval=400),
    )
    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    scheduler = BoardScheduler(
        _Plane(front),
        HarmonicClock((100, 200, 400, 800)),
        arbiter,
        lambda: ports,
    )

    for _ in range(3):
        scheduler.on_tick()
    assert not ports[0].updates and not ports[1].updates
    scheduler.on_tick()
    assert len(ports[0].updates) == len(ports[1].updates) == 1
    ports[0].futures[0].set_result("first")
    ports[1].futures[0].set_result("second")
    arbiter.drain(
        lambda panel_id: next(
            (port for port in ports if port.panel_id == panel_id), None
        )
    )
    assert ports[0].presented is publication
    assert ports[1].presented is publication


def test_a_failed_member_sinks_the_batch_and_is_the_only_one_blamed() -> None:
    """All-or-nothing is about PIXELS, not about blame.

    One member's failure ends the shot for the whole group, and the others
    are simply unpresented: they will draw the next shot.  Handing them the
    failing member's error is how one panel's trouble ended up marked on
    every card on the board.
    """

    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    first.futures[0].set_result("first")
    second.futures[0].set_exception(RuntimeError("worker failed"))
    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))
    assert not first.accepted and not second.accepted
    assert len(second.rejected) == 1
    assert str(second.rejected[0][1]) == "worker failed"
    assert not first.rejected and len(first.finished) == 1


def test_a_rebuilt_panel_marks_nobody_at_all() -> None:
    """A panel whose host was replaced refuses the update staged for the old
    one.  That is the currency guard doing its job, on a panel that is about
    to draw against its new host -- so the shot leaves unpresented and no
    card, not even the rebuilt one's, is marked with anything.
    """

    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    healthy = _Port("healthy", "camera/frame")
    rebuilt = _Port("rebuilt", "camera/frame")
    assert arbiter.enqueue_group((healthy, rebuilt), front)
    healthy.futures[0].set_result("healthy")
    rebuilt.futures[0].set_result("rebuilt")
    rebuilt.acceptable = False  # its host was replaced under the staged render
    arbiter.drain(
        lambda panel_id: {"healthy": healthy, "rebuilt": rebuilt}.get(panel_id)
    )
    assert not healthy.accepted and not rebuilt.accepted
    assert not rebuilt.rejected and len(rebuilt.finished) == 1
    assert not healthy.rejected and len(healthy.finished) == 1


def test_a_superseded_member_abandons_its_whole_batch_without_an_error() -> None:
    """A latest-only host cancels a queued render when a newer frame arrives.

    That is flow control, not failure -- so no member is rejected and no panel
    goes red.  But the batch is one causal group frozen from one front:
    presenting the surviving sibling now would put it one shot ahead of the
    member whose render was coalesced away.  The whole batch therefore leaves
    unpresented, and the newer batch already queued behind it presents every
    member of the group together.
    """

    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    assert first.futures[0].cancel()
    second.futures[0].set_result("second")

    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))

    assert first.finished == [first.updates[0]]
    assert second.finished == [second.updates[0]]
    assert not first.rejected and not second.rejected
    assert not first.accepted and not second.accepted
    assert second.presented is None


def test_a_render_that_raised_cancellation_is_superseded_not_failed() -> None:
    """The worker may surface its own supersession as a raised CancelledError."""

    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    first.futures[0].set_exception(CancelledError())
    second.futures[0].set_result("second")

    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))

    assert first.finished == [first.updates[0]]
    assert second.finished == [second.updates[0]]
    assert not first.rejected and not second.rejected
    assert not first.accepted and not second.accepted


def test_a_batch_whose_every_member_was_superseded_just_finishes() -> None:
    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    assert first.futures[0].cancel()
    assert second.futures[0].cancel()

    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))

    assert first.finished == [first.updates[0]]
    assert second.finished == [second.updates[0]]
    assert not first.rejected and not second.rejected
    assert not first.accepted and not second.accepted


def test_a_sibling_error_still_never_marks_the_superseded_member() -> None:
    """One member superseded, one failed: the batch is abandoned, not rejected.

    Supersession means a newer batch for the same group is already queued;
    that batch re-renders the failed member too, so its error resurfaces
    there if it is real -- while a red mark for an abandoned batch would be a
    complaint about pixels nobody was going to show.
    """

    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    assert first.futures[0].cancel()
    second.futures[0].set_exception(RuntimeError("worker failed"))

    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))

    assert first.finished == [first.updates[0]]
    assert second.finished == [second.updates[0]]
    assert not first.rejected and not second.rejected


class _Plane:
    def __init__(self, front: SignalFront) -> None:
        self.front = front
        self.latest = {}
        self.freezes = 0
        self.front_signals: frozenset[str] = frozenset()
        self.edges: frozenset[tuple[str, str]] = frozenset()
        self.parents: dict[object, tuple] = {}

    def set_front_signals(self, signal_names) -> None:
        self.front_signals = frozenset(signal_names)

    def freeze(self):
        self.freezes += 1
        return self.front

    def latest_publication(self, signal_name):
        return self.latest.get(signal_name, self.front.publication(signal_name))

    def publication_roots(self, publication):
        roots = set()
        pending = [publication]
        while pending:
            current = pending.pop()
            parents = self.parents.get(current, ())
            if parents:
                pending.extend(parents)
            else:
                roots.add(current.event_ref)
        return frozenset(roots)

    def follower_edges(self):
        return self.edges


def test_due_panel_stages_on_the_next_publication_wake_without_advancing_clock() -> None:
    old = _front("camera/frame", sequence=1)
    new = _front("camera/frame", sequence=2)
    old_publication = old.publication("camera/frame")
    assert old_publication is not None
    plane = _Plane(old)
    port = _Port("camera", "camera/frame")
    port.presented = old_publication
    port.presented_refs = (old_publication.event_ref,)
    arbiter = SurfaceBatchArbiter(_Sink())
    scheduler = BoardScheduler(
        plane,
        HarmonicClock((100, 200)),
        arbiter,
        lambda: (port,),
    )

    # The deadline arrives before the source publication.  It remains a
    # lightweight admission debt; there is no stale frame to render.
    scheduler.on_tick()
    assert not port.updates

    # Publication may arrive while Pause is active.  Existing travelling
    # cohorts may finish, but a not-yet-admitted source must remain frozen.
    plane.front = new
    scheduler.stage_owed(admit_new=False)
    assert not port.updates

    # The ordinary owner wake spends the authored deadline immediately and
    # does not consume a second HarmonicClock tick.
    scheduler.stage_owed()
    assert len(port.updates) == 1

    # A later due revision remains only Plane latest while this heavy surface
    # travels; it cannot become a second full-frame/Fit queue entry.
    newest = _front("camera/frame", sequence=3)
    plane.front = newest
    scheduler.on_tick()
    assert len(port.updates) == 1

    port.futures[0].set_result("revision-2")
    arbiter.drain(lambda _panel_id: port)
    scheduler.stage_owed()
    assert len(port.updates) == 2
    assert port.updates[-1].publication is newest.publication("camera/frame")


def test_explicit_presentation_debt_restages_an_unchanged_screen_front() -> None:
    front = _front("camera/frame", sequence=1)
    publication = front.publication("camera/frame")
    assert publication is not None
    plane = _Plane(front)
    port = _Port("camera", "camera/frame")
    port.presented = publication
    port.presented_refs = (publication.event_ref,)
    port.presentation_current = False
    arbiter = SurfaceBatchArbiter(_Sink())
    scheduler = BoardScheduler(
        plane,
        HarmonicClock((100, 200)),
        arbiter,
        lambda: (port,),
    )

    scheduler.invalidate_presentations((port.panel_id,))
    scheduler.stage_owed()

    assert len(port.updates) == 1
    assert port.updates[0].publication is publication
    assert port.presented_refs == (publication.event_ref,)


def test_two_views_of_one_signal_flip_together_as_one_cohort() -> None:
    """Two views of one signal are one shot on screen — they flip as one.

    Each panel stages its own batch at its own cadence, but the batches
    carry the same shot roots and the arbiter assembles them into ONE
    cohort: both present in one accept pass, and when either view's render
    is coalesced away for a newer frame the WHOLE cohort leaves
    unpresented — the group skips to the newest shot together instead of
    one view running a shot ahead of the other.
    """

    front = _front()
    plane = _Plane(front)
    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    fast = _Port("fast", "camera/frame", interval=100)
    slow = _Port("slow", "camera/frame", interval=100)
    clock = HarmonicClock((100, 200, 400, 800))
    scheduler = BoardScheduler(plane, clock, arbiter, lambda: (fast, slow))

    scheduler.on_tick()
    assert len(fast.updates) == len(slow.updates) == 1

    # The slow view's render is coalesced away: the whole shot leaves
    # unpresented; the newer shot presents both views together.
    assert slow.futures[0].cancel()
    fast.futures[0].set_result("fast")
    arbiter.drain(lambda panel_id: {"fast": fast, "slow": slow}.get(panel_id))
    assert not fast.accepted and not slow.accepted
    assert not fast.rejected and not slow.rejected
    assert fast.finished == [fast.updates[0]]
    assert slow.finished == [slow.updates[0]]


def test_a_displayed_follower_joins_its_shot_within_the_open_window() -> None:
    """A fit-signal batch staged one tick late still flips with its shot.

    The pair engine publishes a panel's fit signal during the shot's own
    commit, so the rolling trace of that fit stages on the NEXT tick.  With
    both sides of the follower edge on the board, the shot's cohort keeps
    its join window open one extra tick boundary: the source's batch waits,
    the follower's batch joins, and the whole shot presents in one accept
    pass — never the camera one shot ahead of its own fit trace.
    """

    camera_front = _front("camera/frame", sequence=7)
    camera_publication = camera_front.publication("camera/frame")
    assert camera_publication is not None
    fit_front = _front("@logic/panel/center", sequence=1)
    fit_publication = fit_front.publication("@logic/panel/center")
    assert fit_publication is not None
    object.__setattr__(
        fit_publication,
        "direct_parent_refs",
        (camera_publication.event_ref,),
    )
    both = SignalFront(
        {
            "camera/frame": camera_front.value("camera/frame"),
            "@logic/panel/center": fit_front.value("@logic/panel/center"),
        },
        {
            "camera/frame": camera_publication,
            "@logic/panel/center": fit_publication,
        },
    )

    plane = _Plane(camera_front)
    plane.edges = frozenset({("camera/frame", "@logic/panel/center")})
    plane.parents = {fit_publication: (camera_publication,)}
    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    camera = _Port("camera", "camera/frame", interval=100)
    trace = _Port("trace", "@logic/panel/center", interval=100)
    clock = HarmonicClock((100, 200, 400, 800))
    scheduler = BoardScheduler(plane, clock, arbiter, lambda: (camera, trace))

    # Tick 1: the camera's pair is on the plane, the follower is not yet.
    scheduler.on_tick()
    assert len(camera.updates) == 1 and not trace.updates
    camera.futures[0].set_result("camera")
    arbiter.drain(lambda panel_id: {"camera": camera, "trace": trace}.get(panel_id))
    # The window is open: the complete camera batch waits for its follower.
    assert not camera.accepted

    # The follower publication arrives during the source render.  Its owner
    # wake stages the already-due follower immediately, without consuming a
    # second 100 ms display tick.
    plane.front = both
    scheduler.stage_owed()
    assert len(trace.updates) == 1
    trace.futures[0].set_result("trace")
    arbiter.drain(lambda panel_id: {"camera": camera, "trace": trace}.get(panel_id))
    assert camera.presented is camera_publication
    assert trace.presented is fit_publication
    assert len(camera.accepted) == len(trace.accepted) == 1


def test_completion_wake_does_not_bypass_a_not_due_follower() -> None:
    camera_front = _front("camera/frame", sequence=7)
    camera_publication = camera_front.publication("camera/frame")
    fit_front = _front("@logic/panel/center", sequence=1)
    fit_publication = fit_front.publication("@logic/panel/center")
    assert camera_publication is not None and fit_publication is not None
    object.__setattr__(
        fit_publication,
        "direct_parent_refs",
        (camera_publication.event_ref,),
    )
    both = SignalFront(
        {
            "camera/frame": camera_front.value("camera/frame"),
            "@logic/panel/center": fit_front.value("@logic/panel/center"),
        },
        {
            "camera/frame": camera_publication,
            "@logic/panel/center": fit_publication,
        },
    )
    plane = _Plane(camera_front)
    plane.edges = frozenset({("camera/frame", "@logic/panel/center")})
    plane.parents = {fit_publication: (camera_publication,)}
    arbiter = SurfaceBatchArbiter(_Sink())
    camera = _Port("camera", "camera/frame", interval=100)
    trace = _Port("trace", "@logic/panel/center", interval=200)
    scheduler = BoardScheduler(
        plane,
        HarmonicClock((100, 200)),
        arbiter,
        lambda: (camera, trace),
    )

    scheduler.on_tick()
    camera.futures[0].set_result("camera")
    arbiter.drain(lambda panel_id: {"camera": camera, "trace": trace}.get(panel_id))
    assert len(camera.accepted) == 1
    plane.front = both
    scheduler.stage_owed()
    assert not trace.updates


def test_due_coherent_component_stages_on_its_completion_wake() -> None:
    old_camera = _front("camera/frame", sequence=1)
    old_roi = _front("roi/frame", sequence=1)
    new_camera = _front("camera/frame", sequence=2)
    new_roi = _front("roi/frame", sequence=2)
    old_camera_publication = old_camera.publication("camera/frame")
    old_roi_publication = old_roi.publication("roi/frame")
    new_camera_publication = new_camera.publication("camera/frame")
    new_roi_publication = new_roi.publication("roi/frame")
    assert (
        old_camera_publication is not None
        and old_roi_publication is not None
        and new_camera_publication is not None
        and new_roi_publication is not None
    )

    def combined(camera: SignalFront, roi: SignalFront) -> SignalFront:
        return SignalFront(
            {
                "camera/frame": camera.value("camera/frame"),
                "roi/frame": roi.value("roi/frame"),
            },
            {
                "camera/frame": camera.publication("camera/frame"),
                "roi/frame": roi.publication("roi/frame"),
            },
        )

    plane = _Plane(combined(old_camera, old_roi))
    plane.parents = {
        old_roi_publication: (old_camera_publication,),
        new_roi_publication: (new_camera_publication,),
    }
    plane.latest["camera/frame"] = new_camera_publication
    camera = _Port("camera", "camera/frame")
    roi = _Port("roi", "roi/frame")
    camera.presented = old_camera_publication
    camera.presented_refs = (old_camera_publication.event_ref,)
    roi.presented = old_roi_publication
    roi.presented_refs = (old_roi_publication.event_ref,)
    arbiter = SurfaceBatchArbiter(_Sink())
    scheduler = BoardScheduler(
        plane,
        HarmonicClock((100, 200)),
        arbiter,
        lambda: (camera, roi),
    )

    # The due tick sees the new source publication, but the coherent front is
    # intentionally held on the old shot until its Area-ROI sibling exists.
    scheduler.on_tick()
    assert not camera.updates and not roi.updates

    # The processor completion advances the coherent component.  That completion
    # wake spends the existing cadence decision; it does not need another
    # HarmonicClock tick, and the two same-shot surfaces stay atomic.
    plane.front = combined(new_camera, new_roi)
    scheduler.stage_owed()
    assert len(camera.updates) == len(roi.updates) == 1
    camera.futures[0].set_result("camera")
    roi.futures[0].set_result("roi")
    arbiter.drain(lambda panel_id: {"camera": camera, "roi": roi}.get(panel_id))
    assert len(camera.accepted) == len(roi.accepted) == 1


def test_board_scheduler_declares_its_port_signals_on_every_tick() -> None:
    """The reader declares what it reads: the port list IS the front request.

    An undeclared plane builds no lineage components, so every signal floats
    at its own latest publication -- a camera panel one shot ahead of the
    panel derived from it.  The scheduler is the sole declaration authority;
    no membership bookkeeping may exist beside the ports.
    """

    front = _front()
    plane = _Plane(front)
    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    ports: list[_Port] = [_Port("panel", "camera/frame", interval=100)]
    clock = HarmonicClock((100, 200, 400, 800))
    scheduler = BoardScheduler(plane, clock, arbiter, lambda: tuple(ports))

    scheduler.on_tick()
    assert plane.front_signals == frozenset({"camera/frame"})

    ports.append(_Port("second", "roi/value", interval=100))
    scheduler.on_tick()
    assert plane.front_signals == frozenset({"camera/frame", "roi/value"})

    ports.pop(0)
    scheduler.on_tick()
    assert plane.front_signals == frozenset({"roi/value"})


def test_board_scheduler_owes_a_failed_slow_beat_to_the_next_base_tick() -> None:
    front = _front()
    plane = _Plane(front)
    sink = _Sink()
    channels = sink
    arbiter = SurfaceBatchArbiter(channels)
    port = _Port("panel", "camera/frame", interval=2000, fail_prepare=1)
    clock = HarmonicClock((100, 2000))
    scheduler = BoardScheduler(plane, clock, arbiter, lambda: (port,))

    for _ in range(19):
        scheduler.on_tick()
    scheduler.on_tick()  # elapsed 2000: the due prepare fails
    assert not port.updates

    scheduler.on_tick()  # elapsed 2100: not due, but owed and retried
    assert len(port.updates) == 1
    port.futures[0].set_result("ready")
    arbiter.drain(lambda _panel_id: port)
    assert len(port.accepted) == 1

def test_board_scheduler_owes_missing_value_until_the_next_base_tick() -> None:
    empty = SignalFront({})
    complete = _front()
    plane = _Plane(empty)
    channels = _Sink()
    arbiter = SurfaceBatchArbiter(channels)
    port = _Port("panel", "camera/frame", interval=2000)
    clock = HarmonicClock((100, 2000))
    scheduler = BoardScheduler(plane, clock, arbiter, lambda: (port,))
    for _ in range(20):
        scheduler.on_tick()
    assert port.waiting == ["camera/frame"]

    plane.front = complete
    scheduler.on_tick()
    port.futures[0].set_result("ready")
    arbiter.drain(lambda _panel_id: port)
    assert len(port.accepted) == 1


def test_a_paused_display_still_freezes_the_plane_and_advances_the_clock() -> None:
    """Pause withholds STAGING.  It does not idle the instrument.

    on_tick does three things in one call: it freezes a front, it advances
    the display clock, and it stages what is due.  A console that wanted to
    stop the picture used to skip the whole call -- and the freeze is not a
    read, it is the sole pump of the latest-only processor lane, so pausing
    the display also stopped every selection- and fit-derived signal from
    being computed, and stopped the clock that decides when anything is due
    again.

    Withholding staging must accumulate nothing: group_due is a pure
    function of the elapsed clock, so a resumed board stages on its own next
    boundary with its cadence phase intact.
    """

    plane = _Plane(_front())
    arbiter = SurfaceBatchArbiter(_Sink())
    port = _Port("camera", "camera/frame")
    clock = HarmonicClock((100, 200, 400, 800))
    scheduler = BoardScheduler(plane, clock, arbiter, lambda: (port,))

    scheduler.on_tick(stage=False)
    assert plane.freezes == 1, "a paused display must still pump the plane"
    assert clock.base_ms > 0
    assert port.updates == [], "a paused display must stage nothing"

    for _ in range(3):
        scheduler.on_tick(stage=False)
    assert plane.freezes == 4
    assert port.updates == []

    # Resumed, the very next boundary stages -- the phase was never lost.
    staged = 0
    for _ in range(8):
        scheduler.on_tick()
        if port.updates:
            staged = len(port.updates)
            break
    assert staged, "a resumed board must stage on its own next boundary"


def test_a_missing_companion_holds_the_group_until_completion_wake() -> None:

    front = _front("camera/frame")
    arbiter = SurfaceBatchArbiter(_Sink())
    port = _Port("camera", "camera/frame", companions=("occ/sites",))

    assert not arbiter.enqueue_group((port,), front)
    assert port.waiting == ["occ/sites"]
    assert port.futures == []
    # Even when the coherent front withholds the camera too, report the
    # missing companion instead of implying the camera stopped producing.
    assert not arbiter.enqueue_group((port,), SignalFront({}))
    assert port.waiting[-1] == "occ/sites"

    plane = _Plane(front)
    raw = _Port("raw", "camera/frame")
    ports = {raw.panel_id: raw, port.panel_id: port}
    scheduler = BoardScheduler(
        plane, HarmonicClock((100, 2000)), arbiter, lambda: tuple(ports.values())
    )
    scheduler.on_tick()
    assert raw.futures == port.futures == []
    scheduler.stage_owed()
    assert raw.futures == port.futures == []
    # An explicitly invalid result has completed; it must not hold the image
    # or keep the previous judgement alive. The renderer consumes validity.
    companion = _front("occ/sites", valid=False)
    plane.front = SignalFront(
        {**front.signals, **companion.signals},
        {**front.publication_by_signal, **companion.publication_by_signal},
    )
    scheduler.stage_owed()
    assert len(raw.futures) == len(port.futures) == 1
    raw.futures[-1].set_result("complete raw image")
    arbiter.drain(ports.get)
    assert raw.accepted == port.accepted == []
    port.futures[-1].set_result("complete image and overlay")
    arbiter.drain(ports.get)
    assert len(raw.accepted) == len(port.accepted) == 1
