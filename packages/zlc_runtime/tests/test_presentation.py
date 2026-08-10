"""Pure presentation scheduler contracts and window-runtime helpers."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from pathlib import Path
import threading

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    StreamGenerationId,
    ValueSchema,
)
from zlc_runtime.dataset_output import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime.plane import SignalFront, SignalPublication, SignalValue
from zlc_runtime.presentation import (
    BoardScheduler,
    HarmonicClock,
    OwnerChannels,
    OwnerTurn,
    SurfaceBatchArbiter,
    SurfaceUpdate,
)
from zlc_runtime.streams import EventRef, StreamId
from zlc_runtime.window_runtime import (
    cancel_export_commits,
    stage_and_replace_export,
    submit_compute,
)


def _front(name: str = "camera/frame", sequence: int = 1) -> SignalFront:
    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    schema = DatasetSchema(repeat, PointTable(1), None, ValueSchema.scalar(np.dtype("<f8")))
    block = DataBlock(
        BlockId(f"{name}-{sequence}"),
        DatasetRevision(sequence),
        np.asarray([[[float(sequence)]]], dtype=np.float64),
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId(f"{name}-generation")),
        block,
    )
    value = SignalValue(name, snapshot, None, transient=True)
    publication = SignalPublication(
        EventRef(StreamId("camera"), StreamGenerationId("generation"), sequence),
        {name: value},
        object(),
    )
    return SignalFront(
        {name: value},
        {},
        {name: publication},
        {name: frozenset((name,))},
    )


class _Sink:
    def __init__(self) -> None:
        self.calls = 0

    def request_owner_wake(self) -> None:
        self.calls += 1


class _WakePlane:
    def __init__(self) -> None:
        self.callback = None
        self.token = object()
        self.unbound = []

    def bind_owner_wake(self, callback):
        self.callback = callback
        return self.token

    def unbind_owner_wake(self, token):
        assert token is self.token
        self.unbound.append(token)


def test_owner_channels_coalesce_and_borrow_data_wake() -> None:
    sink = _Sink()
    channels = OwnerChannels(sink)
    plane = _WakePlane()
    channels.activate_data(plane)
    plane.callback()
    channels.notify_lifecycle()
    channels.notify_surface()
    assert sink.calls == 3
    assert channels.take() == OwnerTurn(True, True, True)
    assert channels.take() == OwnerTurn(False, False, False)

    channels.deactivate_data()
    assert plane.unbound == [plane.token]
    channels.close()
    channels.notify_lifecycle()
    assert sink.calls == 3
    assert channels.take() == OwnerTurn(False, False, False)


def test_harmonic_clock_uses_the_global_smallest_tick_and_group_maximum() -> None:
    clock = HarmonicClock((100, 200, 800), 200)
    assert clock.base_ms == 100
    assert clock.advance() == 100
    assert not clock.group_due(100, (100, 800))
    assert clock.group_due(800, (100, 800))
    assert clock.rebase((100, 800)) == 100
    assert clock.elapsed_ms == 0
    assert clock.advance() == 100
    with pytest.raises(ValueError):
        clock.rebase((300,))
    with pytest.raises(ValueError):
        HarmonicClock((100, 250), 100)


class _Port:
    def __init__(
        self,
        panel_id: str,
        signal_name: str,
        *,
        interval: int = 100,
        fail_prepare: int = 0,
    ) -> None:
        self.panel_id = panel_id
        self.signal_name = signal_name
        self.display_interval_ms = interval
        self.presented = None
        self.fail_prepare = fail_prepare
        self.acceptable = True
        self.updates = []
        self.observed = []
        self.accepted = []
        self.rejected = []
        self.finished = []
        self.waiting = []
        self.futures = []

    def presented_publication(self):
        return self.presented

    def prepare(self, value, publication):
        if self.fail_prepare:
            self.fail_prepare -= 1
            raise ValueError("synthetic prepare failure")
        future = Future()
        update = SurfaceUpdate(
            self.panel_id,
            len(self.updates) + 1,
            object(),
            publication,
            value,
            future,
            False,
        )
        self.updates.append(update)
        self.futures.append(future)
        return update

    def observe(self, update, operation):
        self.observed.append((update, operation))

    def can_accept(self, _update, _operation):
        return self.acceptable

    def accept(self, update, operation):
        self.presented = update.publication
        self.accepted.append((update, operation))
        return True

    def reject(self, update, error):
        self.rejected.append((update, error))

    def finish_unpresented(self, update):
        self.finished.append(update)

    def report_waiting(self, missing_signal):
        self.waiting.append(missing_signal)


def test_surface_arbiter_is_all_or_nothing_and_wakes_when_done() -> None:
    sink = _Sink()
    channels = OwnerChannels(sink)
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")

    assert arbiter.enqueue_group((first, second), front)
    assert arbiter.pending_batches == 1
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
    assert arbiter.pending_batches == 0


def test_same_shot_siblings_wait_for_one_global_due_tick_and_commit_together() -> None:
    first = _front("camera/frame_0")
    first_value = first.value("camera/frame_0")
    assert first_value is not None
    second_value = SignalValue(
        "camera/frame_1",
        first_value.snapshot,
        None,
        transient=True,
        run_record=first_value.run_record,
    )
    publication = SignalPublication(
        first.publication("camera/frame_0").event_ref,
        {
            "camera/frame_0": first_value,
            "camera/frame_1": second_value,
        },
        object(),
    )
    siblings = frozenset(("camera/frame_0", "camera/frame_1"))
    front = SignalFront(
        dict(publication.signals),
        {},
        {name: publication for name in siblings},
        {name: siblings for name in siblings},
    )
    ports = (
        _Port("first", "camera/frame_0", interval=100),
        _Port("second", "camera/frame_1", interval=400),
    )
    channels = OwnerChannels(_Sink())
    arbiter = SurfaceBatchArbiter(channels)
    scheduler = BoardScheduler(
        _Plane(front),
        HarmonicClock((100, 200, 400, 800), 400),
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
    scheduler.on_owner_turn(lambda: None)
    assert ports[0].presented is publication
    assert ports[1].presented is publication


def test_surface_arbiter_rejects_the_whole_completed_batch_once() -> None:
    channels = OwnerChannels(_Sink())
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    first.futures[0].set_result("first")
    second.futures[0].set_exception(RuntimeError("worker failed"))
    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))
    assert not first.accepted and not second.accepted
    assert len(first.rejected) == len(second.rejected) == 1


def test_a_superseded_member_finishes_quietly_and_its_siblings_still_commit() -> None:
    """A latest-only host cancels a queued render when a newer frame arrives.

    That is flow control -- the newer render is already queued on the same
    host -- so the cancelled member must leave as unpresented, without an
    error, and without dragging the siblings' completed renders down with it.
    Rejected as a batch error, every panel on that camera went red once per
    coalesced frame, saying nothing.
    """

    channels = OwnerChannels(_Sink())
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    assert first.futures[0].cancel()
    second.futures[0].set_result("second")

    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))

    assert first.finished == [first.updates[0]]
    assert not first.rejected and not second.rejected
    assert not first.accepted and len(second.accepted) == 1
    assert second.presented is front.publication("camera/frame")
    assert arbiter.pending_batches == 0


def test_a_render_that_raised_cancellation_is_superseded_not_failed() -> None:
    """The worker may surface its own supersession as a raised CancelledError."""

    channels = OwnerChannels(_Sink())
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    first.futures[0].set_exception(CancelledError())
    second.futures[0].set_result("second")

    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))

    assert first.finished == [first.updates[0]]
    assert not first.rejected and not second.rejected
    assert len(second.accepted) == 1


def test_a_batch_whose_every_member_was_superseded_just_finishes() -> None:
    channels = OwnerChannels(_Sink())
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
    assert arbiter.pending_batches == 0


def test_a_sibling_error_still_never_marks_the_superseded_member() -> None:
    """One member superseded, one failed: only the failure is an error."""

    channels = OwnerChannels(_Sink())
    arbiter = SurfaceBatchArbiter(channels)
    front = _front()
    first = _Port("one", "camera/frame")
    second = _Port("two", "camera/frame")
    assert arbiter.enqueue_group((first, second), front)
    assert first.futures[0].cancel()
    second.futures[0].set_exception(RuntimeError("worker failed"))

    arbiter.drain(lambda panel_id: {"one": first, "two": second}.get(panel_id))

    assert first.finished == [first.updates[0]]
    assert not first.rejected
    assert len(second.rejected) == 1


class _Plane:
    def __init__(self, front: SignalFront) -> None:
        self.front = front
        self.freezes = 0

    def freeze(self):
        self.freezes += 1
        return self.front


def test_board_scheduler_owes_a_failed_slow_beat_to_the_next_base_tick() -> None:
    front = _front()
    plane = _Plane(front)
    sink = _Sink()
    channels = OwnerChannels(sink)
    arbiter = SurfaceBatchArbiter(channels)
    port = _Port("panel", "camera/frame", interval=2000, fail_prepare=1)
    clock = HarmonicClock((100, 2000), 2000)
    clock.rebase((100, 2000))
    scheduler = BoardScheduler(plane, clock, arbiter, lambda: (port,))

    for _ in range(19):
        scheduler.on_tick()
    assert not scheduler.owed_groups
    scheduler.on_tick()  # elapsed 2000: the due prepare fails
    assert scheduler.owed_groups
    assert not port.updates

    scheduler.on_tick()  # elapsed 2100: not due, but owed and retried
    assert not scheduler.owed_groups
    assert len(port.updates) == 1
    port.futures[0].set_result("ready")
    scheduler.on_owner_turn(lambda: None)
    assert len(port.accepted) == 1


def test_board_scheduler_owes_missing_value_until_the_next_base_tick() -> None:
    empty = SignalFront({}, {})
    complete = _front()
    plane = _Plane(empty)
    channels = OwnerChannels(_Sink())
    arbiter = SurfaceBatchArbiter(channels)
    port = _Port("panel", "camera/frame", interval=2000)
    clock = HarmonicClock((100, 2000), 2000)
    clock.rebase((100, 2000))
    scheduler = BoardScheduler(plane, clock, arbiter, lambda: (port,))
    for _ in range(20):
        scheduler.on_tick()
    assert scheduler.owed_groups
    assert port.waiting == ["camera/frame"]

    plane.front = complete
    scheduler.on_tick()
    assert not scheduler.owed_groups
    port.futures[0].set_result("ready")
    scheduler.on_owner_turn(lambda: None)
    assert len(port.accepted) == 1


def test_window_runtime_compute_and_atomic_export(tmp_path: Path) -> None:
    assert submit_compute(lambda value: value + 1, 4).result(timeout=1.0) == 5
    assert submit_compute(lambda: "interactive", latency_sensitive=True).result(timeout=1.0) == "interactive"

    destination = tmp_path / "export.txt"
    cancelled = threading.Event()
    lock = threading.Lock()
    result = stage_and_replace_export(
        destination,
        write_staged=lambda path: path.write_text("ready", encoding="utf-8"),
        cancelled=cancelled,
        commit_lock=lock,
    )
    assert result == destination
    assert destination.read_text(encoding="utf-8") == "ready"
    cancel_export_commits(cancelled=cancelled, commit_lock=lock)
    with pytest.raises(Exception):
        stage_and_replace_export(
            destination,
            write_staged=lambda path: path.write_text("late", encoding="utf-8"),
            cancelled=cancelled,
            commit_lock=lock,
        )
