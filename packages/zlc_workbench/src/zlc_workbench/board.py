"""Driving a board of live panels: the tick, the wake, and who owns which.

Three parties, and the whole difficulty is that they run on different threads:

* the SCHEDULER freezes one signal front per tick and asks each panel to prepare
  from it.  It must not run on the GUI thread -- preparing means drawing.
* the ARBITER holds prepared surfaces until a whole board is ready, then hands
  them over together.  Committing touches widgets, so it must run ON the GUI
  thread.
* the WAKE carries "there is something to commit" from the first to the second,
  coalescing so a burst of ready surfaces causes one turn, not a queue of them.

Nothing ticked and nothing polled before this: the scheduler existed, the wake
primitive existed, and no code connected them, so a live panel could never
update no matter how correct either side was.

Qt appears only in the shim at the bottom, and only to hop threads.  Everything
above it is ordinary Python and can be driven by a test loop, which is how the
seam is verified without a display.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from zlc_runtime import (
    BoardScheduler,
    HarmonicClock,
    SurfaceBatchArbiter,
)

_LOG = logging.getLogger(__name__)


__all__ = [
    "LiveBoard",
    "OwnerWake",
    "attach_qt",
    "attach_qt_owner_turn",
    "attach_qt_worker",
]


class OwnerWake:
    """Coalescing wake: many requests, one turn.

    The runtime asks to be woken; the owner decides when to take the turn.  A
    burst of surfaces finishing must not queue a turn each, or the GUI thread
    spends its time servicing wakes instead of drawing.
    """

    def __init__(self, notify: Callable[[], None] | None = None) -> None:
        self._lock = Lock()
        self._pending = False
        self._notify = notify

    def set_notify(self, notify: Callable[[], None] | None) -> None:
        """Bind the owner-turn trigger after composition has one to give.

        The board exists before the event-loop shim that can hop threads, so
        the trigger arrives late.  Without one, a wake only raises the pending
        flag and the next beat takes the turn -- which is exactly the headless
        behavior every test drives.
        """

        with self._lock:
            self._notify = notify
            pending = self._pending
        if pending and notify is not None:
            notify()

    def request_owner_wake(self) -> None:
        with self._lock:
            if self._pending:
                return
            self._pending = True
            notify = self._notify
        if notify is not None:
            notify()

    def take(self) -> bool:
        """Claim a pending wake, if there is one."""

        with self._lock:
            pending, self._pending = self._pending, False
            return pending

class LiveBoard:
    """A set of panels kept up to date from one signal plane."""

    def __init__(
        self,
        plane: object,
        ports: Callable[[], Sequence[Any]],
        *,
        intervals: Sequence[int],
        notify: Callable[[], None] | None = None,
    ) -> None:
        subscribe = getattr(plane, "subscribe_publications", None)
        if not callable(subscribe):
            raise TypeError("live board requires publication subscription")
        self._closed = False
        self._closing = False
        self._projection_lock = Lock()
        self._projection_futures: set[object] = set()
        # Canonical run assembly and companion projection happen before a
        # RasterPlotHost can accept its PlotInput.  They are display work, so
        # one board-owned worker performs them at the board cadence instead of
        # making the Qt timer callback copy every finite run prefix.
        self._projection_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="zlc-presentation",
        )
        self.wake = OwnerWake(notify)
        self._arbiter = SurfaceBatchArbiter(self.wake)
        self._ports = ports
        self._clock = HarmonicClock(tuple(intervals))
        self._scheduler = BoardScheduler(
            plane,
            self._clock,
            self._arbiter,
            ports,
        )
        self._unsubscribe_publications = subscribe(
            self.wake.request_owner_wake,
        )

    @property
    def intervals(self) -> tuple[int, ...]:
        """The exact refresh domain enforced by this board's scheduler."""

        return self._clock.intervals

    @property
    def base_interval_ms(self) -> int:
        """The clock base: the wall-time cadence the beat must be driven at.

        ``HarmonicClock.advance`` credits one base per tick, so every panel's
        labeled refresh interval is only truthful when the timer that drives
        the beat fires at exactly this period.
        """

        return self._clock.base_ms

    def tick(self) -> object:
        """Freeze one front and stage whatever is due.  NOT the GUI thread."""

        return self._scheduler.on_tick()

    def submit_projection(self, project: Callable[[], object]):
        """Submit one coalesced panel input projection off the owner thread."""

        if not callable(project):
            raise TypeError("panel projection must be callable")
        with self._projection_lock:
            if self._closing:
                raise RuntimeError("live board is closing")
            future = self._projection_executor.submit(project)
            self._projection_futures.add(future)

        def finished(completed: object) -> None:
            with self._projection_lock:
                self._projection_futures.discard(completed)
                closing = self._closing
            if closing:
                self.wake.request_owner_wake()

        future.add_done_callback(finished)
        return future

    def owe_presentation(self, panel_ids: Sequence[str]) -> None:
        """Owe these panels a staging pass regardless of their cadence.

        The display interval paces what the bench PUBLISHES; it must not
        pace what the operator ASKS for.  Refresh used to hope the next
        natural beat covered the panel, which cost a whole cadence when
        the beat was not due.  Owed panels stage on the next tick whatever
        the clock says -- the same debt a held component already uses.
        """

        selected = tuple(str(value) for value in panel_ids)
        if not selected:
            return
        self._scheduler.invalidate_presentations(selected)
        self.wake.request_owner_wake()

    def invalidate_presentations(
        self,
        panel_ids: Sequence[str],
        *,
        targets: Mapping[str, object] | None = None,
    ) -> None:
        """Atomically change representation identity and create display debt."""

        selected = tuple(str(value) for value in panel_ids)
        if not selected:
            return
        ports = {port.panel_id: port for port in self._ports()}
        active: list[str] = []
        for panel_id in selected:
            port = ports.get(panel_id)
            if port is None:
                continue
            if targets is not None and panel_id in targets:
                port.invalidate_presentation(targets[panel_id])
            elif port.presentation_current or port.surface_busy:
                port.invalidate_presentation()
            active.append(panel_id)
        if not active:
            return
        self._scheduler.invalidate_presentations(tuple(active))
        self.wake.request_owner_wake()

    def commit(self, *, admit_new: bool = True) -> None:
        """Put ready boards on screen.  The GUI thread, and only it.

        A STEP of the owner's turn, not the turn itself -- the owner claims
        the wake before it starts working, so that anything arriving while
        the turn runs raises a fresh one.  Claiming it here instead put the
        claim in the middle: work enqueued during the drain that runs before
        this saw a wake still pending, sent no notification, and then had its
        pending flag cleared out from under it.  Measured on a live console,
        that lost wake was 96 ms of a 260 ms press -- a hand waiting on the
        data clock for its own answer.

        Pause and close pass ``admit_new=False``: renders already travelling
        may still finish, while a source publication that arrived after Pause
        cannot advance the frozen board.
        """

        self._scheduler.stage_owed(admit_new=admit_new)
        self._arbiter.drain(self._resolve)
        # Accepting a travelling group releases its ports.  Spend any
        # capacity-one surface debt against Plane latest in this same owner
        # turn, instead of waiting for another 100 ms beat.
        if admit_new:
            self._scheduler.stage_owed()

    def _resolve(self, panel_id: str) -> Any | None:
        for port in self._ports():
            if port.panel_id == panel_id:
                return port
        return None

    @property
    def pending_projection_count(self) -> int:
        with self._projection_lock:
            return len(self._projection_futures)

    def close(self) -> bool:
        """Start shutdown and report True only after every projection is done."""

        with self._projection_lock:
            if self._closed:
                return True
            first = not self._closing
            self._closing = True
            pending = tuple(self._projection_futures)
        if first:
            self._unsubscribe_publications()
            self._scheduler.close()
            self._arbiter.close()
        for future in pending:
            future.cancel()
        if first:
            self._projection_executor.shutdown(wait=False, cancel_futures=True)
        with self._projection_lock:
            if self._projection_futures:
                return False
        # Every tracked Future callback has returned to the executor loop; this
        # final join is therefore an idle-thread retirement, not a work wait.
        self._projection_executor.shutdown(wait=True, cancel_futures=True)
        with self._projection_lock:
            self._closed = True
        return True


def _guarded_slot(turn: Callable[[], None], what: str) -> Callable[[], None]:
    """Wrap a callable that is about to become a Qt slot.

    ONE OWNER for the boundary.  PyQt calls qFatal() on anything that leaves
    a slot, and the process dies where it stands -- taking a running
    experiment, every mounted panel and the traceback with it.  There are
    two hops into the GUI thread in this project, the timer and the
    completion wake, and the guard was written on one of them.

    Logged, not swallowed: the next tick still runs, and the instrument
    outlives the defect.  Callers that drive these directly (tests, headless
    benches) are untouched and still raise.
    """

    def guarded() -> None:
        try:
            turn()
        except Exception:  # noqa: BLE001 -- the boundary IS total
            _LOG.exception("Qt-driven %s failed", what)

    return guarded


def attach_qt(beat: Callable[[], None], *, interval_ms: int) -> Any:
    """Drive one beat from a Qt event loop.

    It takes the beat rather than the board, and that is the whole point.  This
    used to tick and commit the board itself, which is most of a beat and not
    all of it -- so the shipped window ran a different beat from the one every
    test exercises, and Pause, node polling and panel-error reporting were dead
    on screen while passing headlessly.

    A timer callback IS the GUI thread, and it is therefore also the boundary
    where an exception stops being reportable.  ``_guarded_slot`` owns that
    rule for both hops that cross it; see its docstring.
    """

    from PyQt5 import QtCore  # noqa: PLC0415 -- only a Qt application needs this

    if not callable(beat):
        raise TypeError("attach_qt drives a beat, not a board")
    interval = int(interval_ms)
    if interval <= 0:
        raise ValueError("attach_qt interval_ms must be positive")

    timer = QtCore.QTimer()
    timer.setInterval(interval)
    timer.timeout.connect(_guarded_slot(beat, "beat"))
    timer.start()
    return timer


def attach_qt_worker(name: str = "zlc-worker"):
    """Run Qt-free work off the GUI thread and deliver the result back on it.

    The other half of the owner turn: a presenter's slow half -- opening
    devices, scanning a vendor stack -- must not run in a click slot, because
    while it does the event loop never turns and the busy state and status
    lines the presenter writes are never painted.  The work goes to one
    worker; its outcome comes back through the same queued hop the board
    already uses, so everything that touches a window still happens on the
    thread that owns them.
    """

    from concurrent.futures import ThreadPoolExecutor
    from queue import Empty, SimpleQueue

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=str(name))
    inbox: SimpleQueue = SimpleQueue()
    state_lock = Lock()
    pending = 0
    closed = False

    def drain() -> None:
        while True:
            try:
                finish = inbox.get_nowait()
            except Empty:
                return
            finish()

    trigger = attach_qt_owner_turn(drain)

    def run(work, deliver, failed) -> None:
        nonlocal pending
        with state_lock:
            if closed:
                raise RuntimeError("Qt worker is closed")
            pending += 1

        def task() -> None:
            try:
                result = work()
            except BaseException as error:  # noqa: BLE001 -- delivered, not lost
                callback = lambda error=error: failed(error)
            else:
                callback = lambda result=result: deliver(result)

            def finish() -> None:
                nonlocal pending
                try:
                    callback()
                finally:
                    with state_lock:
                        pending -= 1

            inbox.put(finish)
            trigger()

        try:
            pool.submit(task)
        except BaseException:
            with state_lock:
                pending -= 1
            raise

    def close() -> bool:
        nonlocal closed
        with state_lock:
            if closed:
                return True
            if pending:
                return False
            closed = True
        pool.shutdown(wait=True, cancel_futures=True)
        return True

    return run, close


def attach_qt_owner_turn(turn: Callable[[], None]) -> Callable[[], None]:
    """Return a thread-safe trigger that runs ``turn`` on the Qt GUI thread.

    This is the completion-driven half of the beat: a finished render's done
    callback fires on a worker thread, and the surfaces it completed must be
    committed by the GUI thread NOW, not on the next timer beat -- a group's
    latency is its slowest member plus this one queued hop.  The returned
    trigger owns its relay QObject, so the caller only keeps the callable.
    """

    from PyQt5 import QtCore  # noqa: PLC0415 -- only a Qt application needs this

    if not callable(turn):
        raise TypeError("attach_qt_owner_turn relays a turn callable")

    class _OwnerTurnRelay(QtCore.QObject):
        woke = QtCore.pyqtSignal()

    relay = _OwnerTurnRelay()
    # Guarded like the timer, because this is the SAME hop.  Without it the
    # very same exception was a log line when it arrived on the beat and the
    # death of the process when it arrived on a completion wake -- a saved
    # selector that its exact publication cannot honour reaches this one
    # first, through _settle_panel_hosts on the tick after the first surface
    # is accepted.
    relay.woke.connect(
        _guarded_slot(turn, "owner turn"), type=QtCore.Qt.QueuedConnection
    )

    def trigger(relay: "_OwnerTurnRelay" = relay) -> None:
        relay.woke.emit()

    return trigger
