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

from collections.abc import Callable, Sequence
from typing import Any

from zlc_runtime import (
    BoardScheduler,
    HarmonicClock,
    OwnerChannels,
    SurfaceBatchArbiter,
)


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
        self._pending = False
        self._notify = notify

    def set_notify(self, notify: Callable[[], None] | None) -> None:
        """Bind the owner-turn trigger after composition has one to give.

        The board exists before the event-loop shim that can hop threads, so
        the trigger arrives late.  Without one, a wake only raises the pending
        flag and the next beat takes the turn -- which is exactly the headless
        behavior every test drives.
        """

        self._notify = notify

    def request_owner_wake(self) -> None:
        already = self._pending
        self._pending = True
        if not already and self._notify is not None:
            self._notify()

    def take(self) -> bool:
        """Claim a pending wake, if there is one."""

        pending, self._pending = self._pending, False
        return pending

    @property
    def pending(self) -> bool:
        return self._pending


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
        self.wake = OwnerWake(notify)
        self._channels = OwnerChannels(self.wake)
        self._arbiter = SurfaceBatchArbiter(self._channels)
        self._ports = ports
        self._clock = HarmonicClock(tuple(intervals))
        self._scheduler = BoardScheduler(
            plane,
            self._clock,
            self._arbiter,
            ports,
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

    def commit(self) -> None:
        """Put ready boards on screen.  The GUI thread, and only it.

        The wake is claimed first, so a surface finishing while the drain runs
        notifies again instead of being lost inside the turn it missed.
        """

        self.wake.take()
        self._arbiter.drain(self._resolve)

    def _resolve(self, panel_id: str) -> Any | None:
        for port in self._ports():
            if port.panel_id == panel_id:
                return port
        return None

    def close(self) -> None:
        self._scheduler.close()
        self._arbiter.close()


def attach_qt(beat: Callable[[], None], *, interval_ms: int) -> Any:
    """Drive one beat from a Qt event loop.

    It takes the beat rather than the board, and that is the whole point.  This
    used to tick and commit the board itself, which is most of a beat and not
    all of it -- so the shipped window ran a different beat from the one every
    test exercises, and Pause, node polling and panel-error reporting were dead
    on screen while passing headlessly.

    A timer callback IS the GUI thread, which is the one hop that has to be
    here and cannot be anywhere else.
    """

    from PyQt5 import QtCore  # noqa: PLC0415 -- only a Qt application needs this

    if not callable(beat):
        raise TypeError("attach_qt drives a beat, not a board")
    interval = int(interval_ms)
    if interval <= 0:
        raise ValueError("attach_qt interval_ms must be positive")
    timer = QtCore.QTimer()
    timer.setInterval(interval)
    timer.timeout.connect(beat)
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

    def drain() -> None:
        while True:
            try:
                finish = inbox.get_nowait()
            except Empty:
                return
            finish()

    trigger = attach_qt_owner_turn(drain)

    def run(work, deliver, failed) -> None:
        def task() -> None:
            try:
                result = work()
            except BaseException as error:  # noqa: BLE001 -- delivered, not lost
                inbox.put(lambda: failed(error))
            else:
                inbox.put(lambda: deliver(result))
            trigger()

        pool.submit(task)

    return run


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
    relay.woke.connect(turn, type=QtCore.Qt.QueuedConnection)

    def trigger(relay: "_OwnerTurnRelay" = relay) -> None:
        relay.woke.emit()

    return trigger
