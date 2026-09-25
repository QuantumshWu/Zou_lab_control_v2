"""Worker mailbox and Run ownership for one headless owner."""

from __future__ import annotations

from concurrent.futures import Future
import threading
from typing import Callable, NamedTuple


class OwnerCompletion(NamedTuple):
    kind: str
    generation: int
    future: Future


class RunOwnerMailbox:
    """Own asynchronous Run mechanics, never application product policy.

    Every operation runs on a daemon thread of its own.  A run is never
    handed a second operation while one is still out (its host refuses a
    Start until the last one was reaped), so nothing needs a pool -- and a
    pool's worker is joined at interpreter exit: an operation that never
    returns, an operator's Derive stuck in a loop or an SDK stop that hangs,
    would hold the process open after the console had given up waiting.
    """

    def __init__(
        self,
        request_owner_wake: Callable[[], None],
        *,
        thread_name_prefix: str,
    ) -> None:
        self._wake = request_owner_wake
        self._thread_name = thread_name_prefix
        self._lock = threading.Lock()
        self._tracked: set[Future] = set()
        self._completions: list[OwnerCompletion] = []
        self._generation = 0
        self._owner_reaped = True

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def owner_reaped(self) -> bool:
        return self._owner_reaped

    @property
    def worker_idle(self) -> bool:
        with self._lock:
            return not self._tracked and not self._completions

    @property
    def has_pending_owner_work(self) -> bool:
        with self._lock:
            return bool(self._tracked or self._completions)

    def begin_generation(self) -> int:
        self._generation += 1
        self._owner_reaped = False
        return self._generation

    def mark_owner_reaped(self) -> None:
        self._owner_reaped = True

    def submit(
        self,
        kind: str,
        work: Callable[[], object],
        *,
        generation: int | None = None,
    ) -> Future:
        generation = self._generation if generation is None else generation
        future: Future = Future()
        with self._lock:
            self._tracked.add(future)

        def done(completed: Future) -> None:
            with self._lock:
                self._tracked.discard(completed)
                self._completions.append(
                    OwnerCompletion(kind, generation, completed)
                )
            self._wake()

        future.add_done_callback(done)

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = work()
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(result)

        try:
            threading.Thread(
                target=run, name=f"{self._thread_name}-{kind}", daemon=True,
            ).start()
        except BaseException:
            # No thread will ever finish this future: tracked, it would hold
            # the owner busy for good -- every later Start refused, the close
            # refused -- where the operation was simply never started.
            with self._lock:
                self._tracked.discard(future)
            raise
        return future

    def drain_completions(self) -> tuple[OwnerCompletion, ...]:
        with self._lock:
            pending = tuple(self._completions)
            self._completions.clear()
        return pending

    def shutdown(self) -> None:
        if self.has_pending_owner_work:
            raise RuntimeError("cannot close Run owner with pending work")
        if not self._owner_reaped:
            raise RuntimeError("cannot close Run owner before its handle is reaped")


__all__ = ["OwnerCompletion", "RunOwnerMailbox"]
