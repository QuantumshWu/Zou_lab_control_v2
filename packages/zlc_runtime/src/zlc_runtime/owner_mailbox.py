"""Worker mailbox and Run ownership for one headless owner."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
from typing import Callable, NamedTuple


class OwnerCompletion(NamedTuple):
    kind: str
    generation: int
    future: Future


class RunOwnerMailbox:
    """Own asynchronous Run mechanics, never application product policy."""

    def __init__(
        self,
        request_owner_wake: Callable[[], None],
        *,
        thread_name_prefix: str,
        max_workers: int = 2,
    ) -> None:
        self._wake = request_owner_wake
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
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
        future = self._pool.submit(work)
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
        self._pool.shutdown(wait=False)


__all__ = ["OwnerCompletion", "RunOwnerMailbox"]
