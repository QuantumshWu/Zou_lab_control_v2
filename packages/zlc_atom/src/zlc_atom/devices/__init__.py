"""Headless hardware and virtual-device contracts."""

from collections import deque
from typing import Callable
import threading
import time


class RecordQueue:
    """Bounded FIFO of already-owned records, shared by device adapters.

    Adapters construct their records and own SDK start/read/stop. This queue
    checks capture ordinals, preserves every accepted record, and reports
    overflow/failure before handing out data. An optional producer worker is
    stopped and joined here; a device-owned SDK lane is never replaced.
    """

    def __init__(self, what: str, *, join_timeout_seconds: float) -> None:
        self._what = str(what)
        self._join_timeout = float(join_timeout_seconds)
        self._condition = threading.Condition()
        self._queue: deque[object] = deque()
        self._armed = False
        self._accepting = False
        self._expected: int | None = None
        self._buffer_record_count = 1
        self._produced_count = 0
        self._failure: BaseException | None = None
        self._worker: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._ready = threading.Event()

    @property
    def armed(self) -> bool:
        with self._condition:
            return self._armed

    @property
    def accepting(self) -> bool:
        with self._condition:
            return self._accepting

    @property
    def produced_count(self) -> int:
        with self._condition:
            return self._produced_count

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def capacity(self) -> int:
        with self._condition:
            return self._buffer_record_count

    @property
    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    def arm(
        self,
        records: int | None,
        *,
        buffer_record_count: int,
        worker: Callable[[threading.Event], threading.Thread] | None = None,
    ) -> None:
        """Start a capture; ``worker`` builds the producer thread from its stop event."""

        buffer_count = int(buffer_record_count)
        if buffer_count <= 0:
            raise ValueError("buffer_record_count must be positive")
        expected = None if records is None else int(records)
        if expected is not None and expected <= 0:
            raise ValueError("finite records must be positive")
        with self._condition:
            if self._armed:
                raise RuntimeError(f"{self._what} is already armed")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError(f"{self._what} is still finishing its previous capture")
            self._queue.clear()
            self._failure = None
            self._armed = True
            self._accepting = True
            self._expected = expected
            self._buffer_record_count = buffer_count
            self._produced_count = 0
            self._ready.clear()
            if worker is None:
                return
            stop = threading.Event()
            thread = worker(stop)
            self._worker = thread
            self._stop = stop
            try:
                thread.start()
            except BaseException:
                self._worker = None
                self._stop = None
                self._armed = False
                self._accepting = False
                raise

    def mark_ready(self) -> None:
        """The producer has started the hardware needed for this capture."""

        self._ready.set()

    def wait_ready(self, timeout: float) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(f"{self._what} did not become ready")
        with self._condition:
            if self._failure is not None:
                raise RuntimeError(f"{self._what} failed while arming: {self._failure}") from self._failure
            if not self._armed:
                raise RuntimeError(f"{self._what} stopped before becoming ready")

    def push(self, record: object) -> bool:
        """Keep one complete record; failures stop acceptance without dropping."""

        with self._condition:
            if not self._accepting:
                return False
            ordinal = getattr(record, "source_ordinal", None)
            if type(ordinal) is not int or ordinal != self._produced_count:
                self.fail(RuntimeError(
                    f"{self._what} record ordinal {ordinal!r} is not "
                    f"the expected {self._produced_count} "
                    f"(observed_at_ns={time.time_ns()})"
                ))
                return False
            if len(self._queue) >= self._buffer_record_count:
                self.fail(RuntimeError(
                    f"{self._what} capture buffer overflow at record {self._produced_count} "
                    f"(capacity {self._buffer_record_count}, observed_at_ns={time.time_ns()}); "
                    "no queued record was discarded"
                ))
                return False
            self._produced_count += 1
            self._queue.append(record)
            if self._expected is not None and self._produced_count >= self._expected:
                self._accepting = False
            self._condition.notify_all()
            return True

    def fail(self, error: BaseException) -> None:
        """The producer has died; readers learn it, the capture stops accepting."""

        with self._condition:
            if self._failure is None:
                self._failure = error
            self._accepting = False
            self._ready.set()
            self._condition.notify_all()

    def read(self, n: int, *, timeout: float, exact: bool) -> list[object]:
        requested = int(n)
        if requested <= 0:
            raise ValueError("n must be positive")
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                if self._failure is not None:
                    raise RuntimeError(
                        f"{self._what} failed while producing records: {self._failure}"
                    ) from self._failure
                if len(self._queue) >= requested:
                    break
                if not self._armed or not self._accepting:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if exact and len(self._queue) < requested:
                raise TimeoutError(f"{self._what} did not deliver the requested exact records")
            count = requested if exact else min(requested, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def finish(self) -> int:
        """End queue acceptance and join its worker, retaining queued records."""

        with self._condition:
            if not self._armed:
                if self._failure is not None:
                    raise RuntimeError(
                        f"{self._what} failed while producing records: {self._failure}"
                    ) from self._failure
                return self._produced_count
            self._accepting = False
            self._ready.set()
            worker = self._worker
            stop = self._stop
            if stop is not None:
                stop.set()
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout=self._join_timeout)
        with self._condition:
            if worker is not None and worker.is_alive():
                raise RuntimeError(
                    f"{self._what} did not stop producing within {self._join_timeout:g} s"
                )
            self._worker = None
            self._stop = None
            self._armed = False
            self._condition.notify_all()
            if self._failure is not None:
                raise RuntimeError(
                    f"{self._what} failed while producing records: {self._failure}"
                ) from self._failure
            return self._produced_count



__all__ = ["RecordQueue"]
