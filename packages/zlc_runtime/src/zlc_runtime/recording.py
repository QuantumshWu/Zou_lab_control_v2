"""Every event a run publishes that belongs in the record, on its own thread.

A run's data used to exist only in memory: a day of shots was hundreds of
megabytes nobody could open afterwards, an interruption lost all of it, and
the only way to keep any of it was to save a picture of it.  Commit is the
one place every event of every node passes through, so it is where this
belongs.

WHAT IS RECORDED is the output's own declaration to make.  Nothing is
written unless a ``DatasetOutputDeclaration`` says ``recorded=True``, and
the default is off for one measured reason: a camera's finite output is its
raw frames, four and a half megabytes each, and a thousand-event chunk of
them is four gigabytes buffered before a single byte reaches the disk.  An
output belongs in the record when what it accumulates is the science -- the
occupancy, the counts, the survival, the scan -- and the node that produces
it is the only thing that knows.

ON ITS OWN THREAD, for a measured reason too.  One flush is about seventeen
milliseconds almost entirely of fsync -- the same whether it carries fifty
events or a thousand, so no chunk size escapes it -- and the threads that
reach a terminal are the ones polling the hosts, which on the console is
Qt's.  Writing there is a visible hitch every time a run ends.  So ``record``
only hands work over, the queue is bounded, and a disk that cannot keep up
slows the producer instead of growing without limit.

RECORDING NEVER FAILS A RUN.  A full disk, a revoked permission, a schema
the store cannot stack -- none is a reason to stop an experiment that is
working.  The first failure stops the recording and is kept until somebody
is told (:meth:`take_failure`); the run goes on.  Losing the recording is
bad and losing the experiment is worse, and that order is a decision, not an
accident, so it is written here rather than left to a bare ``except``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

from .plane import SignalValue
from .publication_store import DEFAULT_EVENTS_PER_CHUNK, PublicationWriter


#: Where a run's stores live inside its own directory.
RECORDING_DIRECTORY = "published"

#: How many events may wait to be written before a producer is made to wait.
#: A run that outpaces its disk for a moment must not be stalled by it; one
#: that outpaces it for good must not be allowed to eat the machine either.
QUEUE_DEPTH = 4096

_STOP = object()
_FLUSH = object()


__all__ = ["QUEUE_DEPTH", "RECORDING_DIRECTORY", "RunRecorder"]


class RunRecorder:
    """One append-only store per recorded output, under one run directory.

    Per OUTPUT, not per run: two outputs of one node have different schemas
    and different shapes, and a chunk is a stack of events that agree.  Their
    names are the declaration's own -- ``counts``, ``occupied`` -- so what is
    on disk is readable without knowing how signals are keyed internally.
    """

    def __init__(
        self,
        open_directory: Callable[[], Path],
        *,
        events_per_chunk: int = DEFAULT_EVENTS_PER_CHUNK,
        queue_depth: int = QUEUE_DEPTH,
    ) -> None:
        if not callable(open_directory):
            raise TypeError("a recorder is given a way to open its directory")
        # Called ONCE, on the writing thread at the first event, because
        # allocating a numbered run directory is itself a durable act: a node
        # that is configured and never publishes must not leave a folder
        # behind saying it did.
        self._open_directory = open_directory
        self._events_per_chunk = int(events_per_chunk)
        self._root: Path | None = None
        self._writers: dict[str, PublicationWriter] = {}
        self._queue: Queue = Queue(maxsize=int(queue_depth))
        self._events = 0
        self._failure: BaseException | None = None
        #: The same failure, until somebody has been told about it.
        self._unreported: BaseException | None = None
        self._closed = False
        self._thread: Thread | None = None

    # ------------------------------------------------------------- reading
    @property
    def root(self) -> Path | None:
        """Where this run is being written, once anything has been."""

        return self._root

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(sorted(self._writers))

    @property
    def events(self) -> int:
        """Events handed over, including any still waiting to be written."""

        return self._events

    @property
    def durable_events(self) -> int:
        """Events an interruption right now would leave on disk."""

        return sum(writer.durable_events for writer in self._writers.values())

    @property
    def nbytes(self) -> int:
        return sum(writer.nbytes for writer in self._writers.values())

    @property
    def failure(self) -> BaseException | None:
        """Why recording stopped, if it did.  The run did not stop with it."""

        return self._failure

    def take_failure(self) -> BaseException | None:
        """The failure nobody has been told about yet -- once.

        Kept here rather than in whoever reports it: "the run went on and the
        recording did not" is a fact about this recorder, and a recording
        that stops itself and tells nobody is indistinguishable from a
        complete one, which is the whole reason the failure is swallowed at
        all.  ``failure`` stays readable for ever; this hands it over once so
        one report reaches the operator and a beat does not repeat it.
        """

        failure, self._unreported = self._unreported, None
        return failure

    # ------------------------------------------------------------- writing
    def record(self, published: Mapping[str, Any]) -> None:
        """Hand over one commit's recorded outputs, in publication order.

        Called on whichever thread committed -- a node's worker, a processor
        lane, or the owner's poll -- so it does no I/O of its own.
        """

        if self._closed or self._failure is not None:
            return
        for name, value in published.items():
            if not isinstance(value, SignalValue):
                continue
            self._queue.put((str(name), value.snapshot))
            self._events += 1
        self._ensure_thread()

    def flush(self) -> None:
        """Ask for the buffered tail to reach disk; do not wait for it.

        What a terminal generation owes is this, not a close: a NodeHost has
        as many generations as its source has -- a processor whose source
        ends is refused into CANCELLED and a standing re-follow starts it
        again -- and a recorder closed at the first of them would record
        nothing afterwards, silently.

        Asked, not awaited, because the callers of a terminal are the threads
        polling the hosts, and on the console that is Qt's.
        """

        if self._closed or self._failure is not None or self._thread is None:
            return
        self._queue.put(_FLUSH)

    def close(self, timeout: float = 30.0) -> None:
        """Finish: everything handed over reaches disk, then the thread ends.

        This one WAITS.  It runs when the host is retired, which is allowed
        to take as long as finishing honestly takes.
        """

        if self._closed:
            return
        self._closed = True
        thread = self._thread
        if thread is None:
            return
        self._queue.put(_STOP)
        thread.join(timeout=float(timeout))

    # -------------------------------------------------------- the writer
    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = Thread(
                target=self._write_queued, name="zlc-run-recorder", daemon=True
            )
            self._thread.start()

    def _write_queued(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._finish_writers()
                return
            if self._failure is not None:
                # Drained, not written: the queue must never fill behind a
                # recording that has already stopped, because filling it
                # would block the run this refuses to fail.
                continue
            try:
                if item is _FLUSH:
                    for writer in self._writers.values():
                        writer.flush()
                    continue
                name, snapshot = item
                self._writer(name).append(snapshot)
            except BaseException as error:  # noqa: BLE001 -- kept, never raised
                self._stop_recording(error)

    def _writer(self, name: str) -> PublicationWriter:
        found = self._writers.get(name)
        if found is None:
            if self._root is None:
                self._root = Path(self._open_directory()) / RECORDING_DIRECTORY
            found = PublicationWriter(
                self._root / name,
                events_per_chunk=self._events_per_chunk,
                note=name,
            )
            self._writers[name] = found
        return found

    def _stop_recording(self, error: BaseException) -> None:
        self._failure = error
        self._unreported = error
        self._finish_writers()

    def _finish_writers(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as error:  # noqa: BLE001 -- kept, never raised
                if self._failure is None:
                    self._failure = error
                    self._unreported = error
