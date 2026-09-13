"""Every event a run publishes, on disk as it is published.

A run's data used to exist only in memory: a day of shots was hundreds of
megabytes nobody could open afterwards, an interruption lost all of it, and
the only way to keep any of it was to save a picture of it.  Commit is the
one place every event of every node passes through, so it is the one place
this has to be done.

RECORDING NEVER FAILS A RUN.  A full disk, a revoked permission, a schema
the store cannot stack -- none of them is a reason to stop an experiment
that is working.  The first failure stops the recording, is kept, and is
reported by whoever asks; the run goes on.  Losing the recording is bad and
losing the experiment is worse, and that order is a decision, not an
accident, so it is written here rather than left to a bare ``except``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from .dataset import MonitorCoverage
from .publication_store import (
    DEFAULT_EVENTS_PER_CHUNK,
    PublicationWriter,
)
from .plane import SignalValue


#: Where a run's stores live inside its own directory.
RECORDING_DIRECTORY = "published"


__all__ = ["RECORDING_DIRECTORY", "RunRecorder"]


class RunRecorder:
    """One append-only store per published output, under one run directory.

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
    ) -> None:
        if not callable(open_directory):
            raise TypeError("a recorder is given a way to open its directory")
        # Called ONCE, at the first event, because allocating a numbered run
        # directory is itself a durable act: a node that is configured and
        # never publishes must not leave a folder behind saying it did.
        self._open_directory = open_directory
        self._events_per_chunk = int(events_per_chunk)
        self._root: Path | None = None
        self._writers: dict[str, PublicationWriter] = {}
        self._events = 0
        self._failure: BaseException | None = None
        self._closed = False

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
        """Events accepted across every output."""

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

    # ------------------------------------------------------------- writing
    def record(self, published: Mapping[str, SignalValue]) -> None:
        """Append one commit's accumulating outputs.

        A MONITOR is not recorded, and the reason is its own contract: it
        retains only its latest event, so there is no history of it to write
        down.  That contract is also what makes the rule affordable -- the
        monitor in a real run is the camera, whose frames are eight
        megabytes each; recorded, a thousand-event chunk is eight gigabytes
        and the console stalls for seconds buffering it.  What a run
        accumulates is what is written, which is what the plane accumulates.

        Called on the committing thread, immediately after the plane has the
        event, so what is on disk is in the order it was published.
        """

        if self._closed or self._failure is not None:
            return
        try:
            for name, value in published.items():
                if not isinstance(value, SignalValue):
                    continue
                if isinstance(value.coverage, MonitorCoverage):
                    continue
                self._writer(str(name)).append(value.snapshot)
                self._events += 1
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
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException:  # noqa: BLE001 -- already failing
                pass

    def close(self) -> None:
        """Flush what is buffered, so a finished run leaves no tail behind."""

        if self._closed:
            return
        self._closed = True
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as error:  # noqa: BLE001 -- kept, never raised
                if self._failure is None:
                    self._failure = error
