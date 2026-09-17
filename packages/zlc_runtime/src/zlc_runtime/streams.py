"""Ordered future-publication delivery used inside the signal plane."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable, Generic, Iterable, TypeVar

from zlc_data import StreamGenerationId
from zlc_data import canonical_text, nonnegative_integer


PayloadT = TypeVar("PayloadT")
_FOLLOW_TOKEN = object()
DEFAULT_FOLLOW_MAX_PENDING = 1024
DEFAULT_FOLLOW_MAX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class StreamId:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", canonical_text(self.value, "stream id"))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class EventRef:
    stream_id: StreamId
    generation: StreamGenerationId
    sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("event stream_id must be StreamId")
        if not isinstance(self.generation, StreamGenerationId):
            raise TypeError("event generation must be StreamGenerationId")
        object.__setattr__(
            self,
            "sequence",
            nonnegative_integer(self.sequence, "event sequence"),
        )


class StreamError(RuntimeError):
    pass


class StreamGap(StreamError):
    def __init__(self, expected: int, received: int) -> None:
        self.expected = expected
        self.received = received
        super().__init__(
            f"expected stream sequence {expected}, received {received}"
        )


class StreamEndedEarly(StreamError):
    pass


class SourceFailed(StreamError):
    pass


class SourceGenerationEnded(RuntimeError):
    """The followed source's generation ended or was replaced under a follower.

    Not a defect of the follower: the bench stopped or restarted the
    producer beneath a standing derivation.  Typed so a host can end
    CANCELLED on it -- the state an automatic re-follow restarts from --
    while a true failure stays failed for the operator to read.  A
    RuntimeError subclass, so every existing broad handler still catches
    it.
    """



class FollowTap(Generic[PayloadT]):
    """Lossless ordered delivery of source payloads without another identity."""

    def __init__(
        self,
        authority: object,
        *,
        stream: "AcquisitionStream[PayloadT]",
        start_sequence: int,
        live_sequence: int,
        replay: Iterable[tuple[int, PayloadT]] = (),
        max_pending: int = DEFAULT_FOLLOW_MAX_PENDING,
        max_bytes: int = DEFAULT_FOLLOW_MAX_BYTES,
        project: Callable[[PayloadT], PayloadT] | None = None,
        payload_size: Callable[[PayloadT], int] | None = None,
    ) -> None:
        if authority is not _FOLLOW_TOKEN:
            raise PermissionError("FollowTap can only be minted by AcquisitionStream")
        self._stream = stream
        self._condition = threading.Condition()
        if type(max_pending) is not int or max_pending < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("exact follow limits must be positive integers")
        self._queue: deque[tuple[int, PayloadT, int]] = deque()
        self._replay = iter(replay)
        self._max_pending, self._max_bytes = max_pending, max_bytes
        self._queued_bytes = 0
        self._project, self._payload_size = project, payload_size
        self._next_sequence = start_sequence
        self._live_sequence = live_sequence
        self._closed = False
        self._source_finished = False
        self._terminal_error: StreamError | None = None

    def _offer(self, sequence: int, payload: PayloadT) -> bool:
        with self._condition:
            if self._closed or self._source_finished:
                return False
            expected = self._live_sequence
            if sequence != expected:
                raise StreamGap(expected, sequence)
            self._live_sequence += 1
            failure = None
            try:
                payload = payload if self._project is None else self._project(payload)
                size = 0 if self._payload_size is None else self._payload_size(payload)
                if len(self._queue) >= self._max_pending or self._queued_bytes + size > self._max_bytes:
                    failure = SourceFailed(
                        f"exact follower overflow at sequence {sequence}: "
                        f"{len(self._queue) + 1}/{self._max_pending} pending events, "
                        f"{self._queued_bytes + size}/{self._max_bytes} payload bytes"
                    )
            except Exception as error:
                failure = SourceFailed(str(error))
            if failure is not None:
                self._terminal_error = failure
                self._source_finished = True
                self._queue.clear()
                self._queued_bytes = 0
                self._replay = None
                self._condition.notify_all()
                return False
            self._queue.append((sequence, payload, size))
            self._queued_bytes += size
            self._condition.notify()
            return True

    def next(self, timeout: float | None = None) -> PayloadT:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        # Replay reads the owner's immutable chunks lazily. Never hold the
        # tap condition while that reader acquires its owner's lock.
        while True:
            with self._condition:
                replay = self._replay
            if replay is None:
                break
            failure = None
            try:
                sequence, payload = next(replay)
            except StopIteration:
                with self._condition:
                    self._replay = None
                break
            except Exception as error:
                failure = str(error)
                self.close()
                replay = None
            if failure is not None:
                # Do not keep the reader traceback: it may own the rejected
                # oversized event. The existing close releases all backlog.
                raise SourceFailed(failure)
            with self._condition:
                if self._closed or self._replay is None:
                    break
                if sequence != self._next_sequence:
                    raise StreamGap(self._next_sequence, sequence)
                self._next_sequence += 1
                return payload
        with self._condition:
            while not self._queue:
                if self._closed:
                    raise StreamEndedEarly("follow tap is closed")
                if self._source_finished:
                    if self._terminal_error is not None:
                        raise self._terminal_error
                    raise StreamEndedEarly("follow source reached end-of-stream")
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for followed event")
                self._condition.wait(remaining)
            sequence, payload, size = self._queue.popleft()
            self._queued_bytes -= size
            if sequence != self._next_sequence:
                raise StreamGap(self._next_sequence, sequence)
            self._next_sequence += 1
            return payload

    def _source_ended(self, error: StreamError | None) -> None:
        with self._condition:
            self._source_finished = True
            self._terminal_error = error
            self._condition.notify_all()

    def close(self) -> None:
        self._stream._remove_follower(self)
        with self._condition:
            self._closed = True
            self._queue.clear()
            self._queued_bytes = 0
            self._replay = None
            self._condition.notify_all()


class AcquisitionStream(Generic[PayloadT]):
    """One ordered direct-payload source used only by Plane followers."""

    def __init__(
        self,
        next_sequence: int,
    ) -> None:
        self._lock = threading.Lock()
        self._next_sequence = nonnegative_integer(
            next_sequence,
            "next source sequence",
        )
        self._followers: set[FollowTap[PayloadT]] = set()
        self._closed = False
        self._terminal_error: StreamError | None = None

    @classmethod
    def create(
        cls,
        *,
        next_sequence: int,
    ) -> "AcquisitionStream[PayloadT]":
        return cls(next_sequence)

    def follow(
        self,
        replay: Iterable[tuple[int, PayloadT]] = (),
        *,
        replay_start_sequence: int | None = None,
        max_pending: int = DEFAULT_FOLLOW_MAX_PENDING,
        max_bytes: int = DEFAULT_FOLLOW_MAX_BYTES,
        project: Callable[[PayloadT], PayloadT] | None = None,
        payload_size: Callable[[PayloadT], int] | None = None,
    ) -> FollowTap[PayloadT]:
        with self._lock:
            if self._closed:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise StreamEndedEarly("cannot follow a closed stream")
            start = self._next_sequence if replay_start_sequence is None else replay_start_sequence
            tap = FollowTap(
                _FOLLOW_TOKEN,
                stream=self,
                start_sequence=start,
                live_sequence=self._next_sequence,
                replay=replay,
                max_pending=max_pending,
                max_bytes=max_bytes,
                project=project,
                payload_size=payload_size,
            )
            self._followers.add(tap)
            return tap

    def emit(
        self,
        payload: PayloadT,
        *,
        sequence: int,
    ) -> PayloadT:
        sequence = nonnegative_integer(sequence, "source sequence")
        with self._lock:
            if self._closed:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise StreamEndedEarly("cannot emit after end-of-stream")
            if sequence != self._next_sequence:
                raise StreamGap(self._next_sequence, sequence)
            self._next_sequence += 1
            for follower in tuple(self._followers):
                if not follower._offer(sequence, payload):
                    self._followers.discard(follower)
            return payload

    def finish(self) -> None:
        self._close(None)

    def fail(self, error: StreamError) -> None:
        if not isinstance(error, StreamError):
            raise TypeError("source failure must be a StreamError")
        self._close(error)

    def _close(self, error: StreamError | None) -> None:
        with self._lock:
            if self._closed:
                if self._terminal_error is error and error is not None:
                    return
                if self._terminal_error is not None:
                    raise self._terminal_error
                if error is None:
                    return
                raise StreamEndedEarly("completed stream cannot fail")
            self._closed = True
            self._terminal_error = error
            followers = tuple(self._followers)
            self._followers.clear()
            for follower in followers:
                follower._source_ended(error)

    def _remove_follower(self, follower: FollowTap[PayloadT]) -> None:
        with self._lock:
            self._followers.discard(follower)


__all__ = [
    "EventRef",
    "FollowTap",
    "SourceFailed",
    "StreamEndedEarly",
    "StreamError",
    "StreamGap",
    "StreamId",
]
