"""Ordered future-publication delivery used inside the signal plane."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Generic, Iterable, TypeVar

from zlc_data import StreamGenerationId
from zlc_data import canonical_text, nonnegative_integer


PayloadT = TypeVar("PayloadT")
_FOLLOW_TOKEN = object()


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


class FollowTap(Generic[PayloadT]):
    """Lossless ordered delivery of source payloads without another identity."""

    def __init__(
        self,
        authority: object,
        *,
        stream: "AcquisitionStream[PayloadT]",
        start_sequence: int,
        replay: tuple[tuple[int, PayloadT], ...] = (),
    ) -> None:
        if authority is not _FOLLOW_TOKEN:
            raise PermissionError("FollowTap can only be minted by AcquisitionStream")
        self._stream = stream
        self._condition = threading.Condition()
        self._queue: deque[tuple[int, PayloadT]] = deque(replay)
        self._next_sequence = start_sequence
        self._closed = False
        self._source_finished = False
        self._terminal_error: StreamError | None = None

    def _offer(self, sequence: int, payload: PayloadT) -> None:
        with self._condition:
            if self._closed or self._source_finished:
                return
            expected = self._next_sequence + len(self._queue)
            if sequence != expected:
                raise StreamGap(expected, sequence)
            self._queue.append((sequence, payload))
            self._condition.notify()

    def next(self, timeout: float | None = None) -> PayloadT:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
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
            sequence, payload = self._queue.popleft()
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
    ) -> FollowTap[PayloadT]:
        retained = tuple(replay)
        with self._lock:
            if self._closed:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise StreamEndedEarly("cannot follow a closed stream")
            start = self._next_sequence
            if retained:
                start = retained[0][0]
                expected = start
                for sequence, _payload in retained:
                    if sequence != expected:
                        raise StreamGap(expected, sequence)
                    expected += 1
                if expected != self._next_sequence:
                    raise StreamGap(self._next_sequence, expected)
            tap = FollowTap(
                _FOLLOW_TOKEN,
                stream=self,
                start_sequence=start,
                replay=retained,
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
                follower._offer(sequence, payload)
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
