"""Ordered future-publication delivery used inside the signal plane."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Generic, Protocol, TypeVar
import uuid

from zlc_data import StreamGenerationId
from zlc_data import canonical_text, finite_real, nonnegative_integer


PayloadT = TypeVar("PayloadT")
_FOLLOW_TOKEN = object()


class PayloadContract(Protocol[PayloadT]):
    def snapshot(self, payload: PayloadT) -> PayloadT: ...

    def validate(self, payload: PayloadT) -> None: ...


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


@dataclass(frozen=True, slots=True)
class Envelope(Generic[PayloadT]):
    event_ref: EventRef
    payload: PayloadT
    emitted_at: float
    captured_at: float
    direct_parent_refs: tuple[EventRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.event_ref, EventRef):
            raise TypeError("envelope event_ref must be EventRef")
        object.__setattr__(
            self,
            "emitted_at",
            finite_real(self.emitted_at, "emitted_at"),
        )
        object.__setattr__(
            self,
            "captured_at",
            finite_real(self.captured_at, "captured_at"),
        )
        parents = tuple(self.direct_parent_refs)
        if any(not isinstance(parent, EventRef) for parent in parents):
            raise TypeError("direct_parent_refs must contain EventRef values")
        if len(set(parents)) != len(parents):
            raise ValueError("direct_parent_refs cannot contain duplicates")
        object.__setattr__(self, "direct_parent_refs", parents)

    @property
    def sequence(self) -> int:
        return self.event_ref.sequence


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
    """Lossless ordered delivery of events published after subscription."""

    def __init__(
        self,
        authority: object,
        *,
        stream: "AcquisitionStream[PayloadT]",
        start_sequence: int,
    ) -> None:
        if authority is not _FOLLOW_TOKEN:
            raise PermissionError("FollowTap can only be minted by AcquisitionStream")
        self._stream = stream
        self._condition = threading.Condition()
        self._queue: deque[Envelope[PayloadT]] = deque()
        self._next_sequence = start_sequence
        self._closed = False
        self._source_finished = False
        self._terminal_error: StreamError | None = None

    def _offer(self, envelope: Envelope[PayloadT]) -> None:
        with self._condition:
            if self._closed or self._source_finished:
                return
            expected = self._next_sequence + len(self._queue)
            if envelope.sequence != expected:
                raise StreamGap(expected, envelope.sequence)
            self._queue.append(envelope)
            self._condition.notify()

    def next(self, timeout: float | None = None) -> Envelope[PayloadT]:
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
            envelope = self._queue.popleft()
            if envelope.sequence != self._next_sequence:
                raise StreamGap(self._next_sequence, envelope.sequence)
            self._next_sequence += 1
            return envelope

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


class AcquisitionProducer(Generic[PayloadT]):
    """Exclusive write authority for the plane-local future stream."""

    def __init__(self, stream: "AcquisitionStream[PayloadT]", authority: object) -> None:
        self._stream = stream
        self._authority = authority

    def emit(
        self,
        payload: PayloadT,
        *,
        captured_at: float,
        direct_parent_refs: tuple[EventRef, ...] = (),
    ) -> Envelope[PayloadT]:
        return self._stream._emit(
            self._authority,
            payload,
            captured_at=captured_at,
            direct_parent_refs=direct_parent_refs,
        )

    def finish(self) -> None:
        self._stream._finish(self._authority)

    def fail(self, error: StreamError) -> None:
        self._stream._fail(self._authority, error)


class AcquisitionStream(Generic[PayloadT]):
    """One generation feeding only ordered future subscribers."""

    def __init__(
        self,
        stream_id: StreamId,
        generation: StreamGenerationId,
        payload_contract: PayloadContract[PayloadT],
    ) -> None:
        if not isinstance(stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        if not callable(getattr(payload_contract, "snapshot", None)):
            raise TypeError("payload_contract.snapshot must be callable")
        if not callable(getattr(payload_contract, "validate", None)):
            raise TypeError("payload_contract.validate must be callable")
        self.stream_id = stream_id
        self.generation = generation
        self._payload_contract = payload_contract
        self._condition = threading.Condition()
        self._next_sequence = 0
        self._followers: set[FollowTap[PayloadT]] = set()
        self._closed = False
        self._terminal_error: StreamError | None = None
        self._producer_authority = object()

    @classmethod
    def create(
        cls,
        stream_id: StreamId,
        payload_contract: PayloadContract[PayloadT],
    ) -> tuple["AcquisitionStream[PayloadT]", AcquisitionProducer[PayloadT]]:
        stream = cls(
            stream_id,
            StreamGenerationId(uuid.uuid4().hex),
            payload_contract,
        )
        return stream, AcquisitionProducer(stream, stream._producer_authority)

    def follow(self) -> FollowTap[PayloadT]:
        with self._condition:
            if self._closed:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise StreamEndedEarly("cannot follow a closed stream")
            tap = FollowTap(
                _FOLLOW_TOKEN,
                stream=self,
                start_sequence=self._next_sequence,
            )
            self._followers.add(tap)
            return tap

    def _emit(
        self,
        authority: object,
        payload: PayloadT,
        *,
        captured_at: float,
        direct_parent_refs: tuple[EventRef, ...],
    ) -> Envelope[PayloadT]:
        if authority is not self._producer_authority:
            raise PermissionError("stream write authority belongs to another producer")
        payload = self._payload_contract.snapshot(payload)
        self._payload_contract.validate(payload)
        with self._condition:
            if self._closed:
                if self._terminal_error is not None:
                    raise self._terminal_error
                raise StreamEndedEarly("cannot emit after end-of-stream")
            envelope = Envelope(
                EventRef(self.stream_id, self.generation, self._next_sequence),
                payload,
                time.time(),
                captured_at,
                direct_parent_refs,
            )
            self._next_sequence += 1
            for follower in tuple(self._followers):
                follower._offer(envelope)
            return envelope

    def _finish(self, authority: object) -> None:
        self._close(authority, None)

    def _fail(self, authority: object, error: StreamError) -> None:
        if not isinstance(error, StreamError):
            raise TypeError("source failure must be a StreamError")
        self._close(authority, error)

    def _close(self, authority: object, error: StreamError | None) -> None:
        if authority is not self._producer_authority:
            raise PermissionError("stream terminal authority belongs to another producer")
        with self._condition:
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
            self._condition.notify_all()

    def _remove_follower(self, follower: FollowTap[PayloadT]) -> None:
        with self._condition:
            self._followers.discard(follower)


__all__ = [
    "AcquisitionProducer",
    "AcquisitionStream",
    "Envelope",
    "EventRef",
    "FollowTap",
    "PayloadContract",
    "SourceFailed",
    "StreamEndedEarly",
    "StreamError",
    "StreamGap",
    "StreamId",
]
