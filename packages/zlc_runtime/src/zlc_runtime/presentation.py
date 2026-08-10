"""Owner wakes, harmonic presentation cadence, and surface batching."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from numbers import Integral
from threading import Lock
from typing import NamedTuple, Protocol, runtime_checkable

from .plane import SignalDataPlane, SignalFront, SignalPublication, SignalValue


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field} must be positive")
    return normalized


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field} must be non-negative")
    return normalized


@runtime_checkable
class WakeSink(Protocol):
    """Callback target owned by the application event-loop adapter."""

    def request_owner_wake(self) -> None: ...


class OwnerTurn(NamedTuple):
    """One coalesced owner turn taken from the three wake channels."""

    lifecycle: bool
    data: bool
    surface: bool


class OwnerChannels:
    """Coalesce lifecycle, data, and surface notifications for one owner."""

    __slots__ = (
        "_closed",
        "_data_binding",
        "_data_plane",
        "_data_token",
        "_lifecycle_pending",
        "_lock",
        "_sink",
        "_surface_pending",
        "_data_pending",
    )

    def __init__(self, sink: WakeSink) -> None:
        if not callable(getattr(sink, "request_owner_wake", None)):
            raise TypeError("wake sink must provide request_owner_wake()")
        self._sink = sink
        self._lock = Lock()
        self._lifecycle_pending = False
        self._data_pending = False
        self._surface_pending = False
        self._closed = False
        self._data_plane: object | None = None
        self._data_token: object | None = None
        self._data_binding = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _notify(self, channel: str) -> None:
        with self._lock:
            if self._closed:
                return
            setattr(self, f"_{channel}_pending", True)
        self._sink.request_owner_wake()

    def notify_lifecycle(self) -> None:
        self._notify("lifecycle")

    def notify_surface(self) -> None:
        self._notify("surface")

    def notify_data(self) -> None:
        """Receive the callback borrowed by an attached signal data plane."""

        with self._lock:
            if self._closed or self._data_token is None:
                return
            self._data_pending = True
        self._sink.request_owner_wake()

    def activate_data(self, plane: SignalDataPlane) -> None:
        """Borrow the plane's owner-wake callback until deactivation or close."""

        bind = getattr(plane, "bind_owner_wake", None)
        unbind = getattr(plane, "unbind_owner_wake", None)
        if not callable(bind) or not callable(unbind):
            raise TypeError("data plane must provide bind_owner_wake/unbind_owner_wake")
        with self._lock:
            if self._closed:
                raise RuntimeError("owner channels are closed")
            if self._data_token is not None or self._data_binding:
                raise RuntimeError("data owner wake is already active")
            self._data_binding = True
        try:
            token = bind(self.notify_data)
        except BaseException:
            with self._lock:
                self._data_binding = False
            raise
        should_unbind = False
        with self._lock:
            self._data_binding = False
            if self._closed:
                should_unbind = True
            else:
                self._data_plane = plane
                self._data_token = token
        if should_unbind:
            unbind(token)
            raise RuntimeError("owner channels closed during data activation")

    def deactivate_data(self) -> None:
        with self._lock:
            if self._data_binding:
                raise RuntimeError("data owner wake activation is still in progress")
            plane = self._data_plane
            token = self._data_token
            self._data_plane = None
            self._data_token = None
            self._data_pending = False
        if plane is not None and token is not None:
            plane.unbind_owner_wake(token)

    def take(self) -> OwnerTurn:
        with self._lock:
            if self._closed:
                return OwnerTurn(False, False, False)
            turn = OwnerTurn(
                self._lifecycle_pending,
                self._data_pending,
                self._surface_pending,
            )
            self._lifecycle_pending = False
            self._data_pending = False
            self._surface_pending = False
            return turn

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._lifecycle_pending = False
            self._data_pending = False
            self._surface_pending = False
            plane = self._data_plane
            token = self._data_token
            self._data_plane = None
            self._data_token = None
        if plane is not None and token is not None:
            plane.unbind_owner_wake(token)


class HarmonicClock:
    """Pure arithmetic clock for one harmonic set of panel intervals."""

    __slots__ = ("_allowed", "_base_ms", "_default_ms", "_elapsed_ms")

    def __init__(self, intervals: Sequence[int], default_ms: int) -> None:
        normalized = tuple(sorted({_positive_int(value, "display interval") for value in intervals}))
        if not normalized:
            raise ValueError("display interval set must not be empty")
        default = _positive_int(default_ms, "default_ms")
        if default not in normalized:
            raise ValueError("default_ms must belong to the harmonic interval set")
        base = normalized[0]
        if any(value % base for value in normalized):
            raise ValueError("display intervals must be harmonic multiples of the base")
        self._allowed = frozenset(normalized)
        self._default_ms = default
        self._base_ms = base
        self._elapsed_ms = 0

    @property
    def base_ms(self) -> int:
        return self._base_ms

    @property
    def elapsed_ms(self) -> int:
        return self._elapsed_ms

    @property
    def intervals(self) -> tuple[int, ...]:
        return tuple(sorted(self._allowed))

    def _interval(self, value: object) -> int:
        normalized = _positive_int(value, "display interval")
        if normalized not in self._allowed:
            raise ValueError(
                f"display interval {normalized} is not in {self.intervals}"
            )
        return normalized

    def rebase(self, panel_intervals: Iterable[int]) -> int:
        values = tuple(self._interval(value) for value in panel_intervals)
        self._base_ms = min(values, default=self._default_ms)
        self._elapsed_ms = 0
        return self._base_ms

    def advance(self) -> int:
        self._elapsed_ms += self._base_ms
        return self._elapsed_ms

    def group_due(self, elapsed_ms: int, member_intervals: Iterable[int]) -> bool:
        elapsed = _nonnegative_int(elapsed_ms, "elapsed_ms")
        members = tuple(self._interval(value) for value in member_intervals)
        if not members:
            raise ValueError("a presentation group must have at least one interval")
        return elapsed % max(members) == 0


@dataclass(frozen=True, slots=True)
class SurfaceUpdate:
    """One staged operation awaiting board-coherent presentation."""

    panel_id: str
    serial: int
    host_token: object
    publication: SignalPublication
    value: SignalValue
    future: Future
    replacement: bool

    def __post_init__(self) -> None:
        if not isinstance(self.panel_id, str) or not self.panel_id.strip():
            raise ValueError("surface update panel_id must be non-empty text")
        if self.panel_id.strip() != self.panel_id:
            raise ValueError("surface update panel_id must be canonical text")
        object.__setattr__(self, "serial", _positive_int(self.serial, "surface update serial"))
        if not isinstance(self.publication, SignalPublication):
            raise TypeError("surface update requires SignalPublication")
        if not isinstance(self.value, SignalValue):
            raise TypeError("surface update requires SignalValue")
        if self.publication.value(self.value.name) is not self.value:
            raise ValueError("surface update value is not owned by its publication")
        if not isinstance(self.future, Future):
            raise TypeError("surface update future must be Future")
        if type(self.replacement) is not bool:
            raise TypeError("surface update replacement must be bool")


@runtime_checkable
class SurfacePort(Protocol):
    @property
    def panel_id(self) -> str: ...

    @property
    def signal_name(self) -> str: ...

    @property
    def display_interval_ms(self) -> int: ...

    def presented_publication(self) -> SignalPublication | None: ...

    def prepare(
        self,
        value: SignalValue,
        publication: SignalPublication,
    ) -> SurfaceUpdate | None: ...

    def observe(self, update: SurfaceUpdate, operation: object) -> None: ...

    def can_accept(self, update: SurfaceUpdate, operation: object) -> bool: ...

    def accept(self, update: SurfaceUpdate, operation: object) -> bool: ...

    def reject(
        self,
        update: SurfaceUpdate,
        error: BaseException | None,
    ) -> None: ...

    def finish_unpresented(self, update: SurfaceUpdate) -> None: ...

    def report_waiting(self, missing_signal: str) -> None: ...


class SurfaceBatchArbiter:
    """Stage and commit complete surface groups without presenting partial boards."""

    __slots__ = ("_batches", "_channels", "_closed")

    def __init__(self, channels: OwnerChannels) -> None:
        if not callable(getattr(channels, "notify_surface", None)):
            raise TypeError("surface arbiter requires owner channels")
        self._channels = channels
        self._batches: deque[tuple[SurfaceUpdate, ...]] = deque()
        self._closed = False

    @property
    def channels(self) -> OwnerChannels:
        return self._channels

    @property
    def pending_batches(self) -> int:
        return len(self._batches)

    @staticmethod
    def _panel_id(port: SurfacePort) -> str:
        panel_id = getattr(port, "panel_id", None)
        if not isinstance(panel_id, str) or not panel_id.strip() or panel_id.strip() != panel_id:
            raise ValueError("surface port panel_id must be canonical non-empty text")
        return panel_id

    @staticmethod
    def _signal_name(port: SurfacePort) -> str:
        signal_name = getattr(port, "signal_name", None)
        if not isinstance(signal_name, str):
            raise TypeError("surface port signal_name must be text")
        return signal_name

    @staticmethod
    def _finish_unpresented(
        submitted: Sequence[tuple[SurfacePort, SurfaceUpdate]],
    ) -> None:
        for port, update in submitted:
            try:
                port.finish_unpresented(update)
            except BaseException:
                pass

    def enqueue_group(
        self,
        ports: Sequence[SurfacePort],
        front: SignalFront,
    ) -> bool:
        if self._closed:
            return False
        if not isinstance(front, SignalFront):
            raise TypeError("surface group requires SignalFront")
        members = tuple(ports)
        if not members:
            raise ValueError("surface group must contain at least one port")
        panel_ids = tuple(self._panel_id(port) for port in members)
        if len(set(panel_ids)) != len(panel_ids):
            raise ValueError("surface group contains duplicate panel ids")

        inputs: list[tuple[SurfacePort, SignalValue, SignalPublication]] = []
        for port, panel_id in zip(members, panel_ids, strict=True):
            signal_name = self._signal_name(port)
            value = front.value(signal_name)
            publication = front.publication(signal_name)
            if value is None or publication is None:
                port.report_waiting(signal_name)
                return False
            inputs.append((port, value, publication))

        submitted: list[tuple[SurfacePort, SurfaceUpdate]] = []
        try:
            for port, value, publication in inputs:
                update = port.prepare(value, publication)
                if update is None:
                    continue
                if not isinstance(update, SurfaceUpdate):
                    raise TypeError("surface port prepare() must return SurfaceUpdate or None")
                if update.panel_id != self._panel_id(port):
                    raise ValueError("surface update belongs to another panel")
                submitted.append((port, update))
        except BaseException:
            self._finish_unpresented(submitted)
            return False
        if not submitted:
            return False

        batch = tuple(update for _port, update in submitted)
        self._batches.append(batch)
        for update in batch:
            update.future.add_done_callback(
                lambda _future: self._channels.notify_surface()
            )
        return True

    @staticmethod
    def _reject(
        port: SurfacePort | None,
        update: SurfaceUpdate,
        error: BaseException | None,
    ) -> None:
        if port is None:
            return
        try:
            port.reject(update, error)
        except BaseException:
            pass

    @staticmethod
    def _superseded(future: Future) -> bool:
        """Whether the host coalesced this update away for a newer one.

        A latest-only host cancels a queued render when a newer frame arrives
        behind it.  That is flow control -- the newer render is already queued
        on the same host, inside a newer batch for the same group -- so a
        cancelled member is an unpresented update, not a failed one, and must
        never sink the batch it travelled in as an error.
        """

        if future.cancelled():
            return True
        return isinstance(future.exception(), CancelledError)

    def drain(self, resolve: Callable[[str], SurfacePort | None]) -> None:
        if not callable(resolve):
            raise TypeError("surface batch resolver must be callable")
        if self._closed:
            return
        pending: deque[tuple[SurfaceUpdate, ...]] = deque()
        while self._batches:
            batch = self._batches.popleft()
            if not all(update.future.done() for update in batch):
                pending.append(batch)
                continue

            if any(self._superseded(update.future) for update in batch):
                # A batch is one causal group frozen from one front.  When any
                # member was coalesced away, the newer render replacing it is
                # part of a NEWER batch for the same group -- presenting the
                # remaining members now would put half the group one shot
                # ahead of the other half, which is exactly what a batch
                # exists to prevent.  The whole batch leaves unpresented and
                # the newer batch presents every member together.
                resolved: list[tuple[SurfacePort, SurfaceUpdate]] = []
                for update in batch:
                    try:
                        port = resolve(update.panel_id)
                    except BaseException:
                        port = None
                    if port is not None:
                        resolved.append((port, update))
                self._finish_unpresented(resolved)
                continue

            records: list[
                tuple[SurfacePort | None, SurfaceUpdate, object | None, bool]
            ] = []
            batch_error: BaseException | None = None
            for update in batch:
                try:
                    port = resolve(update.panel_id)
                except BaseException as error:
                    port = None
                    batch_error = batch_error or error
                if port is None:
                    batch_error = batch_error or RuntimeError(
                        f"surface port {update.panel_id!r} no longer exists"
                    )
                    records.append((None, update, None, False))
                    continue
                try:
                    operation = update.future.result()
                except BaseException as error:
                    batch_error = batch_error or error
                    records.append((port, update, error, False))
                    continue
                try:
                    port.observe(update, operation)
                except BaseException as error:
                    batch_error = batch_error or error
                    records.append((port, update, error, False))
                    continue
                records.append((port, update, operation, True))

            if batch_error is None:
                for port, update, operation, successful in records:
                    if not successful or port is None:
                        batch_error = batch_error or RuntimeError(
                            "surface batch contains an unusable operation"
                        )
                        break
                    try:
                        accepted = port.can_accept(update, operation)
                    except BaseException as error:
                        batch_error = error
                        break
                    if not accepted:
                        batch_error = RuntimeError(
                            f"surface update {update.panel_id!r} is no longer acceptable"
                        )
                        break

            if batch_error is not None:
                for port, update, _operation, _successful in records:
                    self._reject(port, update, batch_error)
                continue

            accepted: list[tuple[SurfacePort, SurfaceUpdate]] = []
            try:
                for port, update, operation, successful in records:
                    assert port is not None and successful
                    if not port.accept(update, operation):
                        raise RuntimeError(
                            f"surface port {update.panel_id!r} rejected an acceptable update"
                        )
                    accepted.append((port, update))
            except BaseException as error:
                accepted_ids = {id(update) for _port, update in accepted}
                for port, update, _operation, _successful in records:
                    if id(update) not in accepted_ids:
                        self._reject(port, update, error)
        self._batches.extend(pending)

    def cancel_all(self) -> None:
        while self._batches:
            batch = self._batches.popleft()
            for update in batch:
                try:
                    update.future.cancel()
                except BaseException:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel_all()


class BoardScheduler:
    """Freeze one front per display tick and schedule coherent surface groups."""

    __slots__ = (
        "_arbiter",
        "_clock",
        "_closed",
        "_last_front",
        "_owed",
        "_plane",
        "_ports",
    )

    def __init__(
        self,
        plane: SignalDataPlane,
        clock: HarmonicClock,
        arbiter: SurfaceBatchArbiter,
        ports: Callable[[], Sequence[SurfacePort]],
    ) -> None:
        if not callable(getattr(plane, "freeze", None)) or not callable(
            getattr(plane, "set_front_signals", None)
        ):
            raise TypeError("board scheduler requires a signal data plane")
        if not isinstance(clock, HarmonicClock):
            raise TypeError("board scheduler requires HarmonicClock")
        if not isinstance(arbiter, SurfaceBatchArbiter):
            raise TypeError("board scheduler requires SurfaceBatchArbiter")
        if not callable(ports):
            raise TypeError("board scheduler ports must be callable")
        self._plane = plane
        self._clock = clock
        self._arbiter = arbiter
        self._ports = ports
        self._owed: dict[object, bool] = {}
        self._closed = False
        self._last_front = SignalFront({}, {})

    @property
    def owed_groups(self) -> frozenset[object]:
        return frozenset(self._owed)

    @property
    def last_front(self) -> SignalFront:
        return self._last_front

    @staticmethod
    def _presented_publication(port: SurfacePort) -> SignalPublication | None:
        value = getattr(port, "presented_publication", None)
        if callable(value):
            value = value()
        if value is not None and not isinstance(value, SignalPublication):
            raise TypeError("surface port presented_publication must be SignalPublication or None")
        return value

    def _port_map(self) -> dict[str, SurfacePort]:
        values = tuple(self._ports())
        result: dict[str, SurfacePort] = {}
        for port in values:
            panel_id = SurfaceBatchArbiter._panel_id(port)
            if panel_id in result:
                raise ValueError(f"duplicate surface port panel id {panel_id!r}")
            result[panel_id] = port
        return result

    def _resolve_port(self, panel_id: str) -> SurfacePort | None:
        return self._port_map().get(panel_id)

    def on_tick(self) -> SignalFront:
        if self._closed:
            return self._last_front
        # The reader declares what it reads.  The port list IS the truth of
        # what the board shows, so the coherent-front request is projected
        # from it on every tick rather than book-kept beside every panel
        # add/remove/retarget.  Without this declaration the plane has no
        # requested set, builds no lineage components, and every signal
        # floats at its own latest publication -- a camera panel one shot
        # ahead of the panel derived from it.  The plane no-ops on an
        # unchanged set, so this scheduler is the sole declaration authority.
        ports = tuple(self._ports())
        self._plane.set_front_signals(
            {SurfaceBatchArbiter._signal_name(port) for port in ports}
        )
        front = self._plane.freeze()
        if not isinstance(front, SignalFront):
            raise TypeError("signal data plane freeze() must return SignalFront")
        self._last_front = front
        elapsed = self._clock.advance()
        # Batch keys couple exactly what causality couples: ports whose
        # signals share one multi-signal lineage component present as one
        # same-shot batch.  Everything else -- including several views of the
        # SAME signal -- schedules per panel: the front already serves every
        # view one publication per signal, and batching same-signal views
        # would weld their cadences together and turn one slow view's
        # coalesced render into the whole board's abandoned batch.
        members_by_key: dict[object, list[SurfacePort]] = {}
        for port in ports:
            signal_name = SurfaceBatchArbiter._signal_name(port)
            continuous = front.continuous_group(signal_name)
            if continuous and len(continuous) > 1:
                key: object = ("continuous", continuous)
            else:
                key = ("panel", SurfaceBatchArbiter._panel_id(port))
            members_by_key.setdefault(key, []).append(port)

        for key, members in members_by_key.items():
            if isinstance(key, tuple) and key[0] == "continuous":
                # A debt owed by a port scheduled alone earlier carries into
                # the causal group it now belongs to.
                for port in members:
                    panel_key = ("panel", SurfaceBatchArbiter._panel_id(port))
                    if self._owed.pop(panel_key, None) is not None:
                        self._owed[key] = True

        active_keys = set(members_by_key)
        for key in tuple(self._owed):
            if key not in active_keys:
                self._owed.pop(key, None)

        for key, members in members_by_key.items():
            current = True
            intervals: list[int] = []
            for port in members:
                signal_name = SurfaceBatchArbiter._signal_name(port)
                publication = front.publication(signal_name)
                if publication is None or self._presented_publication(port) is not publication:
                    current = False
                intervals.append(getattr(port, "display_interval_ms"))
            if current:
                self._owed.pop(key, None)
                continue
            due = self._clock.group_due(elapsed, intervals)
            if not due and key not in self._owed:
                continue
            if self._arbiter.enqueue_group(tuple(members), front):
                self._owed.pop(key, None)
            else:
                self._owed[key] = True
        return front

    def on_owner_turn(self, poll_lifecycle: Callable[[], None]) -> None:
        if not callable(poll_lifecycle):
            raise TypeError("poll_lifecycle must be callable")
        if self._closed:
            return
        turn = self._arbiter.channels.take()
        if turn.surface:
            self._arbiter.drain(self._resolve_port)
        if turn.lifecycle:
            poll_lifecycle()
        if turn.lifecycle or turn.data:
            self._plane.freeze()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._owed.clear()
        self._arbiter.close()


__all__ = [
    "BoardScheduler",
    "HarmonicClock",
    "OwnerChannels",
    "OwnerTurn",
    "SurfaceBatchArbiter",
    "SurfacePort",
    "SurfaceUpdate",
    "WakeSink",
]
