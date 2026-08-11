"""Owner wakes, harmonic presentation cadence, and surface batching."""

from __future__ import annotations

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

    __slots__ = ("_allowed", "_base_ms", "_elapsed_ms")

    def __init__(self, intervals: Sequence[int]) -> None:
        normalized = tuple(sorted({_positive_int(value, "display interval") for value in intervals}))
        if not normalized:
            raise ValueError("display interval set must not be empty")
        base = normalized[0]
        if any(value % base for value in normalized):
            raise ValueError("display intervals must be harmonic multiples of the base")
        self._allowed = frozenset(normalized)
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
        front: SignalFront,
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


@dataclass(slots=True)
class _ShotCohort:
    """One shot's presentation group: every staged update sharing its roots.

    A cohort is THE same-shot unit: its members flip together in one accept
    pass or not at all.  ``roots`` is the frozen set of shot-root event refs
    the members' publications descend from (None: unresolvable lineage — the
    cohort stays solo).  While unsealed, later staged batches with equal
    roots join.  ``window_panels`` names the displayed follower panels whose
    batches arrive one tick behind their source's pair: the cohort seals the
    moment every one of them has joined, with ``boundaries_left`` display
    ticks (counted only after the staged work completes) as the fallback for
    a follower that never stages — a not-due slow panel must not hold its
    shot open forever.
    """

    roots: frozenset | None
    window_panels: frozenset[str]
    boundaries_left: int
    sealed: bool
    abandoned: bool
    updates: list[SurfaceUpdate]


class SurfaceBatchArbiter:
    """Assemble staged updates into shot cohorts and present them whole.

    One lineage group shows one shot number on screen: every member of a
    cohort is accepted in one owner-thread pass, a cohort that cannot
    complete never half-shows (a superseded member abandons the WHOLE
    cohort — the group skips to the newest shot together), and cohorts that
    share a panel present strictly in formation order so no panel ever
    regresses.
    """

    __slots__ = ("_channels", "_closed", "_cohorts")

    def __init__(self, channels: OwnerChannels) -> None:
        if not callable(getattr(channels, "notify_surface", None)):
            raise TypeError("surface arbiter requires owner channels")
        self._channels = channels
        self._cohorts: list[_ShotCohort] = []
        self._closed = False

    @property
    def channels(self) -> OwnerChannels:
        return self._channels

    @property
    def pending_cohorts(self) -> int:
        return len(self._cohorts)

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
    def _front_signals(port: SurfacePort) -> tuple[str, ...]:
        """Every signal this port must read from ONE front.

        A panel shows one signal but may READ more of the same shot -- an
        image and the judgement annotating it.  Naming them here is what
        puts them in the plane's coherent set; a companion left out of it
        floats at its own latest publication, which is one shot ahead of the
        picture it is drawn on.
        """

        declared = getattr(port, "front_signals", None)
        names = (
            (SurfaceBatchArbiter._signal_name(port),)
            if declared is None
            else tuple(str(name) for name in declared if str(name))
        )
        return tuple(dict.fromkeys(names))

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
        *,
        shot_roots: frozenset | None = None,
        window_panels: frozenset[str] = frozenset(),
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
                update = port.prepare(value, publication, front)
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
        cohort: _ShotCohort | None = None
        if shot_roots is not None:
            for candidate in self._cohorts:
                # An abandoned cohort still collects its shot's stragglers
                # while unsealed: a follower batch for an abandoned shot must
                # leave unpresented with its group, never present alone one
                # shot ahead of (or behind) its source panel.
                if not candidate.sealed and candidate.roots == shot_roots:
                    cohort = candidate
                    break
        if cohort is None:
            cohort = _ShotCohort(
                roots=shot_roots,
                window_panels=window_panels,
                boundaries_left=2 if window_panels else 1,
                # Unresolvable lineage never joins anything: seal now so it
                # presents the moment it completes.
                sealed=shot_roots is None,
                abandoned=False,
                updates=[],
            )
            self._cohorts.append(cohort)
        cohort.updates.extend(batch)
        if not cohort.sealed and cohort.window_panels and cohort.window_panels <= {
            update.panel_id for update in cohort.updates
        }:
            # Every displayed follower is aboard: nothing else can join this
            # shot, so seal now rather than waiting out the fallback window.
            cohort.sealed = True
        for update in batch:
            update.future.add_done_callback(
                lambda _future: self._channels.notify_surface()
            )
        return True

    def tick_boundary(self) -> None:
        """Advance every unsealed cohort's join window by one display tick.

        Called by the scheduler at the end of each tick.  An ordinary cohort
        seals at its formation tick's boundary (same-tick siblings joined).
        An open-window cohort counts boundaries only once its own staged
        work is COMPLETE: a follower's publication is emitted during its
        source pair's commit, so it can only be staged on a tick after that
        commit — counting from formation would close the window while the
        pair was still rendering and strand every follower one cohort
        behind its shot.  Two boundaries after completion guarantee one full
        tick in which the follower's already-published signal gets staged.
        """

        if self._closed:
            return
        for cohort in self._cohorts:
            if cohort.sealed:
                continue
            if cohort.window_panels and not all(
                update.future.done() for update in cohort.updates
            ):
                continue
            cohort.boundaries_left -= 1
            if cohort.boundaries_left <= 0:
                cohort.sealed = True

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
        progressed = True
        while progressed:
            progressed = False
            # A panel's cohorts present strictly in formation order; a cohort
            # sharing any panel with an earlier still-pending cohort waits.
            # Cohorts with disjoint panels never block each other, so a slow
            # solo panel cannot throttle the rest of the board.
            blocked_panels: set[str] = set()
            for cohort in tuple(self._cohorts):
                panel_ids = {update.panel_id for update in cohort.updates}
                if panel_ids & blocked_panels:
                    blocked_panels |= panel_ids
                    continue
                if not cohort.abandoned and any(
                    update.future.done() and self._superseded(update.future)
                    for update in cohort.updates
                ):
                    # One member was coalesced away for a newer frame: the
                    # WHOLE shot leaves unpresented and the group jumps to
                    # the newest shot together — never half a board one shot
                    # ahead of the other half.
                    cohort.abandoned = True
                if not cohort.sealed or not all(
                    update.future.done() for update in cohort.updates
                ):
                    blocked_panels |= panel_ids
                    continue
                if cohort.abandoned:
                    resolved: list[tuple[SurfacePort, SurfaceUpdate]] = []
                    for update in cohort.updates:
                        try:
                            port = resolve(update.panel_id)
                        except BaseException:
                            port = None
                        if port is not None:
                            resolved.append((port, update))
                    self._finish_unpresented(resolved)
                    self._cohorts.remove(cohort)
                    progressed = True
                    continue
                self._present_cohort(cohort, resolve)
                self._cohorts.remove(cohort)
                progressed = True

    def _present_cohort(
        self,
        cohort: _ShotCohort,
        resolve: Callable[[str], SurfacePort | None],
    ) -> None:
        """Present one complete sealed cohort in a single accept pass."""

        records: list[
            tuple[SurfacePort | None, SurfaceUpdate, object | None, bool]
        ] = []
        batch_error: BaseException | None = None
        blamed: SurfaceUpdate | None = None
        for update in cohort.updates:
            try:
                port = resolve(update.panel_id)
            except BaseException as error:
                port = None
                batch_error = batch_error or error
                blamed = blamed or update
            if port is None:
                batch_error = batch_error or RuntimeError(
                    f"surface port {update.panel_id!r} no longer exists"
                )
                blamed = blamed or update
                records.append((None, update, None, False))
                continue
            try:
                operation = update.future.result()
            except BaseException as error:
                batch_error = batch_error or error
                blamed = blamed or update
                records.append((port, update, error, False))
                continue
            try:
                port.observe(update, operation)
            except BaseException as error:
                batch_error = batch_error or error
                blamed = blamed or update
                records.append((port, update, error, False))
                continue
            records.append((port, update, operation, True))

        if batch_error is None:
            for port, update, operation, successful in records:
                if not successful or port is None:
                    batch_error = batch_error or RuntimeError(
                        "surface batch contains an unusable operation"
                    )
                    blamed = blamed or update
                    break
                try:
                    accepted = port.can_accept(update, operation)
                except BaseException as error:
                    batch_error = error
                    blamed = update
                    break
                if not accepted:
                    batch_error = RuntimeError(
                        f"surface update {update.panel_id!r} is no longer acceptable"
                    )
                    blamed = update
                    break

        if batch_error is not None:
            self._abandon(records, batch_error, blamed)
            return

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
            remaining = tuple(
                record for record in records if id(record[1]) not in accepted_ids
            )
            self._abandon(
                remaining, error, remaining[0][1] if remaining else None
            )

    @classmethod
    def _abandon(
        cls,
        records: Sequence[tuple[SurfacePort | None, SurfaceUpdate, object | None, bool]],
        error: BaseException,
        blamed: SurfaceUpdate | None,
    ) -> None:
        """Drop the cohort, telling only the member that could not go.

        The group flips together or not at all, so ONE member's refusal ends
        the shot for every member -- but it is not a FAILURE of the others.
        Handing them all the same error is how a single panel rebuilt under a
        staged render (a cell kind changed, a host replaced) painted "no
        longer acceptable" on every card on the board, each of which was
        about to redraw perfectly on the next shot.  They leave unpresented
        instead, which is what an abandoned cohort already does.
        """

        for port, update, _outcome, _successful in records:
            if port is None:
                continue
            if update is blamed:
                cls._reject(port, update, error)
            else:
                cls._finish_unpresented(((port, update),))

    def cancel_all(self) -> None:
        while self._cohorts:
            cohort = self._cohorts.pop()
            for update in cohort.updates:
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
        if (
            not callable(getattr(plane, "freeze", None))
            or not callable(getattr(plane, "set_front_signals", None))
            or not callable(getattr(plane, "direct_parent_publications", None))
            or not callable(getattr(plane, "follower_edges", None))
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

    def _shot_roots(
        self,
        publication: SignalPublication,
    ) -> frozenset | None:
        """The shot-root event refs one publication descends from.

        Walking ``direct_parent_publications`` terminates at the parentless
        producers (the shot's cameras); a derived or follower publication
        therefore stamps the SAME roots as its source, which is what lets
        the arbiter assemble them into one cohort.  None means the lineage
        could not be resolved — the batch presents solo rather than guessing.
        """

        roots: set[object] = set()
        stack = [publication]
        while stack:
            current = stack.pop()
            refs = current.direct_parent_refs
            if not refs:
                roots.add(current.event_ref)
                continue
            parents = self._plane.direct_parent_publications(current)
            if len(parents) != len(refs):
                return None
            stack.extend(parents)
        return frozenset(roots)

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
        displayed = {
            name
            for port in ports
            for name in SurfaceBatchArbiter._front_signals(port)
        }
        self._plane.set_front_signals(displayed)
        front = self._plane.freeze()
        if not isinstance(front, SignalFront):
            raise TypeError("signal data plane freeze() must return SignalFront")
        self._last_front = front
        elapsed = self._clock.advance()
        # A presentation-paced follower's batch (a rolling trace of a
        # panel's fit signal) is published during its source pair's commit
        # and therefore stages one tick behind it.  Cohorts formed while
        # both sides of a follower edge are on the board hold a join window
        # for exactly those follower panels: the window closes the moment
        # every one has joined, or after the fallback boundaries when one
        # never stages.  Without a displayed follower nothing ever waits.
        edges = self._plane.follower_edges()
        follower_outputs = {
            output
            for source, output in edges
            if source in displayed and output in displayed
        }
        window_panels = frozenset(
            SurfaceBatchArbiter._panel_id(port)
            for port in ports
            if SurfaceBatchArbiter._signal_name(port) in follower_outputs
        )
        # Panels stage individually; the arbiter couples exactly what
        # causality couples by assembling equal shot-root batches into one
        # cohort.  Several views of one signal share roots and flip
        # together; unrelated panels never wait on each other.
        for port in ports:
            signal_name = SurfaceBatchArbiter._signal_name(port)
            panel_id = SurfaceBatchArbiter._panel_id(port)
            publication = front.publication(signal_name)
            if (
                publication is not None
                and self._presented_publication(port) is publication
            ):
                self._owed.pop(panel_id, None)
                continue
            due = self._clock.group_due(
                elapsed, (getattr(port, "display_interval_ms"),)
            )
            if not due and panel_id not in self._owed:
                continue
            if self._arbiter.enqueue_group(
                (port,),
                front,
                shot_roots=(
                    None
                    if publication is None
                    else self._shot_roots(publication)
                ),
                window_panels=(
                    window_panels if publication is not None else frozenset()
                ),
            ):
                self._owed.pop(panel_id, None)
            else:
                self._owed[panel_id] = True
        active_panels = {
            SurfaceBatchArbiter._panel_id(port) for port in ports
        }
        for panel_id in tuple(self._owed):
            if panel_id not in active_panels:
                self._owed.pop(panel_id, None)
        self._arbiter.tick_boundary()
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
