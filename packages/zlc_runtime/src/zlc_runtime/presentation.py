"""Owner wakes, harmonic presentation cadence, and surface batching."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

from .plane import SignalDataPlane, SignalFront, SignalPublication, SignalValue
from .streams import EventRef


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
    front_refs: tuple[EventRef, ...]
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
        refs = tuple(self.front_refs)
        if not refs or any(not isinstance(value, EventRef) for value in refs):
            raise TypeError("surface update front_refs must contain EventRef values")
        object.__setattr__(self, "front_refs", refs)
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

    @property
    def surface_busy(self) -> bool: ...

    @property
    def presentation_current(self) -> bool: ...

    def presented_front_refs(self) -> tuple[EventRef, ...]: ...

    def prepare(
        self,
        value: SignalValue,
        publication: SignalPublication,
        front: SignalFront,
    ) -> SurfaceUpdate | None: ...

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
    updates: list[tuple[SurfacePort, SurfaceUpdate]]


class SurfaceBatchArbiter:
    """Assemble staged updates into shot cohorts and present them whole.

    One lineage group shows one shot number on screen: every member of a
    cohort is accepted in one owner-thread pass, a cohort that cannot
    complete never half-shows (a superseded member abandons the WHOLE
    cohort — the group skips to the newest shot together), and cohorts that
    share a panel present strictly in formation order so no panel ever
    regresses.
    """

    __slots__ = ("_closed", "_cohorts", "_wake")

    def __init__(self, wake: WakeSink) -> None:
        if not callable(getattr(wake, "request_owner_wake", None)):
            raise TypeError("surface arbiter requires an owner wake")
        self._wake = wake
        self._cohorts: list[_ShotCohort] = []
        self._closed = False

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
    def _front_refs(
        port: SurfacePort,
        front: SignalFront,
    ) -> tuple[EventRef, ...] | None:
        refs: list[EventRef] = []
        for name in SurfaceBatchArbiter._front_signals(port):
            publication = front.publication(name)
            if publication is None:
                return None
            refs.append(publication.event_ref)
        return tuple(refs)

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
        formation_complete: bool = False,
    ) -> bool:
        if self._closed:
            return False
        if not isinstance(front, SignalFront):
            raise TypeError("surface group requires SignalFront")
        if type(formation_complete) is not bool:
            raise TypeError("formation_complete must be bool")
        members = tuple(ports)
        if not members:
            raise ValueError("surface group must contain at least one port")
        panel_ids = tuple(self._panel_id(port) for port in members)
        if len(set(panel_ids)) != len(panel_ids):
            raise ValueError("surface group contains duplicate panel ids")

        inputs: list[tuple[SurfacePort, SignalValue, SignalPublication]] = []
        for port, panel_id in zip(members, panel_ids, strict=True):
            front_refs = self._front_refs(port, front)
            if front_refs is None:
                missing = next(
                    name
                    for name in self._front_signals(port)
                    if front.publication(name) is None
                )
                port.report_waiting(missing)
                return False
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
                expected_refs = self._front_refs(port, front)
                if expected_refs is None or update.front_refs != expected_refs:
                    raise ValueError("surface update currency differs from its front")
                submitted.append((port, update))
        except BaseException:
            self._finish_unpresented(submitted)
            return False
        if not submitted:
            return False

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
        # Keep the submitting port beside its operation.  The board's current
        # resolver may stop returning it while the render is in flight (panel
        # removal or replacement), but that original port still owns the
        # pending entry and any staged generation host that must be retired.
        cohort.updates.extend(submitted)
        if formation_complete and not cohort.window_panels:
            cohort.sealed = True
        if not cohort.sealed and cohort.window_panels and cohort.window_panels <= {
            update.panel_id for _port, update in cohort.updates
        }:
            # Every displayed follower is aboard: nothing else can join this
            # shot, so seal now rather than waiting out the fallback window.
            cohort.sealed = True
        for _port, update in submitted:
            update.future.add_done_callback(
                lambda _future: self._wake.request_owner_wake()
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
                update.future.done() for _port, update in cohort.updates
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
                panel_ids = {
                    update.panel_id for _port, update in cohort.updates
                }
                if panel_ids & blocked_panels:
                    blocked_panels |= panel_ids
                    continue
                if not cohort.abandoned and any(
                    update.future.done() and self._superseded(update.future)
                    for _port, update in cohort.updates
                ):
                    # One member was coalesced away for a newer frame: the
                    # WHOLE shot leaves unpresented and the group jumps to
                    # the newest shot together — never half a board one shot
                    # ahead of the other half.
                    cohort.abandoned = True
                if not cohort.sealed or not all(
                    update.future.done() for _port, update in cohort.updates
                ):
                    blocked_panels |= panel_ids
                    continue
                if cohort.abandoned:
                    self._finish_unpresented(cohort.updates)
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
        for origin, update in cohort.updates:
            try:
                port = resolve(update.panel_id)
            except BaseException as error:
                port = None
                batch_error = batch_error or error
                blamed = blamed or update
            if port is not origin:
                batch_error = batch_error or RuntimeError(
                    f"surface port {update.panel_id!r} no longer resolves "
                    "to its submitting port"
                )
                blamed = blamed or update
                self._finish_unpresented(((origin, update),))
                records.append((None, update, None, False))
                continue
            try:
                operation = update.future.result()
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
                    # Nobody failed.  This port's host was replaced after the
                    # update was staged -- a cell kind changed, a panel was
                    # rebuilt -- so the cohort cannot flip and the next shot
                    # renders against the new host.  Told as an error, it put
                    # a red mark on a panel that was about to draw perfectly.
                    self._abandon(records, None, None)
                    return

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
        error: BaseException | None,
        blamed: SurfaceUpdate | None,
    ) -> None:
        """Drop the cohort, telling only the member that could not go.

        The group flips together or not at all, so ONE member's failure ends
        the shot for every member -- but it is not a failure OF the others.
        Handing them all the same error is how one panel's trouble painted
        itself on every card on the board, each of which was about to redraw
        perfectly on the next shot.  They leave unpresented instead, which is
        what an abandoned cohort already does.  With no member at fault
        (``blamed`` is None) that is the whole story: nobody is told.
        """

        for port, update, _outcome, _successful in records:
            if port is None:
                continue
            if error is not None and update is blamed:
                cls._reject(port, update, error)
            else:
                cls._finish_unpresented(((port, update),))

    def cancel_all(self) -> None:
        while self._cohorts:
            cohort = self._cohorts.pop()
            for _port, update in cohort.updates:
                try:
                    update.future.cancel()
                except BaseException:
                    pass
            self._finish_unpresented(cohort.updates)

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
        "_admission_owed",
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
            or not callable(getattr(plane, "follower_edges", None))
            or not callable(getattr(plane, "latest_publication", None))
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
        self._owed: set[str] = set()
        # A due panel whose source has not advanced owes only a SURFACE
        # admission, not a render.  The publication and its indexed Dataset
        # already exist in Runtime.  Keeping this separate lets a travelling
        # cohort finish during Pause without advancing the frozen board.
        self._admission_owed: set[str] = set()
        self._closed = False
        self._last_front = SignalFront({}, {})

    @staticmethod
    def _presented_front_refs(port: SurfacePort) -> tuple[EventRef, ...]:
        value = getattr(port, "presented_front_refs", None)
        if not callable(value):
            raise TypeError("surface port must provide presented_front_refs()")
        refs = tuple(value())
        if any(not isinstance(item, EventRef) for item in refs):
            raise TypeError(
                "surface port presented_front_refs must contain EventRef values"
            )
        return refs

    def _port_map(self) -> dict[str, SurfacePort]:
        values = tuple(self._ports())
        result: dict[str, SurfacePort] = {}
        for port in values:
            panel_id = SurfaceBatchArbiter._panel_id(port)
            if panel_id in result:
                raise ValueError(f"duplicate surface port panel id {panel_id!r}")
            result[panel_id] = port
        return result

    def _blocked_surface_panels(
        self,
        ports: Sequence[SurfacePort],
        front: SignalFront,
        eligible: set[str],
    ) -> tuple[set[str], dict[str, frozenset | None]]:
        candidates: dict[str, frozenset | None] = {}
        busy_roots: set[frozenset] = set()
        busy_unresolved: set[str] = set()
        for port in ports:
            panel_id = SurfaceBatchArbiter._panel_id(port)
            if panel_id not in eligible:
                continue
            refs = SurfaceBatchArbiter._front_refs(port, front)
            if refs is None or self._presented_front_refs(port) == refs:
                continue
            roots = self._port_shot_roots(port, front)
            candidates[panel_id] = roots
            busy = getattr(port, "surface_busy", None)
            if type(busy) is not bool:
                raise TypeError("surface port must provide boolean surface_busy")
            if not busy:
                continue
            if roots is None:
                busy_unresolved.add(panel_id)
            else:
                busy_roots.add(roots)
        blocked = {
            panel_id
            for panel_id, roots in candidates.items()
            if panel_id in busy_unresolved
            or (roots is not None and roots in busy_roots)
        }
        return blocked, candidates

    @staticmethod
    def _displayed_signals(ports: Sequence[SurfacePort]) -> set[str]:
        return {
            name
            for port in ports
            for name in SurfaceBatchArbiter._front_signals(port)
        }

    def _follower_outputs(self, displayed: set[str]) -> set[str]:
        return {
            output
            for source, output in self._plane.follower_edges()
            if source in displayed and output in displayed
        }

    def _shot_roots(
        self,
        publication: SignalPublication,
    ) -> frozenset | None:
        """The shot-root event refs one publication descends from.

        Runtime owns publication lineage, including replayed exact events.  A
        derived or follower publication therefore receives the SAME roots as
        its source.  None means lineage could not be resolved, so the batch
        presents solo rather than guessing.
        """

        try:
            return self._plane.publication_roots(publication)
        except RuntimeError:
            return None

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
        displayed = self._displayed_signals(ports)
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
        follower_outputs = self._follower_outputs(displayed)
        due = {
            SurfaceBatchArbiter._panel_id(port): self._clock.group_due(
                elapsed, (getattr(port, "display_interval_ms"),)
            )
            for port in ports
        }
        # A coherent component can intentionally remain on its accepted shot
        # while a processor finishes the new shot's derived siblings.  The
        # raw latest source is then ahead of the frozen front.  Record the due
        # decision for every displayed view of that old shot so the processor's
        # completion wake can stage the whole new component without a second beat.
        held_panels: set[str] = set()
        held_roots: set[frozenset] = set()
        for port in ports:
            panel_id = SurfaceBatchArbiter._panel_id(port)
            held = False
            for name in SurfaceBatchArbiter._front_signals(port):
                latest = self._plane.latest_publication(name)
                current = front.publication(name)
                if latest is not None and (
                    current is None or latest.event_ref != current.event_ref
                ):
                    held = True
                    break
            if not held:
                continue
            held_panels.add(panel_id)
            roots = self._port_shot_roots(port, front)
            if roots is not None:
                held_roots.add(roots)
        if held_roots:
            for port in ports:
                roots = self._port_shot_roots(port, front)
                if roots is not None and roots in held_roots:
                    held_panels.add(SurfaceBatchArbiter._panel_id(port))
        window_panels = frozenset(
            SurfaceBatchArbiter._panel_id(port)
            for port in ports
            if due[SurfaceBatchArbiter._panel_id(port)] and any(
                name in follower_outputs
                for name in SurfaceBatchArbiter._front_signals(port)
            )
        )
        follower_panels = frozenset(
            SurfaceBatchArbiter._panel_id(port)
            for port in ports
            if any(
                name in follower_outputs
                for name in SurfaceBatchArbiter._front_signals(port)
            )
        )
        surface_candidates = {
            SurfaceBatchArbiter._panel_id(port)
            for port in ports
            if (
                due[SurfaceBatchArbiter._panel_id(port)]
                or SurfaceBatchArbiter._panel_id(port) in self._owed
                or SurfaceBatchArbiter._panel_id(port) in self._admission_owed
            )
        }
        blocked_surfaces, candidate_roots = self._blocked_surface_panels(
            ports,
            front,
            surface_candidates,
        )
        # Panels stage individually; the arbiter couples exactly what
        # causality couples by assembling equal shot-root batches into one
        # cohort.  Several views of one signal share roots and flip
        # together; unrelated panels never wait on each other.
        for port in ports:
            signal_name = SurfaceBatchArbiter._signal_name(port)
            panel_id = SurfaceBatchArbiter._panel_id(port)
            publication = front.publication(signal_name)
            front_refs = SurfaceBatchArbiter._front_refs(port, front)
            if (
                front_refs is not None
                and self._presented_front_refs(port) == front_refs
            ):
                waiting_for_data = (
                    panel_id in self._owed
                    and (panel_id in follower_panels or panel_id in held_panels)
                ) or panel_id in window_panels or (
                    panel_id in held_panels
                    and due[panel_id]
                )
                if waiting_for_data:
                    # This follower is due for the source shot being staged,
                    # or this coherent component is waiting for a derived
                    # sibling.  Remember the cadence decision so the matching
                    # completion wake can spend it without another display tick.
                    self._owed.add(panel_id)
                    self._admission_owed.discard(panel_id)
                elif due[panel_id]:
                    # The authored surface deadline was reached before a new
                    # source publication.  Its wake may spend this display
                    # debt immediately instead of waiting for another beat.
                    self._owed.discard(panel_id)
                    self._admission_owed.add(panel_id)
                else:
                    self._owed.discard(panel_id)
                continue
            if (
                not due[panel_id]
                and panel_id not in self._owed
                and panel_id not in self._admission_owed
            ):
                continue
            roots = candidate_roots.get(panel_id)
            if panel_id in blocked_surfaces:
                # No heavy surface FIFO: one travelling member holds its whole
                # same-shot group.  Runtime retains every indexed revision;
                # the display debt points only to whatever front is latest
                # after the current group accepts.
                self._owed.discard(panel_id)
                self._admission_owed.add(panel_id)
                continue
            if self._arbiter.enqueue_group(
                (port,),
                front,
                shot_roots=roots,
                window_panels=(
                    window_panels if publication is not None else frozenset()
                ),
            ):
                self._owed.discard(panel_id)
                self._admission_owed.discard(panel_id)
            else:
                if panel_id not in self._owed:
                    self._admission_owed.add(panel_id)
        active_panels = {
            SurfaceBatchArbiter._panel_id(port) for port in ports
        }
        for panel_id in tuple(self._owed | self._admission_owed):
            if panel_id not in active_panels:
                self._owed.discard(panel_id)
                self._admission_owed.discard(panel_id)
        self._arbiter.tick_boundary()
        return front

    def invalidate_presentations(self, panel_ids: Sequence[str]) -> None:
        """Mark unchanged publications owed after their Dataset view changed."""

        if self._closed:
            return
        active = set(self._port_map())
        for panel_id in panel_ids:
            selected = str(panel_id)
            if selected in active:
                self._admission_owed.add(selected)

    def stage_owed(self, *, admit_new: bool = True) -> SignalFront:
        """Stage already-due surfaces on the completion wake that makes them ready.

        This covers both halves of the same contract: a coherent component
        held while its derived sibling is produced, and a presentation-paced
        follower produced while its source surface renders.  Only debt already
        created by ``on_tick`` is eligible; this wake neither advances the
        clock nor lets a not-due panel bypass its authored interval.
        """

        eligible = self._owed | (self._admission_owed if admit_new else set())
        if self._closed or not eligible:
            return self._last_front
        ports = tuple(self._ports())
        displayed = self._displayed_signals(ports)
        self._plane.set_front_signals(displayed)
        front = self._plane.freeze()
        if not isinstance(front, SignalFront):
            raise TypeError("signal data plane freeze() must return SignalFront")
        self._last_front = front
        active_panels = {
            SurfaceBatchArbiter._panel_id(port) for port in ports
        }
        for panel_id in tuple(self._owed | self._admission_owed):
            if panel_id not in active_panels:
                self._owed.discard(panel_id)
                self._admission_owed.discard(panel_id)

        blocked_surfaces, candidate_roots = self._blocked_surface_panels(
            ports,
            front,
            eligible,
        )
        ready: dict[frozenset, list[SurfacePort]] = {}
        unresolved: list[SurfacePort] = []
        for port in ports:
            panel_id = SurfaceBatchArbiter._panel_id(port)
            if panel_id not in eligible:
                continue
            front_refs = SurfaceBatchArbiter._front_refs(port, front)
            if (
                front_refs is None
                or (
                    self._presented_front_refs(port) == front_refs
                    and port.presentation_current
                )
            ):
                continue
            publication = front.publication(
                SurfaceBatchArbiter._signal_name(port)
            )
            if publication is None:
                continue
            roots = candidate_roots.get(panel_id)
            if panel_id in blocked_surfaces:
                self._owed.discard(panel_id)
                self._admission_owed.add(panel_id)
                continue
            if roots is None:
                unresolved.append(port)
            else:
                ready.setdefault(roots, []).append(port)

        edges = self._plane.follower_edges()
        follower_outputs = {output for _source, output in edges}
        for roots, members in ready.items():
            member_panels = {
                SurfaceBatchArbiter._panel_id(member) for member in members
            }
            source_signals = {
                name
                for member in members
                for name in SurfaceBatchArbiter._front_signals(member)
            }
            window_panels = frozenset(
                SurfaceBatchArbiter._panel_id(candidate)
                for candidate in ports
                if SurfaceBatchArbiter._panel_id(candidate) in self._owed
                and (
                    any(
                        name in follower_outputs
                        for name in SurfaceBatchArbiter._front_signals(candidate)
                    )
                    and (
                        SurfaceBatchArbiter._panel_id(candidate) in member_panels
                        or any(
                            source in source_signals
                            and output
                            in SurfaceBatchArbiter._front_signals(candidate)
                            for source, output in edges
                        )
                    )
                )
            )
            if self._arbiter.enqueue_group(
                tuple(members),
                front,
                shot_roots=roots,
                window_panels=window_panels,
                # With no asynchronous follower, every ready same-root member
                # is submitted in this one call, so this formation is complete
                # without inventing a clock boundary on a completion wake.
                formation_complete=not window_panels,
            ):
                for member in members:
                    panel_id = SurfaceBatchArbiter._panel_id(member)
                    self._owed.discard(panel_id)
                    self._admission_owed.discard(panel_id)
        for port in unresolved:
            if self._arbiter.enqueue_group((port,), front):
                panel_id = SurfaceBatchArbiter._panel_id(port)
                self._owed.discard(panel_id)
                self._admission_owed.discard(panel_id)
        return front

    def _port_shot_roots(
        self,
        port: SurfacePort,
        front: SignalFront,
    ) -> frozenset | None:
        roots: set[object] = set()
        for name in SurfaceBatchArbiter._front_signals(port):
            publication = front.publication(name)
            if publication is None:
                return None
            current = self._shot_roots(publication)
            if current is None:
                return None
            roots.update(current)
        return frozenset(roots)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._owed.clear()
        self._admission_owed.clear()
        self._arbiter.close()


__all__ = [
    "BoardScheduler",
    "HarmonicClock",
    "SurfaceBatchArbiter",
    "SurfacePort",
    "SurfaceUpdate",
    "WakeSink",
]
