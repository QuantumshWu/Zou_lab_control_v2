"""Bounded latest-only live updates for every public plot-data type.

Acquisition threads publish immutable revisions and never execute plotting
code.  One controller worker owns the active update; while it renders, the
ingress retains only the newest pending revision.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
import math
from numbers import Integral
from threading import Event, RLock, Thread, current_thread
from time import monotonic
from typing import Any, Protocol, TypeAlias

from zlc_data import DatasetSchema, OwnedSnapshot

from ._live_channel import LatestRevisionChannel, RevisionedItem
from .data_contract import schema_equal, snapshot_revision, snapshot_schema
from .errors import RevisionError

from .config import PlotLibraryDefaults
from .fit import FitCancelled
from .primitives import ImageFrame, PlotInput, PulseTimelineData


_LivePayload: TypeAlias = PlotInput


@dataclass(frozen=True, slots=True)
class LiveDataRevision:
    """One Pulse timeline paired with its producer-owned revision.

    Dataset snapshots and image frames already own their revisions and travel
    directly.  PulseTimelineData deliberately has no revision field, so this
    is its sole public transport envelope.
    """

    revision: int
    payload: PulseTimelineData

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(
            self.revision, Integral
        ):
            raise RevisionError("revision must be an integer")
        revision = int(self.revision)
        if revision < 0:
            raise RevisionError("revision must be non-negative")
        if not isinstance(self.payload, PulseTimelineData):
            raise TypeError("payload must be PulseTimelineData")
        object.__setattr__(self, "revision", revision)


LiveUpdate: TypeAlias = (
    OwnedSnapshot | ImageFrame | LiveDataRevision
)
LiveContract: TypeAlias = (
    DatasetSchema
    | OwnedSnapshot
    | ImageFrame
    | LiveDataRevision
)


class _LatestPlotIngress(
    LatestRevisionChannel[_LivePayload]
):
    """Thread-safe capacity-one ingress with a fixed payload type.

    Dataset contracts retain exact schema and generation validation.  Pulse
    timeline contracts retain their exact payload class.  Every
    successful publication must advance the producer revision, and replacing a
    pending item records one coalesced update.
    """

    def __init__(
        self,
        contract: LiveContract,
        *,
        initial_revision: int | None = None,
    ) -> None:
        if initial_revision is not None:
            if isinstance(initial_revision, bool) or not isinstance(
                initial_revision, Integral
            ):
                raise RevisionError("initial_revision must be an integer or None")
            initial_revision = int(initial_revision)
            if initial_revision < 0:
                raise RevisionError("initial_revision must be non-negative")
        if isinstance(contract, DatasetSchema):
            payload_type: type[_LivePayload] = OwnedSnapshot
            schema: DatasetSchema | None = contract
            floor = initial_revision
        elif isinstance(contract, OwnedSnapshot):
            payload_type = OwnedSnapshot
            schema = snapshot_schema(contract)
            floor = snapshot_revision(contract)
            if initial_revision is not None and initial_revision != floor:
                raise RevisionError(
                    "initial_revision conflicts with the OwnedSnapshot contract"
                )
        elif isinstance(contract, ImageFrame):
            payload_type = ImageFrame
            schema = snapshot_schema(contract.snapshot)
            floor = contract.revision
            if initial_revision is not None and initial_revision != floor:
                raise RevisionError(
                    "initial_revision conflicts with the ImageFrame contract"
                )
        elif isinstance(contract, LiveDataRevision):
            payload_type = type(contract.payload)
            schema = None
            floor = contract.revision
            if initial_revision is not None and initial_revision != floor:
                raise RevisionError(
                    "initial_revision conflicts with the live-data contract"
                )
        else:
            raise TypeError(
                "contract must be DatasetSchema, OwnedSnapshot, ImageFrame, "
                "or LiveDataRevision"
            )
        super().__init__(initial_revision=floor)
        self._payload_type = payload_type
        self._schema = schema
        self._initial_revision = floor

    @property
    def schema(self) -> DatasetSchema | None:
        return self._schema

    @property
    def payload_type(self) -> type[_LivePayload]:
        return self._payload_type

    @property
    def initial_revision(self) -> int | None:
        return self._initial_revision

    @staticmethod
    def _normalize(update: LiveUpdate) -> tuple[int, _LivePayload]:
        if isinstance(update, OwnedSnapshot):
            return snapshot_revision(update), update
        if isinstance(update, ImageFrame):
            return update.revision, update
        if isinstance(update, LiveDataRevision):
            return update.revision, update.payload
        raise TypeError(
            "update must be OwnedSnapshot, ImageFrame, or LiveDataRevision"
        )

    def publish(self, update: LiveUpdate) -> RevisionedItem[_LivePayload]:
        revision, payload = self._normalize(update)
        with self._condition:
            if type(payload) is not self._payload_type:
                raise TypeError(
                    f"live payload must be {self._payload_type.__name__}, "
                    f"got {type(payload).__name__}"
                )
            if isinstance(payload, OwnedSnapshot):
                assert self._schema is not None
                incoming_schema = snapshot_schema(payload)
            elif isinstance(payload, ImageFrame):
                assert self._schema is not None
                incoming_schema = snapshot_schema(payload.snapshot)
            else:
                incoming_schema = None
            if incoming_schema is not None:
                if (
                    incoming_schema is not self._schema
                    and not schema_equal(self._schema, incoming_schema)
                ):
                    raise ValueError("live payload schema differs from ingress schema")
            return self._accept_item_locked(revision, payload)

class _LiveSession(Protocol):
    """Minimal session surface required by :class:`LivePlotController`."""

    @property
    def defaults(self) -> PlotLibraryDefaults: ...

    def prepare_live_frame(
        self,
        data: _LivePayload,
        *,
        revision: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Future[object]: ...

    def solve_live_frame(self, prepared: object) -> Future[object] | None: ...

    def commit_live_frame(
        self,
        prepared: object,
        solved: object | None = None,
    ) -> object | None: ...

    def publish_live_frame(self, finalization: object) -> None: ...

    def abort_live_frame(self, finalization: object) -> None: ...


_Dispatch = Callable[[Callable[[], None]], Future[Any]]
_PresentationDispatch = Callable[
    [Callable[[], None], Callable[[], None], Callable[[], None]],
    Future[Any],
]


@dataclass(frozen=True, slots=True)
class LiveUpdateError:
    revision: int
    error: Exception


@dataclass(frozen=True, slots=True)
class LivePlotMetrics:
    refresh_interval_ms: int
    maximum_refresh_hz: float
    running: bool
    active: bool
    active_revision: int | None
    published: int
    consumed: int
    coalesced_updates: int
    pending: bool
    update_attempts: int
    successful_updates: int
    failed_updates: int
    cancelled_updates: int
    last_published_revision: int | None
    last_consumed_revision: int | None
    last_presented_revision: int | None
    last_failed_revision: int | None


class _StoppedDispatch(RuntimeError):
    pass


class _RetryLiveFrame(RuntimeError):
    pass


class LivePlotController:
    """Cadenced bridge from immutable data revisions to one plot session.

    A dataset uses its displayed :class:`zlc_data.OwnedSnapshot` or fixed
    :class:`DatasetSchema` as the contract and continues to publish snapshots
    directly.  An :class:`ImageFrame` carries image data and its point overlay
    through the same capacity-one cadence.  A PulseTimeline uses a
    :class:`LiveDataRevision` containing the currently displayed payload as its
    typed contract and publishes subsequent envelopes of that payload class.

    An explicit owner-thread dispatcher is fixed at construction.  Otherwise,
    sessions may provide a stable ``owner_dispatch`` gateway that resolves the
    currently attached host for every callback.  A dispatcher returns the one
    Future that completes after its callback; a generic session without either
    route uses the direct/headless synchronous path.
    """

    __slots__ = (
        "_session",
        "_ingress",
        "_defaults",
        "_dispatch",
        "_control_dispatch",
        "_presentation_dispatch",
        "_lock",
        "_refresh_interval_ms",
        "_stop_event",
        "_control_event",
        "_thread",
        "_running",
        "_closed",
        "_deferred",
        "_active_revision",
        "_active_wake",
        "_attempts",
        "_successful",
        "_failed",
        "_cancelled",
        "_last_presented_revision",
        "_last_failed_revision",
        "_last_error",
    )

    def __init__(
        self,
        session: _LiveSession,
        contract: LiveContract,
        *,
        refresh_interval_ms: int | None = None,
        dispatch: _Dispatch | None = None,
        control_dispatch: _Dispatch | None = None,
        presentation_dispatch: _PresentationDispatch | None = None,
    ) -> None:
        for method_name in (
            "prepare_live_frame",
            "solve_live_frame",
            "commit_live_frame",
            "publish_live_frame",
            "abort_live_frame",
        ):
            if not callable(getattr(session, method_name, None)):
                raise TypeError(f"session must provide callable {method_name}(...)")
        defaults = getattr(session, "defaults", None)
        if not isinstance(defaults, PlotLibraryDefaults):
            raise TypeError("session.defaults must be PlotLibraryDefaults")
        if dispatch is None:
            owner_dispatch = getattr(session, "owner_dispatch", None)
            if owner_dispatch is not None and not callable(owner_dispatch):
                raise TypeError("session.owner_dispatch must be callable or None")
            dispatch = owner_dispatch
        elif not callable(dispatch):
            raise TypeError("dispatch must be callable or None")
        if control_dispatch is None:
            control_dispatch = dispatch
        elif not callable(control_dispatch):
            raise TypeError("control_dispatch must be callable or None")
        if presentation_dispatch is not None and not callable(
            presentation_dispatch
        ):
            raise TypeError("presentation_dispatch must be callable or None")
        candidate = getattr(session, "data_revision", None)
        initial_revision = (
            int(candidate)
            if isinstance(candidate, Integral) and not isinstance(candidate, bool)
            else None
        )
        selected_interval = (
            defaults.live.default_refresh_interval_ms
            if refresh_interval_ms is None
            else _validate_refresh_interval(refresh_interval_ms, defaults)
        )
        # Validate the configured default too; custom defaults remain the sole
        # authority rather than introducing a second cadence literal here.
        _validate_refresh_interval(selected_interval, defaults)
        self._session = session
        self._ingress = _LatestPlotIngress(
            contract,
            initial_revision=initial_revision,
        )
        initial_revision = self._ingress.initial_revision
        self._defaults = defaults
        self._dispatch = dispatch
        self._control_dispatch = control_dispatch
        self._presentation_dispatch = presentation_dispatch
        self._lock = RLock()
        self._refresh_interval_ms = selected_interval
        self._stop_event = Event()
        self._control_event = Event()
        self._thread: Thread | None = None
        self._running = False
        self._closed = False
        self._deferred: RevisionedItem[_LivePayload] | None = None
        self._active_revision: int | None = None
        self._active_wake: Event | None = None
        self._attempts = 0
        self._successful = 0
        self._failed = 0
        self._cancelled = 0
        self._last_presented_revision: int | None = initial_revision
        self._last_failed_revision: int | None = None
        self._last_error: LiveUpdateError | None = None

    @property
    def schema(self) -> DatasetSchema | None:
        return self._ingress.schema

    @property
    def payload_type(self) -> type[_LivePayload]:
        return self._ingress.payload_type

    @property
    def refresh_intervals_ms(self) -> tuple[int, ...]:
        return self._defaults.live.refresh_intervals_ms

    @property
    def refresh_interval_ms(self) -> int:
        with self._lock:
            return self._refresh_interval_ms

    @property
    def maximum_refresh_hz(self) -> float:
        return 1000.0 / min(self._defaults.live.refresh_intervals_ms)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def worker_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def set_refresh_interval(self, interval_ms: int) -> int:
        selected = _validate_refresh_interval(interval_ms, self._defaults)
        with self._lock:
            self._assert_open_locked()
            self._refresh_interval_ms = selected
            self._control_event.set()
            active_wake = self._active_wake
        if active_wake is not None:
            active_wake.set()
        return selected

    def publish(self, update: LiveUpdate) -> LiveUpdate:
        """Publish without waiting for cadence, dispatch, or rendering."""

        with self._lock:
            self._assert_open_locked()
        self._ingress.publish(update)
        return update

    def start(self) -> "LivePlotController":
        with self._lock:
            self._assert_open_locked()
            if self._running:
                return self
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("previous live worker is still stopping")
            self._stop_event.clear()
            self._control_event.clear()
            self._running = True
            worker = Thread(
                target=self._run,
                name="zlc-live-plot",
                daemon=True,
            )
            self._thread = worker
            worker.start()
        return self

    def stop(
        self,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Request worker shutdown and report whether it has stopped."""

        if not isinstance(wait, bool):
            raise TypeError("wait must be bool")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        with self._lock:
            thread = self._thread
            self._stop_event.set()
            self._control_event.set()
            active_wake = self._active_wake
        if active_wake is not None:
            active_wake.set()
        if thread is None:
            return True
        if wait and thread is not current_thread():
            selected_timeout = (
                self._defaults.runtime.shutdown_timeout_ms / 1000.0
                if timeout is None
                else timeout
            )
            thread.join(selected_timeout)
        return not thread.is_alive()

    def close(self, *, timeout: float | None = None) -> bool:
        with self._lock:
            if self._closed:
                thread = self._thread
                return thread is None or not thread.is_alive()
            self._closed = True
        stopped = self.stop(wait=True, timeout=timeout)
        self._ingress.close()
        with self._lock:
            # Pause retains one interrupted revision for resume; terminal close
            # does not.  An active worker already owns its frame locally.
            self._deferred = None
            if stopped:
                self._active_revision = None
                self._active_wake = None
        return stopped

    def pump_once(self) -> bool:
        """Synchronously present the latest revision while no worker runs."""

        with self._lock:
            self._assert_open_locked()
            if self._running:
                raise RuntimeError("pump_once cannot run while the live worker is active")
        frame = self._take_latest_candidate()
        if frame is None:
            return False
        completed = self._process(frame)
        if completed:
            self._clear_deferred(frame)
        return completed

    def metrics(self) -> LivePlotMetrics:
        ingress = self._ingress.metrics()
        with self._lock:
            return LivePlotMetrics(
                refresh_interval_ms=self._refresh_interval_ms,
                maximum_refresh_hz=self.maximum_refresh_hz,
                running=self._running,
                active=self._active_revision is not None,
                active_revision=self._active_revision,
                published=ingress.published,
                consumed=ingress.consumed,
                coalesced_updates=ingress.coalesced_updates,
                pending=ingress.pending or self._deferred is not None,
                update_attempts=self._attempts,
                successful_updates=self._successful,
                failed_updates=self._failed,
                cancelled_updates=self._cancelled,
                last_published_revision=ingress.last_published_revision,
                last_consumed_revision=ingress.last_consumed_revision,
                last_presented_revision=self._last_presented_revision,
                last_failed_revision=self._last_failed_revision,
            )

    @property
    def last_error(self) -> LiveUpdateError | None:
        with self._lock:
            return self._last_error

    def clear_error(self) -> None:
        with self._lock:
            self._last_error = None

    def __enter__(self) -> "LivePlotController":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False

    def _assert_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("live plot controller is closed")

    def _run(self) -> None:
        next_tick = monotonic()
        try:
            while not self._stop_event.is_set():
                wait_seconds = max(0.0, next_tick - monotonic())
                if self._control_event.wait(wait_seconds):
                    self._control_event.clear()
                    if self._stop_event.is_set():
                        break
                    next_tick = monotonic() + self.refresh_interval_ms / 1000.0
                    continue
                frame = self._take_latest_candidate()
                if frame is not None and self._process(frame):
                    self._clear_deferred(frame)
                interval_seconds = self.refresh_interval_ms / 1000.0
                next_tick += interval_seconds
                now = monotonic()
                if next_tick <= now:
                    skipped = math.floor((now - next_tick) / interval_seconds) + 1
                    next_tick += skipped * interval_seconds
        finally:
            with self._lock:
                self._running = False
                self._active_revision = None
                self._active_wake = None
                if self._thread is current_thread():
                    self._thread = None

    def _take_latest_candidate(self) -> RevisionedItem[_LivePayload] | None:
        newest = self._ingress.take_latest()
        with self._lock:
            if newest is not None:
                # One active frame is retained only across an explicit
                # stop/cancelled dispatch.  A newer producer revision replaces
                # it before resume or pump; pointer interaction never defers it.
                self._deferred = newest
            return self._deferred

    def _clear_deferred(self, frame: RevisionedItem[_LivePayload]) -> None:
        with self._lock:
            if self._deferred is frame:
                self._deferred = None

    def _process(self, frame: RevisionedItem[_LivePayload]) -> bool:
        with self._lock:
            self._active_revision = frame.revision
        try:
            self._apply_update(frame)
        except (_StoppedDispatch, _RetryLiveFrame):
            with self._lock:
                self._attempts += 1
                self._cancelled += 1
            return False
        except Exception as error:
            with self._lock:
                self._attempts += 1
                self._failed += 1
                self._last_failed_revision = frame.revision
                self._last_error = LiveUpdateError(frame.revision, error)
        else:
            with self._lock:
                self._attempts += 1
                self._successful += 1
                self._last_presented_revision = frame.revision
        finally:
            with self._lock:
                self._active_revision = None
                self._active_wake = None
        return True

    def _apply_update(self, frame: RevisionedItem[_LivePayload]) -> None:
        preparations: list[Future[object]] = []

        def begin() -> None:
            if self._stop_event.is_set():
                raise _StoppedDispatch("live-frame preparation reached its owner after shutdown")
            prepared = self._session.prepare_live_frame(
                frame.payload,
                revision=frame.revision,
                cancelled=self._stop_event.is_set,
            )
            if not isinstance(prepared, Future):
                raise TypeError("prepare_live_frame must return concurrent.futures.Future")
            preparations.append(prepared)

        self._dispatch_and_wait(self._control_dispatch, begin)
        if len(preparations) != 1:
            raise RuntimeError("live-frame preparation did not return one future")
        try:
            prepared_frame = self._wait_for_future(preparations[0])
        except FitCancelled as error:
            if self._stop_event.is_set():
                raise _StoppedDispatch("live-frame preparation stopped") from error
            raise _RetryLiveFrame("live-frame context changed during preparation") from error

        # The fit half of the pair: solved to completion on the fit executor
        # before the commit is dispatched, so the committed front is born
        # complete.  The controller thread is the natural place to wait — the
        # render worker stays free for interaction while the solver runs.
        solved: object | None = None
        solve_future = self._session.solve_live_frame(prepared_frame)
        if solve_future is not None:
            if not isinstance(solve_future, Future):
                raise TypeError(
                    "solve_live_frame must return concurrent.futures.Future or None"
                )
            solved = self._wait_for_future(solve_future)

        accepted: list[object] = []

        def commit() -> None:
            if self._stop_event.is_set():
                raise _StoppedDispatch("live-frame commit reached its owner after shutdown")
            finalization = self._session.commit_live_frame(
                prepared_frame,
                solved,
            )
            if finalization is None:
                raise _RetryLiveFrame(
                    "live-frame context changed before presentation"
                )
            accepted.append(finalization)

        def published() -> None:
            if len(accepted) == 1:
                self._session.publish_live_frame(accepted[0])

        def abort() -> None:
            if len(accepted) == 1:
                self._session.abort_live_frame(accepted[0])

        presentation_dispatch = self._presentation_dispatch
        if presentation_dispatch is None:
            def present_and_publish() -> None:
                commit()
                published()

            self._dispatch_and_wait(self._dispatch, present_and_publish)
            return
        dispatched = presentation_dispatch(commit, published, abort)
        if not isinstance(dispatched, Future):
            raise TypeError(
                "presentation_dispatch must return concurrent.futures.Future"
            )
        self._wait_for_future(dispatched)

    def _dispatch_and_wait(
        self,
        dispatch: _Dispatch | None,
        callback: Callable[[], None],
    ) -> None:
        if dispatch is None:
            callback()
            return
        dispatched = dispatch(callback)
        if not isinstance(dispatched, Future):
            raise TypeError("dispatch must return concurrent.futures.Future")
        self._wait_for_future(dispatched)

    def _wait_for_future(
        self,
        future: Future[Any],
    ) -> Any:
        wake = Event()
        future.add_done_callback(lambda _future: wake.set())
        with self._lock:
            self._active_wake = wake
        while not future.done():
            if self._stop_event.is_set():
                future.cancel()
            wake.wait()
            wake.clear()
        try:
            return future.result()
        except CancelledError as error:
            if self._stop_event.is_set():
                raise _StoppedDispatch("active live work cancelled during shutdown") from error
            raise


def _validate_refresh_interval(
    interval_ms: int,
    defaults: PlotLibraryDefaults,
) -> int:
    if isinstance(interval_ms, bool) or not isinstance(interval_ms, int):
        raise TypeError("refresh_interval_ms must be an integer preset")
    selected = int(interval_ms)
    if selected not in defaults.live.refresh_intervals_ms:
        raise ValueError(
            f"refresh_interval_ms must be one of {defaults.live.refresh_intervals_ms}"
        )
    return selected


__all__ = [
    "LiveContract",
    "LiveDataRevision",
    "LivePlotController",
    "LivePlotMetrics",
    "LiveUpdate",
    "LiveUpdateError",
]
