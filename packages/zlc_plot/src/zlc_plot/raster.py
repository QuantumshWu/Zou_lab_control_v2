"""Immutable Agg-to-frontend hand-off and serial plot worker.

The Qt adapter consumes only :class:`RasterFront` values.  A live Matplotlib
Figure, Artist, mutable array view, or Qt object never crosses this boundary.
All plot mutations and raster capture run on the single worker owned by
``RasterPlotHost``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future, InvalidStateError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Condition, Event, Lock, Thread, current_thread
from time import monotonic
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from . import _raster_kernels as kernels
from zlc_data.units import UnitRegistry

from ._axis_transform import AxisTransform
from ._validation import finite_real as _finite
from ._validation import text
from .config import DEFAULTS, PlotLibraryDefaults
from .front import (
    RasterBuffer,
    RasterFront,
    RasterIdentity,
    RasterInteractionMap,
    RasterOperation,
)
from .selectors import (
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorState,
)

if TYPE_CHECKING:
    from .fit import (
        FacetFitBatchResult,
        FitEngine,
        FitModelSpec,
        FitOptions,
        FitResult,
    )
    from .layout import SurfacePlan
    from .primitives import ImageFrame, ImagePointOverlay, PlotInput
    from .session import (
        DisplayDescription,
        FitEvent,
        PlotSession,
        SelectionEvent,
        SelectorData,
    )
    from .state import DisplayState
    from .specs import PlotSpec


ValueT = TypeVar("ValueT")
_UNSET = object()


class _DispatchMode(str, Enum):
    CONTROL = "control"
    ADAPTIVE = "adaptive"
    PUBLISH = "publish"
    PRESENTATION = "presentation"

    @property
    def publishes(self) -> bool:
        return self in {_DispatchMode.PUBLISH, _DispatchMode.PRESENTATION}


class _FrameSuperseded(RuntimeError):
    """A prepared frame no longer matches the explicitly changed session context."""


@dataclass(slots=True)
class _DataFrame:
    """One active or latest-only complete presentation input."""

    data: object
    revision: int | None
    completion: Future["RasterOperation[DisplayDescription]"]
    started_at: float = 0.0
    cancel: Event = field(default_factory=Event)
    stage: str = "latest"


def _plot_input_revision(data: object) -> int | None:
    snapshot = getattr(data, "snapshot", data)
    revision = getattr(getattr(snapshot, "ref", None), "revision", None)
    value = getattr(revision, "value", revision)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _pointer_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object | None:
    action = args[0]
    if action in {"move", "leave"}:
        return "pointer-motion"
    if action == "scroll":
        return "pointer-scroll"
    return None

class _HandArbiter:
    """A person's hand outranks every camera IN THE PROCESS.

    :meth:`RasterPlotHost._take_next_task` already rules that a pointer
    beats a data frame inside one host.  The bench found the identical
    competition BETWEEN hosts: a panel's Edit surface and its live card
    render on two threads but share one machine, and the card's
    full-resolution committed frames -- an all-cores kernel plus a
    GIL-held compose -- stalled the drag on the Edit surface for a third
    of a second at a time (measured over 1024x1024 data: per-move p90
    59 ms alone, 354 ms with the sibling live).  The ruling's true scope
    is the machine, so the arbiter is process-wide.

    Yielding costs nothing but freshness: a data frame retains only its
    latest successor, so work deferred while the hand moves collapses to
    one frame the moment it stops.  A hold is never a promise either --
    it expires on its own, sized from what that host's pointer work
    actually costs, so a hand that stops moving (or a widget that dies
    mid-gesture) frees the board within about two frames instead of
    starving it on a bracket that never closed.
    """

    _MINIMUM_HOLD_SECONDS = 0.04
    _MAXIMUM_HOLD_SECONDS = 0.40

    def __init__(self) -> None:
        self._lock = Lock()
        self._held: dict[str, tuple[float, float]] = {}
        self._running: dict[str, int] = {}

    def grip(self, host_key: str) -> None:
        """Pointer work is RUNNING here: hold until it is not.

        An estimate cannot cover the move it is estimating.  Sized from
        the previous move, the hold lapsed under a move that ran longer
        than its predecessor -- and that is exactly when a sibling frame
        must not start, because a slow move means a loaded machine.
        Measured, one 286 ms frame slipped through the middle of a
        153 ms move and cost the drag a third of a second.
        """

        with self._lock:
            self._running[host_key] = self._running.get(host_key, 0) + 1

    def ungrip(self, host_key: str, cost_seconds: float) -> None:
        """Pointer work finished: fall back to a hold sized by its cost."""

        with self._lock:
            remaining = self._running.get(host_key, 0) - 1
            if remaining > 0:
                self._running[host_key] = remaining
            else:
                self._running.pop(host_key, None)
        self.touch(host_key, cost_seconds)

    def touch(self, host_key: str, cost_seconds: float | None = None) -> None:
        """Note pointer work, and size the hold from what it cost."""

        with self._lock:
            _expires, hold = self._held.get(host_key, (0.0, self._MINIMUM_HOLD_SECONDS))
            if cost_seconds is not None:
                hold = min(
                    max(2.0 * float(cost_seconds), self._MINIMUM_HOLD_SECONDS),
                    self._MAXIMUM_HOLD_SECONDS,
                )
            self._held[host_key] = (monotonic() + hold, hold)

    def busy(self) -> bool:
        """Is a hand moving right now, on any surface -- including this one?

        A data frame is speculative: the next publication supersedes it, and
        nobody is waiting for this one in particular.  A hand is not.  The
        question used to exclude the asking host, so a panel's own frames
        never stood aside for the hand ON that panel -- and a move that
        arrived while one was running waited out its whole render.  Measured
        on a live console, that is most of the 93 ms a move costs there.
        """

        now = monotonic()
        with self._lock:
            if self._running:
                return True
            for key, (expires, _hold) in tuple(self._held.items()):
                if expires <= now:
                    del self._held[key]
                else:
                    return True
        return False

    def forget(self, host_key: str) -> None:
        with self._lock:
            self._held.pop(host_key, None)
            self._running.pop(host_key, None)


#: One machine, one arbiter.
_HANDS = _HandArbiter()

#: How long a yielding worker sleeps before re-asking.  A hand moves at
#: 60-125 Hz, so this is short enough to be invisible and long enough
#: never to spin.
_HAND_YIELD_POLL_SECONDS = 0.008


def _is_a_hand(action: str, held: bool) -> bool:
    """Whether this pointer event is a hand the cameras must stand aside for.

    The arbiter's bargain is that the hand's own pixels replace the camera's:
    a drag repaints the thing being dragged on every move, so trading the
    camera's frames for the gesture's is a trade the operator wanted.  A bare
    HOVER makes no pixels at all -- it resolves a candidate and returns
    without publishing -- so the same bargain gives up the camera and buys
    nothing.

    Measured on a live console at real density, one image panel, four panels
    in the process:

        still, selectors on           9.22 fps
        DRAGGING at 100 Hz           85.12 fps   (the gesture's own frames)
        bare HOVER at 100 Hz          0.11 fps   -- and every other panel too

    The mechanism is arithmetic: every raw move arms a hold of at least
    _MINIMUM_HOLD_SECONDS, and a pointer at 60-125 Hz renews it three to five
    times faster than it can expire, so ``busy()`` never goes false and every
    publishing task in the process stands aside for as long as the pointer is
    moving.  Measured thresholds: 29 moves a second still froze it (34 ms
    apart, under the 40 ms floor); 9 a second did not (107 ms apart).

    So the hand is every pointer event EXCEPT a move with no button PHYSICALLY
    down, and the leave that follows it.  A scroll carries no button and is
    still a hand: the wheel repaints, and the operator is waiting for it.

    HELD is the button mask the window system delivered, not the widget's own
    ``_pointer_button``.  That field is cleared the moment a press resolves no
    candidate and no role (backends.py, _finish_pointer), which is exactly
    what an area rubber-band press does -- so every move of the commonest
    selector gesture would have arrived looking like a hover.  Caught by
    test_a_drag_stays_a_hand_from_press_to_release, which read
    press(1), move(None) x5, release(1) off a real widget.
    """

    return not (action in {"move", "leave"} and not held)


@dataclass(slots=True)
class _WorkerTask:
    callback: Callable[[], Any]
    completion: Future[RasterOperation[Any]]
    mode: _DispatchMode
    coalesce_key: object | None
    after_publish: Callable[[], None] | None
    on_abort: Callable[[], None] | None
    #: Pointer work: this task IS the operator's hand, and every other
    #: host's speculative frame yields to it while it moves.
    hand: bool = False


class RasterPlotHost:
    """Serialize one ``PlotSession`` and publish immutable latest fronts.

    The callable factory is invoked on the owned worker, and closing the host
    always closes that session.  A live Figure is never exposed to callers.
    :meth:`from_plot` is the standard immutable-data/spec construction path;
    the raw factory remains available for deliberate ``PlotSession`` subclasses.
    Public mutation methods never run plot work on the caller.  One data/fit
    pair may be active and one complete latest input may wait; superseded
    middle inputs are cancelled without becoming solver failures.  An active
    pair exceeding its deadline reports an invalid fit without waiting for a
    successor to arrive.  Only
    high-frequency controls with the same semantic key coalesce.  Selector,
    fit, unit, size and other distinct commands retain their ordering.
    """

    ACTIVE_FIT_TIMEOUT_SECONDS = 1.0

    def __init__(
        self,
        session_factory: Callable[[], "PlotSession"],
        *,
        close_session: bool = True,
        host_id: str | None = None,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if not isinstance(close_session, bool):
            raise TypeError("close_session must be bool")
        self._session_factory: Callable[[], PlotSession] | None = session_factory
        self._close_session = close_session
        self._session: PlotSession | None = None
        self._session_defaults: PlotLibraryDefaults | None = None
        self._release_session_host: Callable[[], None] | None = None
        self._surface_release: Callable[[], None] | None = None
        self._active_mode: _DispatchMode | None = None
        self._condition = Condition(Lock())
        self._pending: deque[_WorkerTask] = deque()
        #: One frame travels prepare -> solve -> commit.  While it travels,
        #: direct callers retain only the latest complete successor.  The
        #: Board normally applies this admission before the Host; retaining
        #: the same capacity here avoids a second public-Host queue policy.
        self._frame_active: _DataFrame | None = None
        self._frame_latest: _DataFrame | None = None
        #: Wheel ticks accumulated across coalesced scroll tasks.  The one
        #: surviving scroll task drains the whole sum, so a fast wheel burst
        #: renders once with the combined magnitude instead of queueing one
        #: full render per tick.
        self._scroll_steps = 0.0
        self._closing = False
        self._closed = False
        self._ready = False
        self._startup_error: Exception | None = None
        self._host_id = uuid4().hex if host_id is None else text(host_id, "host_id")
        self._sequence = 0
        self._front: RasterFront | None = None
        self._initial_metadata: tuple[object, object] | None = None
        #: The configure target still queued, so a call that coalesces on
        #: top of it carries its fields forward.  See ``configure``.
        self._queued_configuration: dict[str, object] | None = None
        self._initial_error: BaseException | None = None
        #: Built on demand by :meth:`qt_widget`; a headless host never has one.
        self._qt_widget = None
        #: Whether dragging on this plot is allowed.  Held here rather than
        #: only on the widget, because the decision can be made before the
        #: widget exists and must survive until it does.
        self._interaction_enabled = True
        self._front_callbacks: list[Callable[[RasterFront], None]] = []
        self._last_front_callback_error: Exception | None = None
        self._thread = Thread(
            target=self._run,
            name="zlc-raster-plot",
            daemon=True,
        )
        self._thread.start()
        self._initial_front = self._submit(
            lambda: None,
            mode=_DispatchMode.PUBLISH,
            coalesce_key="initial-front",
        )
        initial_metadata = self._submit(
            lambda: (
                self._require_session().describe_display(),
                self._require_session().fit_models,
            ),
            mode=_DispatchMode.CONTROL,
            coalesce_key="initial-metadata",
        )
        initial_metadata.add_done_callback(self._capture_initial_operation)

    @classmethod
    def from_plot(
        cls,
        data: "PlotInput",
        spec: "PlotSpec",
        *,
        size: str | None = None,
        parameters: Mapping[str, object] | None = None,
        defaults: PlotLibraryDefaults = DEFAULTS,
        unit_registry: UnitRegistry | None = None,
        device_pixel_ratio: float = 1.0,
        fit_engine: "FitEngine | None" = None,
    ) -> "RasterPlotHost":
        """Create the session on this host's worker from immutable plot inputs."""

        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        initial_parameters = None if parameters is None else dict(parameters)

        def create_session() -> "PlotSession":
            from .session import PlotSession

            return PlotSession(
                data,
                spec,
                size=size,
                parameters=initial_parameters,
                defaults=defaults,
                unit_registry=unit_registry,
                device_pixel_ratio=device_pixel_ratio,
                fit_engine=fit_engine,
            )

        return cls(create_session)

    @classmethod
    def from_session(
        cls,
        session: "PlotSession",
        *,
        close_session: bool = True,
    ) -> "RasterPlotHost":
        """Move one already-created session behind the raster worker boundary."""

        from .session import PlotSession

        if not isinstance(session, PlotSession):
            raise TypeError("session must be PlotSession")
        return cls(lambda: session, close_session=close_session)

    @property
    def startup_failure(self) -> Exception | None:
        """Why this host never started, or None while it lives (or starts).

        A host that failed startup is PERMANENTLY unusable: every later
        call raises the same reason.  Its owner needs that as a readable
        fact -- a dead surface is retired and REMADE from repaired state,
        never reconfigured -- instead of learning it by catching the same
        exception on every call forever.
        """

        with self._condition:
            return self._startup_error

    def _unusable(self) -> RuntimeError:
        """Why this host will not take work -- the REASON, not the symptom.

        A startup failure sets ``_closing`` and records itself in
        ``_startup_error``, and the refusals that read only ``_closing`` said
        "raster plot host is closing" to every later caller.  So a panel asked
        to draw something the plot kind cannot accept reported a host that was
        shutting down, while the actual sentence -- "CurvePlot requires
        zlc_data.OwnedSnapshot" -- sat in a field nobody read.  Every refusal
        comes through here, so there is one answer to "why".

        Call with ``self._condition`` held.
        """

        if self._startup_error is not None:
            failure = RuntimeError(
                f"raster plot host failed to start: {self._startup_error}"
            )
            failure.__cause__ = self._startup_error
            return failure
        return RuntimeError("raster plot host is closing")

    def _require_session(self) -> "PlotSession":
        with self._condition:
            while (
                not self._ready
                and self._startup_error is None
                and not self._closed
            ):
                self._condition.wait()
            if self._startup_error is not None:
                raise RuntimeError("raster plot session initialization failed") from self._startup_error
            if self._session is None:
                raise RuntimeError("raster plot host closed before session initialization")
            return self._session

    @property
    def closing(self) -> bool:
        """Whether this host can still serve, answered without raising.

        Everything else that reports it -- ``defaults``, ``_require_session``,
        ``_dispatch_session`` -- reports it by RAISING, which is the right
        answer to a caller that asked for work and a fatal one to a Qt event
        handler: PyQt aborts the process on an exception that escapes a slot,
        with no traceback.  A widget outliving its host is ordinary (the
        console retires a host when a panel retargets), so the widget needs a
        question it can ask, not an exception it must catch.
        """

        with self._condition:
            return bool(self._closing or self._closed)

    @property
    def front(self) -> RasterFront | None:
        with self._condition:
            return self._front

    def _capture_initial_operation(
        self,
        completed: Future[RasterOperation[object]],
    ) -> None:
        """Cache the initial worker result so the GUI owner never resolves it."""

        try:
            operation = completed.result()
            metadata = operation.value
            if not isinstance(metadata, tuple) or len(metadata) != 2:
                raise TypeError("initial raster metadata must be a pair")
        except BaseException as error:
            with self._condition:
                self._initial_error = error
        else:
            with self._condition:
                self._initial_metadata = metadata

    @property
    def initial_state(
        self,
    ) -> tuple[tuple[object, object] | None, BaseException | None]:
        """Already-completed metadata/error for non-blocking owner projection."""

        with self._condition:
            return self._initial_metadata, self._initial_error

    @property
    def host_id(self) -> str:
        """Stable opaque identity stamped into every front from this host."""

        return self._host_id

    @property
    def defaults(self) -> PlotLibraryDefaults:
        """The immutable configuration owned by the worker-side session."""

        with self._condition:
            while (
                self._session_defaults is None
                and self._startup_error is None
                and not self._closed
            ):
                self._condition.wait()
            if self._startup_error is not None:
                raise RuntimeError(
                    "raster plot session initialization failed"
                ) from self._startup_error
            if self._session_defaults is None:
                raise RuntimeError("raster plot host is closed")
            return self._session_defaults

    @property
    def logical_size(self) -> tuple[int, int] | None:
        """The size the latest front was laid out at, or None before the first.

        A host that has drawn knows how big its picture is, and whoever mounts
        the widget needs exactly that.  It used to be read off a front the
        caller had to fetch and carry, which is how a preview came to be sized
        once at mount and never again: the number was in a variable someone
        held, not a question the host could be asked.
        """

        with self._condition:
            front = self._front
        return None if front is None else tuple(front.logical_size)

    def qt_widget(self):
        """The Qt widget that shows this host, made once and kept.

        The widget is built HERE because it is this package's widget: a host
        handed across a boundary should not oblige the receiver to know which
        class draws it, and a composition root that constructs Qt widgets is a
        composition root assembling a UI.  Made lazily, so a host used
        headlessly never touches Qt at all.
        """

        widget = self._qt_widget
        if widget is None:
            from .backends import Qt5PlotWidget

            widget = Qt5PlotWidget(self)
            self._qt_widget = widget
            # Whatever was decided before there was a widget to decide it for.
            if not self._interaction_enabled:
                widget.set_interaction_enabled(False)
        return widget

    def set_interaction_enabled(self, enabled: bool) -> None:
        """Allow or suspend dragging on this plot -- selectors, zoom, pan.

        The one control a host outside this package needs over interaction, and
        it belongs on the host rather than on the widget: whoever holds the
        host is deliberately not holding a widget, so without this the answer
        to "let the operator drag on it" was unreachable and the console's
        Selectors switch ended up greying out its own dropdowns instead.

        A host that has not been shown has nothing to gate: interaction is a
        property of the widget, made when the widget is, so this is remembered
        and applied to the widget the moment there is one.
        """

        self._interaction_enabled = bool(enabled)
        widget = self._qt_widget
        if widget is not None:
            widget.set_interaction_enabled(bool(enabled))

    @property
    def interaction_enabled(self) -> bool:
        """Whether dragging on this plot is allowed."""

        return self._interaction_enabled

    def wait_for_front(self, timeout: float | None = None) -> RasterFront:
        """Wait for the first complete raster before exposing a new window."""

        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must be non-negative or None")
        with self._condition:
            front = self._front
            initial = self._initial_front
            closing = self._closing
        if front is not None:
            return front
        if closing:
            with self._condition:
                raise self._unusable()
        if initial is None:
            raise RuntimeError("raster plot host closed before its first front")
        return initial.result(timeout=timeout).front

    def subscribe_front(
        self,
        callback: Callable[[RasterFront], None],
    ) -> Callable[[], None]:
        if not callable(callback):
            raise TypeError("front callback must be callable")
        with self._condition:
            if self._closing:
                raise self._unusable()
            self._front_callbacks.append(callback)

        def unsubscribe() -> None:
            with self._condition:
                if callback in self._front_callbacks:
                    self._front_callbacks.remove(callback)

        return unsubscribe

    def _subscribe_session_event(
        self,
        callback: Callable[..., object],
        install: Callable[
            ["PlotSession", Callable[..., object]],
            Callable[[], None],
        ],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        if not callable(callback):
            raise TypeError("event callback must be callable")
        if not callable(install):
            raise TypeError("event subscription installer must be callable")

        def subscribe() -> Callable[[], Future[RasterOperation[None]]]:
            release = install(self._require_session(), callback)
            if not callable(release):
                raise TypeError("session subscription must return an unsubscribe callback")

            def unsubscribe() -> Future[RasterOperation[None]]:
                return self._submit(release, mode=_DispatchMode.CONTROL)

            return unsubscribe

        return self._submit(subscribe, mode=_DispatchMode.CONTROL)

    def subscribe_display(
        self,
        callback: Callable[["DisplayState"], object],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a display callback on the raster worker."""

        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_display(listener),
        )

    def subscribe_viewport(
        self,
        callback: Callable[[object], object],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a committed viewport callback on the raster worker."""

        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_viewport(listener),
        )

    def subscribe_facet_focus(
        self,
        callback: Callable[[int | None, object, str | None, int], object],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a committed FacetGrid focus callback on the raster worker."""

        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_facet_focus(listener),
        )

    def subscribe_fit(
        self,
        callback: Callable[["FitEvent"], object],
        *,
        replay_current: bool = False,
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a fit callback on the raster worker."""

        if type(replay_current) is not bool:
            raise TypeError("replay_current must be bool")
        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_fit(
                listener,
                replay_current=replay_current,
            ),
        )

    def subscribe_selection(
        self,
        callback: Callable[["SelectionEvent"], object],
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a selection callback on the raster worker."""

        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_selection(listener),
        )

    def _dispatch_callback(
        self,
        callback: Callable[[], ValueT],
        mode: _DispatchMode,
        *,
        after_publish: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> Future[RasterOperation[ValueT]]:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if mode is _DispatchMode.PRESENTATION:
            if not callable(after_publish) or not callable(on_abort):
                raise TypeError("presentation requires finalization and abort callbacks")
        elif after_publish is not None or on_abort is not None:
            raise TypeError("finalization/abort require presentation dispatch")
        if current_thread() is not self._thread:
            return self._submit(
                callback,
                mode=mode,
                after_publish=after_publish,
                on_abort=on_abort,
            )
        completion: Future[RasterOperation[ValueT]] = Future()
        if not completion.set_running_or_notify_cancel():
            return completion
        try:
            operation = self._execute_worker_task(
                callback,
                mode,
                after_publish=after_publish,
                on_abort=on_abort,
            )
        except Exception as error:
            completion.set_exception(error)
        else:
            completion.set_result(operation)
        return completion

    def _execute_worker_task(
        self,
        callback: Callable[[], ValueT],
        mode: _DispatchMode,
        *,
        after_publish: Callable[[], None] | None,
        on_abort: Callable[[], None] | None,
    ) -> RasterOperation[ValueT]:
        """Run the only callback-to-front presentation state machine."""

        promoted = False
        self._active_mode = mode
        try:
            session = self._require_session()
            presentation_epoch = session._raster_presentation_epoch()
            value = callback()
            publishes = mode.publishes or (
                mode is _DispatchMode.ADAPTIVE
                and (
                    bool(getattr(value, "publish_front", False))
                    or session._raster_presentation_epoch()
                    != presentation_epoch
                )
            )
            front = self._capture_front() if publishes else self.front
            if front is None:
                front = self._capture_front()
            if publishes:
                if not self._promote(front):
                    raise RuntimeError("raster host closed before front promotion")
                promoted = True
                # The surface callback is intentionally lossless while a
                # CONTROL task is running.  If this task itself captured the
                # committed surface, remove the now-redundant coalesced edge;
                # otherwise every ordinary PUBLISH setter would create a
                # second indistinguishable front.
                self._discard_surface_sync_tasks()
            if mode is _DispatchMode.PRESENTATION:
                assert after_publish is not None
                after_publish()
            return RasterOperation(value, front)
        except Exception as error:
            if mode is _DispatchMode.PRESENTATION and not promoted:
                assert on_abort is not None
                try:
                    on_abort()
                except Exception:
                    self._require_session().redraw_surface()
            raise
        finally:
            self._active_mode = None

    def _discard_surface_sync_tasks(self) -> None:
        cancelled: list[Future[RasterOperation[Any]]] = []
        with self._condition:
            if not self._pending:
                return
            retained: deque[_WorkerTask] = deque()
            for task in self._pending:
                if task.coalesce_key == "surface-sync":
                    cancelled.append(task.completion)
                else:
                    retained.append(task)
            if len(retained) == len(self._pending):
                return
            self._pending = retained
            self._condition.notify_all()
        for completion in cancelled:
            completion.cancel()

    def _on_session_surface(self) -> None:
        """Queue a front promotion for every committed session surface.

        A surface callback carries no reliable information about which
        dispatch initiated the commit.  Dropping it while another worker task
        is active therefore loses the only edge that can republish a CONTROL
        mutation (and leaves the front stale).  A coalesced PUBLISH task is
        cheap when the current task already promoted a matching frame, and it
        is essential when a control/analysis task committed the surface while
        the worker was busy.
        """

        self._submit(
            lambda: None,
            mode=_DispatchMode.PUBLISH,
            coalesce_key="surface-sync",
        )

    def _dispatch_session(
        self,
        operation: Callable[..., Any],
        *args: Any,
        _mode: _DispatchMode,
        coalesce_key: object | None = None,
        _hand: bool = False,
        **kwargs: Any,
    ) -> Future[RasterOperation[Any]]:
        callback = lambda: operation(*args, **kwargs)
        return self._submit(
            callback,
            mode=_mode,
            coalesce_key=coalesce_key,
            hand=_hand,
        )

    def dispatch(self, callback: Callable[[], None]) -> Future[RasterOperation[None]]:
        """Marshal a session completion onto the serial render worker."""

        return self._dispatch_callback(callback, _DispatchMode.PUBLISH)

    def dispatch_control(
        self,
        callback: Callable[[], None],
    ) -> Future[RasterOperation[None]]:
        """Run owner-affine control work without publishing a raster front."""

        return self._dispatch_callback(callback, _DispatchMode.CONTROL)

    def dispatch_presentation(
        self,
        callback: Callable[[], None],
        after_publish: Callable[[], None],
        on_abort: Callable[[], None],
    ) -> Future[RasterOperation[None]]:
        """Queue one non-reentrant publish/finalize presentation transaction."""

        return self._submit(
            callback,
            mode=_DispatchMode.PRESENTATION,
            after_publish=after_publish,
            on_abort=on_abort,
        )

    def update_data(
        self,
        data: object,
        *,
        revision: int | None = None,
    ) -> Future[RasterOperation["DisplayDescription"]]:
        """Present one data frame as a complete pair through the pipeline.

        The frame travels prepare (projection, off-worker) -> solve (armed
        live fit, analysis executor) -> commit (paint data + overlay + capture,
        one short worker item).  The returned future resolves when the
        committed front is promoted.  While one pair is active, only the
        latest complete successor is retained.  An active deadline reports
        one loud invalid fit; cadence-superseded successors are simply
        cancelled.  A failed solve reports only its own revision so a later
        matching pair can recover the panel.
        """

        return self._enqueue_data_frame(data, revision)

    def update_image_overlay(
        self,
        overlay: "ImagePointOverlay",
    ) -> Future[RasterOperation["ImagePointOverlay"]]:
        """Coalesce independent Image point revisions on the render worker."""

        return self._dispatch_session(
            lambda: self._require_session().update_image_overlay(overlay),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="image-overlay",
        )

    def update_image_frame(
        self,
        frame: "ImageFrame",
    ) -> Future[RasterOperation["DisplayDescription"]]:
        """Present one complete image frame through the pair pipeline."""

        return self._enqueue_data_frame(frame, None)

    # ---------------------------------------------------------- pair pipeline

    def _enqueue_data_frame(
        self,
        data: object,
        revision: int | None,
    ) -> Future[RasterOperation["DisplayDescription"]]:
        if revision is None:
            revision = _plot_input_revision(data)
        completion: Future[RasterOperation["DisplayDescription"]] = Future()
        frame = _DataFrame(
            data,
            revision,
            completion,
        )
        refused: BaseException | None = None
        start: _DataFrame | None = None
        superseded: _DataFrame | None = None
        with self._condition:
            if self._closing:
                refused = self._unusable()
            elif self._frame_active is not None:
                superseded, self._frame_latest = self._frame_latest, frame
            else:
                self._frame_active = frame
                start = frame
            self._condition.notify_all()
        if refused is not None:
            completion.set_exception(refused)
            return completion
        if superseded is not None:
            superseded.cancel.set()
            superseded.completion.cancel()
        if start is not None:
            self._begin_data_frame(start)
        return completion

    def _expire_data_frame(
        self,
        frame: _DataFrame,
        elapsed: float,
    ) -> None:
        """Publish one loud invalid result, then release the held latest input."""

        error = RuntimeError(
            "live fit active deadline exceeded: "
            f"revision={frame.revision}, elapsed={elapsed:.3f}s, "
            f"limit={self.ACTIVE_FIT_TIMEOUT_SECONDS:.3f}s"
        )
        try:
            # Rotation is lock-protected PlotSession control state.  It makes
            # every late callback from this analysis generation harmless.
            self._require_session().expire_active_live_fit()
        except BaseException as cancel_error:
            error.add_note(f"fit cancellation failed: {cancel_error}")

        def publish() -> None:
            if frame.revision is not None:
                self._require_session().publish_live_fit_gap(
                    frame.revision,
                    error,
                )

        dispatched = self._submit(publish, mode=_DispatchMode.CONTROL)

        def published(done: Future[RasterOperation[None]]) -> None:
            try:
                done.result()
            except BaseException as gap_error:
                error.add_note(f"fit gap publication failed: {gap_error}")
            if not frame.completion.done():
                try:
                    frame.completion.set_exception(error)
                except InvalidStateError:
                    pass
            with self._condition:
                current = self._frame_active is frame
                if current:
                    self._frame_active = None
                self._condition.notify_all()
            if current:
                self._advance_data_frame()

        dispatched.add_done_callback(published)

    def _dispatch_failed_frame_gap(
        self,
        prepared: object,
        frame: _DataFrame,
        error: BaseException,
    ) -> None:
        projection = getattr(prepared, "projection", None)
        revision = getattr(projection, "data_revision", None)

        def publish() -> None:
            if revision is not None:
                self._require_session().publish_live_fit_gap(
                    int(revision),
                    error,
                    projection=projection,
                )

        dispatched = self._submit(publish, mode=_DispatchMode.CONTROL)

        def published(done: Future[RasterOperation[None]]) -> None:
            try:
                done.result()
            except BaseException as gap_error:
                error.add_note(f"fit gap publication failed: {gap_error}")
            self._finish_data_frame(frame, error=error)

        dispatched.add_done_callback(published)

    def _begin_data_frame(
        self,
        frame: _DataFrame,
    ) -> None:
        with self._condition:
            if frame.stage != "timed_out":
                frame.stage = "prepare"

        def stage_prepare() -> Future[object]:
            session = self._require_session()
            # Data the session already holds is nothing to do, not a fault.
            # ``prepare_live_frame`` refuses it -- rightly, for a caller
            # asking to advance -- but this caller is a pipeline that can
            # legitimately be handed the same revision twice: the submitter
            # records what it handed over only once the render COMPLETES,
            # and during a gesture the arbiter supersedes renders on
            # purpose, so a committed frame whose future was cancelled
            # looked free and went round again.  The operator saw "data
            # revision must increase" on the card while turning a scene.
            if frame.revision is not None and session.holds_live_revision(
                frame.data, frame.revision
            ):
                raise _FrameSuperseded(
                    "the session already holds this data revision"
                )
            return session.prepare_live_frame(
                frame.data,
                revision=frame.revision,
                cancelled=frame.cancel.is_set,
            )

        dispatched = self._submit(stage_prepare, mode=_DispatchMode.CONTROL)
        dispatched.add_done_callback(
            lambda done: self._on_frame_prepare_submitted(done, frame)
        )

    def _on_frame_prepare_submitted(
        self,
        dispatched: Future[RasterOperation[Future[object]]],
        frame: _DataFrame,
    ) -> None:
        try:
            prepare_future = dispatched.result().value
        except (_FrameSuperseded, CancelledError):
            # The same reading the commit stage already gives it: a frame
            # the session has moved past is finished, not failed.
            self._finish_data_frame(frame, cancelled=True)
            return
        except BaseException as error:
            self._finish_data_frame(frame, error=error)
            return
        prepare_future.add_done_callback(
            lambda done: self._on_frame_prepared(done, frame)
        )

    def _on_frame_prepared(
        self,
        prepare_future: Future[object],
        frame: _DataFrame,
    ) -> None:
        from .fit import FitCancelled

        try:
            prepared = prepare_future.result()
        except (FitCancelled, CancelledError):
            # The preparation was cancelled by a context change (replace_spec,
            # clear_fit, close); the producer's next revision heals the panel.
            self._finish_data_frame(frame, cancelled=True)
            return
        except BaseException as error:
            self._finish_data_frame(frame, error=error)
            return
        if frame.cancel.is_set():
            self._finish_data_frame(frame, cancelled=True)
            return
        try:
            # Analysis completion is not the raster worker.  Submission only
            # touches PlotSession's lock-protected fit state.
            solve_future = self._require_session().solve_live_frame(
                prepared,
                cancelled=frame.cancel.is_set,
            )
        except BaseException as error:
            self._dispatch_failed_frame_gap(prepared, frame, error)
            return
        if solve_future is None:
            self._dispatch_frame_commit(prepared, None, frame)
            return
        with self._condition:
            if frame.stage != "timed_out":
                frame.stage = "solve"
                frame.started_at = monotonic()
                self._condition.notify_all()
        solve_future.add_done_callback(
            lambda done: self._on_frame_solved(done, prepared, frame)
        )

    def _on_frame_solved(
        self,
        solve_future: Future[object],
        prepared: object,
        frame: _DataFrame,
    ) -> None:
        from .fit import FitCancelled

        try:
            solved = solve_future.result()
        except (CancelledError, FitCancelled):
            self._finish_data_frame(frame, cancelled=True)
            return
        except BaseException as error:
            self._dispatch_failed_frame_gap(prepared, frame, error)
            return
        if frame.cancel.is_set():
            self._finish_data_frame(frame, cancelled=True)
            return
        self._dispatch_frame_commit(prepared, solved, frame)

    def _dispatch_frame_commit(
        self,
        prepared: object,
        solved: object | None,
        frame: _DataFrame,
    ) -> None:
        accepted: list[object] = []

        def stage_commit() -> "DisplayDescription":
            if frame.cancel.is_set():
                raise _FrameSuperseded("live frame was cancelled before commit")
            session = self._require_session()
            finalization = session.commit_live_frame(
                prepared,
                solved,
            )
            if finalization is None:
                raise _FrameSuperseded(
                    "live frame superseded before presentation"
                )
            if frame.cancel.is_set():
                session.abort_live_frame(finalization)
                raise _FrameSuperseded("live frame was cancelled during commit")
            accepted.append(finalization)
            return session.describe_display()

        def published() -> None:
            if accepted:
                self._require_session().publish_live_frame(accepted[0])

        def abort() -> None:
            if accepted:
                try:
                    self._require_session().abort_live_frame(accepted[0])
                except Exception:
                    pass

        with self._condition:
            if frame.stage != "timed_out":
                frame.stage = "commit"
        dispatched = self._submit(
            stage_commit,
            mode=_DispatchMode.PRESENTATION,
            after_publish=published,
            on_abort=abort,
        )
        dispatched.add_done_callback(
            lambda done: self._on_frame_committed(done, frame)
        )

    def _on_frame_committed(
        self,
        dispatched: Future[RasterOperation["DisplayDescription"]],
        frame: _DataFrame,
    ) -> None:
        try:
            operation = dispatched.result()
        except (_FrameSuperseded, CancelledError):
            self._finish_data_frame(frame, cancelled=True)
            return
        except BaseException as error:
            self._finish_data_frame(frame, error=error)
            return
        self._finish_data_frame(frame, operation=operation)

    def _finish_data_frame(
        self,
        frame: _DataFrame,
        *,
        operation: RasterOperation["DisplayDescription"] | None = None,
        error: BaseException | None = None,
        cancelled: bool = False,
    ) -> None:
        completion = frame.completion
        with self._condition:
            closing = self._closing
            timed_out = frame.stage == "timed_out"
        # A timed-out frame is settled by the gap publisher, after its
        # invalid FitEvent has been emitted while the panel port still holds
        # the exact source publication.  The cooperative solver may observe
        # cancellation first; letting that callback cancel the Future would
        # race away the one loud deadline error.
        if not timed_out:
            try:
                if cancelled or (error is not None and closing):
                    completion.cancel()
                elif error is not None:
                    if not completion.done():
                        completion.set_exception(error)
                elif not completion.done():
                    completion.set_result(operation)
            except InvalidStateError:
                pass
        # A solver failure is loud on exactly its revision, but it cannot
        # permanently freeze the panel.  A timeout remains active until its
        # invalid event is published; that publisher alone releases latest.
        with self._condition:
            current = self._frame_active is frame
            if current and not timed_out:
                self._frame_active = None
                self._condition.notify_all()
        if current and not timed_out:
            self._advance_data_frame()

    def _advance_data_frame(self) -> None:
        cancelled: _DataFrame | None = None
        start: _DataFrame | None = None
        with self._condition:
            if self._frame_active is not None:
                return
            if self._frame_latest is None:
                return
            if self._closing:
                cancelled, self._frame_latest = self._frame_latest, None
            else:
                start, self._frame_latest = self._frame_latest, None
                self._frame_active = start
            self._condition.notify_all()
        if cancelled is not None:
            cancelled.cancel.set()
            cancelled.completion.cancel()
        if start is not None:
            self._begin_data_frame(start)

    def set_parameter(
        self,
        name: str,
        value: object,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().set_parameter(name, value),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key=("parameter", str(name)),
        )

    def set_parameters(
        self,
        values: Mapping[str, object],
    ) -> Future[RasterOperation["DisplayState"]]:
        prepared = dict(values)
        return self._dispatch_session(
            lambda: self._require_session().set_parameters(prepared),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key=("parameters", tuple(sorted(prepared))),
        )

    def configure(
        self,
        *,
        data: "PlotInput | object" = _UNSET,
        semantic: Mapping[str, object] | None = None,
        parameters: Mapping[str, object] | None = None,
        parameter_updates: Mapping[str, object] | None = None,
        size: str | None = None,
        image_overlay: "ImagePointOverlay | None | object" = _UNSET,
        classifier_thresholds: object = _UNSET,
        selectors: Sequence[SelectorState] | object = _UNSET,
        viewport: RectangleRange | None | object = _UNSET,
        facet_focus: int | None | object = _UNSET,
        fit: Mapping[str, object] | None | object = _UNSET,
        fit_live: bool | object = _UNSET,
    ) -> Future[RasterOperation["DisplayDescription"]]:
        """Submit a desired plot target as one raster operation.

        ``parameter_updates`` identifies the fields authored in this
        transaction while ``parameters`` remains the coalescing-safe complete
        target. Plot, not the embedder, resolves transition-generated values.

        A CALL NAMES THE FIELDS IT MEANS, and one that coalesces on top of a
        queued call carries that call's other fields forward.  Every configure
        shares one coalesce key, so a queued target is REPLACED by the next --
        and the console sends two inside a single busy worker window: the
        Setting form's semantic/parameters/fit, and a sibling gesture's
        viewport or facet_focus.  Whichever arrived first had its fields
        silently dropped, with nothing told: an edit in Setting simply did
        not take, or a mirrored viewport did not follow.
        """

        configuration: dict[str, object] = {}
        # None is "not named" for these four -- they have no clearing value --
        # so a viewport-only call no longer says "and no semantic".
        if semantic is not None:
            configuration["semantic"] = dict(semantic)
        if parameters is not None:
            configuration["parameters"] = dict(parameters)
        if parameter_updates is not None:
            configuration["parameter_updates"] = dict(parameter_updates)
        if size is not None:
            configuration["size"] = size
        if data is not _UNSET:
            configuration["data"] = data
        if image_overlay is not _UNSET:
            configuration["image_overlay"] = image_overlay
        if classifier_thresholds is not _UNSET:
            configuration["classifier_thresholds"] = tuple(classifier_thresholds)
        if selectors is not _UNSET:
            configuration["selectors"] = tuple(selectors)
        if viewport is not _UNSET:
            configuration["viewport"] = viewport
        if facet_focus is not _UNSET:
            configuration["facet_focus"] = facet_focus
        if fit is not _UNSET:
            configuration["fit"] = None if fit is None else dict(fit)
        if fit_live is not _UNSET:
            configuration["fit_live"] = fit_live

        with self._condition:
            queued = self._queued_configuration
            if queued is not None:
                merged = dict(queued)
                merged.update(configuration)
                configuration = merged
            self._queued_configuration = configuration

        def forget() -> None:
            with self._condition:
                if self._queued_configuration is configuration:
                    self._queued_configuration = None

        pending = self._dispatch_session(
            lambda: (forget(), self._require_session().configure(**configuration))[1],
            _mode=_DispatchMode.ADAPTIVE,
            coalesce_key="configuration",
        )

        def remember(completed: Future[RasterOperation[object]]) -> None:
            if completed.cancelled():
                return
            try:
                description = completed.result().value
                models = tuple(description.fit_models)
            except BaseException:
                return
            with self._condition:
                self._initial_metadata = (description, models)

        pending.add_done_callback(remember)
        return pending

    def describe_display(self) -> Future[RasterOperation["DisplayDescription"]]:
        """Return the worker session's immutable control-plane description."""

        return self._dispatch_session(
            lambda: self._require_session().describe_display(),
            _mode=_DispatchMode.CONTROL,
        )

    def describe_semantics(self) -> Future[RasterOperation[object]]:
        """Return the registry-derived semantic edit domain."""

        return self._dispatch_session(
            lambda: self._require_session().describe_semantics(),
            _mode=_DispatchMode.CONTROL,
        )

    def replace_spec(
        self,
        spec: "PlotSpec",
        *,
        parameters: Mapping[str, object] | None = None,
    ) -> Future[RasterOperation["DisplayDescription"]]:
        """Replace semantic roles through the same worker transaction as Qt."""

        return self._dispatch_session(
            lambda: self._require_session().replace_spec(
                spec,
                parameters=parameters,
            ),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="spec",
        )

    def apply_semantic(
        self,
        name: str,
        value: object,
    ) -> Future[RasterOperation["DisplayDescription"]]:
        """Apply one semantic edit, composed against what is currently drawn.

        The caller supplies only what the operator changed.  Composing the
        candidate needs the current spec and the data schema, both of which the
        session holds -- so asking a caller for them is asking it to keep a copy
        of state it does not own.
        """

        return self._dispatch_session(
            lambda: self._require_session().apply_semantic(str(name), value),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="spec",
        )

    def resolved_color_limits(
        self,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[NumericRange]]:
        """Return the effective clim painted by the worker session."""

        return self._dispatch_session(
            lambda: self._require_session().resolved_color_limits(display=display),
            _mode=_DispatchMode.CONTROL,
        )

    def set_labels(
        self,
        *,
        title: str | None | object = _UNSET,
        x: str | None | object = _UNSET,
        y: str | None | object = _UNSET,
        value: str | None | object = _UNSET,
    ) -> Future[RasterOperation["DisplayState"]]:
        """Update title/axis/value text through the serial render worker."""

        updates: dict[str, object] = {}
        for name, selected in (
            ("title", title),
            ("x", x),
            ("y", y),
            ("value", value),
        ):
            if selected is not _UNSET:
                updates[name] = selected
        return self._dispatch_session(
            lambda: self._require_session().set_labels(**updates),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key=("labels", tuple(sorted(updates))),
        )

    def set_relim_mode(self, mode: str) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().set_relim_mode(mode),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="relim-mode",
        )

    def set_y_limits(
        self,
        low: float,
        high: float,
        *,
        fixed: bool = True,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().set_y_limits(low, high, fixed=fixed),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="y-limits",
        )

    def reset_y_limits(
        self,
        *,
        mode: str = "normal",
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().reset_y_limits(mode=mode),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="y-limits",
        )

    def set_color_limits(
        self,
        low: float,
        high: float,
        *,
        fixed: bool = True,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().set_color_limits(
                low,
                high,
                fixed=fixed,
            ),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="color-limits",
        )

    def reset_color_limits(
        self,
        *,
        mode: str = "tight",
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().reset_color_limits(mode=mode),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="color-limits",
        )

    def set_x_limits(
        self,
        low: float,
        high: float,
    ) -> Future[RasterOperation[RectangleRange]]:
        return self._dispatch_session(
            lambda: self._require_session().set_x_limits(low, high),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="viewport",
        )

    def set_view_limits(
        self,
        *,
        x: tuple[float, float] | NumericRange | None = None,
        y: tuple[float, float] | NumericRange | None = None,
    ) -> Future[RasterOperation[RectangleRange]]:
        return self._dispatch_session(
            lambda: self._require_session().set_view_limits(x=x, y=y),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="viewport",
        )

    def set_size(self, preset: str) -> Future[RasterOperation["SurfacePlan"]]:
        return self._dispatch_session(
            lambda: self._require_session().set_size(preset),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="size",
        )

    def set_device_pixel_ratio(self, ratio: float) -> Future[RasterOperation["SurfacePlan"]]:
        selected = _finite(ratio, "device pixel ratio")
        if selected <= 0.0:
            raise ValueError("device pixel ratio must be positive")
        return self._dispatch_session(
            lambda: self._require_session().set_device_pixel_ratio(selected),
            # The Qt observer asks once when its widget is constructed even
            # when composition already built the Host at that exact DPR.
            # A real cross-screen change advances the session presentation
            # epoch and ADAPTIVE captures it; an equal ratio must reuse the
            # existing front instead of publishing identical pixels.
            _mode=_DispatchMode.ADAPTIVE,
            coalesce_key="device-pixel-ratio",
        )

    def set_axis_unit(
        self,
        axis: object,
        unit: str | None,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().set_axis_unit(axis, unit),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key=("axis-unit", axis),
        )

    def set_value_unit(self, unit: str | None) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().set_value_unit(unit),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="value-unit",
        )

    def set_time_unit(self, unit: str | None) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            lambda: self._require_session().set_time_unit(unit),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="time-unit",
        )

    def save(
        self,
        path: str | Path,
        *,
        dpi: float | None = None,
        export_scale: float | None = None,
        **kwargs: Any,
    ) -> Future[RasterOperation[None]]:
        options = dict(kwargs)
        return self._dispatch_session(
            lambda: self._require_session().save(
                path,
                dpi=dpi,
                export_scale=export_scale,
                **options,
            ),
            _mode=_DispatchMode.CONTROL,
        )

    def clear_fit(self) -> Future[RasterOperation[None]]:
        return self._dispatch_session(
            lambda: self._require_session().clear_fit(),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="fit-control",
        )

    def fit_models(self) -> Future[RasterOperation[tuple["FitModelSpec", ...]]]:
        """Return the fit catalogue owned by this host's session."""

        return self._dispatch_session(
            lambda: self._require_session().fit_models,
            _mode=_DispatchMode.CONTROL,
        )

    def selectors(self) -> Future[RasterOperation[tuple[SelectorState, ...]]]:
        """Return the current canonical selector geometry without slicing data."""

        return self._dispatch_session(
            lambda: self._require_session().selectors,
            _mode=_DispatchMode.CONTROL,
        )

    def selector_state(
        self,
        kind: SelectorKind,
        *,
        display: bool = False,
    ) -> Future[RasterOperation[SelectorState]]:
        """Return one selector in canonical or current display coordinates."""

        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(
            lambda: self._require_session().selector_state(kind, display=display),
            _mode=_DispatchMode.CONTROL,
        )

    def selector_data(self, kind: SelectorKind) -> Future[RasterOperation["SelectorData"]]:
        """Materialize selector output only for this explicit application call."""

        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(
            lambda: self._require_session().selector_data(kind),
            _mode=_DispatchMode.CONTROL,
        )

    def remove_selector(
        self,
        kind: SelectorKind,
        *,
        emit_change: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(
            lambda: self._require_session().remove_selector(
                kind,
                emit_change=emit_change,
            ),
            _mode=_DispatchMode.PUBLISH,
        )

    def set_selector_value(
        self,
        kind: SelectorKind,
        value: object,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(
            lambda: self._require_session().set_selector_value(
                kind,
                value,
                display=display,
            ),
            _mode=_DispatchMode.PUBLISH,
        )

    def set_area_selector(
        self,
        x: NumericRange,
        y: NumericRange,
        *,
        display: bool = True,
        emit_change: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            lambda: self._require_session().set_area_selector(
                x,
                y,
                display=display,
                emit_change=emit_change,
            ),
            _mode=_DispatchMode.PUBLISH,
        )

    def set_x_selector(
        self,
        low: float,
        high: float,
        *,
        display: bool = True,
        emit_change: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            lambda: self._require_session().set_x_selector(
                low,
                high,
                display=display,
                emit_change=emit_change,
            ),
            _mode=_DispatchMode.PUBLISH,
        )

    def set_threshold_selector(
        self,
        value: float,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            lambda: self._require_session().set_threshold_selector(
                value,
                display=display,
            ),
            _mode=_DispatchMode.PUBLISH,
        )

    def set_crosshair_selector(
        self,
        x: float,
        y: float,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            lambda: self._require_session().set_crosshair_selector(
                x,
                y,
                display=display,
            ),
            _mode=_DispatchMode.PUBLISH,
        )

    def fit(
        self,
        model: str | "FitModelSpec",
        *,
        selector_kind: SelectorKind | None = None,
        initial: Mapping[str, float] | tuple[float, ...] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: "FitOptions | None" = None,
        live: bool = True,
        fit_all_facets: bool = False,
    ) -> Future[RasterOperation[FitResult | FacetFitBatchResult]]:
        """Fit asynchronously and complete after the accepted front is promoted."""

        if selector_kind is not None and not isinstance(selector_kind, SelectorKind):
            raise TypeError("selector_kind must be SelectorKind or None")
        completion: Future[RasterOperation[FitResult | FacetFitBatchResult]] = Future()

        def fail(error: BaseException) -> None:
            """Set a completion exception exactly once, including callbacks."""

            if completion.done():
                return
            try:
                completion.set_exception(error)
            except InvalidStateError:
                # Cancellation may race the callback between the done check
                # and the state transition; cancellation is already the
                # terminal outcome in that case.
                return

        def succeed(operation: RasterOperation[FitResult | FacetFitBatchResult]) -> None:
            if completion.done():
                return
            try:
                completion.set_result(operation)
            except InvalidStateError:
                return
        started = self._dispatch_session(
            lambda: self._require_session().fit_async(
                model,
                selector_kind=selector_kind,
                initial=initial,
                bounds=bounds,
                options=options,
                live=live,
                fit_all_facets=fit_all_facets,
            ),
            _mode=_DispatchMode.CONTROL,
        )

        def start_finished(
            value: Future[RasterOperation[Future[FitResult | FacetFitBatchResult]]],
        ) -> None:
            try:
                if completion.cancelled():
                    return
                analysis = value.result().value
                if not isinstance(analysis, Future):
                    raise TypeError("session.fit_async must return Future")
            except BaseException as error:
                fail(error)
                return

            def analysis_finished(
                result_future: Future[FitResult | FacetFitBatchResult],
            ) -> None:
                try:
                    if completion.cancelled():
                        return
                    result = result_future.result()
                    if current_thread() is not self._thread:
                        raise RuntimeError(
                            "session fit completion escaped the raster owner thread"
                        )
                    front = self.front
                    if front is None:
                        raise RuntimeError("accepted fit has no promoted raster front")
                    if result.source_revision != front.identity.data_revision:
                        raise RuntimeError(
                            "accepted fit result does not match its promoted front"
                        )
                    succeed(RasterOperation(result, front))
                except BaseException as error:
                    fail(error)

            analysis.add_done_callback(analysis_finished)

        started.add_done_callback(start_finished)
        return completion

    def pointer_event(
        self,
        action: str,
        x: float,
        y: float,
        *,
        button: int | None = None,
        double: bool = False,
        step: float = 0.0,
        key: str | None = None,
        identity: RasterIdentity | None = None,
        axes: AxisTransform | None = None,
        interaction: RasterInteractionMap | None = None,
        held: bool = False,
    ) -> Future[RasterOperation[object]]:
        """Route Qt raster input through PlotSession's interaction engine.

        ``held`` says a mouse button is physically down, which is what
        separates a drag from a hover for :func:`_is_a_hand`.
        """

        selected_action = text(action, "pointer action").lower()
        if selected_action not in {
            "press",
            "move",
            "release",
            "scroll",
            "key",
            "cancel",
            "leave",
        }:
            raise ValueError(f"unknown pointer action {action!r}")
        x_value = _finite(x, "pointer x")
        y_value = _finite(y, "pointer y")
        if identity is not None and not isinstance(identity, RasterIdentity):
            raise TypeError("pointer identity must be RasterIdentity or None")
        if axes is not None and not isinstance(axes, AxisTransform):
            raise TypeError("pointer axes must be AxisTransform or None")
        if axes is not None and identity is None:
            raise ValueError("pointer axes require their painted front identity")
        if interaction is not None and not isinstance(
            interaction, RasterInteractionMap
        ):
            raise TypeError("pointer interaction must be RasterInteractionMap or None")
        if interaction is not None and identity is None:
            raise ValueError("pointer interaction requires its painted front identity")

        def apply() -> object:
            session = self._require_session()
            effective_step = step
            if selected_action == "scroll":
                # Drain before any currency check can raise, so rejected
                # contexts cannot leak accumulated ticks into a later gesture.
                with self._condition:
                    effective_step = self._scroll_steps
                    self._scroll_steps = 0.0
                if effective_step == 0.0:
                    # A newer coalesced scroll task already drained the
                    # accumulated ticks; nothing is left to apply.
                    return session._raster_pointer_state(publish_front=False)
            effective_identity = identity
            effective_axes = axes
            effective_interaction = interaction
            if identity is not None:
                if identity.host_id != self._host_id:
                    raise RuntimeError(
                        "the painted pointer front belongs to another raster host"
                    )
                # There is NO press-time currency check beyond host
                # wiring.  The front a pointer event carries IS what the
                # operator saw: the widget swaps pixels and identity
                # atomically, and the gesture layer interprets every
                # action THROUGH the carried transform into canonical
                # coordinates -- so a press against a one-front-old
                # picture is self-consistent, whatever the session
                # committed meanwhile.  Every equality gate this branch
                # ever held (data revisions, transform membership,
                # layout revisions, painted-selector state) rejected the
                # first gesture after every commit for as long as the
                # frontend ran one front behind -- each was the same
                # disease wearing a different field, and each cost the
                # bench ~a third of its presses during live acquisition.
            return session._raster_pointer_event(
                selected_action,
                x_value,
                y_value,
                button=button,
                double=double,
                step=effective_step,
                key=key,
                axes_snapshot=effective_axes,
            )

        if selected_action == "scroll":
            with self._condition:
                self._scroll_steps += float(step)
        return self._dispatch_session(
            apply,
            _mode=_DispatchMode.ADAPTIVE,
            coalesce_key=_pointer_coalesce((selected_action,), {}),
            _hand=_is_a_hand(selected_action, bool(held)),
        )

    def set_viewport(
        self,
        x: NumericRange,
        y: NumericRange,
        *,
        emit_change: bool = True,
    ) -> Future[RasterOperation[RectangleRange]]:
        """Set visible ranges in the current display units."""

        return self._dispatch_session(
            lambda: self._require_session().set_viewport(
                x,
                y,
                emit_change=emit_change,
            ),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="viewport",
        )

    def focus_facet(
        self,
        index: int,
    ) -> Future[RasterOperation[None]]:
        """Open one cell of this FacetGrid, by the thing that identifies it.

        Its INDEX.  This used to take a pixel-geometry handle (an identity
        and an AxisTransform) borrowed from the current front -- and a
        FOCUSED front publishes geometry for no cell except the one already
        shown, so "show cell j" became inexpressible exactly when a cell was
        already open, and the caller had nothing to do but drop the request.
        An index cannot go stale the way a box can: the session clamps it to
        the cells it has.
        """

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("facet focus takes a cell index")
        if index < 0:
            raise ValueError("facet focus requires a non-negative cell index")

        def apply() -> None:
            self._require_session().focus_facet(index)

        return self._dispatch_session(
            apply,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="facet-presentation",
        )

    def show_facet_overview(
        self,
    ) -> Future[RasterOperation[None]]:
        """Return a focused FacetGrid front to its complete overview."""

        return self._dispatch_session(
            lambda: self._require_session().show_facet_overview(
            ),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="facet-presentation",
        )

    def reset_viewport(
        self,
        *,
        emit_change: bool = True,
    ) -> Future[RasterOperation[None]]:
        return self._dispatch_session(
            lambda: self._require_session().reset_viewport(
                emit_change=emit_change,
            ),
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="viewport",
        )

    def _submit(
        self,
        callback: Callable[[], ValueT],
        *,
        mode: _DispatchMode,
        coalesce_key: object | None = None,
        after_publish: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
        hand: bool = False,
    ) -> Future[RasterOperation[ValueT]]:
        if not isinstance(mode, _DispatchMode):
            raise TypeError("task mode must be _DispatchMode")
        if hand:
            # Held from the gesture's arrival, not from its render: the
            # sibling must stand aside BEFORE it starts a frame this
            # move would then have to wait out.
            _HANDS.touch(self._host_id)
        if mode is _DispatchMode.PRESENTATION:
            if not callable(after_publish) or not callable(on_abort):
                raise TypeError("presentation task requires finalize and abort callbacks")
        elif after_publish is not None or on_abort is not None:
            raise TypeError("finalize/abort are valid only for a presentation task")
        completion: Future[RasterOperation[ValueT]] = Future()
        task = _WorkerTask(
            callback,
            completion,
            mode,
            coalesce_key,
            after_publish,
            on_abort,
            hand,
        )
        superseded: Future[RasterOperation[Any]] | None = None
        with self._condition:
            if self._closing:
                completion.set_exception(self._unusable())
                return completion
            replaced = None
            if coalesce_key is not None:
                # The newest task with this key, wherever it sits.  Checking
                # only the tail meant one interleaved task of any other kind
                # made every following pointer move a separate frame to
                # compose: measured 30-51 uncoalesced motions in a three
                # second drag over a live panel.
                for index in range(len(self._pending) - 1, -1, -1):
                    if self._pending[index].coalesce_key == coalesce_key:
                        replaced = index
                        break
            if replaced is None:
                self._pending.append(task)
            else:
                # COALESCING MAY NOT MOVE WORK EARLIER.  Dropping the new
                # task into the superseded one's slot puts it wherever
                # that one happened to sit -- and a hover move queued
                # before a press sits BEFORE it, so the first move of a
                # drag overtook the press that was supposed to create the
                # gesture.  It then ran with no gesture, answered with no
                # active pan, and its answer cleared the button latch;
                # every move after that was submitted with no button and
                # routed as a hover, so the drag never reached the
                # gesture at all.  Recorded on the operator's console,
                # seven drags in fifty-six.
                #
                # Superseding is about not composing every intermediate
                # move, which taking the tail position does just as well.
                superseded = self._pending[replaced].completion
                del self._pending[replaced]
                self._pending.append(task)
            self._condition.notify()
        # Future.cancel() invokes done callbacks synchronously.  Never run
        # application callbacks while holding the non-reentrant queue lock.
        if superseded is not None:
            superseded.cancel()
        return completion

    def _take_next_task(self) -> _WorkerTask:
        """The next task to run: a pointer first, then arrival order.

        Before the first front exists, its one explicit startup task goes
        first.  Otherwise a queued no-op ADAPTIVE configuration captures an
        unpromoted ghost front merely because ``self.front`` is still None,
        and the startup task immediately captures the same surface again.

        A pointer move and a camera frame are not the same kind of work.  One
        is a person's hand, which is only worth serving while the hand is
        still there; the other is a measurement, which is worth serving
        whenever it is served.  Run strictly in arrival order they compete at
        equal priority, and a drag over a live panel spends its whole time
        behind frames -- measured, a motion waiting on a 22-30 ms commit that
        was queued before it.

        Pointer tasks keep their order AMONG THEMSELVES (press, move, release
        is a sequence), and only one MOVE can be pending at a time because
        they coalesce, so a data frame waits for at most a few of them.
        That order is the queue's, and it holds only because superseding a
        pointer task appends the replacement instead of dropping it into
        the superseded one's slot -- which used to let a drag's first move
        overtake its own press.  See _submit.
        """

        if self._front is None:
            for index, candidate in enumerate(self._pending):
                if candidate.coalesce_key == "initial-front":
                    del self._pending[index]
                    return candidate
        for index, candidate in enumerate(self._pending):
            if candidate.mode is _DispatchMode.ADAPTIVE:
                del self._pending[index]
                return candidate
        return self._pending.popleft()

    def _yields_to_hand(self, task: _WorkerTask) -> bool:
        """Does this task stand aside for a hand -- anybody's, including
        the one on this very surface?

        Only speculative pixels yield -- a data frame or a presentation
        repaint, whose content the next one supersedes anyway.  CONTROL
        work never yields: it carries answers a caller is blocked on.
        Nor does a host without a front yet: putting a surface on screen
        for the first time is not speculative, and a mount that happens
        during someone else's drag (opening Edit, for one) must not wait
        on it.
        """

        if not task.mode.publishes or self._front is None:
            return False
        return _HANDS.busy()

    def _run(self) -> None:
        kernels.configure_worker_threads()
        try:
            try:
                from .session import PlotSession

                factory = self._session_factory
                if factory is None:
                    raise RuntimeError("raster session factory was already released")
                session = factory()
                if not isinstance(session, PlotSession):
                    raise TypeError("session_factory must create PlotSession")
                self._session = session
                defaults = session.defaults
                release = session.attach_host(
                    self,
                    self.dispatch_control,
                    presentation_dispatch=self.dispatch_presentation,
                )
                if not callable(release):
                    raise TypeError("session.attach_host must return a release callback")
                self._session_defaults = defaults
                self._release_session_host = release
                self._surface_release = session.subscribe_surface(
                    self._on_session_surface
                )
            except Exception as error:
                with self._condition:
                    self._startup_error = error
                    pending = tuple(self._pending)
                    self._pending.clear()
                    self._closing = True
                    self._condition.notify_all()
                    # The initial front is one of these tasks.  A caller
                    # already blocked on it must hear the same sentence as
                    # every later refusal -- "failed to start: <reason>" --
                    # not the bare exception, which only a caller who
                    # arrived AFTER the failure was told the reason for.
                    failure = self._unusable()
                for task in pending:
                    if not task.completion.done():
                        task.completion.set_exception(failure)
                return
            else:
                with self._condition:
                    self._ready = True
                    self._condition.notify_all()
            while True:
                expired: tuple[_DataFrame, float] | None = None
                task: _WorkerTask | None = None
                with self._condition:
                    while task is None and expired is None:
                        remaining = None
                        active = self._frame_active
                        if active is not None and active.stage == "solve":
                            elapsed = max(0.0, monotonic() - active.started_at)
                            remaining = self.ACTIVE_FIT_TIMEOUT_SECONDS - elapsed
                            if remaining <= 0.0:
                                active.stage = "timed_out"
                                active.cancel.set()
                                expired = (active, elapsed)
                                break
                        if self._pending:
                            candidate = self._take_next_task()
                            if self._yields_to_hand(candidate):
                                # Exactly where it came from: only a
                                # queue-head publish task can yield, so
                                # arrival order among frames is intact.
                                self._pending.appendleft(candidate)
                                self._condition.wait(
                                    timeout=_HAND_YIELD_POLL_SECONDS
                                )
                                continue
                            task = candidate
                            break
                        if self._closing:
                            return
                        self._condition.wait(timeout=remaining)
                if expired is not None:
                    self._expire_data_frame(*expired)
                    continue
                assert task is not None
                if not task.completion.set_running_or_notify_cancel():
                    with self._condition:
                        self._condition.notify_all()
                    continue
                started = monotonic()
                if task.hand:
                    _HANDS.grip(self._host_id)
                try:
                    operation = self._execute_worker_task(
                        task.callback,
                        task.mode,
                        after_publish=task.after_publish,
                        on_abort=task.on_abort,
                    )
                except Exception as error:
                    task.completion.set_exception(error)
                else:
                    task.completion.set_result(operation)
                finally:
                    if task.hand:
                        # The gap between two moves is one render long,
                        # so the hold has to outlast a render or the
                        # sibling slips a frame in between every pair.
                        _HANDS.ungrip(self._host_id, monotonic() - started)
                    with self._condition:
                        self._condition.notify_all()
        finally:
            # A dead host's hand is nobody's: expiry would free the
            # board on its own, but a closing worker knows sooner.
            _HANDS.forget(self._host_id)
            try:
                session = self._session
                if session is not None:
                    surface_release, self._surface_release = (
                        self._surface_release,
                        None,
                    )
                    if surface_release is not None:
                        try:
                            surface_release()
                        except Exception:
                            pass
                    release, self._release_session_host = (
                        self._release_session_host,
                        None,
                    )
                    if release is not None:
                        release()
                    if self._close_session:
                        session.close()
            finally:
                with self._condition:
                    self._session = None
                    self._session_defaults = None
                    self._session_factory = None
                    self._front = None
                    self._front_callbacks.clear()
                    self._initial_front = None
                    self._closed = True
                    self._condition.notify_all()

    def _capture_front(self) -> RasterFront:
        session = self._require_session()
        plan = session.surface_plan
        width, height = tuple(plan.raster_size)
        raw, actual_height, actual_width = session._raster_capture_rgba_bytes()
        if (actual_height, actual_width) != (height, width):
            raise RuntimeError(
                "session RGBA shape does not match its surface raster size"
            )
        buffer = RasterBuffer(width, height, raw)
        axes_maps = session._raster_axes_snapshot()
        selectors = tuple(session._raster_interaction_snapshot())
        color_limits = session._raster_color_limits_snapshot()
        facet_focus_index = session.facet_focus_index
        revisions = session.revisions
        self._sequence += 1
        identity = RasterIdentity(
            host_id=self._host_id,
            sequence=self._sequence,
            data_generation=session.data_generation,
            data_revision=int(revisions.data),
            image_overlay_revision=session.image_overlay_revision,
            display_revision=int(revisions.display),
            layout_revision=int(revisions.layout),
            kind=str(plan.kind),
            preset=str(plan.preset),
        )
        return RasterFront(
            identity=identity,
            buffer=buffer,
            logical_size=tuple(plan.logical_size),
            logical_dpi=float(plan.logical_dpi),
            device_pixel_ratio=float(plan.device_pixel_ratio),
            interaction=RasterInteractionMap(
                axes=axes_maps,
                selectors=selectors,
                color_limits=color_limits,
                facet_focus_index=facet_focus_index,
            ),
        )

    def _promote(self, front: RasterFront) -> bool:
        with self._condition:
            if self._closing:
                return False
            self._front = front
            callbacks = tuple(self._front_callbacks)
        for callback in callbacks:
            try:
                callback(front)
            except Exception as error:
                with self._condition:
                    self._last_front_callback_error = error
                continue
        return True

    #: How long close waits for the worker by default.
    #:
    #: Bounded, because the caller is usually the GUI thread and the worker's
    #: exit path shuts down the analysis executor with wait=True -- so an
    #: unbounded join could park the whole window behind a fit that had not
    #: finished.  Long
    #: enough that a normal exit is never cut short; finite so a wedged worker
    #: is a reported failure rather than a frozen application.
    CLOSE_SECONDS = 30.0

    def close(self, *, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must be non-negative or None")
        if timeout is None:
            timeout = self.CLOSE_SECONDS
        with self._condition:
            if self._closing:
                thread = self._thread
                pending: tuple[_WorkerTask, ...] = ()
                frames: tuple[_DataFrame, ...] = ()
            else:
                self._closing = True
                pending = tuple(self._pending)
                self._pending.clear()
                active = self._frame_active
                if active is not None:
                    active.cancel.set()
                latest, self._frame_latest = self._frame_latest, None
                if latest is not None:
                    latest.cancel.set()
                frames = (
                    (() if active is None else (active,))
                    + (() if latest is None else (latest,))
                )
                self._front_callbacks.clear()
                self._condition.notify_all()
                thread = self._thread
        # See _submit(): cancellation callbacks may re-enter this host.
        for task in pending:
            task.completion.cancel()
        for frame in frames:
            frame.completion.cancel()
        if thread is not None and thread is not current_thread():
            thread.join(timeout)
        # The host made the widget and keeps it, so the host ends it: a
        # widget left mounted on a closed host goes on delivering mouse
        # events, and the first question it asks -- the selector handle
        # radius, out of ``defaults`` -- raises inside ``mouseMoveEvent``
        # and takes the whole application down.  Only from the widget's own
        # thread; from anywhere else the widget's own guard stands.
        widget = self._qt_widget
        if widget is not None:
            try:
                from PyQt5 import QtCore as _QtCore

                if _QtCore.QThread.currentThread() == widget.thread():
                    widget.close_adapter()
            except Exception:
                pass
        stopped = thread is None or not thread.is_alive()
        if stopped:
            with self._condition:
                self._initial_front = None
                if self._thread is thread:
                    self._thread = None
        return stopped

    def __enter__(self) -> "RasterPlotHost":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


__all__ = [
    "RasterBuffer",
    "RasterFront",
    "RasterIdentity",
    "RasterInteractionMap",
    "RasterOperation",
    "RasterPlotHost",
]
