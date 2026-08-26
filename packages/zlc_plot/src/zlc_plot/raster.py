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
from io import BytesIO
from pathlib import Path
from threading import Condition, Event, Lock, Thread, current_thread
from time import monotonic
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import uuid4

import numpy as np
from PIL import Image
from zlc_durable import atomic_write_bytes

from .units import UnitRegistry

from ._axis_transform import AxisTransform
from ._validation import finite_real as _finite
from ._validation import integer, optional_nonempty_text, readonly_copy, text
from .config import DEFAULTS, PlotLibraryDefaults
from .selectors import (
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorSnapshot,
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


@dataclass(frozen=True, slots=True)
class RasterBuffer:
    """One tightly packed, owned, immutable RGBA8888 image."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        width = integer(self.width, "raster width", minimum=1)
        height = integer(self.height, "raster height", minimum=1)
        if not isinstance(self.pixels, bytes):
            raise TypeError("raster pixels must be owned immutable bytes")
        if len(self.pixels) != width * height * 4:
            raise ValueError("RGBA8888 byte length does not match raster dimensions")

    def as_rgba(self, *, copy: bool = False) -> np.ndarray:
        """Return this exact RGBA8888 buffer as a read-only array.

        The default view shares the immutable byte storage.  ``copy=True``
        returns independent storage while retaining the read-only contract.
        Neither path renders, rescales, or color-converts the image.
        """

        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean")
        rgba = np.frombuffer(self.pixels, dtype=np.uint8).reshape(
            self.height,
            self.width,
            4,
        )
        if copy:
            rgba = readonly_copy(rgba)
        return rgba

    def encode(self, format: str = "PNG", **options: object) -> bytes:
        """Encode the current physical pixels without rendering or resizing."""

        selected_format = text(format, "image format").upper()
        output = BytesIO()
        Image.frombytes(
            "RGBA",
            (self.width, self.height),
            self.pixels,
        ).save(output, format=selected_format, **options)
        return output.getvalue()

    def save(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        **options: object,
    ) -> None:
        """Encode the current physical pixels to ``path`` without rerendering."""

        target = Path(path)
        selected = format
        if selected is None:
            selected = target.suffix.removeprefix(".")
        atomic_write_bytes(
            target,
            self.encode(text(selected, "image format"), **options)
        )


@dataclass(frozen=True, slots=True)
class RasterIdentity:
    """Exact session state represented by one accepted raster."""

    host_id: str
    sequence: int
    data_generation: str | None
    data_revision: int
    image_overlay_revision: int | None
    display_revision: int
    layout_revision: int
    kind: str
    preset: str

    def __post_init__(self) -> None:
        for name in (
            "sequence",
            "data_revision",
            "display_revision",
            "layout_revision",
        ):
            value = integer(getattr(self, name), name, minimum=0)
            assert value is not None
            object.__setattr__(self, name, value)
        for name in ("host_id", "kind", "preset"):
            object.__setattr__(self, name, text(getattr(self, name), name))
        object.__setattr__(
            self,
            "data_generation",
            optional_nonempty_text(self.data_generation, "data_generation"),
        )
        object.__setattr__(
            self,
            "image_overlay_revision",
            integer(
                self.image_overlay_revision,
                "image_overlay_revision",
                minimum=0,
                optional=True,
            ),
        )

    def same_surface(self, other: object) -> bool:
        """Return whether two fronts share one current interactive surface."""

        return isinstance(other, RasterIdentity) and (
            self.host_id,
            self.display_revision,
            self.layout_revision,
            self.kind,
            self.preset,
        ) == (
            other.host_id,
            other.display_revision,
            other.layout_revision,
            other.kind,
            other.preset,
        )


@dataclass(frozen=True, slots=True)
class RasterInteractionMap:
    """Everything pointer handling may read from the exact painted front."""

    axes: tuple[AxisTransform, ...]
    selectors: tuple[SelectorState, ...]
    color_limits: NumericRange | None = None
    facet_focus_index: int | None = None

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes or any(not isinstance(value, AxisTransform) for value in axes):
            raise ValueError("interaction map requires AxisTransform values")
        selectors = SelectorSnapshot(tuple(self.selectors)).committed
        color_limits = self.color_limits
        if color_limits is not None and not isinstance(color_limits, NumericRange):
            raise TypeError("color_limits must be NumericRange or None")
        focus = integer(
            self.facet_focus_index,
            "facet_focus_index",
            minimum=0,
            optional=True,
        )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "selectors", selectors)
        object.__setattr__(self, "color_limits", color_limits)
        object.__setattr__(self, "facet_focus_index", focus)

@dataclass(frozen=True, slots=True)
class RasterFront:
    """One complete immutable frontend value promoted atomically."""

    identity: RasterIdentity
    buffer: RasterBuffer
    logical_size: tuple[int, int]
    logical_dpi: float
    device_pixel_ratio: float
    interaction: RasterInteractionMap

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RasterIdentity):
            raise TypeError("front identity must be RasterIdentity")
        if not isinstance(self.buffer, RasterBuffer):
            raise TypeError("front buffer must be RasterBuffer")
        width, height = tuple(self.logical_size)
        object.__setattr__(
            self,
            "logical_size",
            (
                integer(width, "logical width", minimum=1),
                integer(height, "logical height", minimum=1),
            ),
        )
        logical_dpi = _finite(self.logical_dpi, "logical dpi")
        if logical_dpi <= 0.0:
            raise ValueError("logical dpi must be positive")
        ratio = _finite(self.device_pixel_ratio, "device pixel ratio")
        if ratio <= 0.0:
            raise ValueError("device pixel ratio must be positive")
        if not isinstance(self.interaction, RasterInteractionMap):
            raise TypeError("front interaction must be RasterInteractionMap")


@dataclass(frozen=True, slots=True)
class RasterOperation(Generic[ValueT]):
    """A worker result and the exact front painted after that result."""

    value: ValueT
    front: RasterFront


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

@dataclass(slots=True)
class _WorkerTask:
    callback: Callable[[], Any]
    completion: Future[RasterOperation[Any]]
    mode: _DispatchMode
    coalesce_key: object | None
    after_publish: Callable[[], None] | None
    on_abort: Callable[[], None] | None


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
        self._host_id = uuid4().hex
        self._sequence = 0
        self._front: RasterFront | None = None
        self._initial_metadata: tuple[object, object] | None = None
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
    ) -> Future[RasterOperation[Callable[[], Future[RasterOperation[None]]]]]:
        """Install a fit callback on the raster worker."""

        return self._subscribe_session_event(
            callback,
            lambda session, listener: session.subscribe_fit(listener),
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
        **kwargs: Any,
    ) -> Future[RasterOperation[Any]]:
        callback = lambda: operation(*args, **kwargs)
        return self._submit(
            callback,
            mode=_mode,
            coalesce_key=coalesce_key,
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
            return self._require_session().prepare_live_frame(
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
        fit_live: bool = True,
    ) -> Future[RasterOperation["DisplayDescription"]]:
        """Submit one complete desired plot target as one raster operation.

        ``parameter_updates`` identifies the fields authored in this
        transaction while ``parameters`` remains the coalescing-safe complete
        target. Plot, not the embedder, resolves transition-generated values.
        """

        configuration = {
            "semantic": None if semantic is None else dict(semantic),
            "parameters": None if parameters is None else dict(parameters),
            "parameter_updates": (
                None if parameter_updates is None else dict(parameter_updates)
            ),
            "size": size,
        }
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
        configuration["fit_live"] = fit_live
        pending = self._dispatch_session(
            lambda: self._require_session().configure(**configuration),
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
            _mode=_DispatchMode.PUBLISH,
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

    def _pointer_event(
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
    ) -> Future[RasterOperation[object]]:
        """Route Qt raster input through PlotSession's interaction engine."""

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
                # Identity CURRENCY belongs to the press alone.  A scroll
                # is self-relative view navigation (a 3D wheel tick commits
                # the camera and bumps the display revision, so demanding
                # currency made every following tick in the frontend's
                # one-front lag window bounce); moves and releases are
                # anchored to the front their own press validated.
                if selected_action == "press":
                    revisions = session.revisions
                    plan = session.surface_plan
                    if (
                        effective_identity is None
                        or int(revisions.display)
                        != effective_identity.display_revision
                        or int(revisions.layout)
                        != effective_identity.layout_revision
                        or str(plan.kind) != effective_identity.kind
                        or str(plan.preset) != effective_identity.preset
                    ):
                        raise RuntimeError(
                            "the painted pointer front is no longer layout-compatible"
                        )
                    # A press means what the OPERATOR saw -- and the painted
                    # front it carries IS what they saw: the widget swaps
                    # pixels and identity atomically, so the transform that
                    # arrives with the event always matches the picture the
                    # operator pressed on, and the gesture layer interprets
                    # the press THROUGH that transform into canonical
                    # coordinates.  Demanding that it also equal the
                    # session's CURRENT transform re-rejected the first
                    # press after every commit for as long as the frontend
                    # ran one front behind -- data currency, transform
                    # currency, one disease.  What still gates: the
                    # grabbable interaction state below, because a grab
                    # resolves against current selectors.
                    if effective_interaction is not None:
                        # A press consumes the GRABBABLE geometry: region
                        # edges and threshold lines.  The crosshair is a
                        # marker -- nothing grabs it -- yet every pick
                        # republishes a front carrying it, and demanding
                        # marker equality rejected the very next press for
                        # as long as the frontend lagged one front behind
                        # (every orbit after a pick, in practice).  The
                        # clim handles live on the distribution rail, so
                        # their currency gates rail presses alone.
                        def _grabbable(states: object) -> tuple:
                            return tuple(
                                state
                                for state in states
                                if state.kind is not SelectorKind.CROSSHAIR
                            )

                        stale_selectors = _grabbable(
                            session._raster_interaction_snapshot()
                        ) != _grabbable(effective_interaction.selectors)
                        on_rail = (
                            getattr(effective_axes, "role", None)
                            == "distribution"
                        )
                        stale_limits = on_rail and (
                            session._raster_color_limits_snapshot()
                            != effective_interaction.color_limits
                        )
                        if stale_selectors or stale_limits:
                            raise RuntimeError(
                                "the painted interaction state is no longer current"
                            )
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
    ) -> Future[RasterOperation[ValueT]]:
        if not isinstance(mode, _DispatchMode):
            raise TypeError("task mode must be _DispatchMode")
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
                superseded = self._pending[replaced].completion
                self._pending[replaced] = task
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
        is a sequence), and only one can be pending at a time because they
        coalesce, so a data frame waits for at most one of them.
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

    def _run(self) -> None:
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
                for task in pending:
                    if not task.completion.done():
                        task.completion.set_exception(error)
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
                            task = self._take_next_task()
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
                    with self._condition:
                        self._condition.notify_all()
        finally:
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
