"""Immutable Agg-to-frontend hand-off and serial plot worker.

The Qt adapter consumes only :class:`RasterFront` values.  A live Matplotlib
Figure, Artist, mutable array view, or Qt object never crosses this boundary.
All plot mutations and raster capture run on the single worker owned by
``RasterPlotHost``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from threading import Condition, Lock, Thread, current_thread
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
    from .live import LiveContract, LivePlotController
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
EventT = TypeVar("EventT")
_UNSET = object()


def _axis_at_normalized(
    front: "RasterFront",
    x: float,
    y: float,
) -> AxisTransform | None:
    """Resolve the current front's axis under a normalized pointer."""

    return next(
        (
            axis
            for axis in front.interaction.axes
            if axis.bounds[0] <= x <= axis.bounds[2]
            and axis.bounds[1] <= y <= axis.bounds[3]
        ),
        None,
    )


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
    source_revisions: tuple[int, ...]

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
        revisions = tuple(self.source_revisions)
        if not revisions:
            raise ValueError("front source_revisions cannot be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in revisions
        ):
            raise TypeError("front source_revisions must contain non-negative integers")
        # Ties are legal: a seeded rolling history carries one sample per
        # repeat of the same source revision.  Only regressions are invalid.
        if any(left > right for left, right in zip(revisions, revisions[1:])):
            raise ValueError("front source_revisions must be non-decreasing")
        object.__setattr__(self, "source_revisions", tuple(int(value) for value in revisions))


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


_CoalesceResolver = Callable[[tuple[Any, ...], Mapping[str, Any]], object | None]


def _parameter_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object:
    return ("parameter", str(args[0]))


def _parameters_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object:
    return ("parameters", tuple(sorted(args[0])))


def _axis_unit_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object:
    return ("axis-unit", args[0])


def _labels_coalesce(
    _args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> object:
    return ("labels", tuple(sorted(kwargs)))


def _pointer_coalesce(args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> object | None:
    return "pointer-motion" if args[0] == "move" else None

@dataclass(slots=True)
class _WorkerTask:
    callback: Callable[[], Any]
    completion: Future[RasterOperation[Any]]
    mode: _DispatchMode
    coalesce_key: object | None
    after_publish: Callable[[], None] | None
    on_abort: Callable[[], None] | None


class _WorkerSessionAdapter:
    """Concentrate worker-only PlotSession presentation details."""

    def __init__(self, host: "RasterPlotHost") -> None:
        self._host = host

    def _session(self) -> "PlotSession":
        if current_thread() is not self._host._thread:
            raise RuntimeError("plot session access must run on the raster worker")
        return self._host._require_session()

    @property
    def data_revision(self) -> int | None:
        front = self._host.front
        return None if front is None else front.identity.data_revision

    @property
    def defaults(self) -> PlotLibraryDefaults:
        return self._host.defaults

    def update_data(
        self,
        data: object,
        *,
        revision: int | None = None,
    ) -> object:
        return self._session().update_data(data, revision=revision)

    def update_image_overlay(self, overlay: object) -> object:
        return self._session().update_image_overlay(overlay)

    def update_image_frame(self, frame: object) -> object:
        return self._session().update_image_frame(frame)

    def set_parameter(self, name: str, value: object) -> object:
        return self._session().set_parameter(name, value)

    def set_parameters(self, values: Mapping[str, object]) -> object:
        return self._session().set_parameters(values)

    def configure(self, **configuration: object) -> object:
        return self._session().configure(**configuration)

    def describe_display(self) -> object:
        return self._session().describe_display()

    def describe_semantics(self) -> object:
        return self._session().describe_semantics()

    def replace_spec(self, spec: "PlotSpec", *, parameters: Mapping[str, object] | None = None) -> object:
        return self._session().replace_spec(spec, parameters=parameters)

    def apply_semantic(self, name: str, value: object) -> object:
        return self._session().apply_semantic(name, value)

    def resolved_color_limits(self, *, display: bool = True) -> object:
        return self._session().resolved_color_limits(display=display)

    def set_labels(self, **updates: object) -> object:
        return self._session().set_labels(**updates)

    def set_relim_mode(self, mode: str) -> object:
        return self._session().set_relim_mode(mode)

    def set_y_limits(self, low: float, high: float, *, fixed: bool = True) -> object:
        return self._session().set_y_limits(low, high, fixed=fixed)

    def reset_y_limits(self, *, mode: str = "normal") -> object:
        return self._session().reset_y_limits(mode=mode)

    def set_color_limits(self, low: float, high: float, *, fixed: bool = True) -> object:
        return self._session().set_color_limits(low, high, fixed=fixed)

    def reset_color_limits(self, *, mode: str = "tight") -> object:
        return self._session().reset_color_limits(mode=mode)

    def set_x_limits(self, low: float, high: float) -> object:
        return self._session().set_x_limits(low, high)

    def set_view_limits(self, *, x: object = None, y: object = None) -> object:
        return self._session().set_view_limits(x=x, y=y)

    def set_size(self, preset: str) -> object:
        return self._session().set_size(preset)

    def set_device_pixel_ratio(self, ratio: float) -> object:
        return self._session().set_device_pixel_ratio(ratio)

    def set_axis_unit(self, axis: object, unit: str | None) -> object:
        return self._session().set_axis_unit(axis, unit)

    def set_value_unit(self, unit: str | None) -> object:
        return self._session().set_value_unit(unit)

    def set_time_unit(self, unit: str | None) -> object:
        return self._session().set_time_unit(unit)

    def save(self, path: str | Path, **options: object) -> object:
        return self._session().save(path, **options)

    def clear_fit(self) -> object:
        return self._session().clear_fit()

    def fit_models(self) -> object:
        return self._session().fit_models

    def selectors(self) -> object:
        return self._session().selectors

    def selector_state(self, kind: SelectorKind, *, display: bool = False) -> object:
        return self._session().selector_state(kind, display=display)

    def selector_data(self, kind: SelectorKind) -> object:
        return self._session().selector_data(kind)

    def remove_selector(self, kind: SelectorKind) -> object:
        return self._session().remove_selector(kind)

    def set_selector_value(self, kind: SelectorKind, value: object, *, display: bool = True) -> object:
        return self._session().set_selector_value(kind, value, display=display)

    def set_area_selector(self, x: NumericRange, y: NumericRange, *, display: bool = True) -> object:
        return self._session().set_area_selector(x, y, display=display)

    def set_x_selector(self, low: float, high: float, *, display: bool = True) -> object:
        return self._session().set_x_selector(low, high, display=display)

    def set_threshold_selector(self, value: float, *, display: bool = True) -> object:
        return self._session().set_threshold_selector(value, display=display)

    def set_crosshair_selector(self, x: float, y: float, *, display: bool = True) -> object:
        return self._session().set_crosshair_selector(x, y, display=display)

    def fit(self, model: object, **options: object) -> object:
        return self._session().fit_async(model, **options)

    def set_viewport(self, x: NumericRange, y: NumericRange) -> object:
        return self._session().set_viewport(x, y)

    def show_facet_overview(self) -> object:
        return self._session().show_facet_overview()

    def reset_viewport(self) -> object:
        return self._session().reset_viewport()

    def focus_facet(self, index: int) -> object:
        return self._session().focus_facet(index)

    def prepare_live_frame(
        self,
        data: object,
        *,
        revision: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Future[object]:
        return self._session().prepare_live_frame(
            data,
            revision=revision,
            cancelled=cancelled,
        )

    def commit_live_frame(self, prepared: object) -> object | None:
        return self._session().commit_live_frame(prepared)

    def finalize_live_frame(self, finalization: object) -> None:
        self._session().finalize_live_frame(finalization)

    def abort_live_frame(self, finalization: object) -> None:
        self._session().abort_live_frame(finalization)

    def capture_rgba_bytes(self) -> tuple[bytes, int, int]:
        return self._session()._raster_capture_rgba_bytes()

    def capture_rgba(
        self,
        *,
        redraw: bool = False,
    ) -> np.ndarray:
        return np.asarray(
            self._session()._raster_capture_rgba(redraw=redraw),
            dtype=np.uint8,
        )

    def interaction_state(
        self,
    ) -> tuple[SelectorState, ...]:
        return tuple(self._session()._raster_interaction_snapshot())

    def color_limits(self) -> NumericRange | None:
        return self._session()._raster_color_limits_snapshot()

class RasterPlotHost:
    """Serialize one ``PlotSession`` and publish immutable latest-only fronts.

    The callable factory is invoked on the owned worker, and closing the host
    always closes that session.  A live Figure is never exposed to callers.
    :meth:`from_plot` is the standard immutable-data/spec construction path;
    the raw factory remains available for deliberate ``PlotSession`` subclasses.
    Public mutation methods never run plot work on the caller.  Pending data and
    high-frequency controls coalesce only with the same semantic key; selector,
    fit, unit, size and other distinct commands retain their ordering.
    """

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
        self._worker_adapter = _WorkerSessionAdapter(self)
        self._condition = Condition(Lock())
        self._pending: deque[_WorkerTask] = deque()
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
        callback: Callable[[EventT], object],
        install: Callable[
            ["PlotSession", Callable[[EventT], object]],
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

    def live_controller(
        self,
        contract: "LiveContract",
        *,
        refresh_interval_ms: int | None = None,
    ) -> "LivePlotController":
        """Create a live producer bridge whose mutations stay on this worker."""

        from .live import LivePlotController

        self.wait_for_front()
        return LivePlotController(
            self._worker_adapter,
            contract,
            refresh_interval_ms=refresh_interval_ms,
            dispatch=self.dispatch,
            control_dispatch=self.dispatch_control,
            presentation_dispatch=self.dispatch_presentation,
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
            value = callback()
            publishes = mode.publishes or (
                mode is _DispatchMode.ADAPTIVE
                and bool(getattr(value, "publish_front", False))
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
    ) -> Future[RasterOperation[None]]:
        return self._dispatch_session(
            self._worker_adapter.update_data,
            data,
            revision=revision,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="data",
        )

    def update_image_overlay(
        self,
        overlay: "ImagePointOverlay",
    ) -> Future[RasterOperation["ImagePointOverlay"]]:
        """Coalesce independent Image point revisions on the render worker."""

        return self._dispatch_session(
            self._worker_adapter.update_image_overlay,
            overlay,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="image-overlay",
        )

    def update_image_frame(
        self,
        frame: "ImageFrame",
    ) -> Future[RasterOperation["ImageFrame"]]:
        """Coalesce complete image frames on the serial render worker."""

        return self._dispatch_session(
            self._worker_adapter.update_image_frame,
            frame,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="data",
        )

    def set_parameter(
        self,
        name: str,
        value: object,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            self._worker_adapter.set_parameter,
            name,
            value,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key=("parameter", str(name)),
        )

    def set_parameters(
        self,
        values: Mapping[str, object],
    ) -> Future[RasterOperation["DisplayState"]]:
        prepared = dict(values)
        return self._dispatch_session(
            self._worker_adapter.set_parameters,
            prepared,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key=("parameters", tuple(sorted(prepared))),
        )

    def configure(
        self,
        *,
        semantic: Mapping[str, object] | None = None,
        parameters: Mapping[str, object] | None = None,
        size: str | None = None,
        image_overlay: "ImagePointOverlay | None | object" = _UNSET,
        fit_model: str | None | object = _UNSET,
        classifier_thresholds: Sequence[float | None] | object = _UNSET,
    ) -> Future[RasterOperation["DisplayDescription"]]:
        """Submit the complete desired state as one coalesced worker job."""

        configuration = {
            "semantic": None if semantic is None else dict(semantic),
            "parameters": None if parameters is None else dict(parameters),
            "size": size,
        }
        if image_overlay is not _UNSET:
            configuration["image_overlay"] = image_overlay
        if fit_model is not _UNSET:
            configuration["fit_model"] = fit_model
        if classifier_thresholds is not _UNSET:
            configuration["classifier_thresholds"] = tuple(classifier_thresholds)
        return self._dispatch_session(
            self._worker_adapter.configure,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="configuration",
            **configuration,
        )

    def describe_display(self) -> Future[RasterOperation["DisplayDescription"]]:
        """Return the worker session's immutable control-plane description."""

        return self._dispatch_session(
            self._worker_adapter.describe_display,
            _mode=_DispatchMode.CONTROL,
        )

    def describe_semantics(self) -> Future[RasterOperation[object]]:
        """Return the registry-derived semantic edit domain."""

        return self._dispatch_session(
            self._worker_adapter.describe_semantics,
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
            self._worker_adapter.replace_spec,
            spec,
            parameters=parameters,
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
            self._worker_adapter.apply_semantic,
            str(name),
            value,
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
            self._worker_adapter.resolved_color_limits,
            display=display,
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
            self._worker_adapter.set_labels,
            **updates,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key=("labels", tuple(sorted(updates))),
        )

    def set_relim_mode(self, mode: str) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            self._worker_adapter.set_relim_mode,
            mode,
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
            self._worker_adapter.set_y_limits,
            low,
            high,
            fixed=fixed,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="y-limits",
        )

    def reset_y_limits(
        self,
        *,
        mode: str = "normal",
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            self._worker_adapter.reset_y_limits,
            mode=mode,
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
            self._worker_adapter.set_color_limits,
            low,
            high,
            fixed=fixed,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="color-limits",
        )

    def reset_color_limits(
        self,
        *,
        mode: str = "tight",
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            self._worker_adapter.reset_color_limits,
            mode=mode,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="color-limits",
        )

    def set_x_limits(
        self,
        low: float,
        high: float,
    ) -> Future[RasterOperation[RectangleRange]]:
        return self._dispatch_session(
            self._worker_adapter.set_x_limits,
            low,
            high,
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
            self._worker_adapter.set_view_limits,
            x=x,
            y=y,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="viewport",
        )

    def set_size(self, preset: str) -> Future[RasterOperation["SurfacePlan"]]:
        return self._dispatch_session(
            self._worker_adapter.set_size,
            preset,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="size",
        )

    def set_device_pixel_ratio(self, ratio: float) -> Future[RasterOperation["SurfacePlan"]]:
        selected = _finite(ratio, "device pixel ratio")
        if selected <= 0.0:
            raise ValueError("device pixel ratio must be positive")
        return self._dispatch_session(
            self._worker_adapter.set_device_pixel_ratio,
            selected,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="device-pixel-ratio",
        )

    def set_axis_unit(
        self,
        axis: object,
        unit: str | None,
    ) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            self._worker_adapter.set_axis_unit,
            axis,
            unit,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key=("axis-unit", axis),
        )

    def set_value_unit(self, unit: str | None) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            self._worker_adapter.set_value_unit,
            unit,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="value-unit",
        )

    def set_time_unit(self, unit: str | None) -> Future[RasterOperation["DisplayState"]]:
        return self._dispatch_session(
            self._worker_adapter.set_time_unit,
            unit,
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
            self._worker_adapter.save,
            path,
            dpi=dpi,
            export_scale=export_scale,
            **options,
            _mode=_DispatchMode.CONTROL,
        )

    def clear_fit(self) -> Future[RasterOperation[None]]:
        return self._dispatch_session(
            self._worker_adapter.clear_fit,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="fit-control",
        )

    def fit_models(self) -> Future[RasterOperation[tuple["FitModelSpec", ...]]]:
        """Return the fit catalogue owned by this host's session."""

        return self._dispatch_session(
            self._worker_adapter.fit_models,
            _mode=_DispatchMode.CONTROL,
        )

    def selectors(self) -> Future[RasterOperation[tuple[SelectorState, ...]]]:
        """Return the current canonical selector geometry without slicing data."""

        return self._dispatch_session(
            self._worker_adapter.selectors,
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
            self._worker_adapter.selector_state,
            kind,
            display=display,
            _mode=_DispatchMode.CONTROL,
        )

    def selector_data(self, kind: SelectorKind) -> Future[RasterOperation["SelectorData"]]:
        """Materialize selector output only for this explicit application call."""

        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(
            self._worker_adapter.selector_data,
            kind,
            _mode=_DispatchMode.CONTROL,
        )

    def remove_selector(self, kind: SelectorKind) -> Future[RasterOperation[SelectorState]]:
        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        return self._dispatch_session(
            self._worker_adapter.remove_selector,
            kind,
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
            self._worker_adapter.set_selector_value,
            kind,
            value,
            display=display,
            _mode=_DispatchMode.PUBLISH,
        )

    def set_area_selector(
        self,
        x: NumericRange,
        y: NumericRange,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            self._worker_adapter.set_area_selector,
            x,
            y,
            display=display,
            _mode=_DispatchMode.PUBLISH,
        )

    def set_x_selector(
        self,
        low: float,
        high: float,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            self._worker_adapter.set_x_selector,
            low,
            high,
            display=display,
            _mode=_DispatchMode.PUBLISH,
        )

    def set_threshold_selector(
        self,
        value: float,
        *,
        display: bool = True,
    ) -> Future[RasterOperation[SelectorState]]:
        return self._dispatch_session(
            self._worker_adapter.set_threshold_selector,
            value,
            display=display,
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
            self._worker_adapter.set_crosshair_selector,
            x,
            y,
            display=display,
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
            self._worker_adapter.fit,
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
            live=live,
            fit_all_facets=fit_all_facets,
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
            effective_identity = identity
            effective_axes = axes
            effective_interaction = interaction
            if identity is not None:
                if identity.host_id != self._host_id:
                    raise RuntimeError(
                        "the painted pointer front belongs to another raster host"
                    )
                if selected_action == "press":
                    current_front = self.front
                    if current_front is None:
                        raise RuntimeError("raster host has no current pointer front")
                    # A live frame may have been promoted after the browser/Qt
                    # event was sampled.  The press is still valid: resolve
                    # its transform and selector hit map from the latest
                    # complete front instead of rejecting the gesture.
                    effective_identity = current_front.identity
                    effective_axes = _axis_at_normalized(
                        current_front,
                        x_value,
                        y_value,
                    )
                    effective_interaction = current_front.interaction
                revisions = session.revisions
                plan = session.surface_plan
                if (
                    effective_identity is None
                    or int(revisions.display) != effective_identity.display_revision
                    or int(revisions.layout) != effective_identity.layout_revision
                    or str(plan.kind) != effective_identity.kind
                    or str(plan.preset) != effective_identity.preset
                ):
                    raise RuntimeError(
                        "the painted pointer front is no longer layout-compatible"
                    )
                if selected_action == "press":
                    if (
                        session.data_generation != effective_identity.data_generation
                        or session.data_revision != effective_identity.data_revision
                        or session.image_overlay_revision
                        != effective_identity.image_overlay_revision
                    ):
                        raise RuntimeError(
                            "the current pointer front is no longer session-compatible"
                        )
                    if effective_axes is not None:
                        current_axes = session._raster_axes_snapshot()
                        if effective_axes not in current_axes:
                            raise RuntimeError(
                                "the painted pointer transform is no longer current"
                            )
                    if effective_interaction is not None and (
                        tuple(session._raster_interaction_snapshot())
                        != effective_interaction.selectors
                        or session._raster_color_limits_snapshot()
                        != effective_interaction.color_limits
                    ):
                        raise RuntimeError(
                            "the painted interaction state is no longer current"
                        )
            return session._raster_pointer_event(
                selected_action,
                x_value,
                y_value,
                button=button,
                double=double,
                step=step,
                key=key,
                axes_snapshot=effective_axes,
            )

        return self._dispatch_session(
            apply,
            _mode=_DispatchMode.ADAPTIVE,
            coalesce_key=_pointer_coalesce((selected_action,), {}),
        )

    def set_viewport(
        self,
        x: NumericRange,
        y: NumericRange,
    ) -> Future[RasterOperation[RectangleRange]]:
        """Set visible ranges in the current display units."""

        return self._dispatch_session(
            self._worker_adapter.set_viewport,
            x,
            y,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="viewport",
        )

    def focus_facet(
        self,
        identity: RasterIdentity,
        axes: AxisTransform,
    ) -> Future[RasterOperation[None]]:
        """Open one cell from the exact compatible FacetGrid overview."""

        if not isinstance(identity, RasterIdentity):
            raise TypeError("identity must be RasterIdentity")
        if not isinstance(axes, AxisTransform):
            raise TypeError("axes must be AxisTransform")
        if identity.kind != "facet_grid" or axes.role != "facet_cell":
            raise TypeError("facet focus requires a FacetGrid cell")
        if axes.cell_index is None:
            raise ValueError("facet focus requires a cell index")

        def apply() -> None:
            session = self._require_session()
            revisions = session.revisions
            if (
                identity.host_id != self._host_id
                or revisions.display != identity.display_revision
                or revisions.layout != identity.layout_revision
            ):
                raise RuntimeError(
                    "the overview display/layout changed before facet focus"
                )
            session.focus_facet(axes.cell_index)

        return self._dispatch_session(
            apply,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="facet-presentation",
        )

    def show_facet_overview(self) -> Future[RasterOperation[None]]:
        """Return a focused FacetGrid front to its complete overview."""

        return self._dispatch_session(
            self._worker_adapter.show_facet_overview,
            _mode=_DispatchMode.PUBLISH,
            coalesce_key="facet-presentation",
        )

    def reset_viewport(self) -> Future[RasterOperation[None]]:
        return self._dispatch_session(
            self._worker_adapter.reset_viewport,
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
            if (
                coalesce_key is not None
                and self._pending
                and self._pending[-1].coalesce_key == coalesce_key
            ):
                pending = self._pending[-1]
                superseded = pending.completion
                self._pending[-1] = task
            else:
                self._pending.append(task)
            self._condition.notify()
        # Future.cancel() invokes done callbacks synchronously.  Never run
        # application callbacks while holding the non-reentrant queue lock.
        if superseded is not None:
            superseded.cancel()
        return completion

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
                with self._condition:
                    while not self._pending and not self._closing:
                        self._condition.wait()
                    if not self._pending and self._closing:
                        return
                    task = self._pending.popleft()
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
        raw, actual_height, actual_width = self._worker_adapter.capture_rgba_bytes()
        if (actual_height, actual_width) != (height, width):
            raise RuntimeError(
                "session RGBA shape does not match its surface raster size"
            )
        buffer = RasterBuffer(width, height, raw)
        axes_maps = session._raster_axes_snapshot()
        selectors = self._worker_adapter.interaction_state()
        color_limits = self._worker_adapter.color_limits()
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
            source_revisions=session._raster_source_revisions_snapshot(),
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
    #: exit path shuts down a fit pool with wait=True -- so an unbounded join
    #: could park the whole window behind a fit that had not finished.  Long
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
            else:
                self._closing = True
                pending = tuple(self._pending)
                self._pending.clear()
                self._front_callbacks.clear()
                self._condition.notify_all()
                thread = self._thread
        # See _submit(): cancellation callbacks may re-enter this host.
        for task in pending:
            task.completion.cancel()
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
