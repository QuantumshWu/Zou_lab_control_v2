"""Public stateful plotting API shared by notebooks, Qt5 and headless use."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from threading import Event, RLock, current_thread
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeAlias, TypeVar
import math

import numpy as np

from zlc_data import OwnedSnapshot
from zlc_data.snapshot_projection import axis_catalog, indexed_schemas_compatible
from zlc_durable import atomic_write_file

from .data_contract import (
    DEFAULT_UNITS,
    UnitRegistry,
    resolve_unit,
    schema_equal,
    schema_value_unit,
    snapshot_generation,
    snapshot_revision,
    snapshot_schema,
)
from .errors import RevisionError

from ._axis_transform import AxisTransform, canvas_physical_size
from ._gesture_engine import (
    _OrbitGesture,
    _ColorGesture,
    _ColorLimitDrag,
    _PanGesture,
    _PointerGesture,
    _SelectorGesture,
    area_drag_handle,
    pan_rectangle,
    range_endpoint_hit,
)
from ._session_fit import FitSessionMixin, _WarmSeed
from ._session_gesture import GestureSessionMixin
from ._session_live import LiveSessionMixin
from ._session_state import (
    _AcceptedFit,
    _FitPresentation,
    _FitResolution,
    FitEvent,
    _LiveFitRequest,
    _LiveFrameFinalization,
    _LiveFrameSnapshot,
    _PointerUpdate,
    _PreparedLiveFrame,
    _ProjectionPresentation,
    _ResolvedFit,
    SelectionChange,
    _StartedFitRequest,
)
from ._pulse_time import pulse_time_scale
from ._fit_projection import (
    FitProjection,
    FitScope,
    FitSelection,
    ProjectionContext,
    RollingHistoryPoint,
)
from ._selector_scene import (
    ColorLimitCandidate,
)
from ._validation import readonly_copy as _readonly
from .config import DEFAULTS, PlotLibraryDefaults
from .fit import (
    FitCancelled,
    FitEngine,
    FacetFitBatchResult,
    FitModelSpec,
    FitOptions,
    FitParameterDisplay,
    FitResult,
)
from .kinds import AxisDomain, AxisRef, PlotKind
from ._kinds import handler_for
from .layout import FacetTopology, SurfacePlan, resolve_surface
from .parameters import ParameterSchema, RenderEffect
from .primitives import (
    ImageFrame,
    ImagePointOverlay,
    PlotInput,
    PulseAnalogTrace,
    PulseBlock,
    PulseChannel,
    PulseDacScanSegment,
    PulseRepeatMarker,
    PulseScanRegion,
    PulseTimelineData,
)
from .rendering import MatplotlibRenderer, RenderFrame, _image_coordinate_aspect
from .selectors import (
    CrosshairPoint,
    NumericRange,
    RectangleRange,
    _SelectorController,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
    SelectorValue,
    _classifier_threshold_target_from_subject,
    normalize_classifier_threshold_targets,
)
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSpec,
    PulseTimelinePlot,
    RollingPlot,
    parameter_schema_for,
    semantic_spec,
)
from .state import DisplayState, DisplayStateStore
from .semantics import (
    SemanticDescription,
    composed_spec,
    describe_semantics,
    updated_spec,
)
from .session_policy import replace_spec_initial_state


# Semantic probes are cheap validation work, but the edit surface can contain
# many candidate specs (especially facet choices).  Keep the cache bounded and
# make the bound visible to both the UI adapter and its performance guards.
_SEMANTIC_PROBE_CACHE_MAX = 256


_ProjectionInput = OwnedSnapshot | PulseTimelineData
SurfaceCallback = Callable[[], object]
HostDispatch = Callable[[Callable[[], Any]], Future[Any]]
HostPresentationDispatch = Callable[
    [Callable[[], None], Callable[[], None], Callable[[], None]],
    Future[Any],
]
DisplayCallback = Callable[[DisplayState], object]
FitCallback = Callable[["FitEvent | None"], object]
SelectionCallback = Callable[["SelectionEvent"], object]
_CallbackT = TypeVar("_CallbackT", bound=Callable[..., object])
_EventT = TypeVar("_EventT")
_ResultT = TypeVar("_ResultT")


_ANALYSIS_THREAD_PREFIX = "zlc-analysis"
_UNSET = object()
_CONFIGURATION_STATE_NAMES = (
    "_spec", "_parameter_schema", "_projection", "_image_overlay",
    "_viewport", "_focused_facet_index", "_facet_focus_index", "_accepted_fit",
    "_classifier_results", "_classifier_overlays", "_classifier_thresholds",
    "_classifier_gaussian_components",
    "_history", "_layout_revision", "_size", "_fit_context_generation",
    "_fit_request_generation", "_fit_batch_revision", "_fit_cancel",
    "_live_fit_cancel", "_live_fit_request", "_live_fit_future",
    "_live_fit_completion", "_presentation_epoch",
)


def _validated_device_pixel_ratio(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("device pixel ratio must be a positive finite number")
    selected = float(value)
    if not math.isfinite(selected) or selected <= 0.0:
        raise ValueError("device pixel ratio must be a positive finite number")
    return selected


@dataclass(frozen=True, slots=True)
class SessionRevisions:
    data: int
    display: int
    layout: int


@dataclass(frozen=True, slots=True)
class DisplayDescription:
    """Immutable control-plane snapshot for a notebook or GUI frontend."""

    kind: PlotKind
    size: str
    size_choices: tuple[str, ...]
    parameter_schema: ParameterSchema
    display_state: DisplayState
    parameter_choices: Mapping[str, tuple[object, ...]]
    limits: RectangleRange
    viewport: RectangleRange | None
    semantics: SemanticDescription
    fit_models: tuple[FitModelSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PlotKind):
            raise TypeError("display description kind must be PlotKind")
        if not isinstance(self.size, str) or not self.size:
            raise ValueError("display description size must be non-empty")
        size_choices = tuple(self.size_choices)
        if self.size not in size_choices:
            raise ValueError("current display size must be one of size_choices")
        if not isinstance(self.parameter_schema, ParameterSchema):
            raise TypeError("display description requires ParameterSchema")
        if not isinstance(self.display_state, DisplayState):
            raise TypeError("display description requires DisplayState")
        if not isinstance(self.semantics, SemanticDescription):
            raise TypeError("display description requires SemanticDescription")
        object.__setattr__(self, "fit_models", tuple(self.fit_models))
        parameter_choices = {
            str(name): tuple(values)
            for name, values in self.parameter_choices.items()
        }
        unknown = tuple(
            name for name in parameter_choices if name not in self.parameter_schema
        )
        if unknown:
            joined = ", ".join(repr(name) for name in unknown)
            raise KeyError(f"parameter choices refer to unknown parameter(s): {joined}")
        object.__setattr__(self, "size_choices", size_choices)
        object.__setattr__(
            self,
            "parameter_choices",
            MappingProxyType(parameter_choices),
        )
        if not isinstance(self.limits, RectangleRange):
            raise TypeError("display description limits must be RectangleRange")
        if self.viewport is not None and not isinstance(self.viewport, RectangleRange):
            raise TypeError("display description viewport must be RectangleRange or None")



@dataclass(frozen=True, slots=True)
class SelectionData:
    selector: SelectorState
    mask: np.ndarray
    flat_indices: np.ndarray
    canonical_values: np.ndarray
    display_values: np.ndarray
    canonical_coordinates: Mapping[AxisRef, np.ndarray]
    display_coordinates: Mapping[AxisRef, np.ndarray]
    data_revision: int
    facet_index: int | None = None
    source_revisions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask", _readonly(self.mask, dtype=bool))
        object.__setattr__(self, "flat_indices", _readonly(self.flat_indices, dtype=np.int64))
        object.__setattr__(self, "canonical_values", _readonly(self.canonical_values))
        object.__setattr__(self, "display_values", _readonly(self.display_values))
        canonical = {
            key: _readonly(value) for key, value in self.canonical_coordinates.items()
        }
        display = {key: _readonly(value) for key, value in self.display_coordinates.items()}
        object.__setattr__(self, "canonical_coordinates", MappingProxyType(canonical))
        object.__setattr__(self, "display_coordinates", MappingProxyType(display))
        revisions = tuple(self.source_revisions) or (self.data_revision,)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in revisions
        ):
            raise TypeError("source_revisions must contain non-negative integers")
        object.__setattr__(self, "source_revisions", tuple(int(value) for value in revisions))


@dataclass(frozen=True, slots=True)
class PulseTimelineSelectionData:
    """Immutable timeline records intersecting a selector's source-time span."""

    selector: SelectorState
    display_selector: SelectorState
    channels: tuple[PulseChannel, ...]
    blocks: tuple[PulseBlock, ...]
    analog_traces: tuple[PulseAnalogTrace, ...]
    scan_regions: tuple[PulseScanRegion, ...]
    scan_dac_segments: tuple[PulseDacScanSegment, ...]
    repeat_markers: tuple[PulseRepeatMarker, ...]
    data_revision: int
    source_revisions: tuple[int, ...] = ()

    @property
    def canonical_value(self) -> SelectorValue:
        return self.selector.value

    @property
    def display_value(self) -> SelectorValue:
        return self.display_selector.value

    def __post_init__(self) -> None:
        if isinstance(self.data_revision, bool) or not isinstance(
            self.data_revision, Integral
        ):
            raise TypeError("data_revision must be an integer")
        revision = int(self.data_revision)
        if revision < 0:
            raise ValueError("data_revision must be non-negative")
        record_fields = (
            ("channels", PulseChannel),
            ("blocks", PulseBlock),
            ("analog_traces", PulseAnalogTrace),
            ("scan_regions", PulseScanRegion),
            ("scan_dac_segments", PulseDacScanSegment),
            ("repeat_markers", PulseRepeatMarker),
        )
        for name, record_type in record_fields:
            records = tuple(getattr(self, name))
            if any(not isinstance(record, record_type) for record in records):
                raise TypeError(f"{name} must contain {record_type.__name__} values")
            object.__setattr__(self, name, records)
        object.__setattr__(self, "data_revision", revision)
        revisions = tuple(self.source_revisions) or (revision,)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in revisions
        ):
            raise TypeError("source_revisions must contain non-negative integers")
        object.__setattr__(self, "source_revisions", tuple(int(value) for value in revisions))


SelectorData: TypeAlias = SelectionData | PulseTimelineSelectionData


@dataclass(frozen=True, slots=True)
class SelectionSubject:
    """Which upstream quantities a selector's bounds cut.

    Resolved by the projection that emitted the event, because the answer is
    not a property of the selector: a histogram's x bounds cut the value
    quantity while a curve's cut the x coordinate, and a semantic edit moves
    either one under the operator's hands.  A consumer that asks the session
    afterwards can therefore be answered about a different projection than the
    one the operator drew on -- which is silent and wrong, so the answer
    travels with the event instead.

    ``None`` means the bounds do not cut a named upstream axis at all; a
    histogram's value quantity is the ordinary case.

    The scope is the canonical named panel/facet restriction.  A repeat is
    structural rather than named, so its resolved source row travels separately
    as repeat_index.  Coordinate frames travel beside x/y because selector
    values alone cannot identify their producer coordinate system.
    """

    plot_kind: PlotKind
    x: AxisRef | None
    y: AxisRef | None
    x_coordinate_frame: str | None = None
    y_coordinate_frame: str | None = None
    scope: tuple[tuple[AxisRef, object], ...] = ()
    repeat_index: int | None = None


@dataclass(frozen=True, slots=True)
class SelectionEvent:
    """One selector lifecycle event in canonical and display coordinates."""

    change: SelectionChange
    selector: SelectorState
    display_selector: SelectorState
    data_revision: int
    data_generation: str | None
    subject: SelectionSubject
    classifier_thresholds: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class _SelectionSubscription:
    callback: SelectionCallback
    selector_kind: SelectorKind | None


def _keeps_history(spec: object) -> bool:
    """Whether this drawing looks back over the session's past shots.

    History is the session's accumulation for one signal, so what decides
    whether it is kept is whether anything CONSUMES it -- a rolling trace
    along its x, a distribution pooled into its bins.  Gating retention on
    "is this a rolling plot" made the shots the property of one kind, and
    switching a panel to a distribution of the same signal threw them away.
    """

    from .specs import HistogramPlot, RollingPlot, semantic_spec

    return isinstance(semantic_spec(spec), (RollingPlot, HistogramPlot))


class PlotSession(FitSessionMixin, LiveSessionMixin, GestureSessionMixin):
    """One public plot surface with immutable input snapshots and public APIs.

    Data, display and layout revisions are independent. A fixed-size/layout
    edit is composed inside the existing Figure and promoted only after it has
    drawn successfully; the previously accepted front surface is
    untouched until then.
    """

    def __init__(
        self,
        data: PlotInput,
        spec: PlotSpec,
        *,
        size: str | None = None,
        parameters: Mapping[str, object] | None = None,
        defaults: PlotLibraryDefaults = DEFAULTS,
        unit_registry: UnitRegistry | None = None,
        device_pixel_ratio: float = 1.0,
        dispatch: HostDispatch | None = None,
        fit_engine: FitEngine | None = None,
    ) -> None:
        if not isinstance(defaults, PlotLibraryDefaults):
            raise TypeError("defaults must be PlotLibraryDefaults")
        data, initial_image_frame = self._split_image_frame(data, spec)
        FitProjection._validate_input(data, spec)
        self._lock = RLock()
        self._render_lock = RLock()
        self._ownership_gate = RLock()
        self._session_identity = object()
        self._closed = False
        self._defaults = defaults
        if unit_registry is not None and not isinstance(unit_registry, UnitRegistry):
            raise TypeError("unit_registry must be UnitRegistry or None")
        self._unit_registry = unit_registry
        self._spec = spec
        self._parameter_schema = parameter_schema_for(spec, style=defaults.style)
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        initial_parameters = {} if parameters is None else dict(parameters)
        deferred_fixed_limits: dict[str, object] | None = None
        if initial_parameters.get("relim_mode") == "fixed":
            for low_name, high_name in (
                ("color_min", "color_max"),
                ("y_min", "y_max"),
            ):
                if (
                    low_name in self._parameter_schema
                    and high_name in self._parameter_schema
                    and initial_parameters.get(low_name) is None
                    and initial_parameters.get(high_name) is None
                ):
                    deferred_fixed_limits = {
                        "relim_mode": "fixed",
                        low_name: None,
                        high_name: None,
                    }
                    initial_parameters.pop("relim_mode", None)
                    initial_parameters.pop(low_name, None)
                    initial_parameters.pop(high_name, None)
                    break
        self._display_store = DisplayStateStore(
            self._parameter_schema,
            initial_parameters,
        )
        self._size = (
            None if size is None else defaults.layout.validate_preset(size)
        )
        self._device_pixel_ratio = _validated_device_pixel_ratio(device_pixel_ratio)
        if dispatch is not None and not callable(dispatch):
            raise TypeError("dispatch must be callable or None")
        self._dispatch = dispatch
        self._presentation_dispatch: HostPresentationDispatch | None = None
        self._host_owner: object | None = None
        self._host_previous_dispatch: HostDispatch | None = None
        self._host_previous_presentation_dispatch: HostPresentationDispatch | None = None
        self._layout_revision = 0
        self._image_overlay = (
            None if initial_image_frame is None else initial_image_frame.overlay
        )
        self._renderer: MatplotlibRenderer | None = None
        self._surface_callbacks: list[SurfaceCallback] = []
        self._display_callbacks: list[DisplayCallback] = []
        self._viewport_callbacks: list[Callable[[object], object]] = []
        self._fit_callbacks: list[FitCallback] = []
        self._selection_subscriptions: list[_SelectionSubscription] = []
        self._selector_controller = _SelectorController()
        if fit_engine is not None and not isinstance(fit_engine, FitEngine):
            raise TypeError("fit_engine must be FitEngine or None")
        self._fit_engine = fit_engine or FitEngine()
        self._accepted_fit: _AcceptedFit | None = None
        self._classifier_results: tuple[FitResult | None, ...] = ()
        self._classifier_overlays = ()
        self._classifier_thresholds: tuple[float | None, ...] = ()
        self._classifier_gaussian_components: tuple[
            Mapping[str, float] | None, ...
        ] = ()
        self._analysis_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=_ANALYSIS_THREAD_PREFIX,
        )
        self._fit_cancel = Event()
        #: Request-scoped cancellation for live pair solves.  Set only on
        #: re-arm, replace_spec, and close: accepted FIFO pairs keep running
        #: under data pressure, so pairing survives solve > period.
        self._live_fit_cancel = Event()
        self._live_prepare_cancel = Event()
        self._live_prepare_future: Future[_PreparedLiveFrame] | None = None
        self._fit_context_generation = 0
        self._fit_request_generation = 0
        self._live_fit_request: _LiveFitRequest | None = None
        self._live_fit_future: Future[FitResult | FacetFitBatchResult] | None = None
        self._live_fit_completion: Future[FitResult | FacetFitBatchResult] | None = None
        self._fit_warm_starts: dict[tuple[int, str, int | None], _WarmSeed] = {}
        self._fit_batch_revision = 0
        self._viewport: RectangleRange | None = None
        self._history: tuple[RollingHistoryPoint, ...] = ()
        self._focused_facet_index: int | None = (
            0 if isinstance(spec, FacetGridPlot) else None
        )
        self._facet_focus_index: int | None = None
        self._gesture: _PointerGesture | None = None
        self._click_history: dict[int, tuple[float, float, float]] = {}
        initial_revision = snapshot_revision(data) if isinstance(data, OwnedSnapshot) else 0
        self._projection = FitProjection(
            data=data,
            revision=initial_revision,
            spec=self._spec,
            context=self._projection_context(),
            unit_registry=self._unit_registry,
            defaults=self._defaults,
            histogram_projection=None,
        )
        self._rebuild_projection()
        self._refresh_threshold_classifier()
        if _keeps_history(self._spec):
            self._history = self._projection.rolling_history
        self._presentation_epoch = 0
        # One configure unions existing owners' effects before one final paint.
        self._configuration_effects: RenderEffect | None = None
        self._configuration_display_events: list[DisplayState] | None = None
        self._configuration_fit_events: list[FitEvent | None] | None = None
        self._configuration_fit_commit_actions: list[Callable[[], None]] | None = None
        # Feasibility probes are cached per candidate spec for the current
        # dataset generation; see _semantic_feasibility.
        self._semantic_probe_generation: object = None
        self._semantic_probe_cache: dict[
            tuple[object, object], str | int | None
        ] = {}
        plan = self._resolve_plan()
        # Automatic sizing is an initial recommendation.  Once consumed, the
        # resulting named preset is authoritative just like a user selection.
        self._size = plan.preset
        renderer = MatplotlibRenderer(spec, plan, style=defaults.style)
        self._update_renderer(renderer, RenderEffect.LAYOUT)
        self._renderer = renderer
        if deferred_fixed_limits is not None:
            previous = self.display_state
            prepared = self._parameter_schema.prepare_updates(
                deferred_fixed_limits
            )
            self._materialize_fixed_limits(prepared, previous)
            candidate = self._parameter_schema._transition_prepared(
                previous.values,
                prepared,
            )
            self._display_store._commit_prepared(previous, candidate)

    @staticmethod
    def _split_image_frame(
        data: PlotInput,
        spec: PlotSpec,
    ) -> tuple[_ProjectionInput, ImageFrame | None]:
        if not isinstance(data, ImageFrame):
            return data, None
        if not isinstance(semantic_spec(spec), ImagePlot):
            raise TypeError("ImageFrame requires ImagePlot")
        return data.snapshot, data

    @staticmethod
    def _same_image_overlay(
        left: ImagePointOverlay,
        right: ImagePointOverlay,
    ) -> bool:
        def same_snapshot(
            first: OwnedSnapshot | None,
            second: OwnedSnapshot | None,
        ) -> bool:
            if first is None or second is None:
                return first is second
            return first.exactly_equals(second)

        return bool(
            left.revision == right.revision
            and left.point_ids == right.point_ids
            and left.labels == right.labels
            and left.static_statuses == right.static_statuses
            and same_snapshot(left.status, right.status)
            and np.array_equal(left.coordinates, right.coordinates)
        )

    @classmethod
    def _validate_image_frame_overlay(
        cls,
        previous: ImagePointOverlay | None,
        incoming: ImagePointOverlay,
    ) -> None:
        """Keep the point-layer revision monotonic across both update APIs."""

        if previous is None:
            return
        if incoming.revision < previous.revision:
            raise RevisionError(
                "ImageFrame overlay revision cannot move backwards"
            )
        if incoming.revision == previous.revision and not cls._same_image_overlay(
            previous,
            incoming,
        ):
            raise RevisionError(
                "one image overlay revision cannot identify different content"
            )

    def _projection_context(self) -> ProjectionContext:
        with self._lock:
            return ProjectionContext(
                display_state=self.display_state,
                selector_snapshot=self._selector_controller.snapshot(),
                viewport=self._viewport,
                focused_facet_index=self._focused_facet_index,
                rolling_history=self._history,
            )

    @property
    def _projected(self) -> FitProjection:
        """Return the current projection under one immutable session context."""

        return self._projection._with_context(self._projection_context())

    def _rebuild_projection(self, *, payload_only: bool = False) -> None:
        self._projection._reproject(
            context=self._projection_context(),
            payload_only=payload_only,
        )

    @property
    def _view(self) -> Any:
        return self._projection.view

    @property
    def _payload(self) -> Any:
        return self._projection.payload

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("plot session is closed")

    @property
    def spec(self) -> PlotSpec:
        return self._spec

    @property
    def _semantic_spec(self) -> PlotSpec:
        """The spec that decides what this session DRAWS.

        A FacetGrid delegates to its cell, so every per-plot facility -- the
        point overlay, the colour surface, the selector sources -- gates on
        this and never on the outer layout spec.
        """

        return semantic_spec(self._spec)

    @property
    def defaults(self) -> PlotLibraryDefaults:
        """Immutable configuration used by this session and its frontends."""

        return self._defaults

    @property
    def surface_plan(self) -> SurfacePlan:
        with self._render_lock:
            assert self._renderer is not None
            return self._renderer.plan

    def _raster_axes_snapshot(
        self,
    ) -> tuple[AxisTransform, ...]:
        """Return immutable axis geometry without exposing Matplotlib objects."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            return tuple(
                self._axis_transform_for_axis(axis)
                for axis in self._renderer.figure.axes
                if bool(axis.get_visible())
            )

    def _raster_source_revisions_snapshot(self) -> tuple[int, ...]:
        with self._render_lock:
            with self._lock:
                self._assert_open()
                revisions = tuple(getattr(self._payload, "source_revisions", ()))
                return tuple(int(value) for value in revisions) or (self.data_revision,)

    def _canonical_axes_limits(
        self,
        role: str,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if role == "distribution":
            values = tuple(
                self._projected._display_scalar_to_canonical(
                    value,
                    self._projected._value_quantity(),
                )
                if self._view is not None
                else float(value)
                for value in y_limits
            )
            return values, (0.0, 0.0)
        if self._view is None:
            return tuple(map(float, x_limits)), tuple(map(float, y_limits))
        x_values = tuple(
            self._display_x_scalar_to_canonical(value) for value in x_limits
        )
        y_values = (
            tuple(map(float, y_limits))
            if self._projected._is_histogram_plot()
            else tuple(
                self._projected._display_scalar_to_canonical(
                    value,
                    self._projected._y_ref_or_value(),
                )
                for value in y_limits
            )
        )
        return x_values, y_values

    def _axis_transform_for_axis(self, axis: Any) -> AxisTransform:
        assert self._renderer is not None
        width, height = canvas_physical_size(self._renderer.figure.canvas)
        bbox = axis.get_window_extent()
        role, separator, suffix = str(axis.get_gid() or "main").partition(":")
        cell_index = int(suffix) if separator and suffix.isdigit() else None
        display_x = tuple(map(float, axis.get_xlim()))
        display_y = tuple(map(float, axis.get_ylim()))
        canonical_x, canonical_y = self._canonical_axes_limits(
            role or "main",
            display_x,
            display_y,
        )
        return AxisTransform(
            role or "main",
            cell_index,
            (
                float(bbox.x0) / width,
                1.0 - float(bbox.y1) / height,
                float(bbox.x1) / width,
                1.0 - float(bbox.y0) / height,
            ),
            display_x,
            display_y,
            canonical_x,
            canonical_y,
        )

    def _axis_for_transform(self, transform: AxisTransform) -> Any | None:
        assert self._renderer is not None
        for axis in self._renderer.axes.get(transform.role, ()):
            gid = str(axis.get_gid() or "")
            _role, separator, suffix = gid.partition(":")
            cell_index = int(suffix) if separator and suffix.isdigit() else None
            if cell_index == transform.cell_index:
                return axis
        return None

    def _selector_axes(self, state: SelectorState) -> Any | None:
        """Resolve backend axes at the session/renderer boundary."""

        assert self._renderer is not None
        if isinstance(self._spec, FacetGridPlot):
            axes = self._renderer.axes.get("facet_cell", ())
            index = (
                self._focused_facet_index
                if state.facet_index is None
                else state.facet_index
            )
            if index is None or index < 0 or index >= len(axes):
                return None
            return axes[index]
        return self._renderer.primary_axes

    def _raster_capture_rgba(
        self,
        *,
        redraw: bool = False,
    ) -> np.ndarray:
        """Capture the worker-owned, already-composed canvas."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                return self._renderer.capture_rgba(redraw=redraw)

    def _raster_capture_rgba_bytes(self) -> tuple[bytes, int, int]:
        """The composed canvas as raw bytes, for a caller that wants bytes."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                return self._renderer.capture_rgba_bytes()

    def _raster_presentation_epoch(self) -> int:
        with self._lock:
            self._assert_open()
            return int(self._presentation_epoch)

    def _raster_interaction_snapshot(
        self,
    ) -> tuple[SelectorState, ...]:
        """Return the exact display-space selector state painted by this session."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            # Gesture candidates are baked into the raster alongside the
            # committed states, so the painted snapshot is always complete
            # and expressed in the same space as the painted axes.  An
            # overview used to answer with nothing, which is what left a
            # frontend unable to see -- and therefore unable to grab -- the
            # very selectors the grid had just painted in its cells.
            return self._painted_selector_snapshot().states

    def _raster_color_limits_snapshot(self) -> NumericRange | None:
        """Return the effective display-space clim painted into a raster front."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            if not self._renderer.axes.get("distribution"):
                return None
            try:
                low, high = self._renderer.resolved_color_limits()
            except TypeError:
                return None
            return NumericRange(*sorted((float(low), float(high))))

    def redraw_surface(self) -> None:
        """Rebuild the current canvas front after a host canvas is attached."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                self._renderer.draw()

    @property
    def parameter_schema(self) -> ParameterSchema:
        return self._parameter_schema

    @property
    def display_state(self) -> DisplayState:
        return self._display_store.state

    def _unit_parameter_sources(self) -> Mapping[str, Any]:
        if self._view is None:
            return MappingProxyType({})
        semantic = self._projected._semantic_spec()
        sources: dict[str, Any] = {
            "value_display_unit": self._projected._value_quantity(),
        }
        x_ref = getattr(semantic, "x", None)
        y_ref = getattr(semantic, "y", None)
        if isinstance(x_ref, AxisRef):
            sources["x_display_unit"] = self._projected._coordinate(x_ref)
        if isinstance(y_ref, AxisRef):
            sources["y_display_unit"] = self._projected._coordinate(y_ref)
        if isinstance(self._spec, FacetGridPlot):
            sources["facet_display_unit"] = self._projected._coordinate(
                self._spec.facet
            )
        return MappingProxyType(sources)

    def _parameter_choice_overrides(self) -> Mapping[str, tuple[object, ...]]:
        result: dict[str, tuple[object, ...]] = {}
        if self._view is None:
            if "x_display_unit" in self._parameter_schema:
                choices = self._parameter_schema["x_display_unit"].choices
                result["x_display_unit"] = tuple(map(str, choices))
            return MappingProxyType(result)
        registry = self._unit_registry or DEFAULT_UNITS
        for name, source in self._unit_parameter_sources().items():
            if name not in self._parameter_schema:
                continue
            compatible = []
            for symbol in registry.canonical_symbols():
                target = resolve_unit(symbol, registry)
                if source.canonical_unit.compatible_with(target):
                    compatible.append(symbol)
            result[name] = tuple(compatible)
        return MappingProxyType(result)

    def _current_display_limits(self) -> RectangleRange:
        assert self._renderer is not None
        axis = self._renderer.primary_axes
        x_low, x_high = sorted(map(float, axis.get_xlim()))
        y_low, y_high = sorted(map(float, axis.get_ylim()))
        return RectangleRange(
            self._viewport_x_from_axes(NumericRange(x_low, x_high)),
            NumericRange(y_low, y_high),
        )

    def describe_display(self) -> DisplayDescription:
        """Return one complete immutable snapshot for external controls."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            semantics = self.describe_semantics()
            return DisplayDescription(
                kind=self._spec.kind,
                size=self.surface_plan.preset,
                size_choices=self._defaults.layout.size_names,
                parameter_schema=self._parameter_schema,
                display_state=self.display_state,
                parameter_choices=self._parameter_choice_overrides(),
                limits=self._current_display_limits(),
                viewport=self._viewport,
                semantics=semantics,
                fit_models=self.fit_models,
            )

    def describe_semantics(self) -> SemanticDescription:
        """Return the registry-derived semantic edit domain for this session.

        Choice domains are checked against the live projection: an option
        whose edit the projection or layout would reject is delivered
        disabled with its rejection reason instead of failing on click.
        """

        with self._lock:
            self._assert_open()
            data = self._projection.data
            schema = snapshot_schema(data) if isinstance(data, OwnedSnapshot) else None
            spec = self._spec
            return describe_semantics(
                schema,
                spec,
                layout=self._defaults.layout,
                feasibility=self._semantic_feasibility,
            )

    def _semantic_feasibility(self, name: str, value: object) -> str | None:
        """Return why one semantic edit would be rejected, or None if viable.

        The probe runs validation only, never aggregation, and its cost is
        split along what each check actually depends on.  The EXPENSIVE
        candidate validation (spec composition, the projection constructor,
        the unit-aware view, the registry ``validate`` handler -- the same
        checks every DataView projection method runs) is a function of the
        dataset generation and the (current spec, candidate) pair alone, so
        it is cached on exactly that: a display-parameter edit invalidates
        NOTHING, and a describe after one re-pays zero validation sweeps.
        The cheap facet layout gate reads the LIVE surface inputs (size,
        DPR) against the cached facet cell count, so it is re-evaluated on
        every call instead of being folded into the key.  The cache is
        bounded because a long-lived UI can visit unbounded candidate
        combinations.
        """

        with self._lock:
            data = self._projection.data
            schema = snapshot_schema(data) if isinstance(data, OwnedSnapshot) else None
            try:
                candidate = updated_spec(schema, self._spec, name, value)
            except Exception as error:
                return str(error) or type(error).__name__
            if candidate == self._spec:
                return None
            generation = (
                snapshot_generation(data)
                if isinstance(data, OwnedSnapshot)
                else None
            )
            if generation != self._semantic_probe_generation:
                self._semantic_probe_generation = generation
                self._semantic_probe_cache = {}
            cache_key = (candidate, self._spec)
            cache = self._semantic_probe_cache
            if cache_key not in cache:
                result: str | int | None
                try:
                    result = self._validate_candidate_spec(candidate)
                except Exception as error:
                    result = str(error) or type(error).__name__
                if len(cache) >= _SEMANTIC_PROBE_CACHE_MAX:
                    cache.pop(next(iter(cache)))
                cache[cache_key] = result
            cached = cache[cache_key]
            if isinstance(cached, str):
                return cached
            if cached is not None:
                # The layout capacity is judged from the facet domain sizes;
                # the committed path derives the identical topology from its
                # payload.
                try:
                    resolve_surface(
                        self._size,
                        candidate.kind,
                        FacetTopology(cell_count=max(int(cached), 1)),
                        device_pixel_ratio=self._device_pixel_ratio,
                        layout=self._defaults.layout,
                        style=self._defaults.style,
                    )
                except Exception as error:
                    return str(error) or type(error).__name__
            return None

    def _validate_candidate_spec(self, spec: PlotSpec) -> int | None:
        """Run the replacement validation front without building any payload.

        Returns the facet cell count for FacetGrid candidates -- the one
        input the caller's per-call layout gate needs -- and None for every
        other kind.  The layout gate itself lives with the caller because it
        depends on live surface inputs (size, DPR) this validation must not
        be keyed on.
        """

        data = self._projection.data
        FitProjection._validate_input(data, spec)
        schema = parameter_schema_for(spec, style=self._defaults.style)
        old_state = self.display_state
        initial_state = replace_spec_initial_state(
            self._spec,
            spec,
            old_state.values,
            schema,
            size=self._size or self.surface_plan.preset,
            viewport=self._viewport,
            parameters=None,
        )
        display_store = DisplayStateStore(
            schema,
            initial_state.parameters,
            initial_revision=old_state.revision + 1,
        )
        projection = FitProjection(
            data=data,
            revision=self.data_revision,
            spec=spec,
            context=ProjectionContext(
                display_store.state,
                SelectorSnapshot(()),
                viewport=initial_state.viewport,
                focused_facet_index=0 if isinstance(spec, FacetGridPlot) else None,
            ),
            unit_registry=self._unit_registry,
            defaults=self._defaults,
            histogram_projection=None,
        )
        if not isinstance(data, OwnedSnapshot):
            return None
        projection._build_view()
        view = projection._view
        assert view is not None
        handler_for(spec).validate(view, spec)
        if isinstance(spec, FacetGridPlot):
            return int(view.facet_cell_count(spec))
        return None


    @property
    def selectors(self) -> tuple[SelectorState, ...]:
        """Return the effective immutable selector front."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            return self._resolved_selector_snapshot().states

    @property
    def facet_focus_index(self) -> int | None:
        """Presented FacetGrid cell, or ``None`` while showing the overview."""

        with self._lock:
            if not isinstance(self._spec, FacetGridPlot):
                return None
            return self._facet_focus_index

    def cancel_interaction(self) -> SelectorState | None:
        """Discard the transient selector candidate and release pointer state."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            return self._cancel_gesture()

    @property
    def last_fit(self) -> FitResult | FacetFitBatchResult | None:
        with self._lock:
            return None if self._accepted_fit is None else self._accepted_fit.result

    @property
    def fit_status(self) -> str | None:
        """Whether the painted fit matches the current data and fit context."""

        with self._lock:
            accepted = self._accepted_fit
            if accepted is None:
                return None
            current = (
                accepted.result.source_revision == self.data_revision
                and accepted.context_generation == self._fit_context_generation
            )
        return "current" if current else "lagging"

    @property
    def fit_models(self) -> tuple[FitModelSpec, ...]:
        """Semantically and dimensionally valid models for the painted plot."""

        with self._render_lock:
            target = self._projected._fit_target()
            if target is None:
                return ()
            return tuple(
                model
                for model in self._fit_engine.registry.models_for(target)
                if self._projected._fit_model_units_compatible(model)
            )

    def _resolve_fit_model(self, model: str | FitModelSpec) -> FitModelSpec:
        resolved = self._fit_engine.registry.get(model) if isinstance(model, str) else model
        if not isinstance(resolved, FitModelSpec):
            raise TypeError("model must be a registered model id or FitModelSpec")
        return resolved

    @property
    def data_revision(self) -> int:
        with self._lock:
            return self._projection.data_revision

    @property
    def data_generation(self) -> str | None:
        """Dataset generation behind the current frame, if the kind uses one."""

        with self._lock:
            data = self._projection.data
            return snapshot_generation(data) if isinstance(data, OwnedSnapshot) else None

    @property
    def revisions(self) -> SessionRevisions:
        with self._lock:
            return SessionRevisions(
                data=self.data_revision,
                display=self.display_state.revision,
                layout=self._layout_revision,
            )

    @property
    def source_revisions(self) -> tuple[int, ...]:
        """Ordered source revisions represented by the current projection."""

        return self._raster_source_revisions_snapshot()


    def _resolve_plan(self) -> SurfacePlan:
        return self._surface_plan_for(self._spec, self._payload, self.display_state)

    def _surface_plan_for(
        self,
        spec: PlotSpec,
        payload: Any,
        state: DisplayState,
    ) -> SurfacePlan:
        """Resolve the layout plan a (spec, payload) pair produces.

        Shared by the committed render path and the semantic feasibility
        probe, so layout rejections (the facet cell cap) come from one
        authority in both.
        """

        topology = None
        if isinstance(spec, FacetGridPlot):
            cell_count = len(tuple(getattr(payload, "cells", ())))
            topology = FacetTopology(
                cell_count=max(cell_count, 1),
                cell_aspect=(
                    self._drawn_image_aspect(payload)
                    if isinstance(semantic_spec(spec), ImagePlot)
                    else None
                ),
            )
        side_distribution = None
        if isinstance(spec, RollingPlot):
            side_distribution = bool(state["side_distribution"])
        return resolve_surface(
            self._size,
            spec.kind,
            topology,
            device_pixel_ratio=self._device_pixel_ratio,
            rolling_side_distribution=side_distribution,
            image_aspect=(
                self._drawn_image_aspect(payload)
                if isinstance(spec, ImagePlot)
                else None
            ),
            layout=self._defaults.layout,
            style=self._defaults.style,
        )

    def _drawn_image_aspect(self, payload: Any) -> float | None:
        """Return the shape the renderer will actually DRAW an image at.

        ``_image_coordinate_aspect`` is the one rule: when the two axes
        measure the same physical dimension the renderer pads the shorter
        span and the drawn box is square (1.0 here); when they do not it
        leaves Matplotlib in ``auto`` and the image fills whatever slot it is
        given (``None``, which the layout reads as "no preference").

        Asking a different question -- the pixel counts -- shaped the
        overview slots for a ratio nothing draws, leaving dead space around
        every cell.  One payload-level answer serves both surfaces: a facet
        of image cells asks it of a cell, a standalone image of itself.
        """

        cells = tuple(getattr(payload, "cells", ()))
        if cells:
            payload = getattr(cells[0], "payload", None)
        if payload is None or not hasattr(payload, "x") or not hasattr(payload, "y"):
            return 1.0
        return (
            1.0 if _image_coordinate_aspect(payload.x, payload.y) is not None else None
        )

    def _update_renderer(
        self,
        renderer: MatplotlibRenderer,
        effects: RenderEffect,
    ) -> None:
        deferred = self._configuration_effects
        if deferred is not None:
            self._configuration_effects = deferred | effects
            return
        gesture = self._gesture
        viewport = (
            gesture.candidate
            if isinstance(gesture, _PanGesture) and gesture.candidate is not None
            else self._projected.viewport
        )
        view_limits = None
        if viewport is not None:
            axes_x = self._viewport_x_to_axes(viewport.x)
            view_limits = (
                (axes_x.low, axes_x.high),
                self._viewport_y_to_axes(viewport.y),
            )
        classifier_thresholds = self._classifier_thresholds_for_render()
        display_classifier_thresholds = tuple(
            None
            if value is None
            else self._projected._canonical_scalar_to_display(
                value,
                self._projected._value_quantity(),
            )
            for value in classifier_thresholds
        )
        classifier_labels = self._classifier_labels(
            classifier_thresholds,
            display_classifier_thresholds,
        )
        with renderer.raster_transaction():
            renderer.present(RenderFrame(
                payload=self._payload,
                state=self.display_state,
                effects=effects,
                data_revision=self.data_revision,
                fit_overlays=(
                    ()
                    if self._accepted_fit is None
                    else self._accepted_fit.overlays
                ),
                fit_model_id=(
                    None
                    if self._accepted_fit is None
                    else str(self._accepted_fit.result.model.model_id)
                ),
                classifier_overlays=self._classifier_overlays,
                classifier_thresholds=display_classifier_thresholds,
                classifier_labels=classifier_labels,
                image_overlay=self._image_overlay,
                selectors=self._painted_selector_snapshot(),
                facet_index=self._focused_facet_index,
                facet_focus_index=self._facet_focus_index,
                view_limits=view_limits,
            ))
        self._presentation_epoch += 1
        # The single surface-commit notification point.  Every present lands
        # here, so no mutation path can forget to notify (the selector
        # install path historically had no site-local notification and left
        # raster hosts on stale fronts).  Observers run on the render thread
        # and must be non-blocking.
        self._notify_surface_callbacks(tuple(self._surface_callbacks))

    def _render_current(
        self,
        effects: RenderEffect,
        *,
        schedule_fit: bool = False,
    ) -> None:
        with self._render_lock:
            assert self._renderer is not None
            self._update_renderer(self._renderer, effects)

    def _present_projection_transaction(
        self,
        projection: FitProjection,
        *,
        image_overlay: ImagePointOverlay | None,
        accepted_fit: _AcceptedFit | None,
    ) -> _ProjectionPresentation:
        """Swap one complete projected frame, restoring the old frame on failure."""

        if not isinstance(projection, FitProjection):
            raise TypeError("projection must be FitProjection")
        old_plan = self.surface_plan
        old_count = (
            old_plan.facet_topology.cell_count
            if isinstance(self._spec, FacetGridPlot)
            else None
        )
        new_count = (
            len(tuple(getattr(projection.payload, "cells", ())))
            if isinstance(self._spec, FacetGridPlot)
            else None
        )
        if old_count != new_count:
            self._cancel_gesture()
        with self._lock:
            previous = (
                self._projection,
                self._image_overlay,
                self._accepted_fit,
                self._classifier_results,
                self._classifier_overlays,
                self._classifier_thresholds,
                self._classifier_gaussian_components,
                self._focused_facet_index,
                self._facet_focus_index,
                self._viewport,
                self._history,
                self._layout_revision,
            )
            self._projection = projection
            if _keeps_history(self._spec):
                self._history = projection.rolling_history
            self._image_overlay = image_overlay
            self._accepted_fit = accepted_fit
            self._refresh_threshold_classifier()
            if isinstance(self._spec, FacetGridPlot):
                assert new_count is not None
                self._clamp_facet_state(new_count)
                plan = self._resolve_plan() if old_count != new_count else None
            else:
                plan = None
            if plan is not None:
                self._layout_revision += 1
        try:
            if plan is not None:
                self._apply_layout_plan(
                    plan,
                    schedule_fit=False,
                )
            else:
                self._render_current(
                    RenderEffect.BASE_GEOMETRY,
                    schedule_fit=False,
                )
        except Exception:
            with self._lock:
                (
                    self._projection,
                    self._image_overlay,
                    self._accepted_fit,
                    self._classifier_results,
                    self._classifier_overlays,
                    self._classifier_thresholds,
                    self._classifier_gaussian_components,
                    self._focused_facet_index,
                    self._facet_focus_index,
                    self._viewport,
                    self._history,
                    self._layout_revision,
                ) = previous
            try:
                if plan is not None:
                    self._apply_layout_plan(
                        old_plan,
                        schedule_fit=False,
                    )
                else:
                    self._render_current(
                        RenderEffect.BASE_GEOMETRY,
                        schedule_fit=False,
                    )
            except Exception:
                self.redraw_surface()
            raise
        return _ProjectionPresentation(
            projection,
            *previous,
            old_plan,
        )

    def _abort_projection_presentation(
        self,
        presentation: _ProjectionPresentation,
    ) -> None:
        """Restore a drawn projection that never reached the frontend."""

        if not isinstance(presentation, _ProjectionPresentation):
            raise TypeError("presentation must be a projection presentation")
        with self._render_lock:
            with self._lock:
                if self._projection is not presentation.committed_projection:
                    raise RuntimeError(
                        "projection presentation is no longer current"
                    )
                current_plan = self.surface_plan
                self._projection = presentation.previous_projection
                self._image_overlay = presentation.previous_image_overlay
                self._accepted_fit = presentation.previous_accepted_fit
                self._classifier_results = presentation.previous_classifier_results
                self._classifier_overlays = presentation.previous_classifier_overlays
                self._classifier_thresholds = presentation.previous_classifier_thresholds
                self._classifier_gaussian_components = (
                    presentation.previous_classifier_gaussian_components
                )
                self._focused_facet_index = (
                    presentation.previous_focused_facet_index
                )
                self._facet_focus_index = presentation.previous_facet_focus_index
                self._viewport = presentation.previous_viewport
                self._history = presentation.previous_rolling_history
                self._layout_revision = presentation.previous_layout_revision
            if current_plan != presentation.previous_plan:
                self._apply_layout_plan(
                    presentation.previous_plan,
                    schedule_fit=False,
                )
            else:
                self._render_current(
                    RenderEffect.BASE_GEOMETRY,
                    schedule_fit=False,
                )

    def _apply_layout_plan(
        self,
        plan: SurfacePlan,
        *,
        schedule_fit: bool = False,
    ) -> None:
        with self._render_lock:
            self._cancel_gesture()
            assert self._renderer is not None
            renderer = self._renderer
            renderer.relayout(
                plan,
                facet_index=self._focused_facet_index,
                facet_focus_index=self._facet_focus_index,
            )
            self._update_renderer(renderer, RenderEffect.LAYOUT)
            with self._lock:
                self._assert_open()

    def set_parameter(self, name: str, value: object) -> DisplayState:
        return self.set_parameters({name: value})

    def configure(
        self,
        *,
        data: PlotInput | object = _UNSET,
        semantic: Mapping[str, object] | None = None,
        parameters: Mapping[str, object] | None = None,
        parameter_updates: Mapping[str, object] | None = None,
        size: str | None = None,
        image_overlay: ImagePointOverlay | None | object = _UNSET,
        classifier_thresholds: object = _UNSET,
        selectors: Sequence[SelectorState] | object = _UNSET,
        viewport: RectangleRange | None | object = _UNSET,
        facet_focus: int | None | object = _UNSET,
        fit: Mapping[str, object] | None | object = _UNSET,
        fit_live: bool = True,
    ) -> DisplayDescription:
        """Apply one target once; an identical target does no work.

        A complete parameter target may carry its current authored delta so
        transition normalization stays exact even when queued targets coalesce.
        """

        if selectors is not _UNSET:
            if isinstance(selectors, (str, bytes)):
                raise TypeError("selectors must be a sequence of SelectorState")
            selector_target = SelectorSnapshot(tuple(selectors)).committed
        else:
            selector_target = None
        threshold_target = (
            _UNSET
            if classifier_thresholds is _UNSET
            else normalize_classifier_threshold_targets(classifier_thresholds)
        )
        if viewport is not _UNSET and viewport is not None and not isinstance(
            viewport, RectangleRange
        ):
            raise TypeError("viewport must be RectangleRange, None, or omitted")
        if facet_focus is not _UNSET and facet_focus is not None:
            if isinstance(facet_focus, bool) or not isinstance(facet_focus, int):
                raise TypeError("facet_focus must be an integer, None, or omitted")
            if facet_focus < 0:
                raise ValueError("facet_focus must be non-negative")
        if fit is not _UNSET and fit is not None and not isinstance(fit, Mapping):
            raise TypeError("fit must be a mapping, None, or omitted")
        if type(fit_live) is not bool:
            raise TypeError("fit_live must be bool")
        selected_data = data
        selected_overlay = _UNSET
        if data is not _UNSET:
            selected_data, image_frame = self._split_image_frame(data, self._spec)
            selected_overlay = _UNSET if image_frame is None else image_frame.overlay
            FitProjection._validate_input(selected_data, self._spec)

        with self._render_lock:
            if self._configuration_effects is not None:
                raise RuntimeError("plot configuration is already in progress")
            previous_state = self._configuration_state_snapshot()
            self._configuration_effects = RenderEffect.NONE
            self._configuration_display_events = []
            self._configuration_fit_events = []
            self._configuration_fit_commit_actions = []
            try:
                self._apply_configuration(
                    semantic=semantic,
                    parameters=parameters,
                    parameter_updates=parameter_updates,
                    size=size,
                    image_overlay=image_overlay,
                    classifier_thresholds=threshold_target,
                )
                if selected_data is not _UNSET:
                    assert isinstance(selected_data, (OwnedSnapshot, PulseTimelineData))
                    current_data = self._projection.data
                    if isinstance(selected_data, OwnedSnapshot):
                        if not isinstance(current_data, OwnedSnapshot):
                            raise TypeError("indexed data cannot replace a timeline")
                        current_schema = snapshot_schema(current_data)
                        selected_schema = snapshot_schema(selected_data)
                        if not indexed_schemas_compatible(
                            current_schema, selected_schema
                        ):
                            raise ValueError(
                                "configuration data requires one indexed Dataset layout"
                            )
                        current_revision = snapshot_revision(current_data)
                        selected_revision = snapshot_revision(selected_data)
                        if selected_revision < current_revision:
                            raise ValueError("configuration data revision moved backwards")
                        if (
                            selected_revision == current_revision
                            and selected_data.ref == current_data.ref
                        ):
                            selected_data = _UNSET
                    else:
                        raise TypeError(
                            "same-publication configuration data requires an indexed Dataset"
                        )
                if selected_data is not _UNSET:
                    projection = self._projection._fork_frozen(
                        data=selected_data,
                        revision=snapshot_revision(selected_data),
                        context=self._projection_context(),
                    )
                    projection._build_view_and_payload()
                    self._present_projection_transaction(
                        projection,
                        image_overlay=(
                            self._image_overlay
                            if selected_overlay is _UNSET
                            else selected_overlay
                        ),
                        accepted_fit=self._accepted_fit,
                    )
                if facet_focus is not _UNSET:
                    if facet_focus is None:
                        if self._facet_focus_index is not None:
                            self.show_facet_overview()
                    elif facet_focus != self._facet_focus_index:
                        self.focus_facet(facet_focus)
                if selector_target is not None:
                    managed = {
                        SelectorKind.X_RANGE,
                        SelectorKind.AREA,
                        SelectorKind.THRESHOLD,
                    }
                    current = {
                        state.kind: state
                        for state in self._selector_controller.states()
                        if state.kind in managed
                    }
                    wanted = {
                        state.kind: state
                        for state in selector_target
                        if state.kind in managed
                    }
                    for kind in managed:
                        before = current.get(kind)
                        after = wanted.get(kind)
                        if after is None:
                            if before is not None:
                                self.remove_selector(kind, emit_change=False)
                            continue
                        if (
                            before is None
                            or before.value != after.value
                            or before.facet_index != after.facet_index
                        ):
                            self._install_selector_state(
                                after,
                                emit_change=False,
                            )
                if viewport is not _UNSET:
                    self._set_viewport_state(viewport, emit_change=False)
                if fit is not _UNSET:
                    self._configure_fit_target(
                        {} if fit is None else fit,
                        live=fit_live,
                    )
                effects = self._configuration_effects
                display_events = tuple(self._configuration_display_events or ())
                fit_events = tuple(self._configuration_fit_events or ())
                self._configuration_effects = None
                self._configuration_display_events = None
                self._configuration_fit_events = None
                if effects != RenderEffect.NONE:
                    self._render_current(effects, schedule_fit=False)
                fit_commit_actions = tuple(
                    self._configuration_fit_commit_actions or ()
                )
                self._configuration_fit_commit_actions = None
                description = self.describe_display()
            except BaseException:
                self._restore_configuration_state(previous_state)
                self._configuration_effects = None
                self._configuration_display_events = None
                self._configuration_fit_events = None
                self._configuration_fit_commit_actions = None
                raise

        for action in fit_commit_actions:
            action()
        if display_events:
            self._notify_display(display_events[-1])
        for event in fit_events:
            self._notify_fit(event)
        return description

    def _configuration_state_snapshot(self) -> dict[str, object]:
        assert self._renderer is not None
        snapshot = {name: getattr(self, name) for name in _CONFIGURATION_STATE_NAMES}
        snapshot.update({
            "_display_store": self._display_store,
            "display_state": self.display_state,
            "selector_states": self._selector_controller.states(),
            "fit_warm_starts": dict(self._fit_warm_starts),
            "renderer_plan": self._renderer.plan,
        })
        return snapshot

    def _restore_configuration_state(self, snapshot: Mapping[str, object]) -> None:
        display_store = snapshot["_display_store"]
        display_state = snapshot["display_state"]
        current_store = self._display_store
        if current_store is display_store and current_store.state is not display_state:
            current_store._restore_prepared(current_store.state, display_state)
        selector_controller = _SelectorController()
        for state in tuple(snapshot["selector_states"]):
            selector_controller.install(state)
        for name in _CONFIGURATION_STATE_NAMES:
            setattr(self, name, snapshot[name])
        self._display_store = display_store
        self._selector_controller = selector_controller
        self._fit_warm_starts = dict(snapshot["fit_warm_starts"])
        assert self._renderer is not None
        self._renderer.spec = self._spec
        self._renderer.relayout(
            snapshot["renderer_plan"],
            facet_index=self._focused_facet_index,
            facet_focus_index=self._facet_focus_index,
        )

    def _apply_configuration(
        self,
        *,
        semantic: Mapping[str, object] | None = None,
        parameters: Mapping[str, object] | None = None,
        parameter_updates: Mapping[str, object] | None = None,
        size: str | None = None,
        image_overlay: ImagePointOverlay | None | object = _UNSET,
        classifier_thresholds: object = _UNSET,
    ) -> DisplayDescription:
        """Apply semantic/display/layout state inside ``configure``.

        The caller supplies state, not a render strategy.  Semantic choices are
        composed in memory, display values are differenced as one mapping, and
        size/overlay effects join the same renderer update.
        """

        if semantic is not None and not isinstance(semantic, Mapping):
            raise TypeError("semantic must be a mapping or None")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        if parameter_updates is not None and not isinstance(
            parameter_updates,
            Mapping,
        ):
            raise TypeError("parameter_updates must be a mapping or None")
        semantic_values = {} if semantic is None else dict(semantic)
        display_values = {} if parameters is None else dict(parameters)
        with self._lock:
            self._assert_open()
            data = self._projection.data
            schema = snapshot_schema(data) if isinstance(data, OwnedSnapshot) else None
            # One bag, one composition: a configuration is a description of
            # the finished plot, not a sequence of gestures to replay.
            candidate_spec = composed_spec(
                schema,
                self._spec,
                {str(name): value for name, value in semantic_values.items()},
            )
            spec_changed = candidate_spec != self._spec

        if spec_changed:
            description = self.replace_spec(
                candidate_spec,
                parameters=display_values,
                size=size,
                image_overlay=image_overlay,
                classifier_thresholds=classifier_thresholds,
            )
        else:
            self._set_configuration_values(
                display_values,
                authored_values=parameter_updates,
                size=_UNSET if size is None else size,
                image_overlay=image_overlay,
                classifier_thresholds=classifier_thresholds,
            )
            description = self.describe_display()

        return description

    def set_labels(
        self,
        *,
        title: str | None | object = _UNSET,
        x: str | None | object = _UNSET,
        y: str | None | object = _UNSET,
        value: str | None | object = _UNSET,
    ) -> DisplayState:
        """Update visible text artists without rebuilding data artists.

        ``None`` resets title/axis/value text to the PlotSpec/data-declared
        automatic value.  Use an empty string to hide text deliberately.
        """

        updates: dict[str, object] = {}
        for name, selected in (
            ("title", title),
            ("x_label", x),
            ("y_label", y),
            ("value_label", value),
        ):
            if selected is _UNSET:
                continue
            updates[name] = selected
        if not updates:
            return self.display_state
        return self.set_parameters(updates)

    def _prepare_value_unit_ranges(
        self,
        prepared: dict[str, object],
        previous: DisplayState,
        authored_names: frozenset[str],
    ) -> None:
        if (
            self._view is None
            or not isinstance(self._projection.data, OwnedSnapshot)
            or "value_display_unit" not in prepared
            or prepared["value_display_unit"] == previous.values.get("value_display_unit")
        ):
            return
        if (
            prepared.get("relim_mode") != "fixed"
            and "relim_mode" in prepared
        ):
            return
        semantic = self._projected._semantic_spec()
        if isinstance(semantic, (CurvePlot, RollingPlot)):
            range_names = ("y_min", "y_max")
        elif isinstance(semantic, ImagePlot):
            range_names = ("color_min", "color_max")
        else:
            return
        selected_unit = prepared["value_display_unit"]
        target_unit = (
            schema_value_unit(snapshot_schema(self._projection.data), self._unit_registry or DEFAULT_UNITS)
            if selected_unit is None
            else resolve_unit(selected_unit, self._unit_registry)
        )
        source_unit = self._projected._value_quantity().display_unit
        for name in range_names:
            if name not in self._parameter_schema:
                continue
            if name in authored_names:
                continue
            current = prepared.get(name, previous.values.get(name))
            if current is None:
                continue
            converted = source_unit.convert_value_to((float(current),), target_unit)
            prepared[name] = float(np.asarray(converted).reshape(-1)[0])

    def _materialize_fixed_limits(
        self,
        prepared: dict[str, object],
        previous: DisplayState,
    ) -> None:
        if "relim_mode" not in self._parameter_schema:
            return
        mode = prepared.get("relim_mode", previous.values["relim_mode"])
        if mode != "fixed":
            return
        candidate = dict(previous.values)
        candidate.update(prepared)
        assert self._renderer is not None
        for low_name in self._parameter_schema.names:
            if not low_name.endswith("_min"):
                continue
            high_name = f"{low_name[:-4]}_max"
            if high_name not in self._parameter_schema:
                continue
            if candidate[low_name] is not None and candidate[high_name] is not None:
                continue
            if low_name == "color_min":
                low, high = sorted(self._renderer.resolved_color_limits())
            else:
                low, high = sorted(map(float, self._renderer.primary_axes.get_ylim()))
            if candidate[low_name] is None:
                prepared[low_name] = low
            if candidate[high_name] is None:
                prepared[high_name] = high

    def _validate_projection_unit_updates(
        self,
        prepared: Mapping[str, object],
    ) -> None:
        """Resolve unit changes before committing display state."""

        if self._view is None:
            return
        for name, quantity in self._unit_parameter_sources().items():
            if name not in prepared or prepared[name] is None:
                continue
            target = resolve_unit(prepared[name], self._unit_registry)
            if not quantity.canonical_unit.compatible_with(target):
                raise ValueError(
                    f"{name} {target.symbol!r} is incompatible with "
                    f"{quantity.canonical_unit.symbol!r}"
                )

    def _invalidate_fit_context(self) -> int:
        """Mark a non-data fit context as stale without starting a fit.

        A live fit is a data-frame operation.  Selector, viewport, unit, and
        layout changes may make the painted result lag the current interaction,
        but they must never launch another solve (or cancel the already
        admitted data-frame solve).  The current accepted overlay therefore
        remains the stable front until either a newer data revision is
        committed or the caller explicitly requests/clears a fit.
        """
        with self._lock:
            self._fit_context_generation += 1
            # This event belongs to an explicit/ad-hoc fit submitted through
            # fit()/fit_async().  The clock cancellation token belongs to an
            # already admitted live data frame and is deliberately untouched.
            self._commit_fit_actions(self._fit_cancel.set)
            return self._fit_context_generation

    def _clear_fit_presentation(self) -> bool:
        """Clear the one accepted result/selection/overlay state group."""

        changed = self._accepted_fit is not None
        self._accepted_fit = None
        return changed

    def _refresh_accepted_fit_overlays(
        self,
        accepted: _AcceptedFit,
    ) -> _AcceptedFit:
        """Repaint fit overlays in the current display units without solving."""

        projection = self._projected
        if isinstance(accepted.result, FacetFitBatchResult):
            refreshed: list[object] = []
            for index, (result, selection, previous) in enumerate(
                zip(
                    accepted.result.results,
                    accepted.selections,
                    accepted.overlays,
                    strict=True,
                )
            ):
                if result is None or selection is None:
                    refreshed.append(previous)
                    continue
                cell_projection = projection._with_context(
                    replace(projection._context, focused_facet_index=index)
                )
                refreshed.append(
                    cell_projection._make_fit_overlay(result, selection)
                )
            return replace(
                accepted,
                overlay=None,
                overlays=tuple(refreshed),
            )
        if accepted.selection is None:
            raise RuntimeError("single fit acceptance has no selection")
        return replace(
            accepted,
            overlay=projection._make_fit_overlay(
                accepted.result,
                accepted.selection,
            ),
        )

    def set_parameters(self, values: Mapping[str, object]) -> DisplayState:
        """Apply one complete display mapping through the effect-owned path."""

        return self._set_configuration_values(values)

    def _set_configuration_values(
        self,
        values: Mapping[str, object],
        *,
        authored_values: Mapping[str, object] | None = None,
        size: str | object = _UNSET,
        image_overlay: ImagePointOverlay | None | object = _UNSET,
        classifier_thresholds: object = _UNSET,
    ) -> DisplayState:
        """Commit display, layout and Image overlay with one renderer update."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
                prepared = self._parameter_schema.prepare_updates(values)
                authored = (
                    prepared
                    if authored_values is None
                    else self._parameter_schema.prepare_updates(authored_values)
                )
                authored_names = frozenset(
                    name for name, value in authored.items() if value is not None
                )
                previous = self.display_state
                self._validate_projection_unit_updates(prepared)
                self._materialize_fixed_limits(prepared, previous)
                if authored_values is not None:
                    transition_values = dict(authored)
                    if transition_values.get("relim_mode") == "fixed":
                        for low_name in self._parameter_schema.names:
                            if not low_name.endswith("_min"):
                                continue
                            high_name = f"{low_name[:-4]}_max"
                            if high_name not in self._parameter_schema:
                                continue
                            if transition_values.get(low_name) is None:
                                transition_values[low_name] = prepared[low_name]
                            if transition_values.get(high_name) is None:
                                transition_values[high_name] = prepared[high_name]
                    authored_candidate = self._parameter_schema._transition_prepared(
                        previous.values,
                        transition_values,
                    )
                    for name in self._parameter_schema.names:
                        if (
                            name in transition_values
                            or authored_candidate[name] != previous[name]
                        ):
                            prepared[name] = authored_candidate[name]
                self._prepare_value_unit_ranges(
                    prepared,
                    previous,
                    authored_names,
                )
                candidate = self._parameter_schema._transition_prepared(
                    previous.values,
                    prepared,
                )
                accepted_changes = frozenset(
                    name
                    for name in self._parameter_schema.names
                    if previous.values[name] != candidate[name]
                )
                parameter_effects = self._parameter_schema.effects_for(
                    accepted_changes
                ) if accepted_changes else RenderEffect.NONE
                selected_size = (
                    self.surface_plan.preset
                    if size is _UNSET
                    else self._defaults.layout.validate_preset(size)
                )
                size_changed = (
                    size is not _UNSET
                    and selected_size != self.surface_plan.preset
                )
                previous_overlay = self._image_overlay
                if image_overlay is _UNSET:
                    selected_overlay = previous_overlay
                    overlay_changed = False
                else:
                    if image_overlay is not None and not isinstance(
                        image_overlay, ImagePointOverlay
                    ):
                        raise TypeError(
                            "image_overlay must be ImagePointOverlay or None"
                        )
                    selected_overlay = image_overlay
                    if selected_overlay is not None:
                        if not isinstance(self._semantic_spec, ImagePlot):
                            raise TypeError("image overlay requires ImagePlot")
                        self._validate_image_frame_overlay(
                            previous_overlay,
                            selected_overlay,
                        )
                    overlay_changed = (
                        previous_overlay is not selected_overlay
                        and (
                            previous_overlay is None
                            or selected_overlay is None
                            or not self._same_image_overlay(
                                previous_overlay,
                                selected_overlay,
                            )
                        )
                    )
                effects = parameter_effects
                if size_changed:
                    effects |= RenderEffect.LAYOUT
                if overlay_changed:
                    effects |= RenderEffect.OVERLAY
                thresholds_changed = (
                    classifier_thresholds is not _UNSET
                    and tuple(classifier_thresholds)
                    != self._classifier_threshold_targets_state()
                )
                if thresholds_changed:
                    effects |= RenderEffect.OVERLAY
                if effects == RenderEffect.NONE:
                    return previous
                unit_affecting = bool(
                    parameter_effects & RenderEffect.VIEW_PROJECTION
                )
                canonical_viewport = (
                    self._projected._viewport_in_canonical()
                    if unit_affecting
                    and self._viewport is not None
                    and self._view is not None
                    else None
                )
                if effects & (
                    RenderEffect.INTERACTION_REPROJECT | RenderEffect.LAYOUT
                ):
                    self._cancel_gesture()
                old_plan = self.surface_plan
                previous_projection = self._projection._with_context(
                    self._projection_context()
                )
                previous_values = (
                    self._viewport,
                    self._accepted_fit,
                    self._classifier_results,
                    self._classifier_overlays,
                    self._classifier_thresholds,
                    self._classifier_gaussian_components,
                    self._fit_context_generation,
                    self._layout_revision,
                    self._size,
                    self._image_overlay,
                )
                state = (
                    self._display_store._commit_prepared(previous, candidate)
                    if accepted_changes
                    else previous
                )
                fit_cancel: Event | None = None
                layout_attempted = False
                try:
                    changed = accepted_changes
                    if (
                        isinstance(self._spec, PulseTimelinePlot)
                        and "x_display_unit" in changed
                        and self._viewport is not None
                    ):
                        assert isinstance(self._projection.data, PulseTimelineData)
                        old_factor, _old_unit = pulse_time_scale(
                            self._projection.data,
                            previous.values.get("x_display_unit"),
                        )
                        new_factor, _new_unit = pulse_time_scale(
                            self._projection.data,
                            state.values.get("x_display_unit"),
                        )
                        source_x = NumericRange(
                            self._viewport.x.low / old_factor,
                            self._viewport.x.high / old_factor,
                        )
                        self._viewport = RectangleRange(
                            NumericRange(
                                source_x.low * new_factor,
                                source_x.high * new_factor,
                            ),
                            self._viewport.y,
                        )
                    unit_projection_changed = bool(
                        effects & RenderEffect.VIEW_PROJECTION
                    )
                    payload_projection_changed = bool(
                        effects & RenderEffect.PAYLOAD_PROJECTION
                    )
                    fit_selection_changed = bool(
                        effects & RenderEffect.FIT_SELECTION
                    )
                    if fit_selection_changed:
                        self._fit_context_generation += 1
                        fit_cancel = self._fit_cancel
                    if unit_projection_changed:
                        self._rebuild_projection()
                        if canonical_viewport is not None:
                            self._viewport = self._viewport_from_canonical(
                                canonical_viewport
                            )
                    elif payload_projection_changed:
                        self._rebuild_projection(payload_only=True)
                    if unit_affecting and self._accepted_fit is not None:
                        self._accepted_fit = self._refresh_accepted_fit_overlays(
                            self._accepted_fit
                        )
                    if size_changed:
                        self._size = selected_size
                    if overlay_changed:
                        self._image_overlay = selected_overlay
                    classifier_changed = bool(
                        "threshold_classifier" in accepted_changes
                        or (
                            self._threshold_classifier_enabled()
                            and parameter_effects
                            & (
                                RenderEffect.VIEW_PROJECTION
                                | RenderEffect.PAYLOAD_PROJECTION
                                | RenderEffect.FIT_SELECTION
                            )
                        )
                    )
                    if classifier_changed:
                        self._refresh_threshold_classifier()
                    if thresholds_changed:
                        self._set_classifier_thresholds_state(classifier_thresholds)
                    plan = (
                        self._resolve_plan()
                        if effects & RenderEffect.LAYOUT
                        else None
                    )
                    if plan is not None:
                        self._layout_revision += 1
                except Exception:
                    if accepted_changes:
                        self._display_store._restore_prepared(state, previous)
                    self._projection = previous_projection
                    (
                        self._viewport,
                        self._accepted_fit,
                        self._classifier_results,
                        self._classifier_overlays,
                        self._classifier_thresholds,
                        self._classifier_gaussian_components,
                        self._fit_context_generation,
                        self._layout_revision,
                        self._size,
                        self._image_overlay,
                    ) = previous_values
                    raise
            try:
                if plan is not None:
                    layout_attempted = True
                    self._apply_layout_plan(
                        plan,
                        schedule_fit=False,
                    )
                else:
                    self._render_current(
                        effects,
                        schedule_fit=False,
                    )
            except Exception:
                with self._lock:
                    if accepted_changes:
                        self._display_store._restore_prepared(state, previous)
                    self._projection = previous_projection
                    (
                        self._viewport,
                        self._accepted_fit,
                        self._classifier_results,
                        self._classifier_overlays,
                        self._classifier_thresholds,
                        self._classifier_gaussian_components,
                        self._fit_context_generation,
                        self._layout_revision,
                        self._size,
                        self._image_overlay,
                    ) = previous_values
                try:
                    if layout_attempted or self.surface_plan != old_plan:
                        self._apply_layout_plan(
                            old_plan,
                            schedule_fit=False,
                        )
                    else:
                        self._render_current(
                            RenderEffect.LAYOUT,
                            schedule_fit=False,
                        )
                except Exception:
                    self.redraw_surface()
                raise
            if fit_cancel is not None:
                self._commit_fit_actions(fit_cancel.set)
        if accepted_changes:
            self._notify_display(state)
        return state

    def _prepare_replacement(
        self,
        spec: PlotSpec,
        parameters: Mapping[str, object] | None,
        size: str,
    ) -> tuple[Any, Any, DisplayStateStore, int | None, FitProjection]:
        """Build and validate everything a spec replacement would commit.

        Shared by ``replace_spec`` and the semantic feasibility probe so a
        candidate is judged by exactly the validation the real replacement
        runs.  Nothing here mutates session state.
        """

        data = self._projection.data
        FitProjection._validate_input(data, spec)
        old_state = self.display_state
        schema = parameter_schema_for(spec, style=self._defaults.style)
        initial_state = replace_spec_initial_state(
            self._spec,
            spec,
            old_state.values,
            schema,
            size=size,
            viewport=self._viewport,
            parameters=parameters,
        )
        display_store = DisplayStateStore(
            schema,
            initial_state.parameters,
            initial_revision=old_state.revision + 1,
        )
        focused = 0 if isinstance(spec, FacetGridPlot) else None
        # History belongs to the SIGNAL this session is showing, so it
        # survives a spec replacement -- switching a panel from a rolling
        # trace to a distribution of the same shots must not throw the shots
        # away.  What does not survive is a change to what one history point
        # IS: a rolling trace whose group or reduction changed reseeds,
        # because its stored samples were reduced the old way.
        retained_history: tuple[RollingHistoryPoint, ...] = ()
        if _keeps_history(spec) and _keeps_history(self._spec):
            same_sample = not (
                isinstance(spec, RollingPlot)
                and isinstance(self._spec, RollingPlot)
                and (
                    spec.group != self._spec.group
                    or spec.reduction != self._spec.reduction
                )
            )
            if same_sample:
                retained_history = self._history
        projection = FitProjection(
            data=data,
            revision=self.data_revision,
            spec=spec,
            context=ProjectionContext(
                display_store.state,
                SelectorSnapshot(()),
                viewport=initial_state.viewport,
                focused_facet_index=focused,
                rolling_history=retained_history,
            ),
            unit_registry=self._unit_registry,
            defaults=self._defaults,
            histogram_projection=None,
        )
        projection._build_view_and_payload()
        return schema, initial_state, display_store, focused, projection

    def apply_semantic(self, name: str, value: object) -> DisplayDescription:
        """Apply one semantic edit -- a kind, an axis role, a reduction.

        The composition lives here because both of its inputs already do: the
        current spec and the schema of the data being drawn.  Every embedder
        used to do this itself, which meant every embedder kept its own shadow
        copy of the spec, dug the schema out of the frame it happened to have
        started from, and reimplemented the same two lines -- three copies of
        one rule, each able to drift from the session that actually renders.

        ``updated_spec`` stays the single composition authority; this only
        gives it the two things it needs and submits what it returns.
        """

        with self._lock:
            self._assert_open()
            data = self._projection.data
            schema = snapshot_schema(data) if isinstance(data, OwnedSnapshot) else None
            candidate = updated_spec(schema, self._spec, str(name), value)
            unchanged = candidate is self._spec or candidate == self._spec
        if unchanged:
            return self.describe_display()
        return self.replace_spec(candidate)

    def replace_spec(
        self,
        spec: PlotSpec,
        *,
        parameters: Mapping[str, object] | None = None,
        size: str | None = None,
        image_overlay: ImagePointOverlay | None | object = _UNSET,
        classifier_thresholds: object = _UNSET,
    ) -> DisplayDescription:
        """Atomically replace semantics and final presentation on one Figure."""

        if not isinstance(spec, (CurvePlot, ImagePlot, HistogramPlot, RollingPlot,
                                 FacetGridPlot, PulseTimelinePlot)):
            raise TypeError("spec must be a supported PlotSpec")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        threshold_target = (
            _UNSET
            if classifier_thresholds is _UNSET
            else normalize_classifier_threshold_targets(classifier_thresholds)
        )
        with self._render_lock:
            with self._lock:
                self._assert_open()
                selected_size = self._defaults.layout.validate_preset(
                    self.surface_plan.preset if size is None else size
                )
                if image_overlay is not _UNSET:
                    if image_overlay is not None and not isinstance(
                        image_overlay, ImagePointOverlay
                    ):
                        raise TypeError(
                            "image_overlay must be ImagePointOverlay or None"
                        )
                    if image_overlay is not None and not isinstance(
                        semantic_spec(spec), ImagePlot
                    ):
                        raise TypeError("image overlay requires ImagePlot")
                    if image_overlay is not None:
                        self._validate_image_frame_overlay(
                            self._image_overlay,
                            image_overlay,
                        )
                (
                    schema,
                    initial_state,
                    display_store,
                    focused,
                    projection,
                ) = self._prepare_replacement(spec, parameters, selected_size)
                assert self._renderer is not None
                renderer = self._renderer
                old_plan = renderer.plan
                previous = (
                    self._spec,
                    self._parameter_schema,
                    self._display_store,
                    self._projection,
                    self._selector_controller,
                    self._image_overlay,
                    self._viewport,
                    self._focused_facet_index,
                    self._facet_focus_index,
                    self._accepted_fit,
                    self._classifier_results,
                    self._classifier_overlays,
                    self._classifier_thresholds,
                    self._classifier_gaussian_components,
                    self._history,
                    self._layout_revision,
                    self._size,
                )
                self._cancel_gesture()
                self._spec = spec
                self._parameter_schema = schema
                self._display_store = display_store
                self._projection = projection
                self._selector_controller = _SelectorController()
                self._image_overlay = (
                    (
                        self._image_overlay
                        if image_overlay is _UNSET
                        else image_overlay
                    )
                    if isinstance(semantic_spec(spec), ImagePlot)
                    else None
                )
                self._size = selected_size
                self._viewport = initial_state.viewport
                self._focused_facet_index = focused
                self._facet_focus_index = None
                self._accepted_fit = None
                self._history = (
                    projection.rolling_history if _keeps_history(spec) else ()
                )
                self._layout_revision += 1
            try:
                # Layout resolution can reject a spec (for example the facet
                # cell cap); it must stay inside the rollback envelope so a
                # rejected replacement never leaves half-committed state.
                self._refresh_threshold_classifier()
                if threshold_target is not _UNSET:
                    self._set_classifier_thresholds_state(threshold_target)
                renderer.spec = spec
                plan = self._resolve_plan()
                renderer.relayout(
                    plan,
                    facet_index=self._focused_facet_index,
                    facet_focus_index=self._facet_focus_index,
                )
                self._update_renderer(renderer, RenderEffect.LAYOUT)
            except Exception:
                with self._lock:
                    (
                        self._spec,
                        self._parameter_schema,
                        self._display_store,
                        self._projection,
                        self._selector_controller,
                        self._image_overlay,
                        self._viewport,
                        self._focused_facet_index,
                        self._facet_focus_index,
                        self._accepted_fit,
                        self._classifier_results,
                        self._classifier_overlays,
                        self._classifier_thresholds,
                        self._classifier_gaussian_components,
                        self._history,
                        self._layout_revision,
                        self._size,
                    ) = previous
                    renderer.spec = self._spec
                renderer.relayout(
                    old_plan,
                    facet_index=self._focused_facet_index,
                    facet_focus_index=self._facet_focus_index,
                )
                self._update_renderer(renderer, RenderEffect.LAYOUT)
                raise
            with self._lock:
                fit_cancel = self._fit_cancel
                live_fit_cancel = self._live_fit_cancel
                live_prepare_cancel = self._live_prepare_cancel
                self._fit_context_generation += 1
                self._fit_request_generation += 1
                self._fit_warm_starts.clear()
                self._live_fit_cancel = Event()
                completion = self._live_fit_completion
                self._live_fit_completion = None
                self._live_fit_request = None
                self._live_fit_future = None
                description = self.describe_display()

        def retire_replaced_fit() -> None:
            fit_cancel.set()
            live_fit_cancel.set()
            live_prepare_cancel.set()
            if completion is not None and not completion.done():
                completion.set_exception(
                    FitCancelled("plot specification replaced")
                )

        self._commit_fit_actions(retire_replaced_fit)
        self._notify_display(description.display_state)
        return description

    def set_size(self, preset: str) -> SurfacePlan:
        selected = self._defaults.layout.validate_preset(preset)
        callbacks: tuple[SurfaceCallback, ...] = ()
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if selected == self._size:
                    return self.surface_plan
                self._cancel_gesture()
                previous_size = self._size
                previous_revision = self._layout_revision
                previous_plan = self.surface_plan
                try:
                    self._size = selected
                    plan = self._resolve_plan()
                    self._layout_revision += 1
                except Exception:
                    self._size = previous_size
                    self._layout_revision = previous_revision
                    raise
            try:
                self._apply_layout_plan(
                    plan,
                    schedule_fit=False,
                )
            except Exception:
                with self._lock:
                    self._size = previous_size
                    self._layout_revision = previous_revision
                try:
                    self._apply_layout_plan(
                        previous_plan,
                        schedule_fit=False,
                    )
                except Exception:
                    self.redraw_surface()
                raise
        return plan

    def focus_facet(self, index: int) -> None:
        """Open one FacetGrid cell as the full interactive plot surface."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            target_changed = self._select_facet(index)
            assert self._renderer is not None
            presentation_changed = self._facet_focus_index != index
            if presentation_changed:
                self._cancel_gesture()
                self._facet_focus_index = index
            if target_changed or presentation_changed:
                self._render_current(
                    RenderEffect.BASE_GEOMETRY
                    | RenderEffect.AXIS_TRANSFORM
                    | RenderEffect.CHROME
                )

    def show_facet_overview(self) -> None:
        """Return a focused FacetGrid cell to its non-interactive overview."""

        if not isinstance(self._spec, FacetGridPlot):
            raise TypeError("facet overview is available only for FacetGridPlot")
        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            if self._facet_focus_index is None:
                return
            self._cancel_gesture()
            cleared_viewport = self._viewport is not None
            self._viewport = None
            if cleared_viewport:
                # Closing a cell is an interaction, not a request to forget the
                # answer: the accepted overlay lags until a newer data revision
                # or an explicit clear, exactly as _invalidate_fit_context says.
                self._invalidate_fit_context()
            self._facet_focus_index = None
            self._render_current(
                RenderEffect.BASE_GEOMETRY
                | RenderEffect.AXIS_TRANSFORM
                | RenderEffect.CHROME
            )

    def _select_facet(self, index: int) -> bool:
        """Route cell-local state without changing overview/focus presentation."""

        if not isinstance(self._spec, FacetGridPlot):
            raise TypeError("facet selection is available only for FacetGridPlot")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("facet index must be an integer")
        cells = tuple(getattr(self._payload, "cells", ()))
        if index < 0 or index >= len(cells):
            raise IndexError("facet index is outside the current grid")
        changed = index != self._focused_facet_index
        if changed:
            self._cancel_gesture()
            self._focused_facet_index = index
            self._viewport = None
            # Opening another cell is looking at the same measurement more
            # closely.  The accepted result of a facet fit is a per-cell batch
            # over ALL cells (a live fit on a grid is forced to all_facets), so
            # it does not even depend on which cell is open -- deleting it here
            # left the operator with no fit, nothing to re-solve it (an armed
            # live fit only resolves on a NEW data revision), and a console
            # that refuses to re-apply an identical fit record, which is why
            # recovery needed a trip through "no model" first.
            self._invalidate_fit_context()
        return changed

    def _clamp_facet_state(self, cell_count: int) -> None:
        """Keep selected and open cells valid after a payload topology change."""

        if cell_count <= 0:
            selected = None
            opened = None
        else:
            selected = (
                cell_count - 1
                if self._focused_facet_index is None
                else min(self._focused_facet_index, cell_count - 1)
            )
            opened = selected if self._facet_focus_index is not None else None
        if (
            selected != self._focused_facet_index
            or opened != self._facet_focus_index
        ):
            self._focused_facet_index = selected
            self._facet_focus_index = opened
            self._viewport = None

    def set_device_pixel_ratio(self, ratio: float) -> SurfacePlan:
        return self._set_device_pixel_ratio(ratio, preserve_native_canvas=False)

    def _set_device_pixel_ratio(
        self,
        ratio: float,
        *,
        preserve_native_canvas: bool,
    ) -> SurfacePlan:
        selected = _validated_device_pixel_ratio(ratio)
        callbacks: tuple[SurfaceCallback, ...] = ()
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if selected == self._device_pixel_ratio:
                    return self.surface_plan
                self._cancel_gesture()
                previous_ratio = self._device_pixel_ratio
                previous_revision = self._layout_revision
                previous_plan = self.surface_plan
                try:
                    self._device_pixel_ratio = selected
                    plan = self._resolve_plan()
                    self._layout_revision += 1
                    if preserve_native_canvas:
                        assert self._renderer is not None
                        self._renderer.plan = plan
                        figure = self._renderer.figure
                        figure._original_dpi = plan.logical_dpi
                        figure._set_dpi(plan.dpi, forward=False)
                except Exception:
                    self._device_pixel_ratio = previous_ratio
                    self._layout_revision = previous_revision
                    raise
            try:
                if preserve_native_canvas:
                    self._render_current(
                        RenderEffect.LAYOUT,
                        schedule_fit=False,
                    )
                else:
                    self._apply_layout_plan(
                        plan,
                        schedule_fit=False,
                    )
            except Exception:
                with self._lock:
                    self._device_pixel_ratio = previous_ratio
                    self._layout_revision = previous_revision
                    assert self._renderer is not None
                    self._renderer.plan = previous_plan
                    figure = self._renderer.figure
                    figure._original_dpi = previous_plan.logical_dpi
                    figure._set_dpi(previous_plan.dpi, forward=False)
                try:
                    if preserve_native_canvas:
                        self._render_current(
                            RenderEffect.LAYOUT,
                            schedule_fit=False,
                        )
                    else:
                        self._apply_layout_plan(
                            previous_plan,
                            schedule_fit=False,
                        )
                except Exception:
                    self.redraw_surface()
                raise
        return plan

    def set_axis_unit(self, axis: AxisRef, unit: str | None) -> DisplayState:
        if not isinstance(axis, AxisRef):
            raise TypeError("axis must be AxisRef")
        target_name: str | None = None
        semantic = self._semantic_spec
        if getattr(semantic, "x", None) == axis:
            target_name = "x_display_unit"
        elif getattr(semantic, "y", None) == axis:
            target_name = "y_display_unit"
        elif isinstance(self._spec, FacetGridPlot) and axis == self._spec.facet:
            target_name = "facet_display_unit"
        if target_name is None:
            raise ValueError("axis is not a displayed x, y or facet axis in this plot")
        if target_name not in self._parameter_schema:
            raise TypeError(
                f"this plot kind does not expose {target_name!r}"
            )
        return self.set_parameter(target_name, unit)

    def set_value_unit(self, unit: str | None) -> DisplayState:
        return self.set_parameter("value_display_unit", unit)

    def set_time_unit(self, unit: str | None) -> DisplayState:
        """Set a PulseTimeline display unit, or ``None`` for automatic scaling."""

        if not isinstance(self._spec, PulseTimelinePlot):
            raise TypeError("set_time_unit is available only for PulseTimelinePlot")
        return self.set_parameter("x_display_unit", unit)

    def set_color_limits(
        self,
        low: float,
        high: float,
        *,
        fixed: bool = True,
    ) -> DisplayState:
        """Atomically edit an Image or Facet-image color range."""

        if "color_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no color limits")
        values: dict[str, object] = {"color_min": low, "color_max": high}
        if fixed:
            values["relim_mode"] = "fixed"
        return self.set_parameters(values)

    def resolved_color_limits(self, *, display: bool = True) -> NumericRange:
        """Return the color range painted by the current image front."""

        if not isinstance(display, bool):
            raise TypeError("display must be bool")
        if "color_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no color limits")
        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            value = NumericRange(*self._renderer.resolved_color_limits())
            if display or self._view is None:
                return value
            return self._projected._display_range_to_canonical(
                value,
                self._projected._value_quantity(),
            )

    def set_relim_mode(self, mode: str) -> DisplayState:
        """Select tight, normal, or fixed automatic axis scaling."""

        if "relim_mode" not in self._parameter_schema:
            raise TypeError("this plot kind has no relim mode")
        return self.set_parameter("relim_mode", mode)

    def set_y_limits(
        self,
        low: float,
        high: float,
        *,
        fixed: bool = True,
    ) -> DisplayState:
        """Atomically edit a curve value or histogram count y range."""

        if "y_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no configurable y limits")
        values: dict[str, object] = {"y_min": low, "y_max": high}
        if fixed:
            values["relim_mode"] = "fixed"
        return self.set_parameters(values)

    def reset_y_limits(self, *, mode: str = "normal") -> DisplayState:
        if "y_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no configurable y limits")
        return self.set_parameters(
            {"relim_mode": mode, "y_min": None, "y_max": None}
        )

    def reset_color_limits(self, *, mode: str = "tight") -> DisplayState:
        if "color_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no color limits")
        return self.set_parameters(
            {"relim_mode": mode, "color_min": None, "color_max": None}
        )

    def set_x_limits(self, low: float, high: float) -> RectangleRange:
        """Set the visible x range in the current display unit."""

        selected_x = NumericRange(float(low), float(high))
        with self._render_lock:
            with self._lock:
                self._assert_open()
            current = self._current_display_limits()
        return self.set_viewport(selected_x, current.y)

    def set_view_limits(
        self,
        *,
        x: tuple[float, float] | NumericRange | None = None,
        y: tuple[float, float] | NumericRange | None = None,
    ) -> RectangleRange:
        """Set either or both visible ranges in current display units."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            current = self._current_display_limits()

        def selected_range(
            value: tuple[float, float] | NumericRange | None,
            fallback: NumericRange,
            name: str,
        ) -> NumericRange:
            if value is None:
                return fallback
            if isinstance(value, NumericRange):
                return value
            try:
                low_value, high_value = value
            except (TypeError, ValueError) as error:
                raise TypeError(f"{name} limits must contain two values") from error
            return NumericRange(float(low_value), float(high_value))

        return self.set_viewport(
            selected_range(x, current.x, "x"),
            selected_range(y, current.y, "y"),
        )

    def update_data(
        self,
        data: PlotInput,
        *,
        revision: int | None = None,
    ) -> None:
        """Present new data, optionally preserving a producer-owned revision.

        OwnedSnapshot keeps its intrinsic revision. PulseTimeline callers may
        provide the revision from a live transport envelope; direct
        calls without one advance the current session revision by exactly one.
        An armed live fit makes this frame a pair: the solve runs to
        completion on the caller thread and the overlay is accepted into the
        same presented front as its data — the frame is born complete.
        Hosted panels pipeline the same pair off the render worker through
        ``prepare_live_frame``/``solve_live_frame``/``commit_live_frame``.
        """

        data, image_frame = self._split_image_frame(data, self._spec)
        image_overlay = _UNSET if image_frame is None else image_frame.overlay
        FitProjection._validate_input(data, self._spec)
        if revision is not None:
            if isinstance(revision, bool) or not isinstance(revision, Integral):
                raise TypeError("revision must be an integer or None")
            revision = int(revision)
            if revision < 0:
                raise ValueError("revision must be non-negative")
        if image_frame is not None and revision is not None:
            raise ValueError("ImageFrame revision is owned by its snapshot")

        with self._render_lock:
            with self._lock:
                self._assert_open()
                if isinstance(data, OwnedSnapshot):
                    assert isinstance(self._projection.data, OwnedSnapshot)
                    data_revision = snapshot_revision(data)
                    if revision is not None and revision != data_revision:
                        raise ValueError(
                            "OwnedSnapshot revision must equal the supplied revision"
                        )
                    previous_schema = snapshot_schema(self._projection.data)
                    next_schema = snapshot_schema(data)
                    if not (
                        schema_equal(previous_schema, next_schema)
                        or indexed_schemas_compatible(previous_schema, next_schema)
                    ):
                        raise ValueError("data schema must remain exactly constant")
                    previous_revision = snapshot_revision(self._projection.data)
                    if data_revision <= previous_revision:
                        raise ValueError(
                            "data revision must increase: "
                            f"{data_revision} <= {previous_revision}"
                        )
                    if image_frame is not None:
                        self._validate_image_frame_overlay(
                            self._image_overlay,
                            image_frame.overlay,
                        )
                else:
                    next_revision = (
                        self.data_revision + 1
                        if revision is None
                        else revision
                    )
                    if next_revision <= self.data_revision:
                        raise ValueError(
                            "data revision must increase: "
                            f"{next_revision} <= {self.data_revision}"
                        )
                selected_revision = (
                    snapshot_revision(data)
                    if isinstance(data, OwnedSnapshot)
                    else next_revision
                )
                projection = self._projection._fork_frozen(
                    data=data,
                    revision=selected_revision,
                    context=self._projection_context(),
                )
                projection._build_view_and_payload()
                accepted_overlay = (
                    self._image_overlay
                    if image_overlay is _UNSET
                    else image_overlay
                )
            started = self._pair_started(projection)
            accepted_fit = None
            resolution = None
            fit_event = None
            solved = None
            if started is not None:
                solved = self._solve_live_pair(started)
                accepted_fit, resolution, fit_event = self._accept_pair_fit(
                    solved,
                    projection,
                )
            if fit_event is not None:
                self._notify_fit(fit_event)
            try:
                self._present_projection_transaction(
                    projection,
                    image_overlay=accepted_overlay,
                    accepted_fit=accepted_fit,
                )
            except Exception:
                self._restore_live_fit_completion(resolution)
                raise
        if accepted_fit is not None and solved is not None:
            self._remember_fit_warm_starts(
                solved.result,
                request_generation=solved.started.request_generation,
                selections=accepted_fit.selections,
            )
        if resolution is not None:
            self._resolve_fit_completion(resolution)

    def update_image_frame(self, frame: ImageFrame) -> ImageFrame:
        """Present image data and its point layer in one render transaction."""

        if not isinstance(frame, ImageFrame):
            raise TypeError("frame must be ImageFrame")
        if not isinstance(self._semantic_spec, ImagePlot):
            raise TypeError("ImageFrame requires ImagePlot")
        self.update_data(frame)
        return frame

    @property
    def image_overlay(self) -> ImagePointOverlay | None:
        """Latest immutable canonical point layer for an Image plot."""

        with self._lock:
            return self._image_overlay

    @property
    def image_overlay_revision(self) -> int | None:
        with self._lock:
            overlay = self._image_overlay
            return None if overlay is None else overlay.revision

    def update_image_overlay(self, overlay: ImagePointOverlay) -> ImagePointOverlay:
        """Present one strictly newer point-layer revision without touching image data."""

        if not isinstance(overlay, ImagePointOverlay):
            raise TypeError("overlay must be ImagePointOverlay")
        if not isinstance(self._semantic_spec, ImagePlot):
            raise TypeError("image point overlays require ImagePlot")
        with self._render_lock:
            with self._lock:
                self._assert_open()
                previous = self._image_overlay
                if previous is not None and overlay.revision <= previous.revision:
                    raise RevisionError(
                        "image overlay revision must strictly increase"
                    )
                self._image_overlay = overlay
            try:
                self._render_current(RenderEffect.OVERLAY)
            except Exception:
                self._image_overlay = previous
                try:
                    self._render_current(RenderEffect.OVERLAY)
                except Exception:
                    self.redraw_surface()
                raise
        return overlay

    def subscribe_surface(self, callback: SurfaceCallback) -> Callable[[], None]:
        """Observe every surface commit, invoked on the rendering thread.

        The callback fires synchronously after each presented frame, under
        the render transaction; it must be non-blocking and must not reenter
        the session (queue work and return, as the raster host does).
        """

        return self._subscribe_callback(self._surface_callbacks, callback)

    def _subscribe_callback(
        self,
        callbacks: list[_CallbackT],
        callback: _CallbackT,
    ) -> Callable[[], None]:
        """Register one observer and return its idempotent release edge."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._assert_open()
            callbacks.append(callback)

        released = False

        def unsubscribe() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                if callback in callbacks:
                    callbacks.remove(callback)

        return unsubscribe

    def subscribe_display(self, callback: DisplayCallback) -> Callable[[], None]:
        """Observe accepted display-state changes on the attached host thread."""

        return self._subscribe_callback(self._display_callbacks, callback)

    def subscribe_viewport(
        self, callback: Callable[[object], object]
    ) -> Callable[[], None]:
        """Observe committed viewport geometry in canonical and display units."""

        return self._subscribe_callback(self._viewport_callbacks, callback)

    def _notify_display(self, state: DisplayState) -> None:
        deferred = self._configuration_display_events
        if deferred is not None:
            deferred.append(state)
            return
        with self._lock:
            callbacks = tuple(self._display_callbacks)
        self._notify_callbacks(callbacks, state)

    @staticmethod
    def _notify_surface_callbacks(
        callbacks: Sequence[SurfaceCallback],
    ) -> None:
        """Notify independent surface observers after a front is committed."""

        for callback in callbacks:
            try:
                callback()
            except Exception:
                continue

    def _notify_callbacks(
        self,
        callbacks: Sequence[Callable[[_EventT], object]],
        event: _EventT,
    ) -> None:
        """Marshal one event to isolated application observers."""

        if not callbacks:
            return

        def invoke() -> None:
            for callback in callbacks:
                try:
                    callback(event)
                except Exception:
                    # One application callback must not disable later observers.
                    continue

        self.owner_dispatch(invoke)

    def owner_dispatch(self, callback: Callable[[], _ResultT]) -> Future[_ResultT]:
        """Run through the current owner and always return its completion.

        This bound method is a stable gateway: callers may retain it before an
        interactive host is attached.  Host attachment/release and the
        headless direct path share one ownership gate, so a direct callback
        cannot overlap a transition to a notebook or raster owner.
        """

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._ownership_gate:
            with self._lock:
                dispatch = self._dispatch
            if dispatch is None:
                completion: Future[_ResultT] = Future()
                if not completion.set_running_or_notify_cancel():
                    return completion
                try:
                    value = callback()
                except Exception as error:
                    completion.set_exception(error)
                else:
                    completion.set_result(value)
                return completion
            try:
                completion = dispatch(callback)
            except Exception as error:
                failed: Future[_ResultT] = Future()
                failed.set_exception(error)
                return failed
            if not isinstance(completion, Future):
                invalid: Future[_ResultT] = Future()
                invalid.set_exception(
                    TypeError("host dispatch must return concurrent.futures.Future")
                )
                return invalid
            return completion

    def attach_host(
        self,
        owner: object,
        dispatch: HostDispatch,
        *,
        presentation_dispatch: HostPresentationDispatch | None = None,
    ) -> Callable[[], None]:
        """Attach exactly one interactive canvas host and return its release edge."""

        if owner is None:
            raise TypeError("host owner must not be None")
        if not callable(dispatch):
            raise TypeError("dispatch must be callable")
        if presentation_dispatch is not None and not callable(
            presentation_dispatch
        ):
            raise TypeError("presentation_dispatch must be callable or None")
        with self._ownership_gate:
            with self._lock:
                self._assert_open()
                if self._host_owner is not None:
                    if self._host_owner is owner:
                        raise RuntimeError("interactive host is already attached")
                    raise RuntimeError(
                        "PlotSession supports one interactive host; create another "
                        "session over the same immutable snapshot for a second view"
                    )
                self._host_owner = owner
                self._host_previous_dispatch = self._dispatch
                self._host_previous_presentation_dispatch = (
                    self._presentation_dispatch
                )
                self._dispatch = dispatch
                self._presentation_dispatch = presentation_dispatch

        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            with self._ownership_gate:
                with self._render_lock:
                    with self._lock:
                        if self._host_owner is not owner:
                            return
                    if not self._closed:
                        self._cancel_gesture()
                    with self._lock:
                        previous = self._host_previous_dispatch
                        previous_presentation = (
                            self._host_previous_presentation_dispatch
                        )
                        self._host_owner = None
                        self._host_previous_dispatch = None
                        self._host_previous_presentation_dispatch = None
                        self._dispatch = previous
                        self._presentation_dispatch = previous_presentation

        return release

    def set_x_selector(
        self,
        low: float,
        high: float,
        *,
        display: bool = True,
        emit_change: bool = True,
    ) -> SelectorState:
        value = NumericRange(low, high)
        with self._render_lock:
            with self._lock:
                self._assert_open()
            if display:
                if self._view is not None:
                    value = self._projected._display_range_to_canonical(
                        value, self._projected._x_selector_source()
                    )
                elif isinstance(self._spec, PulseTimelinePlot):
                    value = self._pulse_display_range_to_source(value)
            return self._install_selector_state(
                SelectorState(
                    SelectorKind.X_RANGE,
                    value,
                    facet_index=self._focused_facet_index,
                ),
                emit_change=emit_change,
            )

    def set_area_selector(
        self,
        x: NumericRange,
        y: NumericRange,
        *,
        display: bool = True,
        emit_change: bool = True,
    ) -> SelectorState:
        value = RectangleRange(x, y)
        with self._render_lock:
            with self._lock:
                self._assert_open()
            if display:
                value = self._area_display_to_canonical(value)
            return self._install_selector_state(
                SelectorState(
                    SelectorKind.AREA,
                    value,
                    facet_index=self._focused_facet_index,
                ),
                emit_change=emit_change,
            )

    def _install_selector_state(
        self,
        state: SelectorState,
        *,
        emit_change: bool = True,
        finished_gesture: _SelectorGesture | None = None,
    ) -> SelectorState:
        """Atomically replace the one selector owned by ``state.kind``."""

        self._require_stable_selector(state)
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if (
                    finished_gesture is None
                    and isinstance(self._gesture, _SelectorGesture)
                ):
                    self._cancel_gesture()
                fit_previous = self._fit_context_generation
                fit_cancel = self._fit_cancel
                request = self._live_fit_request
                previous, stored = (
                    self._selector_controller.install(state)
                    if finished_gesture is None
                    else self._selector_controller._commit_finished(state)
                )
                classifier_previous = self._classifier_thresholds
                if (
                    stored.kind is SelectorKind.THRESHOLD
                    and self._threshold_classifier_enabled()
                    and self._classifier_thresholds
                ):
                    index = 0 if stored.facet_index is None else stored.facet_index
                    if 0 <= index < len(self._classifier_thresholds):
                        updated = list(self._classifier_thresholds)
                        updated[index] = float(stored.value)
                        self._classifier_thresholds = tuple(updated)
                affects_fit = self._selector_change_affects_fit(
                    stored.kind,
                    request,
                )
                context_changed = affects_fit
                if context_changed:
                    # Selection changes only make the accepted data-bound fit
                    # lag the current interaction.  They do not start a new
                    # solve and do not remove the stable painted overlay.
                    self._fit_context_generation += 1
            try:
                self._render_current(
                    RenderEffect.OVERLAY,
                    schedule_fit=False,
                )
            except Exception:
                with self._lock:
                    self._selector_controller._rollback_install(stored, previous)
                    self._classifier_thresholds = classifier_previous
                    self._fit_context_generation = fit_previous
                try:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
                except Exception:
                    self.redraw_surface()
                raise
            if context_changed:
                self._commit_fit_actions(fit_cancel.set)
        if emit_change:
            self._emit_selection(
                SelectionChange.ADDED if previous is None else SelectionChange.UPDATED,
                stored,
            )
        return stored

    def _selector_change_affects_fit(
        self,
        kind: SelectorKind,
        request: _LiveFitRequest | None,
    ) -> bool:
        if request is not None:
            return self._fit_request_uses_selector(request, kind)
        accepted = self._accepted_fit
        return bool(
            accepted is not None
            and (
                kind in {SelectorKind.AREA, SelectorKind.X_RANGE}
                or (
                    accepted.selection is not None
                    and accepted.selection.selector_kind is kind
                )
            )
        )

    def set_threshold_selector(
        self, value: float, *, display: bool = True
    ) -> SelectorState:
        with self._render_lock:
            with self._lock:
                self._assert_open()
            canonical = (
                self._projected._display_scalar_to_canonical(
                    value,
                    self._projected._value_quantity(),
                )
                if display and self._view is not None
                else value
            )
            return self._install_selector_state(SelectorState(
                SelectorKind.THRESHOLD,
                canonical,
                facet_index=self._focused_facet_index,
            ))

    def set_crosshair_selector(
        self, x: float, y: float, *, display: bool = True
    ) -> SelectorState:
        with self._render_lock:
            with self._lock:
                self._assert_open()
            point = CrosshairPoint(x, y)
            if display:
                if self._view is not None:
                    point = CrosshairPoint(
                        self._display_x_scalar_to_canonical(x),
                        y
                        if self._projected._is_histogram_plot()
                        else self._projected._display_scalar_to_canonical(
                            y, self._projected._y_ref_or_value()
                        ),
                    )
                elif isinstance(self._spec, PulseTimelinePlot):
                    point = CrosshairPoint(self._pulse_display_x_to_source(x), y)
            return self._install_selector_state(SelectorState(
                SelectorKind.CROSSHAIR,
                point,
                facet_index=self._focused_facet_index,
            ))

    def selector_state(
        self,
        kind: SelectorKind,
        *,
        display: bool = False,
    ) -> SelectorState:
        with self._render_lock:
            with self._lock:
                self._assert_open()
            state = next(
                (
                    item
                    for item in self._resolved_selector_snapshot().states
                    if item.kind is kind
                ),
                None,
            )
            if state is None:
                raise KeyError(kind)
            if not display:
                return state
            if self._view is not None:
                return self._projected._display_selector_state(state)
            return self._special_display_selector_state(state)

    def set_selector_value(
        self,
        kind: SelectorKind,
        value: SelectorValue,
        *,
        display: bool = True,
    ) -> SelectorState:
        """Update a selector without exposing controller or canonical-unit details."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            current = self._selector_controller.state(kind)
            canonical = self._canonical_selector_value(current, value, display=display)
            candidate = replace(current, value=canonical)
            self._require_stable_selector(candidate)
            return self._install_selector_state(candidate)

    def _canonical_selector_value(
        self,
        current: SelectorState,
        value: SelectorValue,
        *,
        display: bool,
    ) -> SelectorValue:
        canonical = value
        if display and self._view is not None:
            if current.kind is SelectorKind.X_RANGE:
                if not isinstance(value, NumericRange):
                    raise TypeError("x-range selector requires NumericRange")
                canonical = self._projected._display_range_to_canonical(
                    value, self._projected._x_selector_source()
                )
            elif current.kind is SelectorKind.AREA:
                if not isinstance(value, RectangleRange):
                    raise TypeError("area selector requires RectangleRange")
                canonical = self._area_display_to_canonical(value)
            elif current.kind is SelectorKind.CROSSHAIR:
                if not isinstance(value, CrosshairPoint):
                    raise TypeError("crosshair selector requires CrosshairPoint")
                canonical = CrosshairPoint(
                    self._display_x_scalar_to_canonical(value.x),
                    value.y
                    if self._projected._is_histogram_plot()
                    else self._projected._display_scalar_to_canonical(
                        value.y, self._projected._y_ref_or_value()
                    ),
                )
            elif current.kind is SelectorKind.THRESHOLD:
                canonical = self._projected._display_scalar_to_canonical(
                    float(value), self._projected._value_quantity()
                )
        elif display and isinstance(self._spec, PulseTimelinePlot):
            if current.kind is SelectorKind.X_RANGE:
                if not isinstance(value, NumericRange):
                    raise TypeError("x-range selector requires NumericRange")
                canonical = self._pulse_display_range_to_source(value)
            elif current.kind is SelectorKind.AREA:
                if not isinstance(value, RectangleRange):
                    raise TypeError("area selector requires RectangleRange")
                canonical = self._area_display_to_canonical(value)
            elif current.kind is SelectorKind.CROSSHAIR:
                if not isinstance(value, CrosshairPoint):
                    raise TypeError("crosshair selector requires CrosshairPoint")
                canonical = CrosshairPoint(
                    self._pulse_display_x_to_source(value.x), value.y
                )
        return canonical

    def remove_selector(
        self,
        kind: SelectorKind,
        *,
        emit_change: bool = True,
    ) -> SelectorState:
        """Remove a selector and notify external subscribers."""

        cancelled_fit: Future[FitResult | FacetFitBatchResult] | None = None
        withdrawn = False
        with self._render_lock:
            with self._lock:
                self._assert_open()
                state = self._selector_controller.state(kind)
                gesture = self._gesture
                if (
                    isinstance(gesture, _SelectorGesture)
                    and gesture.kind is state.kind
                ):
                    self._cancel_gesture()
                previous_fit = (
                    self._fit_context_generation,
                    self._fit_request_generation,
                    self._live_fit_completion,
                    self._live_fit_request,
                    self._live_fit_future,
                    self._accepted_fit,
                )
                fit_cancel = self._fit_cancel
                live_fit_cancel = self._live_fit_cancel
                request = self._live_fit_request
                bound_request = bool(
                    request is not None and request.selector_kind is state.kind
                )
                affects_fit = self._selector_change_affects_fit(
                    state.kind,
                    request,
                )
                self._selector_controller.remove(kind)
                # Taking the line away is how an operator says "classify where
                # the fit says", so the choice it recorded goes with it.  The
                # refresh path removes this selector too, but through the
                # controller directly: that one is repainting the same answer,
                # not withdrawing it.
                chosen_previous = self._classifier_thresholds
                if kind is SelectorKind.THRESHOLD and self._classifier_thresholds:
                    index = 0 if state.facet_index is None else state.facet_index
                    if 0 <= index < len(self._classifier_thresholds):
                        cleared = list(self._classifier_thresholds)
                        cleared[index] = None
                        self._classifier_thresholds = tuple(cleared)
                if affects_fit:
                    self._fit_context_generation += 1
                    if bound_request:
                        # Removing the bound selector un-arms the live fit —
                        # a request replacement, so the pair token turns over.
                        self._fit_request_generation += 1
                        self._fit_warm_starts.clear()
                        self._live_fit_cancel = Event()
                        cancelled_fit = self._live_fit_completion
                        self._live_fit_completion = None
                        self._live_fit_request = None
                        self._live_fit_future = None
                        withdrawn = self._clear_fit_presentation()
            try:
                if self._renderer is not None:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
            except Exception:
                with self._lock:
                    self._selector_controller._restore_removed(state)
                    self._classifier_thresholds = chosen_previous
                    (
                        self._fit_context_generation,
                        self._fit_request_generation,
                        self._live_fit_completion,
                        self._live_fit_request,
                        self._live_fit_future,
                        self._accepted_fit,
                    ) = previous_fit
                try:
                    if self._renderer is not None:
                        self._render_current(
                            RenderEffect.OVERLAY,
                            schedule_fit=False,
                        )
                except Exception:
                    self.redraw_surface()
                raise

        def retire_selector_fit() -> None:
            if not affects_fit:
                return
            fit_cancel.set()
            if bound_request:
                # Un-arming cancels only the live pair solve; an in-flight
                # data-frame preparation is fit-agnostic and continues.
                live_fit_cancel.set()
            if cancelled_fit is not None and not cancelled_fit.done():
                cancelled_fit.set_exception(
                    FitCancelled(f"fit selector removed: {state.kind.value}")
                )

        self._commit_fit_actions(retire_selector_fit)
        if withdrawn:
            self._notify_fit(None)
        if emit_change:
            self._emit_selection(SelectionChange.REMOVED, state)
        return state

    def subscribe_selection(
        self,
        callback: SelectionCallback,
        *,
        selector_kind: SelectorKind | None = None,
    ) -> Callable[[], None]:
        """Observe selector geometry changes on the host thread.

        Selection events deliberately contain no sliced data.  Call
        :meth:`selector_data` explicitly when the current snapshot's selected
        values are actually needed.
        """

        if not callable(callback):
            raise TypeError("callback must be callable")
        if selector_kind is not None and not isinstance(selector_kind, SelectorKind):
            raise TypeError("selector_kind must be SelectorKind or None")
        subscription = _SelectionSubscription(callback, selector_kind)
        with self._lock:
            self._assert_open()
            self._selection_subscriptions.append(subscription)

        released = False

        def unsubscribe() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                if subscription in self._selection_subscriptions:
                    self._selection_subscriptions.remove(subscription)

        return unsubscribe

    def _emit_selection(self, change: SelectionChange, state: SelectorState) -> None:
        with self._render_lock:
            with self._lock:
                subscriptions = tuple(
                    item
                    for item in self._selection_subscriptions
                    if item.selector_kind in (None, state.kind)
                )
            if not subscriptions:
                return
            display_state = (
                self._special_display_selector_state(state)
                if self._view is None
                else self._projected._display_selector_state(state)
            )
            event = SelectionEvent(
                change,
                state,
                display_state,
                self.data_revision,
                self.data_generation,
                self._selection_subject(state),
                (
                    self._classifier_threshold_targets_state()
                    if state.kind is SelectorKind.THRESHOLD
                    else ()
                ),
            )

        self._notify_callbacks(
            tuple(item.callback for item in subscriptions),
            event,
        )

    def _selection_subject(
        self,
        state: SelectorState | None = None,
    ) -> SelectionSubject:
        """Resolve what the current selectors cut, through the projection.

        The projection already owns this rule for slicing and units; asking it
        here keeps one answer rather than a second copy that can disagree.
        """

        semantic = self._semantic_spec
        if self._view is None:
            return SelectionSubject(semantic.kind, None, None)
        x = self._projected._x_selector_source() if self._has_x_selector_source() else None
        y = self._projected._y_ref_or_value()
        x_ref = x if isinstance(x, AxisRef) else None
        y_ref = y if isinstance(y, AxisRef) else None
        scope, repeat_index = self._selection_scope(state)
        return SelectionSubject(
            semantic.kind,
            x_ref,
            y_ref,
            self._selection_axis_frame(x_ref),
            self._selection_axis_frame(y_ref),
            scope,
            repeat_index,
        )

    def _selection_axis_frame(self, ref: AxisRef | None) -> str | None:
        """The producer coordinate frame of one resolved plot axis."""

        if ref is None or not isinstance(self._projection.data, OwnedSnapshot):
            return None
        if ref.domain is AxisDomain.POINT_ROW:
            return None
        physical_domain = (
            "repeat"
            if ref.domain is AxisDomain.REPEAT
            else "data"
            if ref.domain is AxisDomain.DATA
            else "point"
        )
        for label, axis_id, axis, domain in axis_catalog(
            snapshot_schema(self._projection.data)
        ):
            if domain != physical_domain:
                continue
            if ref.axis_id is not None and ref.axis_id not in {
                str(label),
                str(axis_id),
            }:
                continue
            frame = axis.coordinate_frame
            return None if frame is None else str(frame)
        return None

    def _selection_scope(
        self,
        state: SelectorState | None,
    ) -> tuple[tuple[tuple[AxisRef, object], ...], int | None]:
        """Canonical panel/facet narrowing that existed when a gesture fired."""

        if not isinstance(self._projection.data, OwnedSnapshot):
            return (), None
        schema = snapshot_schema(self._projection.data)
        scope: list[tuple[AxisRef, object]] = []
        repeat_index: int | None = None
        for ref, value in getattr(self._spec, "scope", ()):
            if value == "latest":
                physical_domain = (
                    "repeat"
                    if ref.domain is AxisDomain.REPEAT
                    else "point"
                    if ref.domain in {
                        AxisDomain.POINT_ROW,
                        AxisDomain.POINT_COORDINATE,
                        AxisDomain.POINT_DIMENSION,
                    }
                    else "data"
                )
                matches = tuple(
                    axis
                    for _label, axis_id, axis, domain in axis_catalog(schema)
                    if domain == physical_domain
                    and (
                        ref.axis_id is None
                        or ref.axis_id in {str(axis_id), axis.name}
                    )
                )
                if len(matches) != 1:
                    raise ValueError("latest scope axis is not uniquely present")
                value = matches[0].coordinate_at(matches[0].size - 1)
            if ref.domain is AxisDomain.REPEAT:
                repeat_index = next(
                    index
                    for index in range(schema.repeat_axis.size)
                    if schema.repeat_axis.coordinate_at(index) == value
                )
            else:
                scope.append((ref, value))

        facet_index = (
            self._facet_focus_index
            if state is None
            else state.facet_index
        )
        if isinstance(self._spec, FacetGridPlot) and facet_index is not None:
            if self._spec.facet.domain is AxisDomain.REPEAT:
                repeat_index = int(facet_index)
            else:
                cell = next(
                    item
                    for item in tuple(getattr(self._payload, "cells", ()))
                    if item.facet_index == facet_index
                )
                value = cell.facet_value_canonical
                if isinstance(value, np.generic):
                    value = value.item()
                scope.append((self._spec.facet, value))
        return tuple(scope), repeat_index

    def _classifier_threshold_target_for_index(
        self,
        facet_index: int | None,
        value: object,
    ) -> Mapping[str, object]:
        state = SelectorState(
            SelectorKind.THRESHOLD,
            float(value),
            facet_index=facet_index,
        )
        return _classifier_threshold_target_from_subject(
            self._selection_subject(state),
            state.value,
        )

    def _has_x_selector_source(self) -> bool:
        """Whether an x source exists at all, asked before it is resolved.

        A plot whose x is the value quantity or is unassigned has no x axis to
        name; that is a description, not a failure, so it must not arrive as an
        exception raised inside an event emission.
        """

        semantic = self._semantic_spec
        if isinstance(semantic, HistogramPlot):
            return True  # resolves to the value quantity, which is not an AxisRef
        if isinstance(semantic, RollingPlot):
            return True  # resolves to its ordinal row placeholder
        return isinstance(getattr(semantic, "x", None), AxisRef)

    def selector_data(self, kind: SelectorKind) -> SelectorData:
        """Slice the current immutable snapshot only when explicitly called."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
                if isinstance(self._projection.data, PulseTimelineData):
                    return self._pulse_timeline_selector_data(kind)
                assert self._view is not None
                state = next(
                    (
                        item
                        for item in self._resolved_selector_snapshot().states
                        if item.kind is kind
                    ),
                    None,
                )
                if state is None:
                    raise KeyError(kind)
                axes = self._selector_axes(state)
                mask = self._projected._selector_mask(
                    state,
                    point_transform=(
                        None if axes is None else axes.transData.transform
                    ),
                )
                samples = self._view.samples
                flat = np.flatnonzero(mask.reshape(-1))
                refs = self._semantic_refs()
                canonical_coordinates = {
                    ref: np.asarray(self._projected._coordinate(ref).canonical)[mask]
                    for ref in refs
                }
                display_coordinates = {}
                for ref in refs:
                    coordinate = self._projected._coordinate(ref)
                    displayed = (
                        self._projected._coordinate_values_to_display(
                            np.asarray(coordinate.canonical), ref
                        )
                        if isinstance(self._spec, RollingPlot) and ref == self._projected._x_ref()
                        else np.asarray(coordinate.display)
                    )
                    display_coordinates[ref] = displayed[mask]
                source_payload = self._projection._focused_payload(state.facet_index)
                return SelectionData(
                    state,
                    mask,
                    flat,
                    np.asarray(samples.value.canonical)[mask],
                    np.asarray(samples.value.display)[mask],
                    canonical_coordinates,
                    display_coordinates,
                    self.data_revision,
                    state.facet_index,
                    tuple(getattr(source_payload, "source_revisions", (self.data_revision,))),
                )

    def _pulse_timeline_selector_data(
        self,
        kind: SelectorKind,
    ) -> PulseTimelineSelectionData:
        payload = self._projection.data
        if not isinstance(payload, PulseTimelineData):
            raise TypeError("PulseTimeline selector data requires PulseTimelineData")
        state = self._selector_controller.state(kind)
        display_state = self._special_display_selector_state(state)
        time_range: NumericRange | None
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(state.value, NumericRange)
            time_range = state.value
        elif state.kind is SelectorKind.AREA:
            assert isinstance(state.value, RectangleRange)
            time_range = state.value.x
        elif state.kind is SelectorKind.CROSSHAIR:
            assert isinstance(state.value, CrosshairPoint)
            time_range = NumericRange(state.value.x, state.value.x)
        else:
            time_range = None

        def intersects(start: float, stop: float) -> bool:
            return bool(
                time_range is not None
                and start <= time_range.high
                and stop >= time_range.low
            )

        blocks = tuple(
            record for record in payload.blocks if intersects(record.start, record.stop)
        )
        channel_ids = {record.channel_id for record in blocks}
        channels = tuple(
            record for record in payload.channels if record.channel_id in channel_ids
        )
        analog_traces = tuple(
            record
            for record in payload.analog_traces
            if record.starts and intersects(record.starts[0], record.starts[-1])
        )
        scan_regions = tuple(
            record
            for record in payload.scan_regions
            if intersects(record.start, record.stop)
        )
        scan_dac_segments = tuple(
            record
            for record in payload.scan_dac_segments
            if intersects(record.start, record.stop)
        )
        repeat_markers = tuple(
            record
            for record in payload.repeat_markers
            if intersects(record.start, record.stop)
        )
        return PulseTimelineSelectionData(
            state,
            display_state,
            channels,
            blocks,
            analog_traces,
            scan_regions,
            scan_dac_segments,
            repeat_markers,
            self.data_revision,
            (self.data_revision,),
        )

    def _semantic_refs(self) -> tuple[AxisRef, ...]:
        semantic = self._semantic_spec
        result = []
        for ref in (
            getattr(semantic, "x", None),
            getattr(semantic, "y", None),
            getattr(semantic, "group", None),
            self._spec.facet if isinstance(self._spec, FacetGridPlot) else None,
        ):
            if isinstance(ref, AxisRef) and ref not in result:
                result.append(ref)
        return tuple(result)


    def _viewport_from_canonical(self, viewport: RectangleRange) -> RectangleRange:
        return RectangleRange(
            self._projected._canonical_range_to_display(
                viewport.x, self._projected._x_selector_source()
            ),
            viewport.y
            if self._projected._is_histogram_plot()
            else self._projected._canonical_range_to_display(
                viewport.y, self._projected._y_ref_or_value()
            ),
        )

    @property
    def viewport(self) -> RectangleRange | None:
        """Current explicit display-space zoom/pan region, if the user set one."""

        with self._lock:
            return self._viewport

    def _viewport_y_to_axes(self, value: NumericRange) -> tuple[float, float]:
        if isinstance(self._projected._semantic_spec(), ImagePlot):
            return value.high, value.low
        return value.low, value.high

    def set_viewport(
        self,
        x: NumericRange,
        y: NumericRange,
        *,
        emit_change: bool = True,
    ) -> RectangleRange:
        if not isinstance(x, NumericRange) or not isinstance(y, NumericRange):
            raise TypeError("viewport x and y must be NumericRange")
        selected = RectangleRange(x, y)
        self._set_viewport_state(selected, emit_change=emit_change)
        return selected

    def _set_viewport_state(
        self,
        selected: RectangleRange | None,
        *,
        emit_change: bool = True,
    ) -> bool:
        """Commit viewport and fit authority only after their frame draws."""

        if selected is not None and not isinstance(selected, RectangleRange):
            raise TypeError("selected must be RectangleRange or None")
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if selected == self._viewport:
                    return False
                previous = (
                    self._viewport,
                    self._fit_context_generation,
                )
                fit_cancel = self._fit_cancel
                self._viewport = selected
                self._fit_context_generation += 1
            effects = (
                RenderEffect.AXIS_TRANSFORM
                | RenderEffect.FIT_SELECTION
                | RenderEffect.OVERLAY
            )
            try:
                self._render_current(effects, schedule_fit=False)
            except Exception:
                with self._lock:
                    (
                        self._viewport,
                        self._fit_context_generation,
                    ) = previous
                try:
                    self._render_current(effects, schedule_fit=False)
                except Exception:
                    self.redraw_surface()
                raise
            self._commit_fit_actions(fit_cancel.set)
        if not emit_change:
            return True
        display = selected
        canonical = self._area_display_to_canonical(
            self._current_display_limits() if display is None else display
        )
        subject = self._selection_subject()
        with self._lock:
            callbacks = tuple(self._viewport_callbacks)
        self._notify_callbacks(
            callbacks,
            (canonical, display, subject),
        )
        return True

    def reset_viewport(self, *, emit_change: bool = True) -> None:
        self._set_viewport_state(None, emit_change=emit_change)

    def _pulse_display_x_to_source(self, value: float) -> float:
        return float(value) / self._projected._pulse_x_factor()

    def _pulse_source_x_to_display(self, value: float) -> float:
        return float(value) * self._projected._pulse_x_factor()

    def _pulse_display_range_to_source(
        self, value: NumericRange
    ) -> NumericRange:
        factor = self._projected._pulse_x_factor()
        return NumericRange(value.low / factor, value.high / factor)


    def _viewport_x_to_axes(self, value: NumericRange) -> NumericRange:
        return (
            self._pulse_display_range_to_source(value)
            if isinstance(self._spec, PulseTimelinePlot)
            else value
        )

    def _viewport_x_from_axes(self, value: NumericRange) -> NumericRange:
        return (
            self._projected._pulse_source_range_to_display(value)
            if isinstance(self._spec, PulseTimelinePlot)
            else value
        )

    def _display_x_scalar_to_canonical(self, value: float) -> float:
        if isinstance(self._spec, RollingPlot):
            # The rolling shot axis is a plain ordinal: display == canonical.
            return float(value)
        source = self._projected._x_selector_source()
        quantity = self._projected._coordinate(source) if isinstance(source, AxisRef) else source
        return self._projected._display_scalar_to_canonical(value, quantity)


    def _area_display_to_canonical(
        self,
        value: RectangleRange,
    ) -> RectangleRange:
        if self._view is not None:
            x = self._projected._display_range_to_canonical(
                value.x,
                self._projected._x_selector_source(),
            )
            y = (
                value.y
                if self._projected._is_histogram_plot()
                else self._projected._display_range_to_canonical(
                    value.y,
                    self._projected._y_ref_or_value(),
                )
            )
            return RectangleRange(x, y)
        if isinstance(self._spec, PulseTimelinePlot):
            return RectangleRange(
                self._pulse_display_range_to_source(value.x),
                value.y,
            )
        return value


    def _special_display_selector_state(self, state: SelectorState) -> SelectorState:
        if not isinstance(self._spec, PulseTimelinePlot):
            return state
        value = state.value
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(value, NumericRange)
            value = self._projected._pulse_source_range_to_display(value)
        elif state.kind is SelectorKind.AREA:
            assert isinstance(value, RectangleRange)
            value = self._projected._area_canonical_to_display(value)
        elif state.kind is SelectorKind.CROSSHAIR:
            assert isinstance(value, CrosshairPoint)
            value = CrosshairPoint(
                self._pulse_source_x_to_display(value.x), value.y
            )
        return replace(state, value=value)

    def _resolved_selector_snapshot(self) -> SelectorSnapshot:
        """Return the sole painted/hit-tested state for every selector kind."""

        snapshot = self._selector_controller.snapshot()
        derived = self._derived_threshold_classifier_selector()
        if derived is None:
            return snapshot
        candidate = snapshot.candidate
        active_index = derived.facet_index
        if (
            candidate is not None
            and candidate.kind is SelectorKind.THRESHOLD
            and candidate.facet_index == active_index
        ):
            threshold = candidate
        else:
            threshold = next(
                (
                    state
                    for state in snapshot.committed
                    if state.kind is SelectorKind.THRESHOLD
                    and state.facet_index == active_index
                ),
                derived,
            )
        committed = tuple(
            state
            for state in snapshot.committed
            if state.kind is not SelectorKind.THRESHOLD
        ) + (threshold,)
        return SelectorSnapshot(
            committed,
            candidate if candidate is not None and candidate.kind is not SelectorKind.THRESHOLD else None,
        )

    def _painted_selector_state(self, state: SelectorState) -> SelectorState:
        """Project one selector into the renderer's axes data space.

        DataView-backed kinds paint their axes in display units, so painted
        geometry is the display projection.  Pulse paints its time axis in
        source units and converts only tick and label text, so its painted
        geometry is the canonical state itself; the scene's ``x_label_factor``
        applies the display factor to readout text exactly once.
        """

        if self._view is not None:
            return self._projected._display_selector_state(state)
        return state

    def _painted_selector_snapshot(self) -> SelectorSnapshot:
        """Return every selector in the exact space the renderer paints."""

        snapshot = self._resolved_selector_snapshot()
        return SelectorSnapshot(
            tuple(self._painted_selector_state(state) for state in snapshot.committed),
            None
            if snapshot.candidate is None
            else self._painted_selector_state(snapshot.candidate),
        )

    def _display_selector_snapshot(self) -> SelectorSnapshot:
        snapshot = self._resolved_selector_snapshot()

        def display(state: SelectorState) -> SelectorState:
            return (
                self._projected._display_selector_state(state)
                if self._view is not None
                else self._special_display_selector_state(state)
            )

        return SelectorSnapshot(
            tuple(display(state) for state in snapshot.committed),
            None if snapshot.candidate is None else display(snapshot.candidate),
        )

    def _raster_pointer_state(
        self,
        *,
        publish_front: bool,
    ) -> _PointerUpdate:
        """Return the transient pointer state needed by a raster frontend."""

        snapshot = self._display_selector_snapshot()
        color_candidate = self._display_color_limit_candidate()
        candidate: SelectorState | ColorLimitCandidate | None = (
            color_candidate if color_candidate is not None else snapshot.candidate
        )
        gesture = self._gesture
        # An orbit drag rides the pan pathway: no selector candidate, but
        # the frontend must keep the button latched and the gesture axes.
        active_pan = isinstance(gesture, (_PanGesture, _OrbitGesture))
        axis = (
            gesture.axes
            if gesture is not None and (candidate is not None or active_pan)
            else None
        )
        if axis is None:
            return _PointerUpdate(
                candidate,
                None,
                None,
                active_pan,
                publish_front,
            )
        role, separator, suffix = str(axis.get_gid() or "main").partition(":")
        cell_index = int(suffix) if separator and suffix.isdigit() else None
        return _PointerUpdate(
            candidate,
            role or "main",
            cell_index,
            active_pan,
            publish_front,
        )

    def _raster_pointer_event(
        self,
        action: str,
        x: float,
        y: float,
        *,
        button: int | None = None,
        double: bool = False,
        step: float = 0.0,
        key: str | None = None,
        axes_snapshot: AxisTransform | None = None,
    ) -> _PointerUpdate:
        """Route normalized raster input through the native session handlers.

        Transient gesture candidates are baked into the raster by the same
        matplotlib artists that render committed selectors, so transient and
        committed appearance are identical by construction on every
        frontend.
        """

        from matplotlib.backend_bases import KeyEvent, MouseEvent

        if not isinstance(action, str):
            raise TypeError("raster pointer action must be text")
        selected_action = action.strip().lower()
        if selected_action not in {
            "press",
            "move",
            "release",
            "scroll",
            "key",
            "cancel",
            "leave",
        }:
            raise ValueError(f"unknown raster pointer action {action!r}")
        coordinates = np.asarray((x, y), dtype=float)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("raster pointer coordinates must be finite")
        if button is not None and button not in (1, 2, 3):
            raise ValueError("raster pointer button must be 1, 2, 3, or None")
        if axes_snapshot is not None and not isinstance(
            axes_snapshot,
            AxisTransform,
        ):
            raise TypeError("axes_snapshot must be AxisTransform or None")
        with self._lock:
            self._assert_open()
        presentation_epoch = self._presentation_epoch
        assert self._renderer is not None
        raster_generation = self._renderer.raster_generation
        canvas = self._renderer.figure.canvas
        width, height = canvas_physical_size(canvas)
        pixel_x = float(x) * float(width)
        pixel_y = (1.0 - float(y)) * float(height)
        interaction_transform = axes_snapshot
        def series(action: str, axes: Any | None = None) -> bool:
            return self._renderer.series_focus(
                action, axes, pixel_x, pixel_y,
                hit_radius=self._defaults.interaction.selector_handle_radius_px,
                click_radius=self._defaults.interaction.double_click_radius_px,
            )
        if selected_action == "cancel":
            series("clear")
            self.cancel_interaction()
        elif selected_action == "leave":
            series("leave")
        elif selected_action == "key":
            if str(key or "").lower() == "escape":
                series("clear")
            self._on_key_press(
                KeyEvent(
                    "key_press_event",
                    canvas,
                    key="" if key is None else str(key),
                    x=pixel_x,
                    y=pixel_y,
                )
            )
        elif selected_action == "scroll":
            if float(step) == 0.0:
                return self._raster_pointer_state(publish_front=False)
            direction = "up" if float(step) > 0.0 else "down"
            event = MouseEvent(
                "scroll_event",
                canvas,
                pixel_x,
                pixel_y,
                button=direction,
                step=float(step),
            )
            if interaction_transform is not None:
                event.inaxes = self._axis_for_transform(
                    interaction_transform
                )
            axes = self._renderer.interactive_axes_at(event)
            if not self._renderer.series_focus_scroll(axes, float(step)):
                self._on_scroll(event, interaction_transform=interaction_transform)
        else:
            event = MouseEvent(
                f"button_{selected_action}_event"
                if selected_action != "move"
                else "motion_notify_event",
                canvas,
                pixel_x,
                pixel_y,
                button=button,
                dblclick=bool(double),
            )
            event_axes = (
                self._axis_for_transform(interaction_transform)
                if interaction_transform is not None
                else self._renderer.interactive_axes_at(event)
            )
            if selected_action == "press":
                if button == 1 and not double:
                    series("press", event_axes)
                self._on_button_press(
                    event,
                    interaction_transform=interaction_transform,
                )
            elif selected_action == "move" and button is None:
                series("move", event_axes)
            elif selected_action == "release":
                handled = button == 1 and series("release", event_axes)
                if handled:
                    self.cancel_interaction()
                else:
                    self._on_button_release(event)
            else:
                self._on_motion(event)
        return self._raster_pointer_state(
            # Native (baked) previews redraw the raster without a full
            # presentation pass; either signal means the pixels changed.
            publish_front=(
                self._presentation_epoch != presentation_epoch
                or self._renderer.raster_generation != raster_generation
            ),
        )

    def save(
        self,
        path: str | Path,
        *,
        dpi: float | None = None,
        export_scale: float | None = None,
        **kwargs: Any,
    ) -> None:
        if dpi is not None and export_scale is not None:
            raise ValueError("specify dpi or export_scale, not both")
        if isinstance(dpi, bool) or isinstance(export_scale, bool):
            raise TypeError("dpi and export_scale must be positive numbers or None")
        selected_dpi = self._defaults.layout.export_dpi if dpi is None else float(dpi)
        if export_scale is not None:
            selected_dpi = self.surface_plan.logical_dpi * float(export_scale)
        if not math.isfinite(selected_dpi) or selected_dpi <= 0.0:
            raise ValueError("export dpi must be a positive finite number")
        target = Path(path)
        image_format = target.suffix.removeprefix(".").casefold()
        if not image_format:
            raise ValueError("plot export target must have an image suffix")
        options = dict(kwargs)
        options["format"] = image_format
        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                atomic_write_file(
                    target,
                    lambda stream: self._renderer.save(
                        stream,
                        dpi=selected_dpi,
                        **options,
                    ),
                )

    def rgba(self) -> np.ndarray:
        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                return self._renderer.rgba()

    def close(self) -> None:
        logical_completion: Future[FitResult | FacetFitBatchResult] | None = None
        with self._render_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._cancel_gesture()
                self._fit_cancel.set()
                self._live_fit_cancel.set()
                self._live_prepare_cancel.set()
                self._fit_context_generation += 1
                self._fit_request_generation += 1
                self._fit_warm_starts.clear()
                logical_completion = self._live_fit_completion
                self._live_fit_completion = None
                self._live_fit_request = None
                self._surface_callbacks.clear()
                self._display_callbacks.clear()
                self._viewport_callbacks.clear()
                self._fit_callbacks.clear()
                self._selection_subscriptions.clear()
        if logical_completion is not None and not logical_completion.done():
            logical_completion.set_exception(RuntimeError("plot session is closed"))
        caller_name = current_thread().name
        self._analysis_executor.shutdown(
            wait=not caller_name.startswith(f"{_ANALYSIS_THREAD_PREFIX}_"),
            cancel_futures=True,
        )

    def __enter__(self) -> "PlotSession":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


__all__ = [
    "DisplayDescription",
    "FacetFitBatchResult",
    "FitEvent",
    "FitScope",
    "FitSelection",
    "PlotInput",
    "PlotSession",
    "PulseTimelineSelectionData",
    "SelectionChange",
    "SelectionData",
    "SelectionEvent",
    "SelectorData",
    "SessionRevisions",
]
