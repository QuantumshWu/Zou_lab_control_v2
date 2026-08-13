"""Projection and fit semantics shared by sessions and live workers."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from zlc_data import BlockId, DatasetRevisionRef, OwnedSnapshot
from zlc_data.axis import point_ordinal_axis
from zlc_data.snapshot_projection import restrict_snapshot, value_selection

from .data_contract import (
    DEFAULT_UNITS,
    UnitRegistry,
    resolve_unit,
    snapshot_revision,
)

from ._pulse_time import pulse_time_scale
from .config import PlotLibraryDefaults
from .data_view import (
    AxisValue,
    CurveData,
    CurveSeries,
    HistogramData,
    ImageData,
    QuantityArray,
    RollingSample,
    aligned_histogram_edges,
)
from .fit import (
    FitModelSpec,
    FitParameterDisplay,
    FitResult,
    FitTarget,
    RegularImageFitInput,
    UnitRelation,
    _REGULAR_IMAGE_CAPABILITIES,
)
from .kinds import AxisDomain, AxisRef
from .primitives import PulseTimelineData
from ._fit_scene import (
    FitEllipseGlyph,
    FitOverlay,
    FitPolyline,
)
from ._kinds import handler_for
from .selectors import (
    CrosshairPoint,
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
)
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSpec,
    PulseTimelinePlot,
    RollingPlot,
    semantic_spec,
)
from .state import DisplayState
from ._validation import integer, readonly_copy


class FitScope(str, Enum):
    SELECTOR = "selector"
    VIEWPORT = "viewport"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class RollingHistoryPoint:
    """One successfully projected source revision in rolling history.

    ``shot_index`` is the absolute shot number: the seeded history counts
    0..R-1 and every appended revision takes the next integer.  The point
    owns it because the window cache truncates the history, so position in
    the list cannot recover the absolute axis.
    """

    sample: RollingSample
    shot_index: int


def _broadcast_all_true(mask: np.ndarray) -> bool:
    """True for a stride-0 broadcast plane that is constant True."""

    if mask.size == 0:
        return False
    if any(stride != 0 for stride in mask.strides):
        return False
    return bool(mask.flat[0])


_FIT_SELECTOR_KINDS = frozenset((
    SelectorKind.AREA,
    SelectorKind.X_RANGE,
    SelectorKind.THRESHOLD,
))

_DEFAULT_FIT_SELECTOR_PRIORITY = (
    SelectorKind.AREA,
    SelectorKind.X_RANGE,
)


@dataclass(frozen=True, slots=True)
class FitAuthority:
    selector: SelectorState | None
    viewport: RectangleRange | None
    focused_facet_index: int | None


@dataclass(frozen=True, slots=True, eq=False)
class FitSelection:
    data_revision: int
    scope: FitScope
    coordinates: tuple[np.ndarray, ...]
    observations: np.ndarray
    selected_indices: np.ndarray | None
    facet_index: int | None = None
    selector_kind: SelectorKind | None = None
    regular_image: RegularImageFitInput | None = None
    _authority: FitAuthority | None = None
    source_revisions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, FitScope):
            raise TypeError("fit selection scope must be FitScope")
        object.__setattr__(
            self,
            "data_revision",
            integer(
                self.data_revision,
                "fit selection data_revision",
                minimum=0,
            ),
        )
        regular_image = self.regular_image
        if regular_image is not None and not isinstance(
            regular_image,
            RegularImageFitInput,
        ):
            raise TypeError("regular_image must be RegularImageFitInput or None")
        object.__setattr__(
            self,
            "coordinates",
            tuple(readonly_copy(value, dtype=float) for value in self.coordinates),
        )
        if regular_image is None:
            observations = readonly_copy(self.observations, dtype=float)
        else:
            observations = np.asarray(regular_image.observations).view()
            observations.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        selected = self.selected_indices
        if selected is not None:
            selected = readonly_copy(selected, dtype=np.int64)
        elif regular_image is None:
            raise ValueError("non-image fit selections require selected_indices")
        object.__setattr__(self, "selected_indices", selected)
        object.__setattr__(
            self,
            "facet_index",
            integer(
                self.facet_index,
                "fit facet_index",
                minimum=0,
                optional=True,
            ),
        )
        selector_kind = self.selector_kind
        if selector_kind is not None:
            if not isinstance(selector_kind, SelectorKind):
                raise TypeError("fit selector_kind must be SelectorKind or None")
            if selector_kind not in _FIT_SELECTOR_KINDS:
                raise ValueError("crosshair selectors cannot define a fit")
        revisions = tuple(self.source_revisions) or (self.data_revision,)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in revisions
        ):
            raise TypeError("fit source_revisions must contain non-negative integers")
        object.__setattr__(self, "source_revisions", tuple(int(value) for value in revisions))

    @property
    def sample_count(self) -> int:
        if self.regular_image is None:
            return int(np.asarray(self.observations).size)
        valid = self.regular_image.valid_mask
        return (
            int(self.regular_image.observations.size)
            if valid is None
            else int(np.count_nonzero(valid))
        )


@dataclass(frozen=True, slots=True)
class HistogramProjection:
    bin_count: int
    edges: np.ndarray

    def __post_init__(self) -> None:
        bin_count = integer(
            self.bin_count,
            "histogram projection bin_count",
            minimum=1,
        )
        edges = readonly_copy(self.edges, dtype=float).reshape(-1)
        if edges.size != bin_count + 1:
            raise ValueError("histogram projection has the wrong edge count")
        if not bool(np.all(np.isfinite(edges))) or bool(
            np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("histogram projection edges must be finite and increasing")
        object.__setattr__(self, "bin_count", bin_count)
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True, slots=True)
class ProjectionContext:
    """Immutable owner or worker state consumed by one projection operation."""

    display_state: DisplayState
    selector_snapshot: SelectorSnapshot
    viewport: RectangleRange | None = None
    focused_facet_index: int | None = None
    rolling_history: tuple[RollingHistoryPoint, ...] = ()

    def selector_state(self, kind: SelectorKind) -> SelectorState:
        if not isinstance(kind, SelectorKind):
            raise TypeError("selector kind must be SelectorKind")
        for state in self.selector_snapshot.states:
            if state.kind is kind:
                return state
        raise KeyError(kind)


class FitProjection:
    """Projection state shared by owner-thread sessions and frozen workers."""

    @staticmethod
    def _validate_input(
        data: OwnedSnapshot | PulseTimelineData,
        spec: PlotSpec,
    ) -> None:
        if isinstance(spec, PulseTimelinePlot):
            if not isinstance(data, PulseTimelineData):
                raise TypeError("PulseTimelinePlot requires PulseTimelineData")
        elif isinstance(
            spec,
            (CurvePlot, ImagePlot, HistogramPlot, RollingPlot, FacetGridPlot),
        ):
            if not isinstance(data, OwnedSnapshot):
                raise TypeError(f"{type(spec).__name__} requires zlc_data.OwnedSnapshot")
        else:
            raise TypeError("unsupported plot specification")

    def __init__(
        self,
        *,
        data: OwnedSnapshot | PulseTimelineData,
        revision: int,
        spec: PlotSpec,
        context: ProjectionContext,
        unit_registry: UnitRegistry | None,
        defaults: PlotLibraryDefaults,
        histogram_projection: HistogramProjection | None,
    ) -> None:
        if not isinstance(context, ProjectionContext):
            raise TypeError("context must be ProjectionContext")
        self._spec = spec
        self._context = context
        self._unit_registry = unit_registry
        self._defaults = defaults
        if histogram_projection is not None and not isinstance(
            histogram_projection,
            HistogramProjection,
        ):
            raise TypeError(
                "histogram_projection must be HistogramProjection or None"
            )
        self._histogram_projection = histogram_projection
        self._view = None
        self._scoped_cache: tuple[object, OwnedSnapshot] | None = None
        self._payload = None
        selected_revision = integer(revision, "projection revision", minimum=0)
        self._validate_input(data, self._spec)
        if isinstance(data, OwnedSnapshot) and selected_revision != snapshot_revision(data):
            raise ValueError("OwnedSnapshot revision must equal projection revision")
        self._data = data
        self._revision = selected_revision

    def _with_context(self, context: ProjectionContext) -> "FitProjection":
        """Return a shallow immutable-data view bound to one context snapshot."""

        if not isinstance(context, ProjectionContext):
            raise TypeError("context must be ProjectionContext")
        selected = copy(self)
        selected._context = context
        return selected

    def _fork_frozen(
        self,
        *,
        data: OwnedSnapshot | PulseTimelineData,
        revision: int,
        context: ProjectionContext,
    ) -> "FitProjection":
        """Capture immutable worker inputs using this projection's configuration."""

        return FitProjection(
            data=data,
            revision=revision,
            spec=self._spec,
            context=context,
            unit_registry=self._unit_registry,
            defaults=self._defaults,
            histogram_projection=self._histogram_projection,
        )

    def _reproject(
        self,
        *,
        context: ProjectionContext,
        payload_only: bool = False,
    ) -> None:
        """Atomically rebuild derived view/payload state for one context."""

        if not isinstance(context, ProjectionContext):
            raise TypeError("context must be ProjectionContext")
        previous = (
            self._context,
            self._view,
            self._payload,
            self._histogram_projection,
        )
        try:
            self._context = context
            if payload_only:
                self._build_payload_from_view()
            else:
                self._build_view_and_payload()
        except Exception:
            (
                self._context,
                self._view,
                self._payload,
                self._histogram_projection,
            ) = previous
            raise

    @property
    def display_state(self) -> DisplayState:
        return self._context.display_state

    @property
    def data_revision(self) -> int:
        return self._revision

    @property
    def data(self) -> OwnedSnapshot | PulseTimelineData:
        return self._data

    @property
    def rolling_history(self) -> tuple[RollingHistoryPoint, ...]:
        return tuple(getattr(self, "_rolling_history_cache", self._context.rolling_history))

    @property
    def viewport(self) -> RectangleRange | None:
        return self._context.viewport

    @property
    def view(self) -> Any:
        return self._view

    @property
    def payload(self) -> Any:
        return self._payload

    @property
    def _viewport(self) -> RectangleRange | None:
        return self._context.viewport

    @property
    def _focused_facet_index(self) -> int | None:
        return self._context.focused_facet_index

    def _fit_target(self) -> FitTarget | None:
        semantic = self._semantic_spec()
        handler = handler_for(semantic)
        target = handler.fit_target
        return None if target is None else FitTarget(target)

    def _fit_model_units_compatible(self, model: FitModelSpec) -> bool:
        if self._view is None:
            return False
        try:
            sources = (
                (self._value_quantity(),)
                if self._is_histogram_plot()
                else (
                    (self._coordinate(self._x_ref()),)
                    if model.independent_arity == 1
                    else (
                        self._coordinate(self._x_ref()),
                        self._coordinate(self._y_axis_ref()),
                    )
                )
            )
            return all(
                source.canonical_unit.compatible_with(
                    self._fit_relation_quantity(relation).canonical_unit
                )
                for source, relation in zip(
                    sources,
                    model.coordinate_relations,
                    strict=True,
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _require_fit_model_compatible(self, model: FitModelSpec) -> None:
        target = self._fit_target()
        if target is None or target not in model.targets:
            name = "none" if target is None else target.value
            raise ValueError(
                f"fit model {model.model_id!r} is not authored for {name} plots"
            )
        if not self._fit_model_units_compatible(model):
            raise ValueError(
                f"fit model {model.model_id!r} is incompatible with the plot axes"
            )

    def _build_view_and_payload(self) -> None:
        if not isinstance(self._data, OwnedSnapshot):
            self._view = None
            self._payload = self._data
            return
        self._build_view()
        self._build_payload_from_view()

    def _scoped_data(self) -> OwnedSnapshot:
        """The source, cut down to the axes the panel is scoped to.

        The ONE place a panel narrows its own data.  Everything a panel shows
        is built from the view constructed below -- payload, fit, selectors,
        the side distribution -- so restricting here is what makes a scoped
        panel scoped all the way through, instead of drawing one thing and
        fitting another.
        """

        assert isinstance(self._data, OwnedSnapshot)
        scope = getattr(self._spec, "scope", ())
        if not scope:
            return self._data
        source = self._data
        schema = source.block.schema
        terms: dict[str, object] = {}
        for ref, value in scope:
            if ref.domain is AxisDomain.REPEAT:
                label = schema.repeat_axis.name
            elif ref.domain is AxisDomain.POINT_ROW:
                label = point_ordinal_axis(schema.point_table.row_count).name
            else:
                label = str(ref.axis_id)
            terms[label] = value
        digest = ",".join(f"{label}={value!r}" for label, value in sorted(terms.items()))
        key = (source.ref.block_id, source.ref.revision, digest)
        if self._scoped_cache is not None and self._scoped_cache[0] == key:
            return self._scoped_cache[1]

        def reference_for(derived_schema: object) -> DatasetRevisionRef:
            # Deterministic: the same source and the same scope name the same
            # derived block, so a rebuild does not look like new data to
            # anything downstream that remembers what it last drew.
            return DatasetRevisionRef(
                BlockId(f"{source.ref.block_id.value}|scope:{digest}"),
                source.ref.stream_generation,
                derived_schema.fingerprint,
                source.ref.revision,
            )

        scoped = restrict_snapshot(
            source,
            value_selection(schema, terms),
            reference_for=reference_for,
        )
        self._scoped_cache = (key, scoped)
        return scoped

    def _build_view(self) -> None:
        """Construct the unit-aware DataView without projecting any payload.

        The semantic feasibility probe stops here and runs the registry
        ``validate`` handlers on the view; only a committed replacement pays
        for aggregation.
        """

        assert isinstance(self._data, OwnedSnapshot)
        from .data_view import DataView

        values = self.display_state.values
        overrides: dict[AxisRef, object] = {}
        x_ref = getattr(self._semantic_spec(), "x", None)
        y_ref = getattr(self._semantic_spec(), "y", None)
        if isinstance(self._spec, FacetGridPlot):
            facet_unit = values.get("facet_display_unit")
            if facet_unit is not None:
                overrides[self._spec.facet] = facet_unit
        if x_ref is not None and values.get("x_display_unit") is not None:
            overrides[x_ref] = values.get("x_display_unit")
        if y_ref is not None and values.get("y_display_unit") is not None:
            overrides[y_ref] = values.get("y_display_unit")
        value_unit = values.get("value_display_unit")
        self._view = DataView(
            self._scoped_data(),
            axis_display_units=overrides,
            value_display_unit=value_unit,
            unit_registry=self._unit_registry,
        )

    def _build_payload_from_view(self) -> None:
        """Reproject the plot payload without rebuilding unit-aware DataView state."""

        if not isinstance(self._data, OwnedSnapshot):
            self._payload = self._data
            return
        if self._view is None:
            raise RuntimeError("dataset payload projection requires a DataView")
        handler_for(self._spec).build_payload(self, self._view, self.display_state)

    def _rolling_payload(
        self,
        history: tuple[RollingHistoryPoint, ...],
        *,
        window: int,
    ) -> CurveData:
        """Build one history series per optional rolling group."""

        if window <= 0:
            raise ValueError("rolling window must be positive")
        visible = history[-window:]
        visible_size = len(visible)
        if not visible:
            raise ValueError("rolling history cannot be empty")
        keys: list[tuple[AxisValue, ...]] = []
        for point in visible:
            for key in point.sample.group_keys:
                if key not in keys:
                    keys.append(key)
        # x is the absolute shot index each history point owns, so the axis
        # slides forward as the window fills instead of counting negative
        # "shots ago".
        x_values = np.asarray(
            [point.shot_index for point in visible], dtype=float
        )
        unit = self._view.samples.value.display_unit
        canonical_unit = self._view.samples.value.canonical_unit
        x_unit = resolve_unit("1", DEFAULT_UNITS)
        series: list[CurveSeries] = []
        for key in keys:
            canonical_values = np.full(visible_size, np.nan, dtype=float)
            valid = np.zeros(visible_size, dtype=np.bool_)
            for index, point in enumerate(visible):
                try:
                    source_index = point.sample.group_keys.index(key)
                except ValueError:
                    continue
                if bool(point.sample.valid[source_index]):
                    canonical_values[index] = float(point.sample.values[source_index])
                    valid[index] = True
            display_values = canonical_unit.convert_value_to(canonical_values, unit)
            x = QuantityArray(
                x_values,
                x_values,
                x_unit,
                x_unit,
                "Shot",
            )
            y = QuantityArray(
                canonical_values,
                display_values,
                canonical_unit,
                unit,
                self._view.samples.value.label,
            )
            label = self._view.samples.value.label if not key else ", ".join(
                item.label for item in key
            )
            series.append(
                CurveSeries(
                    x=x,
                    y=y,
                    valid=valid,
                    counts=valid.astype(np.int64),
                    group_key=key,
                    label=label,
                )
            )
        return CurveData(
            revision=visible[-1].sample.revision,
            generation=visible[-1].sample.generation,
            x_ref=AxisRef.point_rows(),
            group_by=(() if self._spec.group is None else (self._spec.group,)),
            series=tuple(series),
            source_revisions=tuple(point.sample.revision for point in visible),
        )

    def _histogram_bins(
        self,
        view: Any,
        state: DisplayState,
    ) -> np.ndarray:
        """Return stable display-unit edges for one histogram projection."""

        count = int(state["bin_count"])
        samples = view.samples
        canonical = np.asarray(samples.value.canonical).reshape(-1)
        valid = np.asarray(samples.valid_mask, dtype=bool).reshape(-1)
        finite = valid & np.isfinite(canonical)
        values = np.asarray(canonical[finite], dtype=float)
        previous = self._histogram_projection
        mode = str(state["relim_mode"])
        retain_domain = (
            previous is not None
            and previous.bin_count == count
            and mode != "tight"
        )
        if retain_domain:
            assert previous is not None
            low = float(previous.edges[0])
            high = float(previous.edges[-1])
            if values.size:
                data_low = float(np.min(values))
                data_high = float(np.max(values))
                if data_low < low or data_high > high:
                    envelope_low = min(low, data_low)
                    envelope_high = max(high, data_high)
                    padding = (
                        self._defaults.projection.histogram_domain_padding_fraction
                        * (envelope_high - envelope_low)
                    )
                    if data_low < low:
                        low = data_low - padding
                    if data_high > high:
                        high = data_high + padding
        else:
            if values.size:
                data_low = float(np.min(values))
                data_high = float(np.max(values))
            else:
                data_low, data_high = 0.0, 1.0
            if data_low == data_high:
                half = max(abs(data_low) * 0.05, 0.5)
                data_low -= half
                data_high += half
            low, high = data_low, data_high
            if mode != "tight":
                padding = (
                    self._defaults.projection.histogram_domain_padding_fraction
                    * (high - low)
                )
                low -= padding
                high += padding

        edges = aligned_histogram_edges(values, count, limits=(low, high))
        selected = HistogramProjection(
            len(edges) - 1,
            edges,
        )
        if previous is None or not (
            previous.bin_count == selected.bin_count
            and np.array_equal(previous.edges, selected.edges)
        ):
            previous = selected
            self._histogram_projection = previous
        assert previous is not None
        return np.asarray(
            samples.value.canonical_unit.convert_value_to(
                previous.edges,
                samples.value.display_unit,
            ),
            dtype=float,
        )

    def _facet_mask(self, facet_index: int | None = None) -> np.ndarray:
        if self._view is None:
            raise TypeError("facet masking requires zlc_data.OwnedSnapshot")
        if not isinstance(self._spec, FacetGridPlot):
            return np.ones(self._view.samples.shape, dtype=bool)
        cells = tuple(getattr(self._payload, "cells", ()))
        selected = self._focused_facet_index if facet_index is None else facet_index
        if selected is None or selected < 0 or selected >= len(cells):
            raise IndexError("facet index is outside the current grid")
        cell = cells[selected]
        coordinate = np.asarray(self._coordinate(self._spec.facet).canonical)
        return np.asarray(
            np.equal(coordinate, cell.facet_value_canonical),
            dtype=bool,
        )

    def _focused_payload(self, facet_index: int | None = None) -> Any:
        if not isinstance(self._spec, FacetGridPlot):
            return self._payload
        cells = tuple(getattr(self._payload, "cells", ()))
        selected = self._focused_facet_index if facet_index is None else facet_index
        if selected is None or selected < 0 or selected >= len(cells):
            raise IndexError("facet index is outside the current grid")
        return cells[selected].payload

    def _rolling_visible_mask(self) -> np.ndarray:
        if self._view is None:
            raise TypeError("rolling masking requires zlc_data.OwnedSnapshot")
        if not isinstance(self._spec, RollingPlot):
            return np.ones(self._view.samples.shape, dtype=bool)
        series = tuple(getattr(self._payload, "series", ()))
        if not series:
            return np.zeros(self._view.samples.shape, dtype=bool)
        shots = np.asarray(series[0].x.canonical, dtype=float).reshape(-1)
        if shots.size == 0:
            return np.zeros(self._view.samples.shape, dtype=bool)
        coordinate = np.asarray(self._coordinate(self._x_ref()).canonical)
        return np.isin(coordinate, shots)

    def _crosshair_sample_mask(
        self,
        state: SelectorState,
        valid: np.ndarray,
        point_transform: Callable[[np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        """Materialize the nearest valid plotted sample in display space."""

        if self._view is None or not isinstance(state.value, CrosshairPoint):
            raise TypeError("crosshair sample lookup requires zlc_data.OwnedSnapshot")
        displayed = self._display_selector_state(state)
        assert isinstance(displayed.value, CrosshairPoint)
        target = displayed.value
        samples = self._view.samples
        semantic = self._semantic_spec()
        candidate = np.asarray(valid, dtype=bool) & np.isfinite(
            np.asarray(samples.value.canonical, dtype=float)
        )
        candidate &= self._rolling_visible_mask()
        if isinstance(semantic, HistogramPlot):
            x_values = np.asarray(samples.value.display, dtype=float)
            y_values = np.full(samples.shape, target.y, dtype=float)
        else:
            x_coordinate = self._coordinate(self._x_ref())
            x_values = (
                self._coordinate_values_to_display(
                    np.asarray(x_coordinate.canonical, dtype=float),
                    self._x_ref(),
                )
                if isinstance(self._spec, RollingPlot)
                else np.asarray(x_coordinate.display, dtype=float)
            )
            y_values = (
                np.asarray(self._coordinate(self._y_axis_ref()).display, dtype=float)
                if isinstance(semantic, ImagePlot)
                else np.asarray(samples.value.display, dtype=float)
            )
        candidate &= np.isfinite(x_values) & np.isfinite(y_values)
        flat_indices = np.flatnonzero(candidate.reshape(-1))
        result = np.zeros(samples.shape, dtype=bool)
        if flat_indices.size == 0:
            return result

        points = np.column_stack(
            (
                x_values.reshape(-1)[flat_indices],
                y_values.reshape(-1)[flat_indices],
            )
        )
        target_point = np.asarray((target.x, target.y), dtype=float)
        if point_transform is not None:
            try:
                transformed = np.asarray(
                    point_transform(np.vstack((points, target_point))),
                    dtype=float,
                )
                if transformed.shape != (points.shape[0] + 1, 2):
                    raise ValueError("point transform returned the wrong shape")
                points, target_point = transformed[:-1], transformed[-1]
            except (TypeError, ValueError):
                pass
        finite = np.all(np.isfinite(points), axis=1)
        if not np.any(finite):
            return result
        flat_indices = flat_indices[finite]
        delta = np.abs(points[finite] - target_point)
        if isinstance(semantic, ImagePlot):
            nearest = int(np.argmin(np.hypot(delta[:, 0], delta[:, 1])))
        else:
            nearest = int(np.lexsort((delta[:, 1], delta[:, 0]))[0])
        result.reshape(-1)[flat_indices[nearest]] = True
        return result

    def _selector_mask(
        self,
        state: SelectorState,
        *,
        point_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        samples = self._view.samples
        mask = np.array(samples.valid_mask, copy=True, dtype=bool)
        mask &= self._facet_mask(state.facet_index)
        value = np.asarray(samples.value.canonical)
        if state.kind is SelectorKind.X_RANGE:
            source = self._x_selector_source()
            coordinate = np.asarray(
                self._coordinate(source).canonical
                if isinstance(source, AxisRef)
                else source.canonical
            )
            assert isinstance(state.value, NumericRange)
            mask &= (coordinate >= state.value.low) & (coordinate <= state.value.high)
        elif state.kind is SelectorKind.AREA:
            assert isinstance(state.value, RectangleRange)
            x_source = self._x_selector_source()
            x = np.asarray(
                self._coordinate(x_source).canonical
                if isinstance(x_source, AxisRef)
                else x_source.canonical
            )
            mask &= (x >= state.value.x.low) & (x <= state.value.x.high)
            semantic = self._semantic_spec()
            if isinstance(semantic, ImagePlot):
                y = np.asarray(
                    self._coordinate(self._y_axis_ref()).canonical
                )
                mask &= (y >= state.value.y.low) & (y <= state.value.y.high)
            elif not isinstance(semantic, HistogramPlot):
                # A one-dimensional curve's vertical display coordinate is
                # the observation itself.  AREA therefore filters both the
                # independent coordinate and the canonical observation; this
                # The raw-sample mask is materialised only by selector_data().
                mask &= (value >= state.value.y.low) & (value <= state.value.y.high)
        elif state.kind is SelectorKind.THRESHOLD:
            mask &= value >= float(state.value)
        elif state.kind is SelectorKind.CROSSHAIR:
            return self._crosshair_sample_mask(state, mask, point_transform)
        return mask & np.isfinite(value)

    def _fit_selector(
        self,
        selector_kind: SelectorKind | None = None,
    ) -> SelectorState | None:
        if selector_kind is not None and not isinstance(selector_kind, SelectorKind):
            raise TypeError("selector_kind must be SelectorKind or None")
        if (
            selector_kind is not None
            and selector_kind not in _FIT_SELECTOR_KINDS
        ):
            raise ValueError("crosshair selectors cannot define a fit")

        states = {
            state.kind: state for state in self._context.selector_snapshot.states
        }

        def usable(state: SelectorState | None) -> bool:
            return bool(
                state is not None
                and state.kind in _FIT_SELECTOR_KINDS
                and state.facet_index == self._focused_facet_index
            )

        if selector_kind is not None:
            selected = states.get(selector_kind)
            if selected is None:
                raise KeyError(selector_kind)
            if not usable(selected):
                raise ValueError(
                    "fit selector must belong to the focused facet and contain "
                    "a numeric selection"
                )
            return selected

        for kind in _DEFAULT_FIT_SELECTOR_PRIORITY:
            selected = states.get(kind)
            if usable(selected):
                return selected
        return None

    def _fit_selection_authority(
        self,
        selector_kind: SelectorKind | None,
    ) -> FitAuthority:
        """Resolve the selector-or-viewport precedence once for all fit paths."""

        selector = self._fit_selector(selector_kind)
        viewport = None
        if selector is None and self._viewport is not None:
            viewport = (
                self._viewport_in_canonical()
                if self._view is not None
                else self._viewport
            )
        return FitAuthority(
            selector,
            viewport,
            self._focused_facet_index,
        )

    def fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
        facet_index: int | None = None,
    ) -> FitSelection:
        """Select the data one fit runs on.

        ``facet_index`` names the cell to compute; the focused facet is the
        cell the operator is looking at, and so the one whose selectors may
        define the domain.  A grid batch computes every cell over the single
        region that was drawn, so the two differ there and nowhere else.
        """

        if self._view is None:
            raise TypeError("fit is available only for zlc_data.OwnedSnapshot plots")
        if not isinstance(model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        self._require_fit_model_compatible(model)
        if facet_index is None:
            facet_index = self._focused_facet_index
        payload = self._focused_payload(facet_index)
        if isinstance(payload, CurveData):
            return self._curve_fit_selection(
                model,
                selector_kind=selector_kind,
                payload=payload,
                facet_index=facet_index,
            )
        if isinstance(payload, ImageData) and (
            model.capabilities & _REGULAR_IMAGE_CAPABILITIES
        ):
            return self._regular_image_fit_selection(
                model,
                selector_kind=selector_kind,
                payload=payload,
                facet_index=facet_index,
            )
        if self._is_histogram_plot():
            if not isinstance(payload, HistogramData):
                raise RuntimeError("histogram projection did not produce histogram data")
            return self._histogram_fit_selection(
                model,
                selector_kind=selector_kind,
                payload=payload,
                facet_index=facet_index,
            )
        if not isinstance(payload, ImageData):
            raise TypeError("unsupported fit projection payload")
        return self._image_fit_selection(
            model,
            selector_kind=selector_kind,
            payload=payload,
            facet_index=facet_index,
        )

    def _curve_fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        payload: CurveData,
        facet_index: int | None = None,
    ) -> FitSelection:
        """Fit the first painted series, with scope applied to that series."""

        if self._view is None:
            raise TypeError("curve fitting requires zlc_data.OwnedSnapshot")
        if model.independent_arity != 1:
            raise ValueError("curve fit models require exactly one independent axis")
        series = tuple(payload.series)
        if not series:
            raise ValueError("painted curve has no series")
        source = series[0]
        x_canonical = np.asarray(source.x.canonical, dtype=float).reshape(-1)
        y_canonical = np.asarray(source.y.canonical, dtype=float).reshape(-1)
        valid = (
            np.asarray(source.valid, dtype=bool).reshape(-1)
            & np.isfinite(x_canonical)
            & np.isfinite(y_canonical)
        )

        authority = self._fit_selection_authority(selector_kind)
        active = authority.selector
        if active is not None:
            value = active.value
            if active.kind is SelectorKind.X_RANGE:
                assert isinstance(value, NumericRange)
                valid &= (x_canonical >= value.low) & (x_canonical <= value.high)
            elif active.kind is SelectorKind.AREA:
                assert isinstance(value, RectangleRange)
                # A fit domain restricts the COORDINATE, never the value being
                # fitted: dropping samples for lying outside the box vertically
                # is outlier surgery nobody asked for, and it deleted the peak
                # from every fit whose box did not reach over it.  An image
                # domain has always meant this (both of its axes ARE
                # coordinates); a curve's y is the observation.
                valid &= (x_canonical >= value.x.low) & (
                    x_canonical <= value.x.high
                )
            elif active.kind is SelectorKind.THRESHOLD:
                valid &= y_canonical >= float(value)
            else:
                raise ValueError("selected geometry cannot define a curve fit domain")
            scope = FitScope.SELECTOR
        elif authority.viewport is not None:
            viewport = authority.viewport
            valid &= (x_canonical >= viewport.x.low) & (
                x_canonical <= viewport.x.high
            )
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL

        indices = np.flatnonzero(valid)
        coordinates = self._fit_coordinate_values_to_solver(
            x_canonical[valid],
            source.x,
            model.coordinate_relations[0],
        )
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(coordinates,),
            observations=y_canonical[valid],
            selected_indices=indices,
            facet_index=facet_index,
            selector_kind=None if active is None else active.kind,
            _authority=authority,
            source_revisions=payload.source_revisions,
        )

    def _histogram_fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        payload: HistogramData,
        facet_index: int | None = None,
    ) -> FitSelection:
        """Fit the exact bins painted by the current histogram projection."""

        if bool(self.display_state["density"]) or bool(
            self.display_state["cumulative"]
        ):
            raise ValueError(
                "histogram fitting requires count projection; set density=False "
                "and cumulative=False"
            )
        canonical = np.asarray(payload.centers.canonical, dtype=float).reshape(-1)
        counts = np.asarray(payload.counts, dtype=float).reshape(-1)
        valid = np.isfinite(canonical) & np.isfinite(counts)

        authority = self._fit_selection_authority(selector_kind)
        active = authority.selector
        if active is not None:
            if active.kind is SelectorKind.X_RANGE:
                value = active.value
                assert isinstance(value, NumericRange)
                valid &= (canonical >= value.low) & (canonical <= value.high)
            elif active.kind is SelectorKind.AREA:
                value = active.value
                assert isinstance(value, RectangleRange)
                # The bin CENTRE is this plot's coordinate; the count is what
                # is being fitted.  A box that does not reach the tallest bins
                # used to delete exactly the peak of the distribution.
                valid &= (canonical >= value.x.low) & (canonical <= value.x.high)
            elif active.kind is SelectorKind.THRESHOLD:
                valid &= canonical >= float(active.value)
            else:
                raise ValueError(
                    "selected geometry cannot define a histogram fit domain"
                )
            scope = FitScope.SELECTOR
        elif authority.viewport is not None:
            viewport = authority.viewport
            valid &= (canonical >= viewport.x.low) & (canonical <= viewport.x.high)
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL

        indices = np.flatnonzero(valid)
        model_centers = self._fit_coordinate_values_to_solver(
            canonical[valid],
            payload.centers,
            model.coordinate_relations[0],
        )
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(model_centers,),
            observations=counts[valid],
            selected_indices=indices,
            facet_index=facet_index,
            selector_kind=None if active is None else active.kind,
            _authority=authority,
            source_revisions=payload.source_revisions,
        )

    def _regular_image_fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        payload: ImageData,
        facet_index: int | None = None,
    ) -> FitSelection:
        """Describe a painted regular image without expanding coordinate grids."""

        x_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.x.canonical, dtype=float),
            payload.x,
            model.coordinate_relations[0],
        ).reshape(-1)
        y_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.y.canonical, dtype=float),
            payload.y,
            model.coordinate_relations[1],
        ).reshape(-1)
        valid, observations, scope, active, authority = self._image_fit_domain(
            payload,
            selector_kind,
        )
        finite_x = np.isfinite(x_solver)
        finite_y = np.isfinite(y_solver)
        if not (bool(np.all(finite_x)) and bool(np.all(finite_y))):
            plane = finite_y[:, None] & finite_x[None, :]
            valid = plane if valid is None else valid & plane

        regular = RegularImageFitInput(
            x_solver,
            y_solver,
            observations,
            valid_mask=(
                None if valid is None or bool(np.all(valid)) else valid
            ),
        )
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(x_solver, y_solver),
            observations=observations,
            selected_indices=None,
            facet_index=facet_index,
            selector_kind=None if active is None else active.kind,
            regular_image=regular,
            _authority=authority,
            source_revisions=payload.source_revisions,
        )

    def _image_fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        payload: ImageData,
        facet_index: int | None = None,
    ) -> FitSelection:
        """Fit the painted image projection without returning to raw samples."""

        if model.independent_arity != 2:
            raise ValueError("image fit models require exactly two independent axes")
        x_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.x.canonical, dtype=float),
            payload.x,
            model.coordinate_relations[0],
        ).reshape(-1)
        y_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.y.canonical, dtype=float),
            payload.y,
            model.coordinate_relations[1],
        ).reshape(-1)
        valid, observations, scope, active, authority = self._image_fit_domain(
            payload,
            selector_kind,
        )
        finite_x = np.isfinite(x_solver)
        finite_y = np.isfinite(y_solver)
        if not (bool(np.all(finite_x)) and bool(np.all(finite_y))):
            plane = finite_y[:, None] & finite_x[None, :]
            valid = plane if valid is None else valid & plane
        x_solver_grid = np.broadcast_to(x_solver[None, :], observations.shape)
        y_solver_grid = np.broadcast_to(y_solver[:, None], observations.shape)
        if valid is None:
            coordinates = (
                x_solver_grid.reshape(-1),
                y_solver_grid.reshape(-1),
            )
            selected_observations = observations.reshape(-1)
            selected_indices = np.arange(observations.size, dtype=np.int64)
        else:
            selected = valid.reshape(-1)
            coordinates = (
                x_solver_grid.reshape(-1)[selected],
                y_solver_grid.reshape(-1)[selected],
            )
            selected_observations = observations.reshape(-1)[selected]
            selected_indices = np.flatnonzero(selected)
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=coordinates,
            observations=selected_observations,
            selected_indices=selected_indices,
            facet_index=facet_index,
            selector_kind=None if active is None else active.kind,
            _authority=authority,
            source_revisions=payload.source_revisions,
        )

    def _image_fit_domain(
        self,
        payload: ImageData,
        selector_kind: SelectorKind | None,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray,
        FitScope,
        SelectorState | None,
        FitAuthority,
    ]:
        """Resolve one canonical mask over the already projected image.

        A returned mask of ``None`` means every pixel is valid.  Integer and
        boolean observations skip the always-true ``np.isfinite`` sweep, and a
        stride-0 all-True broadcast validity plane short-circuits the mask
        chain entirely, so the plain full-frame fit never touches a
        full-image boolean plane.
        """

        x = np.asarray(payload.x.canonical, dtype=float).reshape(-1)
        y = np.asarray(payload.y.canonical, dtype=float).reshape(-1)
        observations = np.asarray(payload.z.canonical)
        valid: np.ndarray | None = None
        source = np.asarray(payload.valid, dtype=bool)
        if not _broadcast_all_true(source):
            valid = source
        if observations.dtype.kind in "fc":
            finite = np.isfinite(observations)
            valid = finite if valid is None else valid & finite
        finite_x = np.isfinite(x)
        finite_y = np.isfinite(y)
        if not (bool(np.all(finite_x)) and bool(np.all(finite_y))):
            plane = finite_y[:, None] & finite_x[None, :]
            valid = plane if valid is None else valid & plane
        authority = self._fit_selection_authority(selector_kind)
        active = authority.selector
        if active is not None:
            value = active.value
            if active.kind is SelectorKind.X_RANGE:
                assert isinstance(value, NumericRange)
                columns = (x >= value.low) & (x <= value.high)
                band = np.broadcast_to(columns[None, :], observations.shape)
                valid = band if valid is None else valid & band
            elif active.kind is SelectorKind.AREA:
                assert isinstance(value, RectangleRange)
                columns = (x >= value.x.low) & (x <= value.x.high)
                rows = (y >= value.y.low) & (y <= value.y.high)
                box = rows[:, None] & columns[None, :]
                valid = box if valid is None else valid & box
            elif active.kind is SelectorKind.THRESHOLD:
                above = observations >= float(value)
                valid = above if valid is None else valid & above
            else:
                raise ValueError("selected geometry cannot define an image fit domain")
            scope = FitScope.SELECTOR
        elif authority.viewport is not None:
            viewport = authority.viewport
            columns = (x >= viewport.x.low) & (x <= viewport.x.high)
            rows = (y >= viewport.y.low) & (y <= viewport.y.high)
            box = rows[:, None] & columns[None, :]
            valid = box if valid is None else valid & box
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL
        return valid, observations, scope, active, authority

    def _viewport_in_canonical(self) -> RectangleRange:
        assert self._viewport is not None
        return RectangleRange(
            self._display_range_to_canonical(
                self._viewport.x, self._x_selector_source()
            ),
            self._viewport.y
            if self._is_histogram_plot()
            else self._display_range_to_canonical(
                self._viewport.y, self._y_ref_or_value()
            ),
        )

    def _fit_relation_quantity(self, relation: UnitRelation) -> Any:
        """Resolve one model unit relation to the plot's authoritative quantity."""

        if relation is UnitRelation.VALUE:
            return self._value_quantity()
        if relation is UnitRelation.AXIS_0:
            return (
                self._value_quantity()
                if self._is_histogram_plot()
                else self._coordinate(self._x_ref())
            )
        if relation is UnitRelation.AXIS_1:
            return self._coordinate(self._y_axis_ref())
        return None

    def _fit_coordinate_values_to_solver(
        self,
        values: np.ndarray,
        source_quantity: Any,
        solver_relation: UnitRelation,
    ) -> np.ndarray:
        """Convert a painted coordinate's canonical values into model units."""

        target_quantity = self._fit_relation_quantity(solver_relation)
        if target_quantity is None:
            raise ValueError(
                "fit coordinate relations must identify a plot coordinate axis"
            )
        source_unit = source_quantity.canonical_unit
        target_unit = target_quantity.canonical_unit
        if not source_unit.compatible_with(target_unit):
            raise ValueError(
                "fit model coordinate axes require compatible canonical units"
            )
        return np.asarray(
            source_unit.convert_value_to(values, target_unit),
            dtype=float,
        )

    def _fit_overlay_curve_domain(
        self,
        result: FitResult,
        selection: FitSelection,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the painted one-dimensional domain in solver/display units.

        The curve is drawn where it was solved.  Outside that window it is not
        a claim about anything, and for a decay -- whose origin is the window
        start -- the extrapolation runs away from the data within a few
        samples.
        """

        payload = self._focused_payload(selection.facet_index)
        if self._is_histogram_plot() and hasattr(payload, "centers"):
            centers = payload.centers
            canonical = self._fit_coordinate_values_to_solver(
                np.asarray(centers.canonical, dtype=float).reshape(-1),
                centers,
                result.model.coordinate_relations[0],
            )
            return self._clip_to_fitted_domain(
                canonical,
                np.asarray(centers.display, dtype=float).reshape(-1),
                selection,
            )

        series = tuple(getattr(payload, "series", ()))
        if not series:
            raise RuntimeError("one-dimensional fit overlay requires a painted series")
        x = series[0].x
        canonical = self._fit_coordinate_values_to_solver(
            np.asarray(x.canonical, dtype=float).reshape(-1),
            x,
            result.model.coordinate_relations[0],
        )
        return self._clip_to_fitted_domain(
            canonical,
            np.asarray(x.display, dtype=float).reshape(-1),
            selection,
        )

    @staticmethod
    def _clip_to_fitted_domain(
        canonical: np.ndarray,
        display: np.ndarray,
        selection: FitSelection,
    ) -> tuple[np.ndarray, np.ndarray]:
        fitted = np.asarray(selection.coordinates[0], dtype=float).reshape(-1)
        fitted = fitted[np.isfinite(fitted)]
        if fitted.size == 0:
            return canonical, display
        inside = (canonical >= float(np.min(fitted))) & (
            canonical <= float(np.max(fitted))
        )
        if not bool(np.any(inside)):
            return canonical, display
        return canonical[inside], display[inside]

    def _fit_solver_coordinate_to_display(
        self,
        values: np.ndarray,
        solver_relation: UnitRelation,
        display_relation: UnitRelation,
    ) -> np.ndarray:
        source_quantity = self._fit_relation_quantity(solver_relation)
        target_quantity = self._fit_relation_quantity(display_relation)
        if source_quantity is None or target_quantity is None:
            raise ValueError("fit coordinate display requires physical axis relations")
        if isinstance(self._spec, RollingPlot) and (
            solver_relation is UnitRelation.AXIS_0
            and display_relation is UnitRelation.AXIS_0
        ):
            # Shot ordinals: canonical and display coincide.
            return np.asarray(values, dtype=float)
        source_unit = source_quantity.canonical_unit
        target_unit = target_quantity.display_unit
        if not source_unit.compatible_with(target_unit):
            raise ValueError("fit coordinate solver and display units are incompatible")
        return np.asarray(
            source_unit.convert_value_to(values, target_unit),
            dtype=float,
        )

    def _fit_overlay_polylines(
        self,
        result: FitResult,
        selection: FitSelection,
    ) -> tuple[FitPolyline, ...]:
        if not result.success or result.model.independent_arity != 1:
            return ()
        presentation = result.model.presentation
        canonical, display_x = self._fit_overlay_curve_domain(result, selection)
        if not presentation.components:
            fitted = result.model.evaluate(
                (canonical,),
                result.parameter_values,
            ).reshape(-1)
            fitted_display = (
                fitted
                if self._is_histogram_plot()
                else self._convert_coordinate_array_to_display(
                    fitted,
                    self._value_quantity(),
                )
            )
            role = "total" if self._is_histogram_plot() else "primary"
            return (FitPolyline(display_x, fitted_display, role=role),)

        source = np.asarray(canonical, dtype=float).reshape(-1)
        finite = source[np.isfinite(source)]
        if finite.size < 2:
            return ()
        sample_count = self._defaults.style.artists.fit_component_sample_count
        dense = np.linspace(float(np.min(finite)), float(np.max(finite)), sample_count)
        display_x = self._fit_solver_coordinate_to_display(
            dense,
            result.model.coordinate_relations[0],
            UnitRelation.AXIS_0,
        )
        component_values: dict[str, np.ndarray] = {}
        for component in presentation.components:
            component_values[component.component_id] = (
                result.model.evaluate_component(
                    component.component_id,
                    (dense,),
                    result.parameter_values,
                ).reshape(-1)
            )
        converted_components = {
            name: (
                values
                if self._is_histogram_plot()
                else self._convert_coordinate_array_to_display(
                    values,
                    self._value_quantity(),
                )
            )
            for name, values in component_values.items()
        }
        ordered_components = tuple(
            component_values[component.component_id]
            for component in presentation.components
        )
        total = ordered_components[0].copy()
        for component_values_array in ordered_components[1:]:
            total += component_values_array
        if not self._is_histogram_plot():
            total = self._convert_coordinate_array_to_display(
                total,
                self._value_quantity(),
            )
        polylines = tuple(
            FitPolyline(
                display_x,
                converted_components[component.component_id],
                role="component",
                component_index=index,
            )
            for index, component in enumerate(presentation.components)
        ) + (FitPolyline(display_x, total, role="total"),)
        return polylines

    def _fit_overlay_ellipse(
        self,
        result: FitResult,
        parameter_display: tuple[FitParameterDisplay, ...],
    ) -> FitEllipseGlyph | None:
        glyph = result.model.presentation.ellipse_glyph
        if glyph is None or not result.success:
            return None
        center_indices = tuple(
            result.model.parameter_index(name) for name in glyph.center_parameters
        )
        center_x = parameter_display[center_indices[0]].value
        center_y = parameter_display[center_indices[1]].value
        radii = []
        for parameter_name, display_relation in zip(
            glyph.radius_parameters,
            (UnitRelation.AXIS_0, UnitRelation.AXIS_1),
            strict=True,
        ):
            radius_index = result.model.parameter_index(parameter_name)
            radius_spec = result.model.parameters[radius_index]
            radius, _unit = self._display_fit_parameter_value(
                radius_spec,
                float(result.parameter_values[radius_index]),
                difference=True,
                display_relation=display_relation,
            )
            radii.append(abs(radius))
        return FitEllipseGlyph(center_x, center_y, radii[0], radii[1])

    def _make_fit_overlay(
        self,
        result: FitResult,
        selection: FitSelection,
    ) -> FitOverlay:
        parameter_display = self._display_fit_parameters(result)
        headline_parameter = next(
            (
                parameter
                for parameter in parameter_display
                if parameter.name == result.model.headline
            ),
            None,
        )
        polylines = self._fit_overlay_polylines(
            result,
            selection,
        )
        return FitOverlay(
            polylines=polylines,
            ellipse_glyph=self._fit_overlay_ellipse(result, parameter_display),
            success=result.success,
            formula=result.model.formula or "",
            parameter_display=parameter_display,
            diagnostic=result.message,
            facet_index=selection.facet_index,
            headline_parameter=headline_parameter,
        )

    def _display_fit_parameters(
        self,
        result: FitResult,
    ) -> tuple[FitParameterDisplay, ...]:
        """Convert fit values and uncertainties into the painted units."""

        rows: list[FitParameterDisplay] = []
        for index, (spec, raw) in enumerate(zip(
            result.model.parameters,
            result.parameter_values,
            strict=True,
        )):
            value, unit = self._display_fit_parameter_value(spec, float(raw))
            error = None
            if result.covariance_valid:
                error, _error_unit = self._display_fit_parameter_value(
                    spec,
                    float(result.standard_errors[index]),
                    difference=True,
                )
                error = abs(error)
            rows.append(
                FitParameterDisplay(
                    name=spec.name,
                    label=spec.display_label or spec.name,
                    value=value,
                    standard_error=error,
                    unit=unit,
                )
            )
        return tuple(rows)

    def _fit_parameter_units(self, model: FitModelSpec) -> Mapping[str, str]:
        """Return canonical units for the canonical solver parameter values."""

        return MappingProxyType({
            spec.name: self._canonical_fit_parameter_unit(spec)
            for spec in model.parameters
        })

    def _canonical_fit_parameter_unit(self, spec: Any) -> str:
        """Resolve a fit parameter's unit without consulting display overrides."""

        relation = spec.unit_relation
        solver_relation = spec.solver_unit_relation
        if relation is UnitRelation.RADIAN:
            if solver_relation is not UnitRelation.RADIAN:
                raise ValueError("radian fit parameters require a radian solver relation")
            return "rad"
        if relation is UnitRelation.VALUE and self._is_histogram_plot():
            if solver_relation is not UnitRelation.VALUE:
                raise ValueError("histogram count parameters require value solver units")
            quantity = self._value_quantity()
            return "" if quantity.canonical_unit.symbol == "1" else quantity.canonical_unit.symbol
        if isinstance(self._spec, RollingPlot) and relation in {
            UnitRelation.AXIS_0,
            UnitRelation.INVERSE_AXIS_0,
        }:
            if relation is UnitRelation.INVERSE_AXIS_0:
                return "1/point"
            return "point"
        if relation is UnitRelation.INVERSE_AXIS_0:
            quantity = self._fit_relation_quantity(UnitRelation.AXIS_0)
            if quantity is None:
                return ""
            canonical = quantity.canonical_unit
            registry = self._unit_registry or DEFAULT_UNITS
            inverse = registry.inverse_for(canonical)
            if inverse is not None:
                return inverse.symbol
            symbol = canonical.symbol
            return "" if symbol == "1" else f"1/{symbol}"
        quantity = self._fit_relation_quantity(solver_relation)
        if quantity is None:
            return ""
        symbol = quantity.canonical_unit.symbol
        return "" if symbol == "1" else symbol

    def _display_fit_parameter_value(
        self,
        spec: Any,
        value: float,
        *,
        difference: bool = False,
        display_relation: UnitRelation | None = None,
    ) -> tuple[float, str]:
        relation = spec.unit_relation if display_relation is None else display_relation
        solver_relation = spec.solver_unit_relation
        if relation is UnitRelation.RADIAN:
            if solver_relation is not UnitRelation.RADIAN:
                raise ValueError("radian display requires a radian solver parameter")
            return value, "rad"
        if relation is UnitRelation.VALUE and self._is_histogram_plot():
            if solver_relation is not UnitRelation.VALUE:
                raise ValueError("histogram count parameters require value solver units")
            return value, "count"
        if isinstance(self._spec, RollingPlot) and relation in {
            UnitRelation.AXIS_0,
            UnitRelation.INVERSE_AXIS_0,
        }:
            if solver_relation is not relation:
                raise ValueError("rolling fit parameters cannot cross unit relations")
            # The rolling shot axis is a plain ordinal (canonical == display
            # == absolute shot index), so fit parameters cross unchanged.
            if relation is UnitRelation.INVERSE_AXIS_0:
                return value, "1/point"
            return value, "point"

        if relation is UnitRelation.INVERSE_AXIS_0:
            if solver_relation is not UnitRelation.INVERSE_AXIS_0:
                raise ValueError("inverse-axis parameters cannot cross unit relations")
            quantity = self._fit_relation_quantity(UnitRelation.AXIS_0)
            if quantity is None:
                return value, ""
            canonical_unit = quantity.canonical_unit
            display_unit = quantity.display_unit
            converted = value * float(display_unit.scale) / float(canonical_unit.scale)
            registry = self._unit_registry or DEFAULT_UNITS
            inverse = registry.inverse_for(display_unit)
            if inverse is not None:
                return converted, inverse.symbol
            symbol = display_unit.symbol
            return converted, "" if symbol == "1" else f"1/{symbol}"

        source_quantity = self._fit_relation_quantity(solver_relation)
        target_quantity = self._fit_relation_quantity(relation)
        if source_quantity is None or target_quantity is None:
            return value, ""
        canonical_unit = source_quantity.canonical_unit
        display_unit = target_quantity.display_unit
        if not canonical_unit.compatible_with(display_unit):
            raise ValueError("fit parameter solver and display units are incompatible")
        if spec.affine_point and not difference:
            converted = float(
                np.asarray(
                    canonical_unit.convert_value_to((value,), display_unit),
                    dtype=float,
                ).reshape(-1)[0]
            )
        else:
            converted = value * float(canonical_unit.scale) / float(display_unit.scale)
        unit = "" if display_unit.symbol == "1" else display_unit.symbol
        return converted, unit

    def _semantic_spec(self) -> Any:
        return semantic_spec(self._spec)

    def _is_histogram_plot(self) -> bool:
        return isinstance(self._semantic_spec(), HistogramPlot)

    def _x_ref(self) -> AxisRef:
        semantic = self._semantic_spec()
        if isinstance(semantic, RollingPlot):
            # Rolling history owns its ordinal x coordinate; this private
            # placeholder is used only where the generic unit/selector code
            # requires an AxisRef-shaped token.
            return AxisRef.point_rows()
        ref = getattr(semantic, "x", None)
        if not isinstance(ref, AxisRef):
            raise TypeError("this plot has no coordinate x axis")
        return ref

    def _x_selector_source(self) -> AxisRef | Any:
        return self._value_quantity() if self._is_histogram_plot() else self._x_ref()

    def _y_axis_ref(self) -> AxisRef:
        semantic = self._semantic_spec()
        ref = getattr(semantic, "y", None)
        if not isinstance(ref, AxisRef):
            raise TypeError("the selected fit model requires a plot y-coordinate axis")
        return ref

    def _coordinate(self, ref: AxisRef) -> Any:
        if self._view is None:
            raise TypeError("coordinate access requires zlc_data.OwnedSnapshot")
        return self._view.coordinate(ref)

    def _value_quantity(self) -> Any:
        if self._view is None:
            raise TypeError("value access requires zlc_data.OwnedSnapshot")
        return self._view.samples.value

    def _y_ref_or_value(self) -> AxisRef | Any:
        semantic = self._semantic_spec()
        ref = getattr(semantic, "y", None)
        return ref if isinstance(ref, AxisRef) else self._value_quantity()

    def _display_scalar_to_canonical(
        self, value: float, source: AxisRef | Any
    ) -> float:
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        unit = quantity.display_unit
        return float(np.asarray(unit.to_canonical([value])).reshape(-1)[0])

    def _canonical_scalar_to_display(
        self, value: float, source: AxisRef | Any
    ) -> float:
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        canonical = quantity.canonical_unit
        display = quantity.display_unit
        return float(np.asarray(canonical.convert_value_to([value], display)).reshape(-1)[0])

    def _display_range_to_canonical(
        self, value: NumericRange, source: AxisRef | Any
    ) -> NumericRange:
        if isinstance(self._spec, RollingPlot) and source == self._x_ref():
            # The rolling shot axis is a plain ordinal: canonical and display
            # coincide, and its placeholder x ref must never be routed through
            # coordinate unit conversion.
            return NumericRange(*sorted((float(value.low), float(value.high))))
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        return NumericRange(
            self._display_scalar_to_canonical(value.low, quantity),
            self._display_scalar_to_canonical(value.high, quantity),
        )

    def _canonical_range_to_display(
        self, value: NumericRange, source: AxisRef | Any
    ) -> NumericRange:
        if isinstance(self._spec, RollingPlot) and source == self._x_ref():
            return NumericRange(*sorted((float(value.low), float(value.high))))
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        return NumericRange(
            self._canonical_scalar_to_display(value.low, quantity),
            self._canonical_scalar_to_display(value.high, quantity),
        )

    @staticmethod
    def _convert_coordinate_array_to_display(values: np.ndarray, quantity: Any) -> np.ndarray:
        return np.asarray(
            quantity.canonical_unit.convert_value_to(values, quantity.display_unit),
            dtype=float,
        )

    def _pulse_x_factor(self) -> float:
        if not isinstance(self._spec, PulseTimelinePlot) or not isinstance(
            self._data, PulseTimelineData
        ):
            raise TypeError("pulse time conversion requires PulseTimelinePlot")
        factor, _unit = pulse_time_scale(
            self._data,
            self.display_state.values.get("x_display_unit"),
        )
        return factor

    def _pulse_source_range_to_display(
        self, value: NumericRange
    ) -> NumericRange:
        factor = self._pulse_x_factor()
        return NumericRange(value.low * factor, value.high * factor)

    def _canonical_x_scalar_to_display(self, value: float) -> float:
        if isinstance(self._spec, RollingPlot):
            return float(value)
        source = self._x_selector_source()
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        return self._canonical_scalar_to_display(value, quantity)

    def _coordinate_values_to_display(
        self, values: np.ndarray, ref: AxisRef
    ) -> np.ndarray:
        if isinstance(self._spec, RollingPlot) and ref == self._x_ref():
            return np.asarray(values, dtype=float)
        return self._convert_coordinate_array_to_display(
            values, self._coordinate(ref)
        )

    def _area_canonical_to_display(
        self,
        value: RectangleRange,
    ) -> RectangleRange:
        if self._view is not None:
            x = self._canonical_range_to_display(
                value.x,
                self._x_selector_source(),
            )
            y = (
                value.y
                if self._is_histogram_plot()
                else self._canonical_range_to_display(
                    value.y,
                    self._y_ref_or_value(),
                )
            )
            return RectangleRange(x, y)
        if isinstance(self._spec, PulseTimelinePlot):
            return RectangleRange(
                self._pulse_source_range_to_display(value.x),
                value.y,
            )
        return value

    def _display_selector_state(self, state: SelectorState) -> SelectorState:
        value = state.value
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(value, NumericRange)
            value = self._canonical_range_to_display(
                value, self._x_selector_source()
            )
        elif state.kind is SelectorKind.AREA:
            assert isinstance(value, RectangleRange)
            value = self._area_canonical_to_display(value)
        elif state.kind is SelectorKind.CROSSHAIR:
            assert isinstance(value, CrosshairPoint)
            value = CrosshairPoint(
                self._canonical_x_scalar_to_display(value.x),
                value.y
                if self._is_histogram_plot()
                else self._canonical_scalar_to_display(value.y, self._y_ref_or_value()),
            )
        elif state.kind is SelectorKind.THRESHOLD:
            value = self._canonical_scalar_to_display(float(value), self._value_quantity())
        return replace(state, value=value)

    def _selector_state_or_none(
        self,
        kind: SelectorKind,
    ) -> SelectorState | None:
        try:
            return self._context.selector_state(kind)
        except KeyError:
            return None

__all__ = [
    "FitAuthority",
    "FitProjection",
    "FitScope",
    "FitSelection",
    "HistogramProjection",
    "ProjectionContext",
]
