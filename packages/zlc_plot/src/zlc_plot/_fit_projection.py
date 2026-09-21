"""Projection and fit semantics shared by sessions and live workers."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable

import math

import numpy as np

from zlc_data import AxisSpec, BlockId, DatasetRevisionRef, OwnedSnapshot, Selection
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID, SHOT_TIME_AXIS_ID, indexed_history_layout, restrict_snapshot, value_selection

from .data_contract import (
    DEFAULT_UNITS,
    Unit,
    UnitRegistry,
    resolve_unit,
    resolve_axis,
    snapshot_generation,
    snapshot_revision,
    schema_value_unit,
)

from ._pulse_time import pulse_time_scale
from .config import PlotLibraryDefaults
from .data_view import (
    AxisValue,
    CurveData,
    CurveSeries,
    FacetData,
    HistogramData,
    ImageData,
    QuantityArray,
    RollingHistory,
    histogram_edges,
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
from .semantics import projection_scope
from .selectors import (
    CrosshairPoint,
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
    Viewport,
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
    Reduction,
)
from .state import DisplayState
from ._validation import integer, readonly_copy


class FitScope(str, Enum):
    SELECTOR = "selector"
    VIEWPORT = "viewport"
    ALL = "all"


def _combine_moment_summaries(
    left_n: np.ndarray,
    left_mean: np.ndarray,
    left_m2: np.ndarray,
    left_single_sem_square: np.ndarray,
    right_n: np.ndarray,
    right_mean: np.ndarray,
    right_m2: np.ndarray,
    right_single_sem_square: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine independent ``(n, mean, M2)`` summaries without raw moments.

    ``single_sem_square`` is meaningful only when the summary contains one
    sample.  It preserves that sample's stated uncertainty for the one case
    where observed scatter cannot estimate an error; once two samples are
    present, their scatter is the estimator and the value becomes NaN.
    """

    count = left_n + right_n
    left_present = left_n > 0
    right_present = right_n > 0
    both = left_present & right_present
    mean = np.where(left_present, left_mean, right_mean)
    m2 = np.where(left_present, left_m2, right_m2)
    single_sem_square = np.where(
        left_present, left_single_sem_square, right_single_sem_square
    )
    if np.any(both):
        delta = right_mean[both] - left_mean[both]
        total = count[both].astype(np.float64)
        left_weight = left_n[both].astype(np.float64)
        right_weight = right_n[both].astype(np.float64)
        mean[both] = left_mean[both] + delta * right_weight / total
        m2[both] = (
            left_m2[both]
            + right_m2[both]
            + np.square(delta) * left_weight * right_weight / total
        )
    single_sem_square[count != 1] = np.nan
    return count, mean, m2, single_sem_square


def _window_moment_summaries(
    shot_n: np.ndarray,
    shot_mean: np.ndarray,
    shot_m2: np.ndarray,
    shot_single_sem_square: np.ndarray,
    span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy reference for the compiled span-block Chan scan."""

    total = shot_n.size
    width = min(span, total)
    block_count = (total + width - 1) // width
    padded = block_count * width
    shot_counts = np.zeros(padded, dtype=np.int64)
    shot_counts[:total] = shot_n
    shot_means = np.zeros(padded, dtype=np.float64)
    shot_means[:total] = shot_mean
    shot_m2s = np.zeros(padded, dtype=np.float64)
    shot_m2s[:total] = shot_m2
    shot_sem_squares = np.full(padded, np.nan, dtype=np.float64)
    shot_sem_squares[:total] = shot_single_sem_square
    shot_counts = shot_counts.reshape(block_count, width)
    shot_means = shot_means.reshape(block_count, width)
    shot_m2s = shot_m2s.reshape(block_count, width)
    shot_sem_squares = shot_sem_squares.reshape(block_count, width)

    suffix_n = shot_counts.copy()
    suffix_mean = shot_means.copy()
    suffix_m2 = shot_m2s.copy()
    suffix_single_sem_square = shot_sem_squares.copy()
    for offset in range(width - 2, -1, -1):
        (
            suffix_n[:, offset],
            suffix_mean[:, offset],
            suffix_m2[:, offset],
            suffix_single_sem_square[:, offset],
        ) = _combine_moment_summaries(
            shot_counts[:, offset],
            shot_means[:, offset],
            shot_m2s[:, offset],
            shot_sem_squares[:, offset],
            suffix_n[:, offset + 1],
            suffix_mean[:, offset + 1],
            suffix_m2[:, offset + 1],
            suffix_single_sem_square[:, offset + 1],
        )

    for offset in range(1, width):
        (
            shot_counts[:, offset],
            shot_means[:, offset],
            shot_m2s[:, offset],
            shot_sem_squares[:, offset],
        ) = _combine_moment_summaries(
            shot_counts[:, offset - 1],
            shot_means[:, offset - 1],
            shot_m2s[:, offset - 1],
            shot_sem_squares[:, offset - 1],
            shot_counts[:, offset],
            shot_means[:, offset],
            shot_m2s[:, offset],
            shot_sem_squares[:, offset],
        )

    if block_count > 1 and width > 1:
        (
            shot_counts[1:, :-1],
            shot_means[1:, :-1],
            shot_m2s[1:, :-1],
            shot_sem_squares[1:, :-1],
        ) = _combine_moment_summaries(
            suffix_n[:-1, 1:],
            suffix_mean[:-1, 1:],
            suffix_m2[:-1, 1:],
            suffix_single_sem_square[:-1, 1:],
            shot_counts[1:, :-1],
            shot_means[1:, :-1],
            shot_m2s[1:, :-1],
            shot_sem_squares[1:, :-1],
        )

    return (
        shot_counts.reshape(-1)[:total],
        shot_means.reshape(-1)[:total],
        shot_m2s.reshape(-1)[:total],
        shot_sem_squares.reshape(-1)[:total],
    )


def _trailing_trace(
    history: RollingHistory,
    column: int,
    span: int,
    *,
    uncertainty: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Mean and standard error over ``span`` shots inside the panel's window.

    Ungrouped, each shot contributes everything it pooled (its stored
    moments), so shots pooling different sample counts weigh in correctly.
    Grouped, each shot contributes its one per-key reduced value -- for a
    per-site trace that IS the shot's one sample.

    ``span`` counts SHOTS, not samples: "the mean of the last hundred
    shots" is a statement about the recent past of the run, whatever each
    of those shots happened to pool.
    """

    total = len(history)
    values = np.asarray(history.values, dtype=np.float64)[:, column]
    contributing = np.asarray(history.valid[:, column], dtype=bool)
    sems = (
        np.full(total, np.nan, dtype=np.float64)
        if not uncertainty or history.sem is None
        else np.asarray(history.sem, dtype=np.float64)[:, column]
    )
    if history.group_keys[column] == ():
        counts = np.asarray(history.counts, dtype=np.int64)[:, column]
        contributing = contributing & (counts > 0)
        shot_n = np.where(contributing, counts, 0)
    else:
        # One per-key value per shot: for a per-site trace that IS the
        # shot's one sample.  Its own SEM is still meaningful when it is the
        # only valid history sample in a trailing window.
        shot_n = contributing.astype(np.int64)

    shot_mean = np.where(contributing, values, 0.0)
    shot_m2 = np.zeros(total, dtype=np.float64)
    pooled = contributing & (shot_n > 1) & np.isfinite(sems)
    shot_m2[pooled] = (
        np.square(sems[pooled])
        * shot_n[pooled]
        * (shot_n[pooled] - 1)
    )
    shot_single_sem_square = np.full(total, np.nan, dtype=np.float64)
    single = contributing & (shot_n == 1) & np.isfinite(sems)
    shot_single_sem_square[single] = np.square(sems[single])

    # A length-span window intersects at most two consecutive span-sized
    # blocks.  The compiled owner performs their Chan/Welford prefix/suffix
    # scan in O(history); the NumPy spelling is its exact reference/fallback.
    from . import _raster_kernels as kernels

    summaries = kernels.trailing_moment_windows(
        shot_n, shot_mean, shot_m2, shot_single_sem_square, span
    )
    if summaries is None:
        summaries = _window_moment_summaries(
            shot_n, shot_mean, shot_m2, shot_single_sem_square, span
        )
    running_n, mean, running_m2, running_single_sem_square = summaries
    sem = None
    if uncertainty:
        variance = np.full(total, np.nan, dtype=np.float64)
        observed = running_n > 1
        variance[observed] = running_m2[observed] / (running_n[observed] - 1)
        stated = (running_n == 1) & np.isfinite(running_single_sem_square)
        variance[stated] = running_single_sem_square[stated]
        with np.errstate(invalid="ignore", divide="ignore"):
            sem = np.sqrt(variance / running_n)
    valid = (running_n > 0) & np.isfinite(mean)
    mean = np.where(valid, mean, np.nan)
    return mean, sem, valid


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


class _Crossing(Enum):
    """How a fit parameter's number crosses between solver and painted units."""

    POINT = "point"
    SPAN = "span"
    INVERSE = "inverse"


@dataclass(frozen=True, slots=True)
class _FitParameterConversion:
    """One fit parameter's unit crossing, the same object read both ways.

    A POINT (a centre, an offset) is a coordinate and converts like one:
    exactly, through the unit's own conversion, so a level such as dBm
    crosses as the power it names.  A SPAN (a width, an amplitude, a
    standard error) or an INVERSE (a frequency read against a time axis)
    has no position and crosses by the ratio of two linear scales; across a
    logarithmic unit it has no value at all -- a width in dBm is not any
    width in watts -- and asking is refused rather than answered with a
    number.  Units of ``None`` mean the number is already on screen.
    """

    parameter: str
    canonical_unit: Unit | None
    display_unit: Unit | None
    symbol: str
    crossing: _Crossing
    #: What the SOLVER's number is in.  Not ``canonical_unit.symbol``: an
    #: inverse parameter's conversion carries the AXIS's units and inverts
    #: when it crosses, so its own canonical unit is their inverse.  An
    #: empty string means the number has no unit.
    canonical_symbol: str = ""

    def to_display(self, value: float) -> float:
        return self._convert(value, self.canonical_unit, self.display_unit)

    def to_canonical(self, value: float) -> float:
        return self._convert(value, self.display_unit, self.canonical_unit)

    def _convert(self, value: float, source: Unit | None, target: Unit | None) -> float:
        if source is None or target is None or source == target:
            return value
        if self.crossing is _Crossing.POINT:
            return float(
                np.asarray(
                    source.convert_value_to((value,), target),
                    dtype=float,
                ).reshape(-1)[0]
            )
        source_scale, target_scale = source.coordinate_scale, target.coordinate_scale
        if source_scale is None or target_scale is None or source_scale[0] != target_scale[0]:
            raise ValueError(
                f"fit parameter {self.parameter!r} is a {self.crossing.value} and has "
                f"no value between {source.symbol!r} and {target.symbol!r}; only a "
                "position crosses a logarithmic unit"
            )
        ratio = source_scale[1] / target_scale[1]
        return value * (ratio if self.crossing is _Crossing.SPAN else 1.0 / ratio)


@dataclass(frozen=True, slots=True)
class FitAuthority:
    """What defines a fit's domain: the committed selector, else the viewport."""

    selector: SelectorState | None
    viewport: Viewport | None


@dataclass(frozen=True, slots=True, eq=False)
class FitSelection:
    data_revision: int
    scope: FitScope
    coordinates: tuple[np.ndarray, ...]
    observations: np.ndarray
    selected_indices: np.ndarray | None
    #: Per-observation standard error in the same canonical unit as the
    #: observations, aligned with them, or None when the projection carries
    #: no uncertainty.  Present sigma means the fit weights by it -- the
    #: uncertainty request is one switch for the band AND the weighting.
    observation_sigma: np.ndarray | None = None
    facet_index: int | None = None
    selector_kind: SelectorKind | None = None
    regular_image: RegularImageFitInput | None = None
    group_key: tuple[AxisValue, ...] = ()

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
        # A regular selection describes that exact validated input; it does
        # not own a second copy of the input's coordinate axes.
        object.__setattr__(
            self,
            "coordinates",
            (
                (regular_image.x_coordinates, regular_image.y_coordinates)
                if regular_image is not None
                else tuple(readonly_copy(value, dtype=float) for value in self.coordinates)
            ),
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
        # The sigma plane is replayed to late subscribers with the other
        # three; it is sealed like them, or it is the one plane through
        # which what the solver weighted by could be rewritten afterwards.
        sigma = self.observation_sigma
        if sigma is not None:
            sigma = readonly_copy(sigma, dtype=float)
            if sigma.shape != observations.shape:
                raise ValueError("observation_sigma must align with the observations")
        object.__setattr__(self, "observation_sigma", sigma)
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
    #: How many bins this projection HAS.  Not how many were asked for:
    #: integer-valued samples bin on integer boundaries, so a request for
    #: sixty bins over a span of twenty-nine counts produces twenty-nine.
    bin_count: int
    #: How many were asked for -- what this domain was cut FOR.  The two
    #: were the same field once, and the retention test below compared the
    #: produced count against the request: on a camera they never matched,
    #: so a domain that was written to be held was re-fitted on every
    #: revision for as long as both meanings shared one name.
    requested_bins: int
    #: The span the edges were cut FROM.  Also not the same as the span they
    #: cover: integer-aligned bins round the domain up to a whole number of
    #: them, so ``edges[0], edges[-1]`` is wider than what was asked for.
    #: Holding the edges' own span as the next revision's domain therefore
    #: widened it again, and again -- a live histogram's value axis grew
    #: from 30 counts to 1200 in ninety frames.
    domain: tuple[float, float]
    edges: np.ndarray

    def __post_init__(self) -> None:
        bin_count = integer(
            self.bin_count,
            "histogram projection bin_count",
            minimum=1,
        )
        object.__setattr__(
            self,
            "requested_bins",
            integer(
                self.requested_bins,
                "histogram projection requested_bins",
                minimum=1,
            ),
        )
        domain_low, domain_high = (float(value) for value in self.domain)
        if not (
            math.isfinite(domain_low)
            and math.isfinite(domain_high)
            and domain_low < domain_high
        ):
            raise ValueError("histogram projection domain must be increasing")
        object.__setattr__(self, "domain", (domain_low, domain_high))
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
    viewport: Viewport | None = None
    focused_facet_index: int | None = None

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
        inherit_view: "DataView | None" = None,
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
        #: What a fit parameter's number is on screen, resolved once per
        #: view: the answer is a function of the parameter's spec and the
        #: view's quantities, and a sixty-four cell grid asked it a
        #: thousand times a frame, every cell walking the unit relations
        #: again for the same twelve parameters of the same model.
        self._fit_conversion_memo: dict[
            tuple[Any, bool, UnitRelation | None], _FitParameterConversion
        ] = {}
        self._histogram_plot = isinstance(semantic_spec(spec), HistogramPlot)
        #: The previous revision's built DataView, handed across the fork so
        #: coordinate-domain work (an np.unique over a million-point axis)
        #: carries over when the coordinate plane did not change.  Consumed
        #: and released by the first _build_view.
        self._inherit_view = inherit_view
        self._scoped_cache: tuple[object, OwnedSnapshot, object, dict] | None = None
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

        frozen = FitProjection(
            data=data,
            revision=revision,
            spec=self._spec,
            context=context,
            unit_registry=self._unit_registry,
            defaults=self._defaults,
            histogram_projection=self._histogram_projection,
            inherit_view=self._view,
        )
        frozen._scoped_cache = self._scoped_cache
        return frozen

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
            self._scoped_cache,
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
                self._scoped_cache,
            ) = previous
            raise

    @property
    def display_state(self) -> DisplayState:
        return self._context.display_state

    @property
    def spec(self) -> Any:
        """What this projection's view and payload were built FROM.

        A payload only means anything beside the spec it was projected
        through, and a live frame is prepared long before it is committed.
        Saying so out loud is what lets the commit refuse a frame whose
        spec has been replaced underneath it.
        """

        return self._spec

    @property
    def data_revision(self) -> int:
        return self._revision

    @property
    def data_generation(self) -> str | None:
        """Which dataset this frame came out of, or None for a kind with no run.

        THE ONE ANSWER.  A pulse timeline is authored, not acquired: it has a
        revision but no run behind it, so "which dataset" has no answer and
        None is that answer.  This used to REFUSE instead, and every caller
        that knew the kind might have no generation wrote the refusal off as
        an absent attribute -- ``getattr(projection, "data_generation", None)``
        -- which is not what a raising property does.  So the Pulse Editor's
        preview died on the one line meant to tolerate it, and the operator
        was told the pulse could not be drawn.
        """

        data = self._data
        return snapshot_generation(data) if isinstance(data, OwnedSnapshot) else None

    @property
    def data(self) -> OwnedSnapshot | PulseTimelineData:
        return self._data

    @property
    def viewport(self) -> Viewport | None:
        return self._context.viewport

    @property
    def view(self) -> Any:
        return self._view

    @property
    def payload(self) -> Any:
        return self._payload

    @property
    def _viewport(self) -> Viewport | None:
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
                    (self._x_quantity(),)
                    if model.independent_arity == 1
                    else (
                        self._x_quantity(),
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
            self._install_view(None)
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
        scope = projection_scope(self._data.block.schema, self._spec)
        rolling = isinstance(self._semantic_spec(), RollingPlot)
        layout = (
            indexed_history_layout(self._data.block.schema)
            if isinstance(self._semantic_spec(), (CurvePlot, HistogramPlot, RollingPlot)) else None
        )
        window = int(self.display_state.values["window"]) if layout is not None or rolling else None
        count = layout.shot_count if layout is not None else self._data.block.schema.repeat_domain.size
        narrowed = window is not None and window < count
        if not scope and not narrowed:
            self._scoped_cache = None
            return self._data
        source = self._data
        schema = source.block.schema
        terms: dict[object, object] = {}
        identity: list[tuple[str, str, object]] = []
        for ref, value in scope:
            resolved = resolve_axis(schema, ref)
            terms[resolved.axis_id] = value
            identity.append(
                (ref.domain.value, ref.axis_id, value)
            )
        digest = f"window={window};" + ",".join(
            f"{domain}:{axis_id}={value!r}"
            for domain, axis_id, value in sorted(identity)
        )
        key = source.ref, digest
        if self._scoped_cache is not None and self._scoped_cache[0] == key:
            return self._scoped_cache[1]
        scope_identity = tuple((domain, axis_id, type(value), value) for domain, axis_id, value in sorted(identity))
        context = source.ref.stream_generation, scope_identity, schema.cell_domain, schema.value_schema
        memo = (dict(self._scoped_cache[3]) if scope and self._scoped_cache is not None
                and self._scoped_cache[2] == context else {})

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

        scoped = source
        if narrowed:
            scoped = restrict_snapshot(
                source,
                None if layout is None else Selection.index_range(PRIMARY_INDEX_AXIS_ID, count - window, count),
                repeat_rows=range(count - window, count) if layout is None else None,
                reference_for=lambda derived: DatasetRevisionRef(
                    BlockId(f"{source.ref.block_id.value}|window:{window}"),
                    source.ref.stream_generation, derived.fingerprint, source.ref.revision,
                ),
            )
        if scope:
            scoped = restrict_snapshot(
                scoped, value_selection(scoped.block.schema, terms), reference_for=reference_for,
                slice_memo=memo,
            )
        self._scoped_cache = (key, scoped, context, memo)
        return scoped

    def _install_view(self, view: "DataView | None") -> None:
        """The view every quantity is read from -- and the end of whatever
        was resolved against the one before it."""

        self._view = view
        self._fit_conversion_memo.clear()

    def _build_view(self) -> None:
        """Construct the unit-aware DataView without projecting a payload."""

        assert isinstance(self._data, OwnedSnapshot)
        from .data_view import DataView

        values = self.display_state.values
        overrides: dict[AxisRef, object] = {}
        x_ref = getattr(self._semantic_spec(), "x", None)
        y_ref = getattr(self._semantic_spec(), "y", None)
        if isinstance(self._spec, FacetGridPlot) and self._spec.facet is not None:
            facet_unit = values.get("facet_display_unit")
            if facet_unit is not None:
                overrides[self._spec.facet] = facet_unit
        if x_ref is not None and values.get("x_display_unit") is not None:
            overrides[x_ref] = values.get("x_display_unit")
        if y_ref is not None and values.get("y_display_unit") is not None:
            overrides[y_ref] = values.get("y_display_unit")
        value_unit = values.get("value_display_unit")
        self._install_view(DataView(
            self._scoped_data(),
            axis_display_units=overrides,
            value_display_unit=value_unit,
            unit_registry=self._unit_registry,
            inherit_domains_from=(
                self._view if self._view is not None else self._inherit_view
            ),
        ))
        self._inherit_view = None

    def _build_payload_from_view(self) -> None:
        """Reproject the plot payload without rebuilding unit-aware DataView state."""

        if not isinstance(self._data, OwnedSnapshot):
            self._payload = self._data
            return
        if self._view is None:
            raise RuntimeError("dataset payload projection requires a DataView")
        if not isinstance(self._spec, HistogramPlot):
            self._view._frequency_carry = None
        handler_for(self._spec).build_payload(self, self._view, self.display_state)

    def _rolling_payload(
        self,
        history: RollingHistory,
        *,
        trailing: int = 1,
        uncertainty: bool = False,
    ) -> CurveData:
        """Build one history series per optional rolling group.

        ``trailing`` is how many shots each drawn point averages: 1 is the
        shot itself, N is the mean of the last N.  ``uncertainty`` draws the
        band -- the standard error of those same N shots when averaging,
        each shot's own pooled standard error when not. The common scoped
        DataView already holds only this panel's window, so neither reduction
        nor trailing statistics can consume another panel's retained records.
        """

        total = len(history)
        if not total:
            raise ValueError("rolling history cannot be empty")
        keys = history.group_keys
        # Each drawn series is a column slice of the history planes -- no
        # per-shot objects, no per-shot key lookup.
        values_plane = np.asarray(history.values, dtype=float)
        valid_plane = np.asarray(history.valid, dtype=bool)
        # Runtime supplies both coordinates: relative source index and actual
        # run-relative time. A window selects records without rebasing either.
        along = self._spec.x
        if along is None or along == AxisRef.point(PRIMARY_INDEX_AXIS_ID.value):
            if history.source_indices is not None:
                source_coordinates = np.asarray(
                    history.source_indices, dtype=float
                )
            else:
                source_coordinates = np.arange(total, dtype=float) - (total - 1)
            x_unit = resolve_unit("1", DEFAULT_UNITS)
            x_label = "Shots from latest"
        elif along == AxisRef.point(SHOT_TIME_AXIS_ID.value) and history.source_times is not None:
            source_coordinates = np.asarray(history.source_times, dtype=float)
            x_unit = self._view.coordinate(along).canonical_unit
            x_label = self._view.coordinate(along).label
        else:
            raise ValueError(
                "a rolling plot places its shots along the shot index or, for a "
                f"history that stamps its shots, the shot-time axis; not {along!r}"
            )
        x_values = source_coordinates
        unit = self._view._value_display_unit
        canonical_unit = schema_value_unit(self._view._schema, self._view._unit_registry)
        value_label = self._view._schema.value_schema.name or "value"
        if trailing == 1:
            values_plane = np.where(valid_plane, values_plane, np.nan)
            display_plane = canonical_unit.convert_value_to(values_plane, unit)
            sem_plane = (
                np.where(valid_plane, np.asarray(history.sem, dtype=float), np.nan)
                if uncertainty and history.sem is not None else None
            )
            for plane in (values_plane, display_plane, sem_plane):
                if plane is not None:
                    plane.setflags(write=False)
        x_values.setflags(write=False)
        x = QuantityArray(
            x_values,
            (x_values if along is None else x_unit.convert_value_to(
                x_values, self._view.coordinate(along).display_unit
            )),
            x_unit,
            x_unit if along is None else self._view.coordinate(along).display_unit,
            x_label,
        )
        series: list[CurveSeries] = []
        for column, key in enumerate(keys):
            sem = None
            if trailing > 1:
                canonical_values, sem, valid = _trailing_trace(
                    history, column, trailing, uncertainty=uncertainty,
                )
                display_values = canonical_unit.convert_value_to(
                    canonical_values, unit
                )
                for plane in (canonical_values, display_values, valid, sem):
                    if plane is not None:
                        plane.setflags(write=False)
            else:
                valid = valid_plane[:, column]
                canonical_values = values_plane[:, column]
                display_values = display_plane[:, column]
                if uncertainty and sem_plane is not None:
                    # A history whose shots state no error has NO band --
                    # None, not a plane of NaN for every consumer to carry,
                    # mask and skip again.
                    sem = sem_plane[:, column]
            y = QuantityArray(
                canonical_values,
                display_values,
                canonical_unit,
                unit,
                value_label,
            )
            label = value_label if not key else ", ".join(
                item.label for item in key
            )
            series.append(
                CurveSeries(
                    x=x,
                    y=y,
                    valid=valid,
                    sem=sem,
                    group_key=key,
                    label=label,
                )
            )
        return CurveData(
            revision=history.revision,
            generation=history.generation,
            x_ref=None,
            group_by=(() if self._spec.group is None else (self._spec.group,)),
            series=tuple(series),
        )

    def _histogram_bins(
        self,
        view: Any,
        state: DisplayState,
        *,
        binned_values: np.ndarray | None = None,
        binned_valid: np.ndarray | None = None,
        frequency: tuple[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return stable display-unit edges for one histogram projection.

        THE DOMAIN COVERS WHAT IS BINNED, so the caller says what that is.
        It is not always this revision's samples: a window is the last N
        shots, and a reduced spec bins one statistic per group, whose spread
        is narrower than the raw pool's by construction -- taken from the raw
        values, a reduced histogram landed in two bins out of twelve.  Named
        for the history alone, this argument only ever answered half of that.
        A window that is binned from its frequency table hands the table
        instead: its extrema are its first and last occupied level, read
        without a pass over the pool.
        """

        count = int(state["bin_count"])
        canonical_unit = schema_value_unit(view._schema, view._unit_registry)
        display_unit = view._value_display_unit
        if frequency is not None:
            if binned_values is not None:
                raise ValueError("a frequency table describes the whole binned pool")
            integral = True
            offset, table = frequency
            occupied = np.flatnonzero(table)
            has_values = bool(occupied.size)
            if has_values:
                data_low = float(offset + int(occupied[0]))
                data_high = float(offset + int(occupied[-1]))
        elif binned_values is None:
            samples = view.samples
            canonical = np.asarray(samples.value.canonical)
            valid = np.asarray(samples.valid_mask, dtype=bool)
            integral = canonical.dtype.kind in "biu"
        else:
            if binned_valid is None:
                raise ValueError("binned validity is required with binned values")
            canonical = np.asarray(binned_values)
            valid = np.asarray(binned_valid, dtype=bool)
            integral = canonical.dtype.kind in "biu"
        if frequency is not None:
            pass
        elif integral:
            has_values = bool(canonical.size) and bool(np.any(valid))
            if has_values:
                limits = (
                    (True, False)
                    if canonical.dtype.kind == "b"
                    else (
                        np.iinfo(canonical.dtype).max,
                        np.iinfo(canonical.dtype).min,
                    )
                )
                data_low = float(
                    np.min(canonical, where=valid, initial=limits[0])
                )
                data_high = float(
                    np.max(canonical, where=valid, initial=limits[1])
                )
        else:
            # Masked extrema in one pass, never a full finite gather: the
            # copy of a two-million-value pool cost more per revision than
            # the whole binning it fed.  The same pass says whether every
            # finite sample is a whole number, which is the edge owner's
            # question and a fact about EVERY sample -- a bounded prefix
            # once answered it for the whole pool and binned the same
            # multiset differently by storage order.
            flat = np.asarray(canonical).reshape(-1)
            mask = (
                None
                if valid is None
                else np.asarray(valid, dtype=bool).reshape(-1)
            )
            # One pass for three numbers.  The reductions below read the pool
            # four times -- isfinite, any, min, max -- and materialise a bool
            # plane as large as it, which on a two-million-value histogram
            # cost more per revision than counting the bins did.
            from . import _raster_kernels as kernels

            extrema = kernels.masked_finite_extrema(flat, mask)
            if extrema is not None:
                finite_count, kernel_low, kernel_high, integral = extrema
                has_values = finite_count > 0
                if has_values:
                    data_low = kernel_low
                    data_high = kernel_high
            else:
                finite = np.isfinite(flat)
                if mask is not None:
                    finite &= mask
                has_values = bool(np.any(finite))
                if has_values:
                    data_low = float(np.min(flat, where=finite, initial=np.inf))
                    data_high = float(
                        np.max(flat, where=finite, initial=-np.inf)
                    )
                integral = bool(
                    np.all(flat == np.floor(flat), where=finite)
                )
        previous = self._histogram_projection
        # The VALUE axis's own mode.  It used to read the count axis's, so
        # an operator asking for a steady count scale silently also asked
        # for a steady value domain, and could not ask for either alone.
        mode = str(state["x_relim_mode"])
        # Retention is what NORMAL means: keep the limits already on
        # screen unless the data leaves them.  Written as "not tight" it
        # also caught FIXED, and retention widens on overflow -- the one
        # thing fixed exists to forbid.  So a pinned axis held only until
        # the first revision that outgrew it, and every limit written
        # afterwards was swallowed: the elif below could never run again
        # for that bin count.
        retain_domain = (
            previous is not None
            and previous.requested_bins == count
            and mode == "normal"
        )
        if retain_domain:
            assert previous is not None
            low, high = previous.domain
            if has_values:
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
        elif mode == "fixed":
            if not has_values:
                data_low, data_high = 0.0, 1.0

            def _written(key: str, fallback: float) -> float:
                # WRITTEN IN DISPLAY UNITS.  "Value minimum" and "Value
                # maximum" sit beside the drawn axis, and a half-supplied
                # pair is completed from axes.get_xlim(), which is display
                # space.  This domain is canonical -- the edges convert on
                # the way out -- so a value axis canonical in 's' and shown
                # in 'ms' read a written 10 as ten SECONDS and drew a
                # 10000 ms axis.
                value = state[key]
                if value is None:
                    return fallback
                return float(
                    display_unit.convert_value_to(
                        np.asarray(float(value), dtype=float),
                        canonical_unit,
                    )
                )

            low = _written("x_min", data_low)
            high = _written("x_max", data_high)
        else:
            if not has_values:
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

        edges = histogram_edges(low, high, count, integral=integral and has_values)
        selected = HistogramProjection(
            len(edges) - 1,
            count,
            (low, high),
            edges,
        )
        if previous is None or not (
            previous.bin_count == selected.bin_count
            and previous.requested_bins == selected.requested_bins
            and previous.domain == selected.domain
            and np.array_equal(previous.edges, selected.edges)
        ):
            previous = selected
            self._histogram_projection = previous
        assert previous is not None
        return np.asarray(
            canonical_unit.convert_value_to(
                previous.edges,
                display_unit,
            ),
            dtype=float,
        )

    def _focused_payload(self, facet_index: int | None = None) -> Any:
        if not isinstance(self._spec, FacetGridPlot):
            return self._payload
        cells = tuple(getattr(self._payload, "cells", ()))
        selected = self._focused_facet_index if facet_index is None else facet_index
        if selected is None or selected < 0 or selected >= len(cells):
            raise IndexError("facet index is outside the current grid")
        return cells[selected].payload

    def _rolling_sample_offsets(self) -> np.ndarray:
        """The rolling x that each sample of THIS revision sits at.

        Indexed samples use the same Dataset coordinate as the drawn series.
        An unindexed current event has only its present-shot offset, zero.
        """

        if self._view is None:
            raise TypeError("rolling offsets require zlc_data.OwnedSnapshot")
        if self._view.has_primary_index:
            ref = self._spec.x or AxisRef.point(str(PRIMARY_INDEX_AXIS_ID))
            return np.asarray(self._view.coordinate(ref).canonical, dtype=float)
        return np.zeros(self._view.samples.shape, dtype=float)

    def _x_sample_canonical(self) -> np.ndarray:
        """Where each sample sits on the x axis, in that axis's own space.

        ONE OWNER.  Rolling's x is a plot-owned shot-offset quantity, not a
        Dataset axis and therefore never receives a synthetic AxisRef.
        """

        if self._is_histogram_plot():
            return np.asarray(self._view.samples.value.canonical)
        if isinstance(self._spec, RollingPlot):
            return self._rolling_sample_offsets()
        source = self._x_selector_source()
        return np.asarray(
            self._coordinate(source).canonical
            if isinstance(source, AxisRef)
            else source.canonical
        )

    def _rolling_visible_mask(self) -> np.ndarray:
        """Which samples of this revision the rolling curve actually draws.

        All of them or none: they share one offset, and the window either
        covers it or does not.
        """

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
        return np.isin(self._rolling_sample_offsets(), shots)

    def _crosshair_sample_mask(
        self,
        state: SelectorState,
        valid: np.ndarray,
        point_transform: Callable[[np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        """Nearest valid sample per facet, using one shared coordinate grid."""
        if self._view is None or not isinstance(state.value, CrosshairPoint):
            raise TypeError("crosshair sample lookup requires zlc_data.OwnedSnapshot")
        displayed = self._display_selector_state(state)
        target = displayed.value
        samples = self._view.samples
        semantic = self._semantic_spec()
        image = isinstance(semantic, ImagePlot)
        if isinstance(semantic, HistogramPlot):
            x_values = np.asarray(samples.value.display)
            y_values = np.broadcast_to(target.y, samples.shape)
        else:
            x_values = (
                np.asarray(self._x_quantity().canonical_unit.convert_value_to(
                    self._x_sample_canonical(), self._x_quantity().display_unit
                ))
                if isinstance(self._spec, RollingPlot)
                else np.asarray(self._coordinate(self._x_ref()).display)
            )
            y_values = (
                np.asarray(self._coordinate(self._y_axis_ref()).display)
                if image else np.asarray(samples.value.display)
            )
        candidate = valid & np.isfinite(x_values) & np.isfinite(y_values)
        if isinstance(self._spec, RollingPlot):
            candidate &= self._rolling_visible_mask()

        def compact(values: np.ndarray) -> np.ndarray:
            # Broadcast coordinates share one axis/grid across all cells.
            return values[tuple(slice(0, 1) if stride == 0 else slice(None)
                                for stride in values.strides)]

        def transformed(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x, y = np.broadcast_arrays(x, y)
            points = np.column_stack((x.reshape(-1), y.reshape(-1)))
            converted = np.asarray(point_transform(points), dtype=float)
            if converted.shape != points.shape:
                raise ValueError("point transform returned the wrong shape")
            return converted[:, 0].reshape(x.shape), converted[:, 1].reshape(y.shape)

        separable = point_transform is None or bool(
            getattr(getattr(point_transform, "__self__", None), "is_separable", False)
        )
        target_x, target_y = target.x, target.y
        y_transformed = None
        if point_transform is not None:
            target_x, target_y = (float(v) for v in
                                  transformed(np.asarray(target.x), np.asarray(target.y)))
        if separable:
            x_distance = compact(x_values)
            if point_transform is not None:
                x_distance, _ = transformed(x_distance, np.asarray(target.y))
            distance = np.abs(x_distance - target_x)
            if image:
                y_distance = compact(y_values)
                if point_transform is not None:
                    _, y_distance = transformed(np.asarray(target.x), y_distance)
                distance = np.hypot(distance, y_distance - target_y)
            elif point_transform is not None and not getattr(point_transform.__self__, "is_affine", False):
                # Nonlinear y transforms may exclude otherwise finite values.
                _, y_transformed = transformed(np.asarray(target.x), compact(y_values))
                candidate &= np.isfinite(y_transformed)
        else:
            # Arbitrary callers may supply a coupled transform; preserve its
            # exact geometry rather than assume independent x/y mappings.
            x_transformed, y_transformed = transformed(compact(x_values), compact(y_values))
            candidate &= np.isfinite(y_transformed)
            distance = np.abs(x_transformed - target_x)
            if image:
                distance = np.hypot(distance, y_transformed - target_y)
        primary = np.where(candidate & np.isfinite(distance), distance, np.inf)
        facet = self._spec.facet if isinstance(self._spec, FacetGridPlot) else None
        contract = None if facet is None else resolve_axis(self._view._schema, facet)
        if contract is None:
            tied = primary == primary.min()
        else:
            # Reduce all non-carrier dimensions once, then group only the
            # carrier rows by the producer's existing axis codes.
            dimension = contract.dimension
            codes = contract.source_indices(self._view._schema)
            other = tuple(i for i in range(primary.ndim) if i != dimension)
            row_minimum = np.min(primary, axis=other) if other else primary
            minimum = np.full(contract.size, np.inf)
            np.minimum.at(minimum, codes, row_minimum)
            shape = [1] * primary.ndim
            shape[dimension] = len(codes)
            tied = primary == minimum[codes].reshape(shape)
        tied &= np.isfinite(primary)
        flat_indices = np.flatnonzero(tied.reshape(-1))
        result = np.zeros(samples.shape, dtype=bool)
        if flat_indices.size == 0:
            return result
        lane_count = 1 if contract is None else contract.size
        if contract is None:
            lanes = np.zeros(flat_indices.size, dtype=np.int64)
        else:
            stride = math.prod(samples.shape[contract.dimension + 1:])
            positions = (flat_indices // stride) % samples.shape[contract.dimension]
            lanes = codes[positions]
        if not image:
            positions = np.unravel_index(flat_indices, samples.shape)
            selected_y = y_values[positions]
            if point_transform is not None:
                if y_transformed is None:
                    _, selected_y = transformed(x_values[positions], selected_y)
                else:
                    selected_y = np.broadcast_to(y_transformed, samples.shape)[positions]
            delta_y = np.abs(selected_y - target_y)
            nearest_y = np.full(lane_count, np.inf)
            np.minimum.at(nearest_y, lanes, np.where(np.isfinite(delta_y), delta_y, np.inf))
            selected = np.isfinite(delta_y) & (delta_y == nearest_y[lanes])
            flat_indices, lanes = flat_indices[selected], lanes[selected]
        nearest = np.full(lane_count, result.size, dtype=np.int64)
        np.minimum.at(nearest, lanes, flat_indices)
        result.reshape(-1)[nearest[nearest < result.size]] = True
        return result

    def _axis_range_plane(
        self, ref: AxisRef, low: float, high: float
    ) -> np.ndarray:
        """Which samples fall inside [low, high] on one axis.

        ASK THE AXIS, NOT EVERY SAMPLE.  A coordinate varies along exactly
        one tensor dimension -- the invariant every dense projection rides
        when it moves ``resolved.dimension`` to the end -- so a range test
        has as many distinct answers as that axis is long, and the plane is
        those answers seen from every position.  Comparing the materialized
        plane twice and ANDing the results asked two million questions to
        learn two thousand: 4.96 ms per gesture against 0.004 for the axis
        and a broadcast, which allocates nothing at all.
        """

        assert self._view is not None
        resolved = self._view._resolve(ref)
        plane = np.asarray(resolved.coordinate.canonical)
        dimension = int(resolved.dimension)
        line = np.asarray(
            plane[
                tuple(
                    slice(None) if axis == dimension else 0
                    for axis in range(plane.ndim)
                )
            ],
            dtype=float,
        )
        keep = (line >= low) & (line <= high)
        return np.broadcast_to(
            keep.reshape(
                [-1 if axis == dimension else 1 for axis in range(plane.ndim)]
            ),
            plane.shape,
        )

    def _x_range_plane(self, low: float, high: float) -> np.ndarray:
        """The x range test, through the axis where an axis owns x.

        A rolling plot's x is a per-sample offset with no axis behind it --
        see _x_sample_canonical -- so it stays a per-sample comparison.
        """

        source = (
            None
            if isinstance(self._spec, RollingPlot)
            else self._x_selector_source()
        )
        if isinstance(source, AxisRef):
            return self._axis_range_plane(source, low, high)
        coordinate = self._x_sample_canonical()
        return (coordinate >= low) & (coordinate <= high)

    def _selector_mask(
        self,
        state: SelectorState,
        *,
        point_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        samples = self._view.samples
        if state.kind is SelectorKind.CROSSHAIR:
            return self._crosshair_sample_mask(state, samples.valid_mask, point_transform)
        mask = np.array(samples.valid_mask, copy=True, dtype=bool)
        if isinstance(self._spec, RollingPlot):
            mask &= self._rolling_visible_mask()
        value = np.asarray(samples.value.canonical)
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(state.value, NumericRange)
            mask &= self._x_range_plane(state.value.low, state.value.high)
        elif state.kind is SelectorKind.AREA:
            assert isinstance(state.value, RectangleRange)
            mask &= self._x_range_plane(state.value.x.low, state.value.x.high)
            semantic = self._semantic_spec()
            if isinstance(semantic, ImagePlot):
                mask &= self._axis_range_plane(
                    self._y_axis_ref(), state.value.y.low, state.value.y.high
                )
            elif not isinstance(semantic, HistogramPlot):
                # A one-dimensional curve's vertical display coordinate is
                # the observation itself.  AREA therefore filters both the
                # independent coordinate and the canonical observation; this
                # The raw-sample mask is materialised only by selector_data().
                mask &= (value >= state.value.y.low) & (value <= state.value.y.high)
        elif state.kind is SelectorKind.THRESHOLD:
            mask &= value >= float(state.value)
        # VALIDITY ALREADY ANSWERED FINITENESS.  ``mask`` starts as the
        # snapshot's validity plane, which DataView folds ``isfinite`` into
        # for float samples and which integer samples satisfy by
        # construction, and every branch above only ever removes samples.
        # Asking again cost a 2-megapixel isfinite pass and an AND -- 3.10
        # ms -- and could not, by construction, clear a single bit.
        return mask

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
            )

        if selector_kind is not None:
            selected = states.get(selector_kind)
            if selected is None:
                raise KeyError(selector_kind)
            if not usable(selected):
                raise ValueError(
                    "fit selector must contain a numeric selection"
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
        return FitAuthority(selector, viewport)

    def _fit_targets(self, *, all_facets: bool = True) -> tuple[tuple[int | None, int | None, tuple[AxisValue, ...]], ...]:
        if isinstance(self._spec, FacetGridPlot):
            cells = tuple((index, cell.payload) for index, cell in enumerate(self._payload.cells)
                          if all_facets or index == (self._focused_facet_index or 0))
        else:
            cells = ((None, self._payload),)
        return tuple(
            (facet_index, group_index, key)
            for facet_index, payload in cells
            for group_index, key in (
                enumerate(payload.group_keys) if isinstance(payload, HistogramData) else ((None, ()),)
            )
        )

    def _has_grouped_histogram(self) -> bool:
        semantic = self._semantic_spec()
        return isinstance(semantic, HistogramPlot) and semantic.group is not None

    def _fit_sample_axes(self, targets: tuple) -> tuple[tuple[str, AxisSpec], ...]:
        """The real retained coordinates, in the same order as flattened fits."""
        axes = []
        if isinstance(self._spec, FacetGridPlot) and self._spec.facet is not None:
            indices = tuple(dict.fromkeys(target[0] for target in targets))
            cells = self._payload.cells
            axes.append((self._spec.facet, tuple(cells[index].facet_value_canonical for index in indices)))
        group = getattr(self._semantic_spec(), "group", None)
        if self._has_grouped_histogram():
            axes.append((group, tuple(dict.fromkeys(target[2][0].canonical for target in targets))))
        projected = []
        for ref, values in axes:
            resolved = self._view._resolve(ref).contract
            original = resolved.domain.axis(resolved.axis_id)
            coordinate = self._coordinate(ref)
            unit = coordinate.canonical_unit.symbol
            labels = None
            if original.coordinate_labels is not None:
                source_values = self._view._domain(ref).values
                by_value = {value.canonical: original.coordinate_labels[value.index] for value in source_values}
                labels = tuple(by_value[value] for value in values)
            projected.append((ref.domain.value, replace(
                original, size=len(values), coordinates=values, unit=None if unit == "1" else unit,
                coordinate_labels=labels, index_origin=0, coordinate_of=None,
            )))
        return tuple(projected)

    def fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
        facet_index: int | None = None,
        group_index: int | None = None,
    ) -> FitSelection:
        """The single-cell call uses the same prepared selection as a batch."""

        return self._prepare_fit_selection(model, selector_kind)(facet_index, group_index)

    def _prepare_fit_selection(
        self,
        model: FitModelSpec,
        selector_kind: SelectorKind | None,
    ) -> Callable[[int | None], FitSelection]:
        """Resolve one request's model, units and domain before its cell loop.

        Only the computed cell changes within a batch. The same canonical
        region is shared by every cell regardless of which cell is focused;
        the closure is local to this request and is not retained.
        """

        if self._view is None:
            raise TypeError("fit is available only for zlc_data.OwnedSnapshot plots")
        if not isinstance(model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        self._require_fit_model_compatible(model)
        authority = self._fit_selection_authority(selector_kind)
        solver_units = tuple(
            self._fit_relation_quantity(relation).canonical_unit
            for relation in model.coordinate_relations
        )

        def select(facet_index: int | None = None, group_index: int | None = None) -> FitSelection:
            if facet_index is None:
                facet_index = self._focused_facet_index
            payload = self._focused_payload(facet_index)
            if isinstance(payload, CurveData):
                return self._curve_fit_selection(
                    model, authority=authority, solver_units=solver_units,
                    payload=payload, facet_index=facet_index,
                )
            if isinstance(payload, ImageData) and (
                model.capabilities & _REGULAR_IMAGE_CAPABILITIES
            ):
                return self._regular_image_fit_selection(
                    model, authority=authority, solver_units=solver_units,
                    payload=payload, facet_index=facet_index,
                )
            if self._is_histogram_plot():
                if not isinstance(payload, HistogramData):
                    raise RuntimeError("histogram projection did not produce histogram data")
                return self._histogram_fit_selection(
                    model, authority=authority, solver_units=solver_units,
                    payload=payload, facet_index=facet_index, group_index=group_index,
                )
            if not isinstance(payload, ImageData):
                raise TypeError("unsupported fit projection payload")
            return self._image_fit_selection(
                model, authority=authority, solver_units=solver_units,
                payload=payload, facet_index=facet_index,
            )

        return select

    def _curve_fit_selection(
        self,
        model: FitModelSpec,
        *,
        authority: FitAuthority,
        solver_units: tuple[Unit, ...],
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
        elif authority.viewport is not None and authority.viewport[0] is not None:
            x_range = authority.viewport[0]
            valid &= (x_canonical >= x_range.low) & (
                x_canonical <= x_range.high
            )
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL

        indices = np.flatnonzero(valid)
        coordinates = self._fit_coordinate_values_to_solver(
            x_canonical[valid],
            source.x,
            solver_units[0],
        )
        sem = getattr(source, "sem", None)
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(coordinates,),
            observations=y_canonical[valid],
            selected_indices=indices,
            observation_sigma=(
                None
                if sem is None
                else np.asarray(sem, dtype=np.float64).reshape(-1)[valid]
            ),
            facet_index=facet_index,
            selector_kind=None if active is None else active.kind,
        )

    def _histogram_fit_selection(
        self,
        model: FitModelSpec,
        *,
        authority: FitAuthority,
        solver_units: tuple[Unit, ...],
        payload: HistogramData,
        facet_index: int | None = None,
        group_index: int | None = None,
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
        if group_index is None:
            if len(payload.group_keys) != 1:
                raise ValueError("a grouped histogram fit must select each distribution")
            group_index = 0
        counts = np.asarray(payload.counts[group_index], dtype=float)
        if not np.any(counts > 0):
            raise ValueError("histogram distribution has no samples")
        valid = np.isfinite(canonical) & np.isfinite(counts)

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
        elif authority.viewport is not None and authority.viewport[0] is not None:
            x_range = authority.viewport[0]
            valid &= (canonical >= x_range.low) & (canonical <= x_range.high)
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL

        indices = np.flatnonzero(valid)
        model_centers = self._fit_coordinate_values_to_solver(
            canonical[valid],
            payload.centers,
            solver_units[0],
        )
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(model_centers,),
            observations=counts[valid],
            selected_indices=indices,
            facet_index=facet_index,
            group_key=payload.group_keys[group_index],
            selector_kind=None if active is None else active.kind,
        )

    def _regular_image_fit_selection(
        self,
        model: FitModelSpec,
        *,
        authority: FitAuthority,
        solver_units: tuple[Unit, ...],
        payload: ImageData,
        facet_index: int | None = None,
    ) -> FitSelection:
        """Describe a painted regular image without expanding coordinate grids."""

        x_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.x.canonical, dtype=float),
            payload.x,
            solver_units[0],
        ).reshape(-1)
        y_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.y.canonical, dtype=float),
            payload.y,
            solver_units[1],
        ).reshape(-1)
        valid, observations, scope, active = self._image_fit_domain(
            payload,
            authority,
        )
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
        )

    def _image_fit_selection(
        self,
        model: FitModelSpec,
        *,
        authority: FitAuthority,
        solver_units: tuple[Unit, ...],
        payload: ImageData,
        facet_index: int | None = None,
    ) -> FitSelection:
        """Fit the painted image projection without returning to raw samples."""

        if model.independent_arity != 2:
            raise ValueError("image fit models require exactly two independent axes")
        x_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.x.canonical, dtype=float),
            payload.x,
            solver_units[0],
        ).reshape(-1)
        y_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.y.canonical, dtype=float),
            payload.y,
            solver_units[1],
        ).reshape(-1)
        valid, observations, scope, active = self._image_fit_domain(
            payload,
            authority,
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
        )

    def _image_fit_domain(
        self,
        payload: ImageData,
        authority: FitAuthority,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray,
        FitScope,
        SelectorState | None,
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
            x_range, y_range = authority.viewport
            columns = (np.ones(x.shape, dtype=np.bool_) if x_range is None
                       else (x >= x_range.low) & (x <= x_range.high))
            rows = (np.ones(y.shape, dtype=np.bool_) if y_range is None
                    else (y >= y_range.low) & (y <= y_range.high))
            box = rows[:, None] & columns[None, :]
            valid = box if valid is None else valid & box
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL
        return valid, observations, scope, active

    def _viewport_in_canonical(self) -> Viewport:
        assert self._viewport is not None
        x_range, y_range = self._viewport
        return (
            None if x_range is None else self._display_range_to_canonical(
                x_range, self._x_selector_source()
            ),
            None if y_range is None else (
                y_range if self._is_histogram_plot()
                else self._display_range_to_canonical(y_range, self._y_ref_or_value())
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
                else self._x_quantity()
            )
        if relation is UnitRelation.AXIS_1:
            return self._coordinate(self._y_axis_ref())
        return None

    def _fit_coordinate_values_to_solver(
        self,
        values: np.ndarray,
        source_quantity: Any,
        target_unit: Unit,
    ) -> np.ndarray:
        """Convert a painted coordinate's canonical values into model units."""

        source_unit = source_quantity.canonical_unit
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
                self._fit_relation_quantity(result.model.coordinate_relations[0]).canonical_unit,
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
            self._fit_relation_quantity(result.model.coordinate_relations[0]).canonical_unit,
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
        evidence = ""
        if math.isfinite(result.evidence):
            # The two-population question, answered where the parameters
            # are read: the BIC gain of two over one, and what it decided.
            verdict = "one population" if result.reduced else "two populations"
            evidence = f"ΔBIC = {result.evidence:.1f}: {verdict}"
        return FitOverlay(
            polylines=polylines,
            ellipse_glyph=self._fit_overlay_ellipse(result, parameter_display),
            success=result.success,
            formula=result.model.formula or "",
            parameter_display=parameter_display,
            diagnostic=result.message,
            facet_index=selection.facet_index,
            group_key=selection.group_key,
            headline_parameter=headline_parameter,
            evidence=evidence,
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
            fixed = spec.name in result.fixed_parameter_names
            if result.covariance_valid and not fixed:
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
        """The unit the SOLVER's number is in, read off the one crossing.

        Which unit a fit parameter is in -- product, unit-free, histogram
        count, rolling ordinal, inverse axis, plain relation -- is one
        decision, and the crossing already makes it for the display side.
        Written out a second time here, the two answers were free to
        disagree, and only one of them was memoised.
        """

        return self._fit_parameter_conversion(spec).canonical_symbol

    def _fit_product_unit(self, spec: Any, *, display: bool) -> Unit:
        """The same VALUE and AXIS_0 vocabulary, multiplied as authored."""

        symbols = []
        for relation in (UnitRelation.VALUE, UnitRelation.AXIS_0):
            factor = replace(spec, unit_relation=relation, solver_unit_relation=relation)
            symbol = (
                self._fit_parameter_conversion(factor).symbol
                if display else self._canonical_fit_parameter_unit(factor)
            )
            # A unitless factor multiplies nothing: count*1 is count.
            if symbol:
                symbols.append(symbol)
        return (self._unit_registry or DEFAULT_UNITS).resolve("*".join(symbols) or "1")

    def _fit_parameter_conversion(
        self,
        spec: Any,
        *,
        difference: bool = False,
        display_relation: UnitRelation | None = None,
    ) -> _FitParameterConversion:
        """Where one fit parameter's number lives on screen, resolved once
        per view (see ``_fit_conversion_memo``)."""

        key = (spec, difference, display_relation)
        conversion = self._fit_conversion_memo.get(key)
        if conversion is None:
            conversion = self._resolve_fit_parameter_conversion(
                spec, difference=difference, display_relation=display_relation
            )
            self._fit_conversion_memo[key] = conversion
        return conversion

    def _resolve_fit_parameter_conversion(
        self,
        spec: Any,
        *,
        difference: bool,
        display_relation: UnitRelation | None,
    ) -> _FitParameterConversion:
        """Resolve where one fit parameter's number lives on screen.

        ``difference`` reads a value as a span even for a point parameter --
        a centre's standard error is a distance, not a place.
        ``display_relation`` reads it against another axis, for a glyph
        drawn where the parameter is not painted.
        """

        relation = spec.unit_relation if display_relation is None else display_relation
        solver_relation = spec.solver_unit_relation
        name = spec.name
        if relation is UnitRelation.VALUE_TIMES_AXIS_0:
            if solver_relation is not relation:
                raise ValueError("product parameters cannot cross unit relations")
            canonical_unit = self._fit_product_unit(spec, display=False)
            display_unit = self._fit_product_unit(spec, display=True)
            if not canonical_unit.compatible_with(display_unit):
                raise ValueError("fit product parameter units require compatible linear scales")
            return _FitParameterConversion(
                name, canonical_unit, display_unit, display_unit.symbol,
                _Crossing.SPAN, canonical_unit.symbol,
            )
        if relation in {UnitRelation.DIMENSIONLESS, UnitRelation.RADIAN}:
            if solver_relation is not relation:
                raise ValueError("unit-free fit parameters cannot cross unit relations")
            symbol = "rad" if relation is UnitRelation.RADIAN else ""
            return _FitParameterConversion(
                name, None, None, symbol, _Crossing.POINT, symbol,
            )
        if relation is UnitRelation.VALUE and self._is_histogram_plot():
            if solver_relation is not UnitRelation.VALUE:
                raise ValueError("histogram count parameters require value solver units")
            return _FitParameterConversion(
                name, None, None, "count", _Crossing.POINT, "count",
            )
        if (isinstance(self._spec, RollingPlot)
                and self._spec.x != AxisRef.point(SHOT_TIME_AXIS_ID.value)) and relation in {
            UnitRelation.AXIS_0,
            UnitRelation.INVERSE_AXIS_0,
        }:
            if solver_relation is not relation:
                raise ValueError("rolling fit parameters cannot cross unit relations")
            # The default rolling coordinate is the unit-free shot ordinal.
            symbol = "1/point" if relation is UnitRelation.INVERSE_AXIS_0 else "point"
            return _FitParameterConversion(
                name, None, None, symbol, _Crossing.POINT, symbol,
            )

        if relation is UnitRelation.INVERSE_AXIS_0:
            if solver_relation is not UnitRelation.INVERSE_AXIS_0:
                raise ValueError("inverse-axis parameters cannot cross unit relations")
            quantity = self._fit_relation_quantity(UnitRelation.AXIS_0)
            if quantity is None:
                return _FitParameterConversion(
                    name, None, None, "", _Crossing.INVERSE, "",
                )
            canonical_unit = quantity.canonical_unit
            display_unit = quantity.display_unit
            registry = self._unit_registry or DEFAULT_UNITS

            def inverted(unit: Unit) -> str:
                found = registry.inverse_for(unit)
                if found is not None:
                    return found.symbol
                return "" if unit.symbol == "1" else f"1/{unit.symbol}"

            return _FitParameterConversion(
                name, canonical_unit, display_unit, inverted(display_unit),
                _Crossing.INVERSE, inverted(canonical_unit),
            )

        source_quantity = self._fit_relation_quantity(solver_relation)
        target_quantity = self._fit_relation_quantity(relation)

        def plain(unit: Unit | None) -> str:
            if unit is None:
                return ""
            return "" if unit.symbol == "1" else unit.symbol

        if source_quantity is None or target_quantity is None:
            # The SOLVER's unit is still knowable when only the display side
            # is missing: a parameter the plot cannot paint is still in the
            # unit it was solved in.
            return _FitParameterConversion(
                name, None, None, "", _Crossing.POINT,
                plain(None if source_quantity is None else source_quantity.canonical_unit),
            )
        canonical_unit = source_quantity.canonical_unit
        display_unit = target_quantity.display_unit
        if not canonical_unit.compatible_with(display_unit):
            raise ValueError("fit parameter solver and display units are incompatible")
        return _FitParameterConversion(
            name,
            canonical_unit,
            display_unit,
            plain(display_unit),
            _Crossing.POINT if spec.affine_point and not difference else _Crossing.SPAN,
            plain(canonical_unit),
        )

    def _display_fit_parameter_value(
        self,
        spec: Any,
        value: float,
        *,
        difference: bool = False,
        display_relation: UnitRelation | None = None,
    ) -> tuple[float, str]:
        """A solver value in the painted unit, with that unit's symbol."""

        conversion = self._fit_parameter_conversion(
            spec,
            difference=difference,
            display_relation=display_relation,
        )
        return conversion.to_display(value), conversion.symbol

    def _canonical_fit_parameter_value(self, spec: Any, displayed: float) -> float:
        """The solver's number for a value typed in the painted unit.

        The exact inverse of ``_display_fit_parameter_value``: the same
        crossing read the other way, so what is typed reads back as typed.
        """

        return self._fit_parameter_conversion(spec).to_canonical(displayed)

    def fit_expression_target(
        self,
        model: FitModelSpec,
        expression: str,
    ) -> dict[str, object]:
        """Parse one compact display-unit expression into a canonical target.

        The operator writes the SYMBOLS the formula prints -- A, tau, x_0 --
        not the internal parameter names.  Those two vocabularies used to be
        different, and only one of them was ever on screen: the model drew
        f(t)=A e^{-(t-t_0)/tau}+B above a box that would only accept
        "amplitude" and "decay_time".

        The canonical target this returns still keys on the internal name,
        which is the identity the solver, the stored fit and every report
        use.  Only what is typed and read back changes.
        """

        if not isinstance(expression, str) or "\n" in expression or "\r" in expression:
            raise ValueError("fit expression must be one line of text")
        fixed: dict[str, float] = {}
        initial: dict[str, float] = {}
        assignments = tuple(map(str.strip, expression.split(",")))
        if expression.strip() and not all(assignments):
            raise ValueError("use comma-separated name=value assignments")
        for assignment in filter(None, assignments):
            if assignment.count("=") != 1:
                raise ValueError("use name=value or name=guess(value)")
            symbol, raw = (part.strip() for part in assignment.split("="))
            parameter = model.parameter_for_symbol(symbol)
            if parameter is None:
                # Say what this model DOES take.  A formula full of symbols
                # over a box that answers "unknown parameter" and stops is
                # the same silence that made the two vocabularies possible.
                raise ValueError(
                    f"{symbol!r} is not a parameter of this model; it takes "
                    + ", ".join(model.symbols)
                )
            name = parameter.name
            if name in fixed or name in initial:
                raise ValueError(f"repeated fit parameter {symbol!r}")
            guessed = raw.startswith("guess(") and raw.endswith(")")
            try:
                displayed = float(raw[6:-1] if guessed else raw)
            except ValueError as error:
                raise ValueError("use name=value or name=guess(value)") from error
            converted = self._canonical_fit_parameter_value(parameter, displayed)
            lower, upper = parameter.bounds
            if not math.isfinite(converted) or not lower <= converted <= upper:
                raise ValueError(f"fit parameter {symbol!r} is outside its domain")
            (initial if guessed else fixed)[name] = converted
        return {
            **{"model": model.model_id},
            **({"fixed": fixed} if fixed else {}),
            **({"initial": initial} if initial else {}),
        }

    def fit_expression_text(
        self,
        model: FitModelSpec,
        target: Any,
    ) -> str:
        """Format a canonical fixed/initial target in current painted units.

        In the SYMBOLS the formula prints, because this text goes straight
        back into the box the operator types in: what it writes out has to
        be something it would accept.
        """

        values = dict(target or {})
        fixed, initial = dict(values.get("fixed") or {}), dict(values.get("initial") or {})
        terms = []
        for parameter in model.parameters:
            source = fixed if parameter.name in fixed else initial
            if parameter.name not in source:
                continue
            value = self._display_fit_parameter_value(
                parameter, float(source[parameter.name])
            )[0]
            literal = "0" if value == 0.0 else repr(value)
            symbol = str(parameter.symbol)
            terms.append(
                f"{symbol}={literal}"
                if source is fixed
                else f"{symbol}=guess({literal})"
            )
        return ", ".join(terms)

    def _semantic_spec(self) -> Any:
        return semantic_spec(self._spec)

    def _is_histogram_plot(self) -> bool:
        return self._histogram_plot

    def _x_ref(self) -> AxisRef:
        semantic = self._semantic_spec()
        ref = getattr(semantic, "x", None)
        if not isinstance(ref, AxisRef):
            raise TypeError("this plot has no coordinate x axis")
        return ref

    def _rolling_x_quantity(self) -> Any:
        payload = self._focused_payload()
        series = tuple(getattr(payload, "series", ()))
        if not series:
            raise TypeError("rolling x coordinate requires a visible series")
        return series[0].x

    def _x_quantity(self) -> Any:
        if isinstance(self._semantic_spec(), RollingPlot):
            return self._rolling_x_quantity()
        return self._coordinate(self._x_ref())

    def _x_selector_source(self) -> AxisRef | Any:
        if self._is_histogram_plot():
            return self._value_quantity()
        if isinstance(self._semantic_spec(), RollingPlot):
            return self._rolling_x_quantity()
        return self._x_ref()

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
        # Units and labels belong to the accepted plotted quantity; a
        # control/fit-description read must not materialize raw history.
        payload = self._payload
        if isinstance(payload, FacetData) and payload.cells:
            payload = payload.cells[0].payload
        if isinstance(payload, CurveData) and payload.series:
            return payload.series[0].y
        if isinstance(payload, ImageData):
            return payload.z
        if isinstance(payload, HistogramData):
            return payload.edges
        return self._view.samples.value

    def _y_ref_or_value(self) -> AxisRef | Any:
        semantic = self._semantic_spec()
        ref = getattr(semantic, "y", None)
        return ref if isinstance(ref, AxisRef) else self._value_quantity()

    def _display_scalar_to_canonical(
        self, value: float, source: AxisRef | Any
    ) -> float:
        """One painted coordinate back in the DATASET's unit.

        The exact inverse of ``_canonical_scalar_to_display``.  It used to
        convert to the unit system's BASE unit (microseconds to seconds)
        instead of the dataset's own unit -- the two agree only for a
        dataset written in base units, so every selector on metres and volts
        round-tripped while a column in microseconds had every drawn box
        stored a million times too small.
        """

        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        display = quantity.display_unit
        canonical = quantity.canonical_unit
        return float(np.asarray(display.convert_value_to([value], canonical)).reshape(-1)[0])

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
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        return NumericRange(
            self._display_scalar_to_canonical(value.low, quantity),
            self._display_scalar_to_canonical(value.high, quantity),
        )

    def _canonical_range_to_display(
        self, value: NumericRange, source: AxisRef | Any
    ) -> NumericRange:
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
        source = self._x_selector_source()
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        return self._canonical_scalar_to_display(value, quantity)

    def _coordinate_values_to_display(
        self, values: np.ndarray, ref: AxisRef
    ) -> np.ndarray:
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
