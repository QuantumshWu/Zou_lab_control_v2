"""Projection and fit semantics shared by sessions and live workers."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable

import math

import numpy as np

from zlc_data import BlockId, DatasetRevisionRef, OwnedSnapshot
from zlc_data.snapshot_projection import restrict_snapshot, value_selection

from .data_contract import (
    DEFAULT_UNITS,
    UnitRegistry,
    resolve_unit,
    resolve_axis,
    snapshot_generation,
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
    RollingHistory,
    aligned_histogram_edges,
    _finite_probe,
    finite_probe,
    _sem_reference,
)
from .fit import (
    FitModelSpec,
    FitParameterDisplay,
    FitResult,
    FitTarget,
    PHOTOELECTRON_AXIS,
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
    Reduction,
)
from .state import DisplayState
from ._validation import integer, readonly_copy


class FitScope(str, Enum):
    SELECTOR = "selector"
    VIEWPORT = "viewport"
    ALL = "all"


def _window_totals(totals: np.ndarray, span: int) -> np.ndarray:
    """Running totals turned into totals over the last ``span`` entries.

    A prefix sum answers "everything up to here"; subtracting the prefix
    that has left the window answers "the last span".  Before the window
    has filled, the subtracted prefix is empty, so the early points are the
    running totals -- the trace begins as everything it has and settles
    into the window without a discontinuity.
    """

    if span >= totals.size:
        return totals
    shifted = np.zeros_like(totals)
    shifted[span:] = totals[:-span]
    return totals - shifted


def _trailing_trace(
    history: RollingHistory,
    column: int,
    start: int,
    span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error over the last ``span`` shots, then sliced.

    Ungrouped, each shot contributes everything it pooled (its stored
    moments), so shots pooling different sample counts weigh in correctly.
    Grouped, each shot contributes its one per-key reduced value -- for a
    per-site trace that IS the shot's one sample.

    ``span`` counts SHOTS, not samples: "the mean of the last hundred
    shots" is a statement about the recent past of the run, whatever each
    of those shots happened to pool.
    """

    total = len(history)
    values = np.asarray(history.values, dtype=float)[:, column]
    contributing = np.asarray(history.valid[:, column], dtype=bool)
    if history.group_keys[column] == ():
        counts = np.asarray(history.counts, dtype=float)[:, column]
        contributing = contributing & (counts > 0.0)
        count = np.where(contributing, counts, 0.0)
        sems = (
            np.full(total, np.nan)
            if history.sem is None
            else np.asarray(history.sem, dtype=float)[:, column]
        )
    else:
        # One per-key value per shot: for a per-site trace that IS the
        # shot's one sample, so it has no spread of its own.
        count = contributing.astype(float)
        sems = np.full(total, np.nan)

    # Square about the data, not about zero -- the same reason every bucket
    # reduction does.  A shot counter's values are small; a fitted optical
    # frequency's are not, and E[x^2] - mean^2 about zero would report a
    # spread made entirely of rounding.
    reference = _sem_reference(values[contributing])
    centred = np.where(contributing, values - reference, 0.0)
    mean_square = centred * centred
    # sem = s / sqrt(count), and E[x^2] about the shot's own mean is
    # mean^2 + s^2 (count - 1) / count.  Shifting the origin does not
    # change a spread, so the same term rides the centred mean.
    stated = contributing & (count > 1.0) & np.isfinite(sems)
    mean_square = mean_square + np.where(
        stated, np.where(stated, sems, 0.0) ** 2 * (count - 1.0), 0.0
    )
    n = np.where(contributing, count, 0.0)
    sums = n * centred
    squares = n * np.where(contributing, mean_square, 0.0)
    running_n = _window_totals(np.cumsum(n), span)
    running_sum = _window_totals(np.cumsum(sums), span)
    running_squares = _window_totals(np.cumsum(squares), span)
    with np.errstate(invalid="ignore", divide="ignore"):
        centred_mean = running_sum / running_n
        spread = np.clip(
            running_squares / running_n - np.square(centred_mean), 0.0, None
        )
        sem = np.sqrt(spread / (running_n - 1.0))
        mean = centred_mean + reference
    sem[running_n < 2.0] = np.nan
    valid = (running_n > 0.0) & np.isfinite(mean)
    mean = np.where(valid, mean, np.nan)
    return mean[start:], sem[start:], valid[start:]

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
    #: Per-observation standard error in the same canonical unit as the
    #: observations, aligned with them, or None when the projection carries
    #: no uncertainty.  Present sigma means the fit weights by it -- the
    #: uncertainty request is one switch for the band AND the weighting.
    observation_sigma: np.ndarray | None = None
    facet_index: int | None = None
    selector_kind: SelectorKind | None = None
    regular_image: RegularImageFitInput | None = None
    _authority: FitAuthority | None = None

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
    viewport: RectangleRange | None = None
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
        #: The previous revision's built DataView, handed across the fork so
        #: coordinate-domain work (an np.unique over a million-point axis)
        #: carries over when the coordinate plane did not change.  Consumed
        #: and released by the first _build_view.
        self._inherit_view = inherit_view
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
            inherit_view=self._view,
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
    def data_generation(self) -> str:
        if not isinstance(self._data, OwnedSnapshot):
            raise RuntimeError("data generation requires an OwnedSnapshot")
        return snapshot_generation(self._data)

    @property
    def data(self) -> OwnedSnapshot | PulseTimelineData:
        return self._data

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

    def _fit_model_needs_photoelectrons(self, model: FitModelSpec) -> str | None:
        """Why a photon-counting model cannot fit this plot's values, or None.

        The Poisson lattice is the integer photoelectron count.  The product's
        photoelectron readout publishes values with no unit; a camera read raw
        publishes its ``count`` (an ADU under an unknown gain), and any other
        unit is not a count at all.  Fitting a raw MOT ROI's pixel histogram
        put a 0.03-photon comb under a 7-ADU peak narrower than seven photons
        allow; the answer is the calibration that converts the camera to
        photoelectrons, not a fit.
        """

        if PHOTOELECTRON_AXIS not in model.capabilities:
            return None
        if not self._is_histogram_plot():
            return "photon-counting fit models fit histograms"
        unit = self._value_quantity().canonical_unit
        if unit.dimension == "dimensionless":
            return None
        return (
            f"fit model {model.model_id!r} counts photoelectrons; these values "
            f"are in {unit.symbol!r}. Read the camera through a photoelectron "
            "calibration, or fit a Gaussian model"
        )

    def _fit_model_units_compatible(self, model: FitModelSpec) -> bool:
        if self._view is None:
            return False
        try:
            if self._fit_model_needs_photoelectrons(model) is not None:
                return False
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
            reason = None
            try:
                reason = self._fit_model_needs_photoelectrons(model)
            except (AttributeError, TypeError, ValueError):
                reason = None
            raise ValueError(
                reason
                or f"fit model {model.model_id!r} is incompatible with the plot axes"
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
        terms: dict[object, object] = {}
        identity: list[tuple[str, str, object]] = []
        for ref, value in scope:
            resolved = resolve_axis(schema, ref)
            terms[resolved.axis_id] = value
            identity.append(
                (ref.domain.value, str(ref.axis_id or ""), value)
            )
        digest = ",".join(
            f"{domain}:{axis_id}={value!r}"
            for domain, axis_id, value in sorted(identity)
        )
        key = (
            source.ref.block_id,
            source.ref.stream_generation,
            source.ref.revision,
            digest,
        )
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
        self._view = DataView(
            self._scoped_data(),
            axis_display_units=overrides,
            value_display_unit=value_unit,
            unit_registry=self._unit_registry,
            inherit_domains_from=(
                self._view if self._view is not None else self._inherit_view
            ),
        )
        self._inherit_view = None

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
        history: RollingHistory,
        *,
        window: int,
        trailing: int = 1,
        uncertainty: bool = False,
    ) -> CurveData:
        """Build one history series per optional rolling group.

        ``trailing`` is how many shots each drawn point averages: 1 is the
        shot itself, N is the mean of the last N.  ``uncertainty`` draws the
        band -- the standard error of those same N shots when averaging,
        each shot's own pooled standard error when not.  The window selects
        the displayed tail and never changes the numbers; data older than
        Runtime's active bounded retention is never reconstructed by Plot,
        so a trailing mean averages what is actually retained.
        """

        if window <= 0:
            raise ValueError("rolling window must be positive")
        total = len(history)
        if not total:
            raise ValueError("rolling history cannot be empty")
        visible_size = min(window, total)
        start = total - visible_size
        keys = history.group_keys
        # Each drawn series is a column slice of the history planes -- no
        # per-shot objects, no per-shot key lookup.
        values_plane = np.asarray(history.values, dtype=float)[start:]
        valid_plane = np.asarray(history.valid, dtype=bool)[start:]
        masked_plane = np.where(valid_plane, values_plane, np.nan)
        sem_plane = None
        if history.sem is not None:
            sem_plane = np.where(
                valid_plane, np.asarray(history.sem, dtype=float)[start:], np.nan
            )
        # x is how many shots ago each point is: 0 is the newest, and the
        # ones behind it count back.  A rolling window shows the last N
        # shots, so what a point MEANS is its distance from now -- the
        # absolute shot number is a fact about the run, not about the
        # picture, and using it slid every label forward on every single
        # revision.  Once the window is full the axis stops moving, which
        # is also what lets a composed frame keep its cached chrome
        # instead of re-laying the tick labels on each shot.
        if history.source_indices is not None:
            source_coordinates = np.asarray(
                history.source_indices[start:], dtype=float
            )
        else:
            source_coordinates = np.arange(start, total, dtype=float)
        x_values = source_coordinates - source_coordinates[-1]
        unit = self._view.samples.value.display_unit
        canonical_unit = self._view.samples.value.canonical_unit
        x_unit = resolve_unit("1", DEFAULT_UNITS)
        display_plane = canonical_unit.convert_value_to(masked_plane, unit)
        x = QuantityArray(
            x_values,
            x_values,
            x_unit,
            x_unit,
            # Not a shot NUMBER: the axis says how far back a point is
            # from the newest one, which is what a rolling window shows.
            "Shots from latest",
        )
        series: list[CurveSeries] = []
        for column, key in enumerate(keys):
            sem = None
            if trailing > 1:
                canonical_values, running_sem, valid = _trailing_trace(
                    history, column, start, trailing
                )
                if uncertainty:
                    sem = running_sem
                display_values = canonical_unit.convert_value_to(
                    canonical_values, unit
                )
            else:
                valid = valid_plane[:, column]
                canonical_values = masked_plane[:, column]
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
                    sem=sem,
                    group_key=key,
                    label=label,
                )
            )
        return CurveData(
            revision=history.revision,
            generation=history.generation,
            x_ref=AxisRef.point_rows(),
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
    ) -> np.ndarray:
        """Return stable display-unit edges for one histogram projection.

        THE DOMAIN COVERS WHAT IS BINNED, so the caller says what that is.
        It is not always this revision's samples: a window is the last N
        shots, and a reduced spec bins one statistic per group, whose spread
        is narrower than the raw pool's by construction -- taken from the raw
        values, a reduced histogram landed in two bins out of twelve.  Named
        for the history alone, this argument only ever answered half of that.
        """

        count = int(state["bin_count"])
        samples = view.samples
        if binned_values is None:
            canonical = np.asarray(samples.value.canonical)
            valid = np.asarray(samples.valid_mask, dtype=bool)
        else:
            if binned_valid is None:
                raise ValueError("binned validity is required with binned values")
            canonical = np.asarray(binned_values)
            valid = np.asarray(binned_valid, dtype=bool)
        integral = canonical.dtype.kind in "biu"
        if integral:
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
            # The edge helper only needs the dtype to choose integer-aligned
            # bins once exact native min/max have been found.
            edge_values = np.empty(
                0,
                dtype=canonical.dtype if has_values else float,
            )
        else:
            # Masked extrema and a bounded probe, never a full finite
            # gather: the copy of a two-million-value pool cost more per
            # revision than the whole binning it fed.
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
                finite_count, kernel_low, kernel_high = extrema
                has_values = finite_count > 0
                if has_values:
                    data_low = kernel_low
                    data_high = kernel_high
                edge_values = finite_probe(flat, mask)
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
                edge_values = _finite_probe(flat, finite)
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
                    samples.value.display_unit.convert_value_to(
                        np.asarray(float(value), dtype=float),
                        samples.value.canonical_unit,
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

        edges = aligned_histogram_edges(edge_values, count, limits=(low, high))
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
            samples.value.canonical_unit.convert_value_to(
                previous.edges,
                samples.value.display_unit,
            ),
            dtype=float,
        )

    def _facet_mask(self, facet_index: int | None = None) -> np.ndarray | None:
        """The samples one cell owns, or None when no cell restricts them.

        NO RESTRICTION IS NOT A FULL PLANE.  A grid of one -- every plot
        that is not a facet -- answered this with a freshly allocated
        all-true array as wide as the dataset, which its one caller then
        ANDed into a mask it could not change: two megapixel passes and an
        allocation, 1.87 ms per gesture, to say nothing.
        """

        if self._view is None:
            raise TypeError("facet masking requires zlc_data.OwnedSnapshot")
        if not isinstance(self._spec, FacetGridPlot):
            return None
        cells = tuple(getattr(self._payload, "cells", ()))
        selected = self._focused_facet_index if facet_index is None else facet_index
        if selected is None or selected < 0 or selected >= len(cells):
            raise IndexError("facet index is outside the current grid")
        cell = cells[selected]
        if self._spec.facet is None:
            return None
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

    def _rolling_sample_offsets(self) -> np.ndarray:
        """The rolling x that each sample of THIS revision sits at.

        The rolling axis is "shots from latest", and one revision is one
        shot: every sample of it shares a single offset.  That offset is
        zero, because the window is a SUFFIX of the history and its x is
        measured from the newest entry (``absolute - absolute[-1]``), so
        the point the window ends on is this revision.  The points behind
        it are history entries, not samples of this snapshot -- nothing
        here can be selected at a negative offset.
        """

        if self._view is None:
            raise TypeError("rolling offsets require zlc_data.OwnedSnapshot")
        return np.zeros(self._view.samples.shape, dtype=float)

    def _x_sample_canonical(self) -> np.ndarray:
        """Where each sample sits on the x axis, in that axis's own space.

        ONE OWNER.  The range selector, the area filter and the crosshair
        each need it, and each derived it from ``_x_ref()`` -- which for a
        rolling plot is a PLACEHOLDER, a token standing in where the
        generic code needs an AxisRef shape.  Reading that token's
        coordinates handed back point-row ORDINALS (0..P-1) to be compared
        against shot OFFSETS (-(N-1)..0): ordinal 0 was the only value the
        two domains shared, so point rows 1..P-1 could never be selected,
        every surviving crosshair candidate reported the same x, and a
        range over the older shots came back with row 0 of every shot.
        """

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
        """Materialize the nearest valid plotted sample in display space."""

        if self._view is None or not isinstance(state.value, CrosshairPoint):
            raise TypeError("crosshair sample lookup requires zlc_data.OwnedSnapshot")
        displayed = self._display_selector_state(state)
        assert isinstance(displayed.value, CrosshairPoint)
        target = displayed.value
        samples = self._view.samples
        semantic = self._semantic_spec()
        # ``valid`` is already the finiteness answer -- see _selector_mask --
        # and asking it again of the CANONICAL values cast them to float64
        # first: an 11.35 ms copy of a 2048-square camera frame, per
        # gesture, to compute a plane that could not clear a bit.  What the
        # crosshair does need is finiteness of the DISPLAY coordinates it
        # measures distances in, and that is asked below.
        candidate = np.array(valid, copy=True, dtype=bool)
        candidate &= self._rolling_visible_mask()
        if isinstance(semantic, HistogramPlot):
            x_values = np.asarray(samples.value.display, dtype=float)
            y_values = np.full(samples.shape, target.y, dtype=float)
        else:
            x_values = (
                self._coordinate_values_to_display(
                    self._x_sample_canonical(), self._x_ref()
                )
                if isinstance(self._spec, RollingPlot)
                else np.asarray(self._coordinate(self._x_ref()).display, dtype=float)
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

        x_candidates = x_values.reshape(-1)[flat_indices]
        y_candidates = y_values.reshape(-1)[flat_indices]
        target_point = np.asarray((target.x, target.y), dtype=float)
        if point_transform is not None:
            # A transform is the one thing that needs the two coordinates
            # interleaved, because it is free to mix them.  Everything else
            # keeps them apart: stacking two million points into one
            # (N, 2) array copies both coordinates again for no reader.
            points = np.column_stack((x_candidates, y_candidates))
            try:
                transformed = np.asarray(
                    point_transform(np.vstack((points, target_point))),
                    dtype=float,
                )
                if transformed.shape != (points.shape[0] + 1, 2):
                    raise ValueError("point transform returned the wrong shape")
                x_candidates = transformed[:-1, 0]
                y_candidates = transformed[:-1, 1]
                target_point = transformed[-1]
            except (TypeError, ValueError):
                pass
        finite = np.isfinite(x_candidates) & np.isfinite(y_candidates)
        if not bool(finite.any()):
            return result
        if not bool(finite.all()):
            flat_indices = flat_indices[finite]
            x_candidates = x_candidates[finite]
            y_candidates = y_candidates[finite]
        delta_x = np.abs(x_candidates - target_point[0])
        delta_y = np.abs(y_candidates - target_point[1])
        if isinstance(semantic, ImagePlot):
            nearest = int(np.argmin(np.hypot(delta_x, delta_y)))
        else:
            # NEAREST IS A MINIMUM, NOT AN ORDER.  ``lexsort`` ranked every
            # one of two million candidates to read element zero -- 384.2 ms
            # of a 520.9 ms hover.  The smallest x distance, then the
            # smallest y distance among those tied for it, names the same
            # sample: both this and ``lexsort`` are stable, so ties resolve
            # to the lowest flat index either way.
            closest_x = delta_x.min()
            tied = np.flatnonzero(delta_x == closest_x)
            nearest = int(
                tied[0]
                if tied.size == 1
                else tied[int(np.argmin(delta_y[tied]))]
            )
        result.reshape(-1)[flat_indices[nearest]] = True
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
        mask = np.array(samples.valid_mask, copy=True, dtype=bool)
        cell = self._facet_mask(state.facet_index)
        if cell is not None:
            mask &= cell
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
        elif state.kind is SelectorKind.CROSSHAIR:
            return self._crosshair_sample_mask(state, mask, point_transform)
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
            _authority=authority,
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
            return "count"
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
            offset = self._display_fit_parameter_value(parameter, 0.0)[0]
            scale = self._display_fit_parameter_value(parameter, 1.0)[0] - offset
            converted = (displayed - offset) / scale
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
