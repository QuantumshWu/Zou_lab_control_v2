"""Fit request orchestration owned by one PlotSession facade."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field, replace
import math
from threading import Event
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence
import warnings

import numpy as np

from ._fit_projection import FitProjection, FitSelection
from ._fit_scene import FitOverlay
from ._gesture_engine import _SelectorGesture
from ._session_state import (
    _AcceptedFit,
    _FitPresentation,
    _FitResolution,
    _LiveFitRequest,
    _PreparedLiveFrame,
    _SolvedLiveFit,
    _StartedFitRequest,
    FitEvent,
)
from .fit import (
    FacetFitBatchResult,
    FitCancelled,
    FitDeadlineExceeded,
    FitModelSpec,
    FitOptions,
    FitResult,
    ParameterDomain,
    UnitRelation,
    _bimodal_classifier_metrics,
)
from .parameters import RenderEffect
from .selectors import (
    SelectorKind,
    SelectorState,
    _classifier_threshold_key,
    normalize_classifier_threshold_targets,
)
from .specs import FacetGridPlot, HistogramPlot

if TYPE_CHECKING:
    from .session import PlotSession

# ``None`` withdraws: there is no accepted fit any more, which is
# what SelectionChange.REMOVED says on the selection channel.
FitCallback = Callable[[FitEvent | None], object]

# Stable warm-start generation for the threshold classifier: its per-cell
# seeds survive data revisions independently of user fit requests.
_CLASSIFIER_REQUEST_GENERATION = -1

# Facet cells are solved serially: scipy's trf holds the GIL for these small
# per-cell problems, and a measured 4-thread pool was ~20% SLOWER than the
# serial loop (25.9 ms pooled vs 14.2 ms serial for 8x4000-point solves).
# Raise this only with a measurement showing the pool winning.

# Warm-seed memory hardening: a degenerate accepted solve must never seed
# later revisions (one poisoned seed used to replicate itself into every
# following frame).  Thresholds, in order of application:
# * an image radius beyond half its axis span leaves no curvature inside the
#   frame, so the fitted center is unconstrained (degenerate accepts observed
#   in the wild pinned the center at a frame edge with a radius tens of
#   frames wide);
# * an image amplitude under 0.1% of the observed value range is
#   indistinguishable from a flat frame, so the geometry parameters carry no
#   information;
# * a reduced chi-square two orders of magnitude above the remembered seed's
#   (and above the data's own noise floor, so exact synthetic fits with
#   near-zero chi-square never trip the ratio) means the scene no longer
#   resembles the one the seed described: the memory is dropped entirely and
#   the next solve runs cold rather than descending from suspect parameters.
# A rejected seed costs exactly one cold seed search on the next revision
# (tens of milliseconds), so false positives are cheap.
_WARM_SEED_MAX_RADIUS_SPAN_FRACTION = 0.5
_WARM_SEED_MIN_AMPLITUDE_FRACTION = 1e-3
_WARM_SEED_CHI_SQUARE_BLOWUP = 100.0
_WARM_SEED_NOISE_FLOOR_RANGE_FRACTION = 1e-6

_UNRESOLVED_WARM_START: Any = object()


@dataclass(frozen=True, slots=True)
class _WarmSeed:
    """One accepted solution remembered as the next revision's warm seed."""

    parameters: tuple[float, ...]
    reduced_chi_square: float
    # (range * _WARM_SEED_NOISE_FLOOR_RANGE_FRACTION)**2 of the accept's own
    # data; the chi-square blow-up ratio is measured above this floor.
    noise_floor: float


class _LiveSelectionCell:
    """Mutable slot carrying a worker-built live selection to acceptance."""

    __slots__ = ("selection",)

    def __init__(self) -> None:
        self.selection: FitSelection | None = None


@dataclass(frozen=True, slots=True)
class _DeferredStartedFitRequest(_StartedFitRequest):
    """A live restart whose selection freeze runs inside the solver task."""

    selection_cell: _LiveSelectionCell = field(default_factory=_LiveSelectionCell)


class FitSessionMixin:

    def expire_active_live_fit(self) -> None:
        """Expire the current live solve while keeping its request armed."""

        with self._lock:
            prepare_cancel = self._live_prepare_cancel
            fit_cancel = self._live_fit_cancel
            # Later admitted pairs inherit a fresh request-scoped token.  The
            # active pair retains the old one and observes it as cancelled.
            self._live_fit_cancel = Event()
        prepare_cancel.set()
        fit_cancel.set()

    def _commit_fit_actions(self, *actions: Callable[[], None]) -> None:
        """Run irreversible fit retirement only after configuration commits."""

        pending = self._configuration_fit_commit_actions
        if pending is None:
            for action in actions:
                action()
            return
        pending.extend(actions)

    def _stamp_fit_batch_revision(
        self,
        result: FitResult | FacetFitBatchResult,
    ) -> FitResult | FacetFitBatchResult:
        """Assign one monotonic publication revision to a completed fit batch."""

        with self._lock:
            self._fit_batch_revision += 1
            batch_revision = self._fit_batch_revision
        if isinstance(result, FacetFitBatchResult):
            return replace(result, batch_revision=batch_revision)
        return result.with_batch_revision(batch_revision)

    @staticmethod
    def _facet_batch_geometry(
        projection: FitProjection,
        cells: Sequence[Any],
    ) -> dict[str, object]:
        """Describe one facet axis identically for successes and gaps."""

        values = tuple(cell.facet_value_canonical for cell in cells)
        numeric = all(
            isinstance(value, (int, float, np.number))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        )
        coordinate = projection._coordinate(projection._spec.facet)
        return {
            "facet": projection._spec.facet,
            "facet_values": values,
            "sample_axis_name": coordinate.label,
            "sample_coordinates": (
                np.asarray(values, dtype=np.float64)
                if numeric
                else np.arange(len(cells), dtype=np.float64)
            ),
            "sample_unit": (
                ""
                if not numeric or coordinate.canonical_unit.symbol == "1"
                else coordinate.canonical_unit.symbol
            ),
            "sample_labels": (
                None if numeric else tuple(cell.label for cell in cells)
            ),
        }

    def _failed_live_fit_event(
        self,
        request: _LiveFitRequest,
        projection: FitProjection,
        source_revision: int,
        error: BaseException,
    ) -> FitEvent:
        """One explicit invalid parameter sample for an unpresented revision."""

        model = request.model
        message = str(error) or type(error).__name__
        units = projection._fit_parameter_units(model)
        if request.all_facets and isinstance(projection._spec, FacetGridPlot):
            cells = tuple(getattr(projection.payload, "cells", ()))
            if not cells:
                raise RuntimeError("failed facet fit has no cells to identify")
            failed = FacetFitBatchResult(
                model=model,
                results=(None,) * len(cells),
                failure_messages=(message,) * len(cells),
                source_revision=int(source_revision),
                overlays=tuple(
                    FitOverlay(
                        success=False,
                        diagnostic=message,
                        facet_index=index,
                    )
                    for index in range(len(cells))
                ),
                parameter_units=units,
                **self._facet_batch_geometry(projection, cells),
            )
            result = self._stamp_fit_batch_revision(failed)
            return FitEvent(result, None, (), "", result.overlays)

        count = len(model.parameters)
        invalid = np.full(count, np.nan, dtype=np.float64)
        failed = FitResult(
            model=model,
            parameter_values=invalid,
            standard_errors=invalid,
            covariance=np.full((count, count), np.nan, dtype=np.float64),
            fitted_values=np.asarray([], dtype=np.float64),
            residuals=np.asarray([], dtype=np.float64),
            selected_indices=np.asarray([], dtype=np.int64),
            source_revision=int(source_revision),
            success=False,
            message=message,
            reduced_chi_square=0.0,
            covariance_valid=False,
            parameter_units=units,
        )
        return FitEvent(self._stamp_fit_batch_revision(failed), None, (), "")

    def publish_live_fit_gap(
        self,
        source_revision: int,
        error: BaseException,
        *,
        projection: FitProjection | None = None,
    ) -> bool:
        """Publish a gap without replacing the last accepted data/fit front."""

        revision = int(source_revision)
        if revision < 0:
            raise ValueError("fit gap source_revision must be non-negative")
        with self._lock:
            if self._closed or self._live_fit_request is None:
                return False
            request = self._live_fit_request
            selected = self._projected if projection is None else projection
        if not isinstance(selected, FitProjection):
            raise TypeError("fit gap projection must be FitProjection or None")
        self._notify_fit(
            self._failed_live_fit_event(request, selected, revision, error)
        )
        return True

    def _forget_fit_warm_starts(self, request_generation: int | None) -> None:
        """Drop seeds after any non-normal solve completion."""

        with self._lock:
            if request_generation is None:
                self._fit_warm_starts.clear()
                return
            for key in tuple(self._fit_warm_starts):
                if key[0] == request_generation:
                    self._fit_warm_starts.pop(key, None)

    def _fit_warm_start(
        self,
        model: FitModelSpec,
        requested: Mapping[str, float] | Sequence[float] | None,
        *,
        facet_index: int | None,
        request_generation: int | None,
    ) -> Mapping[str, float] | Sequence[float] | None:
        """Use the last accepted cell parameters for the same live request."""

        if request_generation is None:
            return requested
        with self._lock:
            warm = self._fit_warm_starts.get(
                (request_generation, model.model_id, facet_index)
            )
        return None if warm is None else warm.parameters

    @staticmethod
    def _observed_value_range(selection: FitSelection | None) -> float | None:
        """Span of the finite observed values behind one fit selection."""

        if selection is None:
            return None
        observations = np.asarray(selection.observations)
        if observations.size == 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = float(np.nanmin(observations))
            high = float(np.nanmax(observations))
        if not math.isfinite(low) or not math.isfinite(high):
            return None
        return high - low

    @classmethod
    def _propose_warm_seed(
        cls,
        fit: FitResult | None,
        selection: FitSelection | None,
    ) -> _WarmSeed | None:
        """Build the seed one accepted result may leave for the next revision.

        Returns None for failed results and for regular-image accepts that
        cannot localize a blob (radius beyond half the axis span, or
        amplitude lost in the value range); see the threshold notes above.
        """

        if fit is None or not fit.success:
            return None
        value_range = cls._observed_value_range(selection)
        regular = None if selection is None else selection.regular_image
        if regular is not None:
            parameters = fit.parameters
            spans = {
                UnitRelation.AXIS_0: float(np.ptp(regular.x_coordinates)),
                UnitRelation.AXIS_1: float(np.ptp(regular.y_coordinates)),
            }
            for spec in fit.model.parameters:
                span = spans.get(spec.unit_relation)
                if (
                    spec.domain is ParameterDomain.POSITIVE
                    and span is not None
                    and float(parameters[spec.name])
                    > span * _WARM_SEED_MAX_RADIUS_SPAN_FRACTION
                ):
                    return None
            amplitude = parameters.get("amplitude")
            if (
                amplitude is not None
                and value_range is not None
                and abs(amplitude)
                < value_range * _WARM_SEED_MIN_AMPLITUDE_FRACTION
            ):
                return None
        noise_floor = (
            0.0
            if value_range is None
            else (value_range * _WARM_SEED_NOISE_FLOOR_RANGE_FRACTION) ** 2
        )
        return _WarmSeed(
            tuple(float(value) for value in fit.parameter_values),
            float(fit.reduced_chi_square),
            noise_floor,
        )

    def _remember_fit_warm_starts(
        self,
        result: FitResult | FacetFitBatchResult,
        *,
        request_generation: int,
        selections: Sequence[FitSelection | None] = (),
    ) -> None:
        """Publish only credible accepted results as seeds for the next revision.

        ``_propose_warm_seed`` rejects degenerate accepts outright, and an
        accept whose reduced chi-square blew up against the remembered
        seed's drops the memory entirely (see the threshold notes above).
        """

        if isinstance(result, FacetFitBatchResult):
            fits: tuple[FitResult | None, ...] = result.results
            keys = tuple(
                (request_generation, result.model.model_id, index)
                for index in range(len(fits))
            )
        else:
            fits = (result,)
            keys = ((request_generation, result.model.model_id, None),)
        padded = tuple(selections)[: len(fits)]
        padded += (None,) * (len(fits) - len(padded))
        # The degeneracy inspection touches the selection's observations, so
        # it runs before the session lock is taken.
        proposed = tuple(
            self._propose_warm_seed(fit, selection)
            for fit, selection in zip(fits, padded, strict=True)
        )
        with self._lock:
            if request_generation != self._fit_request_generation:
                return
            for key, seed in zip(keys, proposed, strict=True):
                if seed is None:
                    self._fit_warm_starts.pop(key, None)
                    continue
                remembered = self._fit_warm_starts.get(key)
                if (
                    remembered is not None
                    and seed.reduced_chi_square
                    > _WARM_SEED_CHI_SQUARE_BLOWUP
                    * max(remembered.reduced_chi_square, seed.noise_floor)
                ):
                    self._fit_warm_starts.pop(key, None)
                    continue
                self._fit_warm_starts[key] = seed

    def fit_selection(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
    ) -> FitSelection:
        """Freeze the exact painted samples that a fit would consume."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
                resolved = self._resolve_fit_model(model)
                return self._projected.fit_selection(
                    resolved,
                    selector_kind=selector_kind,
                )

    def fit(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
        initial: Mapping[str, float] | Sequence[float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: FitOptions | None = None,
        cancelled: Callable[[], bool] | None = None,
        live: bool = True,
        fit_all_facets: bool = False,
    ) -> FitResult | FacetFitBatchResult:
        """Fit now and, by default, keep the same fit active for later data."""

        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        if not isinstance(fit_all_facets, bool):
            raise TypeError("fit_all_facets must be bool")
        if fit_all_facets and live:
            raise ValueError("fit_all_facets cannot be a live fit")
        started = self._begin_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
            live=live,
            all_facets=fit_all_facets,
            logical_completion=None,
        )
        if started is None:
            raise RuntimeError("a synchronous fit request has no selection")
        try:
            result = self._solve_started_fit(started, cancelled=cancelled)
        except BaseException:
            self._forget_fit_warm_starts(started.request_generation)
            raise
        if (
            not started.cancellation.is_set()
            and self.data_revision == result.source_revision
        ):
            event = self._accept_fit(
                result,
                started,
            )
            if event is not None:
                self._notify_fit(event.event)
        return result

    def fit_async(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
        initial: Mapping[str, float] | Sequence[float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: FitOptions | None = None,
        live: bool = True,
        fit_all_facets: bool = False,
    ) -> Future[FitResult | FacetFitBatchResult]:
        """Fit off-thread; a live request resolves on its first accepted revision."""

        if not isinstance(fit_all_facets, bool):
            raise TypeError("fit_all_facets must be bool")
        if fit_all_facets and live:
            raise ValueError("fit_all_facets cannot be a live fit")
        logical_completion: Future[FitResult | FacetFitBatchResult] = Future()
        started = self._begin_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
            live=live,
            all_facets=fit_all_facets,
            logical_completion=logical_completion,
        )
        if started is None:
            return logical_completion
        self._submit_started_fit(
            started,
            live=live,
            logical_completion=logical_completion,
        )
        return logical_completion

    def _configure_fit_target(
        self,
        target: Mapping[str, object],
        *,
        live: bool,
    ) -> FitResult | FacetFitBatchResult | None:
        """Make one authored fit target current without replaying it."""

        if not isinstance(target, Mapping):
            raise TypeError("fit target must be a mapping")
        if type(live) is not bool:
            raise TypeError("fit live must be bool")
        values = dict(target)
        model = values.pop("model", None)
        values.pop("live", None)

        def clear_current() -> None:
            with self._lock:
                changed = self._live_fit_request is not None or self._accepted_fit is not None
            if changed:
                self.clear_fit()

        if model is None or str(model).strip() == "":
            clear_current()
            return None

        if not isinstance(model, FitModelSpec):
            try:
                self._fit_engine.registry.get(str(model))
            except ValueError:
                clear_current()
                return None

        selector_kind = values.pop("selector_kind", None)
        initial = values.pop("initial", None)
        bounds = values.pop("bounds", None)
        options = values.pop("options", None)
        fit_all_facets = values.pop("fit_all_facets", False)
        if values:
            raise TypeError(
                f"unknown fit target fields: {', '.join(sorted(map(str, values)))}"
            )
        if selector_kind is not None and not isinstance(
            selector_kind, SelectorKind
        ):
            selector_kind = SelectorKind(str(selector_kind))
        prepared = self._prepare_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
        )
        prepared = replace(
            prepared,
            all_facets=bool(fit_all_facets) or (
                live and isinstance(self._spec, FacetGridPlot)
            ),
        )
        with self._lock:
            if live and self._live_fit_request == prepared:
                return None
        started = self._begin_fit_request(
            prepared.model,
            selector_kind=None,
            initial=None,
            bounds=None,
            options=None,
            live=live,
            all_facets=prepared.all_facets,
            logical_completion=None,
            prepared_request=prepared,
        )
        if started is None:
            raise RuntimeError("configured fit request has no selection")
        try:
            result = self._solve_started_fit(started)
        except BaseException:
            self._forget_fit_warm_starts(started.request_generation)
            raise
        presentation = self._accept_fit(result, started)
        if presentation is None:
            raise FitCancelled("fit target was superseded before acceptance")
        self._notify_fit(presentation.event)
        return result

    def _fit_facet_batch(
        self,
        projection: FitProjection,
        model: FitModelSpec,
        *,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
        cancelled: Callable[[], bool] | None,
        request_generation: int | None = None,
        selector_kind: SelectorKind | None = None,
    ) -> tuple[FacetFitBatchResult, tuple[FitSelection | None, ...]]:
        """Fit every projected cell and construct overlays through one path.

        A previous result enters as a WARM START -- a candidate that competes
        with the model's own initializer and wins only by fitting better.  It
        used to be handed in as ``initial`` for classifier refreshes, which
        REPLACES the cold seed, with a fallback only if that solve raised.  A
        wrong-but-converged solve does not raise, so once a live classifier
        found a degenerate answer on its first thirty shots, every later
        revision started from it and stayed there for the rest of the run.
        """

        if not isinstance(projection._spec, FacetGridPlot):
            raise TypeError("facet batch projection requires FacetGridPlot")
        payload = projection.payload
        cells = tuple(getattr(payload, "cells", ()))
        if not cells:
            raise ValueError("facet grid has no cells to fit")
        # Warm starts are resolved on the calling thread so cell workers never
        # touch session locks (the classifier solves under the session lock).
        warm_starts = tuple(
            self._fit_warm_start(
                model,
                None,
                facet_index=index,
                request_generation=request_generation,
            )
            for index in range(len(cells))
        )

        def solve_cell(
            index: int,
        ) -> tuple[FitResult | None, str | None, FitSelection | None, FitOverlay]:
            if cancelled is not None and bool(cancelled()):
                raise FitCancelled("facet fit cancelled")
            warm = warm_starts[index]
            try:
                # The request's own selector, not the default priority: the
                # batch used to solve one domain and the accepted record report
                # another, because only the recording path was told.
                # One batch, one domain: the region the operator drew
                # applies to every cell, so only the computed cell varies --
                # the focused facet still says whose selectors may speak.
                selection = projection.fit_selection(
                    model, selector_kind=selector_kind, facet_index=index
                )
                result = self._solve_fit_selection(
                    projection,
                    model,
                    selection,
                    initial=initial,
                    bounds=bounds,
                    options=options,
                    cancelled=cancelled,
                    request_generation=request_generation,
                    warm_start=warm,
                )
                overlay = projection._make_fit_overlay(result, selection)
            except FitCancelled:
                raise
            except Exception as error:
                message = str(error) or type(error).__name__
                return (
                    None,
                    message,
                    None,
                    FitOverlay(
                        success=False,
                        diagnostic=message,
                        facet_index=index,
                    ),
                )
            return result, None, selection, overlay

        outcomes = [solve_cell(index) for index in range(len(cells))]
        results = [outcome[0] for outcome in outcomes]
        failure_messages = [outcome[1] for outcome in outcomes]
        selections = [outcome[2] for outcome in outcomes]
        overlays = [outcome[3] for outcome in outcomes]
        parameter_units = next(
            (
                result.parameter_units
                for result in results
                if result is not None and result.parameter_units
            ),
            projection._fit_parameter_units(model),
        )
        batch = FacetFitBatchResult(
            model=model,
            results=tuple(results),
            failure_messages=tuple(failure_messages),
            source_revision=projection.data_revision,
            overlays=tuple(overlays),
            parameter_units=parameter_units,
            **self._facet_batch_geometry(projection, cells),
        )
        return batch, tuple(selections)

    def _begin_fit_request(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
        live: bool,
        all_facets: bool = False,
        logical_completion: Future[FitResult | FacetFitBatchResult] | None,
        prepared_request: _LiveFitRequest | None = None,
    ) -> _StartedFitRequest | None:
        """Atomically replace fit request authority and freeze one solver input."""

        if not isinstance(live, bool):
            raise TypeError("live must be bool")
        if not isinstance(all_facets, bool):
            raise TypeError("all_facets must be bool")
        if all_facets and not isinstance(self._spec, FacetGridPlot):
            raise TypeError("fit_all_facets requires FacetGridPlot")
        if prepared_request is None:
            request = self._prepare_fit_request(
                model,
                selector_kind=selector_kind,
                initial=initial,
                bounds=bounds,
                options=options,
            )
            request = replace(
                request,
                all_facets=all_facets or (
                    live and isinstance(self._spec, FacetGridPlot)
                ),
            )
        else:
            request = prepared_request
        selection: FitSelection | None = None
        with self._render_lock:
            with self._lock:
                self._assert_open()
                projection = self._projected
                if not request.all_facets:
                    try:
                        selection = projection.fit_selection(
                            request.model,
                            selector_kind=request.selector_kind,
                        )
                    except (TypeError, ValueError):
                        if logical_completion is None or not live:
                            raise
                previous_fit_cancel = self._fit_cancel
                previous_live_fit_cancel = self._live_fit_cancel
                self._fit_context_generation += 1
                self._fit_request_generation += 1
                self._fit_warm_starts.clear()
                superseded = self._live_fit_completion
                self._live_fit_completion = logical_completion if live else None
                cancellation = Event()
                self._fit_cancel = cancellation
                # Replacing the request authority is one of the only three
                # events allowed to cancel an in-flight live pair solve (the
                # others are replace_spec and close).  Data-frame pairs then
                # share the fresh token until the next replacement.
                self._live_fit_cancel = Event()
                self._live_fit_request = request if live else None
                started = (
                    _StartedFitRequest(
                        request=request,
                        selection=selection,
                        projection=projection,
                        cancellation=cancellation,
                        context_generation=self._fit_context_generation,
                        request_generation=self._fit_request_generation,
                    )
                    if selection is not None or request.all_facets
                    else None
                )
        def retire_previous_request() -> None:
            previous_fit_cancel.set()
            previous_live_fit_cancel.set()
            if superseded is not None and not superseded.done():
                superseded.set_exception(FitCancelled("fit request superseded"))

        self._commit_fit_actions(retire_previous_request)
        return started

    def _pair_started(
        self,
        projection: FitProjection,
    ) -> _DeferredStartedFitRequest | None:
        """Freeze the armed live request against one prepared data frame.

        The selection freeze runs inside the solver task (from the captured
        immutable projection), so neither the caller nor the render worker
        pays for it.  Capacity-one admission keeps one solve active; its
        deadline rotates the request-scoped cancellation token so the retained
        latest input inherits a fresh one. Re-arm, replace_spec and close also
        retire the previous request token.
        """

        with self._lock:
            if self._closed or self._live_fit_request is None:
                return None
            return _DeferredStartedFitRequest(
                request=self._live_fit_request,
                selection=None,
                projection=projection,
                cancellation=self._live_fit_cancel,
                context_generation=self._fit_context_generation,
                request_generation=self._fit_request_generation,
            )

    def _solve_live_pair(
        self,
        started: _DeferredStartedFitRequest,
        cancelled: Callable[[], bool] | None = None,
    ) -> _SolvedLiveFit:
        """Solve one exact pair or fail it without advancing visible data."""

        try:
            result = self._solve_started_fit(started, cancelled=cancelled)
            if started.cancellation.is_set() or (
                cancelled is not None and bool(cancelled())
            ):
                raise FitCancelled("live fit pair was cancelled")
        except Exception as error:
            if not isinstance(error, (FitCancelled, FitDeadlineExceeded)):
                self._forget_fit_warm_starts(started.request_generation)
            failed = self._retire_failed_live_presentation(started, error)
            if failed is not None:
                self._resolve_fit_completion(failed)
            raise
        return _SolvedLiveFit(started, result)

    def solve_live_frame(
        self,
        prepared: object,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Future[_SolvedLiveFit] | None:
        """Start the fit half of one data/fit pair on the analysis executor.

        Returns None only when no live fit is armed.  An armed solve either
        resolves with its matching result or raises loudly; it never converts
        failure into a data-only presentation.
        """

        if not isinstance(prepared, _PreparedLiveFrame):
            raise TypeError("prepared must be a prepared live frame")
        if prepared.session_identity is not self._session_identity:
            raise ValueError("prepared live frame belongs to another PlotSession")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        started = self._pair_started(prepared.projection)
        if started is None:
            return None
        try:
            return self._analysis_executor.submit(
                self._solve_live_pair,
                started,
                cancelled,
            )
        except RuntimeError as error:
            raise FitCancelled("analysis executor is not available") from error

    def _accept_pair_fit(
        self,
        solved: _SolvedLiveFit,
        projection: FitProjection,
    ) -> tuple[_AcceptedFit, _FitResolution | None, FitEvent]:
        """Build matching fit state before the pair's one render transaction."""

        result = solved.result
        resolution: _FitResolution | None = None
        with self._lock:
            request_current = (
                not self._closed
                and self._live_fit_request is not None
                and solved.started.request_generation
                == self._fit_request_generation
            )
            if not request_current:
                raise FitCancelled("fit request changed before pair acceptance")
            if solved.started.projection is not projection:
                raise FitCancelled("solved fit does not match the prepared projection")
            accepted, _selections = self._resolved_accepted_fit(
                result,
                solved.started,
                projection,
            )
            event = self._fit_event(accepted)
            completion = self._live_fit_completion
            if completion is not None:
                self._live_fit_completion = None
        if completion is not None:
            resolution = _FitResolution(completion, result=result)
        return accepted, resolution, event

    def _restore_live_fit_completion(
        self,
        resolution: _FitResolution | None,
    ) -> None:
        """Return an unpromoted pair's logical completion to its live request."""

        if resolution is None or resolution.completion.done():
            return
        with self._lock:
            if not self._closed and self._live_fit_completion is None:
                self._live_fit_completion = resolution.completion

    def _prepare_fit_request(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
    ) -> _LiveFitRequest:
        model_spec = self._resolve_fit_model(model)
        if options is not None and not isinstance(options, FitOptions):
            raise TypeError("options must be FitOptions or None")
        frozen_initial: Mapping[str, float] | tuple[float, ...] | None
        if initial is None:
            frozen_initial = None
        elif isinstance(initial, Mapping):
            frozen_initial = MappingProxyType(
                {str(name): float(value) for name, value in initial.items()}
            )
        elif isinstance(initial, (str, bytes)):
            raise TypeError("initial must be a parameter mapping or numeric sequence")
        else:
            try:
                frozen_initial = tuple(float(value) for value in initial)
            except TypeError as error:
                raise TypeError(
                    "initial must be a parameter mapping or numeric sequence"
                ) from error
        frozen_bounds = None
        if bounds is not None:
            if not isinstance(bounds, Mapping):
                raise TypeError("bounds must be a mapping or None")
            prepared_bounds: dict[str, tuple[float | None, float | None]] = {}
            for name, pair in bounds.items():
                try:
                    low, high = pair
                except (TypeError, ValueError) as error:
                    raise TypeError("each fit bound must contain low and high") from error
                prepared_bounds[str(name)] = (
                    None if low is None else float(low),
                    None if high is None else float(high),
                )
            frozen_bounds = MappingProxyType(prepared_bounds)
        if selector_kind is not None and not isinstance(selector_kind, SelectorKind):
            raise TypeError("selector_kind must be SelectorKind or None")
        return _LiveFitRequest(
            model_spec,
            selector_kind,
            frozen_initial,
            frozen_bounds,
            options,
        )

    def _solve_started_fit(
        self,
        started: _StartedFitRequest,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> FitResult | FacetFitBatchResult:
        """Execute one frozen fit request on the caller or analysis worker."""

        def should_cancel() -> bool:
            return started.cancellation.is_set() or (
                cancelled is not None and bool(cancelled())
            )

        if started.request.all_facets:
            batch, _selections = self._fit_facet_batch(
                started.projection,
                started.request.model,
                initial=started.request.initial,
                bounds=started.request.bounds,
                options=started.request.options,
                cancelled=should_cancel,
                request_generation=started.request_generation,
                selector_kind=started.request.selector_kind,
            )
            return self._stamp_fit_batch_revision(batch)
        selection = started.selection
        if selection is None:
            # Deferred live restart: freeze the selection here, off the
            # render worker, from the captured immutable projection.
            try:
                selection = started.projection.fit_selection(
                    started.request.model,
                    selector_kind=started.request.selector_kind,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise FitCancelled(
                    "live fit selection is unavailable for this revision"
                ) from error
            if isinstance(started, _DeferredStartedFitRequest):
                started.selection_cell.selection = selection
        result = self._solve_fit_selection(
            started.projection,
            started.request.model,
            selection,
            initial=started.request.initial,
            bounds=started.request.bounds,
            options=started.request.options,
            cancelled=should_cancel,
            request_generation=started.request_generation,
        )
        return self._stamp_fit_batch_revision(result)

    def _submit_started_fit(
        self,
        started: _StartedFitRequest,
        *,
        live: bool,
        logical_completion: Future[FitResult | FacetFitBatchResult] | None,
    ) -> Future[FitResult | FacetFitBatchResult] | None:
        """Run one started fit on the analysis executor and route completion."""

        try:
            future = self._analysis_executor.submit(self._solve_started_fit, started)
        except Exception as error:
            self._forget_fit_warm_starts(started.request_generation)
            if not live and logical_completion is not None:
                logical_completion.set_exception(error)
            # The next data revision is the only automatic live-fit retry.
            return None
        tracked = False
        if live:
            with self._lock:
                tracked = (
                    not self._closed
                    and started.request_generation == self._fit_request_generation
                    and self._live_fit_request is started.request
                )
                if tracked:
                    self._live_fit_future = future
                else:
                    started.cancellation.set()
        future.add_done_callback(
            lambda completed: self._schedule_fit_completion(
                completed,
                started,
                logical_completion=None if live else logical_completion,
                tracked=tracked,
            )
        )
        return future

    def _solve_fit_selection(
        self,
        projection: FitProjection,
        model: FitModelSpec,
        selection: FitSelection,
        *,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
        cancelled: Callable[[], bool] | None,
        request_generation: int | None = None,
        warm_start: Any = _UNRESOLVED_WARM_START,
    ) -> FitResult:
        if warm_start is _UNRESOLVED_WARM_START:
            warm_start = self._fit_warm_start(
                model,
                None,
                facet_index=selection.facet_index,
                request_generation=request_generation,
            )
        regular = selection.regular_image
        if regular is not None:
            result = self._fit_engine.fit(
                model,
                regular,
                data_revision=selection.data_revision,
                initial=initial,
                warm_start=warm_start,
                bounds=bounds,
                options=options,
                cancelled=cancelled,
            )
        else:
            result = self._fit_engine.fit(
                model,
                selection.coordinates,
                selection.observations,
                observation_sigma=selection.observation_sigma,
                selected_indices=selection.selected_indices,
                data_revision=selection.data_revision,
                initial=initial,
                warm_start=warm_start,
                bounds=bounds,
                options=options,
                cancelled=cancelled,
            )
        if not isinstance(result, FitResult):
            raise TypeError("FitEngine.fit must return FitResult")
        # The engine may bind the model's origin to the window it was handed;
        # that copy is still this model, so identity is the model id.
        if result.model.model_id != model.model_id:
            raise RuntimeError("FitEngine returned a result for another model")
        if result.source_revision != selection.data_revision:
            raise RuntimeError("FitEngine returned a result for another data revision")
        return result.with_parameter_units(
            projection._fit_parameter_units(model)
        )

    def _threshold_classifier_enabled(self) -> bool:
        return bool(
            isinstance(self._semantic_spec, HistogramPlot)
            and self.display_state.values.get("threshold_classifier", False)
        )

    def _authored_classifier_result(
        self,
        model: FitModelSpec,
        selection: FitSelection,
        components: Mapping[str, float],
    ) -> FitResult:
        """Present a caller-owned Gaussian pair without independently refitting it."""

        left_mean = float(components["left_mean"])
        right_mean = float(components["right_mean"])
        left_sigma = float(components["left_sigma"])
        right_sigma = float(components["right_sigma"])
        left_weight = float(components["left_weight"])
        right_weight = float(components["right_weight"])
        x = np.asarray(selection.coordinates[0], dtype=float).reshape(-1)
        observed = np.asarray(selection.observations, dtype=float).reshape(-1)
        base = (
            left_weight
            / left_sigma
            * np.exp(-0.5 * ((x - left_mean) / left_sigma) ** 2)
            + right_weight
            / right_sigma
            * np.exp(-0.5 * ((x - right_mean) / right_sigma) ** 2)
        )
        denominator = float(np.dot(base, base))
        scale = (
            max(0.0, float(np.dot(base, observed)) / denominator)
            if denominator > 0.0
            else 0.0
        )
        values = {
            "center": 0.5 * (left_mean + right_mean),
            "center_splitting": right_mean - left_mean,
            "left_amplitude": scale * left_weight / left_sigma,
            "left_sigma": left_sigma,
            "right_amplitude": scale * right_weight / right_sigma,
            "right_sigma": right_sigma,
        }
        parameters = np.asarray(
            [values[name] for name in model.parameter_names], dtype=float
        )
        fitted = model.evaluate(selection.coordinates, parameters).reshape(-1)
        residuals = observed - fitted
        count = len(parameters)
        result = FitResult(
            model=model,
            parameter_values=parameters,
            standard_errors=np.full(count, np.nan, dtype=float),
            covariance=np.full((count, count), np.nan, dtype=float),
            fitted_values=fitted,
            residuals=residuals,
            selected_indices=selection.selected_indices,
            source_revision=selection.data_revision,
            success=True,
            message="authored Gaussian components",
            reduced_chi_square=float(
                np.dot(residuals, residuals) / max(observed.size - count, 1)
            ),
            covariance_valid=False,
        )
        return result.with_parameter_units(
            self._projected._fit_parameter_units(model)
        )

    def _apply_authored_classifier_components(
        self,
        model: FitModelSpec,
        results: Sequence[FitResult | None],
        overlays: Sequence[FitOverlay],
    ) -> tuple[tuple[FitResult | None, ...], tuple[FitOverlay, ...]]:
        components = self._classifier_gaussian_components
        if len(components) != len(results):
            components = (None,) * len(results)
            self._classifier_gaussian_components = components
        selected_results = list(results)
        selected_overlays = list(overlays)
        facet_grid = isinstance(self._spec, FacetGridPlot)
        for index, authored in enumerate(components):
            if authored is None:
                continue
            if not authored:
                selected_results[index] = None
                selected_overlays[index] = FitOverlay(
                    facet_index=index if facet_grid else None
                )
                continue
            selection = self._projected.fit_selection(
                model,
                facet_index=index if facet_grid else None,
            )
            result = self._authored_classifier_result(
                model, selection, authored
            )
            selected_results[index] = result
            selected_overlays[index] = self._projected._make_fit_overlay(
                result, selection
            )
        return tuple(selected_results), tuple(selected_overlays)

    def _refresh_threshold_classifier(self) -> None:
        """Solve the Distribution classifier without touching accepted fit state.

        The solve is synchronous: the threshold overlay publishes in the same
        front as the data it classifies.  Steady live revisions stay cheap
        because every cell warm-starts from its previous solution (through the
        stable classifier request generation) and cells solve in parallel.
        """

        if not self._threshold_classifier_enabled():
            self._classifier_results = ()
            self._classifier_overlays = ()
            self._classifier_thresholds = ()
            self._classifier_gaussian_components = ()
            try:
                self._selector_controller.remove(SelectorKind.THRESHOLD)
            except KeyError:
                pass
            return

        projection = self._projected
        model = self._resolve_fit_model("bimodal_gaussian")
        if isinstance(self._spec, FacetGridPlot):
            batch, _selections = self._fit_facet_batch(
                projection,
                model,
                initial=None,
                bounds=None,
                options=None,
                cancelled=None,
                request_generation=_CLASSIFIER_REQUEST_GENERATION,
            )
            results = batch.results
            overlays = batch.overlays
        else:
            selection = projection.fit_selection(model)
            warm = self._fit_warm_start(
                model,
                None,
                facet_index=None,
                request_generation=_CLASSIFIER_REQUEST_GENERATION,
            )
            result = self._solve_fit_selection(
                projection,
                model,
                selection,
                initial=None,
                bounds=None,
                options=None,
                cancelled=None,
                request_generation=_CLASSIFIER_REQUEST_GENERATION,
                warm_start=warm,
            )
            results = (result,)
            overlays = (projection._make_fit_overlay(result, selection),)
        self._remember_classifier_warm_starts(model, results)
        results, overlays = self._apply_authored_classifier_components(
            model, results, overlays
        )
        self._classifier_results = results
        self._classifier_overlays = overlays
        # ``_classifier_thresholds`` holds what somebody CHOSE, and nothing
        # else; the fit's own optimum is derived from the results above
        # whenever it is needed.  They shared this one slot, so every new
        # revision recomputed over the operator's dragged threshold and threw
        # it away without saying anything -- measured as a threshold dragged
        # to 150 reading 175.5 one shot later.
        chosen = self._classifier_thresholds
        # A different number of distributions is a different set of them, and
        # a decision about the old ones says nothing about these.
        self._classifier_thresholds = (
            chosen if len(chosen) == len(results) else (None,) * len(results)
        )
        try:
            self._selector_controller.remove(SelectorKind.THRESHOLD)
        except KeyError:
            pass

    def _remember_classifier_warm_starts(
        self,
        model: FitModelSpec,
        results: Sequence[FitResult | None],
    ) -> None:
        """Seed the next classifier refresh from each accepted cell solution."""

        facet = isinstance(self._spec, FacetGridPlot)
        with self._lock:
            for index, result in enumerate(results):
                key = (
                    _CLASSIFIER_REQUEST_GENERATION,
                    model.model_id,
                    index if facet else None,
                )
                seed = self._propose_warm_seed(result, None)
                if seed is None:
                    self._fit_warm_starts.pop(key, None)
                else:
                    self._fit_warm_starts[key] = seed

    @staticmethod
    def _fitted_classifier_threshold(result: object) -> float | None:
        """Where this fit says the two populations separate best."""

        if result is None or not result.success:
            return None
        return _bimodal_classifier_metrics(result)[0]

    def _classifier_thresholds_settled(self) -> tuple[float | None, ...]:
        """What each distribution is classified at: the choice, else the fit.

        THE composition, so a label, an overlay, a painted line and a
        fidelity can never be computed against different thresholds.
        """

        results = self._classifier_results
        if not results:
            return ()
        return tuple(
            self._fitted_classifier_threshold(result) if chosen is None else chosen
            for chosen, result in zip(
                self._classifier_thresholds, results, strict=True
            )
        )

    def _classifier_thresholds_for_render(self) -> tuple[float | None, ...]:
        thresholds = list(self._classifier_thresholds_settled())
        if not thresholds:
            return ()
        snapshot = self._selector_controller.snapshot()
        selected = (
            snapshot.candidate
            if snapshot.candidate is not None
            and snapshot.candidate.kind is SelectorKind.THRESHOLD
            else next(
                (
                    state
                    for state in snapshot.committed
                    if state.kind is SelectorKind.THRESHOLD
                ),
                None,
            )
        )
        if selected is not None:
            index = 0 if selected.facet_index is None else selected.facet_index
            if 0 <= index < len(thresholds):
                thresholds[index] = float(selected.value)
        return tuple(thresholds)

    def _classifier_labels(
        self,
        thresholds: tuple[float | None, ...],
        display_thresholds: tuple[float | None, ...],
    ) -> tuple[str, ...]:
        labels: list[str] = []
        for result, threshold, displayed in zip(
            self._classifier_results,
            thresholds,
            display_thresholds,
            strict=True,
        ):
            if (
                result is None
                or threshold is None
                or displayed is None
                or not result.success
            ):
                labels.append("")
                continue
            _threshold, left, _right, fidelity = _bimodal_classifier_metrics(
                result,
                threshold,
            )
            left_percent = round(100.0 * left, 1)
            right_percent = 100.0 - left_percent
            labels.append(
                f"Threshold {displayed:.4g}\n"
                f"L/R {left_percent:.1f}%/{right_percent:.1f}%\n"
                f"Fidelity {100.0 * fidelity:.1f}%"
            )
        return tuple(labels)

    def _derived_threshold_classifier_selector(self) -> SelectorState | None:
        """The threshold line to paint when no gesture is holding one."""

        settled = self._classifier_thresholds_settled()
        if not self._threshold_classifier_enabled() or not settled:
            return None
        index = 0 if self._focused_facet_index is None else self._focused_facet_index
        if index < 0 or index >= len(settled):
            return None
        threshold = settled[index]
        if threshold is None:
            return None
        return SelectorState(
            SelectorKind.THRESHOLD,
            float(threshold),
            facet_index=(index if isinstance(self._spec, FacetGridPlot) else None),
        )

    def _set_classifier_thresholds_state(
        self,
        thresholds: object,
    ) -> None:
        if not self._threshold_classifier_enabled():
            raise RuntimeError("threshold classifier is not enabled")
        selected = normalize_classifier_threshold_targets(thresholds)
        expected: dict[tuple[object, ...], int] = {}
        facet_grid = isinstance(self._spec, FacetGridPlot)
        for index in range(len(self._classifier_results)):
            target = self._classifier_threshold_target_for_index(
                index if facet_grid else None,
                0.0,
            )
            identity = _classifier_threshold_key(target)
            if identity in expected:
                raise ValueError(
                    "classifier distributions are not uniquely coordinate-addressed"
                )
            expected[identity] = index
        normalized: list[float | None] = [None] * len(self._classifier_results)
        components: list[Mapping[str, float] | None] = [
            None
        ] * len(self._classifier_results)
        for target in selected:
            identity = _classifier_threshold_key(target)
            index = expected.get(identity)
            if index is None:
                raise ValueError(
                    "classifier threshold target does not match a current distribution"
                )
            normalized[index] = float(target["value"])
            if "gaussian_components" in target:
                gaussian = target["gaussian_components"]
                components[index] = (
                    MappingProxyType({})
                    if gaussian is None
                    else MappingProxyType(dict(gaussian))
                )
        previous_components = self._classifier_gaussian_components
        self._classifier_thresholds = tuple(normalized)
        self._classifier_gaussian_components = tuple(components)
        if any(
            current is None and previous is not None
            for current, previous in zip(
                self._classifier_gaussian_components,
                previous_components
                if len(previous_components) == len(components)
                else (None,) * len(components),
                strict=True,
            )
        ):
            self._refresh_threshold_classifier()
        else:
            model = self._resolve_fit_model("bimodal_gaussian")
            results, overlays = self._apply_authored_classifier_components(
                model,
                self._classifier_results,
                self._classifier_overlays,
            )
            self._classifier_results = results
            self._classifier_overlays = overlays
        try:
            self._selector_controller.remove(SelectorKind.THRESHOLD)
        except KeyError:
            pass

    def _classifier_threshold_targets_state(self) -> tuple[Mapping[str, object], ...]:
        facet_grid = isinstance(self._spec, FacetGridPlot)
        targets: list[Mapping[str, object]] = []
        for index, value in enumerate(self._classifier_thresholds):
            if value is None:
                continue
            target = dict(
                self._classifier_threshold_target_for_index(
                    index if facet_grid else None,
                    value,
                )
            )
            if index < len(self._classifier_gaussian_components):
                gaussian = self._classifier_gaussian_components[index]
                if gaussian is not None:
                    target["gaussian_components"] = (
                        None if not gaussian else dict(gaussian)
                    )
            targets.append(target)
        return normalize_classifier_threshold_targets(
            tuple(targets)
        )

    def _schedule_fit_completion(
        self,
        future: Future[FitResult | FacetFitBatchResult],
        started: _StartedFitRequest,
        *,
        logical_completion: Future[FitResult | FacetFitBatchResult] | None,
        tracked: bool,
    ) -> None:
        resolutions: list[_FitResolution] = []
        presentations: list[_FitPresentation] = []

        def accept_and_paint() -> None:
            try:
                resolution, presentation = self._finish_fit_future(
                    future,
                    started,
                    logical_completion=logical_completion,
                    tracked=tracked,
                )
            except Exception as error:
                self._forget_fit_warm_starts(started.request_generation)
                if tracked:
                    failed = self._retire_failed_live_presentation(started, error)
                    if failed is not None:
                        resolutions.append(failed)
                raise
            if resolution is not None:
                resolutions.append(resolution)
            if presentation is not None:
                presentations.append(presentation)

        def finalize_presentation() -> None:
            for resolution in resolutions:
                self._resolve_fit_completion(resolution)
            for presentation in presentations:
                self._notify_fit(presentation.event)

        def abort_presentation() -> None:
            for presentation in reversed(presentations):
                self._abort_fit_presentation(presentation)

        try:
            with self._ownership_gate:
                with self._lock:
                    presentation_dispatch = self._presentation_dispatch
                if presentation_dispatch is None:
                    presented = self.owner_dispatch(accept_and_paint)
                else:
                    presented = presentation_dispatch(
                        accept_and_paint,
                        finalize_presentation,
                        abort_presentation,
                    )
                if not isinstance(presented, Future):
                    raise TypeError(
                        "host presentation dispatch must return "
                        "concurrent.futures.Future"
                    )
        except Exception as error:
            targets = tuple(resolutions) or (
                ()
                if logical_completion is None
                else (_FitResolution(logical_completion, error=error),)
            )
            for resolution in targets:
                self._resolve_fit_completion(resolution)
            return

        def presentation_finished(completion: Future[Any]) -> None:
            try:
                completion.result()
            except Exception as error:
                targets = tuple(resolutions) or (
                    ()
                    if logical_completion is None
                    else (_FitResolution(logical_completion, error=error),)
                )
                for resolution in targets:
                    target = resolution.completion
                    if not target.done():
                        target.set_exception(error)
            else:
                if presentation_dispatch is None:
                    finalize_presentation()

        presented.add_done_callback(presentation_finished)

    def _retire_failed_live_presentation(
        self,
        started: _StartedFitRequest,
        error: Exception,
    ) -> _FitResolution | None:
        """Fail the explicit fit request without scheduling a replacement solve."""

        with self._lock:
            if (
                self._closed
                or self._live_fit_request is not started.request
                or self._fit_request_generation != started.request_generation
            ):
                return None
            completion = self._live_fit_completion
            self._live_fit_completion = None
        return (
            None
            if completion is None
            else _FitResolution(completion, error=error)
        )

    @staticmethod
    def _resolve_fit_completion(resolution: _FitResolution) -> None:
        completion = resolution.completion
        if completion.done():
            return
        if resolution.error is not None:
            completion.set_exception(resolution.error)
        elif resolution.result is not None:
            completion.set_result(resolution.result)
        else:
            completion.set_exception(FitCancelled("fit did not reach presentation"))

    def _finish_fit_future(
        self,
        future: Future[FitResult | FacetFitBatchResult],
        started: _StartedFitRequest,
        *,
        logical_completion: Future[FitResult | FacetFitBatchResult] | None,
        tracked: bool,
    ) -> tuple[_FitResolution | None, _FitPresentation | None]:
        current = False
        if tracked:
            with self._lock:
                current = self._live_fit_future is future
                if current:
                    self._live_fit_future = None
        result: FitResult | None = None
        error: Exception | None = None
        if not future.cancelled():
            try:
                result = future.result()
            except FitCancelled as caught:
                error = caught
            except Exception as caught:
                error = caught
                self._forget_fit_warm_starts(started.request_generation)
                # A solver failure completes this explicit fit request.  A
                # later data revision is the only automatic opportunity to
                # solve again through the live-frame pipeline.
        else:
            error = FitCancelled("fit analysis was cancelled")
        presentation: _FitPresentation | None = None
        if result is not None and not started.cancellation.is_set():
            presentation = self._accept_fit(
                result,
                started,
            )
        accepted = presentation is not None
        resolution: _FitResolution | None = None
        if tracked and current:
            with self._lock:
                request_current = (
                    not self._closed
                    and self._live_fit_request is not None
                    and started.request_generation == self._fit_request_generation
                )
                if request_current and accepted:
                    accepted_completion = self._live_fit_completion
                    self._live_fit_completion = None
                else:
                    accepted_completion = None
            if accepted_completion is not None and result is not None:
                resolution = _FitResolution(accepted_completion, result=result)
        elif logical_completion is not None:
            if error is not None:
                resolution = _FitResolution(logical_completion, error=error)
            elif accepted and result is not None:
                resolution = _FitResolution(logical_completion, result=result)
            else:
                resolution = _FitResolution(
                    logical_completion,
                    error=FitCancelled(
                        "fit result was superseded before presentation"
                    ),
                )
        return resolution, presentation

    def _resolved_accepted_fit(
        self,
        result: FitResult | FacetFitBatchResult,
        started: _StartedFitRequest,
        projection: FitProjection,
    ) -> tuple[_AcceptedFit, tuple[FitSelection | None, ...]]:
        """Resolve one fit against exactly the projection it was solved from."""

        batch = result if isinstance(result, FacetFitBatchResult) else None
        selection = None if batch is not None else self._started_selection(started)
        if result.source_revision != projection.data_revision:
            raise RuntimeError("fit result does not match its data projection")
        if selection is not None and selection.data_revision != projection.data_revision:
            raise RuntimeError("fit selection does not match its data projection")
        if batch is None:
            if selection is None:
                raise RuntimeError("single fit acceptance has no selection")
            overlay = projection._make_fit_overlay(result, selection)
            overlays = (overlay,)
            selections = (selection,)
        else:
            overlay = None
            overlays = batch.overlays
            selections = self._facet_fit_selections(
                projection,
                batch.model,
                selector_kind=started.request.selector_kind,
            )
        return (
            _AcceptedFit(
                result=result,
                selection=selection,
                overlay=overlay,
                overlays=overlays,
                selections=selections,
                context_generation=started.context_generation,
            ),
            selections,
        )

    def _accept_fit(
        self,
        result: FitResult | FacetFitBatchResult,
        started: _StartedFitRequest,
    ) -> _FitPresentation | None:
        batch = result if isinstance(result, FacetFitBatchResult) else None
        with self._render_lock:
            with self._lock:
                if (
                    self._closed
                    or result.source_revision != self.data_revision
                    or started.request_generation != self._fit_request_generation
                ):
                    return None
                if (
                    batch is None
                    and not result.success
                    and isinstance(self._gesture, _SelectorGesture)
                    and self._fit_request_uses_selector(
                        started.request,
                        self._gesture.kind,
                    )
                ):
                    # A transient domain may briefly contain too little signal.
                    # Keep the last complete fit front instead of replacing it
                    # with a failure topology while the pointer is still down.
                    return None
                accepted, selections = self._resolved_accepted_fit(
                    result,
                    started,
                    self._projected,
                )
                previous = self._accepted_fit
                self._accepted_fit = accepted
            try:
                self._render_current(
                    RenderEffect.OVERLAY,
                    schedule_fit=False,
                )
            except Exception:
                with self._lock:
                    if self._accepted_fit is accepted:
                        self._accepted_fit = previous
                try:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
                except Exception:
                    self.redraw_surface()
                raise
        self._remember_fit_warm_starts(
            result,
            request_generation=started.request_generation,
            selections=selections,
        )
        return _FitPresentation(
            self._fit_event(accepted),
            accepted,
            previous,
        )

    @staticmethod
    def _started_selection(started: _StartedFitRequest) -> FitSelection | None:
        """Return the frozen selection, whether eager or worker-built."""

        if started.selection is not None:
            return started.selection
        if isinstance(started, _DeferredStartedFitRequest):
            return started.selection_cell.selection
        return None

    def _facet_fit_selections(
        self,
        projection: FitProjection,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
    ) -> tuple[FitSelection | None, ...]:
        """Recreate cell selections from the same frozen projection only."""

        if not isinstance(projection._spec, FacetGridPlot):
            raise TypeError("facet fit selections require FacetGridPlot")
        cells = tuple(getattr(projection.payload, "cells", ()))
        selections: list[FitSelection | None] = []
        for index, _cell in enumerate(cells):
            try:
                selections.append(
                    projection.fit_selection(
                        model,
                        selector_kind=selector_kind,
                        facet_index=index,
                    )
                )
            except (TypeError, ValueError, KeyError):
                selections.append(None)
        return tuple(selections)

    @staticmethod
    def _fit_event(accepted: _AcceptedFit) -> FitEvent:
        if isinstance(accepted.result, FacetFitBatchResult):
            return FitEvent(
                accepted.result,
                None,
                (),
                "",
                accepted.overlays,
            )
        overlay = accepted.overlay
        if overlay is None or accepted.selection is None:
            raise RuntimeError("single fit acceptance has no overlay or selection")
        return FitEvent(
            accepted.result,
            accepted.selection,
            overlay.parameter_display,
            overlay.formula,
            accepted.overlays,
        )

    def _abort_fit_presentation(self, presentation: _FitPresentation) -> None:
        """Restore the accepted fit preceding an unpromoted raster front."""

        if not isinstance(presentation, _FitPresentation):
            raise TypeError("presentation must be a fit presentation")
        with self._render_lock:
            with self._lock:
                if self._accepted_fit is not presentation.accepted:
                    raise RuntimeError("fit presentation is no longer current")
                self._accepted_fit = presentation.previous
            self._render_current(
                RenderEffect.OVERLAY,
                schedule_fit=False,
            )

    def subscribe_fit(self, callback: FitCallback) -> Callable[[], None]:
        """Observe exact fit results at their one authoritative completion.

        A live data-pair event follows its solve and precedes that pair's
        raster; an explicit/manual fit event follows its accepted overlay
        presentation.  ``None`` withdraws the answer when its request is
        removed.
        """

        return self._subscribe_callback(self._fit_callbacks, callback)

    def _notify_fit(self, event: FitEvent | None) -> None:
        deferred = getattr(self, "_configuration_fit_events", None)
        if deferred is not None:
            deferred.append(event)
            return
        with self._lock:
            callbacks = tuple(self._fit_callbacks)
        self._notify_callbacks(callbacks, event)


    def clear_fit(self) -> None:
        logical_completion: Future[FitResult | FacetFitBatchResult] | None = None
        with self._render_lock:
            with self._lock:
                self._assert_open()
                logical_completion = self._live_fit_completion
                previous = (
                    self._live_fit_request,
                    self._live_fit_completion,
                    self._fit_request_generation,
                    self._fit_context_generation,
                    self._accepted_fit,
                )
                fit_cancel = self._fit_cancel
                live_fit_cancel = self._live_fit_cancel
                self._live_fit_cancel = Event()
                self._live_fit_completion = None
                self._fit_request_generation += 1
                self._fit_warm_starts.clear()
                self._live_fit_request = None
                self._fit_context_generation += 1
                withdrawn = self._clear_fit_presentation()
                # Un-arming touches only fit state: an in-flight data-frame
                # preparation is fit-agnostic under the pair engine and
                # continues untouched.
            try:
                self._render_current(
                    RenderEffect.OVERLAY,
                    schedule_fit=False,
                )
            except Exception:
                with self._lock:
                    (
                        self._live_fit_request,
                        self._live_fit_completion,
                        self._fit_request_generation,
                        self._fit_context_generation,
                        self._accepted_fit,
                    ) = previous
                try:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
                except Exception:
                    self.redraw_surface()
                raise

        def retire_cleared_request() -> None:
            fit_cancel.set()
            live_fit_cancel.set()
            if logical_completion is not None and not logical_completion.done():
                logical_completion.set_exception(
                    FitCancelled("fit request cleared")
                )

        self._commit_fit_actions(retire_cleared_request)
        if withdrawn:
            # Nothing is painted any longer, so nothing is published either:
            # whoever derived signals from this fit hears it go.
            self._notify_fit(None)
