"""Fit request orchestration owned by one PlotSession facade."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from threading import Event
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from ._fit_projection import FitProjection, FitSelection
from ._gesture_engine import _SelectorGesture
from ._session_state import (
    _AcceptedFit,
    _FitPresentation,
    _FitResolution,
    _LiveFitRequest,
    _StartedFitRequest,
    FitEvent,
)
from .fit import FacetFitBatchResult, FitCancelled, FitModelSpec, FitOptions, FitResult
from .parameters import RenderEffect
from .selectors import SelectorKind
from .specs import FacetGridPlot

if TYPE_CHECKING:
    from .session import PlotSession

FitCallback = Callable[[FitEvent], object]


class FitSessionMixin:

    def _stamp_fit_batch_revision(
        self,
        result: FitResult | FacetFitBatchResult,
    ) -> FitResult | FacetFitBatchResult:
        """Assign one monotonic publication revision to a completed fit batch."""

        with self._lock:
            self._fit_batch_revision += 1
            batch_revision = self._fit_batch_revision
        return replace(result, batch_revision=batch_revision)

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
        return warm

    def _remember_fit_warm_starts(
        self,
        result: FitResult | FacetFitBatchResult,
        *,
        request_generation: int,
    ) -> None:
        """Publish only accepted results as seeds for the next revision."""

        with self._lock:
            if request_generation != self._fit_request_generation:
                return
            if isinstance(result, FacetFitBatchResult):
                for index, fit in enumerate(result.results):
                    key = (request_generation, result.model.model_id, index)
                    if fit is None or not fit.success:
                        self._fit_warm_starts.pop(key, None)
                    else:
                        self._fit_warm_starts[key] = tuple(
                            float(value) for value in fit.parameter_values
                        )
                return
            key = (request_generation, result.model.model_id, None)
            if result.success:
                self._fit_warm_starts[key] = tuple(
                    float(value) for value in result.parameter_values
                )
            else:
                self._fit_warm_starts.pop(key, None)

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
        if fit_all_facets:
            if live:
                raise ValueError("fit_all_facets cannot be a live fit")
            return self._fit_all_facets(
                model,
                initial=initial,
                bounds=bounds,
                options=options,
                cancelled=cancelled,
            )
        started = self._begin_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
            live=live,
            logical_completion=None,
        )
        if started is None:
            raise RuntimeError("a synchronous fit request has no selection")
        try:
            result = self._solve_started_fit(started, cancelled=cancelled)
        except BaseException:
            self._forget_fit_warm_starts(started.request_generation)
            raise
        result_revision = (
            result.source_revision
            if isinstance(result, FacetFitBatchResult)
            else result.source_revision
        )
        if not started.cancellation.is_set() and self.data_revision == result_revision:
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
        if fit_all_facets:
            if live:
                raise ValueError("fit_all_facets cannot be a live fit")
            completion: Future[FitResult | FacetFitBatchResult] = Future()

            def solve_batch() -> None:
                try:
                    completion.set_result(
                        self._fit_all_facets(
                            model,
                            initial=initial,
                            bounds=bounds,
                            options=options,
                            cancelled=None,
                        )
                    )
                except Exception as error:
                    completion.set_exception(error)

            self._fit_executor.submit(solve_batch)
            return completion
        logical_completion: Future[FitResult | FacetFitBatchResult] = Future()
        started = self._begin_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
            live=live,
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

    def _fit_all_facets(
        self,
        model: str | FitModelSpec,
        *,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
        cancelled: Callable[[], bool] | None,
    ) -> FacetFitBatchResult:
        if not isinstance(self._spec, FacetGridPlot):
            raise TypeError("fit_all_facets requires FacetGridPlot")
        resolved = self._resolve_fit_model(model)
        with self._render_lock:
            with self._lock:
                self._assert_open()
                projection = self._projected
        batch, _selections = self._fit_facet_batch(
            projection,
            resolved,
            initial=initial,
            bounds=bounds,
            options=options,
            cancelled=cancelled,
        )
        return self._stamp_fit_batch_revision(batch)

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
    ) -> tuple[FacetFitBatchResult, tuple[FitSelection | None, ...]]:
        """Fit every projected cell and construct overlays through one path."""

        if not isinstance(projection._spec, FacetGridPlot):
            raise TypeError("facet batch projection requires FacetGridPlot")
        payload = projection.payload
        cells = tuple(getattr(payload, "cells", ()))
        if not cells:
            raise ValueError("facet grid has no cells to fit")
        results: list[FitResult | None] = []
        failure_messages: list[str | None] = []
        overlays: list[object] = []
        selections: list[FitSelection | None] = []
        for index, cell in enumerate(cells):
            if cancelled is not None and bool(cancelled()):
                raise FitCancelled("facet fit cancelled")
            cell_projection = projection._with_context(
                replace(
                    projection._context,
                    focused_facet_index=index,
                )
            )
            try:
                selection = cell_projection.fit_selection(model)
                result = self._solve_fit_selection(
                    cell_projection,
                    model,
                    selection,
                    initial=initial,
                    bounds=bounds,
                    options=options,
                    cancelled=cancelled,
                    request_generation=request_generation,
                )
                overlay = cell_projection._make_fit_overlay(result, selection)
            except FitCancelled:
                raise
            except Exception as error:
                results.append(None)
                message = str(error) or type(error).__name__
                failure_messages.append(message)
                selections.append(None)
                from ._fit_scene import FitOverlay

                overlays.append(
                    FitOverlay(
                        success=False,
                        diagnostic=message,
                        facet_index=index,
                    )
                )
            else:
                results.append(result)
                failure_messages.append(None)
                selections.append(selection)
                overlays.append(overlay)
        facet_coordinate = projection._coordinate(projection._spec.facet)
        facet_values = tuple(cell.facet_value_canonical for cell in cells)
        numeric_facet = all(
            isinstance(value, (int, float, np.number))
            and not isinstance(value, (bool, np.bool_))
            for value in facet_values
        )
        parameter_units = next(
            (
                result.parameter_units
                for result in results
                if result is not None and result.parameter_units
            ),
            projection._fit_parameter_units(model),
        )
        batch = FacetFitBatchResult(
            facet=projection._spec.facet,
            facet_values=facet_values,
            model=model,
            results=tuple(results),
            failure_messages=tuple(failure_messages),
            source_revision=projection.data_revision,
            overlays=tuple(overlays),
            parameter_units=parameter_units,
            sample_axis_name=facet_coordinate.label,
            sample_coordinates=(
                np.asarray(facet_values, dtype=np.float64)
                if numeric_facet
                else np.arange(len(facet_values), dtype=np.float64)
            ),
            sample_unit=(
                ""
                if not numeric_facet or facet_coordinate.canonical_unit.symbol == "1"
                else facet_coordinate.canonical_unit.symbol
            ),
            sample_labels=(
                None if numeric_facet else tuple(cell.label for cell in cells)
            ),
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
        logical_completion: Future[FitResult | FacetFitBatchResult] | None,
    ) -> _StartedFitRequest | None:
        """Atomically replace fit request authority and freeze one solver input."""

        if not isinstance(live, bool):
            raise TypeError("live must be bool")
        request = self._prepare_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
        )
        request = replace(
            request,
            all_facets=live and isinstance(self._spec, FacetGridPlot),
        )
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
                previous_clock_fit_cancel = self._clock_fit_cancel
                self._fit_context_generation += 1
                self._fit_request_generation += 1
                self._fit_warm_starts.clear()
                superseded = self._live_fit_completion
                self._live_fit_completion = logical_completion if live else None
                cancellation = Event()
                self._fit_cancel = cancellation
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
            previous_fit_cancel.set()
            previous_clock_fit_cancel.set()
        if superseded is not None and not superseded.done():
            superseded.set_exception(FitCancelled("fit request superseded"))
        return started

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
            )
            return self._stamp_fit_batch_revision(batch)
        if started.selection is None:
            raise RuntimeError("a non-facet fit request has no frozen selection")
        result = self._solve_fit_selection(
            started.projection,
            started.request.model,
            started.selection,
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
            future = self._fit_executor.submit(self._solve_started_fit, started)
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
    ) -> FitResult:
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
        if result.model != model:
            raise RuntimeError("FitEngine returned a result for another model")
        if result.source_revision != selection.data_revision:
            raise RuntimeError("FitEngine returned a result for another data revision")
        return result.with_parameter_units(
            projection._fit_parameter_units(model)
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

    def _accept_fit(
        self,
        result: FitResult | FacetFitBatchResult,
        started: _StartedFitRequest,
    ) -> _FitPresentation | None:
        batch = result if isinstance(result, FacetFitBatchResult) else None
        selection = None if batch is not None else started.selection
        result_revision = (
            batch.source_revision
            if batch is not None
            else result.source_revision
        )
        with self._render_lock:
            with self._lock:
                if (
                    self._closed
                    or result_revision != self.data_revision
                    or started.request_generation != self._fit_request_generation
                ):
                    return None
                if selection is not None and selection.data_revision != self.data_revision:
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
                if batch is None:
                    assert selection is not None
                    overlay = self._projected._make_fit_overlay(result, selection)
                    overlays = (overlay,)
                    selections = (selection,)
                else:
                    overlay = None
                    overlays = batch.overlays
                    selections = self._facet_fit_selections(
                        started.projection,
                        batch.model,
                        selector_kind=started.request.selector_kind,
                    )
                previous = self._accepted_fit
                accepted = _AcceptedFit(
                    result=result,
                    selection=selection,
                    overlay=overlay,
                    overlays=overlays,
                    selections=selections,
                    context_generation=started.context_generation,
                )
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
        )
        return _FitPresentation(
            self._fit_event(accepted),
            accepted,
            previous,
        )

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
            cell_projection = projection._with_context(
                replace(projection._context, focused_facet_index=index)
            )
            try:
                selections.append(
                    cell_projection.fit_selection(
                        model,
                        selector_kind=selector_kind,
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
        """Observe results only after they are accepted and painted."""

        return self._subscribe_callback(self._fit_callbacks, callback)

    def _notify_fit(self, event: FitEvent) -> None:
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
                clock_fit_cancel = self._clock_fit_cancel
                self._live_fit_completion = None
                self._fit_request_generation += 1
                self._fit_warm_starts.clear()
                self._live_fit_request = None
                self._fit_context_generation += 1
                self._accepted_fit = None
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
            fit_cancel.set()
            clock_fit_cancel.set()
        if logical_completion is not None and not logical_completion.done():
            logical_completion.set_exception(FitCancelled("fit request cleared"))
