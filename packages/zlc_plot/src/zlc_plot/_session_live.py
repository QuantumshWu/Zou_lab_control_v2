"""Four-stage live frame protocol owned by one PlotSession facade."""

from __future__ import annotations

from concurrent.futures import Future
from numbers import Integral
from threading import Event
from typing import TYPE_CHECKING, Callable

from zlc_data import OwnedSnapshot

from .data_contract import schema_equal, snapshot_revision, snapshot_schema

from ._fit_projection import FitProjection
from ._session_state import (
    _AcceptedFit,
    _LiveFrameFinalization,
    _LiveFrameSnapshot,
    _PreparedLiveFrame,
    _ResolvedFit,
    FitEvent,
)
from .fit import FacetFitBatchResult, FitCancelled, FitResult
from .parameters import RenderEffect
from .primitives import PlotInput

if TYPE_CHECKING:
    from .session import PlotSession


class LiveSessionMixin:

    def prepare_live_frame(
        self,
        data: PlotInput,
        *,
        revision: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Future[_PreparedLiveFrame]:
        """Prepare one incoming data/fit frame without changing the visible state."""

        data, image_frame = self._split_image_frame(
            data,
            self._spec,
        )
        image_overlay = None if image_frame is None else image_frame.overlay
        FitProjection._validate_input(data, self._spec)
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        if isinstance(data, OwnedSnapshot):
            if revision is not None:
                if isinstance(revision, bool) or not isinstance(revision, Integral):
                    raise TypeError("revision must be an integer or None")
                if int(revision) != snapshot_revision(data):
                    raise ValueError(
                        "OwnedSnapshot revision must equal the supplied revision"
                    )
            selected_revision = snapshot_revision(data)
        else:
            if isinstance(revision, bool) or not isinstance(revision, Integral):
                raise TypeError(
                    "PulseTimeline live frames require an integer revision"
                )
            selected_revision = int(revision)
            if selected_revision < 0:
                raise ValueError("revision must be non-negative")

        with self._render_lock:
            with self._lock:
                self._assert_open()
                if isinstance(data, OwnedSnapshot):
                    assert isinstance(self._projection.data, OwnedSnapshot)
                    if not schema_equal(
                        snapshot_schema(self._projection.data), snapshot_schema(data)
                    ):
                        raise ValueError("data schema must remain exactly constant")
                if selected_revision <= self.data_revision:
                    raise ValueError(
                        "data revision must increase: "
                        f"{selected_revision} <= {self.data_revision}"
                    )
                active = self._clock_fit_future
                if active is not None and not active.done():
                    raise RuntimeError("a live frame is already being prepared")
                if image_overlay is not None:
                    self._validate_image_frame_overlay(
                        self._image_overlay,
                        image_overlay,
                    )
                cancellation = Event()
                self._clock_fit_cancel = cancellation
                snapshot = _LiveFrameSnapshot(
                    projection=self._projection._fork_frozen(
                        data=data,
                        revision=selected_revision,
                        context=self._projection_context(),
                    ),
                    base_data_revision=self.data_revision,
                    image_overlay=image_overlay,
                    image_overlay_authority=self._image_overlay,
                    fit_request=self._live_fit_request,
                    fit_context_generation=self._fit_context_generation,
                    fit_request_generation=self._fit_request_generation,
                )
            future = self._clock_fit_executor.submit(
                self._prepare_live_frame_worker,
                snapshot,
                cancellation,
                cancelled,
            )
            with self._lock:
                self._clock_fit_future = future

        def finished(completed: Future[_PreparedLiveFrame]) -> None:
            with self._lock:
                if self._clock_fit_future is completed:
                    self._clock_fit_future = None
            try:
                completed.result()
            except BaseException:
                self._forget_fit_warm_starts(snapshot.fit_request_generation)

        future.add_done_callback(finished)
        return future

    def _prepare_live_frame_worker(
        self,
        snapshot: _LiveFrameSnapshot,
        cancellation: Event,
        external_cancelled: Callable[[], bool] | None,
    ) -> _PreparedLiveFrame:
        def should_cancel() -> bool:
            return cancellation.is_set() or (
                external_cancelled is not None and bool(external_cancelled())
            )

        if should_cancel():
            raise FitCancelled("live frame preparation was cancelled")
        projection = snapshot.projection
        projection._build_view_and_payload()
        request = snapshot.fit_request
        fit: _ResolvedFit | None = None
        if request is not None:
            if request.all_facets:
                batch, selections = self._fit_facet_batch(
                    projection,
                    request.model,
                    initial=request.initial,
                    bounds=request.bounds,
                    options=request.options,
                    cancelled=should_cancel,
                    request_generation=snapshot.fit_request_generation,
                )
                fit = _ResolvedFit(
                    result=self._stamp_fit_batch_revision(batch),
                    selection=None,
                    overlay=None,
                    overlays=batch.overlays,
                    selections=selections,
                )
            else:
                selection = projection.fit_selection(
                    request.model,
                    selector_kind=request.selector_kind,
                )
                result = self._solve_fit_selection(
                    projection,
                    request.model,
                    selection,
                    initial=request.initial,
                    bounds=request.bounds,
                    options=request.options,
                    cancelled=should_cancel,
                    request_generation=snapshot.fit_request_generation,
                )
                result = self._stamp_fit_batch_revision(result)
                overlay = projection._make_fit_overlay(result, selection)
                fit = _ResolvedFit(result, selection, overlay)
        if should_cancel():
            raise FitCancelled("live frame preparation was cancelled")
        return _PreparedLiveFrame(
            session_identity=self._session_identity,
            base_data_revision=snapshot.base_data_revision,
            image_overlay=snapshot.image_overlay,
            image_overlay_authority=snapshot.image_overlay_authority,
            projection=projection,
            fit_request=request,
            fit=fit,
            fit_context_generation=snapshot.fit_context_generation,
            fit_request_generation=snapshot.fit_request_generation,
        )

    def commit_live_frame(
        self,
        prepared: _PreparedLiveFrame,
    ) -> _LiveFrameFinalization | None:
        """Atomically install and draw one fully prepared live frame."""

        if not isinstance(prepared, _PreparedLiveFrame):
            raise TypeError("prepared must be a prepared live frame")
        if prepared.session_identity is not self._session_identity:
            raise ValueError("prepared live frame belongs to another PlotSession")
        fit_event: FitEvent | None = None
        logical_completion: Future[FitResult | FacetFitBatchResult] | None = None
        with self._render_lock:
            with self._lock:
                self._assert_open()
                request_current = self._live_fit_request is prepared.fit_request
                request_generation_current = (
                    prepared.fit_request is None
                    or (
                        self._fit_request_generation
                        == prepared.fit_request_generation
                    )
                )
                prepared_state = prepared.projection.display_state
                current_state = self.display_state
                concurrent_parameter_changes = frozenset(
                    name
                    for name in self._parameter_schema.names
                    if prepared_state.values[name] != current_state.values[name]
                )
                projection_current = not bool(
                    self._parameter_schema.effects_for(
                        concurrent_parameter_changes
                    )
                    & (
                        RenderEffect.VIEW_PROJECTION
                        | RenderEffect.PAYLOAD_PROJECTION
                    )
                )
                data_base_current = (
                    self.data_revision == prepared.base_data_revision
                )
                image_overlay_current = (
                    prepared.image_overlay is None
                    or self._image_overlay is prepared.image_overlay_authority
                )
                authority_current = True
                if (
                    prepared.fit_request is not None
                    and prepared.fit is not None
                    and prepared.fit.selection is not None
                ):
                    try:
                        prepared_authority = prepared.fit.selection._authority
                        current_authority = self._projected._fit_selection_authority(
                            prepared.fit_request.selector_kind
                        )
                        # Selector and viewport edits are intentionally not a
                        # reason to reject an already prepared data frame: the
                        # fit is the result for that immutable data revision
                        # and its snapshot context.  A facet change, however,
                        # changes the target surface and must not paint the
                        # old facet's fit over the new one.
                        authority_current = (
                            prepared_authority is None
                            or current_authority.focused_facet_index
                            == prepared_authority.focused_facet_index
                        )
                    except (KeyError, TypeError, ValueError):
                        authority_current = False
                if (
                    not request_current
                    or not request_generation_current
                    or not projection_current
                    or not data_base_current
                    or not image_overlay_current
                    or not authority_current
                ):
                    return None
                revision = prepared.projection.data_revision
                if revision <= self.data_revision:
                    return None
                fit = prepared.fit
                accepted_image_overlay = (
                    self._image_overlay
                    if prepared.image_overlay is None
                    else prepared.image_overlay
                )
                fit_cancel = self._fit_cancel
            presentation = self._present_projection_transaction(
                prepared.projection,
                image_overlay=accepted_image_overlay,
                accepted_fit=(
                    None
                    if fit is None
                    else _AcceptedFit(
                        result=fit.result,
                        selection=fit.selection,
                        overlay=fit.overlay,
                        overlays=fit.overlays,
                        selections=fit.selections,
                        context_generation=prepared.fit_context_generation,
                    )
                ),
            )
            if fit is not None:
                self._remember_fit_warm_starts(
                    fit.result,
                    request_generation=prepared.fit_request_generation,
                )
            with self._lock:
                if fit is not None:
                    fit_event = self._fit_event(
                        self._accepted_fit
                    )
                    logical_completion = self._live_fit_completion
        return _LiveFrameFinalization(
            self._session_identity,
            fit_event,
            logical_completion,
            presentation,
            fit_cancel,
        )

    def finalize_live_frame(self, finalization: _LiveFrameFinalization) -> None:
        """Notify observers only after the frontend promoted the committed frame."""

        if not isinstance(finalization, _LiveFrameFinalization):
            raise TypeError("finalization must be a live-frame finalization token")
        if finalization.session_identity is not self._session_identity:
            raise ValueError("live-frame finalization belongs to another PlotSession")
        finalization.fit_cancel.set()
        event = finalization.fit_event
        completion = finalization.logical_completion
        if completion is not None:
            with self._lock:
                if self._live_fit_completion is completion:
                    self._live_fit_completion = None
        if completion is not None and not completion.done():
            if event is None:
                completion.set_exception(
                    FitCancelled("live fit frame reached presentation without a result")
                )
            else:
                completion.set_result(event.result)
        if event is not None:
            self._notify_fit(event)

    def abort_live_frame(self, finalization: _LiveFrameFinalization) -> None:
        """Roll back a drawn live frame that the frontend could not promote."""

        if not isinstance(finalization, _LiveFrameFinalization):
            raise TypeError("finalization must be a live-frame finalization token")
        if finalization.session_identity is not self._session_identity:
            raise ValueError("live-frame finalization belongs to another PlotSession")
        self._abort_projection_presentation(finalization.presentation)
