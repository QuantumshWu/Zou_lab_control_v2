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
    _LiveFrameFinalization,
    _LiveFrameSnapshot,
    _PreparedLiveFrame,
)
from .fit import FitCancelled
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
        """Prepare one incoming data frame without changing the visible state."""

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
                active = self._live_prepare_future
                if active is not None and not active.done():
                    raise RuntimeError("a live frame is already being prepared")
                if image_overlay is not None:
                    self._validate_image_frame_overlay(
                        self._image_overlay,
                        image_overlay,
                    )
                cancellation = Event()
                self._live_prepare_cancel = cancellation
                snapshot = _LiveFrameSnapshot(
                    projection=self._projection._fork_frozen(
                        data=data,
                        revision=selected_revision,
                        context=self._projection_context(),
                    ),
                    base_data_revision=self.data_revision,
                    image_overlay=image_overlay,
                    image_overlay_authority=self._image_overlay,
                )
            future = self._live_prepare_executor.submit(
                self._prepare_live_frame_worker,
                snapshot,
                cancellation,
                cancelled,
            )
            with self._lock:
                self._live_prepare_future = future

        def finished(completed: Future[_PreparedLiveFrame]) -> None:
            with self._lock:
                if self._live_prepare_future is completed:
                    self._live_prepare_future = None

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
        if should_cancel():
            raise FitCancelled("live frame preparation was cancelled")
        return _PreparedLiveFrame(
            session_identity=self._session_identity,
            base_data_revision=snapshot.base_data_revision,
            image_overlay=snapshot.image_overlay,
            image_overlay_authority=snapshot.image_overlay_authority,
            projection=projection,
        )

    def commit_live_frame(
        self,
        prepared: _PreparedLiveFrame,
    ) -> _LiveFrameFinalization | None:
        """Atomically install and draw one prepared data frame."""

        if not isinstance(prepared, _PreparedLiveFrame):
            raise TypeError("prepared must be a prepared live frame")
        if prepared.session_identity is not self._session_identity:
            raise ValueError("prepared live frame belongs to another PlotSession")
        with self._render_lock:
            with self._lock:
                self._assert_open()
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
                if (
                    not projection_current
                    or not data_base_current
                    or not image_overlay_current
                ):
                    return None
                revision = prepared.projection.data_revision
                if revision <= self.data_revision:
                    return None
                accepted_image_overlay = (
                    self._image_overlay
                    if prepared.image_overlay is None
                    else prepared.image_overlay
                )
                accepted_fit = (
                    self._accepted_fit
                    if self._live_fit_request is not None
                    else None
                )
            presentation = self._present_projection_transaction(
                prepared.projection,
                image_overlay=accepted_image_overlay,
                accepted_fit=accepted_fit,
            )
        return _LiveFrameFinalization(
            self._session_identity,
            presentation,
        )

    def finalize_live_frame(self, finalization: _LiveFrameFinalization) -> None:
        """Start the current fit only after the data front was promoted."""

        if not isinstance(finalization, _LiveFrameFinalization):
            raise TypeError("finalization must be a live-frame finalization token")
        if finalization.session_identity is not self._session_identity:
            raise ValueError("live-frame finalization belongs to another PlotSession")
        self._restart_live_fit_for_current_data()

    def abort_live_frame(self, finalization: _LiveFrameFinalization) -> None:
        """Roll back a drawn live frame that the frontend could not promote."""

        if not isinstance(finalization, _LiveFrameFinalization):
            raise TypeError("finalization must be a live-frame finalization token")
        if finalization.session_identity is not self._session_identity:
            raise ValueError("live-frame finalization belongs to another PlotSession")
        self._abort_projection_presentation(finalization.presentation)
