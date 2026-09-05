"""Four-stage live frame protocol owned by one PlotSession facade."""

from __future__ import annotations

from concurrent.futures import Future
from numbers import Integral
from threading import Event
from typing import TYPE_CHECKING, Callable

from zlc_data import OwnedSnapshot
from zlc_data.snapshot_projection import indexed_schemas_compatible

from .data_contract import (
    schema_equal,
    snapshot_generation,
    snapshot_revision,
    snapshot_schema,
)

from ._fit_projection import FitProjection
from ._session_state import (
    _LiveFrameFinalization,
    _LiveFrameSnapshot,
    _PreparedLiveFrame,
    _SolvedLiveFit,
)
from .fit import FitCancelled
from .parameters import RenderEffect
from .primitives import PlotInput

if TYPE_CHECKING:
    from .session import PlotSession


class LiveSessionMixin:

    def holds_live_revision(self, data: PlotInput, revision: int) -> bool:
        """Whether this session already holds that data, so there is nothing to do.

        The one place the question is answered.  ``prepare_live_frame``
        REFUSES on it, which is right for a caller asking to advance and
        wrong for a pipeline that can legitimately be handed the same
        revision twice -- so the pipeline asks the question instead of
        catching the refusal, and the two cannot drift apart.
        """

        snapshot = getattr(data, "snapshot", data)
        generation_changed = (
            isinstance(snapshot, OwnedSnapshot)
            and str(snapshot_generation(snapshot)) != str(self.data_generation)
        )
        return (not generation_changed) and int(revision) <= int(
            self.data_revision
        )

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
            if revision is None:
                # Direct pulse-timeline updates advance the session revision
                # by exactly one, matching update_data's contract.
                selected_revision = self.data_revision + 1
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
                    previous_schema = snapshot_schema(self._projection.data)
                    next_schema = snapshot_schema(data)
                    if not (
                        schema_equal(previous_schema, next_schema)
                        or indexed_schemas_compatible(previous_schema, next_schema)
                    ):
                        raise ValueError("data schema must remain exactly constant")
                if self.holds_live_revision(data, selected_revision):
                    # A new RUN restarts its revisions; within one run they
                    # are strictly monotonic.  This pipeline only ever saw
                    # one run per host while generations replaced hosts, so
                    # the guard predates cross-run frames.
                    raise ValueError(
                        "data revision must increase: "
                        f"{selected_revision} <= {self.data_revision}"
                    )
                active = self._live_prepare_future
                if active is not None and not active.done():
                    raise RuntimeError("a live frame is already being prepared")
                if image_overlay is not None and not self._data_generation_changed(
                    self._projection.data, data
                ):
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
            future = self._analysis_executor.submit(
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
        solved: _SolvedLiveFit | None = None,
    ) -> _LiveFrameFinalization | None:
        """Atomically install and draw one complete data/fit pair.

        The frame of a fit-armed panel is data@N together with fit@N: the
        caller solves the pair on the serial analysis executor (``solve_live_frame``)
        before committing, and this commit paints the data and its overlay
        under one render-lock hold — the captured front is born complete.
        An armed pair that did not solve does not advance visible data.
        Un-armed frames commit data-only exactly as before.
        """

        if not isinstance(prepared, _PreparedLiveFrame):
            raise TypeError("prepared must be a prepared live frame")
        if prepared.session_identity is not self._session_identity:
            raise ValueError("prepared live frame belongs to another PlotSession")
        if solved is not None and not isinstance(solved, _SolvedLiveFit):
            raise TypeError("solved must be a solved live fit or None")
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
                # A semantic edit replaces the SPEC, and no parameter-schema
                # check can see that: a fate lands through replace_spec, not
                # through the display bag.  An old-spec payload committed
                # beside the new spec leaves the session holding a pair that
                # was never one accepted view, and the first consumer to ask
                # them a question -- a selector, wanting its subject --
                # refuses to answer.  The producer's next revision projects
                # through the current spec and the panel heals.
                spec_current = prepared.projection.spec == self._spec
                image_overlay_current = (
                    prepared.image_overlay is None
                    or self._image_overlay is prepared.image_overlay_authority
                )
                if (
                    not projection_current
                    or not spec_current
                    or not data_base_current
                    or not image_overlay_current
                ):
                    return None
                revision = prepared.projection.data_revision
                generation_changed = str(
                    prepared.projection.data_generation
                ) != str(self.data_generation)
                if not generation_changed and revision <= self.data_revision:
                    return None
            presentation, resolution = self._present_solved_projection(
                prepared.projection,
                image_overlay=prepared.image_overlay,
                solved=solved,
            )
        return _LiveFrameFinalization(
            self._session_identity,
            presentation,
            resolution,
        )

    def publish_live_frame(self, finalization: _LiveFrameFinalization) -> None:
        """Acknowledge that the committed front reached the frontend."""

        if not isinstance(finalization, _LiveFrameFinalization):
            raise TypeError("finalization must be a live-frame finalization token")
        if finalization.session_identity is not self._session_identity:
            raise ValueError("live-frame finalization belongs to another PlotSession")
        self._emit_projection_focus_change(finalization.presentation)
        if finalization.fit_resolution is not None:
            self._resolve_fit_completion(finalization.fit_resolution)

    def abort_live_frame(self, finalization: _LiveFrameFinalization) -> None:
        """Roll back a drawn live frame that the frontend could not promote."""

        if not isinstance(finalization, _LiveFrameFinalization):
            raise TypeError("finalization must be a live-frame finalization token")
        if finalization.session_identity is not self._session_identity:
            raise ValueError("live-frame finalization belongs to another PlotSession")
        self._abort_projection_presentation(finalization.presentation)
        resolution = finalization.fit_resolution
        if resolution is not None:
            # The pair's accept was rolled back with the frame; give the
            # logical completion back to the request so a later pair can
            # still resolve it.
            self._restore_live_fit_completion(resolution)
