"""Small state records shared by the fit and live session coordinators."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING, Callable, Mapping

from zlc_data import SelectionChange

from ._fit_projection import FitProjection, FitSelection, RollingHistoryPoint
from ._fit_scene import FitOverlay
from ._selector_scene import ColorLimitCandidate
from .selectors import RectangleRange, SelectorKind, SelectorState
from .primitives import ImagePointOverlay


if TYPE_CHECKING:
    from .fit import (
        FacetFitBatchResult,
        FitModelSpec,
        FitOptions,
        FitParameterDisplay,
        FitResult,
    )
    from .layout import SurfacePlan


@dataclass(frozen=True, slots=True)
class FitEvent:
    """A fit result accepted and painted for the current data revision."""

    result: "FitResult | FacetFitBatchResult"
    selection: FitSelection | None
    display_parameters: tuple["FitParameterDisplay", ...]
    formula: str
    overlays: tuple[FitOverlay, ...] = ()


@dataclass(frozen=True, slots=True)
class _LiveFitRequest:
    model: "FitModelSpec"
    selector_kind: SelectorKind | None
    initial: Mapping[str, float] | tuple[float, ...] | None
    bounds: Mapping[str, tuple[float | None, float | None]] | None
    options: "FitOptions" | None
    all_facets: bool = False


@dataclass(frozen=True, slots=True)
class _PointerUpdate:
    """One backend-neutral gesture reduction for raster frontends."""

    candidate: SelectorState | ColorLimitCandidate | None
    role: str | None
    cell_index: int | None
    active_pan: bool
    publish_front: bool


@dataclass(frozen=True, slots=True)
class _StartedFitRequest:
    request: _LiveFitRequest
    selection: FitSelection | None
    projection: FitProjection
    cancellation: Event
    context_generation: int
    request_generation: int


@dataclass(frozen=True, slots=True)
class _LiveFrameSnapshot:
    projection: FitProjection
    base_data_revision: int
    image_overlay: ImagePointOverlay | None
    image_overlay_authority: ImagePointOverlay | None


@dataclass(frozen=True, slots=True)
class _ResolvedFit:
    """One complete solver result, selection authority and painted overlay."""

    result: "FitResult | FacetFitBatchResult"
    selection: FitSelection | None
    overlay: FitOverlay | None
    overlays: tuple[FitOverlay, ...] = ()
    selections: tuple[FitSelection | None, ...] = ()

    def __post_init__(self) -> None:
        overlays = tuple(self.overlays)
        if not overlays and self.overlay is not None:
            overlays = (self.overlay,)
        overlay = self.overlay
        if overlay is None and len(overlays) == 1:
            overlay = overlays[0]
        selections = tuple(self.selections)
        if not selections and self.selection is not None:
            selections = (self.selection,)
        object.__setattr__(self, "overlay", overlay)
        object.__setattr__(self, "overlays", overlays)
        object.__setattr__(self, "selections", selections)

@dataclass(frozen=True, slots=True)
class _PreparedLiveFrame:
    session_identity: object
    base_data_revision: int
    image_overlay: ImagePointOverlay | None
    image_overlay_authority: ImagePointOverlay | None
    projection: FitProjection

@dataclass(frozen=True, slots=True)
class _SolvedLiveFit:
    """One pair-solved live fit travelling from the fit executor to its commit.

    ``result`` is None when the solve was cancelled, hit a caller-authored
    deadline, or failed; the paired data frame then commits without an
    overlay and the next data revision is the only automatic retry.
    """

    started: _StartedFitRequest
    result: "FitResult | FacetFitBatchResult | None"


@dataclass(frozen=True, slots=True)
class _LiveFrameFinalization:
    session_identity: object
    presentation: "_ProjectionPresentation"
    #: The pair's accepted-fit completion, resolved only after the committed
    #: front is promoted (``publish_live_frame``) so a waiting fit future
    #: never observes a front older than its result.
    fit_resolution: "_FitResolution | None" = None


@dataclass(frozen=True, slots=True)
class _FitResolution:
    completion: Future["FitResult | FacetFitBatchResult"]
    result: "FitResult | FacetFitBatchResult" | None = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class _AcceptedFit(_ResolvedFit):
    """One atomically accepted fit result and its painted presentation."""

    context_generation: int = 0

@dataclass(frozen=True, slots=True)
class _ProjectionPresentation:
    """A drawn projection plus the exact state needed to abort publication."""

    committed_projection: FitProjection
    previous_projection: FitProjection
    previous_image_overlay: ImagePointOverlay | None
    previous_accepted_fit: _AcceptedFit | None
    previous_classifier_results: tuple["FitResult | None", ...]
    previous_classifier_overlays: tuple[FitOverlay, ...]
    previous_classifier_thresholds: tuple[float | None, ...]
    previous_focused_facet_index: int | None
    previous_facet_focus_index: int | None
    previous_viewport: RectangleRange | None
    previous_rolling_history: tuple[RollingHistoryPoint, ...]
    previous_layout_revision: int
    previous_plan: "SurfacePlan"


@dataclass(frozen=True, slots=True)
class _FitPresentation:
    event: object
    accepted: _AcceptedFit
    previous: _AcceptedFit | None


__all__ = [
    "_AcceptedFit", "_FitPresentation", "_FitResolution", "_LiveFitRequest",
    "_LiveFrameFinalization", "_LiveFrameSnapshot", "_PointerUpdate",
    "_PreparedLiveFrame", "_ProjectionPresentation", "_ResolvedFit",
    "_SolvedLiveFit", "_StartedFitRequest", "FitEvent", "SelectionChange",
]
