"""Small registry records; kind-specific behavior lives beside its spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..kinds import PlotKind
# render(renderer, payload, state, *, axes, key, **pooled) paints ONE surface.
# The surface is supplied, never assumed: the same call draws a standalone
# plot and one FacetGrid cell, which is why a cell can no longer silently
# miss a facility the standalone plot has.  ``pooled`` carries the facts the
# GRID resolved for every cell at once (a shared colour scale, one x span,
# one histogram binning) and is empty for a standalone plot.
RenderHandler = Callable[..., None]
PayloadHandler = Callable[[Any, Any, Any], None]
AdmitsHandler = Callable[[Any], bool]
DefaultSpecHandler = Callable[[Any], Any]
# label_roles(spec) maps each PlotLabels slot this kind uses to the semantic
# role the slot describes, e.g. ("x", ("axis", spec.x)) or ("y", ("value",)).
# Label carry-over across semantic edits matches these roles, never slots.
LabelRolesHandler = Callable[[Any], tuple[tuple[str, tuple], ...]]
# validate(view, spec) raises when the projection would reject the spec on
# this dataset.  It runs no aggregation: the semantic feasibility probe calls
# it instead of building payloads, and every DataView projection method calls
# the same underlying checks, so probe and build can never drift.
ValidateHandler = Callable[[Any, Any], None]


@dataclass(frozen=True, slots=True)
class KindHandler:
    kind: PlotKind
    display_name: str
    spec_type: type
    fit_target: str | None
    render: RenderHandler
    build_payload: PayloadHandler
    semantic_fields: tuple[str, ...]
    admits: AdmitsHandler
    default_spec: DefaultSpecHandler
    label_roles: LabelRolesHandler
    validate: ValidateHandler

    def __post_init__(self) -> None:
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("kind display_name must be non-empty text")
        if not callable(self.default_spec):
            raise TypeError("kind default_spec must be callable")
        if not callable(self.label_roles):
            raise TypeError("kind label_roles must be callable")
        if not callable(self.validate):
            raise TypeError("kind validate must be callable")
