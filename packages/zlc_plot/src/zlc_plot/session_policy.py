"""Pure state policy for replacing an authored plot specification.

The render session owns transactions; this module only decides which initial
state is eligible to cross the semantic rebuild boundary. Keeping that
decision independent makes Qt, notebook and headless callers use exactly the
same retention rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ._kinds import handler_for
from .parameters import ParameterSchema
from .selectors import RectangleRange
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotLabels,
    PlotSpec,
    PulseTimelinePlot,
    RollingPlot,
    semantic_spec,
)

_PLOT_SPEC_TYPES = (
    CurvePlot,
    ImagePlot,
    HistogramPlot,
    RollingPlot,
    FacetGridPlot,
    PulseTimelinePlot,
)


@dataclass(frozen=True, slots=True)
class ReplaceSpecInitialState:
    """Initial display/session values selected for one semantic replacement."""

    parameters: Mapping[str, object]
    size: str
    viewport: RectangleRange | None

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Mapping):
            raise TypeError("replacement parameters must be a mapping")
        if not isinstance(self.size, str) or not self.size:
            raise ValueError("replacement size must be non-empty text")
        if self.viewport is not None and not isinstance(self.viewport, RectangleRange):
            raise TypeError("replacement viewport must be RectangleRange or None")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def _viewport_signature(spec: PlotSpec) -> tuple[object, ...]:
    """Return only coordinate roles that define viewport units/geometry."""

    semantic = semantic_spec(spec)
    result: list[object] = [spec.kind]
    for name in ("x", "y"):
        result.append(getattr(semantic, name, None))
    if isinstance(spec, FacetGridPlot):
        result.append(spec.facet)
    return tuple(result)


def _viewport_is_valid(old_spec: PlotSpec, new_spec: PlotSpec) -> bool:
    """A viewport is display-space state and survives only an equivalent view."""

    return _viewport_signature(old_spec) == _viewport_signature(new_spec)


def _retain_parameters(
    old_parameters: Mapping[str, object],
    schema: ParameterSchema,
) -> dict[str, object]:
    """Retain values one at a time, dropping values invalid for the new schema."""

    if not isinstance(old_parameters, Mapping):
        raise TypeError("old parameters must be a mapping")
    current = schema.initial_values()
    retained: dict[str, object] = {}
    for name in schema.names:
        if name not in old_parameters:
            continue
        try:
            candidate = schema.prepare_transition(current, {name: old_parameters[name]})
        except (TypeError, ValueError, KeyError):
            continue
        current = candidate
        retained[name] = candidate[name]
    return retained


def merge_labels(old_spec: PlotSpec, new_spec: PlotSpec) -> PlotLabels:
    """Carry authored labels across a semantic edit by role, never by slot.

    Each kind declares which semantic role every label slot describes
    (registry ``label_roles``).  A label survives only when the new spec has
    a slot with the identical role, and it moves to that slot: a curve's
    y-axis value label becomes a histogram's x label, while the curve's
    x-axis label is dropped because no histogram slot describes that axis.
    Verbatim slot copying is exactly the audit bug this replaces — it left
    a histogram x axis reading "Time (mV)".
    """

    old_labels = old_spec.labels
    carriers = {
        role: getattr(old_labels, slot)
        for slot, role in handler_for(old_spec).label_roles(old_spec)
        if getattr(old_labels, slot)
    }
    slots = {
        slot: carriers[role]
        for slot, role in handler_for(new_spec).label_roles(new_spec)
        if role in carriers
    }
    return PlotLabels(**slots)


def replace_spec_initial_state(
    old_spec: PlotSpec,
    new_spec: PlotSpec,
    old_parameters: Mapping[str, object],
    new_schema: ParameterSchema,
    *,
    size: str,
    viewport: RectangleRange | None = None,
    parameters: Mapping[str, object] | None = None,
) -> ReplaceSpecInitialState:
    """Compute the initial state for a semantic rebuild without side effects.

    Display parameters are independently revalidated against ``new_schema``;
    units and fixed-size choices therefore survive whenever the new kind has
    the same field. A viewport survives when the plot kind and coordinate roles
    remain compatible. Grouping and reduction do not change display-axis units,
    but a changed facet source invalidates it.
    Selector and fit state are intentionally absent: the caller must create
    fresh empty interaction/fit state for the new projection.
    """

    if not isinstance(old_spec, _PLOT_SPEC_TYPES) or not isinstance(
        new_spec, _PLOT_SPEC_TYPES
    ):
        raise TypeError("old_spec and new_spec must be PlotSpec values")
    if not isinstance(new_schema, ParameterSchema):
        raise TypeError("new_schema must be ParameterSchema")
    if parameters is not None and not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping or None")
    retained = _retain_parameters(old_parameters, new_schema)
    if parameters:
        retained.update(parameters)
    selected_viewport = viewport if _viewport_is_valid(old_spec, new_spec) else None
    return ReplaceSpecInitialState(retained, size, selected_viewport)


__all__ = ["ReplaceSpecInitialState", "merge_labels", "replace_spec_initial_state"]
