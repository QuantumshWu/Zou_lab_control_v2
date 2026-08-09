"""Workbench-owned state shared by a panel's Setting and Edit views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from zlc_plot import PlotKind, describe_semantics, updated_spec


__all__ = ["PanelFrozenData", "PanelState", "compose_panel_spec"]


def _plain_state(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Own one shallow plain mapping without inventing a second state model."""

    return MappingProxyType(dict(values))


def _document_value(value: Any) -> Any:
    """Project typed plot choices into the plain values a layout can store."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _document_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_document_value(item) for item in value]
    return str(value)


def restore_semantic_choice(description: object, name: str, saved: object) -> object:
    """Resolve one layout value through the plot-owned semantic choices."""

    field_for = getattr(description, "field", None)
    if not callable(field_for):
        return saved
    field = field_for(str(name))
    for candidate in tuple(getattr(field, "choice_values", ())):
        if candidate == saved or _document_value(candidate) == saved:
            return candidate
    return saved


def compose_panel_spec(schema: object, spec: object, state: "PanelState") -> object:
    """Compose saved semantics before a host performs its first render."""

    if not isinstance(state, PanelState):
        raise TypeError("state must be PanelState")
    candidate = spec
    for name, saved in state.semantic.items():
        description = describe_semantics(schema, candidate)
        value = restore_semantic_choice(description, str(name), saved)
        candidate = updated_spec(schema, candidate, str(name), value)
    return candidate


@dataclass(frozen=True, slots=True)
class PanelState:
    """The single replace-only configuration record for one plot panel."""

    signal: str
    kind: str
    size: str
    interval_ms: int
    title: str
    cell_kind: str = ""
    semantic: Mapping[str, Any] = field(default_factory=dict)
    display: Mapping[str, Any] = field(default_factory=dict)
    fit: Mapping[str, Any] = field(default_factory=dict)
    overlay_signal: str = ""

    def __post_init__(self) -> None:
        interval = int(self.interval_ms)
        if interval <= 0:
            raise ValueError("a display interval must be positive")
        kind = str(self.kind)
        cell_kind = str(self.cell_kind)
        resolved_kind = PlotKind(kind)
        if resolved_kind is PlotKind.FACET_GRID:
            if cell_kind not in {
                PlotKind.CURVE.value,
                PlotKind.IMAGE.value,
                PlotKind.HISTOGRAM.value,
            }:
                raise ValueError(
                    "FacetGrid cell kind must be curve, image, or histogram"
                )
        elif cell_kind:
            raise ValueError("only a FacetGrid panel has a cell kind")
        object.__setattr__(self, "signal", str(self.signal))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "cell_kind", cell_kind)
        object.__setattr__(self, "size", str(self.size))
        object.__setattr__(self, "interval_ms", interval)
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "semantic", _plain_state(self.semantic))
        object.__setattr__(self, "display", _plain_state(self.display))
        object.__setattr__(self, "fit", _plain_state(self.fit))
        overlay_signal = str(self.overlay_signal).strip()
        if overlay_signal and resolved_kind is not PlotKind.IMAGE:
            raise ValueError("only an Image panel can select an overlay signal")
        object.__setattr__(self, "overlay_signal", overlay_signal)

    def document(self) -> dict[str, Any]:
        """Return the JSON-shaped part of a reusable layout document."""

        return {
            "signal": self.signal,
            "title": self.title,
            "kind": self.kind,
            "cell_kind": self.cell_kind,
            "size": self.size,
            "interval_ms": self.interval_ms,
            "semantic": _document_value(self.semantic),
            "display": _document_value(self.display),
            "fit": _document_value(self.fit),
            "overlay_signal": self.overlay_signal,
        }


@dataclass(frozen=True, slots=True)
class PanelFrozenData:
    """The exact data revision shown in Edit, independent of ``PanelState``."""

    signal: str
    publication: object | None
    snapshot: object
    plot_input: object | None = None
    run_chain: tuple[Mapping[str, Any], ...] = ()
    overlay: Mapping[str, Any] = field(default_factory=dict)
