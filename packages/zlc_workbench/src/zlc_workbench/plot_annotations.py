"""Portable, plot-kind-neutral annotations owned by a prepared panel.

Panel state describes authored controls.  These values are producer-authored
annotations over that state: they must survive Save/Open, but they are not a
second plot specification and they are applied only through the plot host's
public control plane.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping


__all__ = ["PanelPlotAnnotations", "apply_panel_plot_annotations"]


@dataclass(frozen=True, slots=True)
class PanelPlotAnnotations:
    """Generic annotations needed to replay one exact prepared plot."""

    facet_thresholds: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        thresholds: list[float | None] = []
        for value in tuple(self.facet_thresholds):
            if value is None:
                thresholds.append(None)
                continue
            if isinstance(value, bool):
                raise TypeError("facet thresholds must be finite numbers or None")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError("facet thresholds must be finite numbers or None")
            thresholds.append(normalized)
        object.__setattr__(self, "facet_thresholds", tuple(thresholds))

    @property
    def empty(self) -> bool:
        return not self.facet_thresholds

    def document(self) -> Mapping[str, object]:
        if self.empty:
            return {}
        return {"facet_thresholds": list(self.facet_thresholds)}

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> "PanelPlotAnnotations":
        values = document.get("facet_thresholds", ())
        if not isinstance(values, (tuple, list)):
            raise TypeError("facet_thresholds annotation must be a sequence")
        return cls(tuple(values))


def apply_panel_plot_annotations(
    host: object,
    annotations: PanelPlotAnnotations,
    *,
    display: bool,
) -> tuple[object, ...]:
    """Apply annotations through the same public host seam used at runtime."""

    if not isinstance(annotations, PanelPlotAnnotations):
        raise TypeError("annotations must be PanelPlotAnnotations")
    if annotations.empty:
        return ()
    setter = getattr(host, "set_facet_thresholds", None)
    if not callable(setter):
        raise TypeError("plot host cannot restore saved facet thresholds")
    return (
        setter(annotations.facet_thresholds, display=bool(display)),
    )
