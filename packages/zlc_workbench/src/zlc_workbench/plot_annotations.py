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


__all__ = ["PanelPlotAnnotations"]


@dataclass(frozen=True, slots=True)
class PanelPlotAnnotations:
    """Generic annotations needed to replay one exact prepared plot."""

    classifier_thresholds: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        thresholds: list[float | None] = []
        for value in tuple(self.classifier_thresholds):
            if value is None:
                thresholds.append(None)
                continue
            if isinstance(value, bool):
                raise TypeError("classifier thresholds must be finite numbers or None")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError("classifier thresholds must be finite numbers or None")
            thresholds.append(normalized)
        object.__setattr__(self, "classifier_thresholds", tuple(thresholds))

    @property
    def empty(self) -> bool:
        return not self.classifier_thresholds

    def document(self) -> Mapping[str, object]:
        if self.empty:
            return {}
        return {"classifier_thresholds": list(self.classifier_thresholds)}

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> "PanelPlotAnnotations":
        values = document.get("classifier_thresholds", ())
        if not isinstance(values, (tuple, list)):
            raise TypeError("classifier_thresholds annotation must be a sequence")
        return cls(tuple(values))
