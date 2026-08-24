"""Discoverable frame-survival processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
)

from .processor import SURVIVAL_OUTPUTS, FrameSurvivalProcessor

#: Nothing to author: the forward frame pairs follow from the data itself,
#: so the form is empty on purpose -- an operator picks the occupancy
#: signal and presses Start.
FRAME_SURVIVAL_SCHEMA = AuthoringSchema(())


def _build(*, source_signal: str, **values: object) -> FrameSurvivalProcessor:
    FRAME_SURVIVAL_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    return FrameSurvivalProcessor(source_signal=selected_source)


LOGIC_NODE = LogicNodeDescriptor(
    "frame_survival",
    NodeKind.PROCESSOR,
    FRAME_SURVIVAL_SCHEMA,
    input_specs=(DatasetInputSpec("occupied", None, "exact"),),
    outputs=SURVIVAL_OUTPUTS,
    build=_build,
)

__all__ = ["FRAME_SURVIVAL_SCHEMA", "LOGIC_NODE"]
