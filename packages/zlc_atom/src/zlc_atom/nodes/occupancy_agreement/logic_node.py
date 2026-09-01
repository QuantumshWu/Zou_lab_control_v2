"""Discoverable occupancy-agreement processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
)

from .processor import OCCUPANCY_AGREEMENT_OUTPUTS, OccupancyAgreementProcessor


OCCUPANCY_AGREEMENT_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "first_occupancy_frame",
            "int",
            "First occupancy frame",
            0,
            minimum=0,
        ),
        AuthoringField(
            "counts_frame",
            "int",
            "Counts frame",
            1,
            minimum=0,
        ),
        AuthoringField(
            "second_occupancy_frame",
            "int",
            "Second occupancy frame",
            2,
            minimum=0,
        ),
    )
)


def _build(*, source_signal: str, **values: object) -> OccupancyAgreementProcessor:
    authored = OCCUPANCY_AGREEMENT_SCHEMA.project_values(values)
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    return OccupancyAgreementProcessor(
        source_signal=selected_source,
        first_occupancy_frame=authored["first_occupancy_frame"],
        counts_frame=authored["counts_frame"],
        second_occupancy_frame=authored["second_occupancy_frame"],
    )


LOGIC_NODE = LogicNodeDescriptor(
    "occupancy_agreement",
    NodeKind.PROCESSOR,
    OCCUPANCY_AGREEMENT_SCHEMA,
    input_specs=(
        DatasetInputSpec(
            "counts",
            "occupancy.counts",
            "exact",
            sibling_outputs=("occupied",),
        ),
    ),
    outputs=OCCUPANCY_AGREEMENT_OUTPUTS,
    build=_build,
)

__all__ = ["LOGIC_NODE", "OCCUPANCY_AGREEMENT_SCHEMA"]
