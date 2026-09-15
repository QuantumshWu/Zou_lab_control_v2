"""Discoverable oscilloscope measurement descriptor: one acquisition, one event."""

from __future__ import annotations

from zlc_runtime import DatasetOutputDeclaration

from zlc_atom.devices.waveform.tek_scope import TIME_PER_DIV_FIELD, volts_per_div_field
from zlc_atom.nodes._framework.descriptor import (
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    NodePreviewSpec,
)
from zlc_atom.nodes.waveform import (
    WaveformMeasurementNode,
    WaveformMeasurementRequest,
    waveform_authoring_schema,
)


#: One acquisition is one shot: a scope's record is already a whole
#: triggered window, and a shot is what triggers it.  The history the
#: output declares is what lets a Rolling panel follow a derived scalar.
VOLTAGE_OUTPUT = DatasetOutputDeclaration(
    "voltage", "waveform.voltage", index_by_source=True
)

#: Every acquisition the scope completes is read.
SCOPE_MEASUREMENT_SCHEMA = waveform_authoring_schema(read_interval_seconds=0.0)

#: The knobs this run takes over: the time base and every channel's scale
#: decide the record's geometry and volts, and a change under a capture
#: would change the schema inside one generation.
SCOPE_PROTECTED_FIELDS = (
    TIME_PER_DIV_FIELD,
    *(volts_per_div_field(channel) for channel in (1, 2, 3, 4)),
)


def _build(
    *,
    sampler: object,
    sampler_key: str,
    signal_plane: object,
    **values: object,
) -> WaveformMeasurementNode:
    authored = SCOPE_MEASUREMENT_SCHEMA.project_values(values)
    return WaveformMeasurementNode(
        sampler=sampler,  # type: ignore[arg-type]
        request=WaveformMeasurementRequest(
            sampler_key=sampler_key,
            repeat=int(authored["repeat"]),
            read_interval_seconds=float(authored["read_interval_seconds"]),
        ),
        signal_plane=signal_plane,
        outputs=(VOLTAGE_OUTPUT,),
        producer="scope_measurement",
    )


LOGIC_NODE = LogicNodeDescriptor(
    "scope_measurement",
    NodeKind.MEASUREMENT,
    SCOPE_MEASUREMENT_SCHEMA,
    reports_ready=True,
    outputs=(VOLTAGE_OUTPUT,),
    node_previews=(NodePreviewSpec(VOLTAGE_OUTPUT, "curve"),),
    device_requirements=(
        DeviceRequirement("waveform.source", "sampler", SCOPE_PROTECTED_FIELDS),
    ),
    build=_build,
)

__all__ = ["LOGIC_NODE", "SCOPE_MEASUREMENT_SCHEMA", "SCOPE_PROTECTED_FIELDS", "VOLTAGE_OUTPUT"]
