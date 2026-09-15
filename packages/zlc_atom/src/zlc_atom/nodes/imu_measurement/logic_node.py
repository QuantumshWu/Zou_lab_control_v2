"""Discoverable IMU measurement descriptor: the four quantities one packet carries."""

from __future__ import annotations

from zlc_runtime import DatasetOutputDeclaration

from zlc_atom.devices.waveform.wheeltec_n100 import N100_OUTPUTS
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


#: One signal per quantity, named by the packet's own vocabulary; the axes
#: of each are a COMPONENT axis on the dataset, so the vocabulary does not
#: change with how the stream is grouped into events.
IMU_OUTPUTS = tuple(
    DatasetOutputDeclaration(output.name, f"waveform.{output.name}")
    for output in N100_OUTPUTS
)
MAGNETIC_FIELD_OUTPUT = IMU_OUTPUTS[0]

#: A hundred packets an event: at the module's 100-400 Hz that is a quarter
#: to a whole second of field per publication, and four to ten commits a
#: second whatever the rate.
IMU_MEASUREMENT_SCHEMA = waveform_authoring_schema(records_per_event=100)


def _build(
    *,
    sampler: object,
    sampler_key: str,
    signal_plane: object,
    **values: object,
) -> WaveformMeasurementNode:
    authored = IMU_MEASUREMENT_SCHEMA.project_values(values)
    return WaveformMeasurementNode(
        sampler=sampler,  # type: ignore[arg-type]
        request=WaveformMeasurementRequest(
            sampler_key=sampler_key,
            repeat=int(authored["repeat"]),
            records_per_event=int(authored["records_per_event"]),
        ),
        signal_plane=signal_plane,
        outputs=IMU_OUTPUTS,
        producer="imu_measurement",
    )


LOGIC_NODE = LogicNodeDescriptor(
    "imu_measurement",
    NodeKind.MEASUREMENT,
    IMU_MEASUREMENT_SCHEMA,
    reports_ready=True,
    outputs=IMU_OUTPUTS,
    node_previews=(NodePreviewSpec(MAGNETIC_FIELD_OUTPUT, "curve"),),
    device_requirements=(DeviceRequirement("waveform.source", "sampler"),),
    build=_build,
)

__all__ = ["IMU_MEASUREMENT_SCHEMA", "IMU_OUTPUTS", "LOGIC_NODE", "MAGNETIC_FIELD_OUTPUT"]
