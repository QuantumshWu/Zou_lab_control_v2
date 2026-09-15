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


#: One signal per quantity, named by the packet's own vocabulary, each a
#: COMPONENT axis of its channels.  Every packet read is one shot, and each
#: signal declares source-index history so a Rolling panel can lease the
#: last N shots from the Runtime.
IMU_OUTPUTS = tuple(
    DatasetOutputDeclaration(
        output.name, f"waveform.{output.name}", index_by_source=True
    )
    for output in N100_OUTPUTS
)
MAGNETIC_FIELD_OUTPUT = IMU_OUTPUTS[0]

#: Read every packet the module sends: at its 100-400 Hz that is a shot
#: every 2.5 to 10 ms.  An interval turns the stream into a sampling.
IMU_MEASUREMENT_SCHEMA = waveform_authoring_schema(read_interval_seconds=0.0)


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
            read_interval_seconds=float(authored["read_interval_seconds"]),
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
    node_previews=(NodePreviewSpec(MAGNETIC_FIELD_OUTPUT, "rolling"),),
    device_requirements=(DeviceRequirement("waveform.source", "sampler"),),
    build=_build,
)

__all__ = ["IMU_MEASUREMENT_SCHEMA", "IMU_OUTPUTS", "LOGIC_NODE", "MAGNETIC_FIELD_OUTPUT"]
