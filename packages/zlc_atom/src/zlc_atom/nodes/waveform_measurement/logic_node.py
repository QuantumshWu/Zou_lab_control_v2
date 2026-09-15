"""Discoverable waveform measurement: one node for every waveform source."""

from __future__ import annotations

from collections.abc import Mapping

from zlc_atom.nodes._framework.descriptor import (
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
)

from .measurement import (
    WAVEFORM_MEASUREMENT_SCHEMA,
    WaveformMeasurementNode,
    WaveformMeasurementRequest,
    waveform_outputs,
    waveform_preview,
)


def _outputs(values: Mapping[str, object], devices: Mapping[str, object]) -> tuple:
    """One signal per quantity the bound source carries; nothing until one is bound."""

    del values
    source = devices.get("sampler")
    return () if source is None else waveform_outputs(source.outputs)


def _previews(values: Mapping[str, object], devices: Mapping[str, object]) -> tuple:
    del values
    source = devices.get("sampler")
    return () if source is None else (waveform_preview(source),)


def _build(
    *,
    sampler: object,
    sampler_key: str,
    signal_plane: object,
    **values: object,
) -> WaveformMeasurementNode:
    authored = WAVEFORM_MEASUREMENT_SCHEMA.project_values(values)
    return WaveformMeasurementNode(
        sampler=sampler,  # type: ignore[arg-type]
        request=WaveformMeasurementRequest(
            sampler_key=sampler_key,
            repeat=int(authored["repeat"]),
            read_interval_seconds=float(authored["read_interval_seconds"]),
        ),
        signal_plane=signal_plane,
        producer="waveform_measurement",
    )


LOGIC_NODE = LogicNodeDescriptor(
    "waveform_measurement",
    NodeKind.MEASUREMENT,
    WAVEFORM_MEASUREMENT_SCHEMA,
    reports_ready=True,
    # What it publishes is what its instrument carries: four quantities off
    # an IMU packet, one voltage off a scope.  The run freezes every knob
    # the instrument has, because a capture takes the instrument as it
    # stands.
    declare_outputs=_outputs,
    declare_previews=_previews,
    device_requirements=(DeviceRequirement("waveform.source", "sampler", None),),
    build=_build,
)

__all__ = ["LOGIC_NODE"]
