"""Public, headless logic-node declarations and composition helpers."""

from importlib.resources import files

from ._framework.descriptor import (
    ArtifactCodec,
    ArtifactInputSpec,
    ArtifactOutputSpec,
    DatasetInputSpec,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    ResolvedArtifact,
    ResolvedWorkspaceResource,
    SelectionMapping,
    NodePreviewSpec,
    WorkspaceResourceSpec,
)
from ._framework.discovery import discover_logic_nodes


def calibration_pulse_template_bytes() -> bytes:
    """Return the shipped imaging template exactly as packaged."""

    return (
        files("zlc_atom.nodes.calibration")
        .joinpath("imaging_template.json")
        .read_bytes()
    )


def scan_pulse_template_bytes() -> bytes:
    """Return the shipped MOT field-scan template exactly as packaged."""

    return (
        files("zlc_atom.nodes.scan")
        .joinpath("mot_field_template.json")
        .read_bytes()
    )


def temperature_pulse_template_bytes() -> bytes:
    """Return the shipped release-recapture (temperature) template.

    Two probe windows around a variable trap-off release period whose
    duration is the ``t_off`` API parameter -- the knob a temperature scan
    turns while it watches occupancy survival.
    """

    return (
        files("zlc_atom.nodes.scan")
        .joinpath("temperature_template.json")
        .read_bytes()
    )


__all__ = [
    "ArtifactCodec",
    "ArtifactInputSpec",
    "ArtifactOutputSpec",
    "DatasetInputSpec",
    "DeviceRequirement",
    "LogicNodeDescriptor",
    "NodeKind",
    "ResolvedArtifact",
    "ResolvedWorkspaceResource",
    "SelectionMapping",
    "NodePreviewSpec",
    "WorkspaceResourceSpec",
    "calibration_pulse_template_bytes",
    "scan_pulse_template_bytes",
    "temperature_pulse_template_bytes",
    "discover_logic_nodes",
]
