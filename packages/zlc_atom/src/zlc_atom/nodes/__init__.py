"""Public, headless logic-node declarations and composition helpers."""

from importlib.resources import files

from ._framework.descriptor import (
    ArtifactCodec,
    ArtifactInputSpec,
    ArtifactOutputSpec,
    DatasetInputSpec,
    DeviceAccess,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    OutputSpec,
    ResolvedArtifact,
    ResolvedWorkspaceResource,
    SelectionMapping,
    TaskPreviewSpec,
    WorkspaceResourceSpec,
)
from ._framework.discovery import discover_logic_nodes


def calibration_pulse_template_bytes() -> bytes:
    """Return the shipped v2 imaging template exactly as packaged."""

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


__all__ = [
    "ArtifactCodec",
    "ArtifactInputSpec",
    "ArtifactOutputSpec",
    "DatasetInputSpec",
    "DeviceAccess",
    "DeviceRequirement",
    "LogicNodeDescriptor",
    "NodeKind",
    "OutputSpec",
    "ResolvedArtifact",
    "ResolvedWorkspaceResource",
    "SelectionMapping",
    "TaskPreviewSpec",
    "WorkspaceResourceSpec",
    "calibration_pulse_template_bytes",
    "scan_pulse_template_bytes",
    "discover_logic_nodes",
]
