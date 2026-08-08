"""Public, headless logic-node declarations and composition helpers."""

from importlib.resources import files

from ._framework.descriptor import ArtifactInputSpec, DatasetInputSpec
from ._framework.discovery import discover_logic_nodes


def calibration_pulse_template_bytes() -> bytes:
    """Return the shipped v2 imaging template exactly as packaged."""

    return (
        files("zlc_atom.nodes.calibration")
        .joinpath("imaging_template.json")
        .read_bytes()
    )


__all__ = [
    "ArtifactInputSpec",
    "DatasetInputSpec",
    "calibration_pulse_template_bytes",
    "discover_logic_nodes",
]
