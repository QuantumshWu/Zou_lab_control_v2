"""Minimal host package for the frozen pulse-streamer device."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"
_PACKAGE_DIR = Path(__file__).resolve().parent
if _PACKAGE_DIR.name != "zlc_pulse" or __package__ != "zlc_pulse":
    raise ImportError(f"unexpected zlc_pulse installation path: {_PACKAGE_DIR}")

from .codec import PULSE_TREE_FORMAT, sequence_from_tree, sequence_to_tree
from .model import (
    ANALOG_MODE_CHOICES,
    MINIMUM_REPEAT_COUNT,  # noqa: E402
    TIME_UNIT_CHOICES,
    TIME_UNIT_TO_NS,
    align_to_grid,
    cycle_binding_kind,
    AnalogStep,
    OutputDelay,
    PulseApiParameter,
    PulseFieldRef,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseSlot,
    PulseTarget,
    RepeatRegion,
)
from .compile import compile_sequence  # noqa: E402
from .binding import (  # noqa: E402
    authored_api_values,
    pulse_field_value,
    resolve_api_parameters,
)
from .wire import load_streamer_config  # noqa: E402
from .manifest import pulse_target_from_xdc  # noqa: E402
from .device import PulseStreamer  # noqa: E402
from .endpoint import (  # noqa: E402
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT,
)
from .scan import (  # noqa: E402
    api_parameter_columns_for,
    resolve_scan_point,
    scan_columns_for,
    scan_rows_from_wire,
    scan_rows_to_wire,
    scan_table_template,
    validate_scan_table,
)
from .transport import (  # noqa: E402
    MemoryRegisterTransport,
    UartError,
    UartRegisterTransport,
    VivadoAxiRegisterTransport,
)

# This is the final user-facing package surface.  Keep implementation types
# and test transports importable from their owning submodules, not this file.
__all__ = [
    "PulseStreamer",
    "RemotePulseStreamer",
    "connect",
    "serve",
    "PulseSequence",
    "PulseApiParameter",
    "PulsePeriod",
    "AnalogStep",
    "PulsePortSpec",
    "PulseTarget",
    "PulseSlot",
    "PulseFieldRef",
    "OutputDelay",
    "MINIMUM_REPEAT_COUNT",
    "PULSE_TREE_FORMAT",
    "sequence_from_tree",
    "sequence_to_tree",
    "RepeatRegion",
    "compile_sequence",
    "pulse_target_from_xdc",
    "load_streamer_config",
    "UartRegisterTransport",
    "VivadoAxiRegisterTransport",
    "MemoryRegisterTransport",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT",
    "TIME_UNIT_CHOICES",
    "TIME_UNIT_TO_NS",
    "ANALOG_MODE_CHOICES",
    "align_to_grid",
    "cycle_binding_kind",
    "resolve_scan_point",
    "authored_api_values",
    "resolve_api_parameters",
    "pulse_field_value",
    "api_parameter_columns_for",
    "scan_columns_for",
    "scan_table_template",
    "validate_scan_table",
    "scan_rows_to_wire",
    "scan_rows_from_wire",
    "RemoteError",
    "UartError",
    "BackendResolutionError",
    "__version__",
]


def __getattr__(name: str):
    if name in {
        "RemotePulseStreamer", "RemoteError", "BackendResolutionError", "serve", "connect",
    }:
        from .remote import (
            BackendResolutionError,
            RemoteError,
            RemotePulseStreamer,
            connect,
            serve,
        )

        globals().update(
            {
                "RemotePulseStreamer": RemotePulseStreamer,
                "RemoteError": RemoteError,
                "BackendResolutionError": BackendResolutionError,
                "serve": serve,
                "connect": connect,
            }
        )
        return globals()[name]
    raise AttributeError(name)
