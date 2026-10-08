"""Minimal host package for the frozen pulse-streamer device."""

from __future__ import annotations

from .codec import (
    CONFIG_VALUES_DIRECTORY,
    CURRENT_CONFIG_VALUES,
    read_pulse_document,
    PULSE_TREE_FORMAT,
    SUBPULSE_TREE_FORMAT,
    read_subpulse,
    write_subpulse,
    subpulse_from_tree,
    subpulse_to_tree,
    CONFIG_VALUES_FORMAT,
    config_values_from_tree,
    config_values_to_tree,
    read_config_values,
    write_config_values,
    sequence_from_tree,
    sequence_to_tree,
)
from .model import (
    ANALOG_MODE_CHOICES,
    MINIMUM_BRACKET_COUNT,
    TIME_UNIT_CHOICES,
    canonical_time_unit,
    nanoseconds_per,
    align_to_grid,
    AnalogStep,
    MAXIMUM_REPEAT_COUNT,
    OutputDelay,
    PulseBracket,
    PulseComponent,
    Subpulse,
    group_component,
    ungroup_component,
    extract_subpulse,
    insert_subpulse,
    replace_component,
    remove_component,
    PulseFieldRef,
    PERIOD_KIND_PERIOD,
    PERIOD_KIND_SPACER,
    PERIOD_KINDS,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseBinding,
    PulseTarget,
)
from .compile import analog_levels, compile_sequence
from .binding import (
    apply_api_values,
    apply_config_values,
    authored_api_entries,
    authored_api_values,
    api_bindings_in_period_order,
    field_label,
    normalize_binding_values,
    config_parameter_key,
    prune_orphaned_bindings,
    convert_time,
    pulse_field_value,
    resolve_api_parameters,
)
from .wire import load_streamer_config
from .manifest import pulse_target_from_xdc
from .device import PulseStreamer
from .endpoint import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT,
)
from .scan import (
    api_parameter_columns_for,
    prepare_scan_application,
    resolve_scan_point,
    scan_columns_for,
    scan_rows_from_wire,
    scan_rows_to_wire,
    scan_table_template,
    validate_scan_table,
)
from .transport import (
    MemoryRegisterTransport,
)

# This is the final user-facing package surface.  Keep implementation types
# and test transports importable from their owning submodules, not this file.
__all__ = [
    "CONFIG_VALUES_DIRECTORY",
    "CURRENT_CONFIG_VALUES",
    "read_config_values",
    "write_config_values",
    "PulseStreamer",
    "RemotePulseStreamer",
    "connect",
    "serve",
    "PulseSequence",
    "PulseComponent",
    "Subpulse",
    "group_component",
    "ungroup_component",
    "extract_subpulse",
    "insert_subpulse",
    "replace_component",
    "remove_component",
    "PERIOD_KIND_PERIOD",
    "PERIOD_KIND_SPACER",
    "PERIOD_KINDS",
    "PulsePeriod",
    "AnalogStep",
    "PulsePortSpec",
    "PulseTarget",
    "PulseBinding",
    "PulseFieldRef",
    "OutputDelay",
    "MINIMUM_BRACKET_COUNT",
    "PULSE_TREE_FORMAT",
    "SUBPULSE_TREE_FORMAT",
    "read_subpulse",
    "write_subpulse",
    "subpulse_from_tree",
    "subpulse_to_tree",
    "read_pulse_document",
    "sequence_from_tree",
    "sequence_to_tree",
    "MAXIMUM_REPEAT_COUNT",
    "PulseBracket",
    "compile_sequence",
    "pulse_target_from_xdc",
    "load_streamer_config",
    "MemoryRegisterTransport",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT",
    "TIME_UNIT_CHOICES",
    "canonical_time_unit",
    "nanoseconds_per",
    "ANALOG_MODE_CHOICES",
    "align_to_grid",
    "analog_levels",
    "resolve_scan_point",
    "CONFIG_VALUES_FORMAT",
    "config_values_from_tree",
    "config_values_to_tree",
    "apply_api_values",
    "apply_config_values",
    "authored_api_entries",
    "authored_api_values",
    "api_bindings_in_period_order",
    "field_label",
    "normalize_binding_values",
    "config_parameter_key",
    "prune_orphaned_bindings",
    "convert_time",
    "resolve_api_parameters",
    "pulse_field_value",
    "api_parameter_columns_for",
    "prepare_scan_application",
    "scan_columns_for",
    "scan_table_template",
    "validate_scan_table",
    "scan_rows_to_wire",
    "scan_rows_from_wire",
    "RemoteError",
    "LocalPulseService",
]


def __getattr__(name: str):
    if name in {
        "RemotePulseStreamer", "RemoteError", "serve", "connect",
        "LocalPulseService",
    }:
        from .remote import (
            LocalPulseService,
            RemoteError,
            RemotePulseStreamer,
            connect,
            serve,
        )

        globals().update(
            {
                "RemotePulseStreamer": RemotePulseStreamer,
                "RemoteError": RemoteError,
                "serve": serve,
                "connect": connect,
                "LocalPulseService": LocalPulseService,
            }
        )
        return globals()[name]
    raise AttributeError(name)
