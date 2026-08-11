"""What every scan shares: its plan, its ports, its dataset, its editor.

This package is a LIBRARY, not a node -- it carries no ``logic_node.py`` and
discovery does not list it.  Two nodes stand on it, and the difference
between them is one sentence each: ``seamless_scan`` lets the BOARD advance
the points from its own scan table, ``stepped_scan`` has the HOST advance
them, one applied point at a time.  Everything else -- what a plan is, which
ports a bench offers, how a captured point lands in the dataset, how the plan
is authored -- is the same for both and lives here.
"""

from .dataset import (
    SCAN_OUTPUT,
    ScanDatasetWriter,
    ScanLiveSlot,
    scan_dataset_schema,
)
from .plan import (
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    SCAN_PULSE_CONTRACT,
    SCAN_PULSE_RESOURCE,
    ScanAxis,
    ScanPlan,
    ScanPort,
    bind_plan,
    load_scan_template,
    plan_from_authored,
    scan_ports_for,
    scan_ports_for_devices,
)
from .source import drain_backlog, follow_source, next_source_value

__all__ = [
    "DEVICE_PARAM_FAMILY",
    "PULSE_PARAM_FAMILY",
    "SCAN_OUTPUT",
    "SCAN_PULSE_CONTRACT",
    "SCAN_PULSE_RESOURCE",
    "ScanAxis",
    "ScanDatasetWriter",
    "ScanLiveSlot",
    "ScanPlan",
    "ScanPort",
    "bind_plan",
    "drain_backlog",
    "follow_source",
    "load_scan_template",
    "next_source_value",
    "plan_from_authored",
    "scan_dataset_schema",
    "scan_ports_for",
    "scan_ports_for_devices",
]
