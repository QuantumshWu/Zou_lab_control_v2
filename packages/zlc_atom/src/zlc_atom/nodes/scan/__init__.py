"""What every scan shares: its plan, its ports, its dataset, its editor.

This package is a LIBRARY, not a node -- it carries no ``logic_node.py`` and
discovery does not list it.  Two nodes stand on it, and the difference
between them is one sentence each: ``seamless_scan`` lets the BOARD advance
the points from its own scan table, ``stepped_scan`` has the HOST advance
them, one applied point at a time.  Everything else -- what a plan is, which
ports a bench offers, how a captured point lands in the dataset, how the plan
is authored -- is the same for both and lives here.

The board-advanced LOOP lives here too, for one reason: it has a second
consumer.  The ``temperature`` Task scans ``t_off`` and publishes survival,
not frames; it is not allowed to import the ``seamless_scan`` node and it must
not grow a second copy of the loop, so the loop is library and the node
package is its authoring form.  The host-advanced loop still has exactly one
consumer and stays in ``stepped_scan`` until it has two.
"""

from .dataset import (
    SCAN_OUTPUT,
    ScanDatasetWriter,
    scan_dataset_schema,
    scan_repeat_domain,
)
from .seamless import MANUAL_AXIS_REQUEST, SeamlessScanMeasurement
from .plan import (
    SCAN_PLAN_SELECTIONS,
    DEVICE_PARAM_FAMILY,
    MANUAL_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    SCAN_PULSE_CONTRACT,
    SEAMLESS_PULSE_RESOURCE,
    STEPPED_PULSE_RESOURCE,
    ScanAxis,
    ScanPlan,
    ScanPort,
    bind_plan,
    host_advanced_port,
    hardware_scan_ports_for,
    load_seamless_template,
    manual_axis,
    manual_axis_name,
    split_outer_axes,
    load_stepped_template,
    slots_from_plan,
    api_overrides_from_authored,
    apply_api_overrides,
    api_overrides_to_authored,
    plan_from_authored,
    scan_ports_for,
    scan_ports_for_devices,
)
from .source import (
    PublishedSignalSource,
    check_cancelled,
    wait_for_board,
    watched_signal_source,
)

__all__ = [
    "SCAN_PLAN_SELECTIONS",
    "PublishedSignalSource",
    "watched_signal_source",
    "check_cancelled",
    "wait_for_board",
    "DEVICE_PARAM_FAMILY",
    "MANUAL_AXIS_REQUEST",
    "MANUAL_PARAM_FAMILY",
    "PULSE_PARAM_FAMILY",
    "SCAN_OUTPUT",
    "SCAN_PULSE_CONTRACT",
    "SEAMLESS_PULSE_RESOURCE",
    "STEPPED_PULSE_RESOURCE",
    "ScanAxis",
    "ScanDatasetWriter",
    "ScanPlan",
    "ScanPort",
    "SeamlessScanMeasurement",
    "bind_plan",
    "host_advanced_port",
    "hardware_scan_ports_for",
    "load_seamless_template",
    "load_stepped_template",
    "manual_axis",
    "manual_axis_name",
    "split_outer_axes",
    "slots_from_plan",
    "api_overrides_from_authored",
    "apply_api_overrides",
    "api_overrides_to_authored",
    "plan_from_authored",
    "scan_dataset_schema",
    "scan_repeat_domain",
    "scan_ports_for",
    "scan_ports_for_devices",
]
