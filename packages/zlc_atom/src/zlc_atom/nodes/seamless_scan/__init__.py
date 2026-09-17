"""Seamless scan Measurement: the board advances the plan from its scan table.

The scan owner lives in ``nodes/scan``; this package declares its authoring form.
"""

from .logic_node import LOGIC_NODE, SEAMLESS_SCAN_SCHEMA

__all__ = [
    "LOGIC_NODE",
    "SEAMLESS_SCAN_SCHEMA",
]
