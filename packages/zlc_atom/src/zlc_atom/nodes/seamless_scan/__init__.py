"""Seamless scan Measurement: the board advances the plan from its scan table.

The loop itself lives in ``nodes/scan`` -- it has a second consumer, the
temperature Task -- and this package is the authoring form that offers it.
"""

from .logic_node import LOGIC_NODE, SEAMLESS_SCAN_SCHEMA

__all__ = [
    "LOGIC_NODE",
    "SEAMLESS_SCAN_SCHEMA",
]
