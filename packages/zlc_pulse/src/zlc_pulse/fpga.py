"""FPGA build projections and capacity estimates.

Runtime clients should import geometry, packing, and compatibility checks from
the wire module. This module is the explicit home for build-time emitters and
resource-budget tooling.
"""

from __future__ import annotations

from typing import Sequence

from . import wire as _wire
from .wire import (
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_CONFIG_PATH,
    DEFAULT_FPGA_PART,
    DEFAULT_TARGET_PCT,
    FROZEN_CLOCK_HZ,
    FROZEN_SLOT_MUL_WIDTH,
    FPGA_PARTS,
    GEOMETRY_VH_FILENAME,
    FpgaPartProfile,
    SolvedCapacity,
    build_ip_sizes,
    check_config_capacity,
    default_coeff_frac_bits,
    default_slot_mul_width,
    emit_geom_tcl,
    emit_geometry_vh,
    estimate_resources,
    format_capacity_report,
    load_streamer_config,
    params_from_config,
    part_profile,
    solve_capacity,
)

__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FPGA_PART",
    "DEFAULT_TARGET_PCT",
    "FROZEN_CLOCK_HZ",
    "FROZEN_SLOT_MUL_WIDTH",
    "FPGA_PARTS",
    "GEOMETRY_VH_FILENAME",
    "FpgaPartProfile",
    "SolvedCapacity",
    "build_ip_sizes",
    "check_config_capacity",
    "default_coeff_frac_bits",
    "default_slot_mul_width",
    "emit_geom_tcl",
    "emit_geometry_vh",
    "estimate_resources",
    "format_capacity_report",
    "load_streamer_config",
    "params_from_config",
    "part_profile",
    "solve_capacity",
]


def _main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest-selected FPGA capacity/geometry command."""

    return _wire._main(argv)
