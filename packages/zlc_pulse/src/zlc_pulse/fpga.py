"""The build-time surface: the ``fpga`` command and the names a build imports.

The capacity estimator, the geometry emitters and the strict config door are
defined in the wire module, beside the geometry they project; this module only
names them for the launchers, the build scripts and ``zlc fpga``.
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
    FPGA_PARTS,
    GEOMETRY_VH_FILENAME,
    FpgaPartProfile,
    SolvedCapacity,
    build_ip_sizes,
    check_config_capacity,
    emit_geom_tcl,
    emit_geometry_vh,
    estimate_resources,
    format_capacity_report,
    load_streamer_config,
    params_from_config,
    part_profile,
    require_streamer_config,
    solve_capacity,
)

__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FPGA_PART",
    "DEFAULT_TARGET_PCT",
    "FROZEN_CLOCK_HZ",
    "FPGA_PARTS",
    "GEOMETRY_VH_FILENAME",
    "FpgaPartProfile",
    "SolvedCapacity",
    "build_ip_sizes",
    "check_config_capacity",
    "emit_geom_tcl",
    "emit_geometry_vh",
    "estimate_resources",
    "format_capacity_report",
    "load_streamer_config",
    "params_from_config",
    "part_profile",
    "require_streamer_config",
    "solve_capacity",
]


def _main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest-selected FPGA capacity/geometry command."""

    return _wire._main(argv)
