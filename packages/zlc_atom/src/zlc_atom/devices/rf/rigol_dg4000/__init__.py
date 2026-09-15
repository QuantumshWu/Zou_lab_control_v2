"""A Rigol DG4000-series function generator behind the RF source contract."""

from .source import (
    Dg4000Sighting,
    RigolDg4000Config,
    RigolDg4000RfSource,
    discover_dg4000,
    is_dg4000,
)

__all__ = [
    "Dg4000Sighting",
    "RigolDg4000Config",
    "RigolDg4000RfSource",
    "discover_dg4000",
    "is_dg4000",
]
