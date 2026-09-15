"""A Tektronix oscilloscope as a waveform source."""

from .source import (
    TIME_PER_DIV_FIELD,
    TekScopeConfig,
    TekScopeWaveformSource,
    discover_tek_scopes,
    is_tektronix,
    volts_per_div_field,
)

__all__ = [
    "TIME_PER_DIV_FIELD",
    "TekScopeConfig",
    "TekScopeWaveformSource",
    "discover_tek_scopes",
    "is_tektronix",
    "volts_per_div_field",
]
