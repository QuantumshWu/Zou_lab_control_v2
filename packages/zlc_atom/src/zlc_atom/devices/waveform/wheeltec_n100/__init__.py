"""The WHEELTEC N100 inertial module as a waveform source."""

from .source import (
    DEFAULT_BAUD,
    FRAME_HEAD,
    FRAME_TAIL,
    IMU_PACKET,
    N100_OUTPUTS,
    WheeltecN100Config,
    WheeltecN100WaveformSource,
    discover_n100,
    drain_imu_samples,
)

__all__ = [
    "DEFAULT_BAUD",
    "FRAME_HEAD",
    "FRAME_TAIL",
    "IMU_PACKET",
    "N100_OUTPUTS",
    "WheeltecN100Config",
    "WheeltecN100WaveformSource",
    "discover_n100",
    "drain_imu_samples",
]
