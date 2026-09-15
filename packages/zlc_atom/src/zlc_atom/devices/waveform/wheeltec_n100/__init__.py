"""The WHEELTEC N100 inertial module as a waveform source."""

from .console import FdiConfigConsole
from .source import (
    IMU_RATE_PARAMETER,
    PACKET_RATE_LADDER_HZ,
    DEFAULT_BAUD,
    FRAME_HEAD,
    FRAME_TAIL,
    IMU_PACKET,
    MAX_PACKET_RATE_HZ,
    N100_OUTPUTS,
    WheeltecN100Config,
    WheeltecN100WaveformSource,
    discover_n100,
    drain_imu_samples,
    header_crc8,
    payload_crc16,
    rate_ladder_index,
    wake_from_config_mode,
)

__all__ = [
    "IMU_RATE_PARAMETER",
    "PACKET_RATE_LADDER_HZ",
    "DEFAULT_BAUD",
    "FdiConfigConsole",
    "FRAME_HEAD",
    "FRAME_TAIL",
    "IMU_PACKET",
    "MAX_PACKET_RATE_HZ",
    "N100_OUTPUTS",
    "WheeltecN100Config",
    "WheeltecN100WaveformSource",
    "discover_n100",
    "drain_imu_samples",
    "header_crc8",
    "payload_crc16",
    "rate_ladder_index",
    "wake_from_config_mode",
]
