"""The WHEELTEC N100 inertial module as a waveform source."""

from .console import FdiConfigConsole, IMU_PACKET_NAME
from .source import (
    PACKET_RATE_LADDER_HZ,
    DEFAULT_BAUD,
    FRAME_HEAD,
    FRAME_TAIL,
    IMU_PACKET,
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
    "IMU_PACKET_NAME",
    "PACKET_RATE_LADDER_HZ",
    "DEFAULT_BAUD",
    "FdiConfigConsole",
    "FRAME_HEAD",
    "FRAME_TAIL",
    "IMU_PACKET",
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
