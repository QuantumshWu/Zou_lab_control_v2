"""The WHEELTEC N100 inertial module as a waveform source."""

from .console import FdiConfigConsole, PacketRate
from .source import (
    DEFAULT_BAUD,
    FRAME_HEAD,
    FRAME_TAIL,
    IMU_PACKET,
    MAX_PACKET_RATE_HZ,
    N100_OUTPUTS,
    OPERATOR_PARAMETER_PREFIXES,
    WheeltecN100Config,
    WheeltecN100WaveformSource,
    discover_n100,
    drain_imu_samples,
    header_crc8,
    packet_id_of,
    packet_rate_field,
    payload_crc16,
)

__all__ = [
    "DEFAULT_BAUD",
    "FdiConfigConsole",
    "FRAME_HEAD",
    "FRAME_TAIL",
    "IMU_PACKET",
    "MAX_PACKET_RATE_HZ",
    "N100_OUTPUTS",
    "OPERATOR_PARAMETER_PREFIXES",
    "PacketRate",
    "WheeltecN100Config",
    "WheeltecN100WaveformSource",
    "discover_n100",
    "drain_imu_samples",
    "header_crc8",
    "packet_id_of",
    "packet_rate_field",
    "payload_crc16",
]
