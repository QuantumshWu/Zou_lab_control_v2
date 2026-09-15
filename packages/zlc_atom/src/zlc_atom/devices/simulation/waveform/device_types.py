"""The virtual waveform sources this bench can install: an IMU and a scope."""

from __future__ import annotations

import numpy as np

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.devices.waveform.binding import bind_waveform_source
from zlc_atom.devices.waveform.contract import WaveformOutput
from zlc_atom.devices.waveform.wheeltec_n100 import N100_OUTPUTS
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from ..authoring import simulation_world_config
from ..world import SimulationWorld
from .source import VirtualWaveformConfig, VirtualWaveformSource


VIRTUAL_IMU_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "packet_rate_hz", "float", "Packet rate (Hz)", 400.0, minimum=1.0
        ),
    )
)

VIRTUAL_SCOPE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "sample_rate_hz", "float", "Sample rate (Hz)", 10000.0, minimum=1.0
        ),
        AuthoringField("record_samples", "int", "Record samples", 1000, minimum=1),
        AuthoringField("channels", "int", "Channels", 2, minimum=1, maximum=4),
    )
)


def _imu_factory(context, key: str, values: dict) -> InstalledLeaf:
    """An N100 on this world: the packet stream reads the world's bias field."""

    if not isinstance(context.world, SimulationWorld):
        raise TypeError("waveform.virtual_imu requires the installation SimulationWorld")
    authored = VIRTUAL_IMU_SCHEMA.project_values(values)
    world = context.world
    rng = np.random.default_rng((world.config.seed, 0x1D0))
    gravity = np.asarray((0.0, 0.0, 9.80665), dtype=np.float64)

    def samples(times: np.ndarray) -> np.ndarray:
        count = int(times.size)
        out = np.empty((count, 10), dtype=np.float32)
        out[:, 0:3] = np.asarray(world.magnetic_field_microtesla()) + rng.normal(
            0.0, 0.15, (count, 3)
        )
        out[:, 3:6] = rng.normal(0.0, 0.002, (count, 3))
        out[:, 6:9] = gravity + rng.normal(0.0, 0.02, (count, 3))
        out[:, 9] = 300.0 + rng.normal(0.0, 0.01, count)
        return out

    source = VirtualWaveformSource(
        VirtualWaveformConfig(float(authored["packet_rate_hz"]), 1, N100_OUTPUTS),
        sample_source=samples,
    )
    return bind_waveform_source(
        context, key, source, f"virtual-imu:{key}", "waveform.virtual_imu"
    )


def _scope_factory(context, key: str, values: dict) -> InstalledLeaf:
    """A scope whose channels see a 50 Hz line at a different amplitude each."""

    authored = VIRTUAL_SCOPE_SCHEMA.project_values(values)
    channels = int(authored["channels"])
    rng = np.random.default_rng(0x5C0)
    amplitudes = 0.1 * (np.arange(channels, dtype=np.float64) + 1.0)
    phases = np.arange(channels, dtype=np.float64) * (np.pi / 4.0)

    def samples(times: np.ndarray) -> np.ndarray:
        phase = 2.0 * np.pi * 50.0 * times[:, None] + phases[None, :]
        return amplitudes[None, :] * np.sin(phase) + rng.normal(
            0.0, 0.005, (int(times.size), channels)
        )

    source = VirtualWaveformSource(
        VirtualWaveformConfig(
            float(authored["sample_rate_hz"]),
            int(authored["record_samples"]),
            (
                WaveformOutput(
                    "voltage",
                    "V",
                    tuple(f"CH{index + 1}" for index in range(channels)),
                    tuple(range(channels)),
                ),
            ),
        ),
        sample_source=samples,
    )
    return bind_waveform_source(
        context, key, source, f"virtual-scope:{key}", "waveform.virtual_scope"
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "waveform.virtual_imu",
        "waveform",
        VIRTUAL_IMU_SCHEMA,
        ("waveform.source",),
        factory=_imu_factory,
        world_config=simulation_world_config,
    ),
    DeviceTypeDescriptor(
        "waveform.virtual_scope",
        "waveform",
        VIRTUAL_SCOPE_SCHEMA,
        ("waveform.source",),
        factory=_scope_factory,
    ),
)

__all__ = ["DEVICE_TYPES", "VIRTUAL_IMU_SCHEMA", "VIRTUAL_SCOPE_SCHEMA"]
