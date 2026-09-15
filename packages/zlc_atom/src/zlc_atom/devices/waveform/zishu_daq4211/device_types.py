"""The ZishuTech DAQ-4211 this bench can install, and how to find one."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.devices.waveform.binding import bind_waveform_source
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from .source import (
    INPUT_RANGES,
    SUPPORTED_MODEL,
    ChannelReading,
    ZishuDaq4211Config,
    ZishuDaq4211WaveformSource,
    discover_daq4211,
)


#: One row per analog input in use, saying what is wired to it.  The card
#: measures volts; what those volts MEAN is the bench's statement, and this
#: is where the bench states it -- an FLC 100 is 50 µT per volt about its own
#: 2.5 V reference, a bare tap is one volt per volt about zero.  Rows sharing
#: a quantity become one published signal with those channels' labels, which
#: is how three FLC 100 heads become one three-component magnetic field.
CHANNEL_COLUMNS = (
    AuthoringField("channel", "int", "AI channel", 0, minimum=0, required=True),
    AuthoringField("quantity", "str", "Quantity", "voltage", required=True),
    AuthoringField("unit", "str", "Unit", "V", required=True),
    AuthoringField("label", "str", "Channel label", "AI0", required=True),
    AuthoringField(
        "unit_per_volt",
        "float",
        "Unit per volt",
        1.0,
        required=True,
        description="the sensor's scale: 50 for an FLC 100 read in uT",
    ),
    AuthoringField(
        "offset_volts",
        "float",
        "Zero at (V)",
        0.0,
        description="the volts that mean zero: 2.5 for an FLC 100 read single-ended",
    ),
)

ZISHU_DAQ4211_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "str", "Device serial", "", required=True),
        AuthoringField(
            "sample_rate_hz",
            "int",
            "Sample rate (Hz)",
            10000,
            minimum=1,
            unit="Hz",
            description="every enabled channel is sampled at this rate, together",
        ),
        AuthoringField(
            "record_samples",
            "int",
            "Samples per record",
            100,
            minimum=1,
            description="one record is one shot; the card streams in ~50 ms packets",
        ),
        AuthoringField(
            "input_range",
            "choice",
            "Input range",
            "10",
            choices=tuple(
                AuthoringChoice(f"{value:g}", f"±{value:g} V")
                for value in sorted(INPUT_RANGES, reverse=True)
            ),
            description="all channels share one range",
        ),
        AuthoringField(
            "readings",
            "rows",
            "Channels",
            (
                {
                    "channel": 0,
                    "quantity": "voltage",
                    "unit": "V",
                    "label": "AI0",
                    "unit_per_volt": 1.0,
                    "offset_volts": 0.0,
                },
            ),
            required=True,
            columns=CHANNEL_COLUMNS,
            description="one row per analog input in use",
        ),
        AuthoringField(
            "timeout_seconds", "float", "Record timeout (s)", 2.0, minimum=0.05
        ),
    )
)


def authored_config(values: dict) -> ZishuDaq4211Config:
    """One card's configuration, as the operator wrote it down."""

    authored = ZISHU_DAQ4211_SCHEMA.project_values(values)
    return ZishuDaq4211Config(
        serial=str(authored["serial"]),
        readings=tuple(
            ChannelReading(
                channel=int(row["channel"]),
                quantity=str(row["quantity"]),
                unit=str(row["unit"]),
                label=str(row["label"]),
                unit_per_volt=float(row["unit_per_volt"]),
                offset_volts=float(row["offset_volts"]),
            )
            for row in authored["readings"]
        ),
        sample_rate_hz=int(authored["sample_rate_hz"]),
        record_samples=int(authored["record_samples"]),
        input_range_volts=float(authored["input_range"]),
        timeout_seconds=float(authored["timeout_seconds"]),
    )


def _factory(context, key: str, values: dict) -> InstalledLeaf:
    source = ZishuDaq4211WaveformSource(authored_config(values))
    return bind_waveform_source(
        context, key, source, source.identity, "waveform.zishu_daq4211"
    )


def _discover() -> tuple[DeviceInstanceConfig, ...]:
    serials = discover_daq4211()
    if not serials:
        raise RuntimeError(
            f"no {SUPPORTED_MODEL} answered: plug the card in over USB (or add "
            "its address in DAQ2-Explorer for a LAN card) and make sure no "
            "other program holds it -- libdaq2 gives one process the card"
        )

    def named(serial: str) -> str:
        return "daq_" + "".join(c if c.isalnum() else "_" for c in serial)

    return tuple(
        DeviceInstanceConfig(
            instance_id=named(serial),
            role=named(serial),
            type_id="waveform.zishu_daq4211",
            parameters=ZISHU_DAQ4211_SCHEMA.project_values({"serial": serial}),
        )
        for serial in serials
    )


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "waveform.zishu_daq4211",
        "waveform",
        ZISHU_DAQ4211_SCHEMA,
        ("waveform.source",),
        factory=_factory,
        discover=_discover,
    ),
)

__all__ = ["CHANNEL_COLUMNS", "DEVICE_TYPES", "ZISHU_DAQ4211_SCHEMA", "authored_config"]
