"""The RF source contract, driven over faked transports.

Per the bench's one virtualization rule the fake is the lowest layer only:
these tests run the REAL drivers -- their SCPI vocabulary, their unit
conversions, their grid refusals, their read-back discipline -- over a link
or library that answers from memory.
"""

from __future__ import annotations

import pytest

from zlc_atom.devices.rf.contract import (
    FREQUENCY_FIELD,
    OUTPUT_FIELD,
    POWER_FIELD,
)
from zlc_atom.devices.rf.rigol_dg4000 import RigolDg4000Config, RigolDg4000RfSource
from zlc_atom.devices.rf.vaunix_lms import VaunixLmsConfig, VaunixLmsRfSource
from zlc_atom.devices.simulation.rf import InMemoryLmsLibrary, virtual_rf_source


class _ScpiInstrument:
    """A DG4000's worth of SCPI, answered from per-channel register dicts."""

    def __init__(self) -> None:
        self.registers = {
            channel: {"FREQ": 1000.0, "VOLT": -30.0, "OUT": "OFF"}
            for channel in ("1", "2")
        }
        self.log: list[str] = []

    @staticmethod
    def _channel(command: str) -> str:
        upper = command.upper()
        for token in (":SOURCE", ":OUTPUT"):
            at = upper.find(token)
            if at >= 0:
                return upper[at + len(token)]
        raise AssertionError(f"no channel in {command!r}")

    def write(self, command: str) -> None:
        self.log.append(command)
        upper = command.upper()
        registers = self.registers[self._channel(command)]
        if ":FREQUENCY " in upper:
            registers["FREQ"] = float(command.split()[-1])
        elif ":VOLTAGE " in upper and ":UNIT" not in upper:
            registers["VOLT"] = float(command.split()[-1])
        elif upper.startswith(":OUTPUT"):
            registers["OUT"] = command.split()[-1].upper()

    def query(self, command: str) -> str:
        self.log.append(command)
        upper = command.upper()
        if upper == "*IDN?":
            return "RIGOL TECHNOLOGIES,DG4162,DG4E0000000001,00.01.12"
        registers = self.registers[self._channel(command)]
        if ":FREQUENCY?" in upper:
            return f"{registers['FREQ']:.6E}"
        if ":VOLTAGE?" in upper:
            return f"{registers['VOLT']:.4E}"
        if upper.startswith(":OUTPUT"):
            return registers["OUT"]
        raise AssertionError(f"unexpected query {command!r}")

    def close(self) -> None:
        self.log.append("<closed>")


def _rigol(**overrides) -> tuple[RigolDg4000RfSource, _ScpiInstrument]:
    instrument = _ScpiInstrument()
    config = RigolDg4000Config(resource="TCPIP0::198.51.100.7::INSTR", **overrides)
    return RigolDg4000RfSource(config, link=instrument), instrument


def test_one_instrument_is_one_instance_with_every_channel_s_knobs() -> None:
    """Channels are the device's own structure, never the operator's to manage.

    One DG4162 is one card offering six knobs -- ch1/ch2 each with
    frequency, power and output -- and tuning one channel must not move the
    other.  DBM is pinned on BOTH channels at open.
    """

    source, instrument = _rigol()
    assert ":SOURce1:VOLTage:UNIT DBM" in instrument.log
    assert ":SOURce2:VOLTage:UNIT DBM" in instrument.log

    names = [field.metadata.name for field in source.tunable_fields()]
    assert names == [
        "ch1_frequency_hz",
        "ch1_power_dbm",
        "ch1_output_enabled",
        "ch2_frequency_hz",
        "ch2_power_dbm",
        "ch2_output_enabled",
    ]

    assert source.tune("ch1_frequency_hz", 80e6) == 80e6
    assert source.tune("ch2_frequency_hz", 5e6) == 5e6
    assert instrument.registers["1"]["FREQ"] == 80e6
    assert instrument.registers["2"]["FREQ"] == 5e6
    assert source.tune("ch2_output_enabled", True) is True
    assert instrument.registers["1"]["OUT"] == "OFF", (
        "tuning one channel must not move the other"
    )
    values = source.tunable_values()
    assert values["ch1_frequency_hz"] == 80e6
    assert values["ch2_frequency_hz"] == 5e6
    assert values["ch2_output_enabled"] is True


def test_bounds_are_bench_policy_and_refuse_before_writing() -> None:
    source, instrument = _rigol(frequency_high_hz=1e6)
    written = {
        channel: dict(registers)
        for channel, registers in instrument.registers.items()
    }
    with pytest.raises(ValueError, match="ch2_frequency_hz must lie in"):
        source.tune("ch2_frequency_hz", 2e6)
    with pytest.raises(ValueError, match="ch1_power_dbm must lie in"):
        source.tune("ch1_power_dbm", 99.0)
    with pytest.raises(TypeError, match="ch1_output_enabled takes a bool"):
        source.tune("ch1_output_enabled", 1)
    with pytest.raises(ValueError, match="no tunable field"):
        source.tune("frequency_hz", 1e5)
    assert instrument.registers == written, "a refusal must not touch hardware"


def test_every_accepted_tune_advances_the_settings_epoch() -> None:
    source, _instrument = _rigol()
    first = source.settings_provenance()
    source.tune("ch1_frequency_hz", 10e6)
    second = source.settings_provenance()
    assert second["settings_epoch"] == first["settings_epoch"] + 1
    assert second["device_session_id"] == first["device_session_id"]
    assert "DG4162" in str(second["device_session_id"])


def test_the_scan_facing_fields_carry_bounds_and_units() -> None:
    source, _instrument = _rigol()
    by_name = {field.metadata.name: field for field in source.tunable_fields()}
    for channel in ("ch1", "ch2"):
        frequency = by_name[f"{channel}_frequency_hz"].metadata
        assert (frequency.minimum, frequency.maximum) == (1e3, 160e6)
        assert frequency.unit == "Hz"
        assert frequency.label.startswith(channel.upper())
        assert by_name[f"{channel}_power_dbm"].metadata.unit == "dBm"
        # The output switch is a control, not an axis: unbounded on purpose,
        # so scan_ports_for_devices never offers it.
        output = by_name[f"{channel}_output_enabled"].metadata
        assert output.minimum is None and output.maximum is None
    for field in by_name.values():
        assert field.live_write
        assert field.dependency_group == (field.metadata.name,)


def test_the_lab_brick_speaks_its_own_units_and_refuses_off_grid() -> None:
    source = virtual_rf_source(VaunixLmsConfig(serial=1001))
    # 10 Hz frequency grid, quarter-dB power grid: representable values pass
    # exactly, everything else is refused BEFORE the write, naming the grid.
    assert source.tune(FREQUENCY_FIELD, 1_000_000_010.0) == 1_000_000_010.0
    with pytest.raises(ValueError, match="10.*Hz grid"):
        source.tune(FREQUENCY_FIELD, 1_000_000_005.0)
    assert source.tune(POWER_FIELD, -3.25) == -3.25
    with pytest.raises(ValueError, match="0.25.*dBm grid"):
        source.tune(POWER_FIELD, -3.1)
    assert source.tune(OUTPUT_FIELD, True) is True


def test_the_virtual_brick_is_the_real_driver_over_a_memory_library() -> None:
    library = InMemoryLmsLibrary((7,))
    source = VaunixLmsRfSource(VaunixLmsConfig(serial=7), library=library)
    assert type(source) is VaunixLmsRfSource
    source.tune(FREQUENCY_FIELD, 2.5e9)
    assert library.get_frequency(7) == 250_000_000  # the DLL's 10 Hz units
    source.close()
    with pytest.raises(RuntimeError, match="not open"):
        library.get_frequency(7)


def test_a_missing_brick_is_a_named_lookup_error() -> None:
    with pytest.raises(LookupError, match="serial 42"):
        VaunixLmsRfSource(
            VaunixLmsConfig(serial=42), library=InMemoryLmsLibrary((7,))
        )
