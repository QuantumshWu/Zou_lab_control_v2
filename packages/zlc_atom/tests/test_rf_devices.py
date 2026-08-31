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
    """A DG4000's worth of SCPI, answered from a register dict."""

    def __init__(self) -> None:
        self.registers = {"FREQ": 1000.0, "VOLT": -30.0, "OUT": "OFF"}
        self.log: list[str] = []

    def write(self, command: str) -> None:
        self.log.append(command)
        upper = command.upper()
        if ":FREQUENCY " in upper:
            self.registers["FREQ"] = float(command.split()[-1])
        elif ":VOLTAGE " in upper and ":UNIT" not in upper:
            self.registers["VOLT"] = float(command.split()[-1])
        elif upper.startswith(":OUTPUT"):
            self.registers["OUT"] = command.split()[-1].upper()

    def query(self, command: str) -> str:
        self.log.append(command)
        upper = command.upper()
        if upper == "*IDN?":
            return "RIGOL TECHNOLOGIES,DG4162,DG4E0000000001,00.01.12"
        if ":FREQUENCY?" in upper:
            return f"{self.registers['FREQ']:.6E}"
        if ":VOLTAGE?" in upper:
            return f"{self.registers['VOLT']:.4E}"
        if upper.startswith(":OUTPUT"):
            return self.registers["OUT"]
        raise AssertionError(f"unexpected query {command!r}")

    def close(self) -> None:
        self.log.append("<closed>")


def _rigol(**overrides) -> tuple[RigolDg4000RfSource, _ScpiInstrument]:
    instrument = _ScpiInstrument()
    config = RigolDg4000Config(resource="TCPIP0::198.51.100.7::INSTR", **overrides)
    return RigolDg4000RfSource(config, link=instrument), instrument


def test_the_rigol_pins_dbm_once_and_returns_what_it_read_back() -> None:
    source, instrument = _rigol()
    assert ":SOURce1:VOLTage:UNIT DBM" in instrument.log

    effective = source.tune(FREQUENCY_FIELD, 80e6)
    assert effective == 80e6
    assert instrument.registers["FREQ"] == 80e6
    assert source.tune(POWER_FIELD, -3.0) == -3.0
    assert source.tune(OUTPUT_FIELD, True) is True
    assert source.tunable_values() == {
        FREQUENCY_FIELD: 80e6,
        POWER_FIELD: -3.0,
        OUTPUT_FIELD: True,
    }


def test_bounds_are_bench_policy_and_refuse_before_writing() -> None:
    source, instrument = _rigol(frequency_high_hz=1e6)
    written = dict(instrument.registers)
    with pytest.raises(ValueError, match="frequency_hz must lie in"):
        source.tune(FREQUENCY_FIELD, 2e6)
    with pytest.raises(ValueError, match="power_dbm must lie in"):
        source.tune(POWER_FIELD, 99.0)
    with pytest.raises(TypeError, match="output_enabled takes a bool"):
        source.tune(OUTPUT_FIELD, 1)
    with pytest.raises(ValueError, match="no tunable field"):
        source.tune("phase_deg", 0.0)
    assert instrument.registers == written, "a refusal must not touch hardware"


def test_every_accepted_tune_advances_the_settings_epoch() -> None:
    source, _instrument = _rigol()
    first = source.settings_provenance()
    source.tune(FREQUENCY_FIELD, 10e6)
    second = source.settings_provenance()
    assert second["settings_epoch"] == first["settings_epoch"] + 1
    assert second["device_session_id"] == first["device_session_id"]
    assert "DG4162" in str(second["device_session_id"])


def test_the_scan_facing_fields_carry_bounds_and_units() -> None:
    source, _instrument = _rigol()
    by_name = {field.metadata.name: field for field in source.tunable_fields()}
    frequency = by_name[FREQUENCY_FIELD].metadata
    assert (frequency.minimum, frequency.maximum) == (1e3, 160e6)
    assert frequency.unit == "Hz"
    assert by_name[POWER_FIELD].metadata.unit == "dBm"
    # The output switch is a control, not an axis: unbounded on purpose, so
    # scan_ports_for_devices never offers it.
    output = by_name[OUTPUT_FIELD].metadata
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
