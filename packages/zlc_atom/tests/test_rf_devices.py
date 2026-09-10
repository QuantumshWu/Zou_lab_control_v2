"""The RF source contract, driven over faked transports.

Per the bench's one virtualization rule the fake is the lowest layer only:
these tests run the REAL drivers -- their SCPI vocabulary, their unit
conversions, their grid refusals, their read-back discipline -- over a link
or library that answers from memory.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from zlc_atom.devices.rf.contract import (
    FREQUENCY_FIELD,
    OUTPUT_FIELD,
    POWER_FIELD,
)
from zlc_atom.devices.rf.rigol_dg4000 import RigolDg4000Config, RigolDg4000RfSource
from zlc_atom.devices.rf.vaunix_lms import (
    CtypesLmsLibrary,
    VaunixLmsConfig,
    VaunixLmsRfSource,
)
from zlc_atom.devices.simulation.rf import InMemoryLmsLibrary, virtual_rf_source


class _ScpiInstrument:
    """A DG4000's worth of SCPI, answered from per-channel register dicts.

    The amplitude cap steps down with frequency the way a DG4162's does
    (10 Vpp to 20 MHz, 5 Vpp to 60 MHz, 2.5 Vpp above), and a frequency
    write lowers a standing amplitude the new frequency cannot carry --
    the instrument's own behaviour, not the driver's.
    """

    _AMPLITUDE_CAPS_VPP = ((20e6, 10.0), (60e6, 5.0), (math.inf, 2.5))

    def __init__(self) -> None:
        # A channel's amplitude UNIT, output LOAD and WAVEFORM are settings
        # of the instrument like any other: the driver reads them, and a
        # test that could not spell them could not tell whether it wrote
        # them.
        self.registers = {
            channel: {
                "FREQ": 1000.0,
                "VOLT": -30.0,
                "OUT": "OFF",
                "UNIT": "DBM",
                "LOAD": 50.0,
                "FUNC": "SIN",
            }
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

    @classmethod
    def _amplitude_cap(cls, registers: dict) -> float:
        """The most this channel may output at its frequency, in its unit."""

        vpp = next(
            cap for ceiling, cap in cls._AMPLITUDE_CAPS_VPP if registers["FREQ"] <= ceiling
        )
        if registers["UNIT"] == "VPP":
            return vpp
        vrms = vpp / (2.0 * math.sqrt(2.0))
        if registers["UNIT"] == "VRMS":
            return vrms
        # dBm into the channel's load, to the two decimals the panel shows.
        return round(10.0 * math.log10(vrms * vrms / registers["LOAD"] / 1e-3), 2)

    def write(self, command: str) -> None:
        self.log.append(command)
        upper = command.upper()
        registers = self.registers[self._channel(command)]
        if ":FREQUENCY " in upper:
            registers["FREQ"] = float(command.split()[-1])
            registers["VOLT"] = min(registers["VOLT"], self._amplitude_cap(registers))
        elif ":VOLTAGE:UNIT " in upper:
            unit = command.split()[-1].upper()
            registers["VOLT"] = self._convert_amplitude(registers["VOLT"], registers["UNIT"], unit, registers)
            registers["UNIT"] = unit
        elif ":VOLTAGE " in upper:
            value = command.split()[-1]
            for unit in ("VPP", "VRMS", "DBM"):
                if value.upper().endswith(unit):
                    registers["VOLT"] = self._convert_amplitude(float(value[:-len(unit)]), unit, registers["UNIT"], registers)
                    break
            else:
                registers["VOLT"] = float(value)
        elif ":IMPEDANCE " in upper:
            tail = command.split()[-1].upper()
            registers["LOAD"] = (
                float("inf") if tail.startswith("INF") else float(tail)
            )
        elif ":FUNCTION " in upper:
            registers["FUNC"] = command.split()[-1].upper()
        elif upper.startswith(":OUTPUT"):
            registers["OUT"] = command.split()[-1].upper()

    @staticmethod
    def _convert_amplitude(value, source, target, registers):
        if source == target:
            return value
        ratio = 2.0 * math.sqrt(2.0) if registers["FUNC"] == "SIN" else 2.0
        rms = (math.sqrt(1e-3 * 10.0 ** (value / 10.0) * registers["LOAD"])
               if source == "DBM" else value / ratio if source == "VPP" else value)
        return (10.0 * math.log10(rms * rms / registers["LOAD"] / 1e-3)
                if target == "DBM" else rms * ratio if target == "VPP" else rms)

    def query(self, command: str) -> str:
        self.log.append(command)
        upper = command.upper()
        if upper == "*IDN?":
            return "RIGOL TECHNOLOGIES,DG4162,DG4E0000000001,00.01.12"
        registers = self.registers[self._channel(command)]
        # The instrument's own limits, as a DG4162 answers them.
        if ":FREQUENCY? MIN" in upper:
            return "1.000000E-06"
        if ":FREQUENCY? MAX" in upper:
            return "1.600000E+08"
        if ":VOLTAGE? MIN" in upper:
            return "-6.0000E+01" if registers["UNIT"] == "DBM" else "1.0000E-03"
        if ":VOLTAGE? MAX" in upper:
            return f"{self._amplitude_cap(registers):.4E}"
        if ":FREQUENCY?" in upper:
            return f"{registers['FREQ']:.6E}"
        if ":VOLTAGE:UNIT?" in upper:
            return registers["UNIT"]
        if ":FUNCTION?" in upper:
            return registers["FUNC"]
        if ":VOLTAGE?" in upper:
            # Echo the memory register without adding test-only quantization.
            return f"{registers['VOLT']:.17g}"
        if ":IMPEDANCE?" in upper:
            load = registers["LOAD"]
            return "INF" if load == float("inf") else f"{load:.4E}"
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
    other.
    """

    source, instrument = _rigol()

    names = [field.metadata.name for field in source.tunable_fields()]
    assert names == [
        "ch1_frequency",
        "ch1_power",
        "ch1_output_enabled",
        "ch2_frequency",
        "ch2_power",
        "ch2_output_enabled",
        "frequency_low",
        "frequency_high",
        "power_low",
        "power_high",
    ]

    assert source.tune("ch1_frequency", 80e6) == 80e6
    assert source.tune("ch2_frequency", 5e6) == 5e6
    assert instrument.registers["1"]["FREQ"] == 80e6
    assert instrument.registers["2"]["FREQ"] == 5e6
    assert source.tune("ch2_output_enabled", True) is True
    assert instrument.registers["1"]["OUT"] == "OFF", (
        "tuning one channel must not move the other"
    )
    values = source.tunable_values()
    assert values["ch1_frequency"] == 80e6
    assert values["ch2_frequency"] == 5e6
    assert values["ch2_output_enabled"] is True


def test_bounds_are_bench_policy_and_refuse_before_writing() -> None:
    source, instrument = _rigol(frequency_high_hz=1e6, power_high_dbm=10.0)
    written = {
        channel: dict(registers)
        for channel, registers in instrument.registers.items()
    }
    with pytest.raises(ValueError, match="ch2_frequency must lie in"):
        source.tune("ch2_frequency", 2e6)
    with pytest.raises(ValueError, match="ch1_power must lie in"):
        source.tune("ch1_power", 99.0)
    with pytest.raises(TypeError, match="ch1_output_enabled takes a bool"):
        source.tune("ch1_output_enabled", 1)
    with pytest.raises(ValueError, match="no tunable field"):
        source.tune("frequency", 1e5)
    assert instrument.registers == written, "a refusal must not touch hardware"


def test_only_an_effective_change_advances_the_settings_epoch() -> None:
    """The epoch counts changes of the instrument's state, and the session
    id names THIS connection.

    A tune to the value a knob already stood at advanced the epoch, so a
    control panel re-projected and a settings record grew a revision for a
    setting that had not moved.  And the session id was the ``*IDN?``
    string: a generator closed and reopened with its knobs elsewhere came
    back under the same id, so nothing downstream could tell a risk
    acceptance bound to the first session from one bound to the second.
    """

    source, _instrument = _rigol()
    first = source.settings_provenance()
    assert source.tune("ch1_frequency", 10e6) == 10e6
    second = source.settings_provenance()
    assert second["settings_epoch"] == first["settings_epoch"] + 1
    assert second["device_session_id"] == first["device_session_id"]

    assert source.tune("ch1_frequency", 10e6) == 10e6
    assert source.settings_provenance() == second, (
        "landing where the knob already stands is not a settings change"
    )
    assert source.tune("ch1_output_enabled", False) is False
    assert source.settings_provenance() == second
    assert source.tune("ch1_output_enabled", True) is True
    assert source.settings_provenance()["settings_epoch"] == (
        second["settings_epoch"] + 1
    )

    # The instrument's name is a label; the session is this connection.
    assert "DG4162" in source.identity
    assert "DG4162" not in str(second["device_session_id"])
    source.close()
    reopened, _instrument = _rigol()
    assert (
        reopened.settings_provenance()["device_session_id"]
        != second["device_session_id"]
    )
    assert reopened.settings_provenance()["settings_epoch"] == 0


def test_the_scan_facing_fields_carry_bounds_and_units() -> None:
    source, _instrument = _rigol(
        frequency_low_hz=1e3,
        frequency_high_hz=160e6,
        power_low_dbm=-30.0,
        power_high_dbm=10.0,
    )
    by_name = {field.metadata.name: field for field in source.tunable_fields()}
    for channel in ("ch1", "ch2"):
        frequency = by_name[f"{channel}_frequency"].metadata
        assert (frequency.minimum, frequency.maximum) == (1e3, 160e6)
        assert frequency.unit == "Hz"
        assert frequency.label.startswith(channel.upper())
        assert by_name[f"{channel}_power"].metadata.unit == "dBm"
        # The instrument's own limits ride beside the effective bounds, so
        # a panel can show which fence is the binding one: here the bench
        # authored the low edge and the instrument owns the high one.
        assert by_name[f"{channel}_frequency"].device_limits == (1e-6, 160e6)
        assert by_name[f"{channel}_power"].device_limits == (-60.0, 23.98)
        # The output switch is a control, not an axis: unbounded on purpose,
        # so scan_ports_for_devices never offers it.
        output = by_name[f"{channel}_output_enabled"]
        assert output.metadata.minimum is None and output.metadata.maximum is None
        assert output.device_limits is None
        assert by_name[f"{channel}_frequency"].live_write
    for name in ("frequency_low", "frequency_high", "power_low", "power_high"):
        assert by_name[name].device_limits is None, "a policy edge has no instrument limit"
    for field in by_name.values():
        assert field.dependency_group == (field.metadata.name,)


def test_optional_window_is_one_init_and_control_policy() -> None:
    """Two fences, one range: the instrument's limits and the bench's window.

    The window is adjusted on the control panel with plain Apply (never
    live), or through the same ``tune`` API -- and the scan add-axis combo
    must never offer it: the window fields are non-live and unbounded,
    which is exactly what scan_ports_for_devices excludes.  The knobs
    themselves are always offered, over the TIGHTER of the instrument's
    own limits and the window on each side -- the one range the panel, an
    external ``tune`` and a scan all obey -- so a knob is sweepable as soon
    as the instrument states a range, and an edge nobody authored forbids
    nothing.  Tightening an edge past a channel's CURRENT value is refused
    by name: policy may fence a knob in, never silently drag a set output
    to a new value; so is a window that leaves nothing of the instrument's
    range, because no knob position could ever satisfy it.
    """

    from zlc_atom.nodes.scan.plan import scan_ports_for_devices

    source, instrument = _rigol()
    by_name = {field.metadata.name: field for field in source.tunable_fields()}
    for name in (
        "frequency_low",
        "frequency_high",
        "power_low",
        "power_high",
    ):
        window = by_name[name]
        assert not window.live_write, "the window applies, never live"
        assert window.metadata.minimum is None and window.metadata.maximum is None
        assert window.current is None

    ports = {port.port.split(":")[-1]: port for port in scan_ports_for_devices({"rf": source})}
    assert set(ports) == {
        "ch1_frequency", "ch1_power", "ch2_frequency", "ch2_power"
    }, "with no window, a knob is swept over the instrument's own range"
    assert (ports["ch1_frequency"].lo, ports["ch1_frequency"].hi) == (1e-6, 160e6)
    assert (ports["ch1_power"].lo, ports["ch1_power"].hi) == (-60.0, 23.98)
    assert not any("FREQuency " in command for command in instrument.log), (
        "omitting all policy edges must not move hardware at open"
    )
    # With no window at all, the instrument's own limit fences direct
    # control too -- refused by name before anything is written.
    with pytest.raises(ValueError, match="ch1_frequency must lie in"):
        source.tune("ch1_frequency", 200e6)
    assert instrument.registers["1"]["FREQ"] == 1000.0

    assert source.tune("frequency_low", 1e3) == 1e3
    assert source.tune("frequency_high", 80e6) == 80e6
    assert source.tune("power_low", -30.0) == -30.0
    assert source.tune("power_high", 10.0) == 10.0
    ports = scan_ports_for_devices({"rf": source})
    offered = {port.port.split(":")[-1] for port in ports}
    assert offered == {
        "ch1_frequency", "ch1_power", "ch2_frequency", "ch2_power"
    }

    before = source.settings_provenance()["settings_epoch"]
    assert source.tune("frequency_high", None) is None
    assert source.tunable_values()["frequency_high"] is None
    assert source.settings_provenance()["settings_epoch"] == before + 1
    cleared = next(
        port for port in scan_ports_for_devices({"rf": source})
        if port.port.endswith(":ch1_frequency")
    )
    assert (cleared.lo, cleared.hi) == (1e3, 160e6), (
        "clearing an edge hands that side back to the instrument's own limit"
    )
    # A window looser than the instrument on one side widens nothing: the
    # tighter fence is the range, whichever of the two it is.
    assert source.tune("frequency_high", 200e6) == 200e6
    loose = next(
        port for port in scan_ports_for_devices({"rf": source})
        if port.port.endswith(":ch1_frequency")
    )
    assert (loose.lo, loose.hi) == (1e3, 160e6)
    with pytest.raises(ValueError, match="ch1_frequency must lie in"):
        source.tune("ch1_frequency", 170e6)
    assert source.tune("frequency_high", 80e6) == 80e6
    # The channel knobs' own scan bounds follow the window immediately.
    by_name = {field.metadata.name: field for field in source.tunable_fields()}
    assert by_name["ch1_frequency"].metadata.maximum == 80e6
    assert by_name["ch1_frequency"].device_limits == (1e-6, 160e6)

    source.tune("ch1_frequency", 50e6)
    with pytest.raises(ValueError, match="strand ch1_frequency at 5e"):
        source.tune("frequency_high", 20e6)
    with pytest.raises(ValueError, match="empty window"):
        source.tune("frequency_low", 90e6)

    # A window entirely outside the instrument's range is a contradiction,
    # refused on the panel and at Init alike -- and Init opens no session
    # for it.
    fresh, _instrument = _rigol()
    with pytest.raises(ValueError, match="leaves nothing of the instrument"):
        fresh.tune("frequency_low", 200e6)
    assert fresh.tunable_values()["frequency_low"] is None
    refused = _ScpiInstrument()
    with pytest.raises(ValueError, match="leaves nothing of the instrument"):
        RigolDg4000RfSource(
            RigolDg4000Config(
                resource="TCPIP0::198.51.100.7::INSTR", frequency_low_hz=200e6
            ),
            link=refused,
        )
    assert refused.log[-1] == "<closed>"


def test_connecting_reads_the_instrument_and_moves_nothing() -> None:
    """Plugging in is an act of observation, not of control.

    Whatever the instrument was doing when the operator connected IS the
    experiment's state.  Opening used to drag a knob that idled outside
    the authored bench window to the nearest edge -- silently, on both
    channels, for frequency and power -- which is the same act this class
    refuses by name when a window is TIGHTENED past a live knob.  A knob
    outside policy is a reading to show, and the panel is where the
    operator finds out.
    """

    instrument = _ScpiInstrument()
    instrument.registers["1"]["FREQ"] = 90e6      # above the window below
    instrument.registers["2"]["VOLT"] = 25.0      # above the window below
    before = {
        channel: dict(registers)
        for channel, registers in instrument.registers.items()
    }
    source = RigolDg4000RfSource(
        RigolDg4000Config(
            resource="TCPIP0::198.51.100.7::INSTR",
            frequency_low_hz=1e3,
            frequency_high_hz=80e6,
            power_low_dbm=-30.0,
            power_high_dbm=10.0,
        ),
        link=instrument,
    )
    assert instrument.registers == before, "connecting moved the instrument"
    # Questions only: the instrument's limits are read at connect, and a
    # read is not a write.
    assert not any(
        command.startswith(":SOURce") and "?" not in command
        for command in instrument.log
    ), instrument.log

    values = source.tunable_values()
    assert values["ch1_frequency"] == 90e6
    assert values["ch2_power"] == 25.0
    by_name = {field.metadata.name: field for field in source.tunable_fields()}
    assert by_name["ch1_frequency"].current == 90e6
    assert by_name["ch1_frequency"].metadata.maximum == 80e6

    # Policy still bounds what may be COMMANDED, from outside as much as in.
    with pytest.raises(ValueError, match="ch1_frequency must lie in"):
        source.tune("ch1_frequency", 85e6)
    assert instrument.registers["1"]["FREQ"] == 90e6
    assert source.tune("ch1_frequency", 50e6) == 50e6


def test_a_channel_in_volts_is_converted_through_its_own_load() -> None:
    """The unit on the front panel is the operator's, so the driver adapts.

    Pinning DBM at open made every later read and write mean dBm by
    force, at the price of changing a setting nobody asked to change --
    and on a high-Z channel that pin is not even legal.  The unit is read
    instead, and volts are converted through the load the channel states
    it is driving.
    """

    instrument = _ScpiInstrument()
    instrument.registers["1"]["UNIT"] = "VPP"
    instrument.registers["1"]["LOAD"] = 50.0
    instrument.registers["1"]["VOLT"] = 0.632456   # 0 dBm into 50 ohms
    source, _instrument = (
        RigolDg4000RfSource(
            RigolDg4000Config(resource="TCPIP0::198.51.100.7::INSTR"),
            link=instrument,
        ),
        instrument,
    )
    assert instrument.registers["1"]["UNIT"] == "VPP", "opening changed a unit"
    assert source.tunable_values()["ch1_power"] == pytest.approx(0.0, abs=1e-3)

    # A written dBm lands as the volts THIS channel is displaying, and the
    # read-back that follows converts straight back.
    assert source.tune("ch1_power", -6.0) == pytest.approx(-6.0, abs=1e-3)
    assert instrument.registers["1"]["UNIT"] == "VPP"
    assert instrument.registers["1"]["VOLT"] == pytest.approx(0.3170, abs=1e-3)

    # The other channel is untouched, in its own unit.
    assert instrument.registers["2"]["UNIT"] == "DBM"
    assert source.tunable_values()["ch2_power"] == -30.0

    # The reported failing value and a 135 -> 247 mVpp, ten-point sweep
    # must survive SCPI serialization. Device readback remains the answer.
    from zlc_atom.nodes.scan.devices import tune_value

    points = [0.135 + index * (0.247 - 0.135) / 9 for index in range(10)]
    requested_values = [-12.83089524390248] + [
        10.0 * math.log10((vpp / (2.0 * math.sqrt(2.0))) ** 2 / 50.0 / 1e-3)
        for vpp in points
    ]
    for requested in requested_values:
        actual = tune_value(source, "ch1_power", requested)
        sent = next(command for command in reversed(instrument.log)
                    if command.startswith(":SOURce1:VOLTage "))
        expected_vpp = math.sqrt(1e-3 * 10.0 ** (requested / 10.0) * 50.0) * (2.0 * math.sqrt(2.0))
        assert float(sent.split()[-1].removesuffix("VPP")) == expected_vpp
        assert actual == source.tunable_values()["ch1_power"]
    source.close()


def test_selected_units_write_native_amplitude_and_restore_the_raw_pair() -> None:
    from zlc_atom.authoring import read_tunable_in_unit, tune_in_unit
    from zlc_atom.nodes.scan.devices import ScanDeviceKnobs

    field = "ch1_power"
    port = "device:rf:" + field
    for old_unit, old_value, symbol in (
        ("DBM", -20.123456789, "dBm"),
        ("VRMS", 0.019123456789123, "Vrms"),
        ("VPP", 0.055123456789123, "Vpp"),
    ):
        instrument = _ScpiInstrument()
        instrument.registers["1"].update(UNIT=old_unit, VOLT=old_value, LOAD=75.0)
        source = RigolDg4000RfSource(
            RigolDg4000Config(resource="memory", power_low_dbm=-40, power_high_dbm=-10),
            link=instrument,
        )
        try:
            standing = read_tunable_in_unit(source, field)
            assert standing.current == old_value and standing.metadata.unit == symbol
            selected = read_tunable_in_unit(source, field, "mVpp")
            assert selected.metadata.maximum == pytest.approx(math.sqrt(8 * 75 * 1e-4) * 1000)
            assert selected.metadata.label.endswith("Power")
            assert read_tunable_in_unit(source, "ch1_frequency", "kHz").metadata.label.endswith("Frequency")
            assert not any(":UNIT " in item for item in instrument.log)
            before = {key: dict(value) for key, value in instrument.registers.items()}
            provenance = source.settings_provenance()
            writes = [item for item in instrument.log if "?" not in item]
            assert source.convert_tunable_value(field, 0.0, "dBm", "mVpp") == pytest.approx(math.sqrt(8 * 75 * 1e-3) * 1000)
            assert source.convert_tunable_value(field, (0.0, -10.0), "dBm", "Vpp") == pytest.approx((math.sqrt(8 * 75 * 1e-3), math.sqrt(8 * 75 * 1e-4)))
            assert source.convert_tunable_value(field, [135.0, 247.0], "mVpp", "dBm") == pytest.approx(tuple(
                10 * math.log10((value / 1000) ** 2 / (8 * 75) / 1e-3) for value in (135.0, 247.0)
            ))
            assert instrument.registers == before and source.settings_provenance() == provenance
            assert [item for item in instrument.log if "?" not in item] == writes
            # Policy edges cover both channels and have no unique load from
            # which to define a voltage; actual channel fields above do.
            for policy in ("power_low", "power_high"):
                for voltage_unit in ("mVpp", "Vrms"):
                    with pytest.raises(ValueError, match="no single channel load"):
                        source.read_tunable_in_unit(policy, voltage_unit)
                    with pytest.raises(ValueError, match="no single channel load"):
                        source.convert_tunable_value(policy, -20.0, "dBm", voltage_unit)
                    with pytest.raises(ValueError, match="no single channel load"):
                        source.tune_in_unit(policy, 135.0, voltage_unit)
            assert source.read_tunable_in_unit("power_high", "mW").current == pytest.approx(.1)
            assert source.tune_in_unit("power_high", .1, "mW") == pytest.approx(.1)
            assert source.tune_in_unit("power_low", None, "mVpp") is None
            source.tune("power_low", -40.0)
            assert [item for item in instrument.log if "?" not in item] == writes
            # The actual instrument may quantize one amplitude. Keep its
            # answer rather than comparing it to the authored coordinate.
            write = instrument.write
            quantized = False
            def first_write_quantized(command):
                nonlocal quantized
                write(command)
                if not quantized and command.startswith(":SOURce1:VOLTage ") and command.endswith("VPP"):
                    instrument.registers["1"]["VOLT"] += 3e-9
                    quantized = True
            instrument.write = first_write_quantized
            knobs = ScanDeviceKnobs({"rf": source})
            instrument.log.clear()
            actual = knobs.move(port, 135.0, "mVpp")
            assert [item for item in instrument.log if "?" in item] == [":SOURce1:VOLTage?"]
            assert actual != 135.0 and actual == instrument.registers["1"]["VOLT"] / 0.001
            sent = next(item for item in instrument.log if item.startswith(":SOURce1:VOLTage ") and item.endswith("VPP"))
            assert float(sent.split()[-1][:-3]) == 135.0 / 1000
            before_move = len(instrument.log)
            knobs.move(port, 220.0, "mVpp")  # legal at 75 ohm, above the 50-ohm policy edge
            assert instrument.log[before_move:] == [":SOURce1:VOLTage 0.22VPP", ":SOURce1:VOLTage?"]
            knobs.restore()
            assert instrument.registers["1"]["UNIT"] == old_unit
            assert instrument.registers["1"]["VOLT"] == old_value
            mode_writes = [item for item in instrument.log if ":UNIT " in item]
            assert mode_writes == ([] if old_unit == "VPP" else [
                ":SOURce1:VOLTage:UNIT VPP", f":SOURce1:VOLTage:UNIT {old_unit}",
            ])
            instrument.log.clear()
            tune_in_unit(source, field, 0.05, "mW")
            assert instrument.registers["1"]["UNIT"] == old_unit
            assert not any(":UNIT " in item for item in instrument.log)
        finally:
            source.close()

    source, instrument = _rigol()
    original = instrument.registers["1"]["VOLT"]
    write = instrument.write
    try:
        for stage in ("unit", "amplitude"):
            failed = False
            def refused(command):
                nonlocal failed
                write(command)
                selected = (command.endswith(":VOLTage:UNIT VPP") if stage == "unit"
                            else ":VOLTage " in command and command.endswith("VPP"))
                if not failed and selected:
                    failed = True
                    raise RuntimeError("instrument write failed")
            instrument.write = refused
            with pytest.raises(RuntimeError, match="instrument write failed"):
                tune_in_unit(source, field, 135.0, "mVpp")
            assert instrument.registers["1"]["UNIT"] == "DBM"
            assert instrument.registers["1"]["VOLT"] == original
            # A successful rollback write is not a confirmed readback.
            before_projection = len(instrument.log)
            assert source.tunable_fields()[1].current is None
            assert source.read_tunable_in_unit(field, "mVpp").current is None
            assert len(instrument.log) == before_projection
            source.refresh_tunable_fields()
        def restore_refused(command):
            if command.endswith(":VOLTage:UNIT DBM"):
                raise RuntimeError("unit restore failed")
            if ":VOLTage " in command and command.endswith("DBM"):
                raise RuntimeError("amplitude restore failed")
            write(command)
            if ":VOLTage " in command and command.endswith("VPP"):
                raise RuntimeError("primary write failed")
        instrument.write = restore_refused
        with pytest.raises(RuntimeError, match="primary write failed") as failure:
            tune_in_unit(source, field, 135.0, "mVpp")
        assert any("unit restore failed" in note for note in failure.value.__notes__)
        assert any("amplitude restore failed" in note for note in failure.value.__notes__)
        before_projection = len(instrument.log)
        assert source.tunable_fields()[1].current is None
        assert source.read_tunable_in_unit(field, "mVpp").current is None
        assert len(instrument.log) == before_projection
        instrument.write = write
        instrument.log.clear()
        assert tune_in_unit(source, field, 220.0, "mVpp") == pytest.approx(220.0)
        assert instrument.log == [":SOURce1:VOLTage:UNIT VPP",
                                  ":SOURce1:VOLTage 0.22VPP", ":SOURce1:VOLTage?"]
    finally:
        source.close()


def test_volts_into_a_high_z_load_is_named_not_guessed() -> None:
    """Delivered power is not defined there, so no number is offered.

    The instrument's power range is read when the connection opens, so a
    channel whose power cannot be stated in dBm is refused at connect --
    with the way out named, and the session released -- rather than as a
    generator whose every panel and scan later fails on the same read.
    """

    instrument = _ScpiInstrument()
    instrument.registers["1"]["UNIT"] = "VPP"
    instrument.registers["1"]["LOAD"] = float("inf")
    with pytest.raises(RuntimeError, match="high-Z load"):
        RigolDg4000RfSource(
            RigolDg4000Config(resource="TCPIP0::198.51.100.7::INSTR"),
            link=instrument,
        )
    assert instrument.log[-1] == "<closed>", "a refused connection is released"
    assert instrument.registers["1"]["VOLT"] == -30.0, "a refusal wrote nothing"


def test_peak_to_peak_volts_are_converted_only_for_a_sine() -> None:
    """Vpp over RMS is a property of the waveform, and only a sine's is known.

    A channel left on a square wave by the previous experiment, in Vpp into
    50 ohms, was read through the sine ratio: 2 Vpp reported as 10 dBm where
    a symmetric square wave delivers 13 dBm.  The driver asks the waveform
    and refuses a shape it cannot convert, naming the way out; RMS and dBm
    channels need no shape and are unaffected.
    """

    instrument = _ScpiInstrument()
    instrument.registers["1"].update(UNIT="VPP", VOLT=2.0, LOAD=50.0, FUNC="SQU")
    with pytest.raises(RuntimeError, match="SQU waveform"):
        RigolDg4000RfSource(
            RigolDg4000Config(resource="TCPIP0::198.51.100.7::INSTR"),
            link=instrument,
        )
    assert instrument.log[-1] == "<closed>"

    instrument = _ScpiInstrument()
    instrument.registers["1"].update(UNIT="VRMS", VOLT=1.0, LOAD=50.0, FUNC="SQU")
    source = RigolDg4000RfSource(
        RigolDg4000Config(resource="TCPIP0::198.51.100.7::INSTR"),
        link=instrument,
    )
    # 1 Vrms into 50 ohms is 20 mW whatever the shape.
    assert source.tunable_values()["ch1_power"] == pytest.approx(13.0103, abs=1e-3)

    # A sine in Vpp converts through 2*sqrt(2); a shape changed under a
    # live connection is caught at the next read or write, and the write
    # never happens.
    instrument.registers["1"].update(UNIT="VPP", VOLT=0.632456, FUNC="SIN")
    assert source.tunable_values()["ch1_power"] == pytest.approx(0.0, abs=1e-3)
    instrument.registers["1"]["FUNC"] = "RAMP"
    with pytest.raises(RuntimeError, match="RAMP waveform"):
        source.refresh_tunable_fields()
    assert instrument.registers["1"]["VOLT"] == 0.632456, "a refusal wrote nothing"


def test_frequency_apply_reads_only_frequency_and_invalidates_amplitude() -> None:
    """Native amplitude limiting never undoes an authored frequency write."""

    source, instrument = _rigol()
    assert source.tune("ch1_power", 20.0) == 20.0
    provenance = source.settings_provenance()
    instrument.log.clear()
    assert source.tune_in_unit("ch1_frequency", 50000.0, "kHz") == 50000.0
    assert instrument.log == [":SOURce1:FREQuency 50000000", ":SOURce1:FREQuency?"]
    assert instrument.registers["1"]["FREQ"] == 50e6
    assert instrument.registers["1"]["VOLT"] == 17.96
    assert source.settings_provenance()["settings_epoch"] == provenance["settings_epoch"] + 1
    fields = {field.metadata.name: field for field in source.tunable_fields()}
    assert fields["ch1_power"].current is None
    assert fields["ch1_power"].device_limits is None
    displayed = source.read_tunable_in_unit("ch1_power", "mVpp")
    assert displayed.current is None and displayed.metadata.unit == "mVpp"
    assert len(instrument.log) == 2, "metadata and epoch do not query the instrument"
    source.refresh_tunable_fields()
    assert source.tunable_fields()[1].current == 17.96

    # Under the cap a frequency write is an ordinary write, and the
    # amplitude stays where it was set.
    assert source.tune("ch1_power", 10.0) == 10.0
    assert source.tune("ch1_frequency", 50e6) == 50e6
    assert instrument.registers["1"]["VOLT"] == 10.0
    assert source.tunable_values()["ch1_power"] == 10.0

    query = instrument.query
    def lost_frequency_readback(command):
        if command == ":SOURce1:FREQuency?":
            raise TimeoutError("frequency readback lost")
        return query(command)
    instrument.query = lost_frequency_readback
    with pytest.raises(TimeoutError, match="frequency readback lost"):
        source.tune("ch1_frequency", 60e6)
    assert instrument.registers["1"]["FREQ"] == 60e6
    before_projection = len(instrument.log)
    assert source.tunable_fields()[0].current is None
    assert source.read_tunable_in_unit("ch1_frequency", "kHz").current is None
    assert len(instrument.log) == before_projection
    instrument.query = query
    source.refresh_tunable_fields()
    assert source.tunable_fields()[0].current == 60e6


def test_what_a_constructor_acquired_the_constructor_releases(monkeypatch) -> None:
    """A source that fails to build has no owner but itself.

    The Rigol opened its VISA session and then failed on ``*IDN?``; the
    Lab Brick opened its USB handle and then failed on a limits read.
    Neither closed what it had opened, and no leaf ever existed to retry
    the close through.  A window the driver cannot honour is refused
    before either transport is opened at all.
    """

    import zlc_atom.devices.rf.rigol_dg4000 as module

    opened: list[str] = []
    instrument = _ScpiInstrument()

    def no_identity(command: str) -> str:
        instrument.log.append(command)
        raise TimeoutError("no answer")

    instrument.query = no_identity
    manager = SimpleNamespace(
        open_resource=lambda resource: opened.append(resource) or instrument
    )
    monkeypatch.setattr(module, "visa_resources", lambda: manager)
    with pytest.raises(TimeoutError, match="no answer"):
        RigolDg4000RfSource(RigolDg4000Config(resource="TCPIP0::198.51.100.7::INSTR"))
    assert opened == ["TCPIP0::198.51.100.7::INSTR"]
    assert instrument.log[-1] == "<closed>", "the session the constructor opened"

    opened.clear()
    with pytest.raises(ValueError, match="frequency bounds must be ordered"):
        RigolDg4000RfSource(
            RigolDg4000Config(
                resource="TCPIP0::198.51.100.7::INSTR",
                frequency_low_hz=2e9,
                frequency_high_hz=1e9,
            )
        )
    assert opened == [], "a window the driver cannot honour opens no session"

    class _LimitsRefused(InMemoryLmsLibrary):
        def get_power_limits(self, handle: int) -> tuple[int, int]:
            raise RuntimeError("firmware refused")

    library = _LimitsRefused((77,))
    with pytest.raises(RuntimeError, match="firmware refused"):
        VaunixLmsRfSource(VaunixLmsConfig(serial=77), library=library)
    with pytest.raises(RuntimeError, match="not open"):
        library.get_frequency(77)

    untouched = InMemoryLmsLibrary((77,))
    with pytest.raises(ValueError, match="frequency bounds must be ordered"):
        VaunixLmsRfSource(
            VaunixLmsConfig(serial=77, frequency_low_hz=2e9, frequency_high_hz=1e9),
            library=untouched,
        )
    assert untouched._open == set(), "a bad window opens no handle"


def test_a_brick_the_sdk_refuses_to_close_stays_owned() -> None:
    """A close status is an answer, and a non-zero one means "still open".

    ``fnLMS_CloseDevice`` returns a status like every other SDK call.  It
    was discarded, so the installation heard "closed", dropped the leaf and
    unbound the brick -- while the SDK still held the handle and nothing
    was left that could retry.  The status is raised now, and the leaf
    stays owned so ``close`` can be tried again.
    """

    from zlc_atom.devices.rf.binding import bind_rf_source
    from zlc_atom.execution import DeviceBroker
    from zlc_atom.install import Installation, InstallationFactoryContext

    physical = {"open": False, "close_attempts": 0}

    def devices(identifiers) -> int:
        identifiers[0] = 7
        return 1

    def opened(_identifier) -> int:
        physical["open"] = True
        return 0

    def refused_close(_identifier) -> int:
        physical["close_attempts"] += 1
        return -2147352576  # the SDK's BAD_HID_IO, a status and not an exception

    library = object.__new__(CtypesLmsLibrary)
    library._dll = SimpleNamespace(
        fnLMS_GetNumDevices=lambda: 1,
        fnLMS_GetDevInfo=devices,
        fnLMS_GetSerialNumber=lambda _identifier: 77,
        fnLMS_InitDevice=opened,
        fnLMS_CloseDevice=refused_close,
        fnLMS_GetMinFreq=lambda _handle: 50_000_000,
        fnLMS_GetMaxFreq=lambda _handle: 800_000_000,
        fnLMS_GetMinPwr=lambda _handle: -160,
        fnLMS_GetMaxPwr=lambda _handle: 40,
        fnLMS_GetFrequency=lambda _handle: 100_000_000,
        fnLMS_GetAbsPowerLevel=lambda _handle: 0,
        fnLMS_GetRF_On=lambda _handle: False,
    )
    source = VaunixLmsRfSource(VaunixLmsConfig(serial=77), library=library)
    with pytest.raises(RuntimeError, match="status -2147352576"):
        source.close()
    assert physical == {"open": True, "close_attempts": 1}

    broker = DeviceBroker()
    leaf = bind_rf_source(
        InstallationFactoryContext(None, broker, {}),
        "rf",
        source,
        "vaunix-lms:77",
        "rf.vaunix_lms",
    )
    installation = Installation({"rf": leaf}, world=None, broker=broker)
    with pytest.raises(BaseExceptionGroup, match="installation close failed"):
        installation.close()
    assert tuple(installation.devices) == ("rf",), (
        "a device that did not close stays owned, so close can be retried"
    )
    assert physical["close_attempts"] == 2
    broker.verify_capability(leaf.binding)


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


def test_the_lab_brick_is_opened_without_being_touched() -> None:
    """The other driver obeys the same rule, through its own transport.

    Opening reads the brick's frequency, power and switch and writes
    none of them -- including when the bench window would have preferred
    different numbers, which is a preference about what may be commanded
    and not a licence to command it.
    """

    library = InMemoryLmsLibrary((77,))
    handle = library.open_device(77)
    library.set_frequency(handle, 1_000_000_000 // 10)
    library.set_power(handle, 4 * 4)
    before = dict(library._registers(handle))
    library.close_device(handle)

    source = VaunixLmsRfSource(
        VaunixLmsConfig(
            serial=77,
            frequency_low_hz=2e9,
            power_low_dbm=10.0,
        ),
        library=library,
    )
    assert dict(library._registers(source._handle)) == before, (
        "opening the brick moved one of its knobs"
    )
    values = source.tunable_values()
    assert values[FREQUENCY_FIELD] == 1e9
    assert values[POWER_FIELD] == 4.0


def test_every_interaction_narrates_at_the_contract_layer(caplog) -> None:
    """Whoever moves a knob, the instrument's log tells the same story.

    The lines land on the ``zlc_atom.devices.rf`` loggers and each ends
    with ``device=<identity>``, which is how a bench window shows one
    instrument's story and nobody else's.
    """

    import logging

    with caplog.at_level(logging.INFO, logger="zlc_atom.devices.rf"):
        source = virtual_rf_source(VaunixLmsConfig(serial=77))
        source.tune(FREQUENCY_FIELD, 1_000_000_000.0)
        with pytest.raises(ValueError):
            source.tune(FREQUENCY_FIELD, 1_000_000_005.0)
    lines = [record.getMessage() for record in caplog.records]
    assert any(
        line.startswith("TUNE field=frequency value=1000000000.0")
        and line.endswith(f"device={source.identity}")
        for line in lines
    )
    assert any(
        line.startswith("TUNE REFUSED field=frequency")
        and line.endswith(f"device={source.identity}")
        for line in lines
    )


def test_vendor_files_live_with_the_family_and_missing_means_instructions(
    tmp_path, monkeypatch
) -> None:
    """The vendor lookup is the folder beside the module, nowhere magic.

    Resolution: an absolute path in vendor/vendor.json, else the file in
    vendor/ itself; missing yields the exact instruction (which file, into
    which folder), which is what the scan strip and the open error show.
    """

    import json

    from zlc_atom.devices.vendor import resolve_vendor_file

    anchor = tmp_path / "family" / "driver.py"
    vendor = tmp_path / "family" / "vendor"
    vendor.mkdir(parents=True)
    anchor.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as caught:
        resolve_vendor_file(str(anchor), "thing.dll", what="the Thing SDK")
    message = str(caught.value)
    assert "copy thing.dll into" in message and str(vendor) in message

    (vendor / "thing.dll").write_bytes(b"")
    assert resolve_vendor_file(
        str(anchor), "thing.dll", what="the Thing SDK"
    ) == str(vendor / "thing.dll")

    elsewhere = tmp_path / "elsewhere.dll"
    elsewhere.write_bytes(b"")
    (vendor / "vendor.json").write_text(
        json.dumps({"thing.dll": str(elsewhere)}), encoding="utf-8"
    )
    assert resolve_vendor_file(
        str(anchor), "thing.dll", what="the Thing SDK"
    ) == str(elsewhere)

    # A relative manifest path would name a different file per launcher
    # working directory; it is refused with the instruction, even when it
    # happens to resolve from here.
    monkeypatch.chdir(tmp_path)
    (vendor / "vendor.json").write_text(
        json.dumps({"thing.dll": "elsewhere.dll"}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="absolute path"):
        resolve_vendor_file(str(anchor), "thing.dll", what="the Thing SDK")

    # The Lab Brick scan surfaces that instruction rather than shrugging.
    import zlc_atom.devices.vendor as vendor_module
    import zlc_atom.devices.rf.device_types as module

    def _missing(_anchor, filename, *, what):
        raise FileNotFoundError(f"{what} is not installed: copy {filename} into ...")

    monkeypatch.setattr(vendor_module, "resolve_vendor_file", _missing)
    with pytest.raises(FileNotFoundError, match="copy vnx_fmsynth.dll into"):
        module._discover_vaunix()


class _VisaBus:
    """A machine's worth of VISA: a resource list, and sessions on demand.

    The fake is the lowest layer, as everywhere else here: the probe's real
    filtering, its real ``*IDN?``, and its real identity rules run over this.
    """

    def __init__(self, instruments: dict, *, refuse: tuple = ()) -> None:
        #: resource -> the ``*IDN?`` answer, or an exception to raise on query
        self.instruments = instruments
        #: resources whose open() fails, the way a busy instrument's does
        self.refuse = tuple(refuse)
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.timeouts: list[int] = []

    def list_resources(self) -> tuple[str, ...]:
        return tuple(self.instruments)

    def open_resource(self, resource: str, **kwargs):
        self.opened.append(resource)
        if resource in self.refuse:
            raise OSError(f"{resource} is in use by another program")
        return _VisaSession(self, resource, self.instruments[resource])


class _VisaSession:
    def __init__(self, bus: _VisaBus, resource: str, answer) -> None:
        self._bus, self._resource, self._answer = bus, resource, answer
        self.timeout = 0

    def query(self, command: str) -> str:
        assert command == "*IDN?", f"the probe asked {command!r}"
        self._bus.timeouts.append(self.timeout)
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer

    def close(self) -> None:
        self._bus.closed.append(self._resource)


def test_a_scpi_instrument_is_found_by_asking_what_it_is() -> None:
    """VISA lists addresses; only *IDN? says what is on the other end.

    So the probe opens each candidate, asks the one universal question, and
    keeps the ones this driver can actually drive.  A scope on the same bus
    answers and is passed over -- it is not a refusal, it is the answer.
    """

    from zlc_atom.devices.rf.rigol_dg4000 import discover_dg4000

    bus = _VisaBus(
        {
            "USB0::0x1AB1::0x0641::DG4E0000000001::INSTR":
                "RIGOL TECHNOLOGIES,DG4162,DG4E0000000001,00.01.12",
            "TCPIP0::198.51.100.7::INSTR":
                "RIGOL TECHNOLOGIES,DG4102,DG4E0000000002,00.01.12",
            "TCPIP0::198.51.100.9::INSTR":
                "KEYSIGHT TECHNOLOGIES,DSOX1204G,CN00000000,01.20",
        }
    )

    found = discover_dg4000(bus)

    assert [sighting.resource for sighting in found] == [
        "USB0::0x1AB1::0x0641::DG4E0000000001::INSTR",
        "TCPIP0::198.51.100.7::INSTR",
    ]
    assert [sighting.serial for sighting in found] == [
        "DG4E0000000001",
        "DG4E0000000002",
    ]
    assert [sighting.model for sighting in found] == ["DG4162", "DG4102"]
    # Every session opened is a session closed, including the scope's: a scan
    # must not leave an instrument held.
    assert sorted(bus.closed) == sorted(bus.opened)


def test_the_probe_never_opens_a_serial_port() -> None:
    """ASRL is where the board's own UART lives, and it is not a question.

    Opening a serial port to ask *IDN? takes it from whoever has it -- on
    this bench, the pulse server that owns the streamer -- and gets nothing
    back.  A signal-generator scan must not be able to do that.
    """

    from zlc_atom.devices.rf.rigol_dg4000 import discover_dg4000

    bus = _VisaBus(
        {
            "ASRL3::INSTR": AssertionError("the probe opened a serial port"),
            "USB0::0x1AB1::0x0641::DG4E0000000001::INSTR":
                "RIGOL TECHNOLOGIES,DG4162,DG4E0000000001,00.01.12",
        }
    )

    found = discover_dg4000(bus)

    assert bus.opened == ["USB0::0x1AB1::0x0641::DG4E0000000001::INSTR"]
    assert len(found) == 1


def test_an_instrument_that_will_not_answer_is_passed_over(caplog) -> None:
    """Busy, silent, or not SCPI: the ordinary case on a shared bus.

    None of them may end the scan, because the instrument the operator IS
    looking for is usually behind one of them in the list.
    """

    from zlc_atom.devices.rf.rigol_dg4000 import discover_dg4000

    bus = _VisaBus(
        {
            "TCPIP0::198.51.100.1::INSTR": TimeoutError("no answer"),
            "TCPIP0::198.51.100.2::INSTR": "",
            "TCPIP0::198.51.100.3::INSTR": "SOME PRINTER,LX-80,,1.0",
            "USB0::0x1AB1::0x0641::DG4E0000000001::INSTR":
                "RIGOL TECHNOLOGIES,DG4162,DG4E0000000001,00.01.12",
        },
        refuse=("TCPIP0::198.51.100.2::INSTR",),
    )

    found = discover_dg4000(bus, timeout_seconds=0.25)

    assert [sighting.serial for sighting in found] == ["DG4E0000000001"]
    # The bound the probe was given is the bound each session got, in ms.
    assert set(bus.timeouts) == {250}


def test_no_visa_at_all_is_an_instruction_not_an_empty_bench(monkeypatch) -> None:
    """The scan strip is where an operator asks "why no instruments?".

    "None found" would be a lie on a machine that has no VISA to look with,
    and it is a lie with no next step in it.
    """

    import zlc_atom.devices.rf.rigol_dg4000 as module

    def _no_backend():
        raise RuntimeError(
            "no VISA backend is available: install NI-VISA system-wide "
            "(or `pip install pyvisa-py`), then restart the bench"
        )

    monkeypatch.setattr(module, "visa_resources", _no_backend)
    with pytest.raises(RuntimeError, match="install NI-VISA"):
        module.discover_dg4000()


def test_a_missing_library_is_not_reported_as_a_missing_backend(monkeypatch) -> None:
    """One sentence for two faults told an operator to install what they had.

    "no VISA backend: pip install pyvisa-py" was raised whether the backend
    was absent or PyVISA had never been imported at all -- and the second is
    what an install into a DIFFERENT interpreter looks like from here.  So
    each failure says which one it is, and names the interpreter that is
    asking, because "installed" is only ever true of one of them.
    """

    import builtins
    import sys

    import zlc_atom.devices.rf.rigol_dg4000 as module

    real_import = builtins.__import__

    def _no_pyvisa(name, *rest):
        if name == "pyvisa":
            raise ModuleNotFoundError("No module named 'pyvisa'")
        return real_import(name, *rest)

    monkeypatch.delitem(sys.modules, "pyvisa", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_pyvisa)
    with pytest.raises(RuntimeError) as caught:
        module.visa_resources()

    message = str(caught.value)
    assert "PyVISA is not installed" in message
    assert sys.executable in message, "which interpreter is the answer"
    assert "backend" not in message, "a missing library is not a missing backend"


def test_a_found_instrument_is_offered_as_an_installable_card(monkeypatch) -> None:
    """What the scan finds must be addable without retyping the address."""

    import zlc_atom.devices.rf.device_types as module
    import zlc_atom.devices.rf.rigol_dg4000 as driver
    from zlc_atom.install import discover_device_catalog

    bus = _VisaBus(
        {
            "TCPIP0::198.51.100.7::INSTR":
                "RIGOL TECHNOLOGIES,DG4162,DG4E0000000002,00.01.12",
        }
    )
    monkeypatch.setattr(driver, "visa_resources", lambda: bus)

    offered = module._discover_rigol()

    assert len(offered) == 1
    card = offered[0]
    assert card.type_id == "rf.rigol_dg4000"
    assert card.instance_id == "dg4000_DG4E0000000002" == card.role
    assert card.parameters["resource"] == "TCPIP0::198.51.100.7::INSTR"
    # The schema fills the rest, so the card installs without further typing.
    assert card.parameters["timeout_seconds"] == 5.0

    # And the scan strip reaches it: the descriptor now declares a discover.
    rigol = next(
        item
        for item in discover_device_catalog().available
        if item.type_id == "rf.rigol_dg4000"
    )
    assert rigol.discover is module._discover_rigol


def test_nothing_to_ask_is_said_out_loud(monkeypatch) -> None:
    """"Found nothing" is only an answer if something was asked.

    VISA's list is far blinder than an operator expects: a LAN instrument
    appears only once it is added in NI MAX, a USB one only once its USB-TMC
    driver is bound.  A Rigol plugged in and working can simply not be in the
    list -- and reporting "no Rigol here" about that bench is a lie the
    operator cannot see through, because the scan looks identical either way.
    """

    import zlc_atom.devices.rf.device_types as module
    import zlc_atom.devices.rf.rigol_dg4000 as driver

    serial_only = _VisaBus({"ASRL3::INSTR": "", "ASRL4::INSTR": ""})
    monkeypatch.setattr(driver, "visa_resources", lambda: serial_only)
    with pytest.raises(RuntimeError) as caught:
        module._discover_rigol()
    message = str(caught.value)
    assert "VISA lists nothing to ask" in message
    assert "ASRL3::INSTR" in message, "say what it DID list"
    assert serial_only.opened == [], "and still open none of them"

    # One probeable address and no Rigol behind it is a real answer: asked,
    # nothing matched, nothing to say.
    a_scope = _VisaBus(
        {"TCPIP0::198.51.100.9::INSTR": "KEYSIGHT,DSOX1204G,CN0,01.20"}
    )
    monkeypatch.setattr(driver, "visa_resources", lambda: a_scope)
    assert module._discover_rigol() == ()
