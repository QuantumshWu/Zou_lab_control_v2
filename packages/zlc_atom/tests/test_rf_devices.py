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
        "frequency_low_hz",
        "frequency_high_hz",
        "power_low_dbm",
        "power_high_dbm",
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
    source, instrument = _rigol(frequency_high_hz=1e6, power_high_dbm=10.0)
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
    source, _instrument = _rigol(
        frequency_low_hz=1e3,
        frequency_high_hz=160e6,
        power_low_dbm=-30.0,
        power_high_dbm=10.0,
    )
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
        assert by_name[f"{channel}_frequency_hz"].live_write
    for field in by_name.values():
        assert field.dependency_group == (field.metadata.name,)


def test_optional_window_is_one_init_and_control_policy() -> None:
    """The bench's safety window moved to the control panel, off Init.

    It is adjusted with plain Apply (never live), or through the same
    ``tune`` API -- and the scan add-axis combo must never offer it: the
    window fields are non-live and unbounded, which is exactly what
    scan_ports_for_devices excludes.  Tightening an edge past a channel's
    CURRENT value is refused by name: policy may fence a knob in, never
    silently drag a set output to a new value.
    """

    from zlc_atom.nodes.scan.plan import scan_ports_for_devices

    source, instrument = _rigol()
    by_name = {field.metadata.name: field for field in source.tunable_fields()}
    for name in (
        "frequency_low_hz",
        "frequency_high_hz",
        "power_low_dbm",
        "power_high_dbm",
    ):
        window = by_name[name]
        assert not window.live_write, "the window applies, never live"
        assert window.metadata.minimum is None and window.metadata.maximum is None
        assert window.current is None

    ports = scan_ports_for_devices({"rf": source})
    assert ports == (), "an unbounded knob has no finite scan-authoring range"
    assert not any("FREQuency " in command for command in instrument.log), (
        "omitting all policy edges must not move hardware at open"
    )

    assert source.tune("frequency_low_hz", 1e3) == 1e3
    assert source.tune("frequency_high_hz", 80e6) == 80e6
    assert source.tune("power_low_dbm", -30.0) == -30.0
    assert source.tune("power_high_dbm", 10.0) == 10.0
    ports = scan_ports_for_devices({"rf": source})
    offered = {port.port.split(":")[-1] for port in ports}
    assert offered == {
        "ch1_frequency_hz", "ch1_power_dbm", "ch2_frequency_hz", "ch2_power_dbm"
    }

    before = source.settings_provenance()["settings_epoch"]
    assert source.tune("frequency_high_hz", None) is None
    assert source.tunable_values()["frequency_high_hz"] is None
    assert source.settings_provenance()["settings_epoch"] == before + 1
    assert not any(
        port.port.endswith(":ch1_frequency_hz") for port in scan_ports_for_devices({"rf": source})
    ), "clearing either edge removes the finite scan range"
    assert source.tune("frequency_high_hz", 80e6) == 80e6
    # The channel knobs' own scan bounds follow the window immediately.
    by_name = {field.metadata.name: field for field in source.tunable_fields()}
    assert by_name["ch1_frequency_hz"].metadata.maximum == 80e6

    source.tune("ch1_frequency_hz", 50e6)
    with pytest.raises(ValueError, match="strand ch1_frequency_hz at 5e"):
        source.tune("frequency_high_hz", 20e6)
    with pytest.raises(ValueError, match="empty window"):
        source.tune("frequency_low_hz", 90e6)


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
    assert not any(line.startswith("OPEN NORMALIZED") for line in lines), lines
    assert any(
        line.startswith("TUNE field=frequency_hz value=1000000000.0")
        and line.endswith(f"device={source.identity}")
        for line in lines
    )
    assert any(
        line.startswith("TUNE REFUSED field=frequency_hz")
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
