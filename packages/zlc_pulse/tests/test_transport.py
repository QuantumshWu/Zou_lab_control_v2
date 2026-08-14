from __future__ import annotations

from types import SimpleNamespace
import sys

from zlc_pulse.transport import InterprocessDeviceLease, VivadoAxiRegisterTransport
from zlc_pulse.transport import uart_frame as framing
from zlc_pulse.transport.axi import JTAG_AXI_OBSERVER_INTERVAL
from zlc_pulse.transport.uart import PySerialLink, UartError, UartRegisterTransport
from zlc_pulse.transport.memory import MemoryRegisterTransport


def test_uart_frame_round_trip_and_crc_guard() -> None:
    reply = framing.encode_reply(7, framing.ST_OK, (0x12345678, 9))
    assert framing.decode_reply(reply) == (7, framing.ST_OK, [0x12345678, 9])
    broken = bytearray(reply)
    broken[-1] ^= 0x01
    try:
        framing.decode_reply(bytes(broken))
    except framing.FrameError:
        pass
    else:
        raise AssertionError("CRC corruption was accepted")


def test_uart_coalescing_respects_gaps_and_frame_limit() -> None:
    pairs = [(index, index + 10) for index in range(5)]
    pairs.extend(((8, 99), (9, 100)))
    assert framing.coalesce_runs(pairs, max_words=3) == [
        (0, [10, 11, 12]),
        (3, [13, 14]),
        (8, [99, 100]),
    ]


def test_axi_burst_split_preserves_4kb_boundary(tmp_path) -> None:
    calls: list[tuple[list[str], str]] = []

    def execute(lines, action, _remaining):
        calls.append((list(lines), action))
        return ""

    transport = VivadoAxiRegisterTransport(state_dir=tmp_path, tcl_executor=execute)
    transport.start()
    transport.write_words(((1023, 1), (1024, 2)))
    assert len(calls) == 1
    writes = [line for line in calls[0][0] if "-type write" in line]
    assert len(writes) == 2
    assert "-address 00000FFC" in writes[0]
    assert "-address 00001000" in writes[1]


def test_device_lease_is_exclusive_until_release(tmp_path) -> None:
    first = InterprocessDeviceLease(tmp_path / "pulse.lock")
    second = InterprocessDeviceLease(tmp_path / "pulse.lock")
    first.acquire()
    try:
        try:
            second.acquire()
        except RuntimeError:
            pass
        else:
            raise AssertionError("a second lease acquired the same device")
    finally:
        first.release()
    second.acquire()
    second.release()


def test_uart_open_disables_modem_control_lines_before_any_write(monkeypatch) -> None:
    records: dict[str, object] = {}

    class FakeSerialPort:
        def __init__(self, *args, **kwargs):
            records["args"] = args
            records["kwargs"] = kwargs
            self.dtr = True
            self.rts = True

        def close(self):
            records["closed"] = True

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=FakeSerialPort))
    link = PySerialLink("COM7", 3_000_000)
    link.open()
    serial_port = link._serial
    assert records["kwargs"] == {
        "timeout": 0.05,
        "write_timeout": 1.0,
        "dsrdtr": False,
        "rtscts": False,
        "xonxoff": False,
    }
    assert serial_port.dtr is False
    assert serial_port.rts is False
    link.close()


def test_uart_crc_status_is_reported_as_crc_error(tmp_path) -> None:
    class FakeLink:
        def open(self):
            pass

        def close(self):
            pass

        def exchange(self, request, *, deadline, stop=None):
            del deadline, stop
            return framing.encode_reply(request[3], framing.ST_CRC_FAIL)

        def write_batch(self, requests, *, deadline, stop=None):
            del requests, deadline, stop
            return []

    transport = UartRegisterTransport(state_dir=tmp_path, link=FakeLink())
    transport.start()
    try:
        try:
            transport.read_word(63)
        except UartError as error:
            assert "CRC" in str(error)
        else:
            raise AssertionError("CRC status was accepted")
    finally:
        transport.close()


def test_observer_intervals_are_transport_specific() -> None:
    assert MemoryRegisterTransport.observer_interval == 0.001
    assert UartRegisterTransport.observer_interval == 0.001
    assert VivadoAxiRegisterTransport.observer_interval == JTAG_AXI_OBSERVER_INTERVAL
    assert JTAG_AXI_OBSERVER_INTERVAL >= 0.05


def test_memory_transport_records_full_list_history_by_default() -> None:
    transport = MemoryRegisterTransport()
    transport.start()

    for index in range(5):
        address = 10_000 + index
        transport.write_words(((address, index),))
        assert transport.read_word(address) == index

    assert isinstance(transport.write_batches, list)
    assert isinstance(transport.read_log, list)
    assert transport.write_batches[:2] == [
        ((10_000, 0),),
        ((10_001, 1),),
    ]
    assert transport.write_batches[-2:] == [
        ((10_003, 3),),
        ((10_004, 4),),
    ]
    assert transport.read_log == [10_000, 10_001, 10_002, 10_003, 10_004]


def test_memory_transport_can_disable_diagnostic_history() -> None:
    transport = MemoryRegisterTransport(record_history=False)
    transport.start()

    for index in range(10_000):
        transport.write_words(((10_000, index),))
        assert transport.read_word(10_000) == index

    assert transport.words[10_000] == 9_999
    assert transport.write_batches == []
    assert transport.read_log == []


def test_default_vivado_discovers_fake_installed_release(monkeypatch, tmp_path) -> None:
    from zlc_pulse.transport import axi as axi_module

    root = tmp_path / "Vivado"
    old = root / "2023.2" / "bin" / "vivado.bat"
    new = root / "2024.1" / "bin" / "vivado.bat"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text("", encoding="utf-8")
    new.write_text("", encoding="utf-8")
    monkeypatch.setattr(axi_module, "VIVADO_SEARCH_ROOTS", (root,))
    monkeypatch.delenv("ZLC_PS_VIVADO_BIN", raising=False)
    assert axi_module._default_vivado() == str(new)
