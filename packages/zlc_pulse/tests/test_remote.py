from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import threading
import time

import pytest

import zlc_pulse.remote as remote_module
import zlc_pulse.transport as transport_module
from zlc_pulse import (
    AnalogStep,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseTarget,
    compile_sequence,
)
from zlc_pulse.device import PulseStreamer
from zlc_pulse.remote import (
    REMOTE_METHODS,
    BackendResolution,
    BackendResolutionError,
    PulseRemoteServer,
    RemoteError,
    RemotePulseStreamer,
    resolve_backend,
)
from zlc_pulse.transport import MemoryRegisterTransport
from zlc_pulse.transport.uart import UartError
from zlc_pulse.transport import uart_frame as framing
from zlc_pulse.wire import CtrlWords, StreamerParams, build_fingerprint, pack_program, pack_scan_rows


def _sequence(*, slotted: bool = False) -> PulseSequence:
    from zlc_pulse import PulseSlot
    from zlc_pulse.model import PulseFieldRef

    target = PulseTarget(
        lanes=("d0", "a0", "a1"),
        ports=(
            PulsePortSpec("d0", "digital", ("d0",)),
            PulsePortSpec("dac", "dac", ("a0", "a1"), bus_index=0),
        ),
    )
    slots = (PulseSlot("duration", PulseFieldRef("duration", "p0"), "ns", "p0_time"),) if slotted else ()
    return PulseSequence(
        target=target,
        time_step_ns=20,
        periods=(
            PulsePeriod("p0", 20, "ns", (1, 0, 0), (AnalogStep("dac", "edge", 0),)),
            PulsePeriod("p1", 20, "ns", (0, 0, 0)),
        ),
        slots=slots,
    )


@contextmanager
def _server(streamer: PulseStreamer, *, client_idle_timeout: float = 300.0):
    server = PulseRemoteServer(("127.0.0.1", 0), streamer, client_idle_timeout=client_idle_timeout)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _client(server: PulseRemoteServer) -> RemotePulseStreamer:
    client = RemotePulseStreamer("127.0.0.1", server.server_address[1], poll_interval=0.001)
    client.open()
    return client


def test_remote_replays_device_path_with_short_done_poll() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6)
    with _server(streamer) as server:
        client = _client(server)
        try:
            client.load(program, source=source)
            client.write_slots((1,))
            client.fire()
            report = client.wait_done(1.0)
            assert report is not None
            assert report.status_reads == (4, 4)
            safe = client.safe()
            assert safe.stable
            state = client.applied()
            assert state is not None
            assert state.source == source
            assert state.slot_values == (1,)
        finally:
            client.close()


def test_remote_safe_interrupts_forever_fire_on_the_same_connection() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6)
    with _server(streamer) as server:
        client = _client(server)
        try:
            client.load(program)
            client.fire(forever=True)
            result: list[object] = []

            def interrupt() -> None:
                result.append(client.safe())

            worker = threading.Thread(target=interrupt)
            worker.start()
            worker.join(timeout=1.0)
            assert not worker.is_alive()
            assert len(result) == 1
            assert result[0].stable
            assert client.snapshot()["firing"] is False
        finally:
            client.disconnect()


def test_remote_logs_lifecycle_events_without_payload_dump(capsys) -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6)
    with _server(streamer) as server:
        client = _client(server)
        try:
            client.load(program, source=source)
            assert client.snapshot()["opened"] is True
            assert client.cursor() == 0
            assert client.applied() is not None
            client.write_slots((1,))
            client.write_scan_table(((1,), (2,)))
            client.fire()
            assert client.wait_done(1.0) is not None
            safe = client.safe()
            assert safe.stable
        finally:
            client.close()

    output = capsys.readouterr().out
    assert "ZLC CLIENT CONNECTED" in output
    assert "ZLC OPEN" in output
    assert "ZLC LOAD" in output
    assert "edges=3" in output
    assert "ZLC FIRE" in output
    assert "mode=ONCE" in output
    assert "reloaded_before_fire=False" in output
    assert "ZLC SNAPSHOT client=127.0.0.1:" in output
    assert "ZLC CURSOR client=127.0.0.1:" in output
    assert "ZLC APPLIED client=127.0.0.1:" in output
    assert "ZLC WRITE SLOTS client=127.0.0.1:" in output
    assert "ZLC WRITE SCAN TABLE client=127.0.0.1:" in output
    assert "ZLC DONE" in output
    assert "ZLC STOP/SAFE" in output
    assert "stable=True" in output
    assert "ZLC CLOSE" in output
    assert "ZLC CLIENT DISCONNECTED" in output
    assert "AnalogStep" not in output


def test_second_client_gets_owner_and_hold_duration_in_busy_reply() -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    streamer = PulseStreamer(MemoryRegisterTransport(geom=geom), geom, 50e6)
    with _server(streamer) as server:
        owner = _client(server)
        contender = RemotePulseStreamer("127.0.0.1", server.server_address[1], request_timeout=1.0)
        try:
            with pytest.raises(RemoteError, match=r"server is busy.*current owner=.*held_for=") as error:
                contender.open()
            assert error.value.remote_type == "RemoteBusyError"
            assert owner.snapshot()["opened"] is True
        finally:
            owner.close()
            contender.disconnect()


def test_idle_client_is_safe_before_owner_release(monkeypatch, capsys) -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    program = compile_sequence(_sequence(), geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6)
    with _server(streamer, client_idle_timeout=0.05) as server:
        events: list[str] = []
        original_safe = streamer.safe
        original_release = server._release_client

        def safe_with_trace():
            events.append("safe")
            return original_safe()

        def release_with_trace(client, *, force=False):
            events.append("release")
            return original_release(client, force=force)

        monkeypatch.setattr(streamer, "safe", safe_with_trace)
        monkeypatch.setattr(server, "_release_client", release_with_trace)
        client = _client(server)
        client.load(program)
        client.fire(forever=True)
        time.sleep(0.15)
        assert streamer.snapshot()["firing"] is False
        assert transport.status == 0
        replacement = _client(server)
        replacement.close()
        client.disconnect()
        assert events[:2] == ["safe", "release"]
    output = capsys.readouterr().out
    assert "ZLC CLIENT IDLE TIMEOUT" in output
    assert "idle timeout; SAFE released" in output


def test_client_endpoint_display_separates_bind_from_connect_host(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        remote_module,
        "_local_ipv4_addresses",
        lambda: ("192.168.0.20", "10.0.0.5"),
    )

    remote_module._print_client_endpoints("0.0.0.0", 18861)

    output = capsys.readouterr().out
    assert "listen_bind=0.0.0.0:18861" in output
    assert "ZLC SERVER ADDRESS" in output
    assert "scope=same_computer address=127.0.0.1:18861" in output
    assert "scope=other_computer address=192.168.0.20:18861" in output
    assert 'same_computer=RemotePulseStreamer("127.0.0.1", 18861' in output
    assert "other_computer=192.168.0.20:18861" in output
    assert 'other_computer=RemotePulseStreamer("192.168.0.20", 18861' in output
    assert "0.0.0.0 is listen-only" in output


def _fake_resolution(tmp_path, outcomes, *, requested="auto", ports=("COM1",)):
    calls: list[tuple[str, float]] = []

    def probe(port: str, timeout: float) -> None:
        calls.append((port, timeout))
        outcome = outcomes.get(port)
        if outcome is not None:
            raise outcome

    result = resolve_backend(
        requested,
        params=StreamerParams(),
        clock_hz=50e6,
        state_dir=str(tmp_path),
        port_provider=lambda: ports,
        probe=probe,
    )
    return result, calls


def test_backend_auto_selects_first_uart_with_matching_word63(tmp_path) -> None:
    result, calls = _fake_resolution(tmp_path, {})

    assert result.backend == "uart"
    assert result.uart_port == "COM1"
    assert "word63 fingerprint match" in result.reason
    assert calls == [("COM1", 0.5)]


def test_backend_auto_skips_failed_port_and_selects_later_uart(tmp_path) -> None:
    result, calls = _fake_resolution(
        tmp_path,
        {"COM1": TimeoutError("UART reply timed out")},
        ports=("COM1", "COM2"),
    )

    assert result.backend == "uart"
    assert result.uart_port == "COM2"
    assert result.attempts == ("COM1: timeout", "COM2: word63 fingerprint matched")
    assert calls == [("COM1", 0.5), ("COM2", 0.5)]


def test_backend_auto_probes_every_os_port_with_physical_usb_first(monkeypatch, tmp_path) -> None:
    class Port:
        def __init__(self, device, description, *, vid=None, pid=None):
            self.device, self.description, self.vid, self.pid = device, description, vid, pid

    discovered = (
        Port("COM3", "Intel AMT Serial-over-LAN"),
        Port("COM6", "USB-SERIAL CH340", vid=0x1A86, pid=0x7523),
        Port("COM5", "Bluetooth serial"),
        Port("COM4", "Bluetooth serial"),
    )
    from serial.tools import list_ports
    monkeypatch.setattr(list_ports, "comports", lambda: discovered)
    calls: list[str] = []

    def probe(port, _timeout):
        calls.append(port)
        if port != "COM4":
            raise TimeoutError("not the pulse streamer")

    result = resolve_backend(
        "auto", params=StreamerParams(), clock_hz=50e6, state_dir=str(tmp_path), probe=probe
    )

    assert result.candidates == ("COM6", "COM3", "COM5", "COM4")
    assert calls == ["COM6", "COM3", "COM5", "COM4"]
    assert result.uart_port == "COM4"


def test_backend_auto_falls_back_to_jtag_after_uart_timeout(tmp_path) -> None:
    result, calls = _fake_resolution(
        tmp_path,
        {"COM1": TimeoutError("UART reply timed out")},
    )

    assert result.backend == "jtag-axi"
    assert result.uart_port is None
    assert result.attempts == ("COM1: timeout",)
    assert "fallback to jtag-axi" in result.reason
    assert calls == [("COM1", 0.5)]


def test_backend_auto_classifies_crc_failure_before_fallback(tmp_path) -> None:
    result, _ = _fake_resolution(
        tmp_path,
        {"COM1": framing.FrameError("UART reply CRC mismatch")},
    )

    assert result.backend == "jtag-axi"
    assert result.attempts == ("COM1: CRC error",)


def test_backend_auto_classifies_fingerprint_mismatch_before_fallback(tmp_path) -> None:
    result, _ = _fake_resolution(
        tmp_path,
        {"COM1": RuntimeError("geometry/layout mismatch: device=0x12345678")},
    )

    assert result.backend == "jtag-axi"
    assert result.attempts == ("COM1: fingerprint mismatch",)


def test_backend_failure_categories_use_real_transport_exceptions() -> None:
    broken = bytearray(framing.encode_reply(1, framing.ST_OK, (0,)))
    broken[-1] ^= 1
    try:
        framing.decode_reply(bytes(broken))
    except framing.FrameError as frame_error:
        assert remote_module._probe_failure_reason(frame_error) == "CRC error"
    else:
        raise AssertionError("corrupted UART frame was accepted")

    assert remote_module._probe_failure_reason(UartError("UART CRC error status in read reply")) == "CRC error"

    mismatch = MemoryRegisterTransport(layout_id=build_fingerprint(StreamerParams()) ^ 1)
    mismatch_streamer = PulseStreamer(mismatch, StreamerParams(), 50e6)
    try:
        mismatch_streamer.open()
    except RuntimeError as error:
        assert remote_module._probe_failure_reason(error) == "fingerprint mismatch"
    else:
        raise AssertionError("geometry mismatch was accepted")
    assert remote_module._probe_failure_reason(TimeoutError("UART reply timed out")) == "timeout"
    assert remote_module._probe_failure_reason(FileNotFoundError("COM7")) == "open failed"


def test_real_uart_crc_status_exception_maps_to_crc_category(tmp_path) -> None:
    class CrcLink:
        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        def exchange(self, request, *, deadline, stop=None):
            return framing.encode_reply(request[3], framing.ST_CRC_FAIL)

        def write_batch(self, requests, *, deadline, stop=None):
            return [framing.encode_reply(request[3], framing.ST_CRC_FAIL) for request in requests]

    transport = transport_module.UartRegisterTransport(state_dir=tmp_path, link=CrcLink())
    transport.start()
    try:
        with pytest.raises(UartError) as failure:
            transport.read_word(CtrlWords.LAYOUT_ID)
        assert remote_module._probe_failure_reason(failure.value) == "CRC error"
    finally:
        transport.close()


def test_backend_auto_distinguishes_open_failure_and_no_ports(tmp_path) -> None:
    open_failed, _ = _fake_resolution(
        tmp_path,
        {"COM1": OSError("could not open port COM1: Access is denied")},
    )
    no_ports, calls = _fake_resolution(tmp_path, {}, ports=())

    assert open_failed.attempts == ("COM1: open failed",)
    assert no_ports.backend == "jtag-axi"
    assert no_ports.attempts == ("no UART ports detected",)
    assert calls == []


def test_backend_auto_missing_pyserial_falls_back_without_probe(tmp_path) -> None:
    def missing_pyserial():
        raise ModuleNotFoundError("No module named 'serial'")

    calls: list[tuple[str, float]] = []
    result = resolve_backend(
        "auto",
        params=StreamerParams(),
        clock_hz=50e6,
        state_dir=str(tmp_path),
        port_provider=missing_pyserial,
        probe=lambda port, timeout: calls.append((port, timeout)),
    )

    assert result.backend == "jtag-axi"
    assert result.attempts[0].startswith("pyserial missing; install with:")
    assert calls == []


def test_uart_probe_reuses_pulse_streamer_word63_open(monkeypatch, tmp_path) -> None:
    params = StreamerParams()
    records: list[tuple[str, int, float, int]] = []
    open_calls: list[int] = []

    class FakeTransport:
        def __init__(self, *, state_dir, port, baud, action_timeout):
            records.append((port, baud, action_timeout, int(state_dir is not None)))
            self.started = False
            self.closed = 0

        def start(self) -> None:
            self.started = True

        def read_word(self, address: int) -> int:
            assert self.started
            assert address == CtrlWords.LAYOUT_ID
            return build_fingerprint(params)

        def close(self) -> None:
            self.closed += 1

    monkeypatch.setattr(transport_module, "UartRegisterTransport", FakeTransport)
    original_open = remote_module.PulseStreamer.open

    def spy_open(streamer):
        open_calls.append(1)
        return original_open(streamer)

    monkeypatch.setattr(remote_module.PulseStreamer, "open", spy_open)

    result = resolve_backend(
        "auto",
        uart_baud=3_000_000,
        params=params,
        clock_hz=50e6,
        state_dir=str(tmp_path),
        port_provider=lambda: ("COM7",),
    )

    assert result.backend == "uart"
    assert result.uart_port == "COM7"
    assert result.attempts == ("COM7: word63 fingerprint matched",)
    assert records == [("COM7", 3_000_000, 0.5, 1)]
    assert len(open_calls) == 1


def test_uart_probe_direct_helper_uses_existing_open_handshake(monkeypatch, tmp_path) -> None:
    params = StreamerParams()
    addresses: list[int] = []

    class FakeTransport:
        def __init__(self, *, state_dir, port, baud, action_timeout):
            self.started = False

        def start(self) -> None:
            self.started = True

        def read_word(self, address: int) -> int:
            addresses.append(address)
            return build_fingerprint(params)

        def close(self) -> None:
            pass

    monkeypatch.setattr(transport_module, "UartRegisterTransport", FakeTransport)

    remote_module._probe_uart_port(
        "COM7",
        0.5,
        params=params,
        clock_hz=50e6,
        state_dir=str(tmp_path),
        baud=3_000_000,
    )

    assert addresses == [CtrlWords.LAYOUT_ID]


def test_explicit_uart_failure_does_not_fallback(tmp_path) -> None:
    with pytest.raises(BackendResolutionError, match="explicit UART backend failed"):
        _fake_resolution(
            tmp_path,
            {"COM3": TimeoutError("UART reply timed out")},
            requested="uart",
            ports=("COM3",),
        )


def test_explicit_jtag_skips_uart_probe(tmp_path) -> None:
    calls: list[tuple[str, float]] = []

    result = resolve_backend(
        "jtag-axi",
        params=StreamerParams(),
        clock_hz=50e6,
        state_dir=str(tmp_path),
        port_provider=lambda: (_ for _ in ()).throw(AssertionError("UART enumeration must be skipped")),
        probe=lambda port, timeout: calls.append((port, timeout)),
    )

    assert result.backend == "jtag-axi"
    assert calls == []


def test_uart_port_override_is_the_only_auto_candidate(tmp_path) -> None:
    calls: list[tuple[str, float]] = []

    result = resolve_backend(
        "auto",
        uart_port="COM9",
        params=StreamerParams(),
        clock_hz=50e6,
        state_dir=str(tmp_path),
        port_provider=lambda: (_ for _ in ()).throw(AssertionError("port enumeration must be skipped")),
        probe=lambda port, timeout: calls.append((port, timeout)),
    )

    assert result.backend == "uart"
    assert result.uart_port == "COM9"
    assert calls == [("COM9", 0.5)]


def test_server_cli_defaults_to_auto_and_accepts_explicit_backends() -> None:
    parser = remote_module.build_arg_parser()

    assert parser.parse_args([]).backend == "auto"
    assert parser.parse_args(["--backend", "jtag-axi"]).backend == "jtag-axi"
    assert parser.parse_args(["--backend", "uart", "--uart-port", "COM3"]).uart_port == "COM3"
    assert parser.parse_args([]).client_idle_timeout == 300.0
    assert parser.parse_args(["--client-idle-timeout", "12.5"]).client_idle_timeout == 12.5


def test_main_logs_final_backend_reason_attempts_and_jtag_note(monkeypatch, capsys, tmp_path) -> None:
    resolution = BackendResolution(
        "auto",
        "jtag-axi",
        None,
        "auto fallback to jtag-axi after UART probe: COM7: timeout",
        ("COM7: timeout",),
        ("COM7",),
    )

    class FakeJtagTransport(MemoryRegisterTransport):
        def __init__(self, *, state_dir):
            del state_dir
            super().__init__(geom=StreamerParams())

    def fake_resolve(*args, **kwargs):
        kwargs["on_candidates"](("COM7",), ())
        return resolution

    monkeypatch.setattr(remote_module, "resolve_backend", fake_resolve)
    monkeypatch.setattr(transport_module, "VivadoAxiRegisterTransport", FakeJtagTransport)
    monkeypatch.setattr(remote_module, "serve", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert remote_module._main(["--backend", "auto", "--state-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "ZLC SERVER STARTING" in output
    assert "endpoint=0.0.0.0:18861" in output
    assert "ZLC UART PROBE" in output
    assert "ports=COM7" in output
    assert "ZLC BACKEND RESOLVED" in output
    assert "selected=jtag-axi" in output
    assert "reason=" in output
    assert "attempts=" in output
    assert "COM7: timeout" in output
    assert "ZLC BACKEND NOTE" in output


def test_main_explicit_backend_failure_is_logged_and_returns_two(monkeypatch, capsys) -> None:
    def fail(*args, **kwargs):
        raise BackendResolutionError("explicit UART backend failed", attempts=("COM3: CRC error",))

    monkeypatch.setattr(remote_module, "resolve_backend", fail)
    assert remote_module._main(["--backend", "uart", "--uart-port", "COM3"]) == 2
    captured = capsys.readouterr()
    assert "ZLC BACKEND FAILED" in captured.out
    assert "COM3: CRC error" in captured.out
    assert "explicit UART backend failed" in captured.err


def test_remote_disconnect_preserves_applied_for_the_next_client(capsys) -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6)
    with _server(streamer) as server:
        client_a = _client(server)
        client_a.load(program, source=source)
        client_a.write_slots((1,))
        client_a.disconnect()

        client_b = _client(server)
        try:
            state = client_b.applied()
            assert state is not None
            assert state.source == source
            assert state.slot_values == (1,)
            assert state.scan_rows == ()
            assert state.forever is False
            assert state.source is not None
            rebuilt = compile_sequence(state.source, geom, 50e6)
            assert pack_program(rebuilt, geom) == pack_program(state.program, geom)
            assert pack_scan_rows((state.slot_values,), geom, 0, 0) == pack_scan_rows(((1,),), geom, 0, 0)
        finally:
            client_b.close()
    assert streamer.applied() is None
    assert "ZLC AUTO-SAFE" in capsys.readouterr().out


def test_remote_surface_is_exactly_the_contract_method_set() -> None:
    """The protocol is a frozen list, and growing it is a deliberate act.

    ``describe`` was added because the set below could open, load and fire a
    board but not ask what the board was, so every client had to read its own
    XDC and config and hope they matched the bitstream.
    """

    assert REMOTE_METHODS == (
        "open",
        "describe",
        "close",
        "load",
        "write_slots",
        "write_scan_table",
        "fire",
        "wait_done",
        "cursor",
        "safe",
        "snapshot",
        "applied",
    )


def test_client_that_drops_its_socket_is_not_a_server_error(capsys) -> None:
    """A client vanishing is the end of a session, not a server fault.

    The handler had only try/finally, so a reset propagated into socketserver's
    handle_error and printed a raw traceback banner right after the clean
    "CLIENT DISCONNECTED" line -- exactly what a user sees when they close a
    notebook.  Here the client's kernel is made to send RST via SO_LINGER(0).
    """

    import socket as socket_module
    import struct as struct_module

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    streamer = PulseStreamer(MemoryRegisterTransport(geom=geom, auto_done=True), geom, 50e6)
    seen: list[BaseException] = []

    with _server(streamer) as server:
        original = server.handle_error

        def spy(request, client_address):
            import sys as sys_module

            seen.append(sys_module.exc_info()[1])
            return original(request, client_address)

        server.handle_error = spy
        client = RemotePulseStreamer("127.0.0.1", server.server_address[1], poll_interval=0.001)
        client.open()
        client._socket.setsockopt(
            socket_module.SOL_SOCKET,
            socket_module.SO_LINGER,
            struct_module.pack("hh", 1, 0),
        )
        client._socket.close()
        deadline = time.monotonic() + 5.0
        while "CLIENT DISCONNECTED" not in capsys.readouterr().out and time.monotonic() < deadline:
            time.sleep(0.01)

    assert seen == [], f"a dropped client reached handle_error: {seen}"
