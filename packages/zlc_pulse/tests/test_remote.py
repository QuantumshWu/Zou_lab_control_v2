from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import socket
import threading
import time
from types import SimpleNamespace

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
    pulse_target_from_xdc,
    sequence_from_tree,
    sequence_to_tree,
)
from zlc_pulse.canonical import canonical_bytes
from zlc_pulse import codec as pulse_codec
from zlc_pulse.device import PulseStreamer
from zlc_pulse.remote import (
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


_BOARD_TARGET = pulse_target_from_xdc()


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
            PulsePeriod("p0", 40, "ns", (1, 0, 0), (AnalogStep("dac", "edge", 0),)),
            PulsePeriod("p1", 40, "ns", (0, 0, 0)),
        ),
        slots=slots,
    )


@contextmanager
def _server(streamer: PulseStreamer):
    server = PulseRemoteServer(("127.0.0.1", 0), streamer)
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


def test_canonical_and_pulse_json_boundaries_do_not_coerce_or_drop_input() -> None:
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        canonical_bytes({1: "not the same key as text one"})

    tree = sequence_to_tree(_sequence())
    tree["periods"][0]["period_id"] = None
    with pytest.raises(TypeError, match="period_id must be non-empty text"):
        sequence_from_tree(tree)

    tree = sequence_to_tree(_sequence())
    tree["periods"][0]["typo"] = 1
    with pytest.raises(ValueError, match="unknown.*period"):
        sequence_from_tree(tree)

    with pytest.raises(ValueError, match="duplicate key.*format"):
        pulse_codec.parse_pulse_tree_json(
            '{"format":"wrong","format":"zlc.pulse.v1"}'
        )
    with pytest.raises(ValueError, match="non-finite JSON constant.*NaN"):
        pulse_codec.parse_pulse_tree_json('{"format":"zlc.pulse.v1","x":NaN}')

    document = sequence_to_tree(_sequence())
    document["editor"] = {
        "visible_ports": None,
        "scan_source": "",
        "scan_rows": [],
        "scan_source_dirty": False,
        "scan_repeats": 0,
    }
    assert pulse_codec.sequence_from_document_tree(document) == _sequence()
    document["editor"]["typo"] = True
    with pytest.raises(ValueError, match="unknown pulse editor field.*typo"):
        pulse_codec.sequence_from_document_tree(document)

    remote_tree = remote_module.encode_tree(_sequence())
    remote_tree["typo"] = 1
    with pytest.raises(ValueError, match="unknown PulseSequence remote field"):
        remote_module.decode_tree(remote_tree)


@pytest.mark.parametrize(
    "payload, message",
    (
        (b'{"id":1,"id":2,"method":"snapshot","params":{}}', "duplicate key.*id"),
        (b'{"id":1,"method":"snapshot","params":{"x":NaN}}', "non-finite JSON constant.*NaN"),
    ),
)
def test_remote_json_frame_rejects_lossy_json(payload: bytes, message: str) -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(len(payload).to_bytes(4, "big") + payload)
        with pytest.raises(ValueError, match=message):
            remote_module._recv_frame(receiver)
    finally:
        sender.close()
        receiver.close()


def _sequence_geometry() -> StreamerParams:
    return replace(
        StreamerParams(),
        channel_count=3,
        bus_count=1,
        bus_width=2,
        max_edges=8,
        bank_size=2,
    )


def test_remote_owner_transfer_requires_stable_safe_and_rejects_stale_handler(
    monkeypatch,
) -> None:
    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=geom), geom, 50e6, target=_BOARD_TARGET
    )
    streamer.open()
    server = PulseRemoteServer(("127.0.0.1", 0), streamer)
    a_server, a_peer = socket.socketpair()
    b_server, b_peer = socket.socketpair()
    original_safe = streamer.safe
    try:
        server.claim_client("A", a_server)

        def fail_safe():
            raise RuntimeError("injected SAFE failure")

        monkeypatch.setattr(streamer, "safe", fail_safe)
        with pytest.raises(RuntimeError, match="takeover.*SAFE"):
            server.claim_client("B", b_server)
        assert server.owner_status()[0] is None
        with pytest.raises(RuntimeError, match="no longer owns"):
            server.dispatch("snapshot", {}, client="A", connection=a_server)

        monkeypatch.setattr(streamer, "safe", original_safe)
        server.claim_client("B", b_server)
        assert server.owner_status()[0] == "B"
        with pytest.raises(RuntimeError, match="no longer owns"):
            server.dispatch("snapshot", {}, client="A", connection=a_server)
    finally:
        for connection in (a_server, a_peer, b_server, b_peer):
            try:
                connection.close()
            except OSError:
                pass
        monkeypatch.setattr(streamer, "safe", original_safe)
        server.server_close()
        streamer.close()


def test_takeover_revokes_and_cancels_an_active_old_command(monkeypatch) -> None:
    source = _sequence()
    geom = _sequence_geometry()
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom)
    streamer = PulseStreamer(transport, geom, 50e6, target=source.target)
    streamer.open()
    server = PulseRemoteServer(("127.0.0.1", 0), streamer)
    a_server, a_peer = socket.socketpair()
    b_server, b_peer = socket.socketpair()
    command_entered = threading.Event()
    natural_release = threading.Event()
    command_cancelled = threading.Event()
    safe_started = threading.Event()
    safe_finished = threading.Event()
    safe_calls: list[str] = []
    events: list[str] = []
    old_failures: list[BaseException] = []
    takeover_failures: list[BaseException] = []
    original_write = transport.write_words
    original_safe = streamer.safe

    def blocked_write(rows, *, stop=None, deadline=None, resend=True):
        if stop is not None and not command_entered.is_set():
            events.append("A command entered transport")
            command_entered.set()
            while not natural_release.is_set():
                if stop.wait(0.01):
                    events.append("A command cancelled")
                    command_cancelled.set()
                    raise RuntimeError("blocked transport command cancelled")
            events.append("A command completed naturally")
        return original_write(
            rows, stop=stop, deadline=deadline, resend=resend
        )

    def recorded_safe():
        phase = "cancel" if not safe_calls else "final"
        safe_calls.append(phase)
        events.append(f"{phase} SAFE started")
        if phase == "cancel":
            safe_started.set()
        result = original_safe()
        events.append(f"{phase} SAFE finished")
        if phase == "final":
            safe_finished.set()
        return result

    def run_old_command() -> None:
        try:
            server.dispatch(
                "load",
                {"program": program, "source": source, "rows": []},
                client="A",
                connection=a_server,
            )
        except BaseException as exc:
            old_failures.append(exc)

    def take_over() -> None:
        try:
            server.claim_client("B", b_server)
        except BaseException as exc:
            takeover_failures.append(exc)

    monkeypatch.setattr(transport, "write_words", blocked_write)
    monkeypatch.setattr(streamer, "safe", recorded_safe)
    old_command = threading.Thread(target=run_old_command)
    takeover = threading.Thread(target=take_over)
    try:
        server.claim_client("A", a_server)
        old_command.start()
        assert command_entered.wait(1.0)
        takeover.start()
        safe_was_prompt = safe_started.wait(0.5)
        cancelled_without_natural_release = command_cancelled.wait(0.5)
        if not cancelled_without_natural_release:
            natural_release.set()
        old_command.join(timeout=2.0)
        takeover.join(timeout=2.0)

        assert not old_command.is_alive()
        assert not takeover.is_alive()
        assert safe_was_prompt is True
        assert cancelled_without_natural_release is True
        assert len(old_failures) == 1
        assert "no longer owns" in str(old_failures[0])
        assert takeover_failures == []
        assert safe_finished.is_set()
        assert events == [
            "A command entered transport",
            "cancel SAFE started",
            "A command cancelled",
            "cancel SAFE finished",
            "final SAFE started",
            "final SAFE finished",
        ]
        assert server.owner_status()[0] == "B"
        assert streamer.applied() is None
        with pytest.raises(RuntimeError, match="no longer owns"):
            server.dispatch("snapshot", {}, client="A", connection=a_server)
    finally:
        natural_release.set()
        old_command.join(timeout=2.0)
        takeover.join(timeout=2.0)
        for connection in (a_server, a_peer, b_server, b_peer):
            try:
                connection.close()
            except OSError:
                pass
        monkeypatch.setattr(transport, "write_words", original_write)
        monkeypatch.setattr(streamer, "safe", original_safe)
        server.server_close()
        streamer.close()


def test_remote_replays_device_path_with_short_done_poll() -> None:
    geom = _sequence_geometry()
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=source.target)
    with _server(streamer) as server:
        client = _client(server)
        try:
            client.load(program, source=source, rows=((1,),))
            client.fire()
            report = client.wait_done(1.0)
            assert report is not None
            assert report.status_reads == (4, 4)
            safe = client.safe()
            assert safe.stable
            state = client.applied()
            assert state is not None
            assert state.source == source
            assert state.rows == ((1,),)
            assert state.cycles == 1
        finally:
            client.close()


def test_remote_safe_interrupts_forever_fire_on_the_same_connection() -> None:
    geom = _sequence_geometry()
    source = _sequence()
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6, target=source.target)
    with _server(streamer) as server:
        client = _client(server)
        try:
            client.load(program)
            client.fire(cycles=None)
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
    geom = _sequence_geometry()
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=source.target)
    with _server(streamer) as server:
        client = _client(server)
        try:
            client.load(program, source=source, rows=((1,), (2,)))
            assert client.snapshot()["opened"] is True
            assert client.cursor() == 0
            assert client.applied() is not None
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
    assert "cycles=1" in output
    assert "reloaded_before_fire=False" in output
    assert "ZLC SNAPSHOT client=127.0.0.1:" in output
    assert "ZLC CURSOR client=127.0.0.1:" in output
    assert "ZLC APPLIED client=127.0.0.1:" in output
    assert "rows=2" in output
    assert "ZLC DONE" in output
    assert "ZLC STOP/SAFE" in output
    assert "stable=True" in output
    assert "ZLC CLOSE" in output
    assert "ZLC CLIENT DISCONNECTED" in output
    assert "AnalogStep" not in output


def test_a_new_client_takes_the_board_and_the_old_connection_is_dropped(capsys) -> None:
    """Arrival decides ownership, which is why nothing has to detect death.

    The client being replaced here is perfectly alive, and it is still the one
    that loses -- because the case that actually happens in the lab is a
    reconnect after a sleeping laptop, where the incumbent is a ghost that no
    amount of asking can be relied on to unmask.
    """

    geom = replace(StreamerParams(), max_edges=8, bank_size=2)
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=geom), geom, 50e6, target=_BOARD_TARGET
    )
    with _server(streamer) as server:
        previous = _client(server)
        previous_address = server.owner_status()[0]
        newcomer = _client(server)
        try:
            assert server.owner_status()[0] not in {None, previous_address}
            assert newcomer.snapshot()["opened"] is True
            # The dropped connection learns it lost the next time it speaks,
            # and is told what happened rather than that a reply was malformed.
            with pytest.raises(OSError, match="that one now holds the board"):
                previous.snapshot()
        finally:
            newcomer.close()
            previous.disconnect()
            newcomer.disconnect()

    output = capsys.readouterr().out
    assert "ZLC CLIENT REPLACED" in output
    assert "by=127.0.0.1:" in output


def test_a_quiet_owner_is_never_disconnected_for_being_quiet() -> None:
    """Sending nothing is what editing looks like, and it must cost nothing.

    An idle timer used to SAFE the outputs and release the board after five
    minutes without a request, which is a description of somebody editing a
    pulse.  Nothing in the handler measures silence any more -- there is no
    deadline on the read at all -- so this asserts what the socket sees: no
    request, no reply, and the board still firing and still owned.
    """

    geom = _sequence_geometry()
    source = _sequence()
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=False)
    streamer = PulseStreamer(transport, geom, 50e6, target=source.target)
    with _server(streamer) as server:
        client = _client(server)
        try:
            client.load(program)
            client.fire(cycles=None)
            owner = server.owner_status()[0]
            time.sleep(0.3)  # not one word from us
            assert streamer.snapshot()["firing"] is True
            assert server.owner_status()[0] == owner
            assert client.snapshot()["firing"] is True
        finally:
            client.close()
            client.disconnect()


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


def test_only_the_configured_uart_is_probed(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(
        remote_module,
        "_probe_uart_port",
        lambda port, timeout, **_kwargs: calls.append((port, timeout)),
    )

    result = resolve_backend(
        "auto",
        uart_port="COM9",
        target=_BOARD_TARGET,
        params=StreamerParams(),
        clock_hz=50e6,
    )

    assert result.backend == "uart"
    assert result.uart_port == "COM9"
    assert calls == [("COM9", 0.5)]


def test_auto_enumerates_usb_first_then_chooses_the_word63_match(monkeypatch) -> None:
    calls: list[str] = []

    def probe(port, _timeout, **_kwargs):
        calls.append(port)
        if port == "COM7":
            raise TimeoutError("not the pulse board")

    monkeypatch.setattr(remote_module, "_probe_uart_port", probe)
    descriptors = (
        SimpleNamespace(device="COM3", vid=None, pid=None),
        SimpleNamespace(device="COM7", vid=0x1234, pid=0x5678),
        SimpleNamespace(device="COM8", vid=None, pid=None),
    )

    result = resolve_backend(
        "auto",
        target=_BOARD_TARGET,
        params=StreamerParams(),
        clock_hz=50e6,
        port_provider=lambda: descriptors,
    )

    assert result.backend == "uart"
    assert result.uart_port == "COM3"
    assert calls == ["COM7", "COM3"]
    assert result.attempts == (
        "COM7: timeout",
        "COM3: word63 fingerprint matched",
    )


@pytest.mark.parametrize(
    "requested, expected",
    (("auto", "jtag-axi"), ("uart", "error"), ("jtag-axi", "jtag-axi")),
)
def test_backend_without_any_serial_port_falls_back_or_fails_loudly(
    monkeypatch, requested: str, expected: str
) -> None:
    monkeypatch.setattr(
        remote_module,
        "_probe_uart_port",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an undeclared UART was probed")
        ),
    )
    if expected == "error":
        with pytest.raises(BackendResolutionError, match="no UART ports detected"):
            resolve_backend(
                requested,
                target=_BOARD_TARGET,
                params=StreamerParams(),
                clock_hz=50e6,
                port_provider=lambda: (),
            )
        return
    result = resolve_backend(
        requested,
        target=_BOARD_TARGET,
        params=StreamerParams(),
        clock_hz=50e6,
        port_provider=lambda: (),
    )
    assert result.backend == expected


def test_server_releases_a_fixed_port_for_the_next_run() -> None:
    port = 0
    for _ in range(2):
        transport = MemoryRegisterTransport(geom=StreamerParams())
        streamer = PulseStreamer(transport, StreamerParams(), 50e6, target=_BOARD_TARGET)
        server = PulseRemoteServer(("127.0.0.1", port), streamer)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = _client(server)
        try:
            client.describe()
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)
            streamer.close()
        assert not thread.is_alive()


def test_startup_failure_prints_the_real_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        remote_module,
        "serve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError(10048, "address already in use")
        ),
    )

    assert remote_module._main(["--backend", "memory", "--port", "0"]) == 3
    error = capsys.readouterr().err
    assert "OSError" in error
    assert "address already in use" in error
    assert "did not enter LISTENING" in error


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
    mismatch_streamer = PulseStreamer(
        mismatch, StreamerParams(), 50e6, target=_BOARD_TARGET
    )
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

    transport = transport_module.UartRegisterTransport(link=CrcLink())
    transport.start()
    try:
        with pytest.raises(UartError) as failure:
            transport.read_word(CtrlWords.LAYOUT_ID)
        assert remote_module._probe_failure_reason(failure.value) == "CRC error"
    finally:
        transport.close()


def test_uart_probe_reuses_pulse_streamer_word63_open(monkeypatch, tmp_path) -> None:
    params = StreamerParams()
    records: list[tuple[str, int, float]] = []
    open_calls: list[int] = []

    class FakeTransport:
        def __init__(self, *, port, baud, action_timeout):
            records.append((port, baud, action_timeout))
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
        target=_BOARD_TARGET,
        params=params,
        clock_hz=50e6,
        uart_port="COM7",
    )

    assert result.backend == "uart"
    assert result.uart_port == "COM7"
    assert result.attempts == ("COM7: word63 fingerprint matched",)
    assert records == [("COM7", 3_000_000, 0.5)]
    assert len(open_calls) == 1


def test_explicit_uart_failure_does_not_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        remote_module,
        "_probe_uart_port",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TimeoutError("UART reply timed out")
        ),
    )
    with pytest.raises(BackendResolutionError, match="explicit UART backend failed"):
        resolve_backend(
            "uart",
            uart_port="COM3",
            target=_BOARD_TARGET,
            params=StreamerParams(),
            clock_hz=50e6,
        )


def test_server_cli_defaults_to_auto_and_accepts_explicit_backends() -> None:
    parser = remote_module.build_arg_parser()

    assert parser.parse_args([]).backend == "auto"
    assert parser.parse_args(["--backend", "jtag-axi"]).backend == "jtag-axi"
    assert parser.parse_args(["--backend", "uart", "--uart-port", "COM3"]).uart_port == "COM3"
    # No knob decides when a quiet client is disconnected, because nothing does.
    assert not hasattr(parser.parse_args([]), "client_idle_timeout")


def test_server_refuses_deployment_config_fallback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        remote_module,
        "load_streamer_config",
        lambda: {
            "source": None,
            "warnings": ("using built-in defaults",),
        },
    )
    assert remote_module._main(["--check-config"]) == 2
    assert "without fallback" in capsys.readouterr().err


def test_main_logs_final_backend_reason_attempts_and_jtag_note(monkeypatch, capsys, tmp_path) -> None:
    resolution = BackendResolution(
        "auto",
        "jtag-axi",
        None,
        "auto fallback to jtag-axi after UART probe: COM7: timeout",
        ("COM7: timeout",),
    )

    class FakeJtagTransport(MemoryRegisterTransport):
        def __init__(self, *, state_dir):
            del state_dir
            super().__init__(geom=StreamerParams())

    def fake_resolve(*args, **kwargs):
        return resolution

    monkeypatch.setattr(remote_module, "resolve_backend", fake_resolve)
    monkeypatch.setattr(transport_module, "VivadoAxiRegisterTransport", FakeJtagTransport)
    monkeypatch.setattr(remote_module, "serve", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert remote_module._main(["--backend", "auto", "--state-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "ZLC SERVER STARTING" in output
    assert "endpoint=0.0.0.0:18861" in output
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
    geom = _sequence_geometry()
    source = _sequence(slotted=True)
    program = compile_sequence(source, geom, 50e6)
    transport = MemoryRegisterTransport(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=source.target)
    with _server(streamer) as server:
        client_a = _client(server)
        client_a.load(program, source=source, rows=((1,),))
        client_a.disconnect()

        client_b = _client(server)
        try:
            state = client_b.applied()
            assert state is not None
            assert state.source == source
            assert state.rows == ((1,),)
            assert state.cycles == 1
            assert state.source is not None
            rebuilt = compile_sequence(state.source, geom, 50e6)
            assert pack_program(rebuilt, geom) == pack_program(state.program, geom)
            assert pack_scan_rows(state.rows, geom, 0, 0, state.cycles) == pack_scan_rows(((1,),), geom, 0, 0, 1)
        finally:
            client_b.close()
    assert streamer.applied() is None
    assert "ZLC AUTO-SAFE" in capsys.readouterr().out


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
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=geom, auto_done=True),
        geom,
        50e6,
        target=_BOARD_TARGET,
    )
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


def test_a_poll_is_logged_only_when_its_answer_changes(capsys) -> None:
    """A poll is a question, not an event.

    wait_done is asked every 10 ms by a client that owns its own poll loop, so
    one five-second shot printed four hundred identical "state=PENDING" lines
    and buried the run they were about.  The same rule makes CURSOR useful for
    the first time: a cursor that stays at 3 says nothing, and a cursor that
    becomes 4 is the scan advancing.
    """

    from zlc_pulse.remote import _forget_polls, _server_log_change

    client = "127.0.0.1:9"
    _forget_polls(client)
    try:
        for _ in range(400):
            _server_log_change("WAIT DONE", client=client, detail="state=PENDING")
        printed = capsys.readouterr().out.splitlines()
        assert len(printed) == 1, printed[:3]

        _server_log_change("WAIT DONE", client=client, detail="state=DONE")
        assert len(capsys.readouterr().out.splitlines()) == 1

        for value in (3, 3, 3, 4, 4, 5):
            _server_log_change("CURSOR", client=client, detail=f"value={value}")
        assert len(capsys.readouterr().out.splitlines()) == 3, "3, 4, 5"
    finally:
        _forget_polls(client)


def test_a_client_that_leaves_takes_its_poll_memory_with_it() -> None:
    """Otherwise a server that runs for months grows one entry per connection."""

    from zlc_pulse.remote import _LAST_POLL, _forget_polls, _server_log_change

    _server_log_change("CURSOR", client="10.0.0.1:1", detail="value=1")
    assert any(key[0] == "10.0.0.1:1" for key in _LAST_POLL)
    _forget_polls("10.0.0.1:1")
    assert not any(key[0] == "10.0.0.1:1" for key in _LAST_POLL)
