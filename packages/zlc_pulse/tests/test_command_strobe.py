"""Pulse command completion, resident replay and RTL numeric oracles."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from zlc_pulse import compile_sequence, pulse_target_from_xdc
from zlc_pulse import device
from zlc_pulse.device import PulseStreamer
from zlc_pulse.transport import MemoryRegisterTransport
from zlc_pulse.wire import CMD_FIRE, CMD_LOAD, CMD_SAFE, CtrlWords, STATUS_RUNNING, StreamerParams

from test_wire_device import _sequence


RTL = Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_top.v"
RTL_DIR = RTL.parent
_BOARD_TARGET = pulse_target_from_xdc()


@pytest.mark.parametrize(
    ("top", "sources", "defines", "marker"),
    (
        (
            "tb_delay_sched",
            ("zlc_edge_streamer.v", "sim/tb_delay_sched.v"),
            (),
            "DELAY-SCHED-PHYSICAL-DONE-OK",
        ),
        (
            "tb_evt_depth",
            ("zlc_edge_streamer.v", "sim/tb_evt_depth.v"),
            (),
            "EVT-DEPTH-STICKY-OVERFLOW-OK",
        ),
        (
            "tb_uart_pipeline",
            ("zlc_uart_bridge.v", "tb_uart_pipeline.v"),
            (),
            "UART-PIPELINE-WATCHDOG-BOUNDS-OK",
        ),
        (
            "tb_safe_gate",
            (
                "zlc_uart_bridge.v",
                "zlc_edge_streamer.v",
                "zlc_pulse_streamer_top.v",
                "sim/tb_t_ff.v",
            ),
            ("ZLC_IVERILOG",),
            "TOP-SAFE-PIN-GATE-OK",
        ),
    ),
)
def test_rtl_contracts_execute_with_nonzero_failure(
    tmp_path: Path,
    top: str,
    sources: tuple[str, ...],
    defines: tuple[str, ...],
    marker: str,
) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("RTL simulation not executed: iverilog and vvp are not installed")

    image = tmp_path / f"{top}.vvp"
    compile_result = subprocess.run(
        [
            iverilog,
            "-g2012",
            "-Wall",
            "-s",
            top,
            "-I",
            str(RTL_DIR),
            *[f"-D{name}" for name in defines],
            "-o",
            str(image),
            *[str(RTL_DIR / source) for source in sources],
        ],
        cwd=RTL_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr

    run_result = subprocess.run(
        [vvp, str(image)],
        cwd=RTL_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    transcript = run_result.stdout + run_result.stderr
    assert run_result.returncode == 0, transcript
    assert marker in transcript, transcript


def test_vivado_rtl_matrix_requires_each_numeric_oracle(tmp_path: Path) -> None:
    """Run the maintained xsim matrix when Vivado and generated BRAM IP exist."""

    suffix = ".bat" if os.name == "nt" else ""
    configured = os.environ.get("ZLC_PS_VIVADO_BIN", "")
    tool_dir = Path(configured).resolve().parent if configured else Path("C:/Xilinx/Vivado/2019.1/bin")
    tools = {
        name: Path(shutil.which(name) or tool_dir / f"{name}{suffix}")
        for name in ("xvlog", "xelab", "xsim")
    }
    if any(not path.is_file() for path in tools.values()):
        pytest.skip("Vivado xsim matrix not executed: xvlog/xelab/xsim are unavailable")

    ip_root = RTL_DIR.parent / "build/ps/ps.srcs/sources_1/ip"
    common = ip_root / "blk_mem_gen_edge_tick/simulation/blk_mem_gen_v8_4.v"
    tick = ip_root / "blk_mem_gen_edge_tick/sim/blk_mem_gen_edge_tick.v"
    mask = ip_root / "blk_mem_gen_edge_mask/sim/blk_mem_gen_edge_mask.v"
    if any(not path.is_file() for path in (common, tick, mask)):
        pytest.skip("Vivado xsim matrix not executed: generated edge-BRAM models are absent")

    engine_markers = {
        "tb_1tick": "ONE-TICK-OK",
        "tb_bus_delay": "BUS-DELAY-OK",
        "tb_da_ttl_align": "DA-TTL-ALIGN-OK",
        "tb_delay_compact": "COMPACT-MAP-OK",
        "tb_delay_sched": "DELAY-SCHED-PHYSICAL-DONE-OK",
        "tb_evt_depth": "EVT-DEPTH-STICKY-OVERFLOW-OK",
        "tb_gapsweep": "GAPSWEEP-OK",
        "tb_loop": "LOOP-OK",
        "tb_ramp_scan": "RAMP-SCAN-OK",
        "tb_scan_wrap": "SEAMLESS-OK",
    }
    markers = {
        **engine_markers,
        "tb_uart_pipeline": "UART-PIPELINE-WATCHDOG-BOUNDS-OK",
        "tb_uart_read_tap": "UART-READ-WRITE-LASTWORD-LAYOUT-OK",
    }
    compile_result = subprocess.run(
        [
            str(tools["xvlog"]),
            str(RTL_DIR / "zlc_edge_streamer.v"),
            str(common),
            str(tick),
            str(mask),
            *[str(RTL_DIR / "sim" / f"{name}.v") for name in engine_markers],
            str(RTL_DIR / "zlc_uart_bridge.v"),
            str(RTL_DIR / "tb_uart_pipeline.v"),
            str(RTL_DIR / "tb_uart_read_tap.v"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr

    forbidden = re.compile(r"Fatal:|\*\*FAIL\*\*|\*\*BAD\*\*|\*\*LATE\*\*|RAMP-SCAN-BAD")
    for top, marker in markers.items():
        snapshot = f"zlc_{top}"
        elaborate = subprocess.run(
            [str(tools["xelab"]), f"work.{top}", "-s", snapshot],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert elaborate.returncode == 0, elaborate.stdout + elaborate.stderr
        simulate = subprocess.run(
            [str(tools["xsim"]), snapshot, "-runall"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        transcript = simulate.stdout + simulate.stderr
        assert simulate.returncode == 0, transcript
        assert marker in transcript and forbidden.search(transcript) is None, transcript
        if top == "tb_1tick":
            assert "SHORT-ONE-SHOT-OK ticks=1" in transcript
            assert "SHORT-ONE-SHOT-OK ticks=2" in transcript


class _Recorder(MemoryRegisterTransport):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.commands: list[int] = []

    def command(self, code, command_id, **kwargs):
        self.commands.append(int(code))
        return super().command(code, command_id, **kwargs)


def _streamer(*, auto_done: bool = True) -> tuple[PulseStreamer, _Recorder, object]:
    geom = StreamerParams()
    transport = _Recorder(geom=geom, auto_done=auto_done)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    return streamer, transport, program


def test_every_fire_in_a_scan_loop_reaches_the_board() -> None:
    """A loaded finite application can be fired repeatedly without a lost edge."""

    streamer, transport, program = _streamer()
    streamer.load(program, rows=((1,),))
    for _ in range(3):
        streamer.fire(run_repeats=1)
        streamer.wait_done(2.0)

    assert transport.commands.count(CMD_FIRE) == 3


def test_loading_the_same_resident_application_does_not_upload_again() -> None:
    streamer, transport, program = _streamer()
    streamer.load(program, rows=((1,),))
    writes = len(transport.write_batches)
    streamer.load(program, rows=((1,),))
    assert transport.commands.count(CMD_LOAD) == 1
    assert len(transport.write_batches) == writes


def test_load_reports_a_loader_that_never_acknowledges() -> None:
    """FIRE requires a fully loaded resident program.

    A load the board never completed must raise here; otherwise every later
    fire is a silent no-op and the host reports a normal DoneReport.
    """

    geom = StreamerParams()

    class _NeverLoads(MemoryRegisterTransport):
        def command(self, code, command_id, **kwargs):
            if code == CMD_LOAD:
                raise TimeoutError("LOAD completion was not acknowledged")
            return super().command(code, command_id, **kwargs)

    transport = _NeverLoads(geom=geom, auto_done=True)
    streamer = PulseStreamer(transport, geom, 50e6, target=_BOARD_TARGET)
    streamer.open()
    program = compile_sequence(_sequence(slotted=True), geom, 50e6)
    device.LOAD_TIMEOUT = 0.05
    try:
        with pytest.raises(TimeoutError, match="LOAD completion"):
            streamer.load(program, rows=((1,),))
    finally:
        device.LOAD_TIMEOUT = 5.0


def test_replaying_a_scan_table_re_arms_the_banks_at_point_zero() -> None:
    """A run restarts at scan point 0, so the banks must hold chunks 0/1.

    During a run the observer streams later chunks over the banks; ``_refill``
    below is that same observer path, driven deterministically.  Without
    re-arming, the replay's point 0 is not resident and the engine stalls with
    no output while every command still looks accepted.
    """

    streamer, transport, program = _streamer()
    rows = tuple((value,) for value in range(1, 3 * StreamerParams().bank_size))
    streamer.load(program, rows=rows)
    streamer.fire(run_repeats=1)
    streamer.wait_done(2.0)

    streamer._refill(2 * StreamerParams().bank_size)  # the observer's own path
    assert transport.read_word(CtrlWords.BANK0_CHUNK) != 0

    streamer.fire(run_repeats=1)
    streamer.wait_done(2.0)
    assert transport.read_word(CtrlWords.BANK0_CHUNK) == 0
    assert transport.read_word(CtrlWords.BANK1_CHUNK) == 1
    assert transport.read_word(CtrlWords.BANK_READY) == 0b11


def test_load_uses_one_safe_then_one_completion_ack_after_the_image() -> None:
    streamer, transport, program = _streamer()
    streamer.load(program, rows=((1,),))
    assert transport.commands == [CMD_SAFE, CMD_LOAD]
    assert len(transport.write_batches) == 3
    assert transport.write_batches[0][-1] == (CtrlWords.COMMAND, CMD_SAFE)
    assert transport.write_batches[1][-1] == (CtrlWords.BANK_READY, 0b11)
    assert transport.write_batches[2][-1] == (CtrlWords.COMMAND, CMD_LOAD)


def test_safe_refuses_missing_or_error_completion_without_guessing(monkeypatch) -> None:
    class _NeverSafe(_Recorder):
        def command(self, code, command_id, **kwargs):
            if code == CMD_SAFE:
                raise TimeoutError("SAFE completion was not acknowledged")
            return super().command(code, command_id, **kwargs)

    transport = _NeverSafe(geom=StreamerParams(), auto_done=True)
    streamer = PulseStreamer(transport, StreamerParams(), 50e6, target=_BOARD_TARGET)
    streamer.open()
    with pytest.raises(TimeoutError, match="SAFE completion"):
        streamer.safe()


def test_a_board_describes_itself_rather_than_letting_a_client_assume(monkeypatch) -> None:
    """The protocol could open, load and fire a board but not ask what it was.

    So a client wanting ports, pins or a clock had to read its own XDC and
    config and hope they were the ones the board was built from -- the exact
    mistake the layout handshake exists to catch, made one layer up.
    """

    from zlc_pulse import load_streamer_config
    from zlc_pulse.device import BoardDescription, PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport

    config = load_streamer_config()
    geometry = config["params"]
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=geometry, auto_done=True),
        geometry,
        config["clock_hz"],
        target=_BOARD_TARGET,
    )

    import zlc_pulse.manifest as manifest

    def _no_late_xdc_read(*_args, **_kwargs):
        raise AssertionError("describe re-read process-global XDC state")

    monkeypatch.setattr(manifest, "pulse_target_from_xdc", _no_late_xdc_read)

    # Nothing is proven before the handshake, so nothing is claimed.
    with pytest.raises(RuntimeError):
        streamer.describe()

    streamer.open()
    try:
        described = streamer.describe()
        assert isinstance(described, BoardDescription)
        assert described.geometry == geometry
        assert len(described.target.raw_lanes) == geometry.channel_count
        assert described.geometry.channel_count == geometry.channel_count
        assert described.geometry.bus_count == geometry.bus_count
        assert described.clock_hz == float(config["clock_hz"])
        assert described.time_step_ns == 1e9 / float(config["clock_hz"])
        # A pin for every lane: the names an operator wires against.
        # One pin map, on the target that owns it.  It used to be carried
        # twice, and the wire silently emptied the nested copy.
        assert set(described.target.package_pins) == set(described.target.raw_lanes)
        assert all(pin.strip() for pin in described.target.package_pins.values())
    finally:
        streamer.close()


def test_the_description_survives_the_wire() -> None:
    """Ports, pins and the clock have to arrive intact or the client still guesses."""

    from zlc_pulse.remote import REMOTE_METHODS, decode_tree, encode_tree
    from zlc_pulse import load_streamer_config
    from zlc_pulse.device import PulseStreamer
    from zlc_pulse.transport import MemoryRegisterTransport

    assert "describe" in REMOTE_METHODS

    config = load_streamer_config()
    geometry = config["params"]
    streamer = PulseStreamer(
        MemoryRegisterTransport(geom=geometry, auto_done=True),
        geometry,
        config["clock_hz"],
        target=_BOARD_TARGET,
    )
    streamer.open()
    try:
        original = streamer.describe()
        restored = decode_tree(encode_tree(original))

        assert restored.target == original.target
        assert dict(restored.target.package_pins) == dict(
            original.target.package_pins
        ), "the wire dropped the pin map"

        assert restored.clock_hz == original.clock_hz
        assert restored.geometry == original.geometry
        assert restored.layout_fingerprint == original.layout_fingerprint
        # The port labels are what an operator reads in an editor.
        assert [port.label for port in restored.target.ports] == [
            port.label for port in original.target.ports
        ]
    finally:
        streamer.close()


def test_the_host_refills_one_chunk_ahead_of_the_cursor() -> None:
    """The one-ahead contract shared by the RTL and host refill path.

    When the engine crosses from chunk c into c+1 it frees the bank holding c
    and the host refills it with c+2.  The host used to wait for the cursor to
    REACH the chunk it was about to write -- one whole bank late -- so any scan
    longer than the two pre-armed chunks stalled at every chunk boundary while
    the host caught up.
    """

    bank = StreamerParams().bank_size
    streamer, _transport, program = _streamer(auto_done=False)
    streamer.load(
        program,
        rows=tuple((value,) for value in range(1, 4 * bank)),
    )
    try:
        streamer.fire(run_repeats=1, scan_repeats=2)
        # Drive the observer's private refill path deterministically below.
        streamer._stop_worker()
        streamer._stop.clear()
        armed = streamer.snapshot()["scan_next_chunk"]
        # The cursor has only entered chunk 1; chunk 2 must already be on its way.
        streamer._refill(bank)
        assert streamer.snapshot()["scan_next_chunk"] > armed
        # CURSOR is cumulative, so the first row of sweep 2 is 4*bank rather
        # than another zero.  Refill derives the sweep directly even if no
        # observer poll happened near the table boundary.
        streamer._refill(3 * bank)
        before_wrap = streamer.snapshot()["scan_next_chunk"]
        streamer._refill(4 * bank)
        assert streamer.snapshot()["scan_next_chunk"] > before_wrap
    finally:
        streamer.safe()


def test_command_retry_returns_original_ack_without_starting_another_run() -> None:
    transport = MemoryRegisterTransport(auto_done=False)
    transport.start()
    transport.command(CMD_LOAD, 1)
    first = transport.command(CMD_FIRE, 2, run_repeats=3)
    transport.publish_execution_readback(status=STATUS_RUNNING, cursor=5)
    assert transport.command(CMD_FIRE, 2, run_repeats=3) == first
    assert transport.cursor_value == 5, "a retried ACK must not reset/replay the engine"
    assert transport.command(CMD_SAFE, 3) == (0, 0)
    assert transport.command(CMD_FIRE, 4, run_repeats=2)[0] == STATUS_RUNNING
