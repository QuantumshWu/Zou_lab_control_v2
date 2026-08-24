from __future__ import annotations

import os
import re
from pathlib import Path
import subprocess
import sys

import pytest

from zlc_pulse.fpga import (
    DEFAULT_CONFIG_PATH,
    FpgaPartProfile,
    emit_geometry_vh,
    estimate_resources,
    load_streamer_config,
    solve_capacity,
)
from zlc_pulse.wire import default_params


ROOT = Path(__file__).resolve().parents[1]


# Files required by the checked-in FPGA build and simulation flow. Their
# meaning is verified below by geometry projection and launcher tests; checkout
# line-ending choices are not experimental facts.
DEPLOYMENT_ASSETS = (
    "fpga/board_config/board.xdc",
    "fpga/board_config/streamer_config.json",
    "fpga/build/geom.tcl",
    "fpga/pulse_streamer/create_project.tcl",
    "fpga/pulse_streamer/diagnose_hw_target.tcl",
    "fpga/pulse_streamer/program_flash.tcl",
    "fpga/pulse_streamer/program_fpga.tcl",
    "fpga/pulse_streamer/sim/.gitignore",
    "fpga/pulse_streamer/sim/replay_t.vh",
    "fpga/pulse_streamer/sim/replay_t_frame.vh",
    "fpga/pulse_streamer/sim/tb_1tick.v",
    "fpga/pulse_streamer/sim/tb_bus_delay.v",
    "fpga/pulse_streamer/sim/tb_da_ttl_align.v",
    "fpga/pulse_streamer/sim/tb_delay_compact.v",
    "fpga/pulse_streamer/sim/tb_delay_sched.v",
    "fpga/pulse_streamer/sim/tb_evt_depth.v",
    "fpga/pulse_streamer/sim/tb_gapsweep.v",
    "fpga/pulse_streamer/sim/tb_loop.v",
    "fpga/pulse_streamer/sim/tb_ramp_scan.v",
    "fpga/pulse_streamer/sim/tb_real_engine.v",
    "fpga/pulse_streamer/sim/tb_scan_wrap.v",
    "fpga/pulse_streamer/sim/tb_t_ff.v",
    "fpga/pulse_streamer/tb_uart_pipeline.v",
    "fpga/pulse_streamer/tb_uart_read_tap.v",
    "fpga/pulse_streamer/zlc_edge_streamer.v",
    "fpga/pulse_streamer/zlc_pulse_streamer_top.v",
    "fpga/pulse_streamer/zlc_uart_bridge.v",
)


def test_deployment_assets_are_present() -> None:
    assert not (ROOT / "fpga/pulse_streamer/host").exists()
    missing = [relative for relative in DEPLOYMENT_ASSETS if not (ROOT / relative).is_file()]
    assert not missing


def _defines(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2)
        for line in text.splitlines()
        if (match := re.match(r"\s*`define\s+(\w+)\s+(.+?)\s*$", line))
    }


def _assert_geometry_header_matches(text: str) -> None:
    assert _defines(text) == _defines(emit_geometry_vh(default_params()))


def test_deployed_geometry_header_matches_wire_projection() -> None:
    header = (ROOT / "fpga/pulse_streamer/zlc_geometry.vh").read_text(encoding="utf-8")
    _assert_geometry_header_matches(header)


def test_geometry_header_regenerates_through_the_documented_package_command(tmp_path: Path) -> None:
    output = tmp_path / "zlc_geometry.vh"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        ""
        if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(ROOT.parents[1]), str(ROOT / "src")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zou_lab_control",
            "fpga",
            "--emit-geometry-vh",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    header = output.read_text(encoding="utf-8")
    _assert_geometry_header_matches(header)
    assert "zlc fpga --emit-geometry-vh" in header


def test_geometry_guard_rejects_macro_and_fingerprint_mutations() -> None:
    header = (ROOT / "fpga/pulse_streamer/zlc_geometry.vh").read_text(encoding="utf-8")
    for name, replacement in (
        ("ZLC_CHANNEL_COUNT", "63"),
        ("ZLC_LAYOUT_FINGERPRINT", "32'h00000000"),
    ):
        mutated = re.sub(
            rf"(`define\s+{name}\s+)([^\n]+)",
            rf"\g<1>{replacement}",
            header,
        )
        with pytest.raises(AssertionError):
            _assert_geometry_header_matches(mutated)


def test_deployed_config_is_the_default_geometry_source(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ZLC_PS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = DEFAULT_CONFIG_PATH.resolve()
    assert expected.is_file()
    loaded = load_streamer_config()
    assert loaded["source"].resolve() == expected
    assert loaded["params"] == default_params()


def test_frozen_35t_uses_98_percent_without_weakening_the_90_percent_default() -> None:
    config = load_streamer_config()
    assert config["target_pct"] == 98.0

    frozen = estimate_resources(
        config["params"], part=config["fpga_part"], target_pct=config["target_pct"]
    )
    assert frozen["lut"] == {
        "used": 20075,
        "budget": 20384,
        "total": 20800,
        "pct": 96.5,
        "ok": True,
    }
    at_default = estimate_resources(config["params"], part=config["fpga_part"])
    assert at_default["lut"]["ok"] is False
    assert at_default["lut"]["budget"] == 18720

    with pytest.raises(ValueError, match=r"90% planning target: LUT 20075 > 18720"):
        solve_capacity(config["fpga_part"])

    planning = solve_capacity("xc7a50t")
    assert planning.all_within_budget()
    assert planning.resource_report["lut"]["budget"] == 29340

    solved = solve_capacity(
        config["fpga_part"], target_pct=config["target_pct"]
    )
    assert solved.all_within_budget()
    assert solved.resource_report == estimate_resources(
        solved.params, part=config["fpga_part"], target_pct=config["target_pct"]
    )


def test_capacity_search_uses_the_estimators_fixed_ramb36_cost() -> None:
    # At 39 tiles the old solver's private +1 formula admitted 4096 edges,
    # while the estimator's routed +3 accounting correctly reports 41.  One
    # estimator authority must instead choose the next 2048-edge geometry.
    part = FpgaPartProfile("boundary", 39, 100000, 100000, 1000, 100000)
    solved = solve_capacity(
        part,
        target_pct=100,
        max_edges_cap=4096,
        engine_logic_luts=0,
        engine_ff=0,
        engine_dsp=0,
    )
    assert solved.params.max_edges == 2048
    assert solved.ramb36_used == 31
    assert solved.ramb36_budget == 39
    assert solved.all_within_budget()
    assert solved.resource_report == estimate_resources(
        solved.params,
        part=part,
        target_pct=100,
        engine_logic_luts=0,
        engine_ff=0,
        engine_dsp=0,
    )


def test_clock_and_safe_pin_boundary_are_explicit() -> None:
    xdc = (ROOT / "fpga/board_config/board.xdc").read_text(encoding="utf-8")
    top = (ROOT / "fpga/pulse_streamer/zlc_pulse_streamer_top.v").read_text(
        encoding="utf-8"
    )
    assert re.search(r"create_clock\s+-period\s+20(?:\.0+)?\s+.*get_ports\s+clk", xdc)
    assert "eng_reset ? 1'b0" in top
    assert "zlc_physical_active && clk_en" in top
    assert "bus_out_final" in top
    assert "eng_reset ? bus_safe_pack : zlc_bus_out" in top


def test_public_done_waits_for_the_physical_tail_and_errors_are_sticky() -> None:
    engine = (ROOT / "fpga/pulse_streamer/zlc_edge_streamer.v").read_text(
        encoding="utf-8"
    )
    top = (ROOT / "fpga/pulse_streamer/zlc_pulse_streamer_top.v").read_text(
        encoding="utf-8"
    )
    assert "draining <= 1'b1" in engine
    assert "if (!delay_runtime_busy)" in engine
    assert "done <= 1'b1" in engine
    playback_start = engine.index(
        "else if (running) begin", engine.index("start_event && !running")
    )
    playback_end = engine.index("else if (draining) begin", playback_start)
    assert "underflow <= 1'b0" not in engine[playback_start:playback_end]
    assert "overflow <= 1'b1" in engine
    assert "zlc_overflow ? {1'b0, ST_ERROR}" in top
    assert "protocol_error ? ST_LINK_ERROR" in top
    assert "bank_ready[0] && bank_chunk0 == 0" in engine


def test_uart_decoder_releases_truncated_frames_and_rejects_bounds() -> None:
    bridge = (ROOT / "fpga/pulse_streamer/zlc_uart_bridge.v").read_text(
        encoding="utf-8"
    )
    assert "FRAME_TIMEOUT_CYCLES" in bridge
    assert "frame_idle >= FRAME_TIMEOUT_CYCLES-1" in bridge
    assert "{rx_byte,f_count[7:0]} > FRAME_WORDS" in bridge
    assert "|f_addr[31:ADDR_WORD_WIDTH]" in bridge
    assert "17'd64" in bridge


def test_fpga_launchers_use_the_package_wire_cli() -> None:
    # In bin\, with everything else a human clicks.  They still drive THIS
    # layer's board, which is why this layer is what checks them.
    launchers = ROOT.parents[1] / "bin"
    build = (launchers / "build_and_program.bat").read_text(encoding="utf-8")
    estimate = (launchers / "estimate_resources.bat").read_text(encoding="utf-8")
    assert "fpga.pulse_streamer.host" not in build
    assert "fpga.pulse_streamer.host" not in estimate
    # Both hardware wrappers use the installed product manifest command; no
    # launcher imports a layer module or mutates PYTHONPATH.
    assert "zou_lab_control fpga" in build
    assert "zou_lab_control fpga" in estimate
