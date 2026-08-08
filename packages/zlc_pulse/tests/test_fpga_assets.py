from __future__ import annotations

import re
from pathlib import Path
import subprocess
import sys

import pytest

from zlc_pulse.fpga import DEFAULT_CONFIG_PATH, emit_geometry_vh, load_streamer_config
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
    "fpga/pulse_streamer/sim/tb_da_clk_phase.v",
    "fpga/pulse_streamer/sim/tb_da_ttl_align.v",
    "fpga/pulse_streamer/sim/tb_delay_compact.v",
    "fpga/pulse_streamer/sim/tb_delay_sched.v",
    "fpga/pulse_streamer/sim/tb_edge_streamer.v",
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
    result = subprocess.run(
        [sys.executable, "-m", "zlc_pulse.fpga", "--emit-geometry-vh", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    header = output.read_text(encoding="utf-8")
    _assert_geometry_header_matches(header)
    assert "python -m zlc_pulse.fpga --emit-geometry-vh" in header


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


def test_deployed_config_is_the_default_geometry_source(monkeypatch) -> None:
    monkeypatch.delenv("ZLC_PS_CONFIG", raising=False)
    monkeypatch.chdir(ROOT)
    expected = (ROOT / "fpga/board_config/streamer_config.json").resolve()
    assert DEFAULT_CONFIG_PATH.resolve() == expected
    loaded = load_streamer_config()
    assert loaded["source"].resolve() == expected
    assert loaded["params"] == default_params()


def test_fpga_launchers_use_the_package_wire_cli() -> None:
    # In bin\, with everything else a human clicks.  They still drive THIS
    # layer's board, which is why this layer is what checks them.
    launchers = ROOT.parents[1] / "bin"
    build = (launchers / "build_and_program.bat").read_text(encoding="utf-8")
    estimate = (launchers / "estimate_resources.bat").read_text(encoding="utf-8")
    assert "fpga.pulse_streamer.host" not in build
    assert "fpga.pulse_streamer.host" not in estimate
    # Through the one entry, not at the layer.  "python -m zlc_pulse.fpga" with
    # this layer's folder on PYTHONPATH does not find the package at all -- the
    # source is under src/ -- so it ran only where a standalone repository was
    # pip-installed, and then it derived the board geometry from that tree.
    assert "zou_lab_control_v2 fpga" in build
    assert "zou_lab_control_v2 fpga" in estimate
