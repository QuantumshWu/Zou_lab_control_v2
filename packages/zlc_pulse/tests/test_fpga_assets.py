from __future__ import annotations

import hashlib
import re
from pathlib import Path
import subprocess
import sys

import pytest

from zlc_pulse.fpga import DEFAULT_CONFIG_PATH, emit_geometry_vh, load_streamer_config
from zlc_pulse.wire import default_params


ROOT = Path(__file__).resolve().parents[1]


# SHA-256 values are the byte-level manifest of the read-only v1 assets copied into
# this repository.  Generated Vivado projects are intentionally not part of it.
FROZEN_ASSET_SHA256 = {
    "fpga/board_config/board.xdc": "3120970730c1e30f1e4b2891adcdcd880221b4f1a8ed3f3adf71cd2a2187ba20",
    "fpga/board_config/streamer_config.json": "fc4a9ceae3222a6d20b5879dee20855838a2dfe8260b81886bb5a5540e3a6622",
    "fpga/build/geom.tcl": "ee37c10f2006dd26fff0fcb1d88f905e21538eac4bcec9d2977ae444e243fed8",
    "fpga/pulse_streamer/create_project.tcl": "7edeb1d5dfabb703cdc3053dd470a3e471f3f61d1107a84611170f848023f58e",
    "fpga/pulse_streamer/diagnose_hw_target.tcl": "b924fad6841588ec2300511369fdaea9b86a9e376f0a247b2b21aa11d43321aa",
    "fpga/pulse_streamer/program_flash.tcl": "ce91e601cb8f748352a7efe3ffb57b8cc7eedfe49082e2434761f7d126c37a89",
    "fpga/pulse_streamer/program_fpga.tcl": "246787e0f688f6c1cadd651b6b2f8447c16bb889d31982187fd10f7f15e2a8bf",
    "fpga/pulse_streamer/sim/.gitignore": "ca7223ab280274de18db9048ea6b856d2968b0690f5aa74110de0759589ae899",
    "fpga/pulse_streamer/sim/replay_t.vh": "63a43191f88ac3a4578abdb854664e4c2e4019879b36937f3ddee3c8e3383b5d",
    "fpga/pulse_streamer/sim/replay_t_frame.vh": "6794465a21b9c71039ffa156d5702260932c83d9985ad750fe710eab673347a0",
    "fpga/pulse_streamer/sim/tb_1tick.v": "7478c3a2b98f9563ad00697e580f910c884934e475b77dabf0776a8b62f7d0c5",
    "fpga/pulse_streamer/sim/tb_bus_delay.v": "812dda5c306181df7842dcc8bd8528c7a91f633f715b5efc9810be0098c86119",
    "fpga/pulse_streamer/sim/tb_da_clk_phase.v": "b781bc45e3b5e627764485fefaa4cf8c23c817a20cacad2d0aff4153ba149a5a",
    "fpga/pulse_streamer/sim/tb_da_ttl_align.v": "9a316fe7036563e7c9cb9de2aebdb4c9bac3dbf2f98a3251d229a0052ec23901",
    "fpga/pulse_streamer/sim/tb_delay_compact.v": "42a2df404e65ec5c7508565587a09d3aa67d158f673e32351c4f5f84d99ce64a",
    "fpga/pulse_streamer/sim/tb_delay_sched.v": "54844983293df4da82ef1c27ed57adf8ddac3414ea34234893824dd3d6501ee2",
    "fpga/pulse_streamer/sim/tb_edge_streamer.v": "d2b883dab5e7ab9a7da6d9e1d6612e7559a004b91c146f5130a13e57295f0b80",
    "fpga/pulse_streamer/sim/tb_evt_depth.v": "c7c1f25126bb3bc4f4b2efe28172e7a099d20ea250409f3e3a6e5eb2ba5144cd",
    "fpga/pulse_streamer/sim/tb_gapsweep.v": "17b0428734bda7847d65fde5d63ff603ee903c74adc583c73921570752193f8f",
    "fpga/pulse_streamer/sim/tb_loop.v": "774ac1158078072c92399b9a58a6c117fec86216d2b707480e0f67a8273a9e1c",
    "fpga/pulse_streamer/sim/tb_ramp_scan.v": "0057f9a4466cc77e9daa0704478c058f49a13b1ae72be9a33317ffe29dde05ee",
    "fpga/pulse_streamer/sim/tb_real_engine.v": "4a5c54f4f4f5666ca817c89212d2101c92881e67695fae77ac21e8e9b8a84e3d",
    "fpga/pulse_streamer/sim/tb_scan_wrap.v": "50a6a653631ce7b37e82bd3bc7217b9f6f189361ea1ed38df5c86a0f073b0097",
    "fpga/pulse_streamer/sim/tb_t_ff.v": "2ebef12255fcc7f68607b58dfbb3bb0983b1c034ddaea4d91e0080b2e688f215",
    "fpga/pulse_streamer/tb_uart_pipeline.v": "b3d0c1ba3969aeab476cb174d4c10efdea1a8b59b2f59ef9ffdfc106747bc944",
    "fpga/pulse_streamer/tb_uart_read_tap.v": "b4a006468f16bdf098161a6f8a431515860eead3bc81db1363c1c140988fa1e3",
    "fpga/pulse_streamer/zlc_edge_streamer.v": "62f20b2944dd4e1b3763e5131bee5acaf5acf036585717e0b5dc2578b5763cb1",
    "fpga/pulse_streamer/zlc_pulse_streamer_top.v": "c391e340464617aab79ed687ddc5db5bfbe442ab4d9f54ddecafde632c02f955",
    "fpga/pulse_streamer/zlc_uart_bridge.v": "ed2c0cb52638997a104cdced385dc3e5ec59e6d0917491c66e4dece92cf91503",
}


def test_frozen_fpga_asset_manifest_is_byte_exact() -> None:
    assert not (ROOT / "fpga/pulse_streamer/host").exists()
    for relative, expected in FROZEN_ASSET_SHA256.items():
        path = ROOT / relative
        assert path.is_file(), relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, relative


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
    assert load_streamer_config()["source"].resolve() == expected


def test_fpga_launchers_use_the_package_wire_cli() -> None:
    # In bin\, with everything else a human clicks.  They still drive THIS
    # layer's board, which is why this layer is what checks them.
    launchers = ROOT.parents[1] / "bin"
    build = (launchers / "build_and_program.bat").read_text(encoding="utf-8")
    estimate = (launchers / "estimate_resources.bat").read_text(encoding="utf-8")
    assert "fpga.pulse_streamer.host" not in build
    assert "zlc_pulse.fpga" in build
    assert "fpga.pulse_streamer.host" not in estimate
    assert "zlc_pulse.fpga" in estimate
