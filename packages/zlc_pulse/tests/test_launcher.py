from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# In bin/, with everything else a human clicks -- it still drives THIS layer's
# board, which is why this layer is what checks it.
LAUNCHER = ROOT.parents[1] / "bin" / "run_server.bat"
BUILD_LAUNCHER = ROOT.parents[1] / "bin" / "build_and_program.bat"
FPGA_SOURCES = ROOT / "fpga" / "pulse_streamer"


def _fake_python(path: Path) -> Path:
    path.write_text("@echo off\necho FAKE_ARGS=%*\nexit /b 0\n", encoding="utf-8")
    return path


def _run_batch(*args: str, cwd: Path, python_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "ZLC_FPGA_PYTHON": str(python_path),
            "ZLC_NO_PAUSE": "1",
            "ZLC_PS_HOST": "127.0.0.1",
            "ZLC_PS_PORT": "18861",
            "PYTHONPATH": "",
        }
    )
    for name in ("ZLC_PY_CMD", "ZLC_PY_PATH"):
        env.pop(name, None)
    arguments = " ".join(f'"{arg}"' if " " in arg else arg for arg in args)
    command = f'cmd.exe /d /s /c ""{LAUNCHER}" {arguments}"'
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_real_batch_wrapper_forwards_exact_modes_without_inner_argument(tmp_path) -> None:
    fake = _fake_python(tmp_path / "fake-python.bat")
    for args in (
        (),
        ("--backend", "jtag-axi"),
        ("--backend", "uart", "--uart-port", "COM7"),
        ("--help",),
    ):
        result = _run_batch(*args, cwd=ROOT, python_path=fake)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "FAKE_ARGS=" in result.stdout
        assert "--inner" not in result.stdout
    no_args = _run_batch(cwd=ROOT, python_path=fake)
    assert "Server listen bind: 127.0.0.1:18861" in no_args.stdout
    assert "remote_server address: 127.0.0.1:18861" in no_args.stdout
    jtag = _run_batch("--backend", "jtag-axi", cwd=ROOT, python_path=fake)
    assert "--backend jtag-axi" in jtag.stdout
    uart = _run_batch("--backend", "uart", "--uart-port", "COM7", cwd=ROOT, python_path=fake)
    assert "--backend uart --uart-port COM7" in uart.stdout


def test_batch_check_config_uses_repo_imports_from_shadow_cwd(tmp_path) -> None:
    shadow = tmp_path / "shadow"
    (shadow / "zlc_pulse").mkdir(parents=True)
    (shadow / "zlc_pulse" / "__init__.py").write_text(
        "raise AssertionError('shadow zlc_pulse was imported')\n", encoding="utf-8"
    )
    result = _run_batch("--check-config", cwd=shadow, python_path=Path(sys.executable))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "config=" in result.stdout
    assert str((ROOT / "fpga" / "board_config" / "streamer_config.json").resolve()) in result.stdout
    assert "shadow zlc_pulse was imported" not in result.stdout + result.stderr


def test_create_project_deletes_only_its_marked_child_and_requires_geometry() -> None:
    source = (FPGA_SOURCES / "create_project.tcl").read_text(encoding="utf-8")
    delete_at = source.index("file delete -force $project_dir")
    guard = source[:delete_at]

    assert ".zlc_generated_project" in guard
    assert "file dirname $out" in guard and "file tail $out" in guard
    assert "file normalize $project_root" in guard
    assert "${project_name}.xpr" in guard
    assert "${project_name}.runs" in guard
    assert "${project_name}.srcs" in guard
    assert "migrating legacy generated Vivado project" in guard
    assert "ZLC_PS_GEOM_TCL is required" in source
    assert "source $zlc_geom_tcl" in source
    assert "if {![info exists zlc_edge_addr_width]}" not in source


def test_hardware_tcl_requires_exactly_one_matching_target_and_device() -> None:
    for name in ("diagnose_hw_target.tcl", "program_fpga.tcl", "program_flash.tcl"):
        source = (FPGA_SOURCES / name).read_text(encoding="utf-8")
        assert "[llength $zlc_targets] != 1" in source, name
        assert "[llength $zlc_devices] != 1" in source, name
        assert "ZLC_PS_FPGA_PART" in source, name
        assert "get_property PART $device" in source, name
        assert "get_parts -quiet $expected_part" in source, name
        assert "get_property DEVICE $expected_parts" in source, name
        assert "string equal -nocase $actual_part $expected_device" in source, name
        assert "lindex $zlc_targets 0" not in source, name
        assert "lindex [get_hw_devices] 0" not in source, name


def test_build_launcher_fails_closed_and_programs_by_default() -> None:
    source = BUILD_LAUNCHER.read_text(encoding="utf-8")
    assert 'set "MODE=build_program"' in source
    assert 'if /I "%MODE%"=="build_program" goto zlc_program' in source
    assert 'if /I "%~1"=="--build-only" (set "MODE=build"' in source
    assert 'if /I "%~1"=="--program-only" (set "MODE=program"' in source
    assert ":zlc_require_config" in source
    require_at = source.index("call :zlc_require_config")
    program_at = source.index(":zlc_program")
    flash_at = source.index('if /I "%MODE%"=="flash"')
    assert require_at < program_at and require_at < flash_at
    assert "duplicate key" in source
    assert "if errorlevel 1 exit /b 1" in source[require_at:]
