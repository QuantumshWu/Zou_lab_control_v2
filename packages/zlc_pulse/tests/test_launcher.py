from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
# The server wrapper is gone -- a bench serves its own board in-process -- so
# the shared _launch.bat plumbing is checked through a wrapper that remains.
LAUNCHER = ROOT.parents[1] / "bin" / "pulse_editor.bat"
SHARED_LAUNCHER = ROOT.parents[1] / "bin" / "_launch.bat"
BUILD_LAUNCHER = ROOT.parents[1] / "bin" / "build_and_program.bat"
ESTIMATE_LAUNCHER = ROOT.parents[1] / "bin" / "estimate_resources.bat"
INSTALL_LAUNCHER = ROOT.parents[1] / "bin" / "install_requirements.bat"
TOOLS_RESOLVER = ROOT / "fpga" / "_resolve_tools.bat"
FPGA_SOURCES = ROOT / "fpga" / "pulse_streamer"


def _fake_python(path: Path) -> Path:
    path.write_text(
        "@echo off\necho FAKE_ARGS=%*\necho FAKE_PYTHONPATH=%PYTHONPATH%\nexit /b 0\n",
        encoding="utf-8",
    )
    return path


def _run_batch(*args: str, cwd: Path, python_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "ZLC_FPGA_PYTHON": str(python_path),
            "ZLC_NO_PAUSE": "1",
            "PYTHONPATH": "",
        }
    )
    for name in ("ZLC_PY_CMD", "ZLC_PY_PATH"):
        env.pop(name, None)
    arguments = " ".join(
        f'"{arg}"' if not arg or " " in arg else arg for arg in args
    )
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
    no_args = _run_batch(cwd=ROOT, python_path=fake)
    assert no_args.returncode == 0, no_args.stdout + no_args.stderr
    assert "FAKE_ARGS=" in no_args.stdout
    assert "--inner" not in no_args.stdout
    assert "-m zou_lab_control pulse_editor" in no_args.stdout
    shared = SHARED_LAUNCHER.read_text(encoding="utf-8")
    resolver = TOOLS_RESOLVER.read_text(encoding="utf-8")
    # The resolver is the one owner of "a launched command imports THIS
    # checkout": it injects the repository root and every layer's src.  The
    # layer list is read from packages/ itself, so a new layer that the
    # resolver forgets is a red test, not a silent import of whatever
    # editable install the interpreter happens to carry.
    layers = sorted(
        path.name
        for path in (ROOT.parents[1] / "packages").iterdir()
        if (path / "src").is_dir()
    )
    assert layers, "packages/ must hold at least one layer with a src tree"
    for layer in layers:
        assert f"%ZLC_TOOL_REPO_ROOT%\packages\{layer}\src" in resolver, layer
    assert 'set "PYTHONPATH=%ZLC_CHECKOUT_PYTHONPATH%;%PYTHONPATH%"' in resolver
    assert 'set "PYTHONPATH=%ZLC_CHECKOUT_PYTHONPATH%"' in resolver
    assert "PYTHONPATH=" not in shared
    seen = no_args.stdout.split("FAKE_PYTHONPATH=", 1)[1]
    assert str(ROOT.parents[1]) in seen
    for layer in layers:
        assert str(ROOT.parents[1] / "packages" / layer / "src") in seen, layer
    installer = INSTALL_LAUNCHER.read_text(encoding="utf-8")
    assert "/installed" in installer
    assert installer.rfind(
        'pushd "%TEMP%"', 0, installer.index("-m zou_lab_control check")
    ) >= 0
    assert '_resolve_tools.bat" python "%ZLC_HOME%"' in BUILD_LAUNCHER.read_text(
        encoding="utf-8"
    )

    exact = _run_batch(
        "value with space", "", "bang!value", "μ-value",
        cwd=ROOT, python_path=fake,
    )
    assert '"value with space" "" bang!value μ-value' in exact.stdout

    estimate_environment = dict(
        os.environ,
        ZLC_FPGA_PYTHON=str(fake),
        ZLC_NO_PAUSE="1",
        PYTHONPATH="",
    )
    estimate = subprocess.run(
        ["cmd.exe", "/d", "/c", str(ESTIMATE_LAUNCHER)],
        cwd=ROOT,
        env=estimate_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert estimate.returncode == 0, estimate.stdout + estimate.stderr
    assert "configured part HAS enough resources" in estimate.stdout
    assert "!ZLC_STATUS!" not in estimate.stdout

    failing = tmp_path / "failing-python.bat"
    failing.write_text(
        "@echo off\nif \"%~1\"==\"-c\" exit /b 0\nexit /b 7\n",
        encoding="utf-8",
    )
    estimate_environment["ZLC_FPGA_PYTHON"] = str(failing)
    failed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(ESTIMATE_LAUNCHER)],
        cwd=ROOT,
        env=estimate_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 7
    assert "failed with code 7" in failed.stdout


def test_create_project_deletes_only_its_project_child_and_requires_geometry() -> None:
    source = (FPGA_SOURCES / "create_project.tcl").read_text(encoding="utf-8")
    delete_at = source.index("file delete -force $project_dir")
    guard = source[:delete_at]

    assert "file dirname $out" in guard and "file tail $out" in guard
    assert "file normalize $project_root" in guard
    assert ".zlc_generated_project" not in source
    assert "Refusing to delete unmarked directory" not in source
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
    vivado_call = source.index('call "%ZLC_PS_VIVADO_BIN%" -mode batch')
    assert source.rfind('pushd "!ZLC_PS_BUILD_ROOT!"', 0, vivado_call) >= 0
    assert 'set "ZLC_TCL_STATUS=!ERRORLEVEL!"' in source[vivado_call:]
    version_call = source.index('call "%ZLC_PS_VIVADO_BIN%" -version')
    assert source.rfind('pushd "!ZLC_PS_BUILD_ROOT!"', 0, version_call) >= 0
