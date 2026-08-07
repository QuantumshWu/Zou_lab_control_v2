from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# In bin/, with everything else a human clicks -- it still drives THIS layer's
# board, which is why this layer is what checks it.
LAUNCHER = ROOT.parents[1] / "bin" / "run_server.bat"


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
