"""Every Windows wrapper selects one installed manifest command exactly once."""

from __future__ import annotations

import pathlib
import re


BIN = pathlib.Path(__file__).resolve().parents[3] / "bin"


def _launchers() -> list[pathlib.Path]:
    return sorted(path for path in BIN.glob("*.bat") if not path.name.startswith("_"))


def test_there_are_launchers_to_check() -> None:
    assert len(_launchers()) >= 6


def test_no_launcher_runs_a_layer_module_directly() -> None:
    """``-m zlc_<layer>`` is reaching past the entry, whatever the PYTHONPATH."""

    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in _launchers()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"-m\s+zlc_\w+", line) and not line.lstrip().lower().startswith("rem")
    ]
    assert offenders == [], "\n".join(offenders)


def test_no_launcher_hides_the_reason_a_python_step_failed() -> None:
    """2>nul on a step whose failure is reported is a generic message and no cause."""

    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in _launchers()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "2>nul" in line
        and "%ZLC_PY_CMD%" in line
        and not line.lstrip().lower().startswith("rem")
    ]
    assert offenders == [], "\n".join(offenders)
