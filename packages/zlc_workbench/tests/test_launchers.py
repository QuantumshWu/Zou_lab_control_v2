"""The launchers are the only part of this project a user runs by clicking.

They are also the only part written in a language whose parser cares about
bytes nobody looks at.  cmd.exe resolves ``goto`` by seeking through the file,
so a batch file saved with bare LF line endings loses labels *by offset*: the
python half of ``_resolve_tools.bat`` resolved fine while its vivado half
answered

    The system cannot find the batch label specified - zlc_vivado_found

for a label sitting in plain sight seven lines below.  A launcher that fails
this way looks like a broken program, not a broken byte, so it costs a
diagnosis every time.

Line endings are the kind of thing a checkout silently decides, which is why
this is a test and not a note: the experiment machine gets its code by pull.

And every wrapper in bin/ selects one installed manifest command exactly once:
it never reaches past the entry into a layer module, never hides why a
python step failed, and never waits for a key while ZLC_NO_PAUSE is set.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BIN = REPO_ROOT / "bin"

# Vivado writes its own launchers into the build tree.  Those are its files,
# they are not synced, and they are not ours to hold to this rule.
GENERATED = ("build", ".Xil", "__pycache__")


def _authored_batch_files() -> list[pathlib.Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.bat")
        if not any(part in GENERATED for part in path.parts)
    )


def _launchers() -> list[pathlib.Path]:
    """The wrappers a user double-clicks: bin/, minus the shared _helpers."""

    return sorted(path for path in BIN.glob("*.bat") if not path.name.startswith("_"))


def test_there_are_launchers_to_check() -> None:
    # Without this the rules below pass loudest when they check nothing --
    # a renamed folder or a moved test file would silently retire them.
    # Nine since the in-process serving round deleted the pulse and SLM
    # server launchers (the console serves both itself).
    assert len(_authored_batch_files()) >= 9
    assert len(_launchers()) >= 6


@pytest.mark.parametrize(
    "batch", _authored_batch_files(), ids=lambda path: path.name
)
def test_batch_files_use_crlf_because_goto_seeks_by_byte(
    batch: pathlib.Path,
) -> None:
    raw = batch.read_bytes()
    bare_lf = raw.replace(b"\r\n", b"").count(b"\n")
    assert bare_lf == 0, (
        f"{batch.relative_to(REPO_ROOT)} has {bare_lf} bare-LF line endings; "
        "cmd.exe will lose goto labels somewhere in it"
    )


def test_the_checkout_rule_is_written_down_not_left_to_each_machine() -> None:
    # Converting the files once fixes this checkout.  Only .gitattributes fixes
    # the next one, which is the machine that runs the experiment.
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in attributes


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


def test_every_pause_yields_to_automation() -> None:
    """A bare ``pause`` blocks whatever runs the window for a key nobody presses.

    ``ZLC_NO_PAUSE`` is the one switch automation and the tests set, so every
    ``pause`` in every batch file -- the shared _launch.bat's included -- is
    guarded by it.  test_launcher reaches only two of them at runtime (a
    failing command through _launch.bat, and estimate_resources.bat); this
    holds the rest.
    """

    guarded = 'if "%ZLC_NO_PAUSE%"=="" pause'
    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in _authored_batch_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\bpause\b", line, re.IGNORECASE)
        and not line.lstrip().lower().startswith("rem")
        and line.strip() != guarded
    ]
    assert offenders == [], "\n".join(offenders)


@pytest.mark.parametrize(
    ("launcher", "command"),
    (
        ("experiment.bat", "task_console"),
        ("figure_viewer.bat", "figure_viewer"),
    ),
)
def test_a_window_launcher_is_one_manifest_wrapper(launcher: str, command: str) -> None:
    source = (BIN / launcher).read_text(encoding="utf-8").lower()
    assert f'set "zlc_command={command}"' in source
    assert 'call "%~dp0_launch.bat" %*' in source
    others = {"task_console", "figure_viewer", "pulse_editor", "device_manager"} - {command}
    assert not any(name in source for name in others), "one wrapper, one command"
    assert "start " not in source
