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
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Vivado writes its own launchers into the build tree.  Those are its files,
# they are not synced, and they are not ours to hold to this rule.
GENERATED = ("build", ".Xil", "__pycache__")


def _authored_batch_files() -> list[pathlib.Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.bat")
        if not any(part in GENERATED for part in path.parts)
    )


def test_there_are_launchers_to_check() -> None:
    # Without this the rule below passes loudest when it is checking nothing --
    # a renamed folder or a moved test file would silently retire the guard.
    # Nine since the in-process serving round deleted the pulse and SLM
    # server launchers (the console serves both itself).
    assert len(_authored_batch_files()) >= 9


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
