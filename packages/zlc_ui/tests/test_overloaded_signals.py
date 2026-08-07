"""Overloaded Qt signals are always subscripted here.

``currentIndexChanged`` and ``activated`` each exist twice in PyQt5 -- once
taking an int, once a str -- and connecting an argument-less Python slot binds
BOTH.  One of those cost this project a deterministic Windows access violation
with no traceback at all: the window simply stopped existing while it was
being built, in a demo whose whole job is to prove the controls work.

The crash depends on the SLOT's arity, which is not something a reader can see
at the connect site, so the rule is the stricter one that can be: name the
overload.  It costs four characters and removes a whole class of silent death.
"""

from __future__ import annotations

import pathlib
import re


SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "zlc_ui"
OVERLOADED = ("currentIndexChanged", "activated", "highlighted", "textActivated")
PATTERN = re.compile(r"\.(" + "|".join(OVERLOADED) + r")\.connect\(")


def test_every_overloaded_signal_connection_names_its_overload() -> None:
    offenders = [
        f"{path.relative_to(SRC)}:{number}: {line.strip()}"
        for path in sorted(SRC.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if PATTERN.search(line) and not line.lstrip().startswith(("#", "rem "))
    ]
    assert offenders == [], "\n".join(offenders)


def test_the_rule_has_something_to_check() -> None:
    # A guard that would pass on an empty search is not a guard.
    connections = sum(
        len(re.findall(r"\.(" + "|".join(OVERLOADED) + r")\[", path.read_text(encoding="utf-8")))
        for path in SRC.rglob("*.py")
    )
    assert connections >= 4
