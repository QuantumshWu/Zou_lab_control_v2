from __future__ import annotations

import ast
from pathlib import Path
import re
import sys


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src" / "zlc_pulse"
#: What this package may reach for.  The point of the list is that zlc_pulse
#: can be lifted out and used to drive the board on its own, so it stays as
#: close to "numpy and a serial port" as the work allows.
#:
#: ``zlc_data`` is here for ONE thing: it owns what a unit is.  A duration is
#: a number and a unit, and the compiler that turns one into device ticks
#: cannot be the only layer with its own opinion about what "us" means -- that
#: is how this package came to hold a second unit table with a different base,
#: a different spelling set and different arithmetic from everybody else's.
#: ``zlc_durable`` is here for the same one reason: it owns what an atomic
#: write and a readable JSON document ARE.  A pulse saved beside its module
#: is a file on disk like every other document this project writes, and a
#: package with its own writer is a package whose files differ from the rest
#: in line endings, temp-file discipline and crash behaviour.
#:
#: The allowance costs no isolation, which the test below is what keeps true.
ALLOWED_TOP_LEVEL = {
    "numpy",
    "serial",
    "zlc_data",
    "zlc_durable",
    "zlc_pulse",
}

#: The allowed layers, each of which must stay as light as this package is.
LIGHT_LAYERS = ("zlc_data", "zlc_durable")


def test_package_import_is_pure() -> None:
    import zlc_pulse

    assert Path(zlc_pulse.__file__).resolve().parent.name == "zlc_pulse"


def _imported_top_levels(path: Path) -> list[str]:
    """Every top-level package one file reaches for."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module.split(".", 1)[0])
    return names


def test_source_imports_only_the_package_and_allowed_dependencies() -> None:
    offenders = [
        (path.name, name)
        for path in SRC.rglob("*.py")
        for name in _imported_top_levels(path)
        if name not in ALLOWED_TOP_LEVEL
        and name not in sys.stdlib_module_names
    ]
    assert offenders == []


def test_the_allowed_layers_carry_nothing_in_behind_them() -> None:
    """An allowed layer may be imported only while it is as light as this one.

    An allowance to import one layer is an allowance to import whatever that
    layer imports.  Written down, they stay a unit vocabulary and a file
    writer; unwritten, it is the day zlc_pulse quietly needs matplotlib.
    """

    offenders = [
        (layer, path.name, name)
        for layer in LIGHT_LAYERS
        for path in (ROOT.parents[0] / layer / "src" / layer).rglob("*.py")
        for name in _imported_top_levels(path)
        if name not in {"numpy", layer}
        and name not in sys.stdlib_module_names
    ]
    assert offenders == []


def test_negative_surface_is_absent() -> None:
    banned = (
        "trigger_schedule",
        "expected_trigger_counts",
        "scan_sweep_count",
        "PulseExecutionForm",
        "rpyc",
        "sha256_text",
        "evidence",
    )
    # The ban is on a CODE surface, so it is read off the code.  Searching
    # the file text made an English word in a comment -- "bounded raw
    # evidence before the parser" -- indistinguishable from a resurrected
    # subsystem, and the only repair available was to reword prose.
    named = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Name):
                named.add(node.id)
            elif isinstance(node, ast.Attribute):
                named.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                named.add(node.name)
            elif isinstance(node, ast.arg):
                named.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                named.add(node.arg)
            elif isinstance(node, ast.alias):
                named.update(node.name.split("."))
                if node.asname:
                    named.add(node.asname)
            elif isinstance(node, ast.ImportFrom) and node.module:
                named.update(node.module.split("."))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                named.add(node.value)
    assert not [
        token for token in banned if any(token in name for name in named)
    ]


def test_production_source_has_no_hardcoded_windows_com_number() -> None:
    text = chr(10).join(path.read_text(encoding="utf-8") for path in SRC.rglob("*.py"))
    assert re.search(r"\bCOM\d+\b", text, flags=re.IGNORECASE) is None
