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
#: The allowance costs no isolation, which the test below is what keeps true.
ALLOWED_TOP_LEVEL = {
    "numpy",
    "serial",
    "zlc_data",
    "zlc_pulse",
}

DATA_SRC = ROOT.parents[0] / "zlc_data" / "src" / "zlc_data"


def test_package_import_is_pure() -> None:
    import zlc_pulse

    assert Path(zlc_pulse.__file__).resolve().parent.name == "zlc_pulse"


_STDLIB = {
    "binascii",
    "collections",
    "dataclasses",
    "enum",
    "fractions",
    "hashlib",
    "json",
    "math",
    "numbers",
    "os",
    "pathlib",
    "re",
    "struct",
    "subprocess",
    "threading",
    "time",
    "typing",
    "zlib",
    "queue",
    "shutil",
    "sys",
    "tempfile",
    "contextlib",
    "io",
    "logging",
    "warnings",
    "abc",
    "types",
    "importlib",
    "__future__",
    "argparse",
    "socket",
    "socketserver",
}


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
        if name not in ALLOWED_TOP_LEVEL and name not in _STDLIB
    ]
    assert offenders == []


def test_the_one_allowed_layer_carries_nothing_in_behind_it() -> None:
    """zlc_data may be imported only while it is as light as this package is.

    An allowance to import one layer is an allowance to import whatever that
    layer imports.  Written down, it stays a unit vocabulary; unwritten, it is
    the day zlc_pulse quietly needs matplotlib.
    """

    offenders = [
        (path.name, name)
        for path in DATA_SRC.rglob("*.py")
        for name in _imported_top_levels(path)
        if name not in {"numpy", "zlc_data"}
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
    text = chr(10).join(path.read_text(encoding="utf-8") for path in SRC.rglob("*.py"))
    assert not [token for token in banned if token in text]


def test_production_source_has_no_hardcoded_windows_com_number() -> None:
    text = chr(10).join(path.read_text(encoding="utf-8") for path in SRC.rglob("*.py"))
    assert re.search(r"\bCOM\d+\b", text, flags=re.IGNORECASE) is None
