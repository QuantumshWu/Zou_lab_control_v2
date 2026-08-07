from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src" / "zlc_pulse"
ALLOWED_TOP_LEVEL = {
    "numpy",
    "serial",
    "zlc_pulse",
}


def test_package_import_is_pure() -> None:
    import zlc_pulse

    assert zlc_pulse.__version__
    assert Path(zlc_pulse.__file__).resolve().parent.name == "zlc_pulse"


def test_source_imports_only_the_package_and_allowed_dependencies() -> None:
    offenders: list[tuple[str, str]] = []
    stdlib = {
        "binascii",
        "collections",
        "dataclasses",
        "enum",
        "fractions",
        "hashlib",
        "json",
        "math",
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
        "warnings",
        "abc",
        "types",
        "importlib",
        "__future__",
        "argparse",
        "socket",
        "socketserver",
    }
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".", 1)[0]]
            else:
                continue
            for name in names:
                if name not in ALLOWED_TOP_LEVEL and name not in stdlib:
                    offenders.append((path.name, name))
    assert offenders == []


def test_negative_surface_is_absent() -> None:
    banned = (
        "trigger_schedule",
        "expected_trigger_counts",
        "visible_ports",
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
