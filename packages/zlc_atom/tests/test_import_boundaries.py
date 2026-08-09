from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "zlc_atom"
PARALLEL_ROOTS = ("zlc_runtime", "zlc_pulse")
EXPECTED_PUBLIC_API = frozenset({"__version__"})
MAX_PUBLIC_NAMES = 2


def _real_public_names(module: object) -> tuple[str, ...]:
    package_name = str(getattr(module, "__name__"))
    package_path = getattr(module, "__path__")
    submodule_names = {
        info.name.rsplit(".", 1)[-1]
        for info in pkgutil.iter_modules(package_path, package_name + ".")
    }
    return tuple(
        name
        for name in dir(module)
        if not name.startswith("_") and name not in submodule_names
    )


def _python_files(*packages: str) -> tuple[Path, ...]:
    paths = tuple(path for package in packages for path in (SRC / package).rglob("*.py"))
    assert paths, f"source scan found no Python files under {packages!r}"
    return paths


def _parallel_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return tuple(name for name in imports if name.startswith(PARALLEL_ROOTS))


def test_top_level_allow_list_is_data_only() -> None:
    import zlc_atom

    assert tuple(zlc_atom.__all__) == ("__version__",)
    assert set(zlc_atom.__all__) == EXPECTED_PUBLIC_API
    assert len(tuple(zlc_atom.__all__)) <= MAX_PUBLIC_NAMES
    assert all(hasattr(zlc_atom, name) for name in EXPECTED_PUBLIC_API)


def test_top_level_real_namespace_is_bounded_and_has_no_annotation_leak() -> None:
    import zlc_atom

    public_names = _real_public_names(zlc_atom)
    assert len(public_names) <= MAX_PUBLIC_NAMES
    assert public_names == ()
    assert not hasattr(zlc_atom, "annotations")


def test_top_level_contract_matches_frozen_document() -> None:
    contract = (ROOT / "docs" / "contract.md").read_text(encoding="utf-8")
    block = contract.split("__all__ = (", 1)[1].split(")\n```", 1)[0]
    names = tuple(re.findall(r'"([A-Za-z_]\w*)"', block))
    import zlc_atom

    assert names == tuple(zlc_atom.__all__)


def test_top_level_submodules_remain_explicitly_importable() -> None:
    module = importlib.import_module("zlc_atom.nodes")
    assert module.__name__ == "zlc_atom.nodes"
    import zlc_atom

    assert "nodes" not in zlc_atom.__all__
    assert "nodes" not in _real_public_names(zlc_atom)


def test_forbidden_imports_are_absent_from_the_package() -> None:
    paths = _python_files("")
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("PyQt5", "matplotlib", "zlc_plot", "zlc_ui", "zlc_neutral_atom", "_VirtualSequencerConnection"):
        assert forbidden not in text


def test_b_half_parallel_import_policy_is_explicit() -> None:
    b_half = _python_files("nodes", "install")
    for path in b_half:
        assert all(name.startswith(PARALLEL_ROOTS) for name in _parallel_imports(path)), path


def test_virtual_runtime_branch_scan_is_non_vacuous_and_clean() -> None:
    paths = _python_files("")
    pattern = re.compile(r"\bif[^\r\n]*\bvirtual\b|\bvirtual\b[^\r\n]*\bif\b", re.IGNORECASE)
    hits = tuple((path, line_number, line) for path in paths for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line))
    assert not hits, hits


def test_simulation_devices_are_a_separate_device_family() -> None:
    """Real-device packages must not own or re-export virtual apparatus code."""

    devices = SRC / "devices"
    simulation = devices / "simulation"
    assert {
        "camera.py",
        "device_types.py",
        "sequencer.py",
        "world.py",
    } <= {path.name for path in simulation.glob("*.py")}
    assert not (devices / "camera" / "virtual.py").exists()
    assert not (devices / "camera" / "world.py").exists()
    assert not (devices / "sequencer" / "virtual.py").exists()

    for package in (devices / "camera", devices / "sequencer"):
        for path in package.rglob("*.py"):
            assert "zlc_atom.devices.simulation" not in path.read_text(encoding="utf-8"), path
