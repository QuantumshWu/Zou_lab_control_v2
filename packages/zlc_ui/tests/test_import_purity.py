"""Mechanical import-boundary guard for the package."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "zlc_ui"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FORBIDDEN_ROOTS = {
    "matplotlib",
    "numpy",
    "zlc_plot",
    "zlc_data",
    "zlc_storage",
    "zlc_neutral_atom",
    "zlc_workbench",
    "Zou_lab_control",
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_modules() -> list[tuple[str, Path]]:
    modules = []
    for path in sorted(PACKAGE.rglob("*.py")):
        modules.append((_module_name(path), path))
    return modules


def _import_root(name: str, *, relative: bool) -> str:
    if relative:
        return "zlc_ui"
    return name.split(".", 1)[0]


def _is_stdlib(root: str) -> bool:
    if root in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(root)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    if spec is None or spec.origin in {None, "built-in", "frozen"}:
        return spec is not None
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    try:
        Path(spec.origin).resolve().relative_to(stdlib)
    except ValueError:
        return False
    return True


def test_package_modules_are_nonempty_and_import_pure() -> None:
    modules = _package_modules()
    assert modules, "the purity guard must scan at least one package module"

    discovered = {name for name, _ in modules}
    assert "zlc_ui" in discovered

    for module_name, path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [_import_root(alias.name, relative=False) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [_import_root(node.module or "", relative=node.level > 0)]
            else:
                continue

            for root in roots:
                assert root not in FORBIDDEN_ROOTS, (
                    f"{module_name} imports forbidden root {root!r}"
                )
                assert (
                    root in {"zlc_ui", "PyQt5", "qframelesswindow"}
                    or _is_stdlib(root)
                ), f"{module_name} imports non-whitelisted root {root!r}"

    for module_name, _ in modules:
        importlib.import_module(module_name)
