from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "zlc_atom"
PARALLEL_ROOTS = ("zlc_runtime", "zlc_pulse")
VIEW_ROOTS = ("PyQt5", "matplotlib", "zlc_plot", "zlc_ui")
COMPOSITION_ROOT = "zlc_workbench"


def _python_files(*packages: str) -> tuple[Path, ...]:
    paths = tuple(path for package in packages for path in (SRC / package).rglob("*.py"))
    assert paths, f"source scan found no Python files under {packages!r}"
    return paths


def _absolute_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return tuple(imports)


def _parallel_imports(path: Path) -> tuple[str, ...]:
    return tuple(
        name for name in _absolute_imports(path) if name.startswith(PARALLEL_ROOTS)
    )


def _is_concrete_plugin(path: Path) -> bool:
    parts = path.relative_to(SRC).parts
    return (
        len(parts) >= 3
        and parts[0] in {"nodes", "devices"}
        and parts[1] != "_framework"
    )


def test_foundation_stays_headless_while_concrete_plugins_may_own_views() -> None:
    paths = _python_files("")
    package_text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("zlc_neutral_atom", "_VirtualSequencerConnection"):
        assert forbidden not in package_text

    foundation_view_imports = tuple(
        (path.relative_to(SRC), imported)
        for path in paths
        if not _is_concrete_plugin(path)
        for imported in _absolute_imports(path)
        if imported.split(".", 1)[0] in VIEW_ROOTS
    )
    assert foundation_view_imports == ()

    plugin_view_imports = {
        (path.relative_to(SRC), imported.split(".", 1)[0])
        for path in paths
        if _is_concrete_plugin(path)
        for imported in _absolute_imports(path)
        if imported.split(".", 1)[0] in VIEW_ROOTS
    }
    assert (Path("nodes/calibration/task.py"), "zlc_plot") in plugin_view_imports


def test_calibration_does_not_depend_on_the_workbench_composition_root(
    tmp_path: Path,
) -> None:
    imports = tuple(
        (path.relative_to(SRC), imported)
        for path in _python_files("nodes/calibration")
        for imported in _absolute_imports(path)
        if imported.split(".", 1)[0] == COMPOSITION_ROOT
    )
    assert imports == ()

    script = r'''
import sys
import zou_lab_control_v2
import zlc_atom
print(zou_lab_control_v2.ROOT)
print(zlc_atom.__file__)
class BlockWorkbench:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] == "zlc_workbench":
            raise ModuleNotFoundError("blocked composition root")
        return None
sys.meta_path.insert(0, BlockWorkbench())
import zlc_atom.nodes.calibration
assert "zlc_workbench" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
        "slm.py",
        "world.py",
    } <= {path.name for path in simulation.glob("*.py")}
    assert not (devices / "camera" / "virtual.py").exists()
    assert not (devices / "camera" / "world.py").exists()
    assert not (devices / "sequencer" / "virtual.py").exists()

    for package in (devices / "camera", devices / "sequencer", devices / "slm"):
        for path in package.rglob("*.py"):
            assert "zlc_atom.devices.simulation" not in path.read_text(encoding="utf-8"), path
