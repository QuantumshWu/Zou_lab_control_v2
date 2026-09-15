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


def _is_concrete_plugin(path: Path) -> bool:
    parts = path.relative_to(SRC).parts
    return (
        len(parts) >= 3
        and parts[0] in {"nodes", "devices"}
        and parts[1] != "_framework"
    )


def test_foundation_stays_headless_while_concrete_plugins_may_own_views() -> None:
    paths = _python_files("")
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
import zou_lab_control
import zlc_atom
print(zou_lab_control.ROOT)
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


def test_reading_the_node_descriptors_does_not_import_a_solver(
    tmp_path: Path,
) -> None:
    """Discovery reads declarations; it must not load anybody's engine.

    A logic node is discovered by importing the module that exports its
    descriptor, so every module that module reaches is on the startup path
    of every process that lists the nodes -- and the task console lists
    them before it shows a window.  Three edges had grown across it: the
    readout's two-state classification took one float from the plot's fit
    engine, the feedback task took an inverse normal and an assignment
    solver, and the SLM's file formats share a module with its phase
    retrieval.  Together they cost 1.27 of the 1.40 s discovery took, and
    put scipy, numba and llvmlite -- some two hundred megabytes -- into a
    GUI process that renders nothing itself: the panels are drawn in the
    render children, which import the engines in parallel, warm, before
    any panel asks.

    Each solver is still reached, unchanged, from inside the function that
    uses it.  This is what keeps it that way.
    """

    blocked = ("scipy", "numba", "llvmlite", "matplotlib")
    script = f'''
import sys
import traceback
import zou_lab_control

BLOCKED = {blocked!r}
reached = []


class NameTheImporter:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in BLOCKED and not reached:
            reached.append((fullname, "".join(traceback.format_stack()[:-1])))
        return None


sys.meta_path.insert(0, NameTheImporter())
from zlc_atom.nodes import discover_logic_nodes

assert discover_logic_nodes(), "discovery found no logic nodes"
if reached:
    name, where = reached[0]
    raise AssertionError("discovery imported " + name + ":" + chr(10) + where)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_virtual_runtime_branch_scan_is_non_vacuous_and_clean() -> None:
    paths = _python_files("")
    pattern = re.compile(r"\bif[^\r\n]*\bvirtual\b|\bvirtual\b[^\r\n]*\bif\b", re.IGNORECASE)
    hits = tuple((path, line_number, line) for path in paths for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line))
    assert not hits, hits


def test_simulation_devices_are_a_separate_device_family() -> None:
    """Real-device packages must not own or re-export virtual apparatus code.

    The virtual apparatus is a family of its own: one world, and beside it a
    folder per virtual device, exactly as a real family carries a folder per
    real device.  What a real family must never carry is the world or a
    stand-in for its own instrument -- a driver that can see the simulation
    is a driver that can be written to please it.
    """

    devices = SRC / "devices"
    simulation = devices / "simulation"
    assert {"__init__.py", "authoring.py", "world.py"} == {
        path.name for path in simulation.glob("*.py")
    }, "the simulation family owns the world and nothing else at its top level"
    for device in ("camera", "rf", "sequencer", "slm", "waveform"):
        assert (simulation / device / "device_types.py").is_file(), device

    for package in (devices / "camera", devices / "sequencer", devices / "slm"):
        for path in package.rglob("*.py"):
            assert "zlc_atom.devices.simulation" not in path.read_text(encoding="utf-8"), path
        assert not any(
            "virtual" in part or "world" in part
            for path in package.rglob("*.py")
            for part in path.relative_to(package).parts
        ), f"{package.name} must not carry a stand-in for its own instrument"
