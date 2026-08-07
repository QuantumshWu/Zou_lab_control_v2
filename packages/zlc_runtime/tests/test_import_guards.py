from __future__ import annotations

import importlib
import json
import os
import pkgutil
import re
import subprocess
import sys
import types
from pathlib import Path

import zlc_runtime


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTRACT = ROOT / "docs" / "contract.md"
FORBIDDEN_IMPORTS = (
    "PyQt5",
    "matplotlib",
    "zlc_plot",
    "zlc_ui",
    "zlc_storage",
    "zlc_neutral_atom",
    "zlc_workbench",
)


def _contract_facade() -> tuple[str, ...]:
    text = CONTRACT.read_text(encoding="utf-8")
    heading = re.search(r"^## 顶层 facade.*$", text, flags=re.MULTILINE)
    assert heading is not None, "contract must contain a top-level facade heading"
    following = text[heading.end() :].splitlines()
    facade_line = next((line for line in following if line.startswith("`")), None)
    assert facade_line is not None, "contract must contain a fenced top-level facade"
    return tuple(re.findall(r"`([^`]+)`", facade_line))


def _runtime_modules() -> tuple[str, ...]:
    discovered = tuple(
        module.name
        for module in pkgutil.walk_packages(
            zlc_runtime.__path__, f"{zlc_runtime.__name__}."
        )
    )
    return (zlc_runtime.__name__, *discovered)


def _public_non_module_names(module: types.ModuleType) -> set[str]:
    module_names = {
        name
        for name, value in vars(module).items()
        if isinstance(value, types.ModuleType)
    }
    return {
        name
        for name in dir(module)
        if not name.startswith("_") and name not in module_names
    }


def test_public_surface_matches_frozen_contract_and_is_nonempty() -> None:
    expected = _contract_facade()
    assert expected
    # Derived from the listing, not re-typed beside it.  Raised for the names
    # zlc_atom and zlc_workbench were ALREADY using through submodules: a
    # sibling reaching past this list is the same surface with the declaration
    # skipped, and nobody can then see how wide it really is.
    assert len(expected) == len(zlc_runtime.__all__)
    assert tuple(zlc_runtime.__all__) == expected
    assert type(zlc_runtime.MAX_PUBLIC_NAMES) is int
    assert zlc_runtime.MAX_PUBLIC_NAMES == len(expected) + 1
    expected_public = {name for name in expected if not name.startswith("_")}
    expected_public.add("MAX_PUBLIC_NAMES")
    actual_public = _public_non_module_names(zlc_runtime)
    assert actual_public == expected_public
    assert len(actual_public) <= zlc_runtime.MAX_PUBLIC_NAMES
    script = """
import json
import types
import zlc_runtime

module_names = {
    name for name, value in vars(zlc_runtime).items()
    if isinstance(value, types.ModuleType)
}
print(json.dumps({
    'all': list(zlc_runtime.__all__),
    'public': sorted(
        name for name in dir(zlc_runtime)
        if not name.startswith('_') and name not in module_names
    ),
    'max': zlc_runtime.MAX_PUBLIC_NAMES,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert tuple(report["all"]) == expected
    assert tuple(report["public"]) == tuple(sorted(expected_public))
    assert report["max"] == zlc_runtime.MAX_PUBLIC_NAMES


def test_facade_names_are_the_concrete_runtime_implementations() -> None:
    from zlc_runtime.host import Node, NodeHost
    from zlc_runtime.live_dataset import LiveDatasetPort
    from zlc_runtime.plane import (
        SignalDataPlane,
        SignalFront,
        SignalPublication,
        SignalValue,
    )
    from zlc_runtime.dataset import DatasetCoverage, MonitorCoverage
    from zlc_runtime.dataset_output import (
        DatasetOutputDeclaration,
        FinalDatasetOutput,
        LiveDatasetOutput,
    )
    from zlc_runtime.plane import DerivedSignalOutput
    from zlc_runtime.presentation import (
        BoardScheduler,
        HarmonicClock,
        OwnerChannels,
        SurfaceBatchArbiter,
        SurfaceUpdate,
    )
    from zlc_runtime.selection_bridge import (
        FitEventValue,
        SelectionBridge,
        SelectionChange,
        SelectionRange,
        SelectionState,
    )
    from zlc_runtime.streams import AcquisitionStream, ExactReservation, FollowTap, MonitorTap

    expected = {
        "SignalDataPlane": SignalDataPlane,
        "SignalValue": SignalValue,
        "SignalPublication": SignalPublication,
        "AcquisitionStream": AcquisitionStream,
        "NodeHost": NodeHost,
        "SelectionBridge": SelectionBridge,
        "SelectionChange": SelectionChange,
        "SelectionRange": SelectionRange,
        "SelectionState": SelectionState,
        "FitEventValue": FitEventValue,
        # Back on the facade because siblings use them.  A name returns the
        # moment one does: reaching into a submodule for something this list
        # does not carry is the same surface with the declaration skipped.
        "BoardScheduler": BoardScheduler,
        "HarmonicClock": HarmonicClock,
        "OwnerChannels": OwnerChannels,
        "SurfaceBatchArbiter": SurfaceBatchArbiter,
        "SurfaceUpdate": SurfaceUpdate,
        "DatasetCoverage": DatasetCoverage,
        "MonitorCoverage": MonitorCoverage,
        "DatasetOutputDeclaration": DatasetOutputDeclaration,
        "FinalDatasetOutput": FinalDatasetOutput,
        "LiveDatasetOutput": LiveDatasetOutput,
        "DerivedSignalOutput": DerivedSignalOutput,
    }
    assert all(getattr(zlc_runtime, name) is value for name, value in expected.items())
    assert isinstance(zlc_runtime.__version__, str)
    assert zlc_runtime.__version__

    removed = {
        "SignalFront": SignalFront,
        "ExactReservation": ExactReservation,
        "MonitorTap": MonitorTap,
        "FollowTap": FollowTap,
        "LiveDatasetPort": LiveDatasetPort,
        "Node": Node,
        "RunHandleLike": importlib.import_module("zlc_runtime._public").RunHandleLike,
    }
    assert all(not hasattr(zlc_runtime, name) for name in removed)
    assert all(
        getattr(
            importlib.import_module(
                {
                    "SignalFront": "zlc_runtime.plane",
                    "ExactReservation": "zlc_runtime.streams",
                    "MonitorTap": "zlc_runtime.streams",
                    "FollowTap": "zlc_runtime.streams",
                    "LiveDatasetPort": "zlc_runtime.live_dataset",
                    "Node": "zlc_runtime.host",
                    "BoardScheduler": "zlc_runtime.presentation",
                    "HarmonicClock": "zlc_runtime.presentation",
                    "OwnerChannels": "zlc_runtime.presentation",
                    "RunHandleLike": "zlc_runtime._public",
                }[name]
            ),
            name,
        )
        is implementation
        for name, implementation in removed.items()
    )


def test_every_runtime_module_is_imported_in_a_fresh_process() -> None:
    modules = _runtime_modules()
    assert modules, "module discovery must not be vacuous"
    script = """
import builtins
import importlib
import json
import sys

seen = set()
real_import = builtins.__import__

def tracing_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name:
        seen.add(name.split('.')[0])
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracing_import
importlib.import_module(sys.argv[1])
print(json.dumps(sorted(seen)))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    stdlib.update(sys.builtin_module_names)
    allowed = stdlib | {"numpy", "zlc_data", "zlc_runtime"}

    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", script, module],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        imported = set(json.loads(result.stdout.strip()))
        assert imported <= allowed, f"{module} imports outside whitelist: {imported - allowed}"


def test_source_scan_is_nonempty_and_has_no_forbidden_dependencies() -> None:
    source_files = tuple(SRC.rglob("*.py"))
    assert source_files, "source scan must inspect at least one Python file"
    violations = {
        f"{path.relative_to(ROOT)}: {term}"
        for path in source_files
        for term in FORBIDDEN_IMPORTS
        if term in path.read_text(encoding="utf-8")
    }
    assert not violations


def test_import_zlc_runtime_does_not_start_threads_or_runtime_services() -> None:
    script = """
import importlib
import threading
import types

before = {thread.ident for thread in threading.enumerate()}
module = importlib.import_module('zlc_runtime')
after = {thread.ident for thread in threading.enumerate()}
assert after == before
module_names = {
    name for name, value in vars(module).items()
    if isinstance(value, types.ModuleType)
}
public = {
    name for name in dir(module)
    if not name.startswith('_') and name not in module_names
}
assert public == {
    name for name in module.__all__ if not name.startswith('_')
} | {'MAX_PUBLIC_NAMES'}
assert len(public) <= module.MAX_PUBLIC_NAMES
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
