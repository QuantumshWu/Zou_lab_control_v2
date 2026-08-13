"""Non-vacuous guards for the small, lazy top-level facade."""

from __future__ import annotations

import importlib
import json
import pkgutil
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "zlc_ui"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPECTED_FACADE = (
    "__version__",
    "BoardMetrics",
    "ConnectionChoiceVM",
    "ConnectionVM",
    "DelayRowVM",
    "FieldVM",
    "FormChoice",
    "FormFieldProps",
    "FormRuntimeContext",
    "FormSpec",
    "PeriodVM",
    "PortRowVM",
    "RepeatVM",
    "ScanPageRecord",
    "ScheduleVM",
    "TargetPortRecord",
    "TargetWidthRule",
    "VALIDATOR_FLOAT",
    "VALIDATOR_INT",
    "WINDOW_SCREEN_FRACTION",
    "capture_window",
    "ensure_qt_app",
    "open_device_control",
    "open_device_manager",
    "open_figure_viewer",
    "open_pulse_editor",
    "open_task_console",
    "use_panel_display_sizes",
)
EXPECTED_PUBLIC = frozenset(name for name in EXPECTED_FACADE if not name.startswith("_"))
MAX_PUBLIC_NAMES = 29

#: GUIs still handed out as widget trees rather than opened behind a handle.
#: Each entry is one window's worth of work, and removing the last one is what
#: finishes the seal -- a new demo may not be added here.
NOT_YET_BEHIND_A_HANDLE = frozenset()

#: The last submodules the notebook still names.  ``acceptance`` is the
#: capture API a CI script calls, and ``fluent`` appears only where the
#: notebook explains the window chrome itself; neither is a window reached
#: around its handle.  A window's package may not be added here.
STILL_REACHED_IN_THE_NOTEBOOK = ()
REMOVED_FACADE_NAMES = ("FlowGraph", "FlowGraphEdge", "FlowGraphNode")


def _real_public_names(package) -> frozenset[str]:
    package_name = package.__name__
    submodules = {
        info.name.rsplit(".", 1)[-1]
        for info in pkgutil.iter_modules(package.__path__, package_name + ".")
    }
    return frozenset(
        name
        for name in dir(package)
        if not name.startswith("_") and name not in submodules
    )


def test_facade_allow_list_and_concrete_public_namespace() -> None:
    package = importlib.import_module("zlc_ui")

    for name in EXPECTED_PUBLIC:
        getattr(package, name)
    assert package.__version__ == "0.1.0"
    assert tuple(package.__all__) == EXPECTED_FACADE

    public_names = _real_public_names(package)
    assert public_names == EXPECTED_PUBLIC
    assert len(public_names) <= MAX_PUBLIC_NAMES
    assert "import_module" not in public_names
    assert "annotations" not in public_names


def test_contract_names_match_facade_bidirectionally() -> None:
    contract = (ROOT / "docs" / "contract.md").read_text(encoding="utf-8")
    match = re.search(r"__all__\s*=\s*\((.*?)\)", contract, re.DOTALL)
    assert match is not None, "contract must contain an explicit __all__ tuple"
    documented = tuple(re.findall(r'"([^"]+)"', match.group(1)))
    assert documented == EXPECTED_FACADE
    assert set(documented) == set(importlib.import_module("zlc_ui").__all__)


def test_removed_names_are_only_available_from_their_submodule() -> None:
    package = importlib.import_module("zlc_ui")
    graph = importlib.import_module("zlc_ui.graph")

    for name in REMOVED_FACADE_NAMES:
        try:
            getattr(package, name)
        except AttributeError:
            pass
        else:
            raise AssertionError(f"removed facade name still resolves: {name}")
        assert hasattr(graph, name)


def test_retained_names_still_point_at_their_implementations() -> None:
    package = importlib.import_module("zlc_ui")
    implementation_paths = {
        "ensure_qt_app": ("zlc_ui.qt", "ensure_qt_app"),
        "FormChoice": ("zlc_ui.form", "FormChoice"),
        "FormFieldProps": ("zlc_ui.form", "FormFieldProps"),
        "FormSpec": ("zlc_ui.form", "FormSpec"),
        "FormRuntimeContext": ("zlc_ui.form", "FormRuntimeContext"),
        "BoardMetrics": ("zlc_ui.board", "BoardMetrics"),
        "open_device_control": ("zlc_ui.windows", "open_device_control"),
    }
    for public_name, (module_name, implementation_name) in implementation_paths.items():
        implementation = getattr(importlib.import_module(module_name), implementation_name)
        assert getattr(package, public_name) is implementation


def test_graph_submodule_import_path_remains_explicitly_usable() -> None:
    from zlc_ui.graph import FlowGraph, FlowGraphEdge, FlowGraphNode

    assert FlowGraph.__module__ == "zlc_ui.graph.flow_graph"
    assert FlowGraphEdge.__module__ == "zlc_ui.graph.flow_graph"
    assert FlowGraphNode.__module__ == "zlc_ui.graph.flow_graph"


def test_examples_and_notebook_do_not_hide_facade_names_behind_submodule_imports() -> None:
    # The tutorial for the OUTSIDE surface: the notebook and each GUI's demo
    # entry.  gallery.py is deliberately not here -- it is the in-package
    # showcase of the controls themselves, nearly all of which are private on
    # purpose, and holding it to the facade would mean it could not show them.
    sources = [
        *(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "examples").glob("demo_*.py")
            if path.name not in NOT_YET_BEHIND_A_HANDLE
        ),
        "\n".join(
            "".join(cell.get("source", []))
            for cell in json.loads((ROOT / "notebooks" / "usage.ipynb").read_text(encoding="utf-8"))["cells"]
            if cell.get("cell_type") == "code"
        ),
    ]
    combined = "\n".join(
        line
        for line in "\n".join(sources).splitlines()
        if not any(area in line for area in STILL_REACHED_IN_THE_NOTEBOOK)
    )
    # Both forms: "import zlc_ui.pulse as p" reaches exactly as far as
    # "from zlc_ui.pulse import ...", and only the second was ever checked.
    assert re.search(r"^from zlc_ui\.\w+ import", combined, re.MULTILINE) is None
    assert re.search(r"^import zlc_ui\.\w+", combined, re.MULTILINE) is None
    for name in EXPECTED_PUBLIC:
        assert re.search(rf"^from zlc_ui import .*\b{re.escape(name)}\b", combined, re.MULTILINE | re.DOTALL)
