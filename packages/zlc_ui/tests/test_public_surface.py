"""Non-vacuous guards for the small, lazy top-level facade."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
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
    assert "import_module" not in public_names
    assert "annotations" not in public_names
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
