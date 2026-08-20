"""Mechanical guard for QApplication ownership and HiDPI ordering."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _function_stack(tree: ast.AST, target: ast.AST) -> list[str]:
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def generic_visit(self, node: ast.AST) -> None:
            if node is target:
                raise StopIteration
            super().generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            try:
                self.generic_visit(node)
            finally:
                stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    try:
        Visitor().visit(tree)
    except StopIteration:
        return stack.copy()
    return []


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_only_ensure_qt_app_constructs_qapplication() -> None:
    constructors: list[tuple[Path, ast.Call, ast.AST]] = []
    attribute_calls: list[tuple[Path, ast.Call]] = []

    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            is_qapplication_constructor = (
                isinstance(function, ast.Name)
                and function.id == "QApplication"
            ) or (
                isinstance(function, ast.Attribute)
                and function.attr == "QApplication"
            )
            if is_qapplication_constructor:
                cursor: ast.AST | None = node
                while cursor is not None and not isinstance(
                    cursor, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    cursor = parents.get(cursor)
                assert cursor is not None
                constructors.append((path, node, cursor))
            elif (
                isinstance(function, ast.Attribute)
                and function.attr == "setAttribute"
                and (
                    (
                        isinstance(function.value, ast.Name)
                        and function.value.id == "QApplication"
                    )
                    or (
                        isinstance(function.value, ast.Attribute)
                        and function.value.attr == "QApplication"
                    )
                )
            ):
                attribute_calls.append((path, node))

    assert len(constructors) == 1, "exactly one QApplication constructor is allowed"
    constructor_path, constructor, owner = constructors[0]
    assert constructor_path == SRC / "zlc_ui" / "qt.py"
    assert isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef))
    assert owner.name == "ensure_qt_app"
    assert attribute_calls, "HiDPI attributes must be configured mechanically"
    assert all(path == constructor_path for path, _ in attribute_calls)
    assert all(call.lineno < constructor.lineno for _, call in attribute_calls)


def test_importing_package_does_not_create_qapplication() -> None:
    code = (
        "import zlc_ui; "
        "from PyQt5.QtWidgets import QApplication; "
        "assert QApplication.instance() is None"
    )
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

def test_ensure_qt_app_sets_reference_font_and_owner_thread() -> None:
    code = """
from PyQt5 import QtCore, QtWidgets
from zlc_ui.fluent import ensure_fluent_scale, fluent_font_size, resolve_fluent_auto_scale
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['test'])
assert ensure_qt_app(['ignored']) is app
assert QtCore.QThread.currentThread() == app.thread()
expected = min(1.25, max(0.72, resolve_fluent_auto_scale(app)))
assert abs(ensure_fluent_scale() - expected) < 1e-9
assert app.font().family() == 'Segoe UI'
assert app.font().pointSize() == fluent_font_size()
"""
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_ensure_qt_app_rejects_preexisting_non_high_dpi_application() -> None:
    code = """
from PyQt5 import QtWidgets
raw_app = QtWidgets.QApplication(['raw-app'])
from zlc_ui.qt import ensure_qt_app
try:
    ensure_qt_app(['test'])
except RuntimeError as error:
    assert 'High-DPI' in str(error)
else:
    raise AssertionError('a non-High-DPI QApplication must not be reused silently')
assert QtWidgets.QApplication.instance() is raw_app
"""
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_resizable_fluent_launcher_cannot_fall_back_to_content_hint_size() -> None:
    code = """
from PyQt5 import QtWidgets
from zlc_ui.fluent import WINDOW_SCREEN_FRACTION, launch_fluent_window, screen_fit_window_size
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['launcher-size'])
body = QtWidgets.QLabel('tiny content hint')
window = launch_fluent_window(body, title='launcher-size', fixed_size=False)
app.processEvents()
target = screen_fit_window_size(WINDOW_SCREEN_FRACTION)
assert window.size() == target, (window.size().width(), window.size().height(), target.width(), target.height())
assert window.width() > body.sizeHint().width()
window.close()
"""
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_reusable_capture_api_rejects_offscreen_ui_acceptance() -> None:
    code = """
from PyQt5 import QtWidgets
from zlc_ui.acceptance import capture_window
from zlc_ui.fluent import open_fluent_window
from zlc_ui.qt import ensure_qt_app

app = ensure_qt_app(['capture-api'])
def factory():
    assert QtWidgets.QApplication.instance() is app
    return QtWidgets.QLabel('capture api')

try:
    capture_window(lambda: open_fluent_window(factory, title='must-reject'))
except RuntimeError as error:
    assert 'offscreen' in str(error).lower()
else:
    raise AssertionError('offscreen capture was accepted as UI acceptance evidence')
"""
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_fluent_window_uses_reference_frameless_shell_and_top_margin() -> None:
    code = """
from PyQt5 import QtWidgets
from qframelesswindow import FramelessWindow
from zlc_ui.fluent import FluentWindow, frameless_content_top_margin
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['test'])
body = QtWidgets.QLabel('body')
window = FluentWindow(widget=body, title='test')
assert isinstance(window, FramelessWindow)
assert window.titleBar is not None
assert window.layout().contentsMargins().top() > 0
assert window.loaded is body
window.show(); app.processEvents()
assert window.layout().contentsMargins().top() == frameless_content_top_margin(window.titleBar)
assert window.loaded.geometry().top() >= window.titleBar.geometry().bottom() + 1, (
    window.loaded.geometry().top(), window.titleBar.geometry().getRect()
)
"""
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
