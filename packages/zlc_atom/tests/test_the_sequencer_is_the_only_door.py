"""Nothing outside zlc_pulse compiles a pulse for a board on its own.

A config parameter's value is baked into the program at COMPILE.  So a caller
that reaches ``compile_sequence`` directly, instead of asking the sequencer
that will play it, compiles the AUTHORED placeholder and fires it -- silently.
``compile_sequence`` cannot refuse this: it is deliberately blind to config
parameters (a declared one is not a hole, it always has a number), and its
signature is frozen by ``zlc_pulse/tests/test_contract.py``.

So the rule cannot be enforced by the compiler, which means it has to be
enforced here.  The one exception is the Pulse Editor with no board attached,
which has nothing to ask and says so.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: file -> the function that is allowed to call it, and why.
ALLOWED = {
    "packages/zlc_workbench/src/zlc_workbench/pulse_editor.py": "compile",
}

ROOT = Path(__file__).resolve().parents[3]


def _direct_callers() -> dict[str, set[str]]:
    """Every function outside zlc_pulse that names ``compile_sequence``."""

    found: dict[str, set[str]] = {}
    for path in sorted((ROOT / "packages").glob("*/src/**/*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("packages/zlc_pulse/"):
            continue  # the compiler's own package
        tree = ast.parse(path.read_text(encoding="utf-8"))
        enclosing: dict[ast.AST, str] = {}
        for node in ast.walk(tree):
            name = getattr(node, "name", None)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    enclosing.setdefault(child, name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if isinstance(called, ast.Attribute):
                called = called.attr
            elif isinstance(called, ast.Name):
                called = called.id
            else:
                continue
            if called == "compile_sequence":
                found.setdefault(relative, set()).add(enclosing.get(node, "<module>"))
    return found


def test_only_the_offline_editor_compiles_a_pulse_without_a_board() -> None:
    callers = _direct_callers()
    assert set(callers) == set(ALLOWED), (
        "a pulse compiled outside the sequencer plays its AUTHORED config "
        "values, not the board's calibrated ones, and nothing says so: "
        f"{sorted(set(callers) - set(ALLOWED))}"
    )
    for relative, functions in callers.items():
        assert functions == {ALLOWED[relative]}, (
            f"{relative} calls compile_sequence from {sorted(functions)}, "
            f"not only from {ALLOWED[relative]!r}"
        )
