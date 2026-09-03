""""Which spec is semantically in play" has exactly ONE answer in the source.

A FacetGrid is a layout; each of its cells draws the cell spec.  Answering
that inline -- ``spec.cell if isinstance(spec, FacetGridPlot) else spec``, or
the same question spelled as a conjunction or an ``if`` block -- is how a
facility ends up honouring the cell in one half of the code and the grid in
the other.  Four user-visible bugs came from exactly that: colour-limit
dragging raised inside a focused image cell, cells were not square, cells
carried no point overlay, and the crosshair lost its value rail.

``zlc_plot.specs.semantic_spec`` is the one authority.  This guard is an AST
walk over the package source, so a new copy cannot be reintroduced by hand.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "zlc_plot"

#: The one definition allowed to unwrap a FacetGrid, as (module, function).
#: ``_kinds/`` is exempt as a package: the facet-grid kind module IS the grid's
#: own semantics, and every other kind module only ever sees a cell spec.
AUTHORITY = ("specs.py", "semantic_spec")


def _modules() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_ROOT.rglob("*.py")
        if "_kinds" not in path.relative_to(SOURCE_ROOT).parts
    )


def _names_facet_grid(node: ast.AST) -> bool:
    """Whether ``node`` contains an ``isinstance(_, FacetGridPlot)`` test."""

    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if not (isinstance(child.func, ast.Name) and child.func.id == "isinstance"):
            continue
        if len(child.args) != 2:
            continue
        for candidate in ast.walk(child.args[1]):
            if isinstance(candidate, ast.Name) and candidate.id == "FacetGridPlot":
                return True
    return False


def _reads_cell(node: ast.AST) -> bool:
    """Whether ``node`` reads a ``.cell`` attribute off anything."""

    return any(
        isinstance(child, ast.Attribute) and child.attr == "cell"
        for child in ast.walk(node)
    )


def _rederivations(tree: ast.AST) -> list[tuple[int, str]]:
    """Every place the FacetGrid unwrap is answered by hand, as (line, form)."""

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            if _names_facet_grid(node.test) and (
                _reads_cell(node.body) or _reads_cell(node.orelse)
            ):
                found.append((node.lineno, "conditional expression"))
        elif isinstance(node, ast.BoolOp):
            if _names_facet_grid(node) and _reads_cell(node):
                found.append((node.lineno, "boolean conjunction"))
        elif isinstance(node, ast.If):
            body = ast.Module(body=[*node.body, *node.orelse], type_ignores=[])
            if _names_facet_grid(node.test) and _reads_cell(body):
                found.append((node.lineno, "if block"))
    return found


def _enclosing_function(tree: ast.AST, line: int) -> str:
    best = "<module>"
    best_line = -1
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= line and node.lineno > best_line:
                end = getattr(node, "end_lineno", node.lineno)
                if line <= end:
                    best = node.name
                    best_line = node.lineno
    return best


def test_semantic_spec_has_one_authority() -> None:
    modules = _modules()
    assert modules, "the guard found no zlc_plot source to walk"
    sites: list[tuple[str, int, str, str]] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, form in _rederivations(tree):
            sites.append(
                (
                    path.name,
                    line,
                    _enclosing_function(tree, line),
                    form,
                )
            )
    unauthorised = [
        site for site in sites if (site[0], site[2]) != AUTHORITY
    ]
    assert not unauthorised, (
        "the FacetGrid semantic unwrap must live only in "
        f"{AUTHORITY[0]}::{AUTHORITY[1]}; found "
        + ", ".join(
            f"{name}:{line} in {function} ({form})"
            for name, line, function, form in unauthorised
        )
    )
    # The authority itself must still be found: a guard that passes because
    # its detector matches nothing at all guards nothing.
    assert [(site[0], site[2]) for site in sites] == [AUTHORITY], sites


def test_the_authority_is_what_every_owner_delegates_to() -> None:
    """Renderer, session and projection expose it; none re-implement it."""

    from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, ImagePlot
    from zlc_plot.specs import semantic_spec

    cell = ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y"))
    grid = FacetGridPlot(AxisRef.repeat("repeat"), cell)
    curve = CurvePlot(AxisRef.point("x"))
    assert semantic_spec(grid) is cell
    assert semantic_spec(curve) is curve
