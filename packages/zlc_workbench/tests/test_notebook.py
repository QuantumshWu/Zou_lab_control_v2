"""The tutorial teaches, and it runs.

A notebook that only reads well is a liability: it drifts from the code and
sends people down paths that stopped working months ago.  A notebook full of
asserts is not a tutorial either -- it is a test that happens to be in a
notebook, and nobody learns from it.

So: every code cell shows its result, no cell asserts, and the whole thing
executes.  The assertions live here.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "usage.ipynb"


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _code_cells() -> list[dict]:
    return [cell for cell in _cells() if cell["cell_type"] == "code"]


def test_the_tutorial_executes_without_error() -> None:
    """Committed with its outputs, so a stale cell shows up in review."""

    errors = [
        output
        for cell in _code_cells()
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == [], errors


def test_every_code_cell_shows_a_result() -> None:
    """A cell that runs and prints nothing teaches nothing."""

    silent = []
    for index, cell in enumerate(_code_cells()):
        source = "".join(cell["source"])
        tree = ast.parse(source)
        prints = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"print", "show"}
        ]
        shows = "plot.show" in source
        if not prints and not shows:
            silent.append(index)
    assert silent == [], f"code cells with no visible result: {silent}"


def test_no_cell_asserts() -> None:
    """Assertions belong in tests; a tutorial shows and explains."""

    for cell in _code_cells():
        tree = ast.parse("".join(cell["source"]))
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]


def test_every_code_cell_is_short_and_introduced() -> None:
    """One idea per cell, and a sentence saying which idea."""

    cells = _cells()
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        assert len(cell["source"]) <= 25, f"cell {index} is {len(cell['source'])} lines"
        assert index > 0 and cells[index - 1]["cell_type"] == "markdown", (
            f"code cell {index} has no explanation before it"
        )


def test_the_tutorial_covers_the_whole_flow() -> None:
    """The round the owner asked for, cell by cell."""

    source = "\n".join("".join(cell["source"]) for cell in _code_cells())
    for step in (
        "ExperimentSession.open",     # device init
        "save_installation_config",   # write the apparatus down
        "load_pulse",                 # set up the pulse
        "prepare(",                   # acquire
        "session.fire",
        "plot.image",                 # look at it
        "save_figure",                # keep it
        "np.load",                    # get it back
        "session.close",
    ):
        assert step in source, f"the tutorial never shows {step}"


def test_the_tutorial_reads_an_archive_without_our_packages() -> None:
    """The durability property is taught, not just implemented.

    Whoever opens an old archive should see it done with numpy and json alone.
    """

    source = "\n".join("".join(cell["source"]) for cell in _code_cells())
    reader = source[source.index("with np.load(") :]
    reader = reader[: reader.index("\n\n")] if "\n\n" in reader else reader
    assert "zlc_" not in reader
