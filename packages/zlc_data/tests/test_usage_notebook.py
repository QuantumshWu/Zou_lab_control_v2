"""Executable tutorial-shape guards for notebooks/usage.ipynb."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "usage.ipynb"


def _source(cell: dict[str, object]) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else str(source)


def test_usage_notebook_is_small_printing_and_executed_tutorial():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]

    assert code_cells
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = _source(cell)
        assert index > 0 and cells[index - 1]["cell_type"] == "markdown"
        assert len(source.splitlines()) <= 25
        assert re.search(r"\bprint\s*\(", source)
        assert not re.search(r"\bassert\b", source)
        outputs = cell["outputs"]
        assert outputs
        assert not any(output["output_type"] == "error" for output in outputs)
        assert any(
            output["output_type"] in {"display_data", "execute_result", "stream"}
            for output in outputs
        )


def test_usage_notebook_teaches_every_facade_name():
    import zlc_data

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    loaded_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert set(zlc_data.__all__) - loaded_names - loaded_attributes == set()
