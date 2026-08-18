from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def test_notebook_offline_prefix_executes_without_hardware(monkeypatch) -> None:
    notebook = json.loads(
        (ROOT / "notebooks/usage.ipynb").read_text(encoding="utf-8")
    )
    monkeypatch.chdir(ROOT)
    namespace: dict[str, object] = {}
    ran: list[str] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code" or not _source(cell).strip():
            continue
        if cell["id"] == "hardware-open":
            break
        with contextlib.redirect_stdout(io.StringIO()):
            exec(
                compile(_source(cell), f"<notebook:{cell['id']}>", "exec"),
                namespace,
                namespace,
            )
        ran.append(cell["id"])
    assert ran and ran[-2:] == ["hardware-config", "hardware-program"]
