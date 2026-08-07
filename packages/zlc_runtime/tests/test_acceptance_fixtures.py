"""Smoke checks for the user-facing notebook and headless demo."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_usage_notebook_is_valid_and_contains_the_acceptance_flow() -> None:
    notebook = json.loads((ROOT / "notebooks" / "usage.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    for marker in (
        "SignalDataPlane",
        "NodeHost",
        "AcquisitionStream",
        "HarmonicClock",
        "same-shot",
    ):
        assert marker in source


def test_headless_demo_once_exits_cleanly() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo_signal_flow.py"), "--once"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "frame=1" in result.stdout
