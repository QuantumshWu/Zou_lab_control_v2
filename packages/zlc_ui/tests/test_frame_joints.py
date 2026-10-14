"""Frame joints are exact at every device-pixel ratio.

A part inside a frame -- a spin box's divider, its hover tint, a chevron, a
table's rules, a card's title band -- is placed FROM THE FRAME and snapped
to device pixels.  Placed from a sub-control rectangle with a hand-tuned
offset, the spin divider stopped half a pixel under the top edge and ran into
the bottom one, and the offsets that hid it at one scale broke at the next.

The measurement is real: each (device-pixel ratio, fluent scale) pair runs
in its own process on the REAL platform with QT_SCALE_FACTOR, because the
offscreen platform only knows a ratio of 1 -- exactly where the defect did
not show.  ``_joint_child.py`` renders, reads the edge, ground and tint
colours back from the render, and reports one verdict per joint; this module
runs the children one at a time and fails with the child's crops named.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import pytest


HERE = Path(__file__).resolve().parent
CHILD = HERE / "_joint_child.py"
SRC = HERE.parent / "src"
REPO_ROOT = HERE.parents[2]
DPRS = (1.0, 1.25, 1.5, 2.0, 3.0)
SCALES = (1.0, 0.85)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="real-platform renders run on Windows")


def _environment(scale_factor: float | None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("QT_QPA_PLATFORM", None)
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), str(SRC)))
    environment.pop("QT_SCALE_FACTOR", None)
    if scale_factor is not None:
        environment["QT_SCALE_FACTOR"] = repr(scale_factor)
    return environment


def _run(argument: str, scale_factor: float | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHILD), argument],
        cwd=str(REPO_ROOT),
        env=_environment(scale_factor),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.fixture(scope="module")
def native_ratio() -> float:
    completed = _run("discover", None)
    if completed.returncode != 0:
        pytest.skip(f"no real platform: {completed.stderr[-400:]}")
    native = json.loads(completed.stdout.strip().splitlines()[-1])["native"]
    if not native:
        pytest.skip("the real platform reports no screen")
    return float(native)


@pytest.fixture(scope="module")
def reports(native_ratio: float, tmp_path_factory) -> Callable[[float, float], dict]:
    """One child per (ratio, scale), run once and read by both tests."""

    cache: dict[tuple[float, float], dict] = {}

    def report(dpr: float, scale: float) -> dict:
        key = (dpr, scale)
        if key not in cache:
            crops = tmp_path_factory.mktemp(f"joints_{dpr:g}_{scale:g}")
            completed = _run(json.dumps({"dpr": dpr, "scale": scale, "crops": str(crops)}), dpr / native_ratio)
            assert completed.returncode != 3, f"the harness did not produce DPR {dpr}: {completed.stdout[-300:]}"
            assert completed.returncode == 0, completed.stderr[-3000:] or completed.stdout[-3000:]
            line = next(
                (text for text in completed.stdout.splitlines() if text.startswith("JOINTS ")), None
            )
            assert line is not None, completed.stdout[-2000:]
            cache[key] = json.loads(line[len("JOINTS "):])
        return cache[key]

    return report


def _is_focus_case(entry: dict) -> bool:
    return "/focus/" in entry["case"] or entry["case"].startswith("focus_ring")


def _failures(entries: list[dict]) -> list[str]:
    return [
        f"{entry['case']} :: {entry['check']} :: {entry['detail']}"
        + (f" [crop {entry['crop']}]" if entry.get("crop") else "")
        for entry in entries
        if not entry.get("skip") and not entry["ok"]
    ]


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dpr", DPRS)
def test_frame_joints(dpr: float, scale: float, reports) -> None:
    report = reports(dpr, scale)
    entries = [entry for entry in report["results"] if not _is_focus_case(entry)]
    assert entries, "the child measured nothing"
    failures = _failures(entries)
    assert not failures, f"DPR {dpr} scale {scale}:\n" + "\n".join(failures)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dpr", DPRS)
def test_focused_frame_joints(dpr: float, scale: float, reports) -> None:
    """The focused field and the focus ring: measured only when the real
    platform gives the test window keyboard focus (another window in front
    can refuse it), and skipped with that reason when it does not."""

    report = reports(dpr, scale)
    entries = [entry for entry in report["results"] if _is_focus_case(entry)]
    if any(entry.get("skip") == "FOCUS-UNAVAILABLE" for entry in entries):
        pytest.skip("FOCUS-UNAVAILABLE: the platform did not give the test window keyboard focus")
    assert entries, "the child measured no focus case"
    failures = _failures(entries)
    assert not failures, f"DPR {dpr} scale {scale}:\n" + "\n".join(failures)
