"""Frame-claims artifacts through finalization, build arguments, the form and a saved layout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_atom.nodes import ArtifactInputSpec, FrameClaim
from zlc_atom.nodes.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.occupancy.logic_node import LOGIC_NODE as OCCUPANCY_NODE
from zlc_runtime import SignalDataPlane
from zlc_workbench.authoring_form import project_artifact_inputs
from zlc_workbench.console_layout import LayoutDocument, LayoutError, LogicLayoutEntry, resolve_layout
from zlc_workbench.logic import LogicCatalog, LogicDraft, build_arguments, finalize_logic_draft


def _calibration() -> TrapCalibration:
    sites = ("site-1",)
    return TrapCalibration(
        SiteMap(sites, np.asarray([[1.0, 1.0]]), [True], [1.0]),
        (ReadoutModel(sites, [1.0], [0.0], [2.0], [True], [1.0]),),
        ReadoutModelKind.BOX,
        FrameContract((3, 3)),
    )


@pytest.fixture
def workspace(tmp_path: Path):
    (tmp_path / "data").mkdir()
    return SimpleNamespace(root=tmp_path, data=tmp_path / "data")


def _saved(workspace, name: str) -> Path:
    path = workspace.data / name
    _calibration().save(path)
    return path


def _finalize(workspace, plane, claims, **kwargs):
    draft = LogicDraft({"model_kind": "default"}, "camera/frames", {}, {"calibrations": claims})
    return finalize_logic_draft(
        OCCUPANCY_NODE, draft,
        installation=SimpleNamespace(devices={}), signal_plane=plane, workspace=workspace, **kwargs,
    )


def test_finalization_resolves_each_file_and_hands_the_claims_to_the_build(workspace) -> None:
    load, readout = _saved(workspace, "load.json"), _saved(workspace, "readout.json")
    plane = SignalDataPlane()
    try:
        finalization = _finalize(workspace, plane, (
            {"path": str(load), "frames": [0]},
            {"path": "readout.json", "frames": ["rest"]},     # relative to the data folder
        ))
        assert finalization.can_start, finalization.issues
        claims = finalization.artifacts["calibrations"]
        assert [claim.value.path for claim in claims] == [load.resolve(), readout.resolve()]
        assert [claim.frames for claim in claims] == [(0,), ("rest",)]
        arguments = build_arguments(OCCUPANCY_NODE, signal_plane=plane, finalization=finalization)
        assert arguments["calibrations"] is claims
        assert all(isinstance(claim, FrameClaim) for claim in claims)
        # A file decoded once is taken from the last answer again.
        again = _finalize(workspace, plane, (
            {"path": str(load), "frames": [0]}, {"path": str(readout), "frames": ["rest"]},
        ), previous=finalization)
        assert again.artifacts["calibrations"][0].value is claims[0].value
    finally:
        plane.close()


def test_claims_are_refused_by_name_and_against_a_known_frame_count(workspace) -> None:
    load, readout = _saved(workspace, "load.json"), _saved(workspace, "readout.json")
    plane = SignalDataPlane()
    try:
        assert "occupancy needs calibrations: add one" in _finalize(workspace, plane, ()).issues
        twice = _finalize(workspace, plane, (
            {"path": str(load), "frames": ["rest"]}, {"path": str(load), "frames": [0]},
        ))
        assert "load.json is listed twice: give it every frame it reads in one row" in twice.issues
        both = _finalize(workspace, plane, (
            {"path": str(load), "frames": ["rest"]}, {"path": str(readout), "frames": ["rest"]},
        ))
        assert "'rest' is claimed by both load.json and readout.json" in both.issues
        claims = ({"path": str(load), "frames": [2]}, {"path": str(readout), "frames": ["rest"]})
        assert _finalize(workspace, plane, claims).can_start, "no count known: nothing to check yet"
        assert _finalize(workspace, plane, claims, source_frames=2).issues == (
            "load.json names frame 2, but the source carries 2 frames (0-1); "
            "numbered middle frames cannot stand for the last frame",
        )
        assert _finalize(workspace, plane, claims, source_frames=4).can_start
        empty_row = _finalize(workspace, plane, ({"path": "", "frames": ["rest"]},))
        assert "Calibrations row 1 has no file yet" in empty_row.issues
        shape = _finalize(workspace, plane, "load.json")
        assert any("frame claims must be a list" in issue for issue in shape.issues)
    finally:
        plane.close()


def test_the_path_form_leaves_frame_claims_to_their_editor() -> None:
    (spec,) = [s for s in OCCUPANCY_NODE.input_specs if isinstance(s, ArtifactInputSpec)]
    assert spec.frame_claims
    assert project_artifact_inputs((spec,), base_dir="d").keys == ()


def test_a_saved_layout_keeps_the_claims_as_written() -> None:
    claims = ({"path": "load.json", "frames": [0]}, {"path": "readout.json", "frames": ["last", "rest"]})
    entry = LogicLayoutEntry(
        "occupancy", "occupancy", {"model_kind": "default"}, "", {}, {"calibrations": claims},
    )
    tree = LayoutDocument((), (entry,), ()).to_tree()
    assert tree["logic"][0]["artifact_inputs"]["calibrations"][1]["frames"] == ["last", "rest"]
    document = LayoutDocument.from_tree(tree)
    resolved = resolve_layout(
        document, catalog=LogicCatalog(), installation=SimpleNamespace(devices={}), panel_ids=(),
    )
    (binding,) = resolved.logic
    assert binding.draft.artifact_inputs == {"calibrations": (
        {"path": "load.json", "frames": (0,)}, {"path": "readout.json", "frames": ("last", "rest")},
    )}
    broken = LogicLayoutEntry(
        "occupancy", "occupancy", {}, "", {}, {"calibrations": [{"path": "a.json", "frames": ["second"]}]},
    )
    with pytest.raises(LayoutError, match="not a frame"):
        resolve_layout(
            LayoutDocument((), (broken,), ()), catalog=LogicCatalog(),
            installation=SimpleNamespace(devices={}), panel_ids=(),
        )
