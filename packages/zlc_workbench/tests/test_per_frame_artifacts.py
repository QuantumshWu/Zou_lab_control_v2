"""A per-frame artifact through finalization, build arguments, the form and a saved layout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.nodes import ArtifactInputSpec, LogicNodeDescriptor, NodeKind
from zlc_atom.nodes.calibration import (
    CALIBRATION_ARTIFACT_CODEC,
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_atom.nodes.occupancy.logic_node import LOGIC_NODE as OCCUPANCY_NODE
from zlc_runtime import SignalDataPlane
from zlc_workbench.authoring_form import project_artifact_inputs
from zlc_workbench.console_layout import LayoutDocument, LogicLayoutEntry, resolve_layout
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


def test_finalization_decodes_each_file_once_and_builds_the_frame_mapping(workspace) -> None:
    shared, strict = _saved(workspace, "shared.json"), _saved(workspace, "strict.json")
    plane = SignalDataPlane()
    try:
        draft = LogicDraft(
            {"model_kind": "default"},
            "camera/frames",
            {},
            {
                "calibration_path": str(shared),
                "calibration_path[2]": "strict.json",          # relative to the data folder
                "calibration_path[3]": str(shared),            # the shared file again
            },
        )
        finalization = finalize_logic_draft(
            OCCUPANCY_NODE, draft,
            installation=SimpleNamespace(devices={}), signal_plane=plane, workspace=workspace,
        )
        assert set(finalization.artifacts) == {
            "calibration_path", "calibration_path[2]", "calibration_path[3]",
        }
        assert finalization.artifacts["calibration_path[3]"] is finalization.artifacts["calibration_path"], (
            "one decode per file, however many frames name it"
        )
        assert finalization.artifact_paths["calibration_path[2]"] == str(strict.resolve())
        arguments = build_arguments(OCCUPANCY_NODE, signal_plane=plane, finalization=finalization)
        assert arguments["calibration"] is finalization.artifacts["calibration_path"]
        assert set(arguments["calibration_by_frame"]) == {2, 3}
        assert arguments["calibration_by_frame"][2] is finalization.artifacts["calibration_path[2]"]
    finally:
        plane.close()


def test_a_frame_key_on_an_artifact_that_is_not_per_frame_is_refused(workspace) -> None:
    plain = LogicNodeDescriptor(
        "plain_consumer",
        NodeKind.TASK,
        AuthoringSchema(()),
        node_previews=(),
        input_specs=(
            ArtifactInputSpec(
                "calibration_path", "Calibration artifact", CALIBRATION_ARTIFACT_CODEC,
                argument_name="calibration",
            ),
        ),
        build=lambda **arguments: arguments,
    )
    assert "calibration_by_frame" not in plain.build_argument_names
    shared = _saved(workspace, "shared.json")
    plane = SignalDataPlane()
    try:
        draft = LogicDraft({}, "", {}, {
            "calibration_path": str(shared), "calibration_path[2]": str(shared),
        })
        finalization = finalize_logic_draft(
            plain, draft,
            installation=SimpleNamespace(devices={}), signal_plane=plane, workspace=workspace,
        )
        assert any("calibration_path[2]" in issue for issue in finalization.issues), finalization.issues
        assert set(finalization.artifacts) == {"calibration_path"}
    finally:
        plane.close()


def test_per_frame_pickers_follow_the_source_cycle() -> None:
    (spec,) = [s for s in OCCUPANCY_NODE.input_specs if isinstance(s, ArtifactInputSpec)]
    assert spec.per_frame
    three = project_artifact_inputs((spec,), base_dir="d", frames_per_cycle=3)
    assert three.keys == (
        "calibration_path", "calibration_path[1]", "calibration_path[2]", "calibration_path[3]",
    )
    assert [field.required for field in three.fields] == [True, False, False, False]
    assert three.fields[0].label == "Calibration artifact (every frame)"
    assert three.fields[2].label == "Calibration artifact · frame 2"
    assert all(field.base_dir == "d" for field in three.fields)
    # One frame, or none known yet: only the plain picker.
    for frames in (0, 1):
        assert project_artifact_inputs((spec,), base_dir="d", frames_per_cycle=frames).keys == (
            "calibration_path",
        )
        assert project_artifact_inputs((spec,), base_dir="d", frames_per_cycle=frames).fields[0].label == (
            "Calibration artifact"
        )


def test_a_saved_layout_keeps_each_frames_artifact() -> None:
    entry = LogicLayoutEntry(
        "occupancy", "occupancy", {"model_kind": "default"}, "", {},
        {"calibration_path": "shared.json", "calibration_path[2]": "strict.json"},
    )
    document = LayoutDocument.from_tree(LayoutDocument((), (entry,), ()).to_tree())
    resolved = resolve_layout(
        document, catalog=LogicCatalog(), installation=SimpleNamespace(devices={}), panel_kinds=(),
    )
    (binding,) = resolved.logic
    assert binding.draft.artifact_inputs == {
        "calibration_path": "shared.json", "calibration_path[2]": "strict.json",
    }
